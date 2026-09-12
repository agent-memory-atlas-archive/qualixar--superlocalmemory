# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3 — Background Maintenance Scheduler.

V3.3.13: Periodically triggers Langevin/Ebbinghaus/Sheaf maintenance
so users don't need to call run_maintenance manually.

Configurable interval via ForgettingConfig.scheduler_interval_minutes.
Defaults to 30 min. Optional forgetting/math work follows
``config.forgetting.enabled``; tier evaluation and bounded housekeeping do not.

Part of Qualixar | Author: Varun Pratap Bhardwaj
License: AGPL-3.0-or-later
"""

from __future__ import annotations

import logging
from datetime import timedelta
import os
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.storage.database import DatabaseManager

logger = logging.getLogger(__name__)



# How much vector-store version history to keep. Every write to the vector
# store creates a version and nothing used to remove them: a 5,561-memory
# store reached 50,580 versions in 17 GB, and each vector operation then had
# to walk that history -- the daemon in GitHub #137 held a core at 90%+ with
# an empty queue. A week is enough to debug a bad projection and short enough
# that the history cannot run away.
_VECTOR_HISTORY_DAYS = float(os.environ.get("SLM_VECTOR_HISTORY_DAYS", "7"))


def compact_vector_store() -> dict:
    """Drop vector-store versions older than the retention window.

    Best-effort and never raises: a maintenance pass that cannot compact must
    not take the daemon down, and the next tick will try again.

    ``delete_unverified`` stays OFF here. It removes files no manifest
    references, which is exactly what an in-flight write is creating -- safe
    only with no writer attached, i.e. the offline repair path.
    """
    try:
        from superlocalmemory.core.backend_orchestrator import get_orchestrator

        orchestrator = get_orchestrator()
        if orchestrator is None:
            logger.warning("vector store compaction skipped: no orchestrator")
            return {"ok": False, "reason": "no orchestrator"}
        backend = orchestrator.get_vector_backend()
        compact = getattr(backend, "compact", None)
        if not callable(compact):
            logger.warning("vector store compaction skipped: backend cannot compact")
            return {"ok": False, "reason": "backend cannot compact"}
        return compact(retention=timedelta(days=_VECTOR_HISTORY_DAYS))
    except Exception as exc:  # noqa: BLE001 -- maintenance is best-effort
        logger.warning("vector store compaction skipped: %s", exc)
        return {"ok": False, "reason": str(exc)}


class MaintenanceScheduler:
    """Background scheduler for periodic math maintenance.

    Runs Langevin/Sheaf/Fisher maintenance at configurable intervals.
    Thread-safe. Auto-stops on garbage collection or explicit stop().
    """

    def __init__(
        self,
        db: DatabaseManager,
        config: SLMConfig,
        profile_id: str = "default",
        embedder: object | None = None,
    ) -> None:
        self._db = db
        self._config = config
        self._profile_id = profile_id
        # v3.8.2 self-heal: when provided, periodic maintenance backfills
        # NULL embeddings so a DB stays fully queryable over time even if
        # facts were stored while the embedder was unavailable. Runs
        # independently of forgetting.enabled (see _run).
        self._embedder = embedder
        self._timer: threading.Timer | None = None
        self._running = False
        self._interval = config.forgetting.scheduler_interval_minutes * 60.0

    def start(self) -> None:
        """Start the periodic scheduler. Idempotent."""
        if self._running:
            return
        self._running = True
        self._schedule_next()
        # v3.8.5: one-shot activation-cache GC ~90s after boot so an upgrade
        # backlog (observed 83k expired rows) clears promptly instead of waiting
        # a full interval — delayed past boot warmup so it never competes for
        # the write lock during the startup window.
        self._initial_gc_timer = threading.Timer(90.0, self._initial_cache_gc)
        self._initial_gc_timer.daemon = True
        self._initial_gc_timer.start()
        # An upgrade arrives with whatever backlog the previous version left.
        # Waiting a full interval for the first graph-metrics pass would mean
        # half an hour of ranking memories as though they had no position in the
        # graph, on exactly the store that just gained the fix. Staggered behind
        # the cache GC so the two never contend for the write lock.
        self._initial_metrics_timer = threading.Timer(
            150.0, self._initial_graph_metrics,
        )
        self._initial_metrics_timer.daemon = True
        self._initial_metrics_timer.start()
        # M048's re-read pass is idempotent and safe to replay (see its
        # verify() docstring), but the migration runner never replays a
        # completed migration and the cycle above only reaches it every
        # ``scheduler_interval_minutes`` (360 by default). A store that
        # drifted since the last cycle stays blocking — ready=false,
        # migrations=false — for up to that whole interval after every
        # restart, not because the fix is slow, but because nothing asks
        # for it sooner. Staggered behind the other two so none of the
        # three one-shots contend for the write lock.
        self._initial_reclassify_timer = threading.Timer(
            210.0, self._initial_reclassify_upcoming,
        )
        self._initial_reclassify_timer.daemon = True
        self._initial_reclassify_timer.start()
        # Vector-store version history is bounded on the maintenance tick,
        # but that tick is armed from boot and only re-armed at the end of
        # _run. A LaunchAgent that restarts inside scheduler_interval_minutes
        # (KeepAlive, MCP clients) never gets a first pass — measured: 3,517
        # scheduler-start lines and zero compact lines on a 37k-version /
        # 48 GB store. Same one-shot pattern as cache GC / graph metrics,
        # staggered so it does not contend for the write lock.
        self._initial_compact_timer = threading.Timer(
            270.0, self._initial_vector_compaction,
        )
        self._initial_compact_timer.daemon = True
        self._initial_compact_timer.start()
        logger.info(
            "Maintenance scheduler started (interval=%dm)",
            self._config.forgetting.scheduler_interval_minutes,
        )

    def _initial_reclassify_upcoming(self) -> None:
        """Best-effort one-shot M048 re-read shortly after boot.

        Runs the same pass the periodic cycle already runs (see below) so a
        store that drifted before this restart converges within seconds
        instead of waiting up to a full maintenance interval with the
        daemon reporting itself not ready.
        """
        if not self._running:
            return
        try:
            from superlocalmemory.storage.migrations import (
                M048_upcoming_holds_only_what_is_upcoming as _reclassify,
            )
            _reclassify.apply(open_connection=self._db.raw_connection)
        except Exception as exc:
            logger.debug("Startup plan re-read skipped: %s", exc)

    def _initial_vector_compaction(self) -> None:
        """Drop stale vector-store versions without waiting a full interval."""
        if not self._running:
            return
        out = compact_vector_store()
        self._record_step(
            "vector compaction",
            bool(out.get("ok")),
            str(out.get("reason") or ""),
        )

    def _initial_cache_gc(self) -> None:
        """Best-effort one-shot activation-cache GC shortly after boot."""
        if not self._running:
            return
        try:
            deleted = self._db.cleanup_activation_cache()
            if deleted > 0:
                logger.info(
                    "Activation-cache GC (startup): %d expired rows removed",
                    deleted,
                )
        except Exception as exc:
            logger.debug("Startup activation-cache GC skipped: %s", exc)

    def _initial_graph_metrics(self) -> None:
        """One-shot catch-up so an upgrade does not rank on stale metrics."""
        if not self._running:
            return
        try:
            from superlocalmemory.core.graph_metrics import (
                compute_graph_metrics,
                metrics_are_stale,
            )
            for profile_id in self._profile_ids():
                stale, why = metrics_are_stale(self._db, profile_id)
                if not stale:
                    continue
                report = compute_graph_metrics(self._db, profile_id)
                if report.ok:
                    logger.info(
                        "Graph metrics at startup (%s): %s", why, report.summary(),
                    )
                else:
                    logger.warning("Graph metrics at startup: %s", report.summary())
        except Exception as exc:
            logger.debug("Startup graph metrics skipped: %s", exc)

    def stop(self) -> None:
        """Stop the scheduler. Idempotent."""
        self._running = False
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None
        _gc_timer = getattr(self, "_initial_gc_timer", None)
        if _gc_timer is not None:
            _gc_timer.cancel()
            self._initial_gc_timer = None
        _metrics_timer = getattr(self, "_initial_metrics_timer", None)
        if _metrics_timer is not None:
            _metrics_timer.cancel()
            self._initial_metrics_timer = None
        _reclassify_timer = getattr(self, "_initial_reclassify_timer", None)
        if _reclassify_timer is not None:
            _reclassify_timer.cancel()
            self._initial_reclassify_timer = None
        _compact_timer = getattr(self, "_initial_compact_timer", None)
        if _compact_timer is not None:
            _compact_timer.cancel()
            self._initial_compact_timer = None
        logger.info("Maintenance scheduler stopped")

    def _schedule_next(self) -> None:
        """Schedule the next maintenance run."""
        if not self._running:
            return
        self._timer = threading.Timer(self._interval, self._run)
        self._timer.daemon = True
        self._timer.start()

    #: Consecutive failures of one step before it stops being a hiccup.
    _ESCALATE_AFTER = 3

    def _record_step(self, step: str, ok: bool, detail: str = "") -> None:
        """Remember whether a maintenance step worked, and say so when it has
        stopped working."""
        counts = getattr(self, "_step_failures", None)
        if counts is None:
            counts = self._step_failures = {}
        if ok:
            if counts.pop(step, 0):
                logger.info("maintenance: %s is working again", step)
            return
        counts[step] = counts.get(step, 0) + 1
        if counts[step] >= self._ESCALATE_AFTER:
            logger.error(
                "maintenance: %s has failed %d cycles in a row (%s). This is "
                "not a transient failure; the work it does is not being done.",
                step, counts[step], detail or "no detail",
            )
        else:
            logger.warning("maintenance: %s failed (%s)", step, detail or "")

    def failing_steps(self) -> dict[str, int]:
        """Steps that have failed on consecutive cycles, and how many.

        Read by the status surfaces, so an operator can see a persistently
        broken maintenance step instead of having to find it in the log.
        """
        return dict(getattr(self, "_step_failures", {}) or {})

    def _note_step_outcomes(self) -> None:
        """Escalate anything still failing after this cycle's steps."""
        failing = self.failing_steps()
        if failing:
            logger.warning(
                "maintenance: %d step(s) still failing: %s",
                len(failing),
                ", ".join(f"{name} x{count}" for name, count in sorted(failing.items())),
            )

    def _run(self) -> None:
        """Execute maintenance + auto-backup check, then schedule next run."""
        if not self._running:
            return
        # v3.8.2 self-heal: bounded NULL-embedding backfill runs every cycle
        # INDEPENDENTLY of forgetting.enabled and across ALL profiles — a fact
        # stored while the embedder was down must become queryable again without
        # the user touching anything. Idempotent + bounded (200/pass) so it
        # converges quietly and is a no-op once coverage is complete.
        if self._embedder is not None:
            try:
                from superlocalmemory.storage.embedding_migrator import (
                    backfill_missing_embeddings,
                )
                r = backfill_missing_embeddings(
                    self._config, self._db, self._embedder,
                    limit=50, all_profiles=True,
                )
                if r.get("embedded"):
                    logger.info(
                        "Self-heal backfill: %d embedded, %d remaining",
                        r["embedded"], r["remaining_null"],
                    )
            except Exception as exc:
                logger.debug("Self-heal backfill skipped: %s", exc)

        # Once per cycle, not per profile: there is one embeddings table.
        # Also before the per-profile Langevin/Ollama work — on a real store
        # that work ran for 11+ minutes and compact never started.
        out = compact_vector_store()
        self._record_step(
            "vector compaction",
            bool(out.get("ok")),
            str(out.get("reason") or ""),
        )

        for profile_id in self._profile_ids():
            if self._config.forgetting.enabled:
                try:
                    from superlocalmemory.core.maintenance import run_maintenance
                    counts = run_maintenance(self._db, self._config, profile_id)
                    logger.info(
                        "Scheduled maintenance complete for %s: %s",
                        profile_id,
                        counts,
                    )
                except Exception as exc:
                    logger.warning(
                        "Scheduled maintenance failed for %s: %s",
                        profile_id,
                        exc,
                    )

            # V3.4.11: Graph pruning (remove orphan edges)
            # v3.8.4-G: thread GraphPruningConfig params so dashboard changes
            # persist and take effect without a daemon restart.
            try:
                from superlocalmemory.core.graph_pruner import prune_graph
                gp = self._config.graph_pruning
                if not gp.enabled:
                    logger.debug(
                        "Graph pruning disabled by config for %s — skipping",
                        profile_id,
                    )
                else:
                    # Fix A: pass DatabaseManager directly → writes serialised through _lock
                    prune_stats = prune_graph(
                        self._db,
                        profile_id,
                        max_degree=gp.max_degree_per_node,
                        min_edge_weight=gp.min_edge_weight,
                    )
                    removed = prune_stats["total_before"] - prune_stats["total_after"]
                    if removed > 0:
                        logger.info(
                            "Graph pruning for %s: %d edges removed", profile_id, removed
                        )
            except AttributeError:
                # Older SLMConfig without graph_pruning field (upgrade safety)
                from superlocalmemory.core.graph_pruner import prune_graph
                prune_stats = prune_graph(self._db, profile_id)
                removed = prune_stats["total_before"] - prune_stats["total_after"]
                if removed > 0:
                    logger.info("Graph pruning for %s: %d edges removed", profile_id, removed)
            except Exception as exc:
                logger.debug("Graph pruning skipped for %s: %s", profile_id, exc)

            # Pruning the graph orphans the lineage of every edge it removed,
            # and nothing had ever deleted from that table — on a real store it
            # had grown to 39% rows describing edges that no longer existed.
            # This runs immediately after so the rows the pass just orphaned are
            # collected in the same pass.
            try:
                from superlocalmemory.storage.lineage_retention import (
                    prune_orphan_lineage,
                )
                report = prune_orphan_lineage(self._db, profile_id=profile_id)
                if report.total:
                    logger.info(
                        "Lineage retention for %s: %d row(s) removed (%s)",
                        profile_id, report.total, report.deleted,
                    )
            except Exception as exc:
                logger.debug("Lineage retention skipped for %s: %s", profile_id, exc)

            # Structural metrics. Recall multiplies a candidate's activation by
            # its PageRank at every hop and biases it toward its query seeds'
            # communities, and both numbers live in fact_importance -- so a
            # memory missing from that table is found by the walk and then
            # ranked as though it had no position in the graph.
            #
            # Nothing scheduled this. It ran only when a consolidation happened
            # to fire or someone called the HTTP endpoint by hand, and on the
            # author's store that meant one run in nine days: 1,036 of 4,034
            # visible memories had no score and no community, and the newest
            # four days of memories had none at all. This runs after pruning so
            # it describes the graph that pruning left behind.
            try:
                from superlocalmemory.core.graph_metrics import (
                    compute_graph_metrics,
                    metrics_are_stale,
                )
                stale, why = metrics_are_stale(self._db, profile_id)
                if stale:
                    backend = None
                    try:
                        from superlocalmemory.core.backend_orchestrator import (
                            get_orchestrator,
                        )
                        orchestrator = get_orchestrator()
                        if orchestrator is not None:
                            backend = orchestrator.get_graph_backend()
                    except Exception:  # noqa: BLE001 -- in-process is the default anyway
                        backend = None
                    report = compute_graph_metrics(
                        self._db, profile_id, backend=backend,
                    )
                    if report.ok:
                        logger.info("Graph metrics (%s): %s", why, report.summary())
                    else:
                        logger.warning("Graph metrics: %s", report.summary())
                else:
                    logger.debug("Graph metrics up to date for %s", profile_id)
                self._record_step("graph metrics", True)
            except Exception as exc:  # noqa: BLE001
                self._record_step("graph metrics", False, str(exc))

            # Lifecycle evaluation must cover every stored profile, not only
            # whichever profile was active when the engine started.
            try:
                from superlocalmemory.core.tier_manager import evaluate_tiers
                stats = evaluate_tiers(self._db, profile_id)
                demoted = stats["demoted_to_warm"] + stats["demoted_to_cold"] + stats["demoted_to_archive"]
                if demoted > 0:
                    logger.info("Tier evaluation for %s: %d facts demoted", profile_id, demoted)
            except Exception as exc:
                logger.debug("Tier evaluation skipped for %s: %s", profile_id, exc)

            # v3.6.6 F-5: Daily core-block recompile with hygiene.
            try:
                from superlocalmemory.core.block_hygiene import _recompile_core_blocks
                _recompile_core_blocks(self._db, self._config, profile_id)
            except Exception as exc:
                logger.debug("Core-block recompile skipped for %s: %s", profile_id, exc)

        # Re-read what is filed as a plan. The one-time pass runs as a
        # migration; the rule it uses keeps getting sharper, and a completed
        # migration is never replayed — so without this the store drifts
        # further from the rule with every release and nothing repairs it. The
        # pass is a pure function of the text and idempotent, so this is a no-op
        # once the store has converged.
        #
        # Once per cycle, not once per profile: it reads the whole table in one
        # sweep with no profile predicate, so running it per profile did the
        # identical global work N times over and took the write lock N times to
        # do it. It also takes and releases that lock per batch, so a memory
        # being saved waits for one batch rather than the whole sweep.
        #
        # An earlier version reached for `_conn` and then `.connection`, and the
        # database manager has neither, so the guard was always False and this
        # had never run once -- the store drifted further from the rule with
        # every release while a block that looks like it repairs that sat here
        # doing nothing.
        try:
            from superlocalmemory.storage.migrations import (
                M048_upcoming_holds_only_what_is_upcoming as _reclassify,
            )
            _reclassify.apply(open_connection=self._db.raw_connection)
            self._record_step("re-reading plans", True)
        except Exception as exc:  # noqa: BLE001
            self._record_step("re-reading plans", False, str(exc))

        # Anything that has failed on several cycles running is not a blip.
        # Every step here logs and continues, which is right -- one broken step
        # must not stop the rest -- but it meant a step that had been failing
        # for a week looked exactly like one that had just hiccupped, and the
        # daemon still reported itself healthy throughout.
        self._note_step_outcomes()

        # Retention. Three tables had a pruner each, written and wired
        # separately; the fourth unbounded table was found by reading a
        # disk-usage report and the fifth by reading the fourth. The policy for
        # every append-shaped table now lives in one registry and this enforces
        # all of them, so a table added without a policy is something the test
        # suite can see rather than something a person has to remember.
        #
        # Once per cycle, not per profile: every rule is keyed either on a row's
        # own age or on whether its referent still exists, and neither is
        # profile-scoped. Placed after the per-profile work so it sweeps rows
        # that pass orphaned -- pruning the graph and demoting tiers is what
        # leaves a lineage or temporal row without a referent.
        # In pieces, each taking and releasing the write lock. Entering
        # ``raw_connection`` is what takes that lock, so entering it once for
        # the whole sweep held it for the whole sweep: measured at 1,480 ms and
        # 123,888 rows on a 1 GB store, which is most of the budget a save is
        # allowed, spent waiting.
        try:
            from superlocalmemory.storage.retention_policy import (
                run_retention_bounded,
            )
            removed = run_retention_bounded(self._db.raw_connection)
            if removed:
                logger.info(
                    "Retention: %s",
                    ", ".join(
                        f"{table} -{count}" for table, count in sorted(removed.items())
                    ),
                )
            self._record_step("retention", True)
        except Exception as exc:  # noqa: BLE001
            self._record_step("retention", False, str(exc))

        # V3.4.10: Check if auto-backup is due
        try:
            from superlocalmemory.infra.backup import BackupManager
            manager = BackupManager(db_path=self._db.db_path)
            filename = manager.check_and_backup()
            if filename:
                logger.info("Auto-backup created: %s", filename)
                self._sync_cloud_destinations(manager)
        except Exception as exc:
            logger.debug("Auto-backup check skipped: %s", exc)

        try:
            from superlocalmemory.cli.pending_store import cleanup_stale
            stats = cleanup_stale()
            if stats["total"] > 0:
                logger.info("Pending cleanup: %s", stats)
        except Exception as exc:
            logger.debug("Pending cleanup skipped: %s", exc)

        # v3.8.5: GC the spreading-activation result cache. Neither cleanup path
        # was ever wired in, so activation_cache grew without bound (observed
        # 83k expired rows, oldest ~3.5 months). Batched + DB-wide (expiry is
        # not profile-scoped), so it runs once per cycle rather than per profile.
        try:
            deleted = self._db.cleanup_activation_cache()
            if deleted > 0:
                logger.info("Activation-cache GC: %d expired rows removed", deleted)
        except Exception as exc:
            logger.debug("Activation-cache GC skipped: %s", exc)

        self._schedule_next()

    def _profile_ids(self) -> tuple[str, ...]:
        """Return every persisted profile with a deterministic fallback."""
        try:
            rows = self._db.execute(
                "SELECT profile_id FROM profiles ORDER BY profile_id",
                (),
            )
            profiles = tuple(
                dict.fromkeys(str(row["profile_id"]) for row in rows if row["profile_id"])
            )
            if profiles:
                return profiles
        except Exception as exc:
            logger.debug("Profile enumeration failed: %s", exc)
        return (self._profile_id,)

    def _sync_cloud_destinations(self, manager: object) -> None:
        """Push latest backup to configured cloud destinations."""
        try:
            from superlocalmemory.infra.cloud_backup import sync_all_destinations
            sync_all_destinations(self._db.db_path)
        except ImportError:
            pass  # cloud_backup module not available yet
        except Exception as exc:
            logger.warning("Cloud sync failed (non-critical): %s", exc)

    def __del__(self) -> None:
        try:
            self.stop()
        except Exception:
            pass
