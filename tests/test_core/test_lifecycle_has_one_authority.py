# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""Retention state is the authority; ``atomic_facts.lifecycle`` mirrors it.

GitHub #136 reported tiers that changed on their own. OBSERVED on the author's
live store, 39 minutes apart, with no user activity:

    11:10   archived 5204 | cold 292 | warm  62 | active   2   -> 3,963 DISAGREE
    11:49   warm  4054 | archived 986 | cold 408 | active 112  ->     0 disagree

The obvious reading is that ``reconcile_profile_lifecycle`` syncing both ways
caused it, and 4.1.15 briefly "fixed" it by making the sync one-way from the
mirror. That was wrong twice over:

  - It broke the decay path. ``POST /api/v3/forgetting/run`` computes
    ``lifecycle_zone`` from ``retention_score`` and then calls reconcile
    precisely to push that into the mirror. Running it the other way made the
    route throw away its own work -- caught by
    ``tests/test_api/test_api_v33.py::TestForgettingRun``.
  - It was not the cause. The oscillation came from the Langevin backfill
    writing a RANDOM ``lifecycle`` for every NULL-position fact, which
    reconcile then faithfully propagated. Fix the writer, not the mirror.

So this file pins two things: the direction, and the property that actually
makes the flapping impossible -- the backfill is deterministic and touches a
fact once, so there is nothing left oscillating for reconcile to carry.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from superlocalmemory.core.lifecycle_state import reconcile_profile_lifecycle
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.schema import create_all_tables

_PROFILE = "default"


@pytest.fixture()
def store(tmp_path: Path) -> DatabaseManager:
    path = tmp_path / "memory.db"
    conn = sqlite3.connect(str(path))
    create_all_tables(conn)
    conn.execute(
        "INSERT INTO memories (memory_id, profile_id, content) "
        "VALUES ('m1', ?, 'source')", (_PROFILE,),
    )
    rows = [
        ("decayed-to-archive", "active", "archive"),
        ("decayed-to-cold", "active", "cold"),
        ("agree", "warm", "warm"),
        ("no-retention-row", "cold", None),
    ]
    for fid, lifecycle, zone in rows:
        conn.execute(
            "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content,"
            " lifecycle, scope, created_at) VALUES (?, 'm1', ?, ?, ?, 'global',"
            " '2026-08-01T00:00:00+00:00')",
            (fid, _PROFILE, f"content for {fid}", lifecycle),
        )
        if zone is not None:
            conn.execute(
                "INSERT INTO fact_retention (fact_id, profile_id,"
                " lifecycle_zone, retention_score) VALUES (?, ?, ?, 0.42)",
                (fid, _PROFILE, zone),
            )
    conn.commit()
    conn.close()
    return DatabaseManager(str(path))


def _lifecycle(db: DatabaseManager, fact_id: str) -> str:
    rows = db.execute(
        "SELECT lifecycle FROM atomic_facts WHERE fact_id = ?", (fact_id,))
    return str(dict(rows[0])["lifecycle"])


def _zone(db: DatabaseManager, fact_id: str) -> str | None:
    rows = db.execute(
        "SELECT lifecycle_zone FROM fact_retention WHERE fact_id = ?", (fact_id,))
    return str(dict(rows[0])["lifecycle_zone"]) if rows else None


class TestRetentionIsTheAuthority:
    def test_a_decayed_zone_reaches_the_mirror(self, store) -> None:
        """What /forgetting/run calls reconcile FOR."""
        reconcile_profile_lifecycle(store, _PROFILE)
        assert _lifecycle(store, "decayed-to-cold") == "cold"

    def test_archive_maps_to_the_mirrors_spelling(self, store) -> None:
        """``fact_retention`` says 'archive'; ``atomic_facts`` says 'archived'.

        Two vocabularies for one tier is how a mapping bug gets written.
        """
        reconcile_profile_lifecycle(store, _PROFILE)
        assert _lifecycle(store, "decayed-to-archive") == "archived"

    def test_the_decayed_zone_itself_survives(self, store) -> None:
        """The regression that inverting this caused: the row vanished.

        ``test_run_forgetting_does_not_touch_archived`` reads this row back
        after the route runs and fails with TypeError when it is gone.
        """
        reconcile_profile_lifecycle(store, _PROFILE)
        assert _zone(store, "decayed-to-archive") == "archive"
        assert _zone(store, "decayed-to-cold") == "cold"

    def test_a_fact_with_no_retention_row_is_imported_from_the_mirror(
        self, store,
    ) -> None:
        """A legacy row with only a mirror value must not be stranded."""
        reconcile_profile_lifecycle(store, _PROFILE)
        assert _zone(store, "no-retention-row") == "cold"

    def test_it_is_idempotent(self, store) -> None:
        reconcile_profile_lifecycle(store, _PROFILE)
        snapshot = {
            fid: (_lifecycle(store, fid), _zone(store, fid))
            for fid in ("decayed-to-archive", "decayed-to-cold",
                        "agree", "no-retention-row")
        }
        assert reconcile_profile_lifecycle(store, _PROFILE) == 0
        assert {
            fid: (_lifecycle(store, fid), _zone(store, fid)) for fid in snapshot
        } == snapshot

    def test_it_does_not_clobber_the_retention_score(self, store) -> None:
        """Only the zone is mirrored. The score is measured elsewhere."""
        reconcile_profile_lifecycle(store, _PROFILE)
        rows = store.execute(
            "SELECT retention_score FROM fact_retention WHERE fact_id = ?",
            ("decayed-to-cold",))
        assert float(dict(rows[0])["retention_score"]) == pytest.approx(0.42)


class TestNothingIsLeftOscillating:
    """The property that actually makes #136's flapping impossible."""

    def test_the_seed_is_stable_across_repeated_passes(self) -> None:
        """The backfill used to hand reconcile a fresh random tier each time.

        Reconcile propagated it faithfully, which is what made the mirror and
        the zone appear to chase each other. Two passes, same answer, means
        there is no oscillation for any direction of sync to carry.
        """
        import numpy as np

        from superlocalmemory.core.maintenance import (
            _LANGEVIN_DIM,
            _seed_langevin_position,
        )

        radii = {
            round(float(np.linalg.norm(_seed_langevin_position(
                2, 45.0, 0.5, 0.3, _LANGEVIN_DIM, fact_id="fact-abc"))), 12)
            for _ in range(30)
        }
        assert len(radii) == 1, f"the seed still varies between passes: {radii}"
