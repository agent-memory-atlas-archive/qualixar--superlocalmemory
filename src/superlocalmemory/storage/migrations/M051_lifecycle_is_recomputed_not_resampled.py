# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Discard the lifecycle positions that were thermal noise.

WHAT WAS WRONG

``core/maintenance.py`` seeded every fact's ``langevin_position`` from an
equilibrium radius that could not reach the ACTIVE band, then ran fifty
unseeded Euler-Maruyama steps and wrote the resulting band into
``atomic_facts.lifecycle``. Measured:

    seed radius, over the ENTIRE input domain      [0.5210, 0.6330]
    radius required for ACTIVE                     < 0.30
    total signal across the whole metadata domain     0.1120
    sd of the 50-step burn-in alone                   0.1225
    signal / noise                                    0.91

400 identical brand-new facts landed cold 50.5%, archived 44.2%, warm 5.2%,
active 0.0%. GitHub #136 reported the visible symptom: three corrections
applied 31 seconds apart put their successors in three different tiers.

The consequence is not cosmetic. ``lifecycle`` gates ``server/routes/timeline``,
``server/routes/insights``, ``learning/pattern_miner`` and
``core/consolidation_engine``, all of which select ``lifecycle = 'active'``. On
the author's store that was 2 facts out of 5,560.

WHY NULL AND NOT A RECOMPUTE HERE

The backfill in ``run_maintenance`` only touches facts whose
``langevin_position`` IS NULL, so a store that already has positions would keep
its noise-derived tiers forever -- the fix alone repairs nothing that already
exists. Clearing the column hands the recomputation to that same backfill,
which is the tested path and now derives the radius from the store's own
forgetting curve. Duplicating the formula here would be a second copy to drift.

``lifecycle`` is deliberately left alone. It is corrected on the next
maintenance pass, which runs at daemon start; until then a store keeps showing
what it showed yesterday rather than being blanked to a value that is equally
untrue. Measured effect of the recomputation on the author's store:

    tier        before            after
    active         2 ( 0.0%)     4412 (79.3%)
    warm          61 ( 1.1%)      947 (17.0%)
    cold         294 ( 5.3%)      196 ( 3.5%)
    archived    5204 (93.6%)        6 ( 0.1%)
"""

from __future__ import annotations

import logging
import sqlite3

logger = logging.getLogger(__name__)

NAME = "M051_lifecycle_is_recomputed_not_resampled"
DB_TARGET = "memory"
_TABLE = "atomic_facts"

# Kept so the catalogue and the DDL-drift hash have something to read. The
# runner prefers apply() when a module defines one, and apply() is what runs.
DDL = """
BEGIN IMMEDIATE;
UPDATE atomic_facts SET langevin_position = NULL
 WHERE langevin_position IS NOT NULL;
COMMIT;
"""


def apply(conn: sqlite3.Connection) -> None:
    """Clear the positions, or do nothing if this store has none to clear.

    ``langevin_position`` is created by ``storage/schema.create_all_tables`` at
    engine init, not by any migration, so there is no migration to depend on
    and no guarantee the column exists when this runs. A store the engine has
    never opened has neither the column nor a lifecycle problem. The same rule
    the visibility clauses follow: an absent column means no work, never an
    exception on every start.
    """
    try:
        columns = {
            str(dict(zip([c[0] for c in conn.execute(
                "PRAGMA table_info(atomic_facts)").description], row)).get("name"))
            for row in conn.execute("PRAGMA table_info(atomic_facts)")
        }
    except sqlite3.Error:
        logger.debug("M051: atomic_facts unreadable; nothing to clear")
        return
    if "langevin_position" not in columns:
        logger.debug("M051: no langevin_position column; nothing to clear")
        return
    cursor = conn.execute(
        "UPDATE atomic_facts SET langevin_position = NULL "
        "WHERE langevin_position IS NOT NULL"
    )
    logger.info("M051: cleared %d noise-derived lifecycle positions",
                cursor.rowcount)

REPAIR_NOT_APPLICABLE = (
    "clears a recomputable column so the maintenance backfill reseeds it; "
    "a repair would re-clear positions the backfill has since computed "
    "correctly, i.e. it would fight the very pass this migration hands the "
    "work to. Re-running the DDL is safe but pointless: the column being "
    "populated again is the success condition, not drift"
)
