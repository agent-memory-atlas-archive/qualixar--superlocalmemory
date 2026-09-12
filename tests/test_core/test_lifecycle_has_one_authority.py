# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""Two columns holding the same fact must not each be the other's source.

GitHub #136. ``reconcile_profile_lifecycle`` synced BOTH ways in one
transaction: ``atomic_facts.lifecycle`` seeded ``fact_retention.lifecycle_zone``
for any fact without a retention row, and ``lifecycle_zone`` then overwrote
``lifecycle`` for every fact that had one. With the maintenance pass writing
``lifecycle`` on its own schedule, the two columns chased each other.

OBSERVED on the author's live store, 39 minutes apart, no user activity:

    11:10   archived 5204 | cold 292 | warm  62 | active   2   -> 3,963 rows DISAGREE
    11:49   warm  4054 | archived 986 | cold 408 | active 112  ->     0 rows disagree

Every subsystem filtering ``lifecycle = 'active'`` -- timeline, insights,
pattern_miner, consolidation_engine -- therefore got a different answer
depending on when it asked.

One direction now: the Langevin position decides ``lifecycle``, and
``lifecycle_zone`` mirrors it. The position is the only state with a physical
meaning; the two columns are views of it.
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
    # Each pair disagrees, in a different direction.
    rows = [
        ("disagree-1", "cold", "warm"),
        ("disagree-2", "warm", "archive"),
        ("disagree-3", "active", "cold"),
        ("agree-1", "warm", "warm"),
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
        "SELECT lifecycle FROM atomic_facts WHERE fact_id = ?", (fact_id,),
    )
    return str(dict(rows[0])["lifecycle"])


def _zone(db: DatabaseManager, fact_id: str) -> str | None:
    rows = db.execute(
        "SELECT lifecycle_zone FROM fact_retention WHERE fact_id = ?", (fact_id,),
    )
    return str(dict(rows[0])["lifecycle_zone"]) if rows else None


class TestTheAtomicMirrorIsTheSource:
    def test_reconcile_never_rewrites_the_lifecycle_column(
        self, store: DatabaseManager,
    ) -> None:
        """The leg that caused the flapping."""
        before = {
            fid: _lifecycle(store, fid)
            for fid in ("disagree-1", "disagree-2", "disagree-3",
                        "agree-1", "no-retention-row")
        }
        reconcile_profile_lifecycle(store, _PROFILE)
        after = {fid: _lifecycle(store, fid) for fid in before}
        assert after == before, f"reconcile moved the authority: {before} -> {after}"

    def test_the_zone_is_brought_into_line_with_it(
        self, store: DatabaseManager,
    ) -> None:
        reconcile_profile_lifecycle(store, _PROFILE)
        assert _zone(store, "disagree-1") == "cold"
        assert _zone(store, "disagree-2") == "warm"
        assert _zone(store, "disagree-3") == "active"

    def test_a_fact_with_no_retention_row_gets_one(
        self, store: DatabaseManager,
    ) -> None:
        reconcile_profile_lifecycle(store, _PROFILE)
        assert _zone(store, "no-retention-row") == "cold"

    def test_archived_maps_to_the_retention_spelling(
        self, tmp_path: Path,
    ) -> None:
        """``atomic_facts`` says 'archived'; ``fact_retention`` says 'archive'.

        Two vocabularies for one tier is how a mapping bug gets written.
        """
        path = tmp_path / "m.db"
        conn = sqlite3.connect(str(path))
        create_all_tables(conn)
        conn.execute(
            "INSERT INTO memories (memory_id, profile_id, content) "
            "VALUES ('m1', ?, 's')", (_PROFILE,),
        )
        conn.execute(
            "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content,"
            " lifecycle, scope, created_at) VALUES ('a1','m1',?,'x','archived',"
            " 'global','2026-08-01T00:00:00+00:00')", (_PROFILE,),
        )
        conn.commit()
        conn.close()
        db = DatabaseManager(str(path))
        reconcile_profile_lifecycle(db, _PROFILE)
        assert _zone(db, "a1") == "archive"

    def test_it_is_idempotent(self, store: DatabaseManager) -> None:
        """A second pass must find nothing left to change."""
        reconcile_profile_lifecycle(store, _PROFILE)
        snapshot = {
            fid: (_lifecycle(store, fid), _zone(store, fid))
            for fid in ("disagree-1", "disagree-2", "disagree-3",
                        "agree-1", "no-retention-row")
        }
        assert reconcile_profile_lifecycle(store, _PROFILE) == 0
        assert {
            fid: (_lifecycle(store, fid), _zone(store, fid)) for fid in snapshot
        } == snapshot

    def test_it_does_not_clobber_the_retention_score(
        self, store: DatabaseManager,
    ) -> None:
        """Only the zone is a mirror. The score is measured elsewhere."""
        reconcile_profile_lifecycle(store, _PROFILE)
        rows = store.execute(
            "SELECT retention_score FROM fact_retention WHERE fact_id = ?",
            ("disagree-1",),
        )
        assert float(dict(rows[0])["retention_score"]) == pytest.approx(0.42)
