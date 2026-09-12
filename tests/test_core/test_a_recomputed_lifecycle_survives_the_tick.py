# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""A tier the maintenance pass computes must still be there after the pass.

WHAT 4.1.15 MISSED, AND WHY THE TEST SUITE DID NOT CATCH IT

4.1.15 fixed the tier computation (#136 part 2) and shipped it. On a real store
the fix then undid itself inside a single maintenance tick:

    M051 cleared the positions                              -> ok
    Langevin backfill recomputed lifecycle, active 111->2631 -> ok
    reconcile_profile_lifecycle ran later in the same tick   -> reverted to 111

The backfill writes ``atomic_facts.lifecycle`` and nothing else.
``fact_retention.lifecycle_zone`` is the authority, reconcile syncs
authority->mirror, and the authority still held the OLD values. So reconcile
did exactly its job and threw the new computation away.

``test_lifecycle_has_one_authority.py`` asserts reconcile syncs the right
DIRECTION, and it passes. It never asserts that a value maintenance just
computed SURVIVES that sync. Direction was the wrong invariant to pin on its
own -- this file pins persistence, which is what the user actually observes.

Every write site in ``run_maintenance`` has the same shape, so all four are
covered here rather than only the one that was reported.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from superlocalmemory.core.lifecycle_state import reconcile_profile_lifecycle
from superlocalmemory.core.maintenance import _persist_lifecycle
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
    # The state a real store is in before the pass: a stale zone the
    # maintenance pass is about to disagree with.
    for fid in ("f1", "f2", "f3"):
        conn.execute(
            "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content,"
            " lifecycle, scope, created_at) VALUES (?, 'm1', ?, ?, 'warm',"
            " 'global', '2026-08-01T00:00:00+00:00')",
            (fid, _PROFILE, f"content {fid}"),
        )
        conn.execute(
            "INSERT INTO fact_retention (fact_id, profile_id, lifecycle_zone,"
            " retention_score) VALUES (?, ?, 'warm', 0.42)", (fid, _PROFILE),
        )
    conn.commit()
    conn.close()
    return DatabaseManager(str(path))


def _lifecycle(db, fact_id):
    rows = db.execute("SELECT lifecycle FROM atomic_facts WHERE fact_id=?", (fact_id,))
    return str(dict(rows[0])["lifecycle"])


def _zone(db, fact_id):
    rows = db.execute(
        "SELECT lifecycle_zone FROM fact_retention WHERE fact_id=?", (fact_id,))
    return str(dict(rows[0])["lifecycle_zone"]) if rows else None


class TestWhatMaintenanceComputesIsWhatSurvives:
    def test_a_recomputed_tier_is_not_reverted_by_reconcile(self, store) -> None:
        """The exact defect: active 2,631 -> 111 inside one tick."""
        _persist_lifecycle(store, _PROFILE, [("f1", "active", None)])
        reconcile_profile_lifecycle(store, _PROFILE)
        assert _lifecycle(store, "f1") == "active", (
            "reconcile reverted a tier the maintenance pass had just computed"
        )

    def test_it_moves_the_authority_not_just_the_mirror(self, store) -> None:
        """Which is the whole reason the revert happened."""
        _persist_lifecycle(store, _PROFILE, [("f1", "active", None)])
        assert _zone(store, "f1") == "active"

    def test_archived_is_written_in_the_authority_s_spelling(self, store) -> None:
        """atomic says 'archived'; fact_retention says 'archive'."""
        _persist_lifecycle(store, _PROFILE, [("f1", "archived", None)])
        assert _zone(store, "f1") == "archive"
        assert _lifecycle(store, "f1") == "archived"
        reconcile_profile_lifecycle(store, _PROFILE)
        assert _lifecycle(store, "f1") == "archived"

    def test_a_mixed_batch_keeps_each_fact_in_its_own_tier(self, store) -> None:
        """The backfill computes a different tier per fact, not one per pass."""
        _persist_lifecycle(store, _PROFILE, [
            ("f1", "active", None), ("f2", "cold", None), ("f3", "archived", None),
        ])
        reconcile_profile_lifecycle(store, _PROFILE)
        assert (_lifecycle(store, "f1"), _lifecycle(store, "f2"),
                _lifecycle(store, "f3")) == ("active", "cold", "archived")

    def test_the_position_is_persisted_alongside(self, store) -> None:
        """Without it the backfill reruns every pass and never converges."""
        _persist_lifecycle(store, _PROFILE, [("f1", "active", [0.1] * 8)])
        rows = store.execute(
            "SELECT langevin_position FROM atomic_facts WHERE fact_id='f1'")
        assert dict(rows[0])["langevin_position"] is not None

    def test_an_empty_batch_is_a_no_op(self, store) -> None:
        _persist_lifecycle(store, _PROFILE, [])
        assert _lifecycle(store, "f1") == "warm"


class TestEveryMaintenanceWriteSiteUsesIt:
    def test_no_site_writes_the_mirror_alone(self) -> None:
        """All four sites had the same shape; fixing one would leave three.

        Walks the AST rather than grepping: a docstring that merely mentions
        the bad pattern is not a bad pattern, and this file's own explanation
        contains one.
        """
        import ast
        import inspect

        from superlocalmemory.core import maintenance

        tree = ast.parse(inspect.getsource(maintenance))
        offenders = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            fn = node.func
            if not (isinstance(fn, ast.Attribute) and fn.attr == "update_fact"):
                continue
            for arg in node.args:
                if isinstance(arg, ast.Dict):
                    keys = [
                        k.value for k in arg.keys
                        if isinstance(k, ast.Constant)
                    ]
                    if "lifecycle" in keys:
                        offenders.append(node.lineno)
        assert not offenders, (
            f"update_fact writes lifecycle directly at line(s) {offenders}; "
            "reconcile will revert those on the same tick — use "
            "_persist_lifecycle"
        )
