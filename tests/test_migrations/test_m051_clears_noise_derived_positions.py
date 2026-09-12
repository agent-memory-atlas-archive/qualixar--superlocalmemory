# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""M051 hands the lifecycle recomputation back to the maintenance backfill.

The fix for #136 part 2 changed how a position is seeded. It repairs nothing
that already exists, because the backfill only touches facts whose
``langevin_position`` IS NULL -- and on any store that has been running, every
fact already has one. On the author's store that was 5,558 of 5,560 facts
holding a tier drawn from thermal noise, with 93.6% marked archived.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from superlocalmemory.storage.migrations import (
    M051_lifecycle_is_recomputed_not_resampled as m051,
)
from superlocalmemory.storage.schema import create_all_tables

_PROFILE = "default"


@pytest.fixture()
def conn(tmp_path: Path) -> sqlite3.Connection:
    c = sqlite3.connect(str(tmp_path / "memory.db"))
    create_all_tables(c)
    c.execute(
        "INSERT INTO memories (memory_id, profile_id, content) "
        "VALUES ('m1', ?, 'source')", (_PROFILE,),
    )
    for fid, pos, lifecycle in (
        ("noisy-1", json.dumps([0.6] * 8), "archived"),
        ("noisy-2", json.dumps([0.2] * 8), "cold"),
        ("never-seeded", None, "active"),
    ):
        c.execute(
            "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content,"
            " langevin_position, lifecycle, scope, created_at) VALUES "
            "(?, 'm1', ?, ?, ?, ?, 'global', '2026-08-01T00:00:00+00:00')",
            (fid, _PROFILE, f"content {fid}", pos, lifecycle),
        )
    c.commit()
    return c


def _positions(c: sqlite3.Connection) -> dict[str, object]:
    return {
        r[0]: r[1]
        for r in c.execute("SELECT fact_id, langevin_position FROM atomic_facts")
    }


def test_it_clears_every_position_so_the_backfill_recomputes(conn) -> None:
    assert sum(v is not None for v in _positions(conn).values()) == 2
    conn.executescript(m051.DDL)
    assert all(v is None for v in _positions(conn).values())


def test_it_leaves_the_lifecycle_column_alone(conn) -> None:
    """Corrected on the next maintenance pass, not blanked to another untruth."""
    before = dict(conn.execute("SELECT fact_id, lifecycle FROM atomic_facts"))
    conn.executescript(m051.DDL)
    assert dict(conn.execute("SELECT fact_id, lifecycle FROM atomic_facts")) == before


def test_it_is_idempotent(conn) -> None:
    conn.executescript(m051.DDL)
    conn.executescript(m051.DDL)
    assert all(v is None for v in _positions(conn).values())


def test_it_is_registered_where_atomic_facts_migrations_belong() -> None:
    """atomic_facts is bootstrapped at engine init, so it defers -- like M011."""
    from superlocalmemory.storage.migration_runner import (
        DEFERRED_MIGRATIONS,
        MIGRATIONS,
    )

    assert any(m.name == m051.NAME for m in DEFERRED_MIGRATIONS)
    assert not any(m.name == m051.NAME for m in MIGRATIONS)
