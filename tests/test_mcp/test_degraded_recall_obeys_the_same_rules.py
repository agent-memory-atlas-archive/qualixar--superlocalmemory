# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""The fire-alarm path is still a read path.

``_sqlite_emergency_recall`` runs when the daemon is unreachable, on the
``session_init`` surface -- so its output is injected into an agent's opening
context. It hand-rolls SQL against ``atomic_facts_fts`` with a bare connection
and, before this file, filtered on ``profile_id`` and age and nothing else.

That put it outside BOTH guarantees at once. ``get_facts_by_ids`` calls itself
"THE PLACE QUARANTINE IS ENFORCED, and the only one"
(storage/database.py) -- this path never calls it.

The docstring on ``visible_fact_clause_for_connection`` asserts that
``mcp/tools_active.py`` "was already clean". It was not, and it says so in a
comment that also warns "grep is not a sufficient test for this". Both reads in
that module were unfiltered. This file is the test that grep could not be.

A degraded answer is allowed to be worse. It is not allowed to be unsafe.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from superlocalmemory.storage.migrations import M011_archive_and_merge
from superlocalmemory.storage.schema import create_all_tables

_PROFILE = "default"
_TOKEN = "runbook"


@pytest.fixture()
def store_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    conn = sqlite3.connect(str(tmp_path / "memory.db"))
    create_all_tables(conn)
    # archive_status arrives with M011, not with create_all_tables.
    conn.executescript(M011_archive_and_merge.DDL)
    conn.execute(
        "INSERT INTO memories (memory_id, profile_id, content) "
        "VALUES ('m1', ?, 'source')", (_PROFILE,),
    )
    rows = [
        ("keep-live", f"The {_TOKEN} lists two approvers.", 0, "live"),
        ("hide-quarantined",
         f"Unfortunately there is no information about the {_TOKEN}.", 1, "live"),
        ("hide-archived", f"The {_TOKEN} was archived last quarter.", 0, "archived"),
        ("hide-retired", f"The {_TOKEN} lives in the old wiki space.", 0, "live"),
    ]
    for fid, content, quarantined, archive_status in rows:
        conn.execute(
            "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content,"
            " quarantined, archive_status, scope, created_at) VALUES "
            "(?, 'm1', ?, ?, ?, ?, 'global', datetime('now'))",
            (fid, _PROFILE, content, quarantined, archive_status),
        )
    conn.execute(
        "INSERT INTO fact_temporal_validity (fact_id, profile_id,"
        " system_created_at, system_expired_at, invalidation_reason) "
        "VALUES ('hide-retired', ?, datetime('now'), datetime('now'),"
        " 'LLM-verified contradiction')", (_PROFILE,),
    )
    conn.commit()
    conn.close()
    return tmp_path


def _ids(response) -> list[str]:
    return sorted(item.fact.fact_id for item in response.results)


def test_the_fire_alarm_path_hides_what_every_other_path_hides(
    store_dir: Path,
) -> None:
    """Quarantined, archived and retired rows are all out of bounds here too."""
    from superlocalmemory.mcp.tools_active import _sqlite_emergency_recall

    got = _ids(_sqlite_emergency_recall(_TOKEN, limit=20, profile_id=_PROFILE))
    assert got == ["keep-live"], (
        f"degraded recall served rows no other path would: {got}"
    )


def test_it_still_answers(store_dir: Path) -> None:
    """A filter that returns nothing would pass the test above vacuously."""
    from superlocalmemory.mcp.tools_active import _sqlite_emergency_recall

    assert _sqlite_emergency_recall(_TOKEN, limit=20, profile_id=_PROFILE).results
