# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""A fact the store has already decided is wrong must not be served as current.

WHY THIS FILE EXISTS, WHICH IS THE POINT OF IT
----------------------------------------------
This is ``test_no_read_path_shows_a_withheld_row.py`` happening a second time,
for a different guarantee, in the same shape.

4.0.10 established that a *withheld* row must not reach a display path, and
enumerated the paths. Supersession is a separate guarantee with a separate
predicate, and it was never given the same treatment. ``review_correction``
writes the retirement to ``fact_temporal_validity.system_expired_at``
(storage/correction_cases.py). ``recall()`` honours it, through
``get_correction_inadmissible_fact_ids`` at ``retrieval/engine.py``. Nothing
else does, because ``visible_fact_clause`` reads only ``archive_status`` and
``quarantined`` -- it has never known that this table exists.

GitHub #136. Measured on the author's real store before the fix:

    facts with system_expired_at set                    261
    of those, still visible to every display path       258
    from corrections                                      7
    from the contradiction / sheaf pipeline             251
    one representative search: 967 hits -> 920 with the fix

So 4.9% of a search result set was facts the store had already decided were
wrong, and correcting a memory made retrieval worse than leaving it alone --
the predecessor keeps its full BM25 weight, because the FTS triggers fire on
INSERT / DELETE / ``UPDATE OF content`` and apply does none of those.

``ORDER BY fts.rank`` carries no time term, which is why the reporter's
re-measurement at 5 minutes and at 40 minutes was identical. Waiting cannot
change this result. Only a predicate can.

If you add a read path that answers "what do I currently know", add it here.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.schema import create_all_tables

_PROFILE = "default"
_SHARED_TOKEN = "deployment"

# Retired by an explicit, reviewed correction -- GitHub #136's own repro.
_CORRECTED_TEXT = f"The {_SHARED_TOKEN} target is Tuesday at 14:00 UTC."
# Retired by the contradiction / sheaf pipeline -- 251 of the author's 261.
_CONTRADICTED_TEXT = f"The {_SHARED_TOKEN} runbook lives in the old wiki space."
# The correction that replaced the first one.
_SUCCESSOR_TEXT = f"The {_SHARED_TOKEN} target is Thursday at 09:00 UTC."
# Never retired by anything.
_LIVE_TEXT = f"The {_SHARED_TOKEN} checklist requires two approvals."


@pytest.fixture()
def store(tmp_path: Path) -> DatabaseManager:
    db = tmp_path / "memory.db"
    conn = sqlite3.connect(str(db))
    create_all_tables(conn)
    conn.execute(
        "INSERT INTO memories (memory_id, profile_id, content) "
        "VALUES ('m1', ?, 'source')", (_PROFILE,),
    )
    rows = [
        ("keep-live", _LIVE_TEXT, 0),
        ("keep-successor", _SUCCESSOR_TEXT, 1),
        ("retired-corrected", _CORRECTED_TEXT, 1),
        ("retired-contradicted", _CONTRADICTED_TEXT, 0),
    ]
    for fid, content, pinned in rows:
        conn.execute(
            "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content,"
            " quarantined, pinned, scope, created_at) "
            "VALUES (?, 'm1', ?, ?, 0, ?, 'global', '2026-08-01T00:00:00+00:00')",
            (fid, _PROFILE, content, pinned),
        )
    # Every fact gets a temporal row. Only the retired ones carry an expiry --
    # that asymmetry is the whole predicate under test.
    for fid in ("keep-live", "keep-successor", "retired-corrected",
                "retired-contradicted"):
        conn.execute(
            "INSERT INTO fact_temporal_validity (fact_id, profile_id,"
            " system_created_at) VALUES (?, ?, '2026-08-01T00:00:00+00:00')",
            (fid, _PROFILE),
        )
    conn.execute(
        "UPDATE fact_temporal_validity SET system_expired_at = ?,"
        " invalidated_by = ?, invalidation_reason = ? WHERE fact_id = ?",
        ("2026-08-02T10:00:00+00:00", "keep-successor",
         "direct_content_correction", "retired-corrected"),
    )
    conn.execute(
        "UPDATE fact_temporal_validity SET system_expired_at = ?,"
        " invalidation_reason = ? WHERE fact_id = ?",
        ("2026-08-02T10:00:00+00:00",
         "LLM-verified contradiction (sheaf pre-filter severity: 0.800)",
         "retired-contradicted"),
    )
    conn.commit()
    conn.close()
    return DatabaseManager(str(db))


def _retired(facts) -> list[str]:
    return sorted(f.fact_id for f in facts if f.fact_id.startswith("retired-"))


def _kept(facts) -> list[str]:
    return sorted(f.fact_id for f in facts if f.fact_id.startswith("keep-"))


def _all_pinned_ids_in_fixture() -> list[str]:
    """Names the rows the fixture pins, so the pin test cannot go vacuous."""
    return ["keep-successor", "retired-corrected"]


class TestEveryPathThatAnswersWhatDoINowKnow:
    def test_full_text_search(self, store: DatabaseManager) -> None:
        """GitHub #136: the `search` tool and the dashboard box land here."""
        facts = store.search_facts_fts(_SHARED_TOKEN, _PROFILE, limit=50)
        assert _retired(facts) == [], (
            "search served a fact the store already retired; this is #136"
        )
        assert _kept(facts), "search returned nothing, so this proves nothing"

    def test_a_contradicted_fact_is_retired_too(
        self, store: DatabaseManager,
    ) -> None:
        """251 of the author's 261 retirements came from contradiction, not
        from a reviewed correction. A predicate that only understood
        correction_cases would miss 96% of them, which is why this one reads
        the expiry itself.
        """
        facts = store.search_facts_fts(_SHARED_TOKEN, _PROFILE, limit=50)
        assert "retired-contradicted" not in [f.fact_id for f in facts]

    def test_pinned_context_injection(self, store: DatabaseManager) -> None:
        """The worst case: a pin is asserted as background truth for a session.

        A retired pin is not merely displayed -- it is handed to an agent as
        something the owner believes.
        """
        pinned = store.get_pinned(_PROFILE)
        # retired-corrected IS pinned in the fixture. Without that this test
        # passes with the fix reverted, which is how a vacuous test ships.
        assert "retired-corrected" in _all_pinned_ids_in_fixture()
        assert _retired(pinned) == [], "a retired fact was injected as a pin"
        assert _kept(pinned) == ["keep-successor"]


class TestTheEscapeHatchStillWorks:
    def test_history_can_still_see_a_retired_fact(
        self, store: DatabaseManager,
    ) -> None:
        """A retirement is an event in the record, not an erasure.

        Timeline, audit and export must still reach these rows. A fact nothing
        can read is a fact nothing can explain, and "why did my memory change"
        is a question this product has to be able to answer.
        """
        facts = store.get_facts_by_ids(
            ["retired-corrected", "retired-contradicted"], _PROFILE,
        )
        assert _retired(facts) == ["retired-contradicted", "retired-corrected"]


class TestTheClauseIsSharedRatherThanRepeated:
    def test_only_the_composer_hand_rolls_the_subquery(self) -> None:
        """Two copies of this predicate would drift. That is how #136 started.

        Counts the NOT-EXISTS *subquery* form specifically. The admission-set
        queries in ``get_correction_inadmissible_fact_ids`` also mention
        ``system_expired_at``, but they SELECT the retired ids rather than
        filtering a display query, so a raw token count would pin the wrong
        number and go green for the wrong reason.
        """
        import inspect

        from superlocalmemory.storage import database

        src = inspect.getsource(database)
        copies = src.count("SELECT 1 FROM fact_temporal_validity")
        assert copies == 1, (
            f"{copies} hand-rolled supersession subqueries in database.py; "
            "exactly one belongs, inside _compose_current_clause"
        )

    def test_search_and_pins_resolve_the_same_predicate(self) -> None:
        """The guarantee is that they cannot drift apart, not that both work."""
        import inspect

        from superlocalmemory.storage.database import DatabaseManager

        for name in ("search_facts_fts", "get_pinned"):
            body = inspect.getsource(getattr(DatabaseManager, name))
            assert "current_fact_clause" in body, (
                f"{name} does not resolve its predicate from current_fact_clause"
            )
            assert "fact_temporal_validity" not in body, (
                f"{name} hand-rolls the supersession predicate again"
            )

    def test_it_degrades_rather_than_raises_on_an_unmigrated_store(
        self, tmp_path: Path,
    ) -> None:
        """No table must mean no filter, never an exception on every read.

        Same rule the quarantine clause follows: a DatabaseManager can be
        pointed at a store engine init never touched, and filtering against an
        absent table would turn a cosmetic gap into total read failure.
        """
        db = tmp_path / "old.db"
        conn = sqlite3.connect(str(db))
        create_all_tables(conn)
        conn.execute(
            "INSERT INTO memories (memory_id, profile_id, content) "
            "VALUES ('m1', ?, 's')", (_PROFILE,),
        )
        conn.execute(
            "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content,"
            " created_at) VALUES ('f1','m1',?,?, '2026-01-01T00:00:00+00:00')",
            (_PROFILE, f"a {_SHARED_TOKEN} from before the table existed"),
        )
        conn.commit()
        conn.execute("DROP TABLE fact_temporal_validity")
        conn.commit()
        conn.close()

        mgr = DatabaseManager(str(db))
        facts = mgr.search_facts_fts(_SHARED_TOKEN, _PROFILE, limit=10)
        assert [f.fact_id for f in facts] == ["f1"]
