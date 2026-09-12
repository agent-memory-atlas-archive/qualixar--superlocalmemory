# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""A drain pass writes vectors once, not once per memory.

GitHub #137's dominant cause. ``_project_vector`` called ``add_vectors`` for a
single fact, and ``add_vectors`` issues a LanceDB ``merge_insert`` -- a
whole-table operation that creates a NEW VERSION of the vector store. One
version per memory, forever, with nothing removing them.

MEASURED on the author's store: 5,561 memories, a 610 MB SQLite database, and
a **17 GB** vector store holding **50,580 versions**. The daemon held a core
at 90%+ with an empty queue and every Python thread parked, because each
vector operation walks that history.

Reproduced in miniature, 300 facts written one at a time:

    before  rows=300  versions=300  size=6.5M
    after   rows=300  versions=  1  size=128K

Batching turns a 200-fact pass into one version instead of 200. The ordering
that matters is unchanged: vectors are durable BEFORE any outbox row clears,
so a crash mid-pass re-queues the work rather than losing it.
"""

from __future__ import annotations

import pytest

from superlocalmemory.core.projection_drain import _VectorBatch


class _Vector:
    """Counts calls the way LanceDB counts versions: one per add_vectors."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def add_vectors(self, ids, embeddings, tiers, profile_id):
        self.calls.append((tuple(ids), tuple(tiers), profile_id))
        return len(ids)


class TestTheBatchCollapsesWrites:
    def test_two_hundred_facts_become_one_write(self) -> None:
        """The whole point: 200 versions -> 1."""
        vec, batch = _Vector(), _VectorBatch()
        for i in range(200):
            batch.add(f"f{i}", [0.1, 0.2], "active", "default")
        batch.flush(vec)
        assert len(vec.calls) == 1, f"made {len(vec.calls)} writes, expected 1"
        assert len(vec.calls[0][0]) == 200, "not every fact reached the write"

    def test_profiles_do_not_get_mixed(self) -> None:
        """add_vectors takes one profile_id, so a batch must split on it."""
        vec, batch = _Vector(), _VectorBatch()
        batch.add("a", [0.1], "active", "work")
        batch.add("b", [0.2], "warm", "personal")
        batch.add("c", [0.3], "active", "work")
        batch.flush(vec)
        by_profile = {c[2]: c[0] for c in vec.calls}
        assert by_profile == {"work": ("a", "c"), "personal": ("b",)}

    def test_tiers_stay_aligned_with_their_facts(self) -> None:
        """A shuffled tier list would silently mis-tier every vector."""
        vec, batch = _Vector(), _VectorBatch()
        batch.add("a", [0.1], "cold", "p")
        batch.add("b", [0.2], "active", "p")
        batch.flush(vec)
        ids, tiers, _ = vec.calls[0]
        assert dict(zip(ids, tiers)) == {"a": "cold", "b": "active"}

    def test_an_empty_pass_writes_nothing(self) -> None:
        """A no-op pass must not create a version either."""
        vec, batch = _Vector(), _VectorBatch()
        batch.flush(vec)
        assert vec.calls == []

    def test_flushing_twice_does_not_rewrite(self) -> None:
        """The buffer empties on flush; a second flush is a no-op."""
        vec, batch = _Vector(), _VectorBatch()
        batch.add("a", [0.1], "active", "p")
        batch.flush(vec)
        batch.flush(vec)
        assert len(vec.calls) == 1

    def test_no_vector_backend_is_not_an_error(self) -> None:
        """Graph-only deployments drain normally."""
        batch = _VectorBatch()
        batch.add("a", [0.1], "active", "p")
        batch.flush(None)
        assert batch.pending == 0
