# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""A background pass that achieved nothing must not immediately run again.

GitHub #137. Two loops decided "keep going" from *activity* rather than from
*progress*, so work that could never succeed kept them at full speed forever.

1. The ingestion materializer slept only when
   ``not pending and not durable_complete and not durable_failed``. It runs one
   operation per pass, so a single operation that fails makes ``durable_failed``
   truthy, which suppressed the sleep -- and the next pass retried the same
   doomed operation with no delay.

2. The projection drain's inner loop repeated while
   ``handled + failed >= DEFAULT_BATCH``. A full batch of permanently failing
   rows satisfies that with ``handled == 0``, so it re-attempted the same 200
   rows without ever waiting, each iteration re-opening ~1000 SQLite
   connections and re-issuing 200 native vector writes.

Both now require forward progress. A failure is not progress.
"""

from __future__ import annotations

from superlocalmemory.core.projection_drain import _should_keep_draining
from superlocalmemory.server.unified_daemon import _materializer_should_idle


class TestTheMaterializer:
    def test_a_pass_that_only_failed_idles(self) -> None:
        """The #137 spin: one failing operation held the loop at full speed."""
        assert _materializer_should_idle(
            pending=[], durable_complete=0, durable_failed=1,
        ) is True

    def test_a_pass_that_completed_work_keeps_going(self) -> None:
        """Draining a real backlog fast is the behaviour worth keeping."""
        assert _materializer_should_idle(
            pending=[], durable_complete=1, durable_failed=0,
        ) is False

    def test_queued_work_keeps_going(self) -> None:
        assert _materializer_should_idle(
            pending=["op-1"], durable_complete=0, durable_failed=0,
        ) is False

    def test_an_idle_queue_idles(self) -> None:
        assert _materializer_should_idle(
            pending=[], durable_complete=0, durable_failed=0,
        ) is True

    def test_progress_alongside_failure_still_progresses(self) -> None:
        """A mixed batch is still draining; only pure failure is not."""
        assert _materializer_should_idle(
            pending=[], durable_complete=3, durable_failed=2,
        ) is False


class TestTheProjectionDrain:
    def test_a_full_batch_of_failures_stops_the_inner_loop(self) -> None:
        """handled == 0 is not progress, however full the batch was."""
        assert _should_keep_draining(handled=0, failed=200, batch=200) is False

    def test_a_full_batch_that_was_handled_keeps_draining(self) -> None:
        """A real backlog must still drain continuously, not one tick a batch."""
        assert _should_keep_draining(handled=200, failed=0, batch=200) is True

    def test_a_partial_batch_stops(self) -> None:
        assert _should_keep_draining(handled=12, failed=0, batch=200) is False

    def test_partial_progress_in_a_full_batch_keeps_draining(self) -> None:
        """Some rows failing must not strand the ones that would succeed."""
        assert _should_keep_draining(handled=150, failed=50, batch=200) is True


class TestTheLoopsStillExistWhereTheirThreadsLookForThem:
    """A pure-predicate test passes even if the loop it guards is gone.

    Extracting these predicates moved code around two long class bodies. An
    editing slip that lands a module-level ``def`` inside a class body makes
    every method below it dead code after a ``return`` -- the module still
    imports, the predicate tests still pass, and the daemon starts a thread
    whose target no longer exists. This is the assertion that notices.
    """

    def test_the_drain_worker_target_is_a_method(self) -> None:
        from superlocalmemory.core.projection_drain import ProjectionDrain

        assert callable(getattr(ProjectionDrain, "_run", None))
        assert callable(getattr(ProjectionDrain, "drain_once", None))
        assert callable(getattr(ProjectionDrain, "stop", None))
        assert callable(getattr(ProjectionDrain, "notify", None))

    def test_the_predicates_are_module_level(self) -> None:
        import ast
        import inspect

        from superlocalmemory.core import projection_drain
        from superlocalmemory.server import unified_daemon

        for module, name in (
            (projection_drain, "_should_keep_draining"),
            (unified_daemon, "_materializer_should_idle"),
        ):
            tree = ast.parse(inspect.getsource(module))
            top = {
                node.name for node in tree.body
                if isinstance(node, ast.FunctionDef)
            }
            assert name in top, f"{name} is not a module-level function"
