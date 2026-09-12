# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""Replacing a backend orchestrator must retire the one it replaced.

GitHub #137. ``LanceDBVectorBackend.close()`` existed and was never called
anywhere in ``src/``. ``set_orchestrator`` overwrote a module global without
stopping the previous orchestrator, and ``_hot_reconfigure_engine`` builds a
new one on every reconfigure and profile switch. Each LanceDB connection owns
a Rust tokio runtime sized to the host's cores, so every replacement leaked a
whole thread pool -- plus an orphaned drain worker still writing into the
projection that had just been swapped out.

MEASURED on the author's machine, one daemon, uptime 11d13h, 14 cores:

    daemon CPU                     89-97%, sustained
    tokio-rt-worker threads        99      (~7 runtimes on a 14-core box)
      one generation at start      14
      later generations            85      <- the leak, visible as thread ids

``BackendOrchestrator.stop()`` only ever stopped the drain thread, so even the
one caller that did shut down cleanly still leaked the native runtime.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from superlocalmemory.core.backend_orchestrator import (
    BackendOrchestrator,
    get_orchestrator,
    set_orchestrator,
)
from superlocalmemory.core.config import SLMConfig
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.schema import create_all_tables


class _RecordingBackend:
    """Stands in for the native handle that owns a tokio runtime."""

    def __init__(self) -> None:
        self.closed = 0

    def close(self) -> None:
        self.closed += 1


def _orchestrator(tmp_path: Path, name: str) -> BackendOrchestrator:
    db_path = tmp_path / f"{name}.db"
    conn = sqlite3.connect(str(db_path))
    create_all_tables(conn)
    conn.commit()
    conn.close()
    config = SLMConfig()
    config.base_dir = str(tmp_path)
    config.data_dir = str(tmp_path)
    orch = BackendOrchestrator(config=config, db=DatabaseManager(str(db_path)))
    orch._lancedb = _RecordingBackend()
    return orch


@pytest.fixture(autouse=True)
def _restore_global_orchestrator():
    previous = get_orchestrator()
    yield
    set_orchestrator.__globals__["_orchestrator"] = previous


def test_replacing_the_orchestrator_closes_the_one_it_replaced(
    tmp_path: Path,
) -> None:
    """The leak itself: 7 runtimes alive where 1 should be."""
    first = _orchestrator(tmp_path, "first")
    second = _orchestrator(tmp_path, "second")
    set_orchestrator(first)
    set_orchestrator(second)
    assert first._lancedb is None or first._lancedb.closed == 1, (
        "the replaced orchestrator kept its native runtime alive"
    )


def test_stop_releases_the_native_backend_not_just_the_drain(
    tmp_path: Path,
) -> None:
    """Even a clean shutdown leaked the runtime, because stop() ignored it."""
    orch = _orchestrator(tmp_path, "solo")
    backend = orch._lancedb
    orch.stop()
    assert backend.closed == 1, "stop() left the native backend open"


def test_stop_is_idempotent(tmp_path: Path) -> None:
    """Shutdown races call this twice; the second must not raise."""
    orch = _orchestrator(tmp_path, "twice")
    backend = orch._lancedb
    orch.stop()
    orch.stop()
    assert backend.closed == 1


def test_setting_the_same_orchestrator_twice_does_not_close_it(
    tmp_path: Path,
) -> None:
    """A no-op re-set must not shut down the live backend under the daemon."""
    orch = _orchestrator(tmp_path, "same")
    set_orchestrator(orch)
    set_orchestrator(orch)
    assert orch._lancedb is not None and orch._lancedb.closed == 0


def test_a_backend_that_raises_on_close_does_not_break_replacement(
    tmp_path: Path,
) -> None:
    """Releasing the old one must never stop the new one being installed."""
    first = _orchestrator(tmp_path, "raises")

    class _Hostile:
        def close(self) -> None:
            raise RuntimeError("native handle already torn down")

    first._lancedb = _Hostile()
    second = _orchestrator(tmp_path, "successor")
    set_orchestrator(first)
    set_orchestrator(second)
    assert get_orchestrator() is second
