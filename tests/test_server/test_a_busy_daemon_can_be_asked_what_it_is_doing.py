# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
"""The daemon can be asked what it is doing without root and without tooling.

GitHub #137 was a daemon at 100% CPU, and on both machines it was investigated
on, nobody could see inside the process:

  - macOS needs root for ``task_for_pid``, so py-spy is unavailable to any user
    who is not in sudoers. On the author's managed corporate Mac, ``sudo``
    refused outright. Eleven days of a pinned core, and the busiest Python
    thread was never identified.
  - The reporter on Ubuntu did get py-spy attached, and it was blind to the
    threads that actually mattered -- native tokio workers carry no Python
    frame -- so they fell back to ``sudo gdb -p`` against a stripped binary.

``faulthandler`` needs no root, no attach and no install. The process dumps its
own stacks on a signal it armed itself.

It shows PYTHON frames only. A native thread with no Python frame stays
invisible, exactly as it was to py-spy -- that limit is stated in the
function's docstring rather than discovered by the next person at 2am.
"""

from __future__ import annotations

import faulthandler
import os
import signal
from pathlib import Path

import pytest

from superlocalmemory.server.unified_daemon import install_thread_dump_signal

pytestmark = pytest.mark.skipif(
    not hasattr(signal, "SIGUSR1"), reason="platform has no SIGUSR1",
)


@pytest.fixture()
def armed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    path = install_thread_dump_signal()
    yield path
    try:
        faulthandler.unregister(signal.SIGUSR1)
    except Exception:
        pass


def test_it_arms_and_reports_where_the_dump_will_go(armed) -> None:
    assert armed is not None
    assert Path(armed).name == "thread-dump.log"


def test_the_signal_actually_writes_a_stack(armed) -> None:
    """The whole point: a running process, asked, answers."""
    os.kill(os.getpid(), signal.SIGUSR1)
    text = Path(armed).read_text()
    assert "Thread" in text or "File " in text, (
        f"SIGUSR1 produced no stack; dump was {text[:200]!r}"
    )
    assert "test_the_signal_actually_writes_a_stack" in text, (
        "the dump does not name the frame that was executing"
    )


def test_it_can_be_asked_more_than_once(armed) -> None:
    """A CPU investigation is a sequence of samples, not one look."""
    os.kill(os.getpid(), signal.SIGUSR1)
    first = Path(armed).read_text()
    os.kill(os.getpid(), signal.SIGUSR1)
    second = Path(armed).read_text()
    assert len(second) > len(first), "the second dump did not append"


def test_arming_never_prevents_the_daemon_starting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A diagnostic that can block startup is worse than no diagnostic.

    The failure is injected at ``faulthandler.register`` rather than by
    monkeypatching ``builtins.open``. Replacing ``open`` globally inside an
    11,000-test session breaks any unrelated machinery that reads a file while
    the patch is live, which is a large blast radius for a small assertion.
    """
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))

    def _explode(*_args, **_kwargs):
        raise OSError("cannot arm handler")

    monkeypatch.setattr(faulthandler, "register", _explode)
    assert install_thread_dump_signal() is None
