"""Cross-process persistence helpers for the BM25 keyword index."""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl


@contextmanager
def index_file_lock(
    lock_path: Path,
    *,
    exclusive: bool,
    timeout_seconds: float = 30.0,
) -> Iterator[None]:
    """Acquire a cross-process lock for keyword index read/write operations."""
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_file:
        deadline = time.monotonic() + timeout_seconds
        while True:
            try:
                _acquire_lock(lock_file, exclusive=exclusive)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    msg = f"Timed out acquiring index lock: {lock_path}"
                    raise TimeoutError(msg) from None
                time.sleep(0.05)
        try:
            yield
        finally:
            _release_lock(lock_file)


def atomic_write_text(path: Path, content: str) -> None:
    """Write text atomically so concurrent readers never see partial files."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f"{path.suffix}.tmp.{os.getpid()}")
    try:
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _acquire_lock(lock_file: Any, *, exclusive: bool) -> None:
    if sys.platform == "win32":
        lock_file.seek(0)
        mode = msvcrt.LK_NBLCK if exclusive else msvcrt.LK_NBRLCK
        msvcrt.locking(lock_file.fileno(), mode, 1)
        return

    mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
    fcntl.flock(lock_file.fileno(), mode | fcntl.LOCK_NB)


def _release_lock(lock_file: Any) -> None:
    if sys.platform == "win32":
        lock_file.seek(0)
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        return

    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
