"""Durable local records and exclusive ownership of a pipeline run."""

from __future__ import annotations

import json
import errno
import os
import sys
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any


class RunLocked(RuntimeError):
    pass


def write_json(path: Path, data: Any) -> None:
    write_text(path, json.dumps(data, ensure_ascii=False, indent=2))


def write_text(path: Path, text: str) -> None:
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def append_record(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_records(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r+b") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                return
            if not line.endswith(b"\n"):
                print(f"warning: recovering interrupted append in {path}; discarding {len(line)} tail bytes",
                      file=sys.stderr)
                stream.seek(offset)
                stream.truncate()
                stream.flush()
                os.fsync(stream.fileno())
                return
            if line.strip():
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(f"non-object record in {path} at byte {offset}")
                yield record


class RunLock:
    """A lock is never unlinked: all owners must continue locking the same inode."""

    def __init__(self, directory: Path):
        self.directory = directory.resolve()
        self.stream = None

    def __enter__(self) -> RunLock:
        self.directory.mkdir(parents=True, exist_ok=True)
        self.stream = (self.directory / ".run.lock").open("a+b")
        try:
            if os.name == "posix":
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            elif os.name == "nt":
                import msvcrt
                self.stream.seek(0)
                if not self.stream.read(1):
                    self.stream.write(b"\0")
                    self.stream.flush()
                self.stream.seek(0)
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                raise RuntimeError(f"run locking is not supported on {os.name}")
        except OSError as e:
            self.stream.close()
            self.stream = None
            if e.errno not in (errno.EAGAIN, errno.EACCES):
                raise RuntimeError(f"cannot lock {self.directory}: {e}") from e
            raise RunLocked(f"{self.directory} is owned by another pipeline or its workers") from e
        return self

    def fileno(self) -> int:
        if self.stream is None:
            raise RuntimeError("run lock is not held")
        return self.stream.fileno()

    def __exit__(self, *_: object) -> None:
        if self.stream is not None:
            # Closing (not LOCK_UN) lets an inherited POSIX descriptor retain ownership
            # until an orphan worker exits after its coordinator crashes.
            self.stream.close()
            self.stream = None
