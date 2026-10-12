"""A single live status line ("judging 120/347") for long pipeline steps."""

from __future__ import annotations

import sys
import threading
from typing import Optional, TextIO


class Progress:
    """Rewrites one line in place on a terminal. Does nothing when the stream isn't a terminal
    (redirected to a file, CI, tests), so logs don't fill up with partial counts."""

    def __init__(self, stream: Optional[TextIO] = None, enabled: Optional[bool] = None):
        self.stream = stream or sys.stderr
        self.enabled = self.stream.isatty() if enabled is None else enabled
        self._shown = False
        self._lock = threading.Lock()

    def update(self, label: str, done: int, total: int) -> None:
        if not self.enabled:
            return
        with self._lock:
            self.stream.write(f"\r\033[K{label} {done}/{total}")
            self.stream.flush()
            self._shown = True

    def clear(self) -> None:
        """Erase the line, so normal output starts on a clean line."""
        if not self._shown:
            return
        with self._lock:
            self.stream.write("\r\033[K")
            self.stream.flush()
            self._shown = False
