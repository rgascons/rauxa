"""The provider-neutral boundary between CLI adapters and the dispatcher."""

from __future__ import annotations

from dataclasses import dataclass
from threading import Event
from typing import Any, Callable, Protocol

from .config import BackendConfig


@dataclass
class AttemptResult:
    answer: dict[str, Any] | None = None
    category: str = ""
    message: str = ""


class CliRunner(Protocol):
    """One runner per dispatch invocation; run() must support concurrent calls.

    Expose the executable at construction and its version after preflight. Preflight
    must not spend model calls. Return validated task JSON or a failed AttemptResult;
    raise BackendUnavailable for setup failures and InterruptedError on cancellation.
    Adapters run fresh, tools-disabled sessions and report owned worker PIDs through
    started. They own isolation, timeouts, and child cleanup, retaining the borrowed
    run-lock descriptor in workers until those workers stop.
    """

    executable: str
    version: str

    def preflight(self) -> None: ...

    def run(
        self, request: dict[str, Any], backend: BackendConfig, cancel: Event,
        started: Callable[[int], None] = ...,
    ) -> AttemptResult: ...
