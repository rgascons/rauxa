"""A single coordinator for file-backed pipeline calls and bounded Copilot workers."""

from __future__ import annotations

import json
import math
import signal
import sys
import threading
import time
from collections import Counter, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import asdict
from typing import Any, Callable

from .config import CopilotSettings, RunConfig
from .controller import Pipeline, PipelineError, RunResult
from .copilot import AttemptResult, CopilotProcess
from .llm import BackendUnavailable, validate_agent_answer, validate_request
from .progress import Progress
from .storage import RunLock, append_record, read_records, write_json


class Dispatcher:
    def __init__(
        self, config: RunConfig, settings: CopilotSettings, *, resume: bool = False,
        log: Callable[[str], None] | None = None,
    ):
        self.config, self.settings, self.resume = config, settings, resume
        self.log = log or (lambda text: print(text, file=sys.stderr, flush=True))
        self.directory = config.run_dir() / "dispatcher"
        self.exchange = config.run_dir() / "agent_exchange"
        self.cancel = threading.Event()
        self.progress = Progress()
        self._journal_lock = threading.Lock()
        self.attempts: Counter[str] = Counter()
        self.last_errors: dict[str, dict[str, Any]] = {}
        self.completed: set[str] = set()
        self.failed: set[str] = set()
        self.used = 0
        self.version = ""
        self.executable = ""

    def _event(self, event: str, **fields: Any) -> None:
        with self._journal_lock:
            append_record(self.directory / "events.jsonl", {"event": event, "time": time.time(), **fields})

    def _recover(self) -> None:
        path = self.directory / "events.jsonl"
        for record in read_records(path):
            if record["event"] == "reserved":
                self.used += 1
                self.attempts[record["id"]] += 1
            elif record["event"] == "result" and record["category"]:
                self.last_errors[record["id"]] = record
        self.completed = {p.stem for p in (self.exchange / "responses").glob("*.json")}
        self.failed = {p.stem for p in (self.exchange / "failures").glob("*.json")}

    def record_disabled(self) -> None:
        """Called with the manual pipeline's run lock held; no CLI preflight is needed."""
        self.directory.mkdir(exist_ok=True)
        self._recover()
        self._event("settings", settings=asdict(self.settings))
        self._snapshot("disabled")
        self.log(f"Copilot settings: {json.dumps(asdict(self.settings), ensure_ascii=False)}")

    def _snapshot(self, status: str, pending: int = 0, active: int = 0) -> dict[str, Any]:
        summary = {
            "status": status, "settings": asdict(self.settings), "executable": self.executable,
            "cli_version": self.version, "calls_used": self.used, "completed": len(self.completed),
            "failed": len(self.failed), "pending": pending, "active": active,
            "retries": sum(max(0, n - 1) for n in self.attempts.values()),
            "execution_log": "dispatcher/events.jsonl",
        }
        write_json(self.directory / "status.json", summary)
        return summary

    def _validate(self) -> None:
        if not self.settings.enabled or self.settings.max_calls is None:
            raise ValueError("dispatcher needs enabled Copilot settings and an explicit call cap")
        names = {s.backend for s in self.config.sources if s.type == "llm"}
        if self.config.judge.enabled:
            names.add(self.config.judge.backend)
        agents = [self.config.backends[name] for name in names if self.config.backends[name].type == "agent"]
        if not agents:
            raise ValueError("Copilot dispatch needs at least one referenced type='agent' backend")
        for source in self.config.sources:
            if source.type != "llm" or self.config.backends[source.backend].type != "agent":
                continue
            for key, minimum in (("count", 0), ("batch_size", 1), ("max_fix_attempts", 0)):
                value = getattr(source, key)
                if type(value) is not int or value < minimum:
                    raise ValueError(f"agent source {key} must be an integer >= {minimum}")
        for backend in agents:
            if not isinstance(backend.model, str) or not backend.model.strip():
                raise ValueError("automated agent backends need explicit model names")
            if type(backend.max_retries) is not int or backend.max_retries < 1:
                raise ValueError("agent max_retries must be a positive total attempt count")
            if (
                isinstance(backend.timeout, bool) or not isinstance(backend.timeout, (int, float))
                or not math.isfinite(backend.timeout) or backend.timeout <= 0
            ):
                raise ValueError("agent timeout must be a positive finite number")

    def _failure(self, request: dict[str, Any]) -> None:
        rid = request["id"]
        error = self.last_errors.get(rid, {
            "category": "interrupted", "message": "last attempt completion is unknown; attempt budget exhausted",
        })
        write_json(self.exchange / "failures" / f"{rid}.json", {
            "id": rid, "category": error["category"], "message": error["message"], "attempts": self.attempts[rid],
        })
        self.failed.add(rid)
        self.log(f"Copilot {request['kind']} {rid}: failed after {self.attempts[rid]} attempts: {error['category']}")

    def _attempt(self, runner: CopilotProcess, request: dict[str, Any], attempt: int) -> AttemptResult:
        if attempt > 1 and self.cancel.wait(min(2 ** min(attempt - 2, 4), 10)):
            raise InterruptedError("Copilot dispatch interrupted")
        return runner.run(
            request, self.config.backends[request["backend"]], self.cancel,
            lambda pid: self._event("started", id=request["id"], attempt=attempt, pid=pid),
        )

    def _publish(self, request: dict[str, Any], result: AttemptResult) -> None:
        rid = request["id"]
        if result.answer is not None:
            response = self.exchange / "responses" / f"{rid}.json"
            if not response.exists():
                write_json(response, result.answer)
            self.completed.add(rid)
        else:
            self.last_errors[rid] = {"category": result.category, "message": result.message}
        self._event("result", id=rid, attempt=self.attempts[rid], category=result.category, message=result.message)

    def _answer_pending(self, ids: list[str], runner: CopilotProcess) -> bool:
        queue = deque()
        for rid in ids:
            # Validate the identifier before constructing a path from it.
            if len(rid) != 32 or any(c not in "0123456789abcdef" for c in rid):
                raise ValueError("legacy or invalid request identity; start a fresh run for automatic dispatch")
            request = json.loads((self.exchange / "requests" / f"{rid}.json").read_text(encoding="utf-8"))
            validate_request(request, rid)
            backend = self.config.backends.get(request["backend"])
            if backend is None or backend.type != "agent" or backend.model != request["model"]:
                raise ValueError(f"request {rid} does not match the configured agent backend")
            response = self.exchange / "responses" / f"{rid}.json"
            if response.exists():
                validate_agent_answer(json.loads(response.read_text(encoding="utf-8")), request["kind"])
                self.completed.add(rid)
            elif rid in self.failed:
                continue
            elif self.attempts[rid] >= backend.max_retries:
                self._failure(request)
            else:
                queue.append(request)
        self.log(f"Copilot: {len(queue)} pending requests; {self.used}/{self.settings.max_calls} calls reserved")
        active = {}
        probed = set()
        last_snapshot = 0.0
        with ThreadPoolExecutor(max_workers=self.settings.workers) as pool:
            try:
                while queue or active:
                    if self.cancel.is_set():
                        raise InterruptedError("Copilot dispatch interrupted")
                    busy_backends = {request["backend"] for request in active.values()}
                    for _ in range(len(queue)):
                        if len(active) >= self.settings.workers or self.used >= self.settings.max_calls:
                            break
                        request = queue.popleft()
                        name, rid = request["backend"], request["id"]
                        # Exercise one real request per backend before fanning out.
                        if name not in probed and name in busy_backends:
                            queue.append(request)
                            continue
                        attempt = self.attempts[rid] + 1
                        self._event("reserved", id=rid, attempt=attempt, backend=name, model=request["model"], kind=request["kind"])
                        self.attempts[rid] = attempt
                        self.used += 1
                        active[pool.submit(self._attempt, runner, request, attempt)] = request
                        busy_backends.add(name)
                    if not active:
                        break
                    done, _ = wait(active, timeout=0.1, return_when=FIRST_COMPLETED)
                    for future in done:
                        request = active.pop(future)
                        rid = request["id"]
                        try:
                            result = future.result()
                        except BackendUnavailable as e:
                            self._event("setup_error", id=rid, attempt=self.attempts[rid], message=str(e))
                            raise
                        probed.add(request["backend"])
                        self._publish(request, result)
                        if result.answer is None:
                            if self.attempts[rid] >= self.config.backends[request["backend"]].max_retries:
                                self._failure(request)
                            else:
                                self.log(f"Copilot {request['kind']} {rid}: {result.category}; retry pending")
                                queue.append(request)
                    self.progress.update(
                        f"Copilot calls ({len(active)} active, {len(queue)} pending)", self.used, self.settings.max_calls,
                    )
                    if time.monotonic() - last_snapshot >= 1:
                        self._snapshot("running", len(queue), len(active))
                        last_snapshot = time.monotonic()
            except BaseException:
                self.cancel.set()
                for future in active:
                    future.cancel()
                # Workers that completed before cancellation still own useful results.
                for future, request in active.items():
                    if future.cancelled():
                        continue
                    try:
                        result = future.result()
                    except (InterruptedError, BackendUnavailable) as e:
                        self._event("aborted", id=request["id"], message=str(e))
                    else:
                        self._publish(request, result)
                raise
            finally:
                self.progress.clear()
        return not queue

    def run(self) -> RunResult:
        self._validate()
        with RunLock(self.config.run_dir()) as lock:
            pipeline = Pipeline(self.config, resume=self.resume, run_lock=lock, log=self.log)
            if pipeline.state.get("agent_protocol") != 2:
                raise PipelineError("legacy agent run cannot be safely automated; resume manually or use a fresh run name")
            self.directory.mkdir(exist_ok=True)
            self._recover()
            self._event("settings", settings=asdict(self.settings))
            self.log(f"Copilot settings: {json.dumps(asdict(self.settings), ensure_ascii=False)}")
            old_handler = None
            if threading.current_thread() is threading.main_thread():
                def interrupt(*_: object) -> None:
                    self.cancel.set()
                    raise InterruptedError("Copilot dispatch interrupted")
                old_handler = signal.signal(signal.SIGTERM, interrupt)
            try:
                runner = CopilotProcess(self.settings, lock.fileno())
                self.executable = runner.executable
                runner.preflight()
                self.version = runner.version
                self._event("preflight", executable=self.executable, version=self.version)
                while True:
                    if self.cancel.is_set():
                        raise InterruptedError("Copilot dispatch interrupted")
                    result = pipeline.run()
                    if result.status != "waiting":
                        summary = self._snapshot(result.status)
                        result.manifest["dispatcher"] = summary
                        write_json(self.config.run_dir() / "manifest.json", result.manifest)
                        self.log(f"Copilot: {self.used} calls reserved, {len(self.failed)} failed requests")
                        return result
                    ids = result.manifest["pending_requests"]
                    if not self._answer_pending(ids, runner):
                        remaining = set(ids) - self.completed - self.failed
                        result.manifest["pending_requests"] = sorted(remaining)
                        result.manifest["dispatcher"] = self._snapshot("budget_paused", len(remaining))
                        self.log(f"Copilot call cap reached ({self.used}/{self.settings.max_calls}); raise it and use --resume")
                        return result
                    pipeline = Pipeline(self.config, resume=True, run_lock=lock, log=self.log)
            except (KeyboardInterrupt, InterruptedError):
                self.cancel.set()
                self._snapshot("interrupted")
                self._event("interrupted")
                raise
            except (OSError, ValueError, RuntimeError) as e:
                self.cancel.set()
                self._snapshot("error")
                self._event("error", message=str(e))
                raise
            finally:
                if old_handler is not None:
                    signal.signal(signal.SIGTERM, old_handler)
