import os
import tempfile
import unittest
from threading import Event
from typing import Any, Callable
from unittest.mock import patch

from rauxa.pipeline.cli_runner import AttemptResult, CliRunner
from rauxa.pipeline.config import BackendConfig, CopilotConfig, resolve_copilot
from rauxa.pipeline.copilot import AttemptResult as CopilotAttemptResult
from rauxa.pipeline.dispatch import Dispatcher
from rauxa.pipeline.llm import BackendUnavailable
from rauxa.pipeline.storage import RunLock, RunLocked, read_records
from test_agent_exchange import answer, run_config


class MemoryRunner:
    executable = "memory-cli"

    def __init__(self, lock_fd: int):
        self.lock_fd = lock_fd
        self.version = ""
        self.preflights = 0
        self.calls = []

    def preflight(self) -> None:
        os.fstat(self.lock_fd)
        self.preflights += 1
        self.version = "memory-cli 1"

    def run(
        self, request: dict[str, Any], backend: BackendConfig, cancel: Event,
        started: Callable[[int], None] = lambda _: None,
    ) -> AttemptResult:
        assert self.preflights == 1
        if cancel.is_set():
            raise InterruptedError("cancelled")
        self.calls.append((request["id"], backend.model))
        started(os.getpid())
        return AttemptResult(answer=answer(int(request["key"].split(":")[-1])))


class TestCliRunner(unittest.TestCase):
    def settings(self, cap):
        return resolve_copilot(
            CopilotConfig(enabled=True, max_calls=cap),
            environ={"RAUXA_COPILOT_BIN": "unused-copilot-executable"},
        )

    def test_copilot_reexports_shared_result(self):
        self.assertIs(CopilotAttemptResult, AttemptResult)

    def test_injected_runner_preflight_budget_resume_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = run_config(tmp, count=2)
            runners = []

            def factory(lock_fd: int) -> CliRunner:
                os.fstat(lock_fd)
                with self.assertRaises(RunLocked):
                    with RunLock(cfg.run_dir()):
                        pass
                runner = MemoryRunner(lock_fd)
                runners.append(runner)
                return runner

            with patch("rauxa.pipeline.dispatch.CopilotProcess", side_effect=AssertionError("Copilot must not be constructed")):
                first = Dispatcher(cfg, self.settings(1), runner_factory=factory, log=lambda _: None).run()
                self.assertEqual(first.status, "waiting")
                final = Dispatcher(cfg, self.settings(2), runner_factory=factory, resume=True, log=lambda _: None).run()

            self.assertEqual(final.status, "complete")
            self.assertEqual(final.selected, 2)
            self.assertEqual([runner.preflights for runner in runners], [1, 1])
            self.assertEqual([len(runner.calls) for runner in runners], [1, 1])
            self.assertEqual(len({runner.calls[0][0] for runner in runners}), 2)
            self.assertTrue(all(runner.calls[0][1] == "test-model" for runner in runners))
            summary = final.manifest["dispatcher"]
            self.assertEqual(summary["calls_used"], 2)
            self.assertEqual(summary["executable"], "memory-cli")
            self.assertEqual(summary["cli_version"], "memory-cli 1")
            events = list(read_records(cfg.run_dir() / "dispatcher" / "events.jsonl"))
            self.assertEqual(len([event for event in events if event["event"] == "started"]), 2)
            with RunLock(cfg.run_dir()):
                pass

    def test_injected_preflight_failure_spends_no_calls(self):
        class UnavailableRunner(MemoryRunner):
            def preflight(self) -> None:
                raise BackendUnavailable("adapter unavailable")

        with tempfile.TemporaryDirectory() as tmp:
            cfg = run_config(tmp, count=1)
            dispatcher = Dispatcher(cfg, self.settings(1), runner_factory=UnavailableRunner, log=lambda _: None)
            with self.assertRaisesRegex(BackendUnavailable, "adapter unavailable"):
                dispatcher.run()
            events = list(read_records(cfg.run_dir() / "dispatcher" / "events.jsonl"))
            self.assertFalse(any(event["event"] == "reserved" for event in events))
            with RunLock(cfg.run_dir()):
                pass


if __name__ == "__main__":
    unittest.main()
