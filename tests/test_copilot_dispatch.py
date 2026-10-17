import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from rauxa.cli import main
from rauxa.pipeline.config import BackendConfig, CopilotConfig, JudgeConfig, resolve_copilot
from rauxa.pipeline.copilot import AttemptResult, CopilotProcess
from rauxa.pipeline.dispatch import Dispatcher
from rauxa.pipeline.llm import BackendUnavailable, managed_request_id
from rauxa.pipeline.storage import RunLock, RunLocked, append_record, read_records
from test_agent_exchange import run_config


@unittest.skipUnless(os.name == "posix", "Copilot dispatch runs on WSL/Linux")
class TestCopilotDispatch(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.binary = self.root / "fake copilot"
        self.binary.write_text(
            (Path(__file__).parent / "fixtures" / "fake_copilot.py").read_text(encoding="utf-8"),
            encoding="utf-8", newline="\n",
        )
        self.binary.chmod(0o700)
        self.events = self.root / "cli.jsonl"
        env = patch.dict(os.environ, {"COPILOT_GITHUB_TOKEN": "fake-test-token",
                                     "RAUXA_FAKE_LOG": str(self.events), "RAUXA_FAKE_MODE": "normal"})
        env.start()
        self.addCleanup(env.stop)

    def settings(self, cap, workers=4):
        return resolve_copilot(
            CopilotConfig(enabled=True, workers=workers, max_calls=cap,
                          specified=frozenset(("enabled", "workers", "max_calls"))),
            command=str(self.binary),
        )

    def logs(self):
        return list(read_records(self.events))

    def test_budget_pause_resume_and_model_routing(self):
        cfg = run_config(self.root)
        cfg.judge = JudgeConfig(enabled=True, backend="judge", workers=2)
        cfg.backends["judge"] = BackendConfig(type="agent", model="judge-model")
        initial = Dispatcher(cfg, self.settings(2), log=lambda _: None).run()
        self.assertEqual(initial.status, "waiting")
        self.assertEqual(initial.manifest["dispatcher"]["calls_used"], 2)
        resumed = Dispatcher(cfg, self.settings(8, workers=2), resume=True, log=lambda _: None).run()
        self.assertEqual(resumed.status, "complete")
        self.assertEqual(resumed.selected, 4)
        self.assertEqual(resumed.manifest["dispatcher"]["calls_used"], 8)
        self.assertEqual(len(self.logs()), 8)
        for record in self.logs():
            self.assertIn("--available-tools=", record["args"])
            self.assertIn("--no-custom-instructions", record["args"])
            self.assertNotIn("--allow-all", record["args"])
            self.assertEqual(record["model"], "judge-model" if record["role"] == "judge" else "test-model")
            self.assertFalse(Path(record["cwd"]).exists())
        history = list(read_records(cfg.run_dir() / "dispatcher" / "events.jsonl"))
        settings = [r["settings"] for r in history if r["event"] == "settings"]
        self.assertEqual([r["max_calls"] for r in settings], [2, 8])
        self.assertEqual(settings[-1]["origins"]["workers"], "toml:copilot.workers")
        again = Dispatcher(cfg, self.settings(8), resume=True, log=lambda _: None).run()
        self.assertEqual(again.status, "complete")
        self.assertEqual(len(self.logs()), 8)

    def test_retry_budget_survives_resume_and_failures_continue(self):
        cfg = run_config(self.root, count=2)
        cfg.backends["writer"].max_retries = 2
        with patch.dict(os.environ, {"RAUXA_FAKE_MODE": "invalid"}):
            first = Dispatcher(cfg, self.settings(1), log=lambda _: None).run()
            self.assertEqual(first.status, "waiting")
            final = Dispatcher(cfg, self.settings(4), resume=True, log=lambda _: None).run()
        self.assertEqual(final.status, "incomplete")
        self.assertEqual(final.manifest["writer_errors"], 2)
        self.assertEqual(final.manifest["dispatcher"]["failed"], 2)
        self.assertEqual(len(self.logs()), 4)
        self.assertEqual(final.manifest["dispatcher"]["retries"], 2)

    def test_repairs_and_negative_judgements(self):
        cfg = run_config(self.root, count=2)
        cfg.sources[0].max_fix_attempts = 1
        cfg.judge = JudgeConfig(enabled=True, backend="judge")
        cfg.backends["judge"] = BackendConfig(type="agent", model="judge-model")
        with patch.dict(os.environ, {"RAUXA_FAKE_MODE": "repair"}):
            result = Dispatcher(cfg, self.settings(6), log=lambda _: None).run()
        self.assertEqual(result.status, "complete")
        self.assertEqual(result.manifest["counts"]["candidates"], 4)
        self.assertEqual([r["role"] for r in self.logs()].count("fix"), 2)
        cfg.name = "negative"
        with patch.dict(os.environ, {"RAUXA_FAKE_MODE": "negative"}):
            result = Dispatcher(cfg, self.settings(4), log=lambda _: None).run()
        self.assertEqual(result.status, "incomplete")
        self.assertEqual(result.manifest["dispatcher"]["failed"], 0)

    def test_setup_errors_stop_without_failing_all_requests(self):
        for mode in ("old_cli", "auth"):
            cfg = run_config(self.root)
            cfg.name = mode
            with patch.dict(os.environ, {"RAUXA_FAKE_MODE": mode}), self.assertRaises(BackendUnavailable):
                Dispatcher(cfg, self.settings(100), log=lambda _: None).run()
            self.assertFalse(list((cfg.run_dir() / "agent_exchange" / "failures").glob("*.json")))
        self.assertEqual(len(self.logs()), 1)  # help/version do not consume calls

    def test_timeout_and_owned_children(self):
        request = {"model": "m", "kind": "write", "messages": [{"role": "user", "content": "Write 1 new pairs."}]}
        with RunLock(self.root / "locked") as lock:
            runner = CopilotProcess(self.settings(1), lock.fileno())
            with patch.dict(os.environ, {"RAUXA_FAKE_MODE": "child"}):
                result = runner.run(request, BackendConfig(timeout=0.4), threading.Event())
        self.assertEqual(result.category, "timeout")
        record = self.logs()[0]
        for pid in (record["pid"], record["child"]):
            stat = Path(f"/proc/{pid}/stat")
            if stat.exists():
                self.assertEqual(stat.read_text().split()[2], "Z", f"process {pid} still running")

    def cli_config(self):
        config = self.root / "run.toml"
        config.write_text(f'''
[run]
name = "cli"
output_dir = {json.dumps(str(self.root))}
target_size = 2
pool_multiplier = 1
[[sources]]
type = "llm"
task = "castellanismes_verbals"
backend = "writer"
count = 2
batch_size = 1
phenomena = ["tenir_que"]
max_fix_attempts = 0
[checks.languagetool]
enabled = false
[backends.writer]
type = "agent"
model = "test-model"
[copilot]
enabled = true
workers = 2
max_calls = 1
''', encoding="utf-8")
        return config

    def test_cli_toml_only_and_overrides(self):
        config = self.cli_config()
        output = StringIO()
        with patch.dict(os.environ, {"RAUXA_COPILOT_BIN": str(self.binary)}), redirect_stdout(output), redirect_stderr(output):
            self.assertEqual(main(["run", str(config)]), 3)
            self.assertEqual(main(["summary", str(self.root / "cli")]), 1)
            self.assertEqual(main(["run", str(config), "--resume", "--copilot-max-calls", "2"]), 0)
            self.assertEqual(main(["summary", str(self.root / "cli"), "--samples", "0"]), 0)
        self.assertIn("env:RAUXA_COPILOT_BIN", output.getvalue())
        self.assertIn("cli:--copilot-max-calls", output.getvalue())

    def test_no_copilot_and_dataset_fingerprint(self):
        config = self.cli_config()
        output = StringIO()
        with redirect_stdout(output), redirect_stderr(output):
            self.assertEqual(main(["run", str(config), "--no-copilot"]), 3)
            self.assertFalse(self.events.exists())
            disabled = json.loads((self.root / "cli" / "dispatcher" / "status.json").read_text())
            self.assertEqual(disabled["status"], "disabled")
            self.assertEqual(disabled["settings"]["origins"]["enabled"], "cli:--no-copilot")
            before = json.loads((self.root / "cli" / "state.json").read_text())["fingerprint"]
            config.write_text(config.read_text().replace("workers = 2", "workers = 3").replace("max_calls = 1", "max_calls = 2"))
            with patch.dict(os.environ, {"RAUXA_COPILOT_BIN": str(self.binary)}):
                self.assertEqual(main(["run", str(config), "--resume"]), 0)
            after = json.loads((self.root / "cli" / "state.json").read_text())["fingerprint"]
            self.assertEqual(before, after)
            config.write_text(config.read_text().replace('model = "test-model"', 'model = "different-model"'))
            self.assertEqual(main(["run", str(config), "--resume"]), 1)
        self.assertIn("config changed", output.getvalue())

    def test_nonzero_errors_and_judge_error_policy(self):
        cfg = run_config(self.root, count=1)
        cfg.backends["writer"].max_retries = 1
        with patch.dict(os.environ, {"RAUXA_FAKE_MODE": "exit"}):
            result = Dispatcher(cfg, self.settings(1), log=lambda _: None).run()
        self.assertEqual(result.status, "incomplete")
        self.assertEqual(result.manifest["dispatcher"]["failed"], 1)
        for policy, selected in (("reject", 0), ("keep", 1)):
            cfg.name = policy
            cfg.judge = JudgeConfig(enabled=True, backend="judge", on_error=policy)
            cfg.backends["judge"] = BackendConfig(type="agent", model="judge-model", max_retries=1)
            Dispatcher(cfg, self.settings(1), log=lambda _: None).run()
            with patch.dict(os.environ, {"RAUXA_FAKE_MODE": "exit"}):
                result = Dispatcher(cfg, self.settings(2), resume=True, log=lambda _: None).run()
            self.assertEqual(result.selected, selected)
            self.assertEqual(result.manifest["judge"]["errors"], 1)

    def test_login_profile_isolation_and_redaction(self):
        home = self.root / "login"
        home.mkdir()
        with patch.dict(os.environ, {"COPILOT_HOME": str(home)}, clear=True):
            env = CopilotProcess._environment(self.root)
            self.assertEqual(env["COPILOT_HOME"], str(home))
            state = home / "config.json"
            state.write_text('// This file is managed automatically.\n{"loggedInUsers": []}\n')
            self.assertEqual(CopilotProcess._environment(self.root)["COPILOT_HOME"], str(home))
            state.write_text('// managed\n{"installedPlugins": ["x"]}\n')
            with self.assertRaisesRegex(BackendUnavailable, "plugin state"):
                CopilotProcess._environment(self.root)
            state.write_text("not json")
            with self.assertRaisesRegex(BackendUnavailable, "cannot verify"):
                CopilotProcess._environment(self.root)
            state.unlink()
            settings = home / "settings.json"
            settings.write_text('{\n  "bashEnv": false\n}\n')
            self.assertEqual(CopilotProcess._environment(self.root)["COPILOT_HOME"], str(home))
            for content in ('{"bashEnv": true}', '{"bashEnv": false, "hooks": {}}', "not json"):
                settings.write_text(content)
                with self.assertRaisesRegex(BackendUnavailable, "customizations found: settings.json"):
                    CopilotProcess._environment(self.root)
            settings.unlink()
            (home / "hooks").mkdir()
            self.assertEqual(CopilotProcess._environment(self.root)["COPILOT_HOME"], str(home))
            (home / "hooks" / "custom.json").write_text("{}")
            with self.assertRaisesRegex(BackendUnavailable, "customizations"):
                CopilotProcess._environment(self.root)
            with patch.dict(os.environ, {"COPILOT_GITHUB_TOKEN": "test-secret", "COPILOT_ALLOW_ALL": "true"}):
                env = CopilotProcess._environment(self.root)
                self.assertNotEqual(env["COPILOT_HOME"], str(home))
                self.assertNotIn("COPILOT_ALLOW_ALL", env)
                self.assertNotIn("test-secret", CopilotProcess._redact("error test-secret"))

    def test_legacy_runs_require_explicit_manual_resume(self):
        from rauxa.pipeline.controller import Pipeline, PipelineError
        cfg = run_config(self.root)
        Pipeline(cfg, log=lambda _: None).run()
        state_path = cfg.run_dir() / "state.json"
        state = json.loads(state_path.read_text())
        del state["agent_protocol"]
        state_path.write_text(json.dumps(state))
        with self.assertRaisesRegex(PipelineError, "legacy"):
            Dispatcher(cfg, self.settings(10), resume=True, log=lambda _: None).run()
        self.assertFalse(self.events.exists())

    def test_invalid_agent_settings_fail_before_calls(self):
        for field, value in (("model", 42), ("timeout", float("nan")), ("max_retries", 0)):
            cfg = run_config(self.root)
            setattr(cfg.backends["writer"], field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                Dispatcher(cfg, self.settings(10), log=lambda _: None).run()
        for field, value in (("batch_size", 0), ("count", True), ("max_fix_attempts", -1)):
            cfg = run_config(self.root)
            setattr(cfg.sources[0], field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                Dispatcher(cfg, self.settings(10), log=lambda _: None).run()
        self.assertFalse(self.events.exists())

    def test_coordinator_sigterm_stops_workers(self):
        config = self.cli_config()
        env = dict(os.environ, RAUXA_COPILOT_BIN=str(self.binary), RAUXA_FAKE_MODE="child",
                   PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
        with (self.root / "coordinator.log").open("w") as log:
            process = subprocess.Popen([sys.executable, "-m", "rauxa", "run", str(config)],
                                       env=env, stdout=log, stderr=log)
            try:
                self.wait_until(lambda: self.events.exists() and self.events.stat().st_size > 0)
                with self.assertRaises(RunLocked):
                    with RunLock(self.root / "cli"):
                        pass
                process.send_signal(signal.SIGTERM)
                self.assertEqual(process.wait(timeout=8), 130)
                with RunLock(self.root / "cli"):
                    pass
                status = json.loads((self.root / "cli" / "dispatcher" / "status.json").read_text())
                self.assertEqual(status["status"], "interrupted")
                self.assertEqual(status["calls_used"], 1)
                self.assertEqual(status["failed"], 0)
            finally:
                if process.poll() is None:
                    process.send_signal(signal.SIGTERM)
                    process.wait(timeout=8)
        for event in self.logs():
            for pid in (event["pid"], event["child"]):
                stat = Path(f"/proc/{pid}/stat")
                if stat.exists():
                    self.assertEqual(stat.read_text().split()[2], "Z")

    def wait_until(self, predicate, timeout=8):
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() >= deadline:
                self.fail("timed out waiting for subprocess condition")
            time.sleep(0.02)

    def test_guard_retains_lock_after_coordinator_crash(self):
        directory = self.root / "orphan"
        worker = Path(__file__).resolve().parents[1] / "src" / "rauxa" / "pipeline" / "copilot_worker.py"
        code = '''
import os, subprocess, sys, time
from pathlib import Path
from rauxa.pipeline.storage import RunLock
with RunLock(Path(sys.argv[1])) as lock:
    process = subprocess.Popen(
        [sys.executable, sys.argv[2], "2", str(lock.fileno()), sys.argv[3], "--model", "m"],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True, pass_fds=(lock.fileno(),))
    process.stdin.write(b"Write 1 new pairs.")
    process.stdin.close()
    deadline = time.monotonic() + 5
    while not Path(os.environ["RAUXA_FAKE_LOG"]).exists():
        if time.monotonic() >= deadline:
            process.terminate()
            process.wait()
            raise RuntimeError("CLI did not start")
        time.sleep(0.01)
    os._exit(0)
'''
        env = dict(os.environ, RAUXA_FAKE_MODE="child", COPILOT_HOME=str(self.root / "profile"),
                   PYTHONPATH=str(Path(__file__).resolve().parents[1] / "src"))
        subprocess.run([sys.executable, "-c", code, str(directory), str(worker), str(self.binary)],
                       env=env, check=True, timeout=8)
        with self.assertRaises(RunLocked):
            with RunLock(directory):
                pass
        def released():
            try:
                with RunLock(directory):
                    return True
            except RunLocked:
                return False
        self.wait_until(released)
        for event in self.logs():
            for pid in (event["pid"], event["child"]):
                stat = Path(f"/proc/{pid}/stat")
                if stat.exists():
                    self.assertEqual(stat.read_text().split()[2], "Z")

    def test_ten_thousand_candidates_bounded_scheduling(self):
        cfg = run_config(self.root, count=10000)
        cfg.backends["judge"] = BackendConfig(type="agent", model="judge-model")
        dispatcher = Dispatcher(cfg, self.settings(10500), log=lambda _: None)
        requests = {}
        responses = {}
        events = []
        concurrency_lock = threading.Lock()
        active = peak = 0

        class FastRunner:
            def run(self, request, backend, cancel, started):
                nonlocal active, peak
                with concurrency_lock:
                    active += 1
                    peak = max(peak, active)
                try:
                    time.sleep(0.0001)
                    if request["kind"] == "write":
                        return AttemptResult(answer={"items": [{"good": "Cal treballar.", "bad": "Hi ha que treballar.",
                                                               "alternatives": []}] * request["count"]})
                    return AttemptResult(answer={"good_is_correct": True, "variants_are_correct": True,
                                                 "bad_has_stated_error": True, "bad_has_other_errors": False,
                                                 "natural": True, "same_meaning": True, "comment": ""})
                finally:
                    with concurrency_lock:
                        active -= 1

        for kind, count, backend in (("write", 500, "writer"), ("judge", 10000, "judge")):
            for index in range(count):
                request = {"version": 2, "kind": kind, "key": f"{kind}:{index}", "count": 20 if kind == "write" else 1,
                           "backend": backend, "model": cfg.backends[backend].model,
                           "messages": [{"role": "user", "content": f"Request {kind} {index}."}]}
                request["id"] = managed_request_id(request)
                requests[request["id"]] = request
        def publish(path, value):
            self.assertNotIn(path.stem, responses)
            responses[path.stem] = value
        with patch.object(Path, "read_text", lambda path, **_: json.dumps(requests[path.stem])), \
             patch.object(Path, "exists", return_value=False), \
             patch("rauxa.pipeline.dispatch.write_json", side_effect=publish), \
             patch.object(dispatcher, "_event", side_effect=lambda event, **fields: events.append((event, fields))), \
             patch.object(dispatcher, "_snapshot"):
            for kind in ("write", "judge"):
                ids = [rid for rid, r in requests.items() if r["kind"] == kind]
                self.assertTrue(dispatcher._answer_pending(ids, FastRunner()))
        self.assertLessEqual(peak, 4)
        self.assertGreater(peak, 1)
        self.assertEqual(len(responses), 10500)
        self.assertEqual(sum(len(answer.get("items", [])) for answer in responses.values()), 10000)
        self.assertEqual(dispatcher.used, 10500)
        self.assertEqual(len({e[1]["id"] for e in events if e[0] == "reserved"}), 10500)

    def test_torn_journal_recovers_explicitly(self):
        path = self.root / "torn.jsonl"
        append_record(path, {"event": "reserved"})
        with path.open("ab") as stream:
            stream.write(b'{"event":')
        warning = StringIO()
        with redirect_stderr(warning):
            self.assertEqual(list(read_records(path)), [{"event": "reserved"}])
        self.assertIn("recovering interrupted append", warning.getvalue())
        self.assertTrue(path.read_bytes().endswith(b"\n"))


if __name__ == "__main__":
    unittest.main()
