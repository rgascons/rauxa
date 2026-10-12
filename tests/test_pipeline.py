import json
import re
import tempfile
import unittest
from pathlib import Path

from rauxa.pipeline import Pipeline, PipelineError
from rauxa.pipeline.calibration import calibrate, cohen_kappa
from rauxa.pipeline.config import (
    BackendConfig, ChecksConfig, JudgeConfig, LanguageToolCheckConfig, RunConfig, SourceConfig,
)
from rauxa.tasks.castellanismes_verbals import PHENOMENON_INFO, TENSES

TASK = "castellanismes_verbals"


class FakeLanguageTool:
    """Flags each calque with its expected rule, except compound tenses of "hi ha que" (a real gap)."""

    def __init__(self):
        self.markers = {p: re.compile(info["marker"], re.IGNORECASE) for p, info in PHENOMENON_INFO.items()}

    def check(self, text):
        if re.search(r"\bhi (va haver|ha hagut) que\b", text, re.IGNORECASE):
            return []
        for phenomenon, marker in self.markers.items():
            if marker.search(text):
                rule = PHENOMENON_INFO[phenomenon]["languagetool_rules"][0]
                return [{"rule": {"id": rule, "category": {"id": "CASTELLANISMES"}}, "message": "calc", "offset": 0, "length": 1}]
        return []


class FakeBackend:
    def __init__(self, model, pairs=None, verdict=None, fail=False):
        self.model = model
        self.pairs = pairs or []
        self.verdict = verdict or {}
        self.fail = fail
        self.calls = []

    def chat_json(self, messages):
        from rauxa.pipeline.llm import LLMError

        self.calls.append(messages)
        if self.fail:
            raise LLMError("backend down")
        user = messages[-1]["content"]
        if "did not pass our checks" in user:
            return {"good": "Dissabte hem d'acabar la maqueta.", "bad": "Dissabte tenim que acabar la maqueta.", "alternatives": []}
        if "Write" in user and "new pairs" in user:
            return {"items": self.pairs}
        base = {"good_is_correct": True, "variants_are_correct": True, "bad_has_stated_error": True,
                "bad_has_other_errors": False, "natural": True, "same_meaning": True, "comment": ""}
        for needle, override in self.verdict.items():
            if needle in user:
                base.update(override)
        return base


def procedural_source(**params):
    return SourceConfig(type="procedural", task=TASK, params={"tenses": list(TENSES), "formats": ["mcq", "judgement", "gec"], **params})


def config(tmp, name="run", **overrides):
    values = dict(
        name=name, target_size=300, seed=3, output_dir=tmp, sources=[procedural_source()],
        checks=ChecksConfig(languagetool=LanguageToolCheckConfig(workers=2)),
        min_share={"phenomenon": {"tenir_que": 0.3, "hi_ha_que": 0.3, "donar_se_compte": 0.3}},
    )
    values.update(overrides)
    return RunConfig(**values)


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class TestProceduralRun(unittest.TestCase):
    def test_complete_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = Pipeline(config(tmp), lt_client=FakeLanguageTool(), log=lambda _: None).run()
            self.assertEqual(result.status, "complete")
            self.assertEqual(result.path.name, "dataset.jsonl")
            items = read_jsonl(result.path)
            self.assertEqual(len(items), 300)
            ids = [e["metadata"]["item_id"] for e in items]
            self.assertEqual(len(ids), len(set(ids)))
            manifest = result.manifest
            self.assertTrue(manifest["shares"]["ok"])
            self.assertIn("no judge used: pairs were kept on the automatic checks alone", manifest["warnings"])
            for entry in items:
                self.assertEqual(entry["metadata"]["qc"]["decision"], {"judge": "not_used"})
            # LanguageTool missing compound tenses is reported, not rejected.
            hi_ha = manifest["languagetool"]["hi_ha_que"]
            self.assertLess(hi_ha["flag_rate"], 1.0)
            self.assertTrue(hi_ha["unflagged_examples"])
            for name in ("candidates.jsonl", "checks.jsonl", "rejected.jsonl", "manifest.json", "state.json"):
                self.assertTrue((Path(tmp) / "run" / name).exists(), name)

    def test_summary_command(self):
        from contextlib import redirect_stderr, redirect_stdout
        from io import StringIO

        from rauxa.cli import main

        with tempfile.TemporaryDirectory() as tmp:
            Pipeline(config(tmp), lt_client=FakeLanguageTool(), log=lambda _: None).run()
            out, err = StringIO(), StringIO()
            with redirect_stdout(out), redirect_stderr(err):
                self.assertEqual(main(["summary", str(Path(tmp) / "run"), "--samples", "2"]), 0)
                self.assertEqual(main(["summary", str(Path(tmp) / "missing")]), 1)
            self.assertIn("Status:   complete", out.getvalue())
            self.assertEqual(out.getvalue().count("  ✓ "), 2)
            self.assertIn("no run found", err.getvalue())

    def test_existing_run_requires_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = config(tmp)
            first = Pipeline(cfg, lt_client=FakeLanguageTool(), log=lambda _: None).run()
            with self.assertRaises(PipelineError):
                Pipeline(cfg, lt_client=FakeLanguageTool(), log=lambda _: None)
            again = Pipeline(cfg, resume=True, lt_client=FakeLanguageTool(), log=lambda _: None).run()
            self.assertEqual(
                [e["metadata"]["item_id"] for e in read_jsonl(first.path)],
                [e["metadata"]["item_id"] for e in read_jsonl(again.path)],
            )

    def test_resume_refuses_changed_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            Pipeline(config(tmp), lt_client=FakeLanguageTool(), log=lambda _: None).run()
            with self.assertRaises(PipelineError):
                Pipeline(config(tmp, target_size=301), resume=True, lt_client=FakeLanguageTool(), log=lambda _: None)

    def test_running_out_of_pairs_gives_incomplete(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = config(
                tmp, target_size=5000, max_draws_per_round=3000,
                sources=[procedural_source(phenomena=["hi_ha_que"], tenses=["present"])], min_share={},
            )
            result = Pipeline(cfg, lt_client=FakeLanguageTool(), log=lambda _: None).run()
            self.assertEqual(result.status, "incomplete")
            self.assertEqual(result.path.name, "dataset.incomplete.jsonl")
            self.assertLess(result.selected, 5000)
            self.assertTrue(any("ran out of unique pairs" in w for w in result.manifest["warnings"]))


class TestLLMRoles(unittest.TestCase):
    PAIRS = [
        {"good": "Aquest hivern haurem d'encendre la llar de foc.", "bad": "Aquest hivern tindrem que encendre la llar de foc.", "alternatives": []},
        {"good": "Els actors han d'assajar cada tarda.", "bad": "Els actors tenen que assajar cada tarda.", "alternatives": []},
        # Fails the orthography check ("de acabar"), so the writer is asked to fix it.
        {"good": "Dissabte hem de acabar la maqueta.", "bad": "Dissabte tenim que acabar la maqueta.", "alternatives": []},
    ]

    def _config(self, tmp, on_error="reject"):
        return config(
            tmp, target_size=50, min_share={},
            sources=[
                procedural_source(phenomena=["tenir_que"]),
                SourceConfig(type="llm", task=TASK, backend="writer", count=3, batch_size=3, phenomena=["tenir_que"]),
            ],
            judge=JudgeConfig(enabled=True, backend="judge", on_error=on_error, workers=2),
            backends={"writer": BackendConfig(base_url="http://x", model="qwen2.5-72b"), "judge": BackendConfig(base_url="http://y", model="qwen3-235b")},
        )

    def test_writer_fix_and_judge(self):
        writer = FakeBackend("qwen2.5-72b", pairs=self.PAIRS)
        judge = FakeBackend("qwen3-235b", verdict={"actors": {"natural": False, "comment": "poc natural"}})
        with tempfile.TemporaryDirectory() as tmp:
            result = Pipeline(
                self._config(tmp), lt_client=FakeLanguageTool(),
                backends={"writer": writer, "judge": judge}, log=lambda _: None,
            ).run()
            run_dir = Path(tmp) / "run"
            candidates = read_jsonl(run_dir / "candidates.jsonl")
            written = [e for e in candidates if e["metadata"]["source"]["type"] == "llm"]
            self.assertEqual(len(written), 4)  # 3 written + 1 fixed
            fixed = [e for e in written if e["metadata"]["revision_of"]]
            self.assertEqual(len(fixed), 1)

            kept_bad = {e["metadata"]["sentence_bad"] for e in read_jsonl(result.path)}
            rejected_bad = {e["metadata"]["sentence_bad"] for e in read_jsonl(run_dir / "rejected.jsonl")}
            self.assertNotIn("Els actors tenen que assajar cada tarda.", kept_bad)
            self.assertIn("Els actors tenen que assajar cada tarda.", rejected_bad)
            self.assertEqual(result.manifest["judge"]["failed_questions"], {"natural": 1})
            self.assertTrue(any("same model family" in w for w in result.manifest["warnings"]))

    def test_judge_error_policy(self):
        for on_error, expect_kept in (("reject", False), ("keep", True)):
            with tempfile.TemporaryDirectory() as tmp:
                result = Pipeline(
                    self._config(tmp, on_error=on_error), lt_client=FakeLanguageTool(),
                    backends={"writer": FakeBackend("qwen2.5-72b", pairs=self.PAIRS),
                              "judge": FakeBackend("gpt-4o", fail=True)},
                    log=lambda _: None,
                ).run()
                self.assertGreater(result.manifest["judge"]["errors"], 0)
                self.assertEqual(result.manifest["judge"]["kept_despite_error"] > 0, expect_kept)
                if not expect_kept:
                    self.assertEqual(result.selected, 0)
                    self.assertEqual(result.status, "incomplete")


class TestAgentBackend(unittest.TestCase):
    """The run stops with status "waiting"; an agent answers the request files; --resume continues."""

    def test_one_request_per_writer_batch(self):
        # Several batches for the same phenomenon must not collapse into one request.
        with tempfile.TemporaryDirectory() as tmp:
            cfg = config(
                tmp, target_size=50, min_share={}, checks=ChecksConfig(languagetool=LanguageToolCheckConfig(enabled=False)),
                sources=[SourceConfig(type="llm", task=TASK, backend="writer", count=12, batch_size=3, phenomena=["tenir_que"])],
                backends={"writer": BackendConfig(type="agent", model="claude-sonnet-5-5")},
            )
            result = Pipeline(cfg, log=lambda _: None).run()
            self.assertEqual(result.status, "waiting")
            self.assertEqual(len(result.manifest["pending_requests"]), 4)

    def test_agent_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = config(
                tmp, target_size=50, min_share={},
                sources=[
                    procedural_source(phenomena=["tenir_que"]),
                    SourceConfig(type="llm", task=TASK, backend="writer", count=3, batch_size=3, phenomena=["tenir_que"]),
                ],
                judge=JudgeConfig(enabled=True, backend="judge", source_types=["llm"], workers=1),
                backends={
                    "writer": BackendConfig(type="agent", model="claude-sonnet-5-5"),
                    "judge": BackendConfig(type="agent", model="gpt-5"),
                },
            )
            # Plays the agent: answers each request with the same logic as the fake API backend.
            agent = FakeBackend("agent", pairs=TestLLMRoles.PAIRS, verdict={"actors": {"natural": False}})
            exchange = Path(tmp) / "run" / "agent_exchange"
            statuses = []
            for attempt in range(10):
                result = Pipeline(cfg, resume=attempt > 0, lt_client=FakeLanguageTool(), log=lambda _: None).run()
                statuses.append(result.status)
                if result.status != "waiting":
                    break
                for request in sorted((exchange / "requests").glob("*.json")):
                    data = json.loads(request.read_text(encoding="utf-8"))
                    self.assertTrue((exchange / "requests" / f"{data['id']}.md").exists())
                    answer = agent.chat_json(data["messages"])
                    (exchange / "responses" / f"{data['id']}.json").write_text(json.dumps(answer, ensure_ascii=False), encoding="utf-8")

            self.assertEqual(statuses[0], "waiting")
            self.assertEqual(result.status, "complete")
            self.assertGreaterEqual(len(statuses), 3)  # write, then fix + judge, then the rest
            self.assertFalse(list((exchange / "requests").glob("*.json")))  # answered requests move to done/

            written = [e for e in read_jsonl(Path(tmp) / "run" / "candidates.jsonl") if e["metadata"]["source"]["type"] == "llm"]
            self.assertEqual(len(written), 4)  # 3 written + 1 fixed
            kept = read_jsonl(result.path)
            self.assertIn("not_used_for_source", {e["metadata"]["qc"]["decision"].get("judge") for e in kept})
            self.assertNotIn("Els actors tenen que assajar cada tarda.", {e["metadata"]["sentence_bad"] for e in kept})
            self.assertTrue(any("played by a coding agent" in w for w in result.manifest["warnings"]))


class TestCalibration(unittest.TestCase):
    def test_kappa(self):
        self.assertEqual(cohen_kappa([(True, True), (False, False)]), 1.0)
        self.assertEqual(cohen_kappa([(True, False), (False, True)]), -1.0)
        self.assertIsNone(cohen_kappa([]))

    def test_calibrate(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            verdict_ok = {"good_is_correct": True, "variants_are_correct": True, "bad_has_stated_error": True,
                          "bad_has_other_errors": False, "natural": True, "same_meaning": True, "passed": True}
            verdict_bad = {**verdict_ok, "natural": False, "passed": False}
            (run / "judge.jsonl").write_text(
                json.dumps({"item_id": "a", "verdict": verdict_ok}) + "\n" + json.dumps({"item_id": "b", "verdict": verdict_bad}) + "\n",
                encoding="utf-8",
            )
            (run / "checks.jsonl").write_text(
                json.dumps({"item_id": "a", "passed": True}) + "\n" + json.dumps({"item_id": "b", "passed": True}) + "\n",
                encoding="utf-8",
            )
            labels = run / "labels.jsonl"
            labels.write_text(
                json.dumps({"item_id": "a", "good_ok": True, "bad_ok": True}) + "\n"
                + json.dumps({"item_id": "b", "good_ok": False, "bad_ok": True, "note": "forçada"}) + "\n",
                encoding="utf-8",
            )
            report = calibrate(labels, run)
            self.assertEqual(report["judge_vs_human"]["item"]["agreement"], 1.0)
            self.assertEqual(report["automatic_checks"]["human_rejection_rate"], 0.5)
            self.assertEqual(report["automatic_checks"]["notes"], ["forçada"])


if __name__ == "__main__":
    unittest.main()
