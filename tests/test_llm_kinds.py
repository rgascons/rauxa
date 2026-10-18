"""LLM writer, fixes and judge for preferences, controls and context (anglicismes)."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from rauxa.pipeline import Pipeline, PipelineError
from rauxa.pipeline.agents import Judge, Writer, WrittenPair, failed_questions
from rauxa.pipeline.config import BackendConfig, ChecksConfig, JudgeConfig, LanguageToolCheckConfig, RunConfig, SourceConfig
from rauxa.pipeline.llm import LLMError, validate_agent_answer
from rauxa.tasks.anglicismes import PHENOMENON_INFO
from rauxa.tasks.castellanismes_verbals import PHENOMENON_INFO as CASTELLANISMES_INFO

TASK = "anglicismes"

ITEMS = {
    "pair": [
        {"good": "Quan la Rosa es va adonar del perill, ja era tard.", "bad": "Quan la Rosa va realitzar el perill, ja era tard.", "alternatives": []},
        # Fails the orthography check ("de anar"): its writer is asked to fix it.
        {"good": "En Lluís es va adonar que havia de anar-hi.", "bad": "En Lluís va realitzar que havia de anar-hi.", "alternatives": []},
    ],
    "preference": [
        {"good": "El jutge no va trobar cap prova contra ells.", "bad": "El jutge no va trobar cap evidència contra ells.",
         "alternatives": [], "context": "Una notícia de successos."},
        {"good": "Tenim proves que el pont té esquerdes.", "bad": "Tenim evidències que el pont té esquerdes.", "alternatives": []},
    ],
    "control": [
        {"sentence": "L'equip va realitzar l'estudi en dos anys."},
        {"sentence": "Ningú no pot negar l'evidència dels fets."},
    ],
}


class KindsBackend:
    """Answers by the kind the system prompt asks for; records every prompt."""

    def __init__(self, model):
        self.model = model
        self.calls = []

    def chat_json(self, messages):
        self.calls.append(messages)
        system, user = messages[0]["content"], messages[-1]["content"]
        kind = "preference" if "preference pair" in system else "control" if "Each item is a control" in system else "pair"
        if "did not pass our checks" in user:
            return {"good": "En Lluís es va adonar que havia d'anar-hi.", "bad": "En Lluís va realitzar que havia d'anar-hi.", "alternatives": []}
        if system.startswith("You are a Catalan linguist"):
            return {"items": ITEMS[kind]}
        if "control item" in user:
            return {"sentence_is_correct": True, "uses_correct_sense": True, "contains_the_form": True, "natural": True, "comment": ""}
        if "preference, not an error" in user:
            verdict = {"good_is_correct": True, "variants_are_correct": True, "bad_is_acceptable": True, "good_is_preferred": True,
                       "only_stated_difference": True, "natural": True, "same_meaning": True, "comment": ""}
            if "pont" in user:
                verdict.update(good_is_preferred=False, comment="cap diferència de registre")
            return verdict
        return {"good_is_correct": True, "variants_are_correct": True, "bad_has_stated_error": True,
                "bad_has_other_errors": False, "natural": True, "same_meaning": True, "comment": ""}


def _config(tmp, **source):
    return RunConfig(
        name="run", target_size=5, seed=2, output_dir=tmp, min_share={},
        sources=[SourceConfig(type="llm", task=TASK, backend="writer", count=8, batch_size=2,
                              phenomena=["sem.realitzar_adonar", "sem.evidencia_proves"], **source)],
        checks=ChecksConfig(languagetool=LanguageToolCheckConfig(enabled=False)),
        judge=JudgeConfig(enabled=True, backend="judge"),
        backends={"writer": BackendConfig(base_url="http://x", model="qwen2.5-72b"),
                  "judge": BackendConfig(base_url="http://y", model="gpt-4o")},
    )


def _read(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class TestKindsRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.writer, cls.judge = KindsBackend("qwen2.5-72b"), KindsBackend("gpt-4o")
        cls.result = Pipeline(_config(cls.tmp.name), backends={"writer": cls.writer, "judge": cls.judge}, log=lambda _: None).run()
        run = Path(cls.tmp.name) / "run"
        cls.candidates = _read(run / "candidates.jsonl")
        cls.judged = {r["item_id"]: r for r in _read(run / "judge.jsonl")}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _by_kind(self, kind):
        return [e for e in self.candidates if e["metadata"]["kind"] == kind]

    def test_batches_cycle_through_phenomenon_and_kind(self):
        systems = [m[0]["content"] for m in self.writer.calls if "new " in m[-1]["content"]]
        self.assertEqual(len(systems), 4)  # (realitzar: pair, control), (evidència: preference, control)
        self.assertEqual(sum("preference pair" in s for s in systems), 1)
        self.assertEqual(sum("Each item is a control" in s for s in systems), 2)

    def test_items_have_their_kind(self):
        self.assertEqual(len(self._by_kind("preference")), 2)
        self.assertEqual(len(self._by_kind("control")), 2)  # both control batches return the same two
        for entry in self._by_kind("control"):
            meta = entry["metadata"]
            self.assertIsNone(meta["sentence_bad"])
            self.assertIn(meta["format"], ("judgement", "gec"))
            self.assertEqual(meta["category"], "semantic")
        for entry in self._by_kind("preference"):
            self.assertEqual(entry["metadata"]["format"], "mcq")
            self.assertIn("registre formal", entry["question"])

    def test_context_reaches_the_item_and_the_judge(self):
        entry = next(e for e in self.candidates if e["metadata"].get("context"))
        self.assertTrue(entry["question"].startswith("Context: «Una notícia de successos.»"))
        judge_prompt = next(m[-1]["content"] for m in self.judge.calls if "cap prova contra ells" in m[-1]["content"])
        self.assertIn("after this context", judge_prompt)

    def test_gec_admits_unchanged_sentences(self):
        for entry in self.candidates:
            if entry["metadata"]["format"] == "gec":
                self.assertIn("Si ja és correcta", entry["question"])

    def test_control_prompts_describe_the_correct_uses(self):
        user = next(m[-1]["content"] for m in self.writer.calls if "Each item is a control" in m[0]["content"])
        self.assertIn(PHENOMENON_INFO["sem.realitzar_adonar"]["correct_uses"], user)
        self.assertIn("- sentence: ", user)  # examples are control sentences

    def test_failed_pair_is_fixed_as_a_pair(self):
        fixed = [e for e in self.candidates if e["metadata"]["revision_of"]]
        self.assertEqual(len(fixed), 1)
        self.assertEqual(fixed[0]["metadata"]["kind"], "pair")

    def test_judge_verdicts_by_kind(self):
        kinds = {r["verdict"].get("kind", "pair") for r in self.judged.values()}
        self.assertEqual(kinds, {"pair", "preference", "control"})
        failed = self.result.manifest["judge"]["failed_questions"]
        self.assertEqual(failed, {"good_is_preferred": 1})
        kept = {e["metadata"]["sentence_good"] for e in _read(self.result.path)}
        self.assertNotIn("Tenim proves que el pont té esquerdes.", kept)


class TestPlans(unittest.TestCase):
    def _pipeline(self, tmp, **source):
        return Pipeline(_config(tmp, **source), backends={"writer": KindsBackend("a"), "judge": KindsBackend("b")}, log=lambda _: None)

    def test_kinds_and_formats_narrow_the_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(self._pipeline(tmp, kinds=["control"]).plans[0],
                             [("sem.realitzar_adonar", "control"), ("sem.evidencia_proves", "control")])
        with tempfile.TemporaryDirectory() as tmp:
            pipeline = self._pipeline(tmp, formats=["mcq"])
            self.assertEqual(pipeline.plans[0], [("sem.realitzar_adonar", "pair"), ("sem.evidencia_proves", "preference")])
            self.assertFalse(pipeline.allow_unchanged[TASK])

    def test_empty_plan_is_an_error(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(PipelineError):
            self._pipeline(tmp, kinds=["control"], formats=["mcq"])

    def test_unknown_kind(self):
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(AssertionError):
            _config(tmp, kinds=["minimal"]).validate()


class TestAnswers(unittest.TestCase):
    def test_agent_answers_by_kind(self):
        validate_agent_answer({"items": [{"sentence": "Correcta."}]}, "write")
        validate_agent_answer({"sentence": "Correcta."}, "fix")
        validate_agent_answer({"sentence_is_correct": True, "uses_correct_sense": True, "contains_the_form": True,
                               "natural": True, "comment": ""}, "judge")
        with self.assertRaises(LLMError):
            validate_agent_answer({"items": [{"sentence": ""}]}, "write")
        with self.assertRaises(LLMError):
            validate_agent_answer({"sentence_is_correct": True, "natural": True, "comment": ""}, "judge")

    def test_failed_questions_by_kind(self):
        self.assertEqual(failed_questions({"kind": "control", "sentence_is_correct": True, "uses_correct_sense": False,
                                           "contains_the_form": True, "natural": True}), ["uses_correct_sense"])
        old = {"good_is_correct": True, "variants_are_correct": True, "bad_has_stated_error": True,
               "bad_has_other_errors": True, "natural": True, "same_meaning": True}
        self.assertEqual(failed_questions(old), ["bad_has_other_errors"])  # no kind: a pair verdict

    def test_writer_parses_by_kind(self):
        class Fixed:
            model = "m"

            def chat_json(self, messages):
                return {"items": [{"sentence": "Va realitzar el projecte.", "context": " "}, {"good": "x", "bad": "y"}]}

        written = Writer(Fixed()).write(PHENOMENON_INFO["sem.realitzar_adonar"], [], 2, kind="control")
        self.assertEqual(written, [WrittenPair("Va realitzar el projecte.", None, [], "control", None)])


class TestUnchangedPrompts(unittest.TestCase):
    """Tasks without kinds keep the original prompts, so agent runs resume and ids stay the same."""

    def test_castellanismes_prompts(self):
        class Recorder:
            model = "m"

            def __init__(self):
                self.messages = []

            def chat_json(self, messages):
                self.messages.append(messages)
                return {"items": [{"good": "A.", "bad": "B."}]} if "new pairs" in messages[-1]["content"] else {"good": "A.", "bad": "B."}

        info = CASTELLANISMES_INFO["tenir_que"]
        recorder = Recorder()
        Writer(recorder).write(info, [("Hem d'anar-hi.", "Tenim que anar-hi.")], 3, key="k")
        Writer(recorder).fix(info, WrittenPair("A.", "B.", ["C."]), ["orthography: {}"], key="k")
        Judge(recorder).judge({"metadata": {"sentence_good": "A.", "sentence_bad": "B.", "correct_variants": ["A."]}}, info)
        digest = hashlib.sha256(json.dumps(recorder.messages, ensure_ascii=False).encode()).hexdigest()[:16]
        self.assertEqual(digest, "5e56bbf4c1be4ba0")  # main before kinds


if __name__ == "__main__":
    unittest.main()
