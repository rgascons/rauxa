"""Item kinds beyond plain pairs: preferences, controls, and context."""

import json
import re
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Any, Optional

from rauxa.dataset import ProceduralDataset
from rauxa.export import export_lm_eval
from rauxa.factory import DATASETS, register_dataset
from rauxa.formats import render, score
from rauxa.pipeline import Pipeline
from rauxa.pipeline.calibration import calibrate
from rauxa.pipeline.checks import LanguageToolCheck, MarkerCheck, MinimalPairCheck, OrthographyCheck, SchemaCheck
from rauxa.pipeline.config import ChecksConfig, LanguageToolCheckConfig, RunConfig, SourceConfig
from rauxa.pipeline.selection import item_id

TASK = "test_item_kinds"

# Looked up by the pipeline through the dataset class's module.
PHENOMENON_INFO = {
    "captura": {
        "error": "«screenshot» per «captura de pantalla».",
        "correction": "«captura de pantalla».",
        "languagetool_rules": ["ANGLICISME"],
        "marker": r"\bscreenshot\b",
    },
}


@dataclass
class KindsConfig:
    seed: Optional[int] = None
    size: int = 500

    def validate(self) -> None:
        pass


def _item(idx: int) -> dict[str, Any]:
    """Cycle pair → preference → control, with unique sentences per index."""
    rng = Random(idx)
    kind = ("pair", "preference", "control")[idx % 3]
    n = f"{idx}"
    if kind == "pair":
        good, bad, fmt = f"El grup {n} ha fet una captura de pantalla.", f"El grup {n} ha fet un screenshot.", "mcq"
        context = None
    elif kind == "preference":
        good, bad, fmt = f"La tasca {n} demana una captura de pantalla.", f"La tasca {n} demana un screenshot.", "mcq"
        context = "Un informe formal."
    else:
        good, bad, fmt = f"El grup {n} ha penjat el document al web.", None, ("judgement", "gec")[idx % 2]
        context = None
    question, answer, extra = render(fmt, good, bad, rng, kind=kind, context=context, allow_unchanged=True)
    meta = {
        "source_dataset": TASK, "source_index": idx, "format": fmt, "phenomenon": "captura",
        "kind": kind, "sentence_good": good, "sentence_bad": bad, "correct_variants": [good],
        "languagetool_rules": ["ANGLICISME"], **extra,
    }
    if context:
        meta["context"] = context
    if kind == "control":
        meta["languagetool_tolerated"] = ["WEB_ANGLICISME"]
    return {"question": question, "answer": answer, "metadata": meta}


class KindsDataset(ProceduralDataset):
    def __init__(self, config: KindsConfig):
        super().__init__(config=config, seed=config.seed, size=config.size)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return _item(idx)


if TASK not in DATASETS:
    register_dataset(TASK, KindsDataset, KindsConfig)


class FakeLanguageTool:
    """Flags «screenshot», and flags «web» with a rule the control tolerates."""

    def check(self, text):
        matches = []
        if re.search(r"\bscreenshot\b", text):
            matches.append({"rule": {"id": "ANGLICISME"}, "message": "anglicisme", "offset": 0, "length": 1})
        if re.search(r"\bweb\b", text):
            matches.append({"rule": {"id": "WEB_ANGLICISME"}, "message": "anglicisme", "offset": 0, "length": 1})
        return matches


class TestRender(unittest.TestCase):
    def test_control_formats(self):
        q, a, extra = render("judgement", "Bon dia.", None, Random(0), kind="control")
        self.assertEqual((a, extra["shown"]), ("sí", "Bon dia."))
        q, a, _ = render("gec", "Bon dia.", None, Random(0), kind="control", allow_unchanged=True)
        self.assertIn("Si ja és correcta, torna-la igual.", q)
        self.assertEqual(a, "Bon dia.")

    def test_preference_wording_never_says_correct(self):
        q, a, extra = render("mcq", "Formal.", "Informal.", Random(0), kind="preference")
        self.assertIn("més adequada en un registre formal", q)
        self.assertNotIn("correcta", q)
        self.assertEqual(extra["options"][extra["target_index"]], "Formal.")

    def test_context_prefix(self):
        q, _, _ = render("mcq", "A.", "B.", Random(0), context="Una reunió de feina.")
        self.assertTrue(q.startswith("Context: «Una reunió de feina.»\n\n"))

    def test_invalid_combinations(self):
        for kwargs in (
            dict(fmt="mcq", bad=None, kind="control"),
            dict(fmt="judgement", bad="B.", kind="preference"),
            dict(fmt="gec", bad="B.", kind="preference"),
            dict(fmt="mcq", bad=None, kind="pair"),
            dict(fmt="judgement", bad="B.", kind="control"),
        ):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                render(kwargs["fmt"], "A.", kwargs["bad"], Random(0), kind=kwargs["kind"])

    def test_scoring_controls(self):
        judgement, gec = _item(2), _item(5)
        self.assertEqual(score("sí", judgement), 1.0)
        self.assertEqual(score("no", judgement), 0.0)
        good = gec["metadata"]["sentence_good"]
        self.assertEqual(score(good, gec), 1.0)  # leaving a correct sentence alone is right
        self.assertEqual(score(good.replace("web", "lloc web"), gec), 0.0)  # overcorrection


class TestChecksAndIdentity(unittest.TestCase):
    def test_checks_on_controls(self):
        control = _item(2)
        self.assertTrue(SchemaCheck().run(control).passed)
        self.assertTrue(OrthographyCheck().run(control).passed)
        self.assertEqual(MarkerCheck({"captura": r"\bweb\b"}).run(control).details, {"skipped": "control"})
        self.assertTrue(MinimalPairCheck(8).run(control).passed)
        result = LanguageToolCheck(FakeLanguageTool(), require_bad_flagged=False).run(control)
        self.assertTrue(result.passed)
        self.assertIn("tolerated_matches", result.details)  # recorded, not hidden

    def test_untolerated_match_rejects_control(self):
        control = _item(2)
        control["metadata"]["languagetool_tolerated"] = []
        self.assertFalse(LanguageToolCheck(FakeLanguageTool(), require_bad_flagged=False).run(control).passed)

    def test_schema_rejects_bad_sentence_on_control(self):
        control = _item(2)
        control["metadata"]["sentence_bad"] = "Hola."
        self.assertFalse(SchemaCheck().run(control).passed)

    def test_identity(self):
        pair = _item(0)
        ids = {item_id(pair)}
        as_control = json.loads(json.dumps(pair))
        as_control["metadata"].update(kind="control", sentence_bad=None)
        with_context = json.loads(json.dumps(pair))
        with_context["metadata"]["context"] = "Un altre context."
        for variant in (as_control, with_context):
            self.assertNotIn(item_id(variant), ids)
            ids.add(item_id(variant))
        plain = json.loads(json.dumps(pair))
        plain["metadata"].pop("kind")
        self.assertEqual(item_id(plain), item_id(pair))  # kind="pair" and no kind are the same item


class TestExport(unittest.TestCase):
    def test_kinds_are_exported_separately(self):
        entries = [_item(i) for i in range(9)]
        with tempfile.TemporaryDirectory() as tmp:
            names = {p.name for p in export_lm_eval(entries, Path(tmp), "k")}
            self.assertEqual(names, {"k_pairs.jsonl", "k_pairs.yaml", "k_preference.jsonl", "k_preference.yaml",
                                     "k_mcq.jsonl", "k_mcq.yaml"})
            pairs = [json.loads(l) for l in (Path(tmp) / "k_pairs.jsonl").read_text(encoding="utf-8").splitlines()]
            prefs = [json.loads(l) for l in (Path(tmp) / "k_preference.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(pairs), 3)
            self.assertEqual(len(prefs), 3)
            self.assertTrue(all(p["kind"] == "preference" and p["context"] for p in prefs))
            self.assertIn('doc_to_text: "{{context}}"', (Path(tmp) / "k_preference.yaml").read_text(encoding="utf-8"))
            self.assertIn('doc_to_text: ""', (Path(tmp) / "k_pairs.yaml").read_text(encoding="utf-8"))
            # Controls have no counterpart to score against and stay out of lm-eval.
            self.assertFalse(any("ha penjat" in l for l in (Path(tmp) / "k_mcq.jsonl").read_text(encoding="utf-8").splitlines()))


class TestPipelineAndReview(unittest.TestCase):
    def test_pipeline_keeps_all_kinds(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = RunConfig(
                name="kinds", target_size=30, seed=0, output_dir=tmp,
                sources=[SourceConfig(type="procedural", task=TASK)],
                checks=ChecksConfig(languagetool=LanguageToolCheckConfig(workers=1)),
                min_share={"kind": {"pair": 0.3, "preference": 0.3, "control": 0.3}},
            )
            result = Pipeline(cfg, lt_client=FakeLanguageTool(), log=lambda _: None).run()
            self.assertEqual(result.status, "complete")
            self.assertEqual(result.manifest["check_failures"], {})
            run_dir = Path(tmp) / "kinds"
            items = [json.loads(l) for l in result.path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual({e["metadata"]["kind"] for e in items}, {"pair", "preference", "control"})
            # Only pairs and preferences count towards LanguageTool coverage.
            candidates = [json.loads(l) for l in (run_dir / "candidates.jsonl").read_text(encoding="utf-8").splitlines()]
            lt = result.manifest["languagetool"]["captura"]
            self.assertEqual(lt["checked"], sum(1 for e in candidates if e["metadata"]["kind"] != "control"))

            labels = run_dir / "labels.jsonl"
            control = next(e for e in items if e["metadata"]["kind"] == "control")
            pair = next(e for e in items if e["metadata"]["kind"] == "pair")
            labels.write_text(
                json.dumps({"item_id": control["metadata"]["item_id"], "good_ok": True, "bad_ok": None}) + "\n"
                + json.dumps({"item_id": pair["metadata"]["item_id"], "good_ok": True, "bad_ok": False}) + "\n",
                encoding="utf-8",
            )
            report = calibrate(labels, run_dir)
            self.assertEqual(report["automatic_checks"]["human_rejected"], 1)  # the pair, not the control

    def test_review_shows_context_and_asks_one_question_for_controls(self):
        from rauxa.pipeline.calibration import review

        with tempfile.TemporaryDirectory() as tmp:
            items_path, labels_path = Path(tmp) / "items.jsonl", Path(tmp) / "labels.jsonl"
            entries = [_item(1), _item(2)]
            for e in entries:
                e["metadata"]["item_id"] = item_id(e)
            items_path.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries), encoding="utf-8")
            printed, asked = [], []

            def answer(prompt):
                asked.append(prompt)
                return "" if "Comentari" in prompt else "s"

            n = review(items_path, labels_path, n=2, seed=0, reviewer="t", input_fn=answer, print_fn=printed.append)
            self.assertEqual(sum("[s/n/q]" in p for p in asked), 3)  # two for the preference, one for the control
            self.assertEqual(n, 2)
            self.assertTrue(any(line.startswith("  Context: ") for line in printed))
            labels = [json.loads(l) for l in labels_path.read_text(encoding="utf-8").splitlines()]
            control_label = next(l for l in labels if l["item_id"] == entries[1]["metadata"]["item_id"])
            self.assertIsNone(control_label["bad_ok"])


if __name__ == "__main__":
    unittest.main()
