import re
import tempfile
import unittest
from pathlib import Path

import rauxa
from rauxa.export import export_lm_eval
from rauxa.formats import score
from rauxa.catalan.checks import orthography_issues
from rauxa.tasks.castellanismes_verbals import PHENOMENA, PHENOMENON_INFO, TENSES

NAME = "castellanismes_verbals"

_CALQUE_MARKERS = {p: re.compile(info["marker"], re.IGNORECASE) for p, info in PHENOMENON_INFO.items()}


class TestDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = rauxa.create_dataset(
            NAME, size=5000, seed=123, tenses=TENSES, formats=("mcq", "judgement", "gec")
        )
        cls.entries = list(cls.data)

    def test_registered(self):
        self.assertIn(NAME, rauxa.DATASETS)

    def test_deterministic(self):
        again = rauxa.create_dataset(NAME, size=5000, seed=123, tenses=TENSES, formats=("mcq", "judgement", "gec"))
        for idx in (0, 17, 4999):
            self.assertEqual(self.data[idx], again[idx])

    def test_seed_changes_items(self):
        other = rauxa.create_dataset(NAME, size=50, seed=124, tenses=TENSES)
        self.assertNotEqual([e["question"] for e in other], [e["question"] for e in self.entries[:50]])

    def test_schema(self):
        for entry in self.entries[:200]:
            self.assertEqual(set(entry), {"question", "answer", "metadata"})
            meta = entry["metadata"]
            self.assertEqual(meta["source_dataset"], NAME)
            self.assertIn(meta["phenomenon"], PHENOMENA)
            self.assertIn(meta["sentence_good"], meta["correct_variants"])

    def test_coverage(self):
        seen = {(e["metadata"]["phenomenon"], e["metadata"]["tense"]) for e in self.entries}
        self.assertEqual({p for p, _ in seen}, set(PHENOMENA))
        self.assertEqual({t for _, t in seen}, set(TENSES))
        self.assertNotIn("imperfect", {t for p, t in seen if p == "donar_se_compte"})

    def test_orthography(self):
        for entry in self.entries:
            meta = entry["metadata"]
            for text in [meta["sentence_bad"], *meta["correct_variants"]]:
                self.assertEqual(orthography_issues(text), [], text)

    def test_calque_only_in_bad_sentence(self):
        for entry in self.entries:
            meta = entry["metadata"]
            marker = _CALQUE_MARKERS[meta["phenomenon"]]
            self.assertIsNotNone(marker.search(meta["sentence_bad"]), meta["sentence_bad"])
            for good in meta["correct_variants"]:
                self.assertIsNone(marker.search(good), good)

    def test_oracle_answers_score_one(self):
        for entry in self.entries:
            self.assertEqual(self.data.score_answer(entry["answer"], entry), 1.0)

    def test_score_answer_fn(self):
        fn = rauxa.get_score_answer_fn(NAME)
        self.assertEqual(fn(self.entries[0]["answer"], self.entries[0]), 1.0)


class TestScoring(unittest.TestCase):
    def _entry(self, fmt):
        return next(
            e for e in rauxa.create_dataset(NAME, size=200, seed=1, formats=(fmt,), phenomena=("hi_ha_que",))
        )

    def test_mcq(self):
        entry = self._entry("mcq")
        letter = entry["answer"]
        wrong = "B" if letter == "A" else "A"
        self.assertEqual(score(f"{letter})", entry), 1.0)
        self.assertEqual(score(f"La resposta correcta és la {letter}.", entry), 1.0)
        self.assertEqual(score(wrong, entry), 0.0)
        self.assertEqual(score(None, entry), 0.0)

    def test_judgement(self):
        entry = self._entry("judgement")
        expected = entry["answer"]
        self.assertEqual(score(expected.capitalize() + ".", entry), 1.0)
        if expected == "sí":
            self.assertEqual(score("Si", entry), 1.0)
        self.assertEqual(score("no" if expected == "sí" else "sí", entry), 0.0)

    def test_gec_accepts_every_variant(self):
        entry = self._entry("gec")
        variants = entry["metadata"]["correct_variants"]
        self.assertEqual(len(variants), 2)
        for variant in variants:
            self.assertEqual(score(f"«{variant.replace(chr(39), '’')}»", entry), 1.0)
        self.assertEqual(score(entry["metadata"]["sentence_bad"], entry), 0.0)


class TestCurriculumAndConfig(unittest.TestCase):
    def test_curriculum(self):
        curriculum = rauxa.create_curriculum(NAME)
        self.assertEqual(curriculum.generate_configuration({"size": 10, "seed": 0}).tenses, ("present",))
        while curriculum.increment_attr_level("tenses"):
            pass
        self.assertEqual(curriculum.generate_configuration({"size": 10, "seed": 0}).tenses, TENSES)

    def test_invalid_config(self):
        with self.assertRaises(AssertionError):
            rauxa.create_dataset(NAME, size=10, phenomena=("donar_se_compte",), tenses=("imperfect",))
        with self.assertRaises(NotImplementedError):
            rauxa.create_dataset(NAME, size=10, norm="avl")


class TestExport(unittest.TestCase):
    def test_lm_eval_export(self):
        entries = list(rauxa.create_dataset(NAME, size=20, seed=3, formats=("mcq", "gec")))
        with tempfile.TemporaryDirectory() as tmp:
            paths = export_lm_eval(entries, Path(tmp), "rauxa_cv")
            names = {p.name for p in paths}
            self.assertIn("rauxa_cv_pairs.yaml", names)
            self.assertIn("rauxa_cv_mcq.yaml", names)
            yaml = (Path(tmp) / "rauxa_cv_pairs.yaml").read_text(encoding="utf-8")
            self.assertIn('doc_to_choice: "{{[sentence_good, sentence_bad]}}"', yaml)


if __name__ == "__main__":
    unittest.main()
