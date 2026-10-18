import re
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import rauxa
from rauxa.catalan.checks import edit_span, orthography_issues
from rauxa.export import export_lm_eval
from rauxa.formats import KIND_FORMATS
from rauxa.tasks.anglicismes import CATALOG, ENTRIES, PHENOMENA, PHENOMENON_INFO, SPECS

NAME = "anglicismes"
FORMATS = ("mcq", "judgement", "gec")

_MARKERS = {p: re.compile(i["marker"], re.IGNORECASE) for p, i in PHENOMENON_INFO.items() if "marker" in i}


class TestCatalog(unittest.TestCase):
    def test_every_accepted_entry_has_frames_and_nothing_else(self):
        accepted = {e["id"] for e in CATALOG["phenomenon"] if e["status"] == "accepted"}
        self.assertEqual(set(SPECS), accepted)
        self.assertEqual(set(PHENOMENA), accepted)

    def test_rejected_entries_are_not_used(self):
        for entry in CATALOG["phenomenon"]:
            if entry["status"] == "rejected":
                self.assertNotIn(entry["id"], PHENOMENON_INFO)

    def test_kinds_follow_the_assessment(self):
        expected = {"normative_error": "pair", "style_preference": "preference"}
        for pid, info in PHENOMENON_INFO.items():
            kinds = set(info["kinds"])
            self.assertTrue(kinds, pid)
            if info["assessment"] == "accepted":
                self.assertEqual(kinds, {"control"}, pid)
            else:
                self.assertIn(expected[info["assessment"]], kinds, pid)
                self.assertEqual(kinds - {expected[info["assessment"]], "control"}, set(), pid)

    def test_info(self):
        for pid, info in PHENOMENON_INFO.items():
            self.assertTrue(info["error"] and info["correction"], pid)
            self.assertEqual(info["category"], ENTRIES[pid]["category"])
            if "pair" in info["kinds"] or "preference" in info["kinds"]:
                self.assertIn("marker", info, pid)

    def test_slot_names_have_no_digits(self):
        # A trailing digit means "a different value from the same list" ({p2}).
        for pid, spec in SPECS.items():
            for name in spec.slots or {}:
                self.assertRegex(name, r"^[a-z_]+$", pid)


class TestDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = rauxa.create_dataset(NAME, size=12000, seed=123)
        cls.entries = list(cls.data)

    def test_registered(self):
        self.assertIn(NAME, rauxa.DATASETS)

    def test_deterministic(self):
        again = rauxa.create_dataset(NAME, size=12000, seed=123)
        for idx in (0, 17, 11999):
            self.assertEqual(self.data[idx], again[idx])

    def test_schema(self):
        for entry in self.entries:
            meta = entry["metadata"]
            self.assertEqual(meta["source_dataset"], NAME)
            self.assertIn(meta["phenomenon"], PHENOMENA)
            self.assertIn(meta["kind"], PHENOMENON_INFO[meta["phenomenon"]]["kinds"])
            self.assertIn(meta["format"], KIND_FORMATS[meta["kind"]])
            self.assertEqual(meta["sentence_good"], meta["correct_variants"][0])
            self.assertEqual(meta["sentence_bad"] is None, meta["kind"] == "control")
            self.assertNotIn(meta["sentence_bad"], meta["correct_variants"])

    def test_coverage(self):
        seen = Counter((e["metadata"]["phenomenon"], e["metadata"]["kind"]) for e in self.entries)
        for pid, info in PHENOMENON_INFO.items():
            for kind in info["kinds"]:
                self.assertGreater(seen[(pid, kind)], 0, (pid, kind))
        kinds = Counter(e["metadata"]["kind"] for e in self.entries)
        self.assertAlmostEqual(kinds["control"] / len(self.entries), 0.12, delta=0.02)
        self.assertAlmostEqual(kinds["preference"] / len(self.entries), 0.08, delta=0.02)

    def test_sentences_are_complete(self):
        for entry in self.entries:
            meta = entry["metadata"]
            for text in [meta["sentence_bad"] or "", *meta["correct_variants"]]:
                self.assertNotRegex(text, r"[\[\]{}|/]", text)
                self.assertNotIn("  ", text)

    def test_orthography(self):
        for entry in self.entries:
            meta = entry["metadata"]
            for text in [*([meta["sentence_bad"]] if meta["sentence_bad"] else []), *meta["correct_variants"]]:
                self.assertEqual(orthography_issues(text), [], text)

    def test_marker_only_in_bad_sentence(self):
        for entry in self.entries:
            meta = entry["metadata"]
            if meta["kind"] == "control":
                continue
            marker = _MARKERS[meta["phenomenon"]]
            self.assertIsNotNone(marker.search(meta["sentence_bad"]), meta["sentence_bad"])
            for good in meta["correct_variants"]:
                self.assertIsNone(marker.search(good), good)

    def test_minimal_pairs(self):
        for entry in self.entries:
            meta = entry["metadata"]
            if meta["kind"] == "control":
                continue
            for good in meta["correct_variants"]:
                span_good, span_bad, shared = edit_span(good, meta["sentence_bad"])
                self.assertTrue(0 < max(span_good, span_bad) <= 8 and shared > 0, (good, meta["sentence_bad"]))

    def test_grammatical_passive_stays_on_the_correct_side(self):
        variants = {v for e in self.entries if e["metadata"]["phenomenon"] == "syn.passiva_datiu"
                    for v in e["metadata"]["correct_variants"]}
        self.assertTrue(any(re.search(r"\bva ser concedida a\b", v) for v in variants))

    def test_gec_prompt_admits_unchanged_sentences(self):
        gec = [e for e in self.entries if e["metadata"]["format"] == "gec"]
        self.assertTrue(gec)
        for entry in gec:
            self.assertIn("Si ja és correcta, torna-la igual.", entry["question"])

    def test_preference_wording(self):
        prefs = [e for e in self.entries if e["metadata"]["kind"] == "preference"]
        self.assertTrue(prefs)
        for entry in prefs:
            self.assertIn("registre formal", entry["question"])
            self.assertEqual(entry["metadata"]["assessment"], "style_preference")

    def test_context(self):
        with_context = [e for e in self.entries if e["metadata"].get("context")]
        self.assertTrue(with_context)
        for entry in with_context:
            self.assertTrue(entry["question"].startswith("Context: «"))

    def test_oracle_answers_score_one(self):
        for entry in self.entries:
            self.assertEqual(self.data.score_answer(entry["answer"], entry), 1.0)

    def test_control_gec_accepts_the_sentence_itself(self):
        entry = next(e for e in self.entries if e["metadata"]["kind"] == "control" and e["metadata"]["format"] == "gec")
        self.assertEqual(self.data.score_answer(entry["metadata"]["sentence_good"], entry), 1.0)


class TestConfig(unittest.TestCase):
    def test_filters(self):
        entries = list(rauxa.create_dataset(NAME, size=300, seed=1, categories=("syntactic",), kinds=("pair",)))
        self.assertEqual({e["metadata"]["category"] for e in entries}, {"syntactic"})
        self.assertEqual({e["metadata"]["kind"] for e in entries}, {"pair"})

    def test_mcq_only_has_no_controls(self):
        entries = list(rauxa.create_dataset(NAME, size=300, seed=1, formats=("mcq",)))
        self.assertEqual({e["metadata"]["kind"] for e in entries}, {"pair", "preference"})
        self.assertNotIn("Si ja és correcta", "".join(e["question"] for e in entries))

    def test_single_phenomenon(self):
        entries = list(rauxa.create_dataset(NAME, size=50, seed=1, phenomena=("acc.web",)))
        self.assertEqual({e["metadata"]["kind"] for e in entries}, {"control"})

    def test_invalid(self):
        with self.assertRaises(AssertionError):
            rauxa.create_dataset(NAME, size=10, phenomena=("sem.assumir_suposar",))  # rejected
        with self.assertRaises(AssertionError):
            rauxa.create_dataset(NAME, size=10, phenomena=("acc.web",), kinds=("pair",))
        with self.assertRaises(AssertionError):
            rauxa.create_dataset(NAME, size=10, preference_share=0.6, control_share=0.6)
        with self.assertRaises(NotImplementedError):
            rauxa.create_dataset(NAME, size=10, norm="avl")

    def test_curriculum(self):
        curriculum = rauxa.create_curriculum(NAME)
        self.assertEqual(curriculum.generate_configuration({"size": 10, "seed": 0}).categories, ("lexical",))
        while curriculum.increment_attr_level("categories"):
            pass
        self.assertEqual(len(curriculum.generate_configuration({"size": 10, "seed": 0}).categories), 3)


class TestExport(unittest.TestCase):
    def test_lm_eval_export(self):
        entries = list(rauxa.create_dataset(NAME, size=200, seed=3))
        with tempfile.TemporaryDirectory() as tmp:
            names = {p.name for p in export_lm_eval(entries, Path(tmp), "rauxa_ang")}
            self.assertIn("rauxa_ang_pairs.yaml", names)
            self.assertIn("rauxa_ang_preference.yaml", names)


if __name__ == "__main__":
    unittest.main()
