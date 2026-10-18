import unittest
from collections import Counter
from random import Random

from rauxa.pipeline.config import RunConfig, SourceConfig
from rauxa.pipeline.selection import select


def _pool() -> list[dict]:
    # "a" has many more distinct items than "b" and "c".
    sizes = {"a": 300, "b": 40, "c": 60}
    return [{"metadata": {"phenomenon": p, "format": ("mcq", "gec")[i % 2], "n": i}}
            for p, size in sizes.items() for i in range(size)]


class TestSelect(unittest.TestCase):
    def test_without_balance_shares_follow_the_pool(self):
        counts = Counter(e["metadata"]["phenomenon"] for e in select(_pool(), 120, {}, Random(0)))
        self.assertGreater(counts["a"], 70)

    def test_balance_by_evens_out_values(self):
        counts = Counter(e["metadata"]["phenomenon"] for e in select(_pool(), 120, {}, Random(0), "phenomenon"))
        self.assertEqual(counts, {"a": 40, "b": 40, "c": 40})

    def test_balance_by_tops_up_from_larger_values(self):
        counts = Counter(e["metadata"]["phenomenon"] for e in select(_pool(), 200, {}, Random(0), "phenomenon"))
        self.assertEqual(counts["b"], 40)
        self.assertEqual(counts["c"], 60)
        self.assertEqual(counts["a"], 100)

    def test_balance_by_keeps_minimum_shares(self):
        picked = select(_pool(), 90, {"format": {"gec": 0.8}}, Random(0), "phenomenon")
        self.assertGreaterEqual(Counter(e["metadata"]["format"] for e in picked)["gec"], 72)

    def test_unset_balance_by_keeps_the_fingerprint(self):
        # Runs started before balance_by existed must still resume.
        config = RunConfig(name="x", target_size=10, sources=[SourceConfig(type="procedural", task="t")])
        self.assertEqual(config.fingerprint(), "1532ef6fa8846962")
        config.balance_by = "phenomenon"
        self.assertNotEqual(config.fingerprint(), "1532ef6fa8846962")


if __name__ == "__main__":
    unittest.main()
