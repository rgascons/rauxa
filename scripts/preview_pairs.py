"""Print generated sentence pairs as a table, for human review.

Usage: python scripts/preview_pairs.py [task] [size] [seed]
"""

import sys

import rauxa

task = sys.argv[1] if len(sys.argv) > 1 else "castellanismes_verbals"
size = int(sys.argv[2]) if len(sys.argv) > 2 else 30
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0

kwargs = {"size": size, "seed": seed}
if task == "castellanismes_verbals":
    kwargs["tenses"] = rauxa.tasks.castellanismes_verbals.TENSES

for entry in rauxa.create_dataset(task, **kwargs):
    m = entry["metadata"]
    detail = m.get("tense") or m.get("kind", "")
    print(f"{m['source_index']:>4} {m['phenomenon']:<26} {detail:<17} ✓ {m['sentence_good']}")
    if m.get("sentence_bad"):
        print(f"{'':>4} {'':<26} {'':<17} ✗ {m['sentence_bad']}")
