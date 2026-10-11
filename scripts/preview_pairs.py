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
    print(f"{m['source_index']:>4} {m['phenomenon']:<16} {m['tense']:<17} ✓ {m['sentence_good']}")
    print(f"{'':>4} {'':<16} {'':<17} ✗ {m['sentence_bad']}")
