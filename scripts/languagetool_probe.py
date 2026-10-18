"""Run every distinct sentence of a task through LanguageTool and report, per phenomenon:

- matches on correct sentences (these would reject the item in the pipeline), with an example;
- the rules that flag the incorrect sentences, and how many go unflagged.

Used to fill in a task's ``languagetool_rules`` and ``languagetool_tolerated``.
Needs a LanguageTool server (see GETTING_STARTED.md).

Usage: python scripts/languagetool_probe.py [task] [draws] [url]
"""

import sys
from collections import Counter, defaultdict

import rauxa
from rauxa.qc.languagetool import LanguageToolClient

task = sys.argv[1] if len(sys.argv) > 1 else "anglicismes"
draws = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
url = sys.argv[3] if len(sys.argv) > 3 else "http://localhost:8081"

client = LanguageToolClient(url=url, language="ca-ES")
good: dict[str, str] = {}
bad: dict[str, str] = {}
for entry in rauxa.create_dataset(task, size=draws, seed=7):
    meta = entry["metadata"]
    for sentence in meta["correct_variants"]:
        good.setdefault(sentence, meta["phenomenon"])
    if meta.get("sentence_bad"):
        bad.setdefault(meta["sentence_bad"], meta["phenomenon"])

good_rules: dict[str, Counter] = defaultdict(Counter)
examples: dict[tuple[str, str], str] = {}
for sentence, phenomenon in good.items():
    for match in client.check(sentence):
        rule = match["rule"]["id"]
        good_rules[phenomenon][rule] += 1
        examples.setdefault((phenomenon, rule), sentence)

bad_rules: dict[str, Counter] = defaultdict(Counter)
totals, unflagged = Counter(), Counter()
for sentence, phenomenon in bad.items():
    rules = {match["rule"]["id"] for match in client.check(sentence)}
    totals[phenomenon] += 1
    unflagged[phenomenon] += not rules
    bad_rules[phenomenon].update(rules)

print(f"{len(good)} correct and {len(bad)} incorrect sentences\n")
print("Matches on correct sentences:")
for phenomenon, rules in sorted(good_rules.items()):
    for rule, n in rules.most_common():
        print(f"  {phenomenon:<28} {rule:<40} {n:>5}  e.g. {examples[(phenomenon, rule)]}")
print("\nRules on incorrect sentences:")
for phenomenon in sorted(totals):
    rules = ", ".join(f"{r} ({n})" for r, n in bad_rules[phenomenon].most_common())
    print(f"  {phenomenon:<28} {totals[phenomenon]:>5} sentences, {unflagged[phenomenon]:>5} unflagged  {rules}")
