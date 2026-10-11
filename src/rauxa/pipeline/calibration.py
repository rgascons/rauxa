"""Human review, and how well the LLM judge and automatic checks match it.

``review`` shows a sample of pairs to a native speaker in the terminal and stores labels.
``calibrate`` compares those labels with the judge's verdicts (agreement, Cohen's kappa) and
measures how many pairs that passed the automatic checks a human would still reject.
"""

from __future__ import annotations

import json
from pathlib import Path
from random import Random
from typing import Any, Callable, Iterable, Optional

from .agents import failed_questions


def _load(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _ask(prompt: str, input_fn: Callable[[str], str]) -> Optional[bool]:
    while True:
        answer = input_fn(prompt).strip().lower()
        if answer in ("s", "sí", "si", "y"):
            return True
        if answer in ("n", "no"):
            return False
        if answer in ("q", "quit"):
            return None


def review(
    items_path: Path, labels_path: Path, *, n: int, seed: int, reviewer: str,
    input_fn: Callable[[str], str] = input, print_fn: Callable[..., None] = print,
) -> int:
    """Label up to ``n`` randomly sampled items not yet labelled. Returns how many were labelled."""
    items = _load(items_path)
    done = {label["item_id"] for label in _load(labels_path)}
    todo = [e for e in items if e["metadata"]["item_id"] not in done]
    Random(seed).shuffle(todo)
    labelled = 0
    for entry in todo[:n]:
        meta = entry["metadata"]
        others = [v for v in meta["correct_variants"] if v != meta["sentence_good"]]
        print_fn(f"\n[{labelled + 1}/{min(n, len(todo))}] fenomen: {meta['phenomenon']}  ({meta['item_id']})")
        print_fn(f"  ✓ {meta['sentence_good']}")
        for v in others:
            print_fn(f"  ✓ {v}")
        print_fn(f"  ✗ {meta['sentence_bad']}")
        good_ok = _ask("  Les frases ✓ són correctes i naturals? [s/n/q] ", input_fn)
        if good_ok is None:
            break
        bad_ok = _ask("  La frase ✗ conté només l'error esperat? [s/n/q] ", input_fn)
        if bad_ok is None:
            break
        note = input_fn("  Comentari (opcional): ").strip()
        with labels_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "item_id": meta["item_id"], "good_ok": good_ok, "bad_ok": bad_ok,
                "note": note, "reviewer": reviewer,
            }, ensure_ascii=False) + "\n")
        labelled += 1
    return labelled


def cohen_kappa(pairs: Iterable[tuple[bool, bool]]) -> Optional[float]:
    pairs = list(pairs)
    n = len(pairs)
    if not n:
        return None
    observed = sum(a == b for a, b in pairs) / n
    p_a = sum(a for a, _ in pairs) / n
    p_b = sum(b for _, b in pairs) / n
    expected = p_a * p_b + (1 - p_a) * (1 - p_b)
    if expected == 1:
        return 1.0 if observed == 1 else 0.0
    return round((observed - expected) / (1 - expected), 4)


def _agreement(pairs: list[tuple[bool, bool]]) -> dict[str, Any]:
    return {
        "n": len(pairs),
        "agreement": round(sum(a == b for a, b in pairs) / len(pairs), 4) if pairs else None,
        "kappa": cohen_kappa(pairs),
    }


def calibrate(labels_path: Path, run_dir: Path) -> dict[str, Any]:
    labels = {label["item_id"]: label for label in _load(labels_path)}
    verdicts = {r["item_id"]: r["verdict"] for r in _load(run_dir / "judge.jsonl") if "verdict" in r}
    checks = {r["item_id"]: r["passed"] for r in _load(run_dir / "checks.jsonl")}

    item, good, bad = [], [], []
    for iid, label in labels.items():
        verdict = verdicts.get(iid)
        if verdict is None:
            continue
        failed = set(failed_questions(verdict))
        item.append((label["good_ok"] and label["bad_ok"], verdict["passed"]))
        good.append((label["good_ok"], not failed & {"good_is_correct", "variants_are_correct", "natural"}))
        bad.append((label["bad_ok"], not failed & {"bad_has_stated_error", "bad_has_other_errors"}))

    passed_checks = [l for iid, l in labels.items() if checks.get(iid)]
    human_rejected = [l for l in passed_checks if not (l["good_ok"] and l["bad_ok"])]
    return {
        "labels": len(labels),
        "judge_vs_human": {"item": _agreement(item), "good_sentences": _agreement(good), "bad_sentence": _agreement(bad)},
        "automatic_checks": {
            "labelled_and_passed": len(passed_checks),
            "human_rejected": len(human_rejected),
            "human_rejection_rate": round(len(human_rejected) / len(passed_checks), 4) if passed_checks else None,
            "notes": [l["note"] for l in human_rejected if l.get("note")][:20],
        },
    }
