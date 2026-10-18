"""Pair ids for deduplication, and selection of the dataset under minimum shares."""

from __future__ import annotations

import hashlib
import math
from random import Random
from typing import Any, Optional

from ..formats import normalize_sentence


def item_id(entry: dict[str, Any]) -> str:
    """Stable id of the underlying item (format-independent), used for dedup.

    A plain pair is identified by its two sentences, as it always has been. Kind and context
    join the key only when present, so a control and a pair sharing a sentence, or the same
    sentences under a different context, are different items while existing ids don't change.
    """
    meta = entry["metadata"]
    kind, context = meta.get("kind", "pair"), meta.get("context")
    key = normalize_sentence(meta["sentence_good"]) + "\x00" + normalize_sentence(meta.get("sentence_bad") or "")
    if kind != "pair" or context:
        key = f"{kind}\x00{normalize_sentence(context or '')}\x00{key}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def _value(entry: dict[str, Any], dim: str) -> str:
    return str(entry["metadata"].get(dim))


def share_report(items: list[dict[str, Any]], min_share: dict[str, dict[str, float]], target: int) -> dict[str, Any]:
    """Share of each value per field, checked against the minimum shares (fractions of ``target``)."""
    report: dict[str, Any] = {"ok": True, "fields": {}}
    for dim, minimums in min_share.items():
        counts: dict[str, int] = {}
        for entry in items:
            counts[_value(entry, dim)] = counts.get(_value(entry, dim), 0) + 1
        rows = {}
        for value in sorted(set(counts) | set(minimums)):
            minimum = minimums.get(value, 0.0)
            count = counts.get(value, 0)
            ok = count >= math.ceil(minimum * target)
            rows[value] = {"count": count, "share": round(count / max(len(items), 1), 4), "min_share": minimum, "ok": ok}
            report["ok"] &= ok
        report["fields"][dim] = rows
    return report


def _interleave(pool: list[dict[str, Any]], dim: str) -> list[dict[str, Any]]:
    """Reorder ``pool`` round-robin over the values of ``dim``, keeping the order within each value."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for entry in pool:
        groups.setdefault(_value(entry, dim), []).append(entry)
    longest = max((len(g) for g in groups.values()), default=0)
    return [g[i] for i in range(longest) for g in groups.values() if i < len(g)]


def select(
    pool: list[dict[str, Any]], target: int, min_share: dict[str, dict[str, float]], rng: Random,
    balance_by: Optional[str] = None,
) -> list[dict[str, Any]]:
    """Pick up to ``target`` items: first enough to meet every minimum share, then the rest at random.

    With ``balance_by``, items are taken round-robin over the values of that metadata field, so
    each value gets as even a count as the pool allows. Without it, a value with more distinct
    items (more template combinations) ends up with a larger share.
    """
    pool = list(pool)
    rng.shuffle(pool)
    if balance_by:
        pool = _interleave(pool, balance_by)
    need = {
        (dim, value): math.ceil(share * target)
        for dim, minimums in min_share.items()
        for value, share in minimums.items()
    }
    selected: list[dict[str, Any]] = []
    used: set[int] = set()
    for i, entry in enumerate(pool):
        if len(selected) >= target or not any(n > 0 for n in need.values()):
            break
        helps = [key for key, n in need.items() if n > 0 and _value(entry, key[0]) == key[1]]
        if helps:
            selected.append(entry)
            used.add(i)
            for key in helps:
                need[key] -= 1
    for i, entry in enumerate(pool):
        if len(selected) >= target:
            break
        if i not in used:
            selected.append(entry)
    return selected
