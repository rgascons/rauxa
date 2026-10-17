"""LLM roles. The writer suggests new sentence pairs and fixes rejected ones; the judge gives
an opinion on a pair. Neither decides anything: the controller keeps or drops pairs based on
the automatic checks, the judge's verdict and the run config.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from random import Random
from typing import Any

from ..formats import render
from .llm import ChatBackend, LLMError, agent_request

# ---------------------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------------------

_WRITER_SYSTEM = """\
You are a Catalan linguist writing test items for a benchmark of normative Catalan \
(standard of the Institut d'Estudis Catalans, central variety). Each item is a minimal pair: \
a fully correct Catalan sentence and the same sentence containing exactly one specific error. \
Answer with JSON only."""

_WRITE_USER = """\
Error to test: {error}
Normative correction: {correction}

Examples of good pairs:
{examples}

Write {n} new pairs. Requirements:
- "good" is a natural, fully normative sentence; "bad" is identical except for the error above.
- "bad" must contain no other error of any kind.
- Vary topic, register, person, tense, sentence length and structure; avoid the example vocabulary.
- One sentence per item, starting with a capital letter and ending with a period.
- "alternatives": other fully correct ways to fix "bad" with a minimal change (may be empty).

Return: {{"items": [{{"good": "...", "bad": "...", "alternatives": ["..."]}}]}}"""

_FIX_USER = """\
This test item for the error «{error}» (correction: {correction}) did not pass our checks:

{item}

What the checks found: {problems}

Fix the item so it meets the original requirements (minimal pair, only that error in \
"bad", fully normative "good"). Return: {{"good": "...", "bad": "...", "alternatives": ["..."]}}"""


@dataclass
class WrittenPair:
    good: str
    bad: str
    alternatives: list[str]


def _pair(raw: dict[str, Any]) -> WrittenPair:
    good, bad = str(raw.get("good", "")).strip(), str(raw.get("bad", "")).strip()
    alternatives = [str(a).strip() for a in raw.get("alternatives", []) or [] if str(a).strip()]
    alternatives = [a for a in dict.fromkeys(alternatives) if a not in (good, bad)]
    if not good or not bad:
        raise LLMError("writer returned an item without good/bad")
    return WrittenPair(good, bad, alternatives)


class Writer:
    def __init__(self, backend: ChatBackend):
        self.backend = backend

    def write(self, info: dict[str, Any], examples: list[tuple[str, str]], n: int, *, key: str = "") -> list[WrittenPair]:
        example_text = "\n".join(f"- good: {g}\n  bad: {b}" for g, b in examples)
        user = _WRITE_USER.format(error=info["error"], correction=info["correction"], examples=example_text, n=n)
        response = agent_request(
            self.backend, [{"role": "system", "content": _WRITER_SYSTEM}, {"role": "user", "content": user}],
            kind="write", key=key, count=n,
        )
        pairs = []
        for raw in response.get("items", []):
            try:
                pairs.append(_pair(raw))
            except LLMError:
                continue
        return pairs

    def fix(self, info: dict[str, Any], pair: WrittenPair, problems: list[str], *, key: str = "") -> WrittenPair:
        item = json.dumps({"good": pair.good, "bad": pair.bad, "alternatives": pair.alternatives}, ensure_ascii=False)
        user = _FIX_USER.format(
            error=info["error"], correction=info["correction"], item=item, problems="; ".join(problems)
        )
        return _pair(
            agent_request(self.backend, [{"role": "system", "content": _WRITER_SYSTEM}, {"role": "user", "content": user}],
                          kind="fix", key=key)
        )


def llm_entry(
    pair: WrittenPair, *, task: str, phenomenon: str, norm: str, formats: list[str], rng: Random,
    languagetool_rules: list[str], writer_model: str, revision_of: str | None = None,
) -> dict[str, Any]:
    """Turn an LLM-written pair into an item with the same shape as procedural items."""
    fmt = rng.choice(formats)
    question, answer, extra = render(fmt, pair.good, pair.bad, rng)
    return {
        "question": question,
        "answer": answer,
        "metadata": {
            "source_dataset": f"llm:{task}",
            "source_index": None,
            "format": fmt,
            "phenomenon": phenomenon,
            "norm": norm,
            "tense": None,
            "sentence_good": pair.good,
            "sentence_bad": pair.bad,
            "correct_variants": [pair.good, *pair.alternatives],
            "languagetool_rules": languagetool_rules,
            "writer_model": writer_model,
            "revision_of": revision_of,
            **extra,
        },
    }


# ---------------------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------------------

_JUDGE_SYSTEM = """\
You are a strict reviewer of normative Catalan (standard of the Institut d'Estudis Catalans). \
You check test items for a benchmark. Judge each sentence on its own merits; do not assume \
the item's labels are right. Answer with JSON only."""

_JUDGE_USER = """\
The item tests this error: {error}
Expected correction: {correction}

Sentence presented as correct: «{good}»
Other sentences presented as correct: {variants}
Sentence presented as containing the error: «{bad}»

Answer with these fields:
- "good_is_correct": is the correct sentence fully normative?
- "variants_are_correct": are all the other correct sentences fully normative? (true if none)
- "bad_has_stated_error": does the incorrect sentence contain exactly the error described?
- "bad_has_other_errors": does it contain any other error?
- "natural": would a native speaker find the correct sentence natural?
- "same_meaning": do the correct and incorrect sentences mean the same apart from the error?
- "comment": one short sentence explaining any answer that makes the item bad.

Return: {{"good_is_correct": true, "variants_are_correct": true, "bad_has_stated_error": true, \
"bad_has_other_errors": false, "natural": true, "same_meaning": true, "comment": ""}}"""

# Questions the judge must answer "true"; "bad_has_other_errors" must be "false".
JUDGE_QUESTIONS = ("good_is_correct", "variants_are_correct", "bad_has_stated_error", "natural", "same_meaning")


def failed_questions(verdict: dict[str, Any]) -> list[str]:
    """Questions with an unacceptable answer (missing or non-boolean answers count as failed)."""
    failed = [k for k in JUDGE_QUESTIONS if verdict.get(k) is not True]
    if verdict.get("bad_has_other_errors") is not False:
        failed.append("bad_has_other_errors")
    return failed


def verdict_passes(verdict: dict[str, Any]) -> bool:
    return not failed_questions(verdict)


class Judge:
    def __init__(self, backend: ChatBackend):
        self.backend = backend

    def judge(self, entry: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
        meta = entry["metadata"]
        others = [v for v in meta["correct_variants"] if v != meta["sentence_good"]]
        user = _JUDGE_USER.format(
            error=info["error"],
            correction=info["correction"],
            good=meta["sentence_good"],
            variants=", ".join(f"«{v}»" for v in others) or "(none)",
            bad=meta["sentence_bad"],
        )
        verdict = agent_request(
            self.backend, [{"role": "system", "content": _JUDGE_SYSTEM}, {"role": "user", "content": user}],
            kind="judge", key=f"judge:{meta.get('item_id', '')}",
        )
        keys = (*JUDGE_QUESTIONS, "bad_has_other_errors")
        cleaned = {k: verdict.get(k) for k in keys}
        cleaned["comment"] = str(verdict.get("comment", ""))[:500]
        cleaned["passed"] = verdict_passes(cleaned)
        return cleaned
