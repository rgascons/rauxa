"""LLM roles. The writer suggests new items and fixes rejected ones; the judge gives an opinion
on an item. Neither decides anything: the controller keeps or drops items based on the
automatic checks, the judge's verdict and the run config.

Each role has a prompt per item kind (see ``rauxa.formats``): a ``pair`` (an error and its
correction), a ``preference`` (two acceptable sentences, one preferred in a formal register) and
a ``control`` (one correct sentence that uses a form often mistaken for an error). Pair prompts
for tasks without kinds are the original ones, word for word, so earlier runs still resume.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from random import Random
from typing import Any, Optional

from ..formats import KIND_FORMATS, render
from .llm import ChatBackend, LLMError, agent_request

# ---------------------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------------------

_WRITER_SYSTEM = """\
You are a Catalan linguist writing test items for a benchmark of normative Catalan \
(standard of the Institut d'Estudis Catalans, central variety). Each item is a minimal pair: \
a fully correct Catalan sentence and the same sentence containing exactly one specific error. \
Answer with JSON only."""

_PREFERENCE_SYSTEM = """\
You are a Catalan linguist writing test items for a benchmark of normative Catalan \
(standard of the Institut d'Estudis Catalans, central variety). Each item is a preference pair: \
two acceptable Catalan sentences that differ in one point, where one of them is preferred in a \
formal register. Neither sentence contains an error. Answer with JSON only."""

_CONTROL_SYSTEM = """\
You are a Catalan linguist writing test items for a benchmark of normative Catalan \
(standard of the Institut d'Estudis Catalans, central variety). Each item is a control: one \
fully correct Catalan sentence that uses a form often mistaken for an error, in a way that is \
correct. A model that changes the sentence is overcorrecting. Answer with JSON only."""

_SYSTEMS = {"pair": _WRITER_SYSTEM, "preference": _PREFERENCE_SYSTEM, "control": _CONTROL_SYSTEM}

# {scope} and {context_rule} are empty, or full lines ending in a newline: with both empty the
# prompt is exactly the original one.
_WRITE_USER = """\
Error to test: {error}
Normative correction: {correction}
{scope}
Examples of good pairs:
{examples}

Write {n} new pairs. Requirements:
- "good" is a natural, fully normative sentence; "bad" is identical except for the error above.
- "bad" must contain no other error of any kind.
- Vary topic, register, person, tense, sentence length and structure; avoid the example vocabulary.
- One sentence per item, starting with a capital letter and ending with a period.
- "alternatives": other fully correct ways to fix "bad" with a minimal change (may be empty).
{context_rule}
Return: {{"items": [{{"good": "...", "bad": "...", "alternatives": ["..."]}}]}}"""

_WRITE_PREFERENCE_USER = """\
Point to test: {error}
Preferred in a formal register: {correction}
{scope}
Examples of good items:
{examples}

Write {n} new items. Requirements:
- "good" is a natural, fully normative sentence that uses the preferred form.
- "bad" is identical except that it uses the other form: acceptable, not an error, but less \
suitable for a formal register. Neither sentence may contain any error.
- Use general contexts where the preference applies, not the specialised ones where the other \
form is the usual term.
- Vary topic, person, tense, sentence length and structure; avoid the example vocabulary.
- One sentence per item, starting with a capital letter and ending with a period.
- "alternatives": other ways of writing "good" with the preferred form and a minimal change \
(may be empty).
{context_rule}
Return: {{"items": [{{"good": "...", "bad": "...", "alternatives": ["..."]}}]}}"""

_WRITE_CONTROL_USER = """\
The incorrect use, which these sentences must avoid: {error}
Where the form is correct: {scope}

Examples of good items:
{examples}

Write {n} new items. Requirements:
- Each "sentence" is natural, fully normative Catalan that contains one of the correct uses \
listed above, so a careful reviewer would leave it unchanged.
- It must not use the form in the incorrect way above, and must contain no other error.
- Vary topic, person, tense, sentence length and structure; avoid the example vocabulary.
- One sentence per item, starting with a capital letter and ending with a period.
{context_rule}
Return: {{"items": [{{"sentence": "..."}}]}}"""

_WRITE_TEMPLATES = {"pair": _WRITE_USER, "preference": _WRITE_PREFERENCE_USER, "control": _WRITE_CONTROL_USER}

_CONTEXT_RULE = (
    '- "context" (optional): one short Catalan sentence shown before the item, only when it is '
    "needed to fix the intended meaning. Leave it out otherwise.\n"
)

_FIX_USER = """\
This test item for the error «{error}» (correction: {correction}) did not pass our checks:

{item}

What the checks found: {problems}

Fix the item so it meets the original requirements (minimal pair, only that error in \
"bad", fully normative "good"). Return: {{"good": "...", "bad": "...", "alternatives": ["..."]}}"""

_FIX_PREFERENCE_USER = """\
This test item for the formal-register preference «{error}» (preferred: {correction}) did not \
pass our checks:

{item}

What the checks found: {problems}

Fix the item so it meets the original requirements (both sentences free of errors, "good" uses \
the preferred form, "bad" differs from it only in using the other form). \
Return: {{"good": "...", "bad": "...", "alternatives": ["..."]}}"""

_FIX_CONTROL_USER = """\
This control item did not pass our checks. It must be a fully correct sentence that uses the \
form in a correct way.
The incorrect use, which it must avoid: {error}
Where the form is correct: {scope}

{item}

What the checks found: {problems}

Fix the sentence so it meets these requirements. Return: {{"sentence": "..."}}"""

_FIX_TEMPLATES = {"pair": _FIX_USER, "preference": _FIX_PREFERENCE_USER, "control": _FIX_CONTROL_USER}


@dataclass
class WrittenPair:
    """An LLM-written item. ``bad`` is ``None`` for a control."""

    good: str
    bad: Optional[str]
    alternatives: list[str] = field(default_factory=list)
    kind: str = "pair"
    context: Optional[str] = None

    def to_json(self) -> str:
        if self.kind == "control":
            item: dict[str, Any] = {"sentence": self.good}
        else:
            item = {"good": self.good, "bad": self.bad, "alternatives": self.alternatives}
        if self.context:
            item["context"] = self.context
        return json.dumps(item, ensure_ascii=False)


def _pair(raw: dict[str, Any], kind: str = "pair") -> WrittenPair:
    context = str(raw.get("context") or "").strip() or None
    if kind == "control":
        sentence = str(raw.get("sentence", "")).strip()
        if not sentence:
            raise LLMError("writer returned a control without a sentence")
        return WrittenPair(sentence, None, [], kind, context)
    good, bad = str(raw.get("good", "")).strip(), str(raw.get("bad", "")).strip()
    alternatives = [str(a).strip() for a in raw.get("alternatives", []) or [] if str(a).strip()]
    alternatives = [a for a in dict.fromkeys(alternatives) if a not in (good, bad)]
    if not good or not bad:
        raise LLMError("writer returned an item without good/bad")
    return WrittenPair(good, bad, alternatives, kind, context)


def _kind_aware(info: dict[str, Any]) -> bool:
    """Whether the task declares item kinds; if not, prompts are the original pair prompts."""
    return "kinds" in info


def _correct_uses(info: dict[str, Any]) -> str:
    """Where the suspect form is correct, for control prompts."""
    return info.get("correct_uses") or info.get("scope", "")


def _scope_line(info: dict[str, Any], kind: str) -> str:
    if kind == "control":
        return _correct_uses(info)
    return f"Where it applies: {info['scope']}\n" if info.get("scope") else ""


def _example_text(examples: list[tuple[str, Optional[str]]], kind: str) -> str:
    if kind == "control":
        return "\n".join(f"- sentence: {g}" for g, _ in examples)
    if kind == "preference":
        return "\n".join(f"- good (preferred): {g}\n  bad (acceptable, less formal): {b}" for g, b in examples)
    return "\n".join(f"- good: {g}\n  bad: {b}" for g, b in examples)


class Writer:
    def __init__(self, backend: ChatBackend):
        self.backend = backend

    def write(
        self, info: dict[str, Any], examples: list[tuple[str, Optional[str]]], n: int, *,
        kind: str = "pair", key: str = "",
    ) -> list[WrittenPair]:
        user = _WRITE_TEMPLATES[kind].format(
            error=info["error"], correction=info["correction"], examples=_example_text(examples, kind), n=n,
            scope=_scope_line(info, kind), context_rule=_CONTEXT_RULE if _kind_aware(info) else "",
        )
        response = agent_request(
            self.backend, [{"role": "system", "content": _SYSTEMS[kind]}, {"role": "user", "content": user}],
            kind="write", key=key, count=n,
        )
        items = response.get("items")
        if not isinstance(items, list) or not items:
            # e.g. a reply cut off mid-list, where the parser can only recover one inner pair.
            # Raising makes the batch a failed call instead of a silent empty success.
            raise LLMError("writer reply has no nonempty 'items' list (truncated or malformed)")
        pairs = []
        for raw in items:
            if not isinstance(raw, dict):
                continue
            try:
                pairs.append(_pair(raw, kind))
            except LLMError:
                continue
        return pairs

    def fix(self, info: dict[str, Any], pair: WrittenPair, problems: list[str], *, key: str = "") -> WrittenPair:
        user = _FIX_TEMPLATES[pair.kind].format(
            error=info["error"], correction=info["correction"], scope=_correct_uses(info),
            item=pair.to_json(), problems="; ".join(problems),
        )
        return _pair(
            agent_request(self.backend, [{"role": "system", "content": _SYSTEMS[pair.kind]}, {"role": "user", "content": user}],
                          kind="fix", key=key),
            pair.kind,
        )


def llm_entry(
    pair: WrittenPair, *, task: str, phenomenon: str, norm: str, formats: list[str], rng: Random,
    languagetool_rules: list[str], writer_model: str, revision_of: str | None = None,
    info: Optional[dict[str, Any]] = None, allow_unchanged: bool = False,
) -> dict[str, Any]:
    """Turn an LLM-written item into an item with the same shape as procedural items.

    ``formats`` is narrowed to the formats the item's kind allows; the caller makes sure one is left.
    """
    info = info or {}
    fmt = rng.choice([f for f in formats if f in KIND_FORMATS[pair.kind]])
    question, answer, extra = render(
        fmt, pair.good, pair.bad, rng, kind=pair.kind, context=pair.context, allow_unchanged=allow_unchanged,
    )
    meta: dict[str, Any] = {
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
    }
    if _kind_aware(info):
        meta.update({"kind": pair.kind, **{k: info[k] for k in ("category", "assessment", "evidence_level") if k in info}})
        if info.get("languagetool_tolerated"):
            meta["languagetool_tolerated"] = list(info["languagetool_tolerated"])
    if pair.context:
        meta["context"] = pair.context
    return {"question": question, "answer": answer, "metadata": meta}


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

_JUDGE_PREFERENCE_USER = """\
The item tests a formal-register preference, not an error: {error}
Preferred form: {correction}

Sentence presented as preferred: «{good}»
Other sentences presented as preferred: {variants}
Sentence presented as acceptable but less suitable for a formal register: «{bad}»

Answer with these fields:
- "good_is_correct": is the preferred sentence fully normative?
- "variants_are_correct": are all the other preferred sentences fully normative? (true if none)
- "bad_is_acceptable": is the less formal sentence itself free of normative errors?
- "good_is_preferred": in a formal register, is the preferred sentence the better choice, for \
the reason described?
- "only_stated_difference": do the two sentences differ only in the point described?
- "natural": would a native speaker find the preferred sentence natural?
- "same_meaning": do the two sentences mean the same?
- "comment": one short sentence explaining any answer that makes the item bad.

Return: {{"good_is_correct": true, "variants_are_correct": true, "bad_is_acceptable": true, \
"good_is_preferred": true, "only_stated_difference": true, "natural": true, "same_meaning": true, \
"comment": ""}}"""

_JUDGE_CONTROL_USER = """\
This is a control item: a sentence that should be correct as it is. It checks that a reader \
does not wrongly "correct" a form often mistaken for this error: {error}
Where the form is correct: {scope}

Sentence: «{good}»

Answer with these fields:
- "sentence_is_correct": is the sentence fully normative as it is?
- "avoids_the_error": is the sentence free of the incorrect use described above?
- "matches_a_correct_use": does the sentence contain one of the correct uses listed under "Where \
the form is correct"? It does not need to contain the incorrect construction itself: a control is \
useful because it is close to the error without being it.
- "natural": would a native speaker find the sentence natural?
- "comment": one short sentence explaining any answer that makes the item bad.

Return: {{"sentence_is_correct": true, "avoids_the_error": true, "matches_a_correct_use": true, \
"natural": true, "comment": ""}}"""

_JUDGE_TEMPLATES = {"pair": _JUDGE_USER, "preference": _JUDGE_PREFERENCE_USER, "control": _JUDGE_CONTROL_USER}

_JUDGE_CONTEXT = "The item is shown after this context, which fixes the intended meaning: «{context}»\n\n"

# Questions the judge must answer "true", per kind; for pairs, "bad_has_other_errors" must be "false".
JUDGE_QUESTIONS = ("good_is_correct", "variants_are_correct", "bad_has_stated_error", "natural", "same_meaning")
KIND_JUDGE_QUESTIONS: dict[str, tuple[str, ...]] = {
    "pair": JUDGE_QUESTIONS,
    "preference": ("good_is_correct", "variants_are_correct", "bad_is_acceptable", "good_is_preferred",
                   "only_stated_difference", "natural", "same_meaning"),
    "control": ("sentence_is_correct", "avoids_the_error", "matches_a_correct_use", "natural"),
}
# Control questions before they were reworded; verdicts saved with them still count.
LEGACY_CONTROL_QUESTIONS = ("sentence_is_correct", "uses_correct_sense", "contains_the_form", "natural")
_NEGATIVE_QUESTIONS = {"pair": ("bad_has_other_errors",), "preference": (), "control": ()}

# Which questions concern the correct sentence(s) and which the other one, for calibration
# against human labels ("good_ok" / "bad_ok").
GOOD_QUESTIONS = {
    "pair": {"good_is_correct", "variants_are_correct", "natural"},
    "preference": {"good_is_correct", "variants_are_correct", "natural"},
    "control": {"sentence_is_correct", "avoids_the_error", "matches_a_correct_use", "natural",
                *LEGACY_CONTROL_QUESTIONS},
}
BAD_QUESTIONS = {
    "pair": {"bad_has_stated_error", "bad_has_other_errors"},
    "preference": {"bad_is_acceptable", "good_is_preferred", "only_stated_difference"},
    "control": set(),
}


def verdict_keys(kind: str) -> tuple[str, ...]:
    return (*KIND_JUDGE_QUESTIONS[kind], *_NEGATIVE_QUESTIONS[kind])


def failed_questions(verdict: dict[str, Any]) -> list[str]:
    """Questions with an unacceptable answer (missing or non-boolean answers count as failed).

    The kind is read from the verdict; verdicts without one are pair verdicts.
    """
    kind = verdict.get("kind", "pair")
    questions = KIND_JUDGE_QUESTIONS[kind]
    if kind == "control" and "contains_the_form" in verdict:
        questions = LEGACY_CONTROL_QUESTIONS
    failed = [k for k in questions if verdict.get(k) is not True]
    failed += [k for k in _NEGATIVE_QUESTIONS[kind] if verdict.get(k) is not False]
    return failed


def verdict_passes(verdict: dict[str, Any]) -> bool:
    return not failed_questions(verdict)


class Judge:
    def __init__(self, backend: ChatBackend):
        self.backend = backend

    def judge(self, entry: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
        meta = entry["metadata"]
        kind = meta.get("kind", "pair")
        others = [v for v in meta["correct_variants"] if v != meta["sentence_good"]]
        user = _JUDGE_TEMPLATES[kind].format(
            error=info["error"],
            correction=info["correction"],
            scope=_correct_uses(info),
            good=meta["sentence_good"],
            variants=", ".join(f"«{v}»" for v in others) or "(none)",
            bad=meta["sentence_bad"],
        )
        if meta.get("context"):
            user = _JUDGE_CONTEXT.format(context=meta["context"]) + user
        verdict = agent_request(
            self.backend, [{"role": "system", "content": _JUDGE_SYSTEM}, {"role": "user", "content": user}],
            kind="judge", key=f"judge:{meta.get('item_id', '')}",
        )
        cleaned: dict[str, Any] = {k: verdict.get(k) for k in verdict_keys(kind)}
        cleaned["comment"] = str(verdict.get("comment", ""))[:500]
        if kind != "pair":
            cleaned["kind"] = kind
        cleaned["passed"] = verdict_passes(cleaned)
        return cleaned
