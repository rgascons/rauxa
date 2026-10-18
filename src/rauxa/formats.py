"""Render an item as a prompt in one of several formats, and score answers.

Item kinds (``metadata["kind"]``, absent means ``"pair"``):
- ``pair``: a normative minimal pair, ``sentence_good`` / ``sentence_bad``.
- ``preference``: the same shape, but ``sentence_bad`` is acceptable and ``sentence_good`` is
  preferred in a formal register. Prompts never call the dispreferred sentence incorrect.
- ``control``: one valid sentence (``sentence_bad`` is ``None``), e.g. a legitimate use of a
  suspect word. Measures false positives and overcorrection.

Pairs and preferences keep ``sentence_good`` / ``sentence_bad`` in their metadata whatever their
format, so they can also be exported as BLiMP-style minimal pairs for log-likelihood scoring.

Formats:
- ``mcq``: pick the correct (or, for preferences, the more formal) sentence; answer is the letter.
- ``judgement``: is this sentence correct? answer is ``sí`` or ``no``.
- ``gec``: rewrite the sentence if needed; answer is the corrected sentence (any of
  ``metadata["correct_variants"]`` is accepted). For a control, the answer is the sentence itself.

An optional ``context`` is shown before the item, to fix the intended meaning.
"""

from __future__ import annotations

import re
import string
import unicodedata
from random import Random
from typing import Any, Optional

FORMATS = ("mcq", "judgement", "gec")
KINDS = ("pair", "preference", "control")
# Formats each kind can be rendered in: a preference has no "incorrect" sentence to judge or
# correct, and a control has no second sentence to choose between.
KIND_FORMATS = {"pair": FORMATS, "preference": ("mcq",), "control": ("judgement", "gec")}
LETTERS = "ABCD"

_MCQ_PROMPT = (
    "Quina d'aquestes frases és correcta segons la normativa del català?\n\n"
    "{options}\n\n"
    "Respon només amb la lletra de l'opció correcta."
)
_PREFERENCE_PROMPT = (
    "Quina d'aquestes frases és més adequada en un registre formal?\n\n"
    "{options}\n\n"
    "Respon només amb la lletra de l'opció més adequada."
)
_JUDGEMENT_PROMPT = (
    "La frase següent és correcta segons la normativa del català?\n\n"
    "«{sentence}»\n\n"
    "Respon només «sí» o «no»."
)
_GEC_PROMPT = (
    "Corregeix la frase següent perquè sigui correcta segons la normativa del català. "
    "Respon només amb la frase corregida.\n\n"
    "«{sentence}»"
)
# Used for every gec item of a task that has controls, so the wording doesn't reveal which
# sentences are already correct.
_GEC_UNCHANGED_PROMPT = (
    "Corregeix la frase següent perquè sigui correcta segons la normativa del català. "
    "Si ja és correcta, torna-la igual. Respon només amb la frase.\n\n"
    "«{sentence}»"
)
_CONTEXT = "Context: «{context}»\n\n"


def render(
    fmt: str, good: str, bad: Optional[str], rng: Random, *,
    kind: str = "pair", context: Optional[str] = None, allow_unchanged: bool = False,
) -> tuple[str, str, dict[str, Any]]:
    """Return ``(question, answer, extra_metadata)`` for one item.

    ``allow_unchanged`` selects the gec wording that admits an already-correct sentence; tasks
    with controls should set it for all their gec items.
    """
    if kind not in KINDS:
        raise ValueError(f"unknown kind '{kind}'; expected one of {KINDS}")
    if fmt not in KIND_FORMATS[kind]:
        raise ValueError(f"format '{fmt}' is not available for kind '{kind}'; use one of {KIND_FORMATS[kind]}")
    if (bad is None) != (kind == "control"):
        raise ValueError("only controls have no incorrect sentence")
    prefix = _CONTEXT.format(context=context) if context else ""
    if fmt == "mcq":
        options = [good, bad]
        rng.shuffle(options)
        target = options.index(good)
        body = "\n".join(f"{LETTERS[i]}) {opt}" for i, opt in enumerate(options))
        template = _PREFERENCE_PROMPT if kind == "preference" else _MCQ_PROMPT
        return prefix + template.format(options=body), LETTERS[target], {"options": options, "target_index": target}
    if fmt == "judgement":
        shown_good = kind == "control" or rng.random() < 0.5
        shown = good if shown_good else bad
        return prefix + _JUDGEMENT_PROMPT.format(sentence=shown), "sí" if shown_good else "no", {"shown": shown}
    template = _GEC_UNCHANGED_PROMPT if allow_unchanged else _GEC_PROMPT
    return prefix + template.format(sentence=good if kind == "control" else bad), good, {}


def normalize_sentence(text: str) -> str:
    """Normalise for exact-match comparison: quotes, apostrophes, case, spacing, final punctuation."""
    text = unicodedata.normalize("NFC", text).strip()
    text = text.replace("’", "'").replace("`", "'")
    text = text.strip("«»\"“”'").strip()
    text = re.sub(r"\s+", " ", text)
    text = text.rstrip(string.punctuation + " ")
    return text.lower()


def _extract_letter(answer: str, n_options: int) -> Optional[str]:
    match = re.search(rf"\b([{LETTERS[:n_options]}])\b", answer.strip())
    return match.group(1) if match else None


def _extract_yes_no(answer: str) -> Optional[str]:
    words = re.findall(r"\w+", answer.lower())
    if not words:
        return None
    if words[0] in ("sí", "si"):
        return "sí"
    if words[0] == "no":
        return "no"
    return None


def score(answer: Optional[str], entry: dict[str, Any]) -> float:
    """Verifier shared by all sentence-based tasks; returns 0.0 or 1.0."""
    if answer is None:
        return 0.0
    meta = entry["metadata"]
    fmt = meta["format"]
    if fmt == "mcq":
        return 1.0 if _extract_letter(answer, len(meta["options"])) == entry["answer"] else 0.0
    if fmt == "judgement":
        return 1.0 if _extract_yes_no(answer) == entry["answer"] else 0.0
    if fmt == "gec":
        accepted = {normalize_sentence(s) for s in meta["correct_variants"]}
        return 1.0 if normalize_sentence(answer) in accepted else 0.0
    raise ValueError(f"unknown format '{fmt}'")
