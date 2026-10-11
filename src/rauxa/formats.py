"""Render a (correct, incorrect) sentence pair as a prompt in one of several formats, and score answers.

Every item keeps ``sentence_good`` / ``sentence_bad`` in its metadata whatever its format, so
any dataset can also be exported as BLiMP-style minimal pairs for log-likelihood scoring.

Formats:
- ``mcq``: pick the correct sentence among lettered options; answer is the letter.
- ``judgement``: is this sentence correct? answer is ``sí`` or ``no``.
- ``gec``: rewrite the incorrect sentence; answer is the corrected sentence (any of
  ``metadata["correct_variants"]`` is accepted).
"""

from __future__ import annotations

import re
import string
import unicodedata
from random import Random
from typing import Any, Optional

FORMATS = ("mcq", "judgement", "gec")
LETTERS = "ABCD"

_MCQ_PROMPT = (
    "Quina d'aquestes frases és correcta segons la normativa del català?\n\n"
    "{options}\n\n"
    "Respon només amb la lletra de l'opció correcta."
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


def render(fmt: str, good: str, bad: str, rng: Random) -> tuple[str, str, dict[str, Any]]:
    """Return ``(question, answer, extra_metadata)`` for one pair."""
    if fmt == "mcq":
        options = [good, bad]
        rng.shuffle(options)
        target = options.index(good)
        body = "\n".join(f"{LETTERS[i]}) {opt}" for i, opt in enumerate(options))
        return _MCQ_PROMPT.format(options=body), LETTERS[target], {"options": options, "target_index": target}
    if fmt == "judgement":
        shown_good = rng.random() < 0.5
        shown = good if shown_good else bad
        return _JUDGEMENT_PROMPT.format(sentence=shown), "sí" if shown_good else "no", {"shown": shown}
    if fmt == "gec":
        return _GEC_PROMPT.format(sentence=bad), good, {}
    raise ValueError(f"unknown format '{fmt}'; expected one of {FORMATS}")


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
    """Verifier shared by all tasks built on sentence pairs; returns 0.0 or 1.0."""
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
