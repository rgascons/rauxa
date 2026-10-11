"""Deterministic checks on Catalan sentences, used by the pipeline's automatic checks."""

from __future__ import annotations

import re

# Errors that templating (or an LLM) can introduce: unelided "de"/article/pronoun, missing
# contraction, spacing problems.
ORTHOGRAPHY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("de_not_elided", re.compile(r"\bde h?[aeiouàèéíòóú]", re.IGNORECASE)),
    ("missing_contraction", re.compile(r"\b(de|a|per) (el|els)\b", re.IGNORECASE)),
    ("article_not_elided", re.compile(r"\b(el|la) h?[aeoàèéòó]", re.IGNORECASE)),
    ("pronoun_not_elided", re.compile(r"\b(em|et|es) h?[aeiouàèéíòóú]", re.IGNORECASE)),
    ("double_space", re.compile(r"\s{2,}")),
    ("space_after_apostrophe", re.compile(r"\w' ")),
)

_TOKEN = re.compile(r"\w+'?|[^\w\s]", re.UNICODE)


def orthography_issues(text: str) -> list[str]:
    """Names of orthographic problems found in ``text`` (empty if none)."""
    issues = [name for name, pattern in ORTHOGRAPHY_PATTERNS if pattern.search(text)]
    if not text[:1].isupper():
        issues.append("not_capitalised")
    if not text.rstrip().endswith((".", "?", "!")):
        issues.append("no_final_punctuation")
    return issues


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def edit_span(a: str, b: str) -> tuple[int, int, int]:
    """Size of the single contiguous edit turning ``a`` into ``b``.

    Returns ``(span_a, span_b, shared)``: tokens that differ in ``a`` and in ``b`` after
    removing the longest common prefix and suffix, and the number of shared tokens.
    """
    ta, tb = tokenize(a), tokenize(b)
    prefix = 0
    while prefix < min(len(ta), len(tb)) and ta[prefix] == tb[prefix]:
        prefix += 1
    suffix = 0
    while (
        suffix < min(len(ta), len(tb)) - prefix
        and ta[len(ta) - 1 - suffix] == tb[len(tb) - 1 - suffix]
    ):
        suffix += 1
    return len(ta) - prefix - suffix, len(tb) - prefix - suffix, prefix + suffix
