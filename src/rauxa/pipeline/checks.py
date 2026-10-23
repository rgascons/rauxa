"""Automatic checks run on every candidate pair. Each returns a ``CheckResult``.

No check calls an LLM. A pair passes only if every enabled check passes; all results are
stored so that rejections can be inspected later.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Protocol

from ..catalan.checks import edit_span, orthography_issues
from ..qc.languagetool import LanguageToolClient, summarize_match
from .config import ChecksConfig


@dataclass
class CheckResult:
    check: str
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Check(Protocol):
    name: str

    def run(self, entry: dict[str, Any]) -> CheckResult: ...


def _sentences(entry: dict[str, Any]) -> tuple[str, Optional[str], list[str]]:
    """Correct sentence, incorrect (or dispreferred) sentence, accepted variants.

    The second element is ``None`` for controls, which have only a valid sentence.
    """
    meta = entry["metadata"]
    return meta["sentence_good"], meta.get("sentence_bad"), meta["correct_variants"]


class SchemaCheck:
    name = "schema"

    def run(self, entry: dict[str, Any]) -> CheckResult:
        problems = []
        if not entry.get("question") or entry.get("answer") in (None, ""):
            problems.append("missing question or answer")
        meta = entry.get("metadata", {})
        kind = meta.get("kind", "pair")
        required = ("sentence_good", "correct_variants", "phenomenon", "format")
        if kind != "control":
            required = ("sentence_good", "sentence_bad", "correct_variants", "phenomenon", "format")
        for key in required:
            if not meta.get(key):
                problems.append(f"missing metadata.{key}")
        if kind not in ("pair", "preference", "control"):
            problems.append(f"unknown kind {kind!r}")
        elif kind == "control" and meta.get("sentence_bad") is not None:
            problems.append("a control has no incorrect sentence")
        if not problems:
            good, bad, variants = _sentences(entry)
            if good not in variants:
                problems.append("sentence_good not in correct_variants")
            if bad is not None and bad in variants:
                problems.append("sentence_bad is listed as a correct variant")
            if len(set(variants)) != len(variants):
                problems.append("duplicate correct variants")
        return CheckResult(self.name, not problems, {"problems": problems})


class OrthographyCheck:
    name = "orthography"

    def run(self, entry: dict[str, Any]) -> CheckResult:
        good, bad, variants = _sentences(entry)
        sentences = variants if bad is None else [bad, *variants]
        issues = {s: orthography_issues(s) for s in sentences}
        issues = {s: i for s, i in issues.items() if i}
        return CheckResult(self.name, not issues, {"issues": issues})


class MarkerCheck:
    """The phenomenon's marker must match the incorrect sentence and no correct variant.

    Skipped for controls: a control can legitimately contain the suspect form (an accepted loan,
    a valid sense), which is exactly what it tests.
    """

    name = "marker"

    def __init__(self, markers: dict[str, str]):
        self.markers = {p: re.compile(m, re.IGNORECASE) for p, m in markers.items()}

    def run(self, entry: dict[str, Any]) -> CheckResult:
        good, bad, variants = _sentences(entry)
        if bad is None:
            return CheckResult(self.name, True, {"skipped": "control"})
        marker = self.markers.get(entry["metadata"]["phenomenon"])
        if marker is None:
            return CheckResult(self.name, True, {"skipped": "no marker for phenomenon"})
        in_bad = bool(marker.search(bad))
        in_good = [v for v in variants if marker.search(v)]
        return CheckResult(self.name, in_bad and not in_good, {"marker_in_bad": in_bad, "marker_in_good": in_good})


class MinimalPairCheck:
    """The incorrect sentence must differ from each correct variant by a single small edit.

    Checking every variant, not only ``sentence_good``, keeps far-off rewrites out of the
    accepted gec answers. They are listed under ``variants_too_far`` when the check fails.
    """

    name = "minimal_pair"

    def __init__(self, max_edit_tokens: int):
        self.max_edit_tokens = max_edit_tokens

    def run(self, entry: dict[str, Any]) -> CheckResult:
        good, bad, variants = _sentences(entry)
        if bad is None:
            return CheckResult(self.name, True, {"skipped": "control"})
        span_good, span_bad, shared = edit_span(good, bad)
        ok = self._minimal(span_good, span_bad, shared)
        details: dict[str, Any] = {"span_good": span_good, "span_bad": span_bad, "shared": shared}
        too_far = [v for v in variants if v != good and not self._minimal(*edit_span(v, bad))]
        if too_far:
            details["variants_too_far"] = too_far
        return CheckResult(self.name, ok and not too_far, details)

    def _minimal(self, span_a: int, span_b: int, shared: int) -> bool:
        return 0 < max(span_a, span_b) <= self.max_edit_tokens and shared > 0


class LanguageToolCheck:
    """Correct variants must get no LanguageTool match; the incorrect sentence should get one.

    ``expected_rule_hit`` records whether LanguageTool flagged the incorrect sentence with one
    of the phenomenon's expected rules. A miss means LanguageTool doesn't cover that case; it
    is reported in the manifest and only rejects the pair if ``require_bad_flagged`` is set.

    ``metadata["languagetool_tolerated"]`` lists rule ids that may fire on a correct variant
    without rejecting it, e.g. a rule that flags an accepted loan in a control. Tolerated
    matches are recorded, never hidden. Controls have no incorrect sentence to flag.
    """

    name = "languagetool"

    def __init__(self, client: LanguageToolClient, require_bad_flagged: bool):
        self.client = client
        self.require_bad_flagged = require_bad_flagged

    def run(self, entry: dict[str, Any]) -> CheckResult:
        good, bad, variants = _sentences(entry)
        tolerated_rules = set(entry["metadata"].get("languagetool_tolerated", []))
        variant_matches, tolerated = {}, {}
        for v in variants:
            matches = [summarize_match(m) for m in self.client.check(v)]
            blocking = [m for m in matches if m["rule_id"] not in tolerated_rules]
            if blocking:
                variant_matches[v] = blocking
            if len(blocking) != len(matches):
                tolerated[v] = [m for m in matches if m["rule_id"] in tolerated_rules]
        if bad is None:
            details = {"variant_matches": variant_matches, "kind": "control"}
            if tolerated:
                details["tolerated_matches"] = tolerated
            return CheckResult(self.name, not variant_matches, details)
        bad_matches = [summarize_match(m) for m in self.client.check(bad)]
        expected = set(entry["metadata"].get("languagetool_rules", []))
        hit = bool(expected & {m["rule_id"] for m in bad_matches})
        passed = not variant_matches and (bool(bad_matches) or not self.require_bad_flagged)
        details = {
            "variant_matches": variant_matches,
            "bad_rules": sorted({m["rule_id"] for m in bad_matches}),
            "bad_flagged": bool(bad_matches),
            "expected_rule_hit": hit,
        }
        if tolerated:
            details["tolerated_matches"] = tolerated
        return CheckResult(self.name, passed, details)


def build_checks(config: ChecksConfig, markers: dict[str, str], lt_client: Optional[LanguageToolClient]) -> list[Check]:
    checks: list[Check] = [SchemaCheck()]
    if config.orthography:
        checks.append(OrthographyCheck())
    if config.marker:
        checks.append(MarkerCheck(markers))
    if config.minimal_pair:
        checks.append(MinimalPairCheck(config.max_edit_tokens))
    if config.languagetool.enabled:
        if lt_client is None:
            raise ValueError("LanguageTool check enabled but no client given")
        checks.append(LanguageToolCheck(lt_client, config.languagetool.require_bad_flagged))
    return checks


def run_checks(checks: list[Check], entry: dict[str, Any]) -> list[CheckResult]:
    """Run checks in order, stopping at the first failure (cheap checks come first)."""
    results = []
    for check in checks:
        result = check.run(entry)
        results.append(result)
        if not result.passed:
            break
    return results
