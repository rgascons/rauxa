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


def _sentences(entry: dict[str, Any]) -> tuple[str, str, list[str]]:
    meta = entry["metadata"]
    return meta["sentence_good"], meta["sentence_bad"], meta["correct_variants"]


class SchemaCheck:
    name = "schema"

    def run(self, entry: dict[str, Any]) -> CheckResult:
        problems = []
        if not entry.get("question") or entry.get("answer") in (None, ""):
            problems.append("missing question or answer")
        meta = entry.get("metadata", {})
        for key in ("sentence_good", "sentence_bad", "correct_variants", "phenomenon", "format"):
            if not meta.get(key):
                problems.append(f"missing metadata.{key}")
        if not problems:
            good, bad, variants = _sentences(entry)
            if good not in variants:
                problems.append("sentence_good not in correct_variants")
            if bad in variants:
                problems.append("sentence_bad is listed as a correct variant")
            if len(set(variants)) != len(variants):
                problems.append("duplicate correct variants")
        return CheckResult(self.name, not problems, {"problems": problems})


class OrthographyCheck:
    name = "orthography"

    def run(self, entry: dict[str, Any]) -> CheckResult:
        good, bad, variants = _sentences(entry)
        issues = {s: orthography_issues(s) for s in [bad, *variants]}
        issues = {s: i for s, i in issues.items() if i}
        return CheckResult(self.name, not issues, {"issues": issues})


class MarkerCheck:
    """The phenomenon's marker must match the incorrect sentence and no correct variant."""

    name = "marker"

    def __init__(self, markers: dict[str, str]):
        self.markers = {p: re.compile(m, re.IGNORECASE) for p, m in markers.items()}

    def run(self, entry: dict[str, Any]) -> CheckResult:
        good, bad, variants = _sentences(entry)
        marker = self.markers.get(entry["metadata"]["phenomenon"])
        if marker is None:
            return CheckResult(self.name, True, {"skipped": "no marker for phenomenon"})
        in_bad = bool(marker.search(bad))
        in_good = [v for v in variants if marker.search(v)]
        return CheckResult(self.name, in_bad and not in_good, {"marker_in_bad": in_bad, "marker_in_good": in_good})


class MinimalPairCheck:
    """The incorrect sentence must differ from the correct one by a single small edit."""

    name = "minimal_pair"

    def __init__(self, max_edit_tokens: int):
        self.max_edit_tokens = max_edit_tokens

    def run(self, entry: dict[str, Any]) -> CheckResult:
        good, bad, _ = _sentences(entry)
        span_good, span_bad, shared = edit_span(good, bad)
        ok = 0 < max(span_good, span_bad) <= self.max_edit_tokens and shared > 0
        return CheckResult(self.name, ok, {"span_good": span_good, "span_bad": span_bad, "shared": shared})


class LanguageToolCheck:
    """Correct variants must get no LanguageTool match; the incorrect sentence should get one.

    ``expected_rule_hit`` records whether LanguageTool flagged the incorrect sentence with one
    of the phenomenon's expected rules. A miss means LanguageTool doesn't cover that case; it
    is reported in the manifest and only rejects the pair if ``require_bad_flagged`` is set.
    """

    name = "languagetool"

    def __init__(self, client: LanguageToolClient, require_bad_flagged: bool):
        self.client = client
        self.require_bad_flagged = require_bad_flagged

    def run(self, entry: dict[str, Any]) -> CheckResult:
        good, bad, variants = _sentences(entry)
        variant_matches = {v: [summarize_match(m) for m in self.client.check(v)] for v in variants}
        variant_matches = {v: m for v, m in variant_matches.items() if m}
        bad_matches = [summarize_match(m) for m in self.client.check(bad)]
        expected = set(entry["metadata"].get("languagetool_rules", []))
        hit = bool(expected & {m["rule_id"] for m in bad_matches})
        passed = not variant_matches and (bool(bad_matches) or not self.require_bad_flagged)
        return CheckResult(
            self.name,
            passed,
            {
                "variant_matches": variant_matches,
                "bad_rules": sorted({m["rule_id"] for m in bad_matches}),
                "bad_flagged": bool(bad_matches),
                "expected_rule_hit": hit,
            },
        )


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
