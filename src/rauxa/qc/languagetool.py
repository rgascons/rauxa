"""Quality control with a LanguageTool HTTP server (local or remote).

A generated pair passes when the correct sentence gets no LanguageTool matches and the
incorrect one gets at least one. Pairs that fail either check are flagged: a match on the
"good" sentence usually means a template or inflection bug, and no match on the "bad" one
means LanguageTool does not cover that error (worth knowing, but not necessarily a bug).

Start a local server with::

    java -cp languagetool-server.jar org.languagetool.server.HTTPServer --port 8081
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any

DEFAULT_URL = "http://localhost:8081"


@dataclass
class LanguageToolClient:
    url: str = DEFAULT_URL
    language: str = "ca-ES"
    # Rule IDs to ignore, e.g. rules that fire on templated text without being real errors.
    disabled_rules: tuple[str, ...] = ()
    timeout: float = 30.0
    _cache: dict[str, list[dict[str, Any]]] = field(default_factory=dict, repr=False)

    def check(self, text: str) -> list[dict[str, Any]]:
        """Return LanguageTool matches for ``text`` (cached per text)."""
        if text in self._cache:
            return self._cache[text]
        params = {"language": self.language, "text": text}
        if self.disabled_rules:
            params["disabledRules"] = ",".join(self.disabled_rules)
        data = urllib.parse.urlencode(params).encode()
        request = urllib.request.Request(f"{self.url.rstrip('/')}/v2/check", data=data, method="POST")
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            matches = json.load(response)["matches"]
        self._cache[text] = matches
        return matches


def summarize_match(match: dict[str, Any]) -> dict[str, Any]:
    return {
        "rule_id": match["rule"]["id"],
        "category": match["rule"].get("category", {}).get("id"),
        "message": match["message"],
        "offset": match["offset"],
        "length": match["length"],
    }


def check_pair(client: LanguageToolClient, good: str, bad: str) -> dict[str, Any]:
    good_matches = [summarize_match(m) for m in client.check(good)]
    bad_matches = [summarize_match(m) for m in client.check(bad)]
    return {
        "good_clean": not good_matches,
        "bad_flagged": bool(bad_matches),
        "good_matches": good_matches,
        "bad_matches": bad_matches,
    }
