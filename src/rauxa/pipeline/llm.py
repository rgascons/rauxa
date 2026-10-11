"""Minimal client for OpenAI-compatible chat completions (OpenAI, vLLM, Ollama, llama.cpp...).

Standard library only. Agents use ``chat_json`` and get a parsed JSON object back.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Protocol

from .config import BackendConfig


class LLMError(RuntimeError):
    pass


class ChatBackend(Protocol):
    model: str

    def chat_json(self, messages: list[dict[str, str]]) -> dict[str, Any]: ...


class OpenAICompatibleBackend:
    def __init__(self, config: BackendConfig):
        self.config = config
        self.model = config.model
        self._api_key = os.environ.get(config.api_key_env, "") if config.api_key_env else ""
        if config.api_key_env and not self._api_key:
            raise LLMError(f"environment variable {config.api_key_env} is not set")

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        request = urllib.request.Request(
            f"{self.config.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode(),
            headers=headers,
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.config.timeout) as response:
            return json.load(response)

    def chat_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
        }
        if self.config.json_mode:
            payload["response_format"] = {"type": "json_object"}
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                body = self._post(payload)
                return parse_json_object(body["choices"][0]["message"]["content"])
            except (urllib.error.URLError, TimeoutError, KeyError, ValueError, LLMError) as e:
                last_error = e
                time.sleep(min(2**attempt, 10))
        raise LLMError(f"{self.model}: failed after {self.config.max_retries} attempts: {last_error}")


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse the first JSON object in ``text`` (tolerates code fences and surrounding prose)."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise LLMError(f"no JSON object in response: {text[:200]!r}") from None
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise LLMError("response JSON is not an object")
    return value


def model_family(model: str) -> str:
    """Rough model family from its name: 'gpt-4o' → 'gpt', 'BSC-LT/salamandra-7b' → 'salamandra'."""
    name = model.lower().rsplit("/", 1)[-1]
    return re.split(r"[-_:.\d]", name, maxsplit=1)[0] or name
