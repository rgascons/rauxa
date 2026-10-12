"""LLM backends. Both return a parsed JSON object from ``chat_json``.

- ``OpenAICompatibleBackend``: OpenAI-compatible chat completions (OpenAI, vLLM, Ollama...).
- ``AgentFileBackend``: requests written to files and answered by a coding agent.

Standard library only.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Protocol

from .config import BackendConfig


class LLMError(RuntimeError):
    pass


class AwaitingAgent(Exception):
    """Raised by ``AgentFileBackend`` when a request has been written but not answered yet."""


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


_REQUEST_TEMPLATE = """\
# Request {rid}

Reply as the assistant in the conversation below. Write a single JSON object and nothing
else to this file (relative to the `agent_exchange` folder):

    responses/{rid}.json

Answer from the conversation alone: don't read the run's other files to decide.

## System

{system}

## User

{user}
"""


class AgentFileBackend:
    """Serves LLM calls through files, so a coding agent can play the LLM roles.

    Each call is identified by a hash of its messages. If ``responses/<id>.json`` exists it is
    returned; otherwise the prompt is written to ``requests/<id>.md`` (and ``.json``) and
    ``AwaitingAgent`` is raised. Prompts are deterministic, so a resumed run asks again for
    exactly the requests it is still missing.
    """

    def __init__(self, config: BackendConfig, exchange_dir: Path):
        self.config = config
        self.model = config.model
        self.exchange_dir = exchange_dir
        self.requests = exchange_dir / "requests"
        self.responses = exchange_dir / "responses"
        self.done = self.requests / "done"
        for d in (self.requests, self.responses, self.done):
            d.mkdir(parents=True, exist_ok=True)
        self.pending: set[str] = set()
        self._lock = threading.Lock()

    @staticmethod
    def request_id(messages: list[dict[str, str]]) -> str:
        blob = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()[:16]

    def chat_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        rid = self.request_id(messages)
        response = self.responses / f"{rid}.json"
        if response.exists():
            try:
                value = parse_json_object(response.read_text(encoding="utf-8"))
            except (ValueError, LLMError) as e:
                raise LLMError(f"agent response {response.name} is not a valid JSON object: {e}") from e
            for suffix in (".md", ".json"):
                request = self.requests / f"{rid}{suffix}"
                if request.exists():
                    os.replace(request, self.done / request.name)
            return value
        md = self.requests / f"{rid}.md"
        if not md.exists():
            system = next((m["content"] for m in messages if m["role"] == "system"), "")
            user = "\n\n".join(m["content"] for m in messages if m["role"] == "user")
            md.write_text(_REQUEST_TEMPLATE.format(rid=rid, system=system, user=user), encoding="utf-8")
            (self.requests / f"{rid}.json").write_text(
                json.dumps({"id": rid, "messages": messages}, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        with self._lock:
            self.pending.add(rid)
        raise AwaitingAgent(rid)


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
