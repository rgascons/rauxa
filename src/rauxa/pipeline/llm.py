"""LLM backends. Both return a parsed JSON object from ``chat_json``.

- ``OpenAICompatibleBackend``: OpenAI-compatible chat completions (OpenAI, vLLM, Ollama...).
- ``AnthropicBackend``: Claude through Anthropic's SDK (the only optional dependency).
- ``AgentFileBackend``: requests written to files and answered by a coding agent.

Standard library only, except ``AnthropicBackend``, which imports the SDK when used.
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
from .storage import write_json, write_text


class LLMError(RuntimeError):
    """One call failed (bad reply, timeout, refusal...). The pipeline records it and carries on."""


class BackendUnavailable(RuntimeError):
    """The backend can't work at all: missing SDK, missing or rejected credentials, unknown
    model or endpoint. Unlike ``LLMError`` this stops the run, so a setup mistake doesn't
    quietly turn into hundreds of failed calls and rejected pairs."""


# HTTP statuses that mean a setup problem rather than a failed call.
_SETUP_STATUSES = {401: "the API key was rejected", 403: "the API key lacks permission", 404: "model or endpoint not found"}


class AwaitingAgent(Exception):
    """Raised by ``AgentFileBackend`` when a request has been written but not answered yet."""


class AgentRequestFailed(LLMError):
    """A dispatched request exhausted its attempts; never submit it again."""


def agent_request(
    backend: ChatBackend, messages: list[dict[str, str]], *, kind: str, key: str, count: int = 1,
) -> dict[str, Any]:
    if isinstance(backend, AgentFileBackend):
        return backend.chat_request(messages, kind=kind, key=key, count=count)
    return backend.chat_json(messages)


def managed_request_id(record: dict[str, Any]) -> str:
    payload = {key: record[key] for key in ("version", "backend", "model", "kind", "key", "count", "messages")}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:32]


def validate_request(record: dict[str, Any], rid: str) -> None:
    if not re.fullmatch(r"[a-f0-9]{32}", rid) or record.get("id") != rid or record.get("version") != 2:
        raise ValueError(f"invalid or legacy agent request {rid!r}; use a fresh run for Copilot dispatch")
    if record.get("kind") not in ("write", "fix", "judge"):
        raise ValueError(f"unknown request kind for {rid}")
    if any(not isinstance(record.get(k), str) or not record[k] for k in ("backend", "model", "key")):
        raise ValueError(f"request {rid} needs explicit backend, model, and logical key")
    if type(record.get("count")) is not int or record["count"] < 1:
        raise ValueError(f"invalid request count for {rid}")
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages or any(
        not isinstance(m, dict) or m.get("role") not in ("system", "user") or not isinstance(m.get("content"), str)
        for m in messages
    ):
        raise ValueError(f"invalid conversation for request {rid}")
    if managed_request_id(record) != rid:
        raise ValueError(f"request {rid} does not match its content")


def validate_agent_answer(value: Any, kind: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise LLMError("answer must be a JSON object")
    if kind == "judge":
        from .agents import verdict_keys  # agents imports this module

        # The answer's shape tells the item kind (pair, preference or control).
        shapes = [verdict_keys(k) for k in ("pair", "preference", "control")]
        complete = any(all(type(value.get(k)) is bool for k in keys) for keys in shapes)
        if not complete or not isinstance(value.get("comment"), str):
            raise LLMError("judge answer needs every boolean verdict the request asks for and a string comment")
    else:
        pairs = value.get("items") if kind == "write" else [value]
        if not isinstance(pairs, list) or not pairs:
            raise LLMError("writer answer needs a nonempty items array")
        for pair in pairs:
            if isinstance(pair, dict) and "sentence" in pair and "good" not in pair:
                # A control: one sentence.
                if not isinstance(pair["sentence"], str) or not pair["sentence"].strip():
                    raise LLMError("each control needs a nonempty sentence string")
                continue
            if not isinstance(pair, dict) or any(
                not isinstance(pair.get(k), str) or not pair[k].strip() for k in ("good", "bad")
            ):
                raise LLMError("each pair needs nonempty good and bad strings")
            alternatives = pair.get("alternatives")
            if not isinstance(alternatives, list) or any(not isinstance(a, str) or not a.strip() for a in alternatives):
                raise LLMError("alternatives must be an array of nonempty strings")
    return value


class ChatBackend(Protocol):
    model: str

    def chat_json(self, messages: list[dict[str, str]]) -> dict[str, Any]: ...


class OpenAICompatibleBackend:
    def __init__(self, config: BackendConfig):
        self.config = config
        self.model = config.model
        self._api_key = os.environ.get(config.api_key_env, "") if config.api_key_env else ""
        if config.api_key_env and not self._api_key:
            raise BackendUnavailable(f"{config.model}: environment variable {config.api_key_env} is not set")

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
        payload: dict[str, Any] = {"model": self.config.model, "messages": messages}
        if self.config.temperature is not None:
            payload["temperature"] = self.config.temperature
        if self.config.json_mode:
            payload["response_format"] = {"type": "json_object"}
        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                body = self._post(payload)
                return parse_json_object(body["choices"][0]["message"]["content"])
            except urllib.error.HTTPError as e:
                if e.code in _SETUP_STATUSES:
                    raise BackendUnavailable(
                        f"{self.model} at {self.config.base_url}: {_SETUP_STATUSES[e.code]} (HTTP {e.code})"
                    ) from e
                last_error = e
                time.sleep(min(2**attempt, 10))
            except (urllib.error.URLError, TimeoutError, KeyError, ValueError, LLMError) as e:
                last_error = e
                time.sleep(min(2**attempt, 10))
        raise LLMError(f"{self.model}: failed after {self.config.max_retries} attempts: {last_error}")


# Models that accept the server-side refusal fallback ("fallbacks": "default").
_FALLBACK_MODELS = frozenset({"claude-fable-5-1", "claude-opus-5-5", "claude-opus-5", "claude-sonnet-5-5"})


class AnthropicBackend:
    """Claude through Anthropic's Python SDK, an optional dependency (``pip install -e ".[anthropic]"``).

    No sampling parameters are sent: current Claude models reject them. Depth of reasoning is
    set with ``effort`` instead. JSON comes from the prompt, and the reply text is parsed with
    the same tolerant parser as the other backends.
    """

    def __init__(self, config: BackendConfig):
        try:
            import anthropic
        except ImportError as e:
            raise BackendUnavailable(
                "backend type 'anthropic' needs Anthropic's SDK: pip install -e \".[anthropic]\""
            ) from e
        self._anthropic = anthropic
        self.config = config
        self.model = config.model
        kwargs: dict[str, Any] = {"max_retries": config.max_retries, "timeout": config.timeout}
        if config.api_key_env:
            key = os.environ.get(config.api_key_env, "")
            if not key:
                raise BackendUnavailable(f"{config.model}: environment variable {config.api_key_env} is not set")
            kwargs["api_key"] = key
        if config.base_url:
            kwargs["base_url"] = config.base_url
        self.client = anthropic.Anthropic(**kwargs)

    def chat_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        anthropic = self._anthropic
        params: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 16000,
            "messages": [m for m in messages if m["role"] != "system"],
        }
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        if system:
            params["system"] = system
        if self.config.effort:
            params["output_config"] = {"effort": self.config.effort}
        try:
            if self.config.fallbacks and self.model in _FALLBACK_MODELS:
                response = self.client.beta.messages.create(
                    **params, betas=["server-side-fallback-2026-07-01"], extra_body={"fallbacks": "default"}
                )
            else:
                response = self.client.messages.create(**params)
        except anthropic.RateLimitError as e:
            raise LLMError(f"{self.model}: rate limited after retries: {e}") from e
        except anthropic.APIStatusError as e:
            if e.status_code in _SETUP_STATUSES:
                raise BackendUnavailable(f"{self.model}: {_SETUP_STATUSES[e.status_code]} (HTTP {e.status_code})") from e
            raise LLMError(f"{self.model}: API error {e.status_code}: {e}") from e
        except anthropic.APIConnectionError as e:
            raise LLMError(f"{self.model}: connection error: {e}") from e
        except TypeError as e:
            # The SDK raises TypeError when it finds no credentials at all.
            if "authentication" in str(e).lower():
                raise BackendUnavailable(
                    f"{self.model}: no Anthropic credentials found. Set ANTHROPIC_API_KEY (or "
                    "api_key_env in the config), or log in with `ant auth login`"
                ) from e
            raise
        if response.stop_reason == "refusal":
            category = getattr(response.stop_details, "category", None)
            raise LLMError(f"{self.model} declined the request (category: {category})")
        if response.stop_reason == "max_tokens":
            raise LLMError(f"{self.model}: reply cut off at max_tokens")
        text = "".join(block.text for block in response.content if block.type == "text")
        return parse_json_object(text)


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

    def __init__(self, config: BackendConfig, exchange_dir: Path, *, name: str = "", version: int = 1):
        self.config = config
        self.model = config.model
        self.name = name
        self.version = version
        self.exchange_dir = exchange_dir
        self.requests = exchange_dir / "requests"
        self.responses = exchange_dir / "responses"
        self.done = self.requests / "done"
        self.failures = exchange_dir / "failures"
        for d in (self.requests, self.responses, self.done, self.failures):
            d.mkdir(parents=True, exist_ok=True)
        self.pending: set[str] = set()
        self._lock = threading.Lock()

    @staticmethod
    def request_id(messages: list[dict[str, str]]) -> str:
        blob = json.dumps(messages, ensure_ascii=False, sort_keys=True)
        return hashlib.sha1(blob.encode()).hexdigest()[:16]

    def chat_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        return self.chat_request(messages, kind="write", key=self.request_id(messages))

    def _archive(self, rid: str) -> None:
        for suffix in (".md", ".json"):
            request = self.requests / f"{rid}{suffix}"
            if request.exists():
                os.replace(request, self.done / request.name)

    def chat_request(self, messages: list[dict[str, str]], *, kind: str, key: str, count: int = 1) -> dict[str, Any]:
        record = {"version": 2, "backend": self.name, "model": self.model,
                  "kind": kind, "key": key, "count": count, "messages": messages}
        rid = managed_request_id(record) if self.version == 2 else self.request_id(messages)
        record["id"] = rid
        response = self.responses / f"{rid}.json"
        if response.exists():
            try:
                value = parse_json_object(response.read_text(encoding="utf-8"))
            except (ValueError, LLMError) as e:
                raise LLMError(f"agent response {response.name} is not a valid JSON object: {e}") from e
            self._archive(rid)
            return value
        failure = self.failures / f"{rid}.json"
        if failure.exists():
            data = json.loads(failure.read_text(encoding="utf-8"))
            if data.get("id") != rid:
                raise ValueError(f"invalid failure record {failure}")
            self._archive(rid)
            raise AgentRequestFailed(f"agent request {rid} failed: {data['category']}: {data['message']}")
        md = self.requests / f"{rid}.md"
        request_json = self.requests / f"{rid}.json"
        if not request_json.exists():
            write_json(request_json, record if self.version == 2 else {"id": rid, "messages": messages})
        if not md.exists():
            system = next((m["content"] for m in messages if m["role"] == "system"), "")
            user = "\n\n".join(m["content"] for m in messages if m["role"] == "user")
            write_text(md, _REQUEST_TEMPLATE.format(rid=rid, system=system, user=user))
        with self._lock:
            self.pending.add(rid)
        raise AwaitingAgent(rid)


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse the first JSON object in ``text`` (tolerates code fences and surrounding prose)."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        # Take the first complete object, ignoring prose or further objects around it.
        decoder = json.JSONDecoder()
        value = None
        start = text.find("{")
        while start >= 0:
            try:
                value, _ = decoder.raw_decode(text, start)
                break
            except json.JSONDecodeError:
                start = text.find("{", start + 1)
        if value is None:
            raise LLMError(f"no JSON object in response: {text[:200]!r}") from None
    if not isinstance(value, dict):
        raise LLMError("response JSON is not an object")
    return value


def model_family(model: str) -> str:
    """Rough model family from its name: 'gpt-4o' → 'gpt', 'BSC-LT/salamandra-7b' → 'salamandra'."""
    name = model.lower().rsplit("/", 1)[-1]
    return re.split(r"[-_:.\d]", name, maxsplit=1)[0] or name
