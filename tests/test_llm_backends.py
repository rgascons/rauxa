import json
import sys
import types
import unittest
from types import SimpleNamespace
from unittest import mock

from rauxa.pipeline.config import BackendConfig
import urllib.error

from rauxa.pipeline.llm import AnthropicBackend, BackendUnavailable, LLMError, OpenAICompatibleBackend

MESSAGES = [
    {"role": "system", "content": "You are a strict reviewer."},
    {"role": "user", "content": "Judge this pair."},
]


def reply(text, stop_reason="end_turn"):
    return SimpleNamespace(
        stop_reason=stop_reason,
        stop_details=SimpleNamespace(category="cyber") if stop_reason == "refusal" else None,
        content=[SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)],
    )


def fake_anthropic(responses):
    """A stand-in for the `anthropic` package that records every call."""
    module = types.ModuleType("anthropic")

    class APIError(Exception):
        pass

    class APIStatusError(APIError):
        def __init__(self, message, status_code=500):
            super().__init__(message)
            self.status_code = status_code

    class RateLimitError(APIStatusError):
        pass

    class APIConnectionError(APIError):
        pass

    calls = []

    class Messages:
        def __init__(self, path):
            self.path = path

        def create(self, **kwargs):
            calls.append((self.path, kwargs))
            response = responses.pop(0)
            if isinstance(response, Exception):
                raise response
            return response

    class Anthropic:
        def __init__(self, **kwargs):
            calls.append(("init", kwargs))
            self.messages = Messages("messages")
            self.beta = SimpleNamespace(messages=Messages("beta.messages"))

    module.Anthropic = Anthropic
    module.APIError = APIError
    module.APIStatusError = APIStatusError
    module.RateLimitError = RateLimitError
    module.APIConnectionError = APIConnectionError
    return module, calls


class TestAnthropicBackend(unittest.TestCase):
    def backend(self, responses, **config):
        module, calls = fake_anthropic(responses)
        patcher = mock.patch.dict(sys.modules, {"anthropic": module})
        patcher.start()
        self.addCleanup(patcher.stop)
        return AnthropicBackend(BackendConfig(type="anthropic", **config)), calls

    def test_request_shape_with_fallbacks(self):
        backend, calls = self.backend([reply('{"natural": true}')], model="claude-opus-5-5", effort="medium")
        self.assertEqual(backend.chat_json(MESSAGES), {"natural": True})
        path, kwargs = calls[1]
        self.assertEqual(path, "beta.messages")
        self.assertEqual(kwargs["model"], "claude-opus-5-5")
        self.assertEqual(kwargs["system"], "You are a strict reviewer.")
        self.assertEqual(kwargs["messages"], [{"role": "user", "content": "Judge this pair."}])
        self.assertEqual(kwargs["output_config"], {"effort": "medium"})
        self.assertEqual(kwargs["betas"], ["server-side-fallback-2026-07-01"])
        self.assertEqual(kwargs["extra_body"], {"fallbacks": "default"})
        self.assertNotIn("temperature", kwargs)

    def test_model_without_fallbacks_uses_plain_messages(self):
        backend, calls = self.backend([reply("```json\n{\"ok\": 1}\n```")], model="claude-haiku-5-5")
        self.assertEqual(backend.chat_json(MESSAGES), {"ok": 1})
        path, kwargs = calls[1]
        self.assertEqual(path, "messages")
        self.assertNotIn("output_config", kwargs)
        self.assertNotIn("betas", kwargs)

    def test_fallbacks_can_be_turned_off(self):
        backend, calls = self.backend([reply("{}")], model="claude-opus-5-5", fallbacks=False)
        backend.chat_json(MESSAGES)
        self.assertEqual(calls[1][0], "messages")

    def test_refusal_and_truncation_raise(self):
        backend, _ = self.backend([reply("", "refusal"), reply('{"a"', "max_tokens")], model="claude-opus-5-5")
        with self.assertRaisesRegex(LLMError, "declined"):
            backend.chat_json(MESSAGES)
        with self.assertRaisesRegex(LLMError, "max_tokens"):
            backend.chat_json(MESSAGES)

    def test_api_errors_become_llm_errors(self):
        module, _ = fake_anthropic([])
        with mock.patch.dict(sys.modules, {"anthropic": module}):
            backend = AnthropicBackend(BackendConfig(type="anthropic", model="claude-sonnet-5-5"))
            module_errors = [module.RateLimitError("slow down", 429), module.APIStatusError("bad", 400), module.APIConnectionError("offline")]
            backend.client.beta.messages.create = mock.Mock(side_effect=module_errors)
            for _ in module_errors:
                with self.assertRaises(LLMError):
                    backend.chat_json(MESSAGES)

    def test_setup_errors_stop_the_run(self):
        module, _ = fake_anthropic([])
        with mock.patch.dict(sys.modules, {"anthropic": module}):
            backend = AnthropicBackend(BackendConfig(type="anthropic", model="claude-opus-5-5"))
            no_credentials = TypeError('"Could not resolve authentication method. Expected one of api_key..."')
            for error in (module.APIStatusError("invalid x-api-key", 401), module.APIStatusError("no such model", 404), no_credentials):
                backend.client.beta.messages.create = mock.Mock(side_effect=error)
                with self.assertRaises(BackendUnavailable):
                    backend.chat_json(MESSAGES)
            backend.client.beta.messages.create = mock.Mock(side_effect=TypeError("unrelated bug"))
            with self.assertRaisesRegex(TypeError, "unrelated bug"):
                backend.chat_json(MESSAGES)

    def test_api_key_env(self):
        module, calls = fake_anthropic([])
        with mock.patch.dict(sys.modules, {"anthropic": module}), mock.patch.dict("os.environ", {"MY_KEY": "sk-test"}):
            AnthropicBackend(BackendConfig(type="anthropic", model="claude-opus-5-5", api_key_env="MY_KEY"))
            self.assertEqual(calls[0][1]["api_key"], "sk-test")
        with mock.patch.dict(sys.modules, {"anthropic": module}), mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(BackendUnavailable, "MY_KEY"):
                AnthropicBackend(BackendConfig(type="anthropic", model="claude-opus-5-5", api_key_env="MY_KEY"))

    def test_missing_sdk_explains_install(self):
        with mock.patch.dict(sys.modules, {"anthropic": None}):
            with self.assertRaisesRegex(BackendUnavailable, r'pip install -e "\.\[anthropic\]"'):
                AnthropicBackend(BackendConfig(type="anthropic", model="claude-opus-5-5"))


class TestOpenAICompatibleBackend(unittest.TestCase):
    def payload_for(self, **config):
        backend = OpenAICompatibleBackend(BackendConfig(base_url="http://x", model="m", **config))
        sent = {}

        def fake_post(payload):
            sent.update(payload)
            return {"choices": [{"message": {"content": json.dumps({"ok": True})}}]}

        backend._post = fake_post
        self.assertEqual(backend.chat_json(MESSAGES), {"ok": True})
        return sent

    def test_rejected_key_stops_the_run(self):
        backend = OpenAICompatibleBackend(BackendConfig(base_url="http://x", model="m"))
        backend._post = mock.Mock(side_effect=urllib.error.HTTPError("http://x", 401, "Unauthorized", {}, None))
        with self.assertRaisesRegex(BackendUnavailable, "rejected"):
            backend.chat_json(MESSAGES)
        self.assertEqual(backend._post.call_count, 1)  # not retried

    def test_temperature_only_when_set(self):
        self.assertNotIn("temperature", self.payload_for())
        self.assertEqual(self.payload_for(temperature=0.0)["temperature"], 0.0)


if __name__ == "__main__":
    unittest.main()
