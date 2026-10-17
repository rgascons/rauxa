import json
import tempfile
import unittest
from pathlib import Path

from rauxa.pipeline.config import BackendConfig, ChecksConfig, LanguageToolCheckConfig, RunConfig, SourceConfig
from rauxa.pipeline.controller import Pipeline
from rauxa.pipeline.llm import (
    AgentFileBackend, AgentRequestFailed, AwaitingAgent, LLMError,
    validate_agent_answer, validate_request,
)
from rauxa.pipeline.storage import RunLocked, write_json


MESSAGES = [{"role": "system", "content": "Write Catalan."}, {"role": "user", "content": "One pair."}]


def run_config(directory, count=4):
    return RunConfig(
        name="run", output_dir=str(directory), target_size=count, pool_multiplier=1,
        sources=[SourceConfig(type="llm", task="castellanismes_verbals", backend="writer",
                              count=count, batch_size=1, phenomena=["tenir_que"], max_fix_attempts=0)],
        checks=ChecksConfig(languagetool=LanguageToolCheckConfig(enabled=False)),
        backends={"writer": BackendConfig(type="agent", model="test-model")},
    )


def answer(batch):
    return {"items": [{"good": f"El grup {batch} ha de treballar.", "bad": f"El grup {batch} té que treballar.",
                       "alternatives": []}]}


class TestAgentExchange(unittest.TestCase):
    def test_identity_metadata_and_legacy(self):
        with tempfile.TemporaryDirectory() as tmp:
            ids = []
            for name, model, key in (("a", "m1", "write:0"), ("b", "m1", "write:0"),
                                     ("a", "m2", "write:0"), ("a", "m1", "write:1")):
                backend = AgentFileBackend(BackendConfig(model=model), Path(tmp), name=name, version=2)
                with self.assertRaises(AwaitingAgent) as pending:
                    backend.chat_request(MESSAGES, kind="write", key=key)
                rid = str(pending.exception)
                ids.append(rid)
                data = json.loads((backend.requests / f"{rid}.json").read_text())
                validate_request(data, rid)
                data["model"] = "tampered"
                with self.assertRaises(ValueError):
                    validate_request(data, rid)
            self.assertEqual(len(set(ids)), 4)
            legacy = AgentFileBackend(BackendConfig(), Path(tmp) / "legacy")
            with self.assertRaises(AwaitingAgent) as pending:
                legacy.chat_json(MESSAGES)
            self.assertEqual(str(pending.exception), legacy.request_id(MESSAGES))

    def test_atomic_failure_is_not_an_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            backend = AgentFileBackend(BackendConfig(model="m"), Path(tmp), name="a", version=2)
            with self.assertRaises(AwaitingAgent) as pending:
                backend.chat_request(MESSAGES, kind="write", key="write:0")
            rid = str(pending.exception)
            write_json(backend.failures / f"{rid}.json", {"id": rid, "category": "timeout", "message": "too slow"})
            with self.assertRaises(AgentRequestFailed):
                backend.chat_request(MESSAGES, kind="write", key="write:0")
            self.assertFalse((backend.responses / f"{rid}.json").exists())

    def test_validation_does_not_retry_negative_verdicts(self):
        negative = {"good_is_correct": False, "variants_are_correct": True, "bad_has_stated_error": True,
                    "bad_has_other_errors": True, "natural": False, "same_meaning": False, "comment": "No."}
        self.assertEqual(validate_agent_answer(negative, "judge"), negative)
        for raw, kind in (({}, "write"), ({"items": []}, "write"), ({"good": "", "bad": "a"}, "fix"),
                          ({**negative, "natural": "false"}, "judge")):
            with self.subTest(raw=raw), self.assertRaises(LLMError):
                validate_agent_answer(raw, kind)

    def test_out_of_order_resume_and_terminal_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = run_config(tmp)
            first = Pipeline(cfg, log=lambda _: None).run()
            exchange = cfg.run_dir() / "agent_exchange"
            requests = [json.loads(p.read_text()) for p in (exchange / "requests").glob("*.json")]
            self.assertEqual(first.status, "waiting")
            for request in requests:
                batch = int(request["key"].split(":")[-1])
                if batch == 0:
                    missing = request
                elif batch == 2:
                    write_json(exchange / "failures" / f"{request['id']}.json",
                               {"id": request["id"], "category": "exit", "message": "failed"})
                else:
                    write_json(exchange / "responses" / f"{request['id']}.json", answer(batch))
            second = Pipeline(cfg, resume=True, log=lambda _: None).run()
            self.assertEqual(second.manifest["pending_requests"], [missing["id"]])
            state = json.loads((cfg.run_dir() / "state.json").read_text())
            self.assertEqual(state["sources"]["0"]["requested"], 3)
            self.assertEqual(state["writer_errors"], 1)
            Pipeline(cfg, resume=True, log=lambda _: None).run()
            write_json(exchange / "responses" / f"{missing['id']}.json", answer(0))
            final = Pipeline(cfg, resume=True, log=lambda _: None).run()
            self.assertEqual(final.status, "incomplete")
            self.assertEqual(final.selected, 3)
            self.assertEqual(final.manifest["writer_errors"], 1)
            self.assertEqual(final.manifest["counts"]["candidates"], 3)

    def test_lock_covers_initialization(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = run_config(tmp)
            with Pipeline(cfg, log=lambda _: None) as first:
                with self.assertRaises(RunLocked):
                    Pipeline(cfg, resume=True, log=lambda _: None)
                first.run()
            self.assertEqual(Pipeline(cfg, resume=True, log=lambda _: None).run().status, "waiting")


if __name__ == "__main__":
    unittest.main()
