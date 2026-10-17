"""The pipeline controller. LLMs only suggest pairs or give opinions; this code decides
which pairs are kept.

Each round: make candidate pairs (procedural generator and/or an LLM writer) → automatic
checks (LLM-written pairs that fail get a limited number of fix attempts) → optional LLM
judge → keep or drop. Rounds repeat until the kept pool reaches
``target_size * pool_multiplier`` with every minimum share reachable, or the sources run out.
The dataset is then selected from the pool and written as ``dataset.jsonl`` when it is
complete, or ``dataset.incomplete.jsonl`` when it is too small or a minimum share is missed.

With ``agent`` backends, LLM calls are written as files for a coding agent to answer. The run
stops with status "waiting" while any are unanswered and continues with ``--resume``.

Run directory (files are append-only or replaced atomically, so a run can resume):

    state.json                 progress: rounds, generator indices, LLM requests
    candidates.jsonl           every pair made — unchecked, do not use as a dataset
    checks.jsonl               automatic check results per pair
    judge.jsonl                judge verdicts (or errors) per pair
    dataset.jsonl | dataset.incomplete.jsonl
    rejected.jsonl             pairs that were dropped, with the reason
    manifest.json              config, counts, check and judge stats, shares, warnings
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from random import Random
from typing import Any, Callable, Optional

from .. import __version__
from ..factory import create_dataset, get_phenomenon_info
from ..qc.languagetool import LanguageToolClient
from .agents import Judge, Writer, WrittenPair, failed_questions, llm_entry
from .checks import build_checks, run_checks
from .config import RunConfig, SourceConfig
from .progress import Progress
from .llm import (
    AgentFileBackend,
    AnthropicBackend,
    AwaitingAgent,
    ChatBackend,
    LLMError,
    OpenAICompatibleBackend,
    model_family,
)
from .selection import item_id, select, share_report
from .storage import RunLock, read_records, write_json

_HUGE = 2**31 - 1

COMPLETE_FILE = "dataset.jsonl"
INCOMPLETE_FILE = "dataset.incomplete.jsonl"


class PipelineError(RuntimeError):
    pass


@dataclass
class RunResult:
    status: str  # "complete", "incomplete", or "waiting" (agent requests to answer)
    path: Path
    selected: int
    manifest: dict[str, Any]


def _append(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read(path: Path) -> list[dict[str, Any]]:
    return list(read_records(path))


def _write_json(path: Path, data: Any) -> None:
    write_json(path, data)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


class Pipeline:
    def __init__(
        self,
        config: RunConfig,
        *,
        resume: bool = False,
        lt_client: Optional[LanguageToolClient] = None,
        backends: Optional[dict[str, ChatBackend]] = None,
        log: Optional[Callable[[str], None]] = None,
        progress: Optional[Progress] = None,
        run_lock: Optional[RunLock] = None,
    ):
        self._owns_lock = run_lock is None
        self._run_lock = run_lock or RunLock(config.run_dir())
        try:
            if self._owns_lock:
                self._run_lock.__enter__()
            elif self._run_lock.directory != config.run_dir().resolve():
                raise PipelineError("run lock belongs to another directory")
            self._run_lock.fileno()
            self._initialize(config, resume=resume, lt_client=lt_client, backends=backends, log=log, progress=progress)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self._owns_lock:
            self._run_lock.__exit__()

    def __enter__(self) -> Pipeline:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _initialize(self, config, *, resume, lt_client, backends, log, progress) -> None:
        config.validate()
        self.cfg = config
        self.dir = config.run_dir()
        self.progress = progress or Progress()
        write_log = log or (lambda msg: print(msg, file=sys.stderr, flush=True))

        def _log(msg: str) -> None:
            self.progress.clear()
            write_log(msg)

        self.log = _log
        self.paths = {
            name: self.dir / name
            for name in ("state.json", "candidates.jsonl", "checks.jsonl", "judge.jsonl", "manifest.json")
        }
        self._open_run_dir(resume)

        self.info = {s.task: get_phenomenon_info(s.task) for s in config.sources}
        markers = {p: i["marker"] for infos in self.info.values() for p, i in infos.items() if "marker" in i}

        if config.checks.languagetool.enabled and lt_client is None:
            lt = config.checks.languagetool
            lt_client = LanguageToolClient(url=lt.url, language=lt.language, disabled_rules=tuple(lt.disabled_rules))
            try:
                lt_client.check("Bon dia.")
            except OSError as e:
                raise PipelineError(
                    f"LanguageTool server not reachable at {lt.url} ({e}). Start it, or set "
                    "[checks.languagetool] enabled = false."
                ) from e
        self.checks = build_checks(config.checks, markers, lt_client)

        self.backends = backends if backends is not None else self._build_backends()
        self.writers = {i: Writer(self.backends[s.backend]) for i, s in enumerate(config.sources) if s.type == "llm"}
        self.judge = Judge(self.backends[config.judge.backend]) if config.judge.enabled else None
        self._datasets: dict[int, Any] = {}

    # -- setup -------------------------------------------------------------------------

    def _open_run_dir(self, resume: bool) -> None:
        state_path = self.paths["state.json"]
        if state_path.exists():
            if not resume:
                raise PipelineError(f"{self.dir} already holds a run; pass --resume or choose another name")
            self.state = json.loads(state_path.read_text(encoding="utf-8"))
            if self.state["fingerprint"] != self.cfg.fingerprint():
                raise PipelineError("config changed since this run started; use a new run name")
        else:
            self.dir.mkdir(parents=True, exist_ok=True)
            self.state = {
                "fingerprint": self.cfg.fingerprint(),
                "rounds_done": 0,
                "sources": {str(i): {"next_index": 0, "requested": 0, "exhausted": False} for i in range(len(self.cfg.sources))},
                "duplicates_skipped": 0,
                "writer_errors": 0,
                "agent_protocol": 2,
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            self._save_state()
        self.candidates: dict[str, dict[str, Any]] = {}
        for entry in _read(self.paths["candidates.jsonl"]):
            self.candidates[entry["metadata"]["item_id"]] = entry
        self.check_records = {r["item_id"]: r for r in _read(self.paths["checks.jsonl"])}
        self.judge_records = {r["item_id"]: r for r in _read(self.paths["judge.jsonl"])}

    def _save_state(self) -> None:
        # Make the records durable before a checkpoint can declare them consumed.
        for name in ("candidates.jsonl", "checks.jsonl", "judge.jsonl"):
            path = self.paths[name]
            if path.exists():
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
        _write_json(self.paths["state.json"], self.state)

    def _build_backends(self) -> dict[str, ChatBackend]:
        needed = {s.backend for s in self.cfg.sources if s.type == "llm"}
        if self.cfg.judge.enabled:
            needed.add(self.cfg.judge.backend)
        backends: dict[str, ChatBackend] = {}
        for name in needed:
            config = self.cfg.backends[name]
            if config.type == "agent":
                backends[name] = AgentFileBackend(
                    config, self.dir / "agent_exchange", name=name, version=self.state.get("agent_protocol", 1),
                )
            elif config.type == "anthropic":
                backends[name] = AnthropicBackend(config)
            else:
                backends[name] = OpenAICompatibleBackend(config)
        return backends

    def _pending_requests(self) -> set[str]:
        """Agent requests raised during this run that have no response yet."""
        pending: set[str] = set()
        for backend in self.backends.values():
            pending |= getattr(backend, "pending", set())
        return pending

    # -- making candidates -------------------------------------------------------------

    def _add_candidate(self, entry: dict[str, Any], source_index: int) -> Optional[dict[str, Any]]:
        iid = item_id(entry)
        if iid in self.candidates:
            self.state["duplicates_skipped"] += 1
            return None
        entry["metadata"]["item_id"] = iid
        entry["metadata"]["source"] = {"index": source_index, "type": self.cfg.sources[source_index].type}
        self.candidates[iid] = entry
        _append(self.paths["candidates.jsonl"], entry)
        return entry

    def _dataset(self, i: int, src: SourceConfig) -> Any:
        if i not in self._datasets:
            self._datasets[i] = create_dataset(src.task, seed=self.cfg.seed + 7919 * i, size=_HUGE, **src.params)
        return self._datasets[i]

    def _make_procedural(self, i: int, src: SourceConfig, n: int) -> list[dict[str, Any]]:
        st = self.state["sources"][str(i)]
        out: list[dict[str, Any]] = []
        tries = 0
        while len(out) < n and not st["exhausted"]:
            if tries >= self.cfg.max_draws_per_round:
                st["exhausted"] = True
                self.log(f"source {i} ({src.task}): no new unique pairs after {tries} draws, out of combinations")
                break
            entry = self._dataset(i, src)[st["next_index"]]
            st["next_index"] += 1
            tries += 1
            added = self._add_candidate(entry, i)
            if added is not None:
                out.append(added)
                tries = 0
        return out

    def _examples(self, task: str, phenomenon: str, batch_index: int, k: int = 4) -> list[tuple[str, str]]:
        """Example pairs for a writer prompt, different for each batch.

        Different examples keep prompts for the same phenomenon distinct: otherwise an agent
        backend, which identifies requests by their content, would reuse the first answer.
        """
        try:
            data = create_dataset(task, seed=self.cfg.seed, size=_HUGE, phenomena=(phenomenon,))
        except (AssertionError, TypeError, ValueError):
            return []
        # Writer examples are normative pairs only; skip preferences and controls. Pair-only
        # tasks take exactly the k items they always took, so prompts don't change.
        examples, j = [], batch_index * k
        while len(examples) < k and j < batch_index * k + 50 * k:
            meta = data[j]["metadata"]
            if meta.get("kind", "pair") == "pair":
                examples.append((meta["sentence_good"], meta["sentence_bad"]))
            j += 1
        return examples

    def _llm_entry(self, i: int, src: SourceConfig, phenomenon: str, pair: WrittenPair, revision_of: Optional[str] = None):
        info = self.info[src.task][phenomenon]
        return llm_entry(
            pair, task=src.task, phenomenon=phenomenon, norm="iec", formats=src.formats,
            rng=Random(f"{self.cfg.seed}:{pair.good}:{pair.bad}"),
            languagetool_rules=info.get("languagetool_rules", []),
            writer_model=self.backends[src.backend].model, revision_of=revision_of,
        )

    def _make_llm(self, i: int, src: SourceConfig, n: int) -> list[dict[str, Any]]:
        if isinstance(self.backends[src.backend], AgentFileBackend) and self.state.get("agent_protocol") == 2:
            return self._make_agent_llm(i, src)
        st = self.state["sources"][str(i)]
        n = min(n, src.count - st["requested"])
        phenomena = src.phenomena or list(self.info[src.task])
        out: list[dict[str, Any]] = []
        consecutive_errors = 0
        waiting_offset = 0  # batches handed to an agent and not answered yet
        total, label = n, f"writing pairs ({self.backends[src.backend].model})"
        self.progress.update(label, 0, total)
        while n > 0 and consecutive_errors < 3:
            batch_index = (st["requested"] + waiting_offset) // max(src.batch_size, 1)
            phenomenon = phenomena[batch_index % len(phenomena)]
            k = min(src.batch_size, n)
            try:
                examples = self._examples(src.task, phenomenon, batch_index)
                pairs = self.writers[i].write(self.info[src.task][phenomenon], examples, k, key=f"write:{i}:{batch_index}")
                consecutive_errors = 0
            except AwaitingAgent:
                waiting_offset += k
                n -= k
                continue
            except LLMError as e:
                self.state["writer_errors"] += 1
                consecutive_errors += 1
                self.log(f"LLM writer {i}: {e}")
                continue
            st["requested"] += k
            n -= k
            self.progress.update(label, total - n, total)
            for pair in pairs:
                added = self._add_candidate(self._llm_entry(i, src, phenomenon, pair), i)
                if added is not None:
                    out.append(added)
        if st["requested"] >= src.count:
            st["exhausted"] = True
        return out

    def _make_agent_llm(self, i: int, src: SourceConfig) -> list[dict[str, Any]]:
        if src.batch_size < 1 or src.count < 0:
            raise PipelineError("agent sources need a positive batch_size and nonnegative count")
        st = self.state["sources"][str(i)]
        batches = st.setdefault("batches", {})
        phenomena = src.phenomena or list(self.info[src.task])
        out = []
        for batch, start in enumerate(range(0, src.count, src.batch_size)):
            key = str(batch)
            if key in batches:
                continue
            k = min(src.batch_size, src.count - start)
            phenomenon = phenomena[batch % len(phenomena)]
            try:
                pairs = self.writers[i].write(
                    self.info[src.task][phenomenon], self._examples(src.task, phenomenon, batch), k,
                    key=f"write:{i}:{batch}",
                )
            except AwaitingAgent:
                continue
            except LLMError as e:
                self.log(f"LLM writer {i}, batch {batch}: {e}")
                self.state["writer_errors"] += 1
                batches[key] = {"status": "failed", "count": k}
            else:
                for pair in pairs:
                    added = self._add_candidate(self._llm_entry(i, src, phenomenon, pair), i)
                    if added is not None:
                        out.append(added)
                batches[key] = {"status": "succeeded", "count": k}
            st["requested"] += k
            self._save_state()
            self.progress.update("reading agent batches", st["requested"], src.count)
        st["exhausted"] = st["requested"] >= src.count
        return out

    def _make_from_llm_sources(self) -> list[dict[str, Any]]:
        """LLM sources always ask for everything they still have to request (their ``count``)."""
        new: list[dict[str, Any]] = []
        for i, src in enumerate(self.cfg.sources):
            st = self.state["sources"][str(i)]
            if src.type == "llm" and not st["exhausted"]:
                new += self._make_llm(i, src, src.count - st["requested"])
        return new

    def _make_procedural_share(self, n: int) -> list[dict[str, Any]]:
        new: list[dict[str, Any]] = []
        procedural = [(i, s) for i, s in enumerate(self.cfg.sources) if s.type == "procedural"]
        total_weight = sum(s.weight for _, s in procedural) or 1.0
        for i, src in procedural:
            new += self._make_procedural(i, src, math.ceil(n * src.weight / total_weight))
        return new

    # -- checks and judge --------------------------------------------------------------

    def _record_checks(self, entry: dict[str, Any], results: list[Any]) -> dict[str, Any]:
        failed = next((r.check for r in results if not r.passed), None)
        record = {
            "item_id": entry["metadata"]["item_id"],
            "passed": failed is None,
            "failed_check": failed,
            "results": [r.to_dict() for r in results],
        }
        self.check_records[record["item_id"]] = record
        _append(self.paths["checks.jsonl"], record)
        return record

    def _check_pending(self) -> None:
        pending = [e for iid, e in self.candidates.items() if iid not in self.check_records]
        if not pending:
            return
        workers = max(1, self.cfg.checks.languagetool.workers)
        with ThreadPoolExecutor(workers) as pool:
            all_results = pool.map(lambda e: run_checks(self.checks, e), pending)
            for done, (entry, results) in enumerate(zip(pending, all_results), 1):
                self._record_checks(entry, results)
                self.progress.update("checking", done, len(pending))

    def _fix_depth(self, iid: str) -> int:
        """How many fixes led to pair ``iid`` (0 for a pair as first written)."""
        depth, meta = 0, self.candidates[iid]["metadata"]
        while meta.get("revision_of") in self.candidates:
            depth += 1
            meta = self.candidates[meta["revision_of"]]["metadata"]
        return depth

    def _fix_pending(self) -> None:
        """Ask the LLM writer to fix pairs that failed the checks, then check the fixes.

        Which pairs still need a fix is worked out from the run files, so this resumes
        correctly when fixes come from an agent over several runs.
        """
        given_up = self.state.setdefault("fix_given_up", [])
        while True:
            fixed_already = {e["metadata"].get("revision_of") for e in self.candidates.values()}
            todo = []
            for iid, rec in self.check_records.items():
                meta = self.candidates[iid]["metadata"]
                if rec["passed"] or meta["source"]["type"] != "llm" or iid in fixed_already or iid in given_up:
                    continue
                if self._fix_depth(iid) < self.cfg.sources[meta["source"]["index"]].max_fix_attempts:
                    todo.append(iid)
            progressed = False
            for done, iid in enumerate(todo):
                self.progress.update("fixing failed pairs", done, len(todo))
                meta = self.candidates[iid]["metadata"]
                i = meta["source"]["index"]
                src = self.cfg.sources[i]
                failing = next(r for r in self.check_records[iid]["results"] if not r["passed"])
                problems = [f"{failing['check']}: {json.dumps(failing['details'], ensure_ascii=False)}"]
                pair = WrittenPair(meta["sentence_good"], meta["sentence_bad"],
                                   [v for v in meta["correct_variants"] if v != meta["sentence_good"]])
                try:
                    fixed = self.writers[i].fix(
                        self.info[src.task][meta["phenomenon"]], pair, problems, key=f"fix:{i}:{iid}",
                    )
                except AwaitingAgent:
                    continue
                except LLMError as e:
                    self.log(f"fix attempt failed: {e}")
                    given_up.append(iid)
                    self._save_state()
                    continue
                candidate = self._add_candidate(self._llm_entry(i, src, meta["phenomenon"], fixed, revision_of=iid), i)
                if candidate is None:
                    given_up.append(iid)
                    continue
                self._record_checks(candidate, run_checks(self.checks, candidate))
                progressed = True
            if not progressed:
                return

    def _sampled_for_audit(self, iid: str) -> bool:
        return int(iid[:8], 16) / 0xFFFFFFFF < self.cfg.judge.sample_rate

    def _judged_source(self, iid: str) -> bool:
        types = self.cfg.judge.source_types
        return not types or self.candidates[iid]["metadata"]["source"]["type"] in types

    def _judge_pending(self) -> None:
        if self.judge is None:
            return
        pending = [
            iid for iid, rec in self.check_records.items()
            if rec["passed"] and iid not in self.judge_records and self._judged_source(iid)
            and (self.cfg.judge.mode == "filter" or self._sampled_for_audit(iid))
        ]
        if not pending:
            return

        def judge(iid: str) -> Optional[dict[str, Any]]:
            entry = self.candidates[iid]
            meta = entry["metadata"]
            info = self.info[self.cfg.sources[meta["source"]["index"]].task][meta["phenomenon"]]
            try:
                return {"item_id": iid, "verdict": self.judge.judge(entry, info)}
            except AwaitingAgent:
                return None
            except LLMError as e:
                return {"item_id": iid, "error": str(e)}

        with ThreadPoolExecutor(max(1, self.cfg.judge.workers)) as pool:
            for done, record in enumerate(pool.map(judge, pending), 1):
                self.progress.update(f"judging ({self.judge.backend.model})", done, len(pending))
                if record is None:
                    continue
                self.judge_records[record["item_id"]] = record
                _append(self.paths["judge.jsonl"], record)

    def _decision(self, iid: str) -> tuple[bool, dict[str, Any]]:
        """Whether pair ``iid`` is kept, and how that was decided (stored on the item)."""
        checks = self.check_records.get(iid)
        if checks is None:
            return False, {"reason": "not_checked"}
        if not checks["passed"]:
            return False, {"reason": "failed_check", "failed_check": checks["failed_check"]}
        judge_cfg = self.cfg.judge
        if not judge_cfg.enabled:
            return True, {"judge": "not_used"}
        if not self._judged_source(iid):
            return True, {"judge": "not_used_for_source"}
        record = self.judge_records.get(iid)
        if record is None:
            if judge_cfg.mode == "audit":
                return True, {"judge": "not_sampled"}
            return False, {"reason": "not_judged"}
        if "error" in record:
            kept = judge_cfg.on_error == "keep"
            return kept, {"judge": "error", "kept_despite_judge_error": kept, "reason": None if kept else "judge_error"}
        passed = record["verdict"]["passed"]
        if judge_cfg.mode == "audit":
            return True, {"judge": "audited", "judge_passed": passed}
        return passed, {"judge": "passed" if passed else "failed", "reason": None if passed else "judge"}

    def _kept_ids(self) -> list[str]:
        return [iid for iid in self.candidates if self._decision(iid)[0]]

    def _share_shortfalls(self, kept_ids: list[str]) -> dict[tuple[str, str], int]:
        """Minimum shares the kept pool cannot meet yet: (field, value) → missing pairs."""
        shortfalls = {}
        for dim, minimums in self.cfg.min_share.items():
            counts = Counter(str(self.candidates[iid]["metadata"].get(dim)) for iid in kept_ids)
            for value, share in minimums.items():
                missing = math.ceil(share * self.cfg.target_size) - counts.get(value, 0)
                if missing > 0:
                    shortfalls[(dim, value)] = missing
        return shortfalls

    # -- main loop ---------------------------------------------------------------------

    def run(self) -> RunResult:
        self._run_lock.fileno()
        try:
            return self._run()
        finally:
            self.progress.clear()
            self.close()

    def _run(self) -> RunResult:
        goal = math.ceil(self.cfg.target_size * self.cfg.pool_multiplier)
        while True:
            self._make_from_llm_sources()
            self._check_pending()
            self._fix_pending()
            self._judge_pending()
            pending = self._pending_requests()
            if pending:
                return self._wait(pending)
            kept_ids = self._kept_ids()
            kept = len(kept_ids)
            shortfalls = self._share_shortfalls(kept_ids)
            checked = len(self.check_records)
            self.log(
                f"round {self.state['rounds_done']}: candidates={len(self.candidates)} checked={checked} "
                f"kept={kept}/{goal} share_shortfalls={ {f'{d}={v}': n for (d, v), n in shortfalls.items()} }"
            )
            if (kept >= goal and not shortfalls) or self.state["rounds_done"] >= self.cfg.max_rounds:
                break
            if all(st["exhausted"] for st in self.state["sources"].values()):
                self.log("all sources exhausted")
                break
            rate = max(kept / checked, 0.2) if checked else 1.0
            # Short on total: make the shortfall. Short only on a minimum share: make more, since
            # sources can't be steered to one value and only part of each batch will help.
            shortfall = goal - kept if kept < goal else 3 * sum(shortfalls.values())
            new = self._make_procedural_share(math.ceil(shortfall / rate))
            self.state["rounds_done"] += 1
            self._save_state()
            if not new and all(iid in self.check_records for iid in self.candidates):
                self.log("no new candidates")
                break
        return self._finish()

    def _wait(self, pending: set[str]) -> RunResult:
        """Stop until a coding agent answers the requests (``agent`` backends only)."""
        self._save_state()
        exchange = self.dir / "agent_exchange"
        self.log(
            f"waiting: {len(pending)} request(s) for the agent in {exchange / 'requests'}. "
            "Answer each one in responses/<id>.json, then run again with --resume."
        )
        return RunResult("waiting", exchange, 0, {"pending_requests": sorted(pending)})

    def _finish(self) -> RunResult:
        kept, rejected = [], []
        for iid, entry in self.candidates.items():
            ok, decision = self._decision(iid)
            checks = self.check_records.get(iid, {})
            annotated = {**entry, "metadata": {**entry["metadata"], "qc": {
                "decision": decision,
                "checks": {r["check"]: r["passed"] for r in checks.get("results", [])},
                "languagetool": next((r["details"] for r in checks.get("results", []) if r["check"] == "languagetool"), None),
                "judge": self.judge_records.get(iid, {}).get("verdict"),
            }}}
            (kept if ok else rejected).append(annotated)

        target = self.cfg.target_size
        selected = select(kept, target, self.cfg.min_share, Random(self.cfg.seed))
        shares = share_report(selected, self.cfg.min_share, target)
        status = "complete" if len(selected) == target and shares["ok"] else "incomplete"
        for name in (COMPLETE_FILE, INCOMPLETE_FILE):
            (self.dir / name).unlink(missing_ok=True)
        path = self.dir / (COMPLETE_FILE if status == "complete" else INCOMPLETE_FILE)
        _write_jsonl(path, selected)
        _write_jsonl(self.dir / "rejected.jsonl", rejected)

        manifest = self._manifest(status, kept, rejected, selected, shares)
        _write_json(self.paths["manifest.json"], manifest)
        self.state["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._save_state()
        self.log(f"{status}: {len(selected)}/{target} pairs → {path}")
        for warning in manifest["warnings"]:
            self.log(f"warning: {warning}")
        return RunResult(status, path, len(selected), manifest)

    def _manifest(self, status, kept, rejected, selected, shares) -> dict[str, Any]:
        check_failures = Counter(r["failed_check"] for r in self.check_records.values() if not r["passed"])
        lt = defaultdict(lambda: Counter())
        missed = defaultdict(list)
        for iid, rec in self.check_records.items():
            details = next((r["details"] for r in rec["results"] if r["check"] == "languagetool"), None)
            meta = self.candidates[iid]["metadata"]
            # Controls have no incorrect sentence, so they say nothing about LanguageTool's coverage.
            if details is None or meta.get("sentence_bad") is None:
                continue
            c = lt[meta["phenomenon"]]
            c["checked"] += 1
            c["bad_flagged"] += details["bad_flagged"]
            c["expected_rule_hit"] += details["expected_rule_hit"]
            if not details["bad_flagged"] and len(missed[meta["phenomenon"]]) < 5:
                missed[meta["phenomenon"]].append(meta["sentence_bad"])

        verdicts = [r["verdict"] for r in self.judge_records.values() if "verdict" in r]
        judge_failed = Counter(k for v in verdicts for k in failed_questions(v))
        reasons = Counter(r["metadata"]["qc"]["decision"].get("reason") for r in rejected)

        warnings = []
        writer_models = {self.backends[s.backend].model for s in self.cfg.sources if s.type == "llm"}
        judge_model = self.backends[self.cfg.judge.backend].model if self.judge else None
        families = lambda models: {model_family(m) for m in models if m}
        if judge_model and model_family(judge_model) in families(writer_models):
            warnings.append(f"judge ({judge_model}) is from the same model family as an LLM writer ({sorted(writer_models)})")
        overlap = families(self.cfg.evaluated_models) & families(writer_models | ({judge_model} if judge_model else set()))
        if overlap:
            warnings.append(f"evaluated models are from the same family as the writer or judge: {sorted(overlap)}")
        for i, st in self.state["sources"].items():
            src = self.cfg.sources[int(i)]
            if st["exhausted"] and src.type == "procedural":
                warnings.append(f"source {i} ({src.task}) ran out of unique pairs; add vocabulary to its templates")
        kept_on_error = sum(1 for r in kept if r["metadata"]["qc"]["decision"].get("kept_despite_judge_error"))
        if kept_on_error:
            warnings.append(f"{kept_on_error} pairs kept even though the judge call failed")
        if not self.cfg.judge.enabled:
            warnings.append("no judge used: pairs were kept on the automatic checks alone")
        agent_backends = sorted(n for n, b in self.backends.items() if isinstance(b, AgentFileBackend))
        if agent_backends:
            warnings.append(
                f"LLM roles {agent_backends} were played by a coding agent through files; "
                "treat this as an approximation of an API run"
            )
            if any(not self.backends[n].model for n in agent_backends):
                warnings.append("agent backend without a model name: same-family checks were skipped for it")
        for dim, rows in shares["fields"].items():
            for value, row in rows.items():
                if not row["ok"]:
                    warnings.append(f"minimum share not met: {dim}={value} is {row['share']:.1%}, minimum {row['min_share']:.0%}")
        if status == "incomplete":
            warnings.append("dataset is incomplete: too few pairs, or a minimum share not met")

        manifest = {
            "rauxa_version": __version__,
            "config": asdict(self.cfg),
            "fingerprint": self.cfg.fingerprint(),
            "status": status,
            "started_at": self.state.get("started_at"),
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "counts": {
                "candidates": len(self.candidates),
                "duplicates_skipped": self.state["duplicates_skipped"],
                "passed_checks": sum(r["passed"] for r in self.check_records.values()),
                "kept": len(kept),
                "rejected": len(rejected),
                "selected": len(selected),
                "by_source_type": dict(Counter(e["metadata"]["source"]["type"] for e in self.candidates.values())),
            },
            "check_failures": dict(check_failures),
            "rejection_reasons": dict(reasons),
            "languagetool": {
                p: {**c, "flag_rate": round(c["bad_flagged"] / c["checked"], 4) if c["checked"] else None,
                    "unflagged_examples": missed[p]}
                for p, c in lt.items()
            },
            "judge": {
                "enabled": self.cfg.judge.enabled,
                "mode": self.cfg.judge.mode,
                "model": judge_model,
                "judged": len(verdicts),
                "passed": sum(v["passed"] for v in verdicts),
                "errors": sum(1 for r in self.judge_records.values() if "error" in r),
                "failed_questions": dict(judge_failed),
                "kept_despite_error": kept_on_error,
            },
            "writer_errors": self.state["writer_errors"],
            "shares": shares,
            "warnings": warnings,
        }
        dispatcher_status = self.dir / "dispatcher" / "status.json"
        if dispatcher_status.exists():
            manifest["dispatcher"] = json.loads(dispatcher_status.read_text(encoding="utf-8"))
        return manifest
