"""The pipeline controller. LLMs only suggest pairs or give opinions; this code decides
which pairs are kept.

Each round: make candidate pairs (procedural generator and/or an LLM writer) → automatic
checks (LLM-written pairs that fail get a limited number of fix attempts) → optional LLM
judge → keep or drop. Rounds repeat until the kept pool reaches
``target_size * pool_multiplier`` with every minimum share reachable, or the sources run out.
The dataset is then selected from the pool and written as ``dataset.jsonl`` when it is
complete, or ``dataset.incomplete.jsonl`` when it is too small or a minimum share is missed.

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
from .llm import ChatBackend, LLMError, OpenAICompatibleBackend, model_family
from .selection import item_id, select, share_report

_HUGE = 2**31 - 1

COMPLETE_FILE = "dataset.jsonl"
INCOMPLETE_FILE = "dataset.incomplete.jsonl"


class PipelineError(RuntimeError):
    pass


@dataclass
class RunResult:
    status: str  # "complete" or "incomplete"
    path: Path
    selected: int
    manifest: dict[str, Any]


def _append(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
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
    ):
        config.validate()
        self.cfg = config
        self.dir = config.run_dir()
        self.log = log or (lambda msg: print(msg, file=sys.stderr, flush=True))
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
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            self._save_state()
        self.candidates: dict[str, dict[str, Any]] = {}
        for entry in _read(self.paths["candidates.jsonl"]):
            self.candidates[entry["metadata"]["item_id"]] = entry
        self.check_records = {r["item_id"]: r for r in _read(self.paths["checks.jsonl"])}
        self.judge_records = {r["item_id"]: r for r in _read(self.paths["judge.jsonl"])}

    def _save_state(self) -> None:
        _write_json(self.paths["state.json"], self.state)

    def _build_backends(self) -> dict[str, ChatBackend]:
        needed = {s.backend for s in self.cfg.sources if s.type == "llm"}
        if self.cfg.judge.enabled:
            needed.add(self.cfg.judge.backend)
        return {name: OpenAICompatibleBackend(self.cfg.backends[name]) for name in needed}

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

    def _examples(self, task: str, phenomenon: str, k: int = 4) -> list[tuple[str, str]]:
        try:
            data = create_dataset(task, seed=self.cfg.seed, size=_HUGE, phenomena=(phenomenon,))
        except (AssertionError, TypeError, ValueError):
            return []
        return [(data[j]["metadata"]["sentence_good"], data[j]["metadata"]["sentence_bad"]) for j in range(k)]

    def _llm_entry(self, i: int, src: SourceConfig, phenomenon: str, pair: WrittenPair, revision_of: Optional[str] = None):
        info = self.info[src.task][phenomenon]
        return llm_entry(
            pair, task=src.task, phenomenon=phenomenon, norm="iec", formats=src.formats,
            rng=Random(f"{self.cfg.seed}:{pair.good}:{pair.bad}"),
            languagetool_rules=info.get("languagetool_rules", []),
            writer_model=self.backends[src.backend].model, revision_of=revision_of,
        )

    def _make_llm(self, i: int, src: SourceConfig, n: int) -> list[dict[str, Any]]:
        st = self.state["sources"][str(i)]
        n = min(n, src.count - st["requested"])
        phenomena = src.phenomena or list(self.info[src.task])
        out: list[dict[str, Any]] = []
        consecutive_errors = 0
        while n > 0 and consecutive_errors < 3:
            phenomenon = phenomena[(st["requested"] // max(src.batch_size, 1)) % len(phenomena)]
            k = min(src.batch_size, n)
            try:
                pairs = self.writers[i].write(self.info[src.task][phenomenon], self._examples(src.task, phenomenon), k)
                consecutive_errors = 0
            except LLMError as e:
                self.state["writer_errors"] += 1
                consecutive_errors += 1
                self.log(f"LLM writer {i}: {e}")
                continue
            st["requested"] += k
            n -= k
            for pair in pairs:
                added = self._add_candidate(self._llm_entry(i, src, phenomenon, pair), i)
                if added is not None:
                    out.append(added)
        if st["requested"] >= src.count:
            st["exhausted"] = True
        return out

    def _make(self, n: int) -> list[dict[str, Any]]:
        new: list[dict[str, Any]] = []
        for i, src in enumerate(self.cfg.sources):
            if src.type == "llm" and not self.state["sources"][str(i)]["exhausted"]:
                new += self._make_llm(i, src, n)
        remaining = max(n - len(new), 0)
        procedural = [(i, s) for i, s in enumerate(self.cfg.sources) if s.type == "procedural"]
        total_weight = sum(s.weight for _, s in procedural) or 1.0
        for i, src in procedural:
            new += self._make_procedural(i, src, math.ceil(remaining * src.weight / total_weight))
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
            all_results = list(pool.map(lambda e: run_checks(self.checks, e), pending))
        for entry, results in zip(pending, all_results):
            record = self._record_checks(entry, results)
            if not record["passed"] and entry["metadata"]["source"]["type"] == "llm":
                self._fix(entry, record)

    def _fix(self, entry: dict[str, Any], record: dict[str, Any]) -> None:
        """Ask the LLM writer to fix a pair that failed the checks, then check it again."""
        i = entry["metadata"]["source"]["index"]
        src = self.cfg.sources[i]
        phenomenon = entry["metadata"]["phenomenon"]
        current, current_record = entry, record
        for _ in range(src.max_fix_attempts):
            failing = next(r for r in current_record["results"] if not r["passed"])
            problems = [f"{failing['check']}: {json.dumps(failing['details'], ensure_ascii=False)}"]
            meta = current["metadata"]
            pair = WrittenPair(meta["sentence_good"], meta["sentence_bad"],
                               [v for v in meta["correct_variants"] if v != meta["sentence_good"]])
            try:
                fixed = self.writers[i].fix(self.info[src.task][phenomenon], pair, problems)
            except LLMError as e:
                self.log(f"fix attempt failed: {e}")
                return
            candidate = self._add_candidate(self._llm_entry(i, src, phenomenon, fixed, revision_of=meta["item_id"]), i)
            if candidate is None:
                return
            current, current_record = candidate, self._record_checks(candidate, run_checks(self.checks, candidate))
            if current_record["passed"]:
                return

    def _sampled_for_audit(self, iid: str) -> bool:
        return int(iid[:8], 16) / 0xFFFFFFFF < self.cfg.judge.sample_rate

    def _judge_pending(self) -> None:
        if self.judge is None:
            return
        pending = [
            iid for iid, rec in self.check_records.items()
            if rec["passed"] and iid not in self.judge_records
            and (self.cfg.judge.mode == "filter" or self._sampled_for_audit(iid))
        ]
        if not pending:
            return

        def judge(iid: str) -> dict[str, Any]:
            entry = self.candidates[iid]
            meta = entry["metadata"]
            info = self.info[self.cfg.sources[meta["source"]["index"]].task][meta["phenomenon"]]
            try:
                return {"item_id": iid, "verdict": self.judge.judge(entry, info)}
            except LLMError as e:
                return {"item_id": iid, "error": str(e)}

        with ThreadPoolExecutor(max(1, self.cfg.judge.workers)) as pool:
            for record in pool.map(judge, pending):
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
        goal = math.ceil(self.cfg.target_size * self.cfg.pool_multiplier)
        while True:
            self._check_pending()
            self._judge_pending()
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
            new = self._make(math.ceil(shortfall / rate))
            self.state["rounds_done"] += 1
            self._save_state()
            if not new and all(iid in self.check_records for iid in self.candidates):
                self.log("no new candidates")
                break
        return self._finish()

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
            if details is None:
                continue
            meta = self.candidates[iid]["metadata"]
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
        families = lambda models: {model_family(m) for m in models}
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
        for dim, rows in shares["fields"].items():
            for value, row in rows.items():
                if not row["ok"]:
                    warnings.append(f"minimum share not met: {dim}={value} is {row['share']:.1%}, minimum {row['min_share']:.0%}")
        if status == "incomplete":
            warnings.append("dataset is incomplete: too few pairs, or a minimum share not met")

        return {
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
