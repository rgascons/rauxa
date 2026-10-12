"""Run configuration, loaded from TOML (see ``configs/`` for examples)."""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional


@dataclass
class BackendConfig:
    """Where LLM calls go.

    - ``openai``: an OpenAI-compatible chat completions endpoint (OpenAI, vLLM, Ollama...).
    - ``agent``: files answered by a coding agent (Claude Code, Codex...). The run stops
      with status "waiting" and resumes once the requests are answered. See AGENT_PIPELINE.md.
    """

    type: Literal["openai", "agent"] = "openai"
    base_url: str = ""
    model: str = ""  # for "agent", the model the agent runs on (recorded, used for family warnings)
    api_key_env: Optional[str] = None  # name of the env var holding the key; None for local servers
    temperature: float = 0.7
    json_mode: bool = True  # send response_format={"type": "json_object"}
    timeout: float = 120.0
    max_retries: int = 3


@dataclass
class SourceConfig:
    """Where candidate pairs come from: the procedural generator, or an LLM writing them."""

    type: Literal["procedural", "llm"]
    task: str
    weight: float = 1.0  # share of procedural candidates
    params: dict[str, Any] = field(default_factory=dict)  # dataset config kwargs (procedural)
    # llm only
    backend: str = "default"
    count: int = 0  # pairs to request in total
    batch_size: int = 20
    max_fix_attempts: int = 2  # times the LLM may fix a pair that failed the checks
    phenomena: list[str] = field(default_factory=list)  # default: all of the task's phenomena
    formats: list[str] = field(default_factory=lambda: ["mcq", "judgement", "gec"])


@dataclass
class LanguageToolCheckConfig:
    enabled: bool = True
    url: str = "http://localhost:8081"
    language: str = "ca-ES"
    disabled_rules: list[str] = field(default_factory=list)
    # Reject pairs whose incorrect sentence LanguageTool does not flag. Off by default:
    # LanguageTool misses some errors, and those pairs are often the most interesting.
    require_bad_flagged: bool = False
    workers: int = 8


@dataclass
class ChecksConfig:
    orthography: bool = True
    marker: bool = True
    minimal_pair: bool = True
    max_edit_tokens: int = 8
    languagetool: LanguageToolCheckConfig = field(default_factory=LanguageToolCheckConfig)


@dataclass
class JudgeConfig:
    enabled: bool = False
    backend: str = "default"
    # "filter": every pair that passed the checks is judged, and dropped on a negative verdict.
    # "audit": a random sample is judged to estimate quality; verdicts drop nothing.
    mode: Literal["filter", "audit"] = "filter"
    sample_rate: float = 0.05  # audit mode only
    # What to do with a pair when the judge call fails (e.g. the endpoint is down).
    on_error: Literal["reject", "keep"] = "reject"
    # Only judge pairs from these source types (e.g. ["llm"]); empty means all.
    source_types: list[str] = field(default_factory=list)
    workers: int = 4


@dataclass
class RunConfig:
    name: str
    target_size: int
    seed: int = 0
    output_dir: str = "out/runs"
    pool_multiplier: float = 1.3  # kept-pool size to aim for, as a multiple of target_size
    max_rounds: int = 5  # make → check → judge rounds before giving up
    max_draws_per_round: int = 200_000  # bound on procedural indices tried per round
    sources: list[SourceConfig] = field(default_factory=list)
    checks: ChecksConfig = field(default_factory=ChecksConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    backends: dict[str, BackendConfig] = field(default_factory=dict)
    # Minimum share of the dataset per metadata field and value, e.g.
    # {"phenomenon": {"tenir_que": 0.25}, "format": {"gec": 0.2}}.
    min_share: dict[str, dict[str, float]] = field(default_factory=dict)
    # Models you will evaluate on this data; used to warn about same-family bias.
    evaluated_models: list[str] = field(default_factory=list)

    def validate(self) -> None:
        assert self.target_size > 0, "target_size must be positive"
        assert self.sources, "at least one source is required"
        assert self.pool_multiplier >= 1.0, "pool_multiplier must be >= 1"
        needed = {s.backend for s in self.sources if s.type == "llm"}
        if self.judge.enabled:
            needed.add(self.judge.backend)
        missing = needed - set(self.backends)
        assert not missing, f"backends not configured: {sorted(missing)}"
        for name in needed:
            backend = self.backends[name]
            if backend.type == "openai":
                assert backend.base_url and backend.model, f"backend '{name}' needs base_url and model"
        for dim, shares in self.min_share.items():
            assert sum(shares.values()) <= 1.0 + 1e-9, f"min_share for '{dim}' sums to more than 1"

    def fingerprint(self) -> str:
        """Hash of the configuration, used to refuse resuming a run with a different config."""
        blob = json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def run_dir(self) -> Path:
        return Path(self.output_dir) / self.name


def load_config(path: str | Path) -> RunConfig:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    run = raw.get("run", {})
    checks_raw = dict(raw.get("checks", {}))
    lt = LanguageToolCheckConfig(**checks_raw.pop("languagetool", {}))
    config = RunConfig(
        **run,
        sources=[SourceConfig(**s) for s in raw.get("sources", [])],
        checks=ChecksConfig(**checks_raw, languagetool=lt),
        judge=JudgeConfig(**raw.get("judge", {})),
        backends={name: BackendConfig(**b) for name, b in raw.get("backends", {}).items()},
        min_share=raw.get("min_share", {}),
    )
    config.validate()
    return config
