"""Dataset registry, mirroring reasoning-gym's ``register_dataset`` / ``create_dataset``."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Callable, Optional

from .curriculum import BaseCurriculum
from .dataset import ProceduralDataset


@dataclass(frozen=True)
class _Entry:
    dataset_cls: type[ProceduralDataset]
    config_cls: type
    curriculum_cls: Optional[type[BaseCurriculum]]


DATASETS: dict[str, _Entry] = {}


def register_dataset(
    name: str,
    dataset_cls: type[ProceduralDataset],
    config_cls: type,
    curriculum_cls: Optional[type[BaseCurriculum]] = None,
) -> None:
    if name in DATASETS:
        raise ValueError(f"dataset '{name}' is already registered")
    DATASETS[name] = _Entry(dataset_cls, config_cls, curriculum_cls)


def _get(name: str) -> _Entry:
    try:
        return DATASETS[name]
    except KeyError:
        raise ValueError(f"unknown dataset '{name}'; available: {sorted(DATASETS)}") from None


def create_dataset(name: str, **kwargs: Any) -> ProceduralDataset:
    """Build a dataset from keyword arguments of its config (including ``seed`` and ``size``)."""
    entry = _get(name)
    config = entry.config_cls(**kwargs)
    config.validate()
    return entry.dataset_cls(config)


def create_curriculum(name: str) -> BaseCurriculum:
    entry = _get(name)
    if entry.curriculum_cls is None:
        raise ValueError(f"dataset '{name}' has no curriculum")
    return entry.curriculum_cls()


def get_phenomenon_info(name: str) -> dict[str, dict[str, Any]]:
    """Per-phenomenon info (error, correction, LanguageTool rules, marker) of task ``name``."""
    module = sys.modules[_get(name).dataset_cls.__module__]
    return getattr(module, "PHENOMENON_INFO", {})


def get_score_answer_fn(name: str) -> Callable[[Optional[str], dict[str, Any]], float]:
    """Return a verifier that scores an answer against an entry of dataset ``name``."""
    entry = _get(name)
    scorer = entry.dataset_cls(entry.config_cls(size=1, seed=0))
    return scorer.score_answer
