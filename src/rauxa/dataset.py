"""Base class for procedural datasets, mirroring reasoning-gym's ``ProceduralDataset``.

Every item is a pure function of ``(config, seed, idx)``: ``__getitem__`` builds it with
``random.Random(seed + idx)``, so any item can be regenerated from its index and a split
of any size is reproducible from ``(config, seed, size)``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sized
from random import Random
from typing import Any, Optional


class ProceduralDataset(ABC, Sized, Iterable[dict[str, Any]]):
    def __init__(self, config: Any, seed: Optional[int] = None, size: int = 500):
        if hasattr(config, "validate"):
            config.validate()
        self.config = config
        self.size = size
        self.seed = seed if seed is not None else Random().randint(0, 2**32)

    def __len__(self) -> int:
        return self.size

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for idx in range(self.size):
            yield self[idx]

    @abstractmethod
    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return ``{"question": str, "answer": str, "metadata": dict}`` for item ``idx``."""

    def score_answer(self, answer: Optional[str], entry: dict[str, Any]) -> float:
        """Default verifier: 1.0 for an exact match, 0.0 otherwise."""
        if answer is None:
            return 0.0
        return 1.0 if answer.strip() == entry["answer"] else 0.0
