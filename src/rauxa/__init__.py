"""rauxa: procedural generator of verifiable Catalan language tasks.

The public interface mirrors reasoning-gym::

    import rauxa
    data = rauxa.create_dataset("castellanismes_verbals", size=10, seed=42)
    for entry in data:
        print(entry["question"], entry["answer"])
        assert data.score_answer(entry["answer"], entry) == 1.0
"""

from . import tasks  # noqa: F401  (registers datasets)
from .curriculum import BaseCurriculum, ScalarAttributeDefinition
from .dataset import ProceduralDataset
from .factory import DATASETS, create_curriculum, create_dataset, get_score_answer_fn, register_dataset

__version__ = "0.1.0"

__all__ = [
    "DATASETS",
    "BaseCurriculum",
    "ProceduralDataset",
    "ScalarAttributeDefinition",
    "create_curriculum",
    "create_dataset",
    "get_score_answer_fn",
    "register_dataset",
]
