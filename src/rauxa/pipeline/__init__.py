"""Generation pipeline: makes candidate pairs, runs automatic checks and an optional LLM judge,
and selects the dataset. LLMs only suggest pairs or give opinions; the code decides what is
kept. See ``controller.py`` for the steps and the run directory layout.
"""

from .config import RunConfig, load_config
from .controller import Pipeline, PipelineError, RunResult

__all__ = ["Pipeline", "PipelineError", "RunConfig", "RunResult", "load_config"]
