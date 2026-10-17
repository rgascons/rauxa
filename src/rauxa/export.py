"""Export generated items: JSONL, and lm-evaluation-harness task configs.

The lm-eval export writes up to three tasks from one JSONL of items:

- ``<name>_pairs``: BLiMP-style minimal pairs scored by log-likelihood, from every ``pair``
  item whatever its format. When items have a ``context``, it is the prompt both sentences
  are scored after.
- ``<name>_preference``: the same for ``preference`` items (formal-register preferences), kept
  apart so a preference is never scored as a correctness question.
- ``<name>_mcq``: the ``mcq`` items, scored by log-likelihood of the option letter.

``control`` items (one valid sentence, judgement or gec format) have no counterpart to score
against and are not exported to lm-eval; they stay in the dataset JSONL.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def write_jsonl(entries: Iterable[dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


_PAIRS_YAML = """\
task: {name}_{suffix}
dataset_path: json
dataset_kwargs:
  data_files:
    test: {data}
test_split: test
output_type: multiple_choice
doc_to_text: {doc_to_text}
doc_to_target: 0
doc_to_choice: "{{{{[sentence_good, sentence_bad]}}}}"
should_decontaminate: true
doc_to_decontamination_query: "{{{{sentence_good}}}} {{{{sentence_bad}}}}"
metric_list:
  - metric: acc
metadata:
  version: 1.0
"""

_MCQ_YAML = """\
task: {name}_mcq
dataset_path: json
dataset_kwargs:
  data_files:
    test: {data}
test_split: test
output_type: multiple_choice
doc_to_text: "{{{{question}}}}\\nResposta:"
doc_to_choice: letters
doc_to_target: target_index
metric_list:
  - metric: acc
metadata:
  version: 1.0
"""


def export_lm_eval(entries: list[dict[str, Any]], out_dir: Path, name: str) -> list[Path]:
    """Write data files and task YAMLs for lm-evaluation-harness into ``out_dir``."""
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    def flat(entry: dict[str, Any], with_context: bool = False) -> dict[str, Any]:
        meta = entry["metadata"]
        row = {
            "id": f"{meta['source_dataset']}-{meta['source_index']}",
            "phenomenon": meta.get("phenomenon"),
            "sentence_good": meta["sentence_good"],
            "sentence_bad": meta["sentence_bad"],
        }
        # Extra fields only when present, so existing exports stay byte-identical.
        for key in ("kind", "category", "assessment"):
            if meta.get(key) is not None and not (key == "kind" and meta[key] == "pair"):
                row[key] = meta[key]
        if with_context:
            row["context"] = meta.get("context") or ""
        return row

    written: list[Path] = []
    for kind, suffix in (("pair", "pairs"), ("preference", "preference")):
        items = [e for e in entries if e["metadata"].get("kind", "pair") == kind]
        if not items and kind != "pair":
            continue
        with_context = any(e["metadata"].get("context") for e in items)
        data = out_dir / f"{name}_{suffix}.jsonl"
        write_jsonl((flat(e, with_context) for e in items), data)
        yaml = out_dir / f"{name}_{suffix}.yaml"
        doc_to_text = '"{{context}}"' if with_context else '""'
        yaml.write_text(_PAIRS_YAML.format(name=name, suffix=suffix, data=data, doc_to_text=doc_to_text), encoding="utf-8")
        written += [data, yaml]

    mcq = [e for e in entries if e["metadata"]["format"] == "mcq"]
    if mcq:
        mcq_data = out_dir / f"{name}_mcq.jsonl"
        write_jsonl(
            (
                {
                    **flat(e),
                    "question": e["question"],
                    "letters": [chr(ord("A") + i) for i in range(len(e["metadata"]["options"]))],
                    "target_index": e["metadata"]["target_index"],
                }
                for e in mcq
            ),
            mcq_data,
        )
        mcq_yaml = out_dir / f"{name}_mcq.yaml"
        mcq_yaml.write_text(_MCQ_YAML.format(name=name, data=mcq_data), encoding="utf-8")
        written += [mcq_data, mcq_yaml]
    return written
