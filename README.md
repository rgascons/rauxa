# rauxa

Procedural generator of verifiable Catalan language tasks, built to produce benchmarks (and
training data) that target **normative Catalan**: Spanish calques, elision and contraction,
weak pronouns, diacritics... Items are built from templates and rules, never by asking an LLM
for the right answer, so the gold answer is correct by construction.

The interface copies [reasoning-gym](https://github.com/open-thought/reasoning-gym), so tasks can
later be plugged into it through an adapter.

```python
import rauxa

data = rauxa.create_dataset("castellanismes_verbals", size=10_000, seed=42, formats=("mcq", "gec"))
entry = data[0]
entry["question"]   # prompt in Catalan
entry["answer"]     # gold answer
entry["metadata"]   # phenomenon, tense, sentence_good / sentence_bad, accepted variants...
data.score_answer("A", entry)  # 0.0 or 1.0
```

## How it maps to reasoning-gym

| reasoning-gym | rauxa |
|---|---|
| Config dataclass with `validate()`, `seed`, `size` | Same (`CastellanismesVerbalsConfig`, ...) |
| `ProceduralDataset`, item `idx` built with `Random(seed + idx)` | Same: any item is regenerable from its index |
| Item `{question, answer, metadata}` | Same; metadata always carries `sentence_good` / `sentence_bad` |
| Answer computed in code | Correct sentence realised by `rauxa.catalan` (conjugation, elision, contractions) |
| `score_answer` verifier | Per-format verifiers (`rauxa.formats.score`) |
| `register_dataset` / `create_dataset` / `get_score_answer_fn` | Same names |
| `BaseCurriculum` | Simplified version with scalar attribute levels |

What is specific to rauxa:

- **Error transformations**: each task builds the incorrect sentence from the same choices as the
  correct one, applying the error real models make.
- **One pair, several formats**: `mcq` (pick the correct sentence), `judgement` (is it correct?
  sí/no) and `gec` (correct it). Every dataset also exports as BLiMP-style minimal pairs.
- **LanguageTool QC**: the correct sentence must get no LanguageTool match; the incorrect one
  should get at least one.

## Tasks

| Task | Phenomena |
|---|---|
| `castellanismes_verbals` | `tenir_que` (→ *haver de*), `hi_ha_que` (→ *cal* / *s'ha de*), `donar_se_compte` (→ *adonar-se*) |

Only the IEC (central) norm is implemented; `norm="avl"` raises `NotImplementedError`.

## Command line

No dependencies are needed to run from a checkout:

```bash
export PYTHONPATH=src
python3 -m rauxa list
python3 -m rauxa generate castellanismes_verbals --size 50000 --seed 1 --formats mcq,judgement,gec -o out/cv.jsonl
python3 -m rauxa qc out/cv.jsonl --lt-url http://localhost:8081 -o out/cv.clean.jsonl --report out/cv.qc.jsonl
python3 -m rauxa export-lm-eval out/cv.clean.jsonl out/lm_eval --name rauxa_cv
python3 scripts/preview_pairs.py castellanismes_verbals 30   # table for human review
```

Then run the benchmark with lm-evaluation-harness:

```bash
lm_eval --model hf --model_args pretrained=BSC-LT/salamandra-7b --include_path out/lm_eval --tasks rauxa_cv_pairs,rauxa_cv_mcq
```

## Tests

```bash
PYTHONPATH=src python3 -m unittest discover -s tests
```

## Layout

```
src/rauxa/
  dataset.py, factory.py, curriculum.py   reasoning-gym style core
  formats.py                              prompt formats and verifiers
  catalan/orthography.py                  elision, articles, contractions, clitics
  catalan/conjugation.py                  small verb paradigms
  tasks/castellanismes_verbals.py         first task
  qc/languagetool.py                      LanguageTool HTTP client
  export.py                               JSONL and lm-eval export
  cli.py
```
