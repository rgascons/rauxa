# Tooling and Architectural Options for a Catalan Synthetic Dataset Generator and Benchmark (as of Oct 2026)

Note on method: about 15 search/fetch calls. "Verified" means read from a primary source page in this session. Points marked (inferred) or (from prior knowledge, unverified) were not confirmed in this session and should be checked before relying on them.

## 1. Procedural generators (reasoning-gym, DyVal, GSM-Symbolic, LiveBench, grammar/morphology tools)

### Takeaway
reasoning-gym (Apache-2.0, more than 100 tasks) is the strongest base for verifiable procedural generation. Its pattern is a config dataclass, a `ProceduralDataset` subclass seeded per item, a `score_answer` verifier and a registry, and it can be copied directly for Catalan-language tasks. For linguistic (inflection/agreement) tasks, Catalan is covered by the GF Resource Grammar Library and by Apertium's apertium-cat morphological generator (GPL-2.0), which is enough for template surface realisation at scale.

### Cited Findings
**reasoning-gym**
- Apache-2.0. Provides "more than 100" tasks. API: `reasoning_gym.create_dataset('<task>', size=..., seed=..., **config)` returns entries with `question`, `answer`, `metadata`. Each dataset exposes `score_answer(answer, entry)` (1.0 = correct). `get_score_answer_fn` serves RL training. `cascade_score` tries string, then numeric, then symbolic matching. Composite datasets use `DatasetSpec(name, weight, config)`. No GitHub releases are listed, and the PyPI version may lag `main` — [GitHub open-thought/reasoning-gym](https://github.com/open-thought/reasoning-gym)
- Paper: "REASONING GYM: Reasoning Environments for Reinforcement Learning with Verifiable Rewards", arXiv 2505.24760 — [arXiv](https://arxiv.org/abs/2505.24760)
- Adding a task: subclass `ProceduralDataset` (in `reasoning_gym/dataset.py`), write a config class, and return dicts with `question`/`answer`/`metadata`. Override `score_answer()` when several answers are valid; it returns a value in [0,1]. Add unit tests modelled on `tests/test_chain_sum.py` and run pre-commit. The guide says to favour tasks where guessing rarely succeeds and to avoid true/false or multiple-choice formats, which give noisy rewards. Prompts should be answerable by an average human — [CONTRIBUTING.md](https://raw.githubusercontent.com/open-thought/reasoning-gym/main/CONTRIBUTING.md)
- Reference implementation `chain_sum.py`:
  - `ChainSumConfig` is a dataclass with `seed` and `size` (default 500) and a `validate()` method.
  - `__getitem__(idx)` uses `random.Random(self.seed + idx)`, so any item can be reproduced from its index.
  - The module ends with `register_dataset(DATASET_NAME, ChainSumDataset, ChainSumConfig, ChainSumCurriculum)`.
  - `ChainSumCurriculum(BaseCurriculum)` uses `RangeAttributeDefinition` difficulty levels (for example, `num_terms` from 2 to 12).
  - [chain_sum.py](https://raw.githubusercontent.com/open-thought/reasoning-gym/main/reasoning_gym/arithmetic/chain_sum.py)

**Dynamic / contamination-resistant benchmark designs**
- DyVal (ICLR 2024, Microsoft; part of PromptBench) generates samples with directed acyclic graphs at controllable complexity across math, logic and algorithm tasks. Model performance drops as complexity rises — [ICLR 2024 poster](https://iclr.cc/virtual/2024/poster/18134); [Microsoft Research](https://www.microsoft.com/en-us/research/?p=1014714); [arXiv 2309.17167](https://ar5iv.org/html/2309.17167)
- LiveBench releases new questions monthly, drawn from recent arXiv papers, news and datasets. It scores against objective ground truth without LLM judges — as described in [BeyondBench arXiv 2509.24210](https://arxiv.org/pdf/2509.24210)
- GSM-Symbolic is classed in the survey literature as a template-based, rule-generated dynamic benchmark — [survey arXiv 2502.17521](https://arxiv.org/pdf/2502.17521v2). (From prior knowledge, unverified here: Apple 2024, arXiv 2410.05229, turns GSM8K items into templates with variable names and numbers, and reports accuracy variance and drops when numbers or clauses change.)
- GSM1k, a held-out set matched to GSM8K, showed accuracy drops of up to 13% for some model families. This is evidence that static benchmarks overfit — cited in [arXiv 2509.24210](https://arxiv.org/pdf/2509.24210)
- Recent "benchmark generation" work, for example "Fine-Grained Benchmark Generation for Comprehensive Evaluation of Foundation Models" (arXiv 2605.18824, 2026), appeared in search results — [arXiv](https://arxiv.org/pdf/2605.18824). It was not read in detail.

**Grammar / morphology tools for Catalan**
- GF Resource Grammar Library: the synopsis lists "currently 38 languages: Afrikaans, Arabic, Bulgarian, Catalan, …". Catalan examples appear in the synopsis tables (for example, "més tebi" for "warmer") — [RGL synopsis](https://www.grammaticalframework.org/lib/doc/synopsis/index.html). Answer: **yes, the RGL has Catalan (Cat)**. Detailed completeness status is in a separate RGL Status document, which was not fetched. The repository is active, with 7,558 commits on master — [gf-rgl GitHub](https://github.com/GrammaticalFramework/gf-rgl)
- Apertium: apertium-cat supports morphological analysis, morphological generation and POS tagging of Catalan. It requires lttoolbox ≥3.7.1, apertium and vislcg3, and is GPL-2.0. apertium-spa-cat (es↔ca) is GPL-2+ per Debian. lttoolbox compiles the same monodix as either an analyser or a generator — [search summary, Apertium/Debian metadata](https://metadata.ftp-master.debian.org/changelogs/main/a/apertium-spa-cat/stable_copyright); [codegraph apertium-spa-cat](https://codegraph.jelmer.uk/apertium-spa-cat/2.2.0-3/)
- FreeLing was LGPL in 2004 — [Carreras et al. 2004](https://nlp.lsi.upc.edu/papers/carreras04.pdf). Its current license was not verified. (From prior knowledge, unverified: FreeLing 4.x is AGPL-3.0 and ships Catalan dictionaries for analysis and generation.)

### Inferences
- Recommended architecture: a reasoning-gym-style package of Catalan task classes. Each class would take a config dataclass, use per-item `Random(seed+idx)`, produce a verifiable answer computed in code and a `score_answer` verifier, and record template and lexicon IDs in `metadata`. Tens of thousands of items is trivial at this scale. Fresh seeds give regenerable private splits.
- Templates should be realised with explicit morphology rather than string concatenation, to handle Catalan features such as gender/number agreement, elision (l', d'), contractions (al, del, pel), apostrophisation and weak pronouns. Options: (a) query an Apertium apertium-cat generator; (b) use GF RGL Cat for full sentence linearisation with agreement guaranteed; (c) use a hand-curated inflection lexicon. Note that GPL tools used offline to *produce data* do not normally impose GPL on the data, but check this.
- reasoning-gym's advice against MCQ applies to RL rewards. For a benchmark, MCQ is still acceptable when it is scored by loglikelihood.
- No SimpleNLG Catalan port was verified in this session.

### Gaps
- GF RGL Catalan completeness level and maintainer status (Status document not fetched).
- SimpleNLG-ES / any Catalan SimpleNLG variant: not searched successfully.
- Current FreeLing license and version.
- PromptBench/DyVal maintenance status in 2026; Dynabench status not checked.
- Whether reasoning-gym has any non-English tasks or i18n hooks (none seen; inferred to be English-only).

## 2. LLM-based generation frameworks (distilabel, YourBench, DataDreamer, Curator, NeMo Data Designer, Magpie, persona-hub)

### Takeaway
Every mature framework is model-agnostic and language-agnostic, so Catalan support depends on the generator LLM rather than the framework. YourBench (Apache-2.0, document→QA, exports to lighteval) is the best fit for grounded QA from Catalan source documents. distilabel and NeMo Data Designer (both Apache-2.0) are the best fit for general pipelines with judges and dedup. distilabel's original authors have left the project.

### Cited Findings
- **YourBench** (Hugging Face) generates single-hop and multi-hop QA from PDF/Word/HTML/text documents.
  - Pipeline: ingestion → summarization → chunking → question generation → `prepare_lighteval`.
  - Providers: OpenAI-compatible endpoints (vLLM, Ollama), OpenAI and HF providers. Apache-2.0.
  - Paper: arXiv 2504.01833 (COLM 2025). No release tags.
  - Multilingual support is not stated as a feature, though language can be specified in the brief.
  - [GitHub huggingface/yourbench](https://github.com/huggingface/yourbench)
- **distilabel** (Argilla/HF) is Apache-2.0.
  - Features: Python `Pipeline` with Ray scaling, many providers (Anthropic, OpenAI, Mistral, Vertex, Ollama, LiteLLM, vLLM, MLX), structured generation (outlines, Instructor), clustering, MinHash dedup and push to the Hub.
  - The README says the original authors "have moved on to other projects" and community collaborators are "working towards the next release". The project is maintained, but at reduced velocity.
  - [GitHub argilla-io/distilabel](https://github.com/argilla-io/distilabel)
- **NVIDIA NeMo Data Designer**:
  - The open-source library is Apache-2.0 (per a third-party guide). It combines statistical samplers with LLM columns.
  - The NGC microservice falls under the NVIDIA Software License and needs NVIDIA AI Enterprise for production. It originated from the Gretel acquisition (March 2025).
  - Docs show v0.9.1, plus 2026 dev notes (a May 2026 plugin framework and a June 2026 Nemotron-Personas pipeline).
  - [NVIDIA docs deployment options](https://docs.nvidia.com/nemo/datadesigner/v0.9.1/concepts/deployment-options.md); [NGC](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/nemo-microservices/containers/data-designer); [OpenRouter blog](https://openrouter.ai/blog/tutorials/distillable-models-and-synthetic-data-pipelines-with-nemo-data-designer/)
- **Bespoke Curator**: `pip install bespokelabs-curator`. Apache-2.0 per an aggregator; the LICENSE file was not verified — [Bespoke docs](https://docs.bespokelabs.ai/bespoke-curator); [rl-list](https://www.rl-list.com/vendors/bespoke-labs)

### Inferences
- From prior knowledge, unverified in this session:
  - DataDreamer (Patel et al., ACL 2024; MIT) focuses on reproducibility and caching.
  - Magpie (arXiv 2406.08464) extracts instructions by prompting aligned models with only the chat-template prefix. For a non-English language this needs a system prompt or prefix that forces the target language, otherwise the output skews heavily English.
  - Persona Hub (Tencent, arXiv 2406.20094) has 1B English personas. These can be translated or localised to Catalan contexts, or Nemotron-Personas-style local personas can be built.
- For Catalan, the generator LLM matters most. Strong frontier models write fluent Catalan but tend toward Spanish calques and Castilianisms. Local open models such as BSC's Salamandra/ALIA were not checked here. Plan for LID and LanguageTool filters (section 4).
- Suggested hybrid: generate procedural skeletons with gold answers in code, then use an LLM only for paraphrase or "story wrapping", and verify that the gold answer is preserved by re-solving or by an invariant check. This is similar to the GSM-Symbolic style but with LLM-based surface diversity.

### Gaps
- Current versions and release dates for distilabel, Curator, DataDreamer and YourBench (no GitHub releases listed).
- Explicit multilingual or Catalan evaluations of any of these frameworks: none found.

## 3. Benchmark harnesses (lm-evaluation-harness, Inspect AI, lighteval, OpenCompass), HF publishing, leaderboards

### Takeaway
lm-evaluation-harness is the natural home: CatalanBench already lives there as `catalan_bench`, and new tasks are YAML files that point to an HF dataset. lighteval offers language-adapted multilingual templates (FineTasks methodology). Inspect AI (MIT) suits generative or agentic tasks that need custom scorers.

### Cited Findings
- **lm-eval task YAML** fields:
  - `dataset_path` (HF Hub ID), `dataset_name`, `test_split` (plus training/validation/fewshot splits).
  - `doc_to_text`, `doc_to_target` and `doc_to_choice` take Jinja2, a string or a function.
  - `output_type` is one of `generate_until` (default), `loglikelihood`, `loglikelihood_rolling` or `multiple_choice`. For multiple choice, `doc_to_target` is an index into the choices.
  - `generation_kwargs`; `filter_list` (regex, take_first, majority_vote); `metric_list` (with `aggregation` and `higher_is_better`); `include` for a base YAML; `!function` for Python.
  - `tag`, and group YAMLs with `aggregate_metric_list`.
  - [task_guide.md](https://raw.githubusercontent.com/EleutherAI/lm-evaluation-harness/main/docs/task_guide.md)
- **catalan_bench** in lm-eval includes:
  - ARC: arc_ca_challenge/easy.
  - Reading/QA: belebele_cat_Latn, catalanqa, coqcat, xquad_ca, openbookqa_ca, terretaqa.
  - Summarisation: cabreu.
  - Acceptability/NLI: catcola, teca, wnli_ca, xnli_ca/xnli_va.
  - Commonsense/paraphrase: copa_ca, piqa_ca, siqa_ca, xstorycloze_ca, parafraseja, paws_ca, phrases_ca, cocoteros_va.
  - Math: mgsm_direct_ca. Translation: flores_ca (16 directions). Other: cieacova.
  - Truthfulness: truthfulqa_va, veritasqa_gen_ca / veritasqa_mc1_ca / veritasqa_mc2_ca.
  - Shared configs in `_arc_ca_common_yaml`, `_cabreu_common_yaml` and `utils.py`. Groups: `catalan_bench` and `flores_ca`. Changelog v2.0–2.5.
  - The citation is IberoBench (Baucells et al., COLING 2025); the CatalanBench paper is listed as "coming soon".
  - [lm-eval catalan_bench](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/lm_eval/tasks/catalan_bench)
- **lighteval multilingual**: prompts are built from translation literals (about 19 keywords such as yes/no/because, plus punctuation) and language-adapted templates. Tasks are parametrised via `LightevalTaskConfig` per language and formulation. Adding a language means editing the translation-literals file — [lighteval docs: contributing to multilingual evaluations](https://huggingface.co/docs/lighteval/main/contributing-to-multilingual-evaluations)
- **FineTasks**: a task-selection method applied to 9 languages. It compares Cloze Formulation (choices not in prompt) with Multi-Choice Formulation (choices in prompt, model selects letter) and includes a guide for extending to new languages — [FineTasks blog](https://huggingfacefw-blogpost-fine-tasks.hf.space/) (full text not retrieved). Whether a Catalan literals entry exists was not verified.
- **Inspect AI**: MIT-licensed, Python ≥3.10, from UK AISI and developed with Meridian Labs. The repo is under UKGovernmentBEIS — [CASRAI guide](https://casrai.org/guides/inspect-ai-evaluation-framework-uk-aisi); [EC OSOR](https://interoperable-europe.ec.europa.eu/collection/open-source-observatory-osor/news/inspect-uk-government-backed-open-source-ai-testing-platform)

### Inferences
- Suggested packaging:
  - Publish the HF dataset with public `test` (and `fewshot`/`validation`) splits plus per-task configs.
  - Ship lm-eval YAMLs: `multiple_choice` for discriminative linguistic items, `generate_until` plus a regex `filter_list` plus `exact_match` for verifiable-answer items.
  - Run locally via `--include_path`, then upstream a PR into `lm_eval/tasks/catalan_bench` or a sibling folder.
  - Optionally add a lighteval task and an Inspect task, reusing the same verifier code (`!function utils.process_results` in lm-eval; a custom scorer in Inspect).
  - (`--include_path` is from prior knowledge; it was not shown on the fetched page.)
- Following FineTasks, report both MCF and CF formulations, and check monotonicity and signal against random baselines before releasing.
- OpenCompass (Apache-2.0, Shanghai AI Lab) was not researched here. It is lower priority for a Catalan/European audience (inferred).
- Leaderboard options (from prior knowledge, unverified): an HF Space using the leaderboard template or the "Open LLM Leaderboard"-style backend, or results tables in the dataset card. The Open LLM Leaderboard was retired in 2025. Re-verify before citing.

### Gaps
- The full YAML of an existing catalan_bench task (not displayed). The exact output_type per task is unverified.
- OpenCompass status; HF leaderboard template status in 2026.
- Whether lighteval has Catalan literals.

## 4. Quality control (LanguageTool, LID, dedup, LLM-as-judge, human sampling)

### Takeaway
LanguageTool (core LGPL-2.1) runs as a local HTTP server with `ca-ES` and `ca-ES-valencia` support and can serve as an automatic grammar filter. However, some rules exist only in the cloud version. For ca/es leakage, use GlotLID (Apache-2.0, about 2,000+ labels) or OpenLID-v3 (GPL-3.0, focused on closely related languages), in addition to fastText lid.176.

### Cited Findings
- **LanguageTool**: the core is LGPL-2.1 and can be used as a library without open-sourcing the application. Dictionaries may carry other licenses (e.g., GPL) — [credativ](https://www.credativ.de/en/wp-json/wp/v2/posts/13292)
- LanguageTool ships an embedded HTTP server — [FreshPorts](https://www.freshports.org/textproc/languagetool/). The `language_tool_python` package downloads and runs the local server jar and accepts `'ca-ES'` — [PyPI](https://pypi.org/project/language-tool-python/2.7.1)
- Catalan is supported, and several 2023 releases added Catalan rules — [LibreOffice extension page](https://extensions.libreoffice.org/fr/extensions/show/languagetool)
- Some rules run only on the cloud endpoint and are not part of the open-source version — [LanguageTool forum](https://forum.languagetool.org/t/locally-server-behaves-different-from-browser-extension/8511)
- **GlotLID** (cis-lmu/glotlid, arXiv 2310.16248): Apache-2.0 per the CHALIS paper; labels look like `__label__eng_Latn`.
  - It covers about 2,000+ languages but uses some training data under more restrictive licenses.
  - [CHALIS arXiv 2606.06088](https://arxiv.org/pdf/2606.06088); [OpenLID-v3 arXiv 2602.13139](https://arxiv.org/pdf/2602.13139); [GlotLID paper](https://arxiv.org/abs/2310.16248)
- **OpenLID-v3** (HPLT, 2026) targets precision on closely related languages. It lists Catalan and is GPL-3.0 — [HF HPLT/OpenLID-v3](https://huggingface.co/HPLT/OpenLID-v3/blob/main/README.md); [arXiv 2602.13139](https://arxiv.org/pdf/2602.13139)
- **fastText lid.176** — [fasttext.cc](https://fasttext.cc/docs/en/language-identification). (From prior knowledge: CC-BY-SA-3.0, includes `ca`.)
- **Dedup**: distilabel includes MinHash dedup — [distilabel](https://github.com/argilla-io/distilabel)

### Inferences
- Suggested QC pipeline:
  1. Run sentence-level LID with GlotLID or OpenLID-v3, and flag items where the `spa_Latn` probability exceeds a threshold or the `cat_Latn` top-1 confidence is low.
  2. Run a local LanguageTool server with Catalan (or Valencian) rules, and reject or repair items with grammar/spelling matches. Whitelist rule IDs that are known to give false positives on templated text.
  3. Run MinHash/n-gram near-dup detection within the set and against public Catalan benchmarks.
  4. Use an LLM judge only for fluency/naturalness and ambiguity. The verifier, not the judge, decides correctness.
  5. Validate a human stratified sample per task/template, for example a few hundred items with two native annotators and agreement reported.
- Sentence-level LID on short items is noisy (inferred). Combine it with lexical Castilianism lists, for example from Softcatalà resources (unverified).
- LLM-judge biases are known from prior literature: position, verbosity and self-preference. Mitigate with swapped-order evaluation, judges from a different model family and calibration against human labels. No specific source was fetched this session.

### Gaps
- Softcatalà-specific LanguageTool rule licensing; the size of the Catalan rule set.
- The GlotLID model card was not read directly. Its Catalan precision against Spanish/Occitan was not verified.
- Literature on LLM-judge bias was not fetched.

## 5. Contamination-resistant design and licensing of LLM-generated data

### Takeaway
Procedural generation with per-item seeds lets you keep a private held-out split and regenerate fresh, isomorphic test sets at each version, as in DyVal, GSM-Symbolic and LiveBench. Combine this with canary strings and versioned releases. For LLM-generated content, check each provider's output-use terms. The most permissive route is open-weight models under Apache/MIT.

### Cited Findings
- Dynamic generation (DyVal graphs) and periodic refresh (LiveBench monthly, objective scoring) are established answers to contamination — [ICLR DyVal](https://iclr.cc/virtual/2024/poster/18134); [arXiv 2509.24210](https://arxiv.org/pdf/2509.24210)
- Static benchmarks show overfitting: GSM1k drops of up to 13% — [arXiv 2509.24210](https://arxiv.org/pdf/2509.24210)
- reasoning-gym's `Random(seed+idx)` makes any split reproducible from (seed, size, config) — [chain_sum.py](https://raw.githubusercontent.com/open-thought/reasoning-gym/main/reasoning_gym/arithmetic/chain_sum.py)

### Inferences
- Design suggestions:
  - Publish the generator code and a public dev/test split generated with seed A.
  - Keep a private split generated with an undisclosed seed and/or held-out templates and lexicon entries, which gives stronger resistance than seed alone.
  - Release a new versioned split (v1, v2…) on a schedule, and report results per version.
  - Embed a canary GUID in the dataset card and files (BIG-bench convention, from prior knowledge) so crawlers and trainers can filter it.
  - Gate the HF dataset (gated access, no-redistribution terms) to reduce scraping.
  - Measure contamination with n-gram overlap and with performance gaps between the public and private splits.
- Licensing (prior knowledge, unverified in this session): OpenAI, Anthropic and Google terms historically restrict using outputs to develop *competing* models. Evaluation use is generally less restricted. Llama-licensed outputs carry naming and attribution conditions. Apache-2.0/MIT open-weight models (e.g., Qwen, Mistral and OLMo variants; BSC Salamandra is Apache-2.0 per prior knowledge) impose no output restrictions. Document the generator model and terms in the dataset card, and choose a data license such as CC-BY-4.0 accordingly. Re-check current ToS text before publication.

### Gaps
- No primary provider ToS pages were fetched. Current (Oct 2026) output-use clauses are unverified.
- No primary source was fetched for the canary-string convention or for HF gated-dataset mechanics.
