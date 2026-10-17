# Run configuration reference

`rauxa run CONFIG.toml` reads one TOML file that describes the whole run: where pairs come from,
how they are checked, whether an LLM judges them, and how large and balanced the dataset must be.
This page lists every key. For a guided first run see [GETTING_STARTED.md](GETTING_STARTED.md);
for working examples see [configs/](configs/).

Unknown keys are an error: the run stops with a `TypeError: ... unexpected keyword argument
'<key>'`, so a typo never silently falls back to a default.

- [How a run uses the config](#how-a-run-uses-the-config)
- [`[run]`](#run)
- [`[[sources]]`](#sources)
- [`[checks]`](#checks)
- [`[judge]`](#judge)
- [`[backends.<name>]`](#backendsname)
- [`[copilot]`](#copilot)
- [`[min_share.<field>]`](#min_sharefield)
- [Changing a config and `--resume`](#changing-a-config-and---resume)

## How a run uses the config

The pipeline works in rounds:

1. **LLM sources write first.** Each one asks for everything left of its `count`, whatever its
   position in the file. Their budget is fixed, so they go first and the free procedural
   generator makes up the difference afterwards. Usually the whole `count` goes out in round 0,
   and later rounds only retry what failed calls left over.
2. **Automatic checks** run on every new pair. An LLM-written pair that fails gets up to
   `max_fix_attempts` rewrites by its writer.
3. **The judge** (if enabled) looks at the pairs that passed the checks, for the source types in
   `judge.source_types`.
4. **The controller counts the kept pool.** The goal is `target_size × pool_multiplier` kept pairs
   *and* every `min_share` reachable. If either is short, the procedural sources make more pairs
   (split by `weight`) and the next round starts.
5. The run stops when the goal is met, after `max_rounds` rounds, or when no source can make
   anything new. Then `target_size` pairs are **selected** from the pool: first enough to cover
   each minimum share, then the rest at random.

The result is `dataset.jsonl` if the target size and every minimum share were met, and
`dataset.incomplete.jsonl` otherwise.

| | Procedural source | LLM source |
|---|---|---|
| Cost | Free, local | API calls |
| How many pairs | As many as needed to fill the pool | `count` requested in total (the writer may return more) |
| When it runs | After the LLM sources, in every round where the pool is short | At the start of each round, until all of `count` has been requested. Normally that all happens in round 0; later rounds only pick up what failed calls (or unanswered agent requests) left over |
| Fix attempts after a failed check | No (dropped) | Yes, `max_fix_attempts` |
| Judged | With `judge.enabled = true`: if `judge.source_types` is empty (the default) or contains `"procedural"` | With `judge.enabled = true`: if `judge.source_types` is empty (the default) or contains `"llm"` |

## `[run]`

| Key | Default | Meaning |
|---|---|---|
| `name` | *required* | Run folder: `<output_dir>/<name>/` |
| `target_size` | *required* | Number of pairs in the final dataset |
| `seed` | `0` | Seeds the procedural generator, the format of each LLM pair and the final selection. Same config and seed, same procedural pairs |
| `output_dir` | `"out/runs"` | Where run folders go |
| `pool_multiplier` | `1.3` | Kept pairs to aim for before selecting, as a multiple of `target_size` (must be ≥ 1). The spare pairs give selection room to meet the minimum shares and to draw at random |
| `max_rounds` | `5` | Make → check → judge rounds before giving up and writing what there is |
| `max_draws_per_round` | `200000` | How many procedural indices a source may try in one round without finding a new unique pair before it is marked exhausted (out of combinations) |
| `evaluated_models` | `[]` | Models you will evaluate on this data. Used only for the manifest warning when they share a model family with the writer or judge |

## `[[sources]]`

One `[[sources]]` table per source; a run needs at least one. Sources are numbered from 0 in the
order they appear. The number goes into each item's `metadata.source.index` and into the
procedural seed, so **reordering sources changes the procedural pairs**.

Keys for both types:

| Key | Default | Meaning |
|---|---|---|
| `type` | *required* | `"procedural"` or `"llm"` |
| `task` | *required* | Task name, e.g. `"castellanismes_verbals"` (`rauxa list` shows them) |

### Procedural sources (`type = "procedural"`)

| Key | Default | Meaning |
|---|---|---|
| `weight` | `1.0` | With several procedural sources, the share of the missing pairs each one makes, proportional to its weight |
| `[sources.params]` | `{}` | Options for the task's generator (below). Don't set `seed` or `size`: the pipeline sets them |

`count`, `backend`, `batch_size`, `max_fix_attempts`, `phenomena` and `formats` at the top level
of a procedural source are **accepted but ignored**. The generator's own phenomena and formats go
in `[sources.params]`.

`[sources.params]` for `castellanismes_verbals`:

| Key | Default | Values |
|---|---|---|
| `phenomena` | all | `tenir_que`, `hi_ha_que`, `donar_se_compte` |
| `tenses` | `["present", "future", "periphrastic_past", "perfect"]` | `present`, `imperfect`, `future`, `conditional`, `periphrastic_past`, `perfect`. `donar_se_compte` is only generated in `present`, `future`, `periphrastic_past` and `perfect` |
| `formats` | `["mcq"]` | `mcq`, `judgement`, `gec`. Each pair gets one, at random |
| `negation_prob` | `0.2` | Chance a sentence is negated |
| `subject_pronoun_prob` | `0.3` | Chance of an explicit subject pronoun |
| `adverb_prob` | `0.6` | Chance of an adverbial (time, place...) |
| `norm` | `"iec"` | Only `iec` is implemented |

Note that the default `formats` here is `["mcq"]`, unlike LLM sources, whose default is all three.

### LLM sources (`type = "llm"`)

| Key | Default | Meaning |
|---|---|---|
| `backend` | `"default"` | Name of a `[backends.<name>]` table |
| `count` | `0` | Pairs to **request** in total. The writer may send back more per batch than asked, and all valid ones are kept, so the number of LLM candidates can exceed `count`. Fixes come on top |
| `batch_size` | `20` | Pairs asked for per call. Each batch is about one phenomenon |
| `phenomena` | all of the task's | Phenomena to write about. Batches cycle through them in order |
| `formats` | `["mcq", "judgement", "gec"]` | Each written pair is rendered in one of these, chosen at random (seeded by the pair) |
| `max_fix_attempts` | `2` | Times a pair that failed a check is sent back to the writer to fix. Each fix is a new candidate, checked again. `0` disables fixing |

`weight` and `[sources.params]` are ignored for LLM sources.

## `[checks]`

Automatic checks, run in this order on every candidate; the first failure stops the rest. A
`schema` check (required fields present, correct sentence among the accepted variants) always
runs first and can't be turned off.

| Key | Default | Meaning |
|---|---|---|
| `orthography` | `true` | No unelided *de*/article/pronoun, no missing contraction, capitalised, final punctuation |
| `marker` | `true` | The phenomenon's regex must match the incorrect sentence and none of the correct variants |
| `minimal_pair` | `true` | The two sentences must differ by one contiguous edit |
| `max_edit_tokens` | `8` | Longest allowed edit for `minimal_pair`, in tokens |

### `[checks.languagetool]`

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `true` | Check pairs against a LanguageTool server. The run stops at the start if the server isn't reachable |
| `url` | `"http://localhost:8081"` | Server address |
| `language` | `"ca-ES"` | LanguageTool language code |
| `disabled_rules` | `[]` | LanguageTool rule ids to ignore |
| `require_bad_flagged` | `false` | Also reject pairs whose incorrect sentence LanguageTool does **not** flag. Off by default: LanguageTool misses some errors, and those pairs are often the most interesting. Either way, the manifest reports the flag rate and examples of misses per phenomenon |
| `workers` | `8` | Parallel requests to the server |

A correct variant that LanguageTool flags always fails the check.

## `[judge]`

An LLM that reads each pair and answers a fixed set of questions (is the good sentence correct,
does the bad one contain exactly the stated error, is it natural, same meaning...). A pair passes
only if all answers are right.

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `false` | Use a judge at all |
| `backend` | `"default"` | Name of a `[backends.<name>]` table |
| `mode` | `"filter"` | See below |
| `sample_rate` | `0.05` | `audit` mode only: share of pairs judged |
| `source_types` | `[]` | Only judge pairs from these source types, e.g. `["llm"]`. Empty means all sources. Pairs from other sources are kept on the checks alone (`metadata.qc.judge = "not_used_for_source"`) |
| `on_error` | `"reject"` | When a judge call fails (timeout, refusal, unparseable reply): `"reject"` drops the pair, `"keep"` keeps it and marks it `kept_despite_judge_error`. Counted in the manifest either way |
| `workers` | `4` | Parallel judge calls |

**Modes:**

- **`filter`**: every pair that passed the checks (and is from a judged source type) is judged,
  and dropped on a failed verdict. This is the quality gate. Cost: one call per pair.
- **`audit`**: a random `sample_rate` of those pairs is judged, and **nothing is dropped**. The
  verdicts are recorded (`metadata.qc.judge = "audited"`, plus `judge_passed`) and summarised in
  the manifest, so you get an estimate of quality at a fraction of the cost. A typical use is
  checking whether template pairs are as correct as assumed. The sample is chosen by pair id, so
  it's the same pairs on every run.

`mode` and `source_types` apply to the whole judge: you can't filter one source type and audit
another in the same run.

Judge quality itself can be measured against native-speaker labels with `rauxa review` and
`rauxa calibrate` (see the [README](README.md#safeguards)).

## `[backends.<name>]`

Named LLM endpoints, referenced by `sources.backend` and `judge.backend`. Only the backends that
are actually referenced need to be configured.

| Key | Default | Applies to | Meaning |
|---|---|---|---|
| `type` | `"openai"` | | `"openai"`, `"anthropic"` or `"agent"` (below) |
| `model` | `""` | all | Model id. Required for `openai`, `anthropic`, and automated Copilot dispatch. For manual `agent`, recorded and used for family warnings |
| `base_url` | `""` | `openai`, `anthropic` | Endpoint. Required for `openai` (e.g. `"http://localhost:8000/v1"`); optional for `anthropic` |
| `api_key_env` | unset | `openai`, `anthropic` | Name of the environment variable holding the key (not the key itself). Unset: `openai` sends no key (fine for local servers); `anthropic` uses the SDK's own lookup (`ANTHROPIC_API_KEY`, or an `ant auth login` profile) |
| `temperature` | unset | `openai` | Sampling temperature. Leave unset for models that reject it (e.g. reasoning models) |
| `json_mode` | `true` | `openai` | Send `response_format = {"type": "json_object"}`. Turn off for servers that don't support it |
| `effort` | unset | `anthropic` | Reasoning effort: `low`, `medium`, `high`, `xhigh`, `max`. Unset uses the model's default. Higher is slower and costs more |
| `fallbacks` | `true` | `anthropic` | If the model declines a request, let the API retry it on another model (supported models only) |
| `timeout` | `120.0` | `openai`, `anthropic`, automated `agent` | Seconds per call; positive and finite for Copilot |
| `max_retries` | `3` | `openai`, `anthropic`, automated `agent` | For OpenAI/Copilot: total attempts per call, at least 1. Anthropic passes this to its SDK as its retry limit. Ignored for manual file responses |

**Types:**

- **`openai`**: any OpenAI-compatible chat completions endpoint: OpenAI, DeepSeek, Gemini, GLM,
  vLLM, Ollama, llama.cpp server... See [configs/castellanismes_full.toml](configs/castellanismes_full.toml).
- **`anthropic`**: Claude through Anthropic's SDK (`pip install -e ".[anthropic]"`). No sampling
  parameters are sent; use `effort` instead. See [configs/castellanismes_claude.toml](configs/castellanismes_claude.toml).
- **`agent`**: no API. Each call is written to `agent_exchange/requests/` and answered by a coding
  agent; the run stops with status `waiting` and continues with `--resume`. See
  [AGENT_PIPELINE.md](AGENT_PIPELINE.md) and [configs/castellanismes_agent.toml](configs/castellanismes_agent.toml).

A setup problem (missing SDK or key, rejected key, unknown model) stops the run straight away; a
single failed call is recorded and the run carries on.

The manifest warns when the judge is from the same model family as a writer (it may favour
pairs written in its own style), and when `evaluated_models` overlap with either.

## `[copilot]`

Automatically answer file-backed `agent` requests using fresh, tools-disabled GitHub Copilot
CLI sessions, entirely inside WSL/Linux. The pipeline runs, dispatches pending requests, and
resumes until finished or paused by its call cap. Existing API backends still work normally.

```toml
[copilot]
workers = 4
max_calls = 500
```

| Key | Default | Meaning |
|---|---|---|
| `enabled` | `true` when `[copilot]` is present, otherwise `false` | Optional; set `false` to keep the section configured but disabled. Automation requires at least one referenced `type = "agent"` backend |
| `workers` | `4` | Positive integer: maximum concurrent CLI calls across writers, repairs, and judges |
| `max_calls` | unset | Required positive integer when enabled: cumulative invocation cap for the whole run, including retries |

Dispatcher settings live **only in TOML**, with no Copilot-specific command-line overrides.
Declaring `[copilot]` enables automation without needing `enabled = true`. Even an empty
section enables it, so `max_calls` must be supplied in TOML unless automation is explicitly
disabled. Without this section, manual/API behavior is unchanged.
`enabled = false` disables automation while retaining its settings.
Existing explicit `enabled = true` settings remain valid. Unknown `[copilot]` keys are errors.

There is deliberately **no executable key in TOML**. Executable resolution is:
`RAUXA_COPILOT_BIN`, then `copilot` on PATH. The chosen value
is one executable name/path, not a shell command with arguments. An invalid explicit choice
is an error, not permission to fall back. The selected binary must pass version and
automation-capability preflight.

Dispatcher settings are parsed separately from `RunConfig` and **never enter its fingerprint**.
Changing workers, the call cap, or executable does not prevent `--resume`. The cap is not
reset on resume: raise it to allow more work. A cap at or below recorded usage launches no
additional calls. Dataset/source/model settings, including backend timeout/retry settings,
remain fingerprinted.

Each invocation records effective settings and their sources (`toml`, `default`,
`env:RAUXA_COPILOT_BIN`, or `path`), the resolved binary and version in
`dispatcher/events.jsonl`. `dispatcher/status.json` and the final manifest's `dispatcher`
section expose current settings and counters; `rauxa summary` displays them even while
the run is waiting. This provenance is outside the fingerprint.
Implicit enabling from section presence is recorded as `toml:copilot`, distinct from
an explicit `toml:copilot.enabled` value.

Calls are reserved durably before launch. A crash can leave a reservation without a
confirmed answer; it still consumes budget. This is not an exactly-once billing guarantee,
nor a token, credit, or currency cap. A valid response is saved atomically and reused on
resume. Budget exhaustion returns exit 3, preserving unfinished requests rather than failing
them. Ctrl+C/SIGTERM returns 130; setup/coordinator errors return 1.

After `max_retries` total attempts, an individual failure is recorded in
`agent_exchange/failures/` and processing continues. A failed writer batch consumes its
requested candidate budget once; a failed repair is abandoned; a failed judge uses
`judge.on_error`. Recognized authentication/model/permission errors stop the run instead.
The dataset may be incomplete, or complete with recorded failures if enough other pairs
survive. Automatic linguistic checks and a negative judge verdict are not transport failures.

See [the WSL setup guide](GETTING_STARTED.md#automated-copilot-cli-in-wsl) for authentication,
isolation requirements, and an example run.

## `[min_share.<field>]`

Minimum share of the **final dataset** for a value of a metadata field. One table per field, one
key per value, shares as fractions of `target_size`:

```toml
[min_share.phenomenon]
tenir_que = 0.25
hi_ha_que = 0.2
donar_se_compte = 0.25

[min_share.format]
gec = 0.2
```

- Any metadata field works: `phenomenon`, `format`, `tense`, `source_dataset`...
- The shares for one field must add up to at most 1. Values not listed have no minimum.
- If the kept pool can't cover a share, the controller asks the procedural sources for more
  pairs (roughly 3× the shortfall, since it can't steer them to one value). If it still can't
  after `max_rounds`, the dataset is written as `dataset.incomplete.jsonl`.
- LLM-written pairs have `tense` unset, so a `tense` minimum is only met by procedural pairs.

## Changing a config and `--resume`

`rauxa run CONFIG --resume` continues an interrupted or waiting run from its run folder. The
dataset config is fingerprinted, and **any dataset configuration change** refuses the resume, including keys that are ignored
(such as `count` on a procedural source) and reordering sources. To change the config, use a new
`name` or delete the old run folder. The operational `[copilot]` section and executable
environment selection are excluded from this fingerprint and can change on resume.
