# Running the pipeline with a coding agent

This file lets you run the rauxa pipeline without paying for an LLM API. A coding agent you
already use (Claude Code, Codex, Gemini CLI, GitHub Copilot CLI...) plays the two LLM roles,
the **writer** and the **judge**, and everything else runs as normal.

**What is real:** the template generator, the automatic checks (including LanguageTool),
deduplication, minimum shares, the manifest and the output files. The prompts are the same ones
the API version sends.

**What is approximated:** the LLM answers. They come from your agent, often one model playing
both roles, in a session that can see the rest of the repo. Treat the result as a good
approximation for learning and small experiments, not as equivalent to an API run. The manifest
records that the run used an agent.

**To start**, open this repository in your agent and tell it:

> Follow the instructions in AGENT_PIPELINE.md.

## Automatic Copilot CLI alternative

For unattended generation inside WSL/Linux, use
[configs/castellanismes_runners_llm_only.toml](configs/castellanismes_runners_llm_only.toml) and follow
[the WSL setup guide](GETTING_STARTED.md#automated-copilot-cli-in-wsl). The `[runners.copilot]`
section configures `concurrency` and the cumulative `max_calls`. It requires explicit
`enabled = true` to allow automated calls that can incur costs; section presence alone
leaves automation disabled. Use `enabled = false` to retain settings without dispatch.
There are no Copilot-specific CLI overrides.
Executable selection is machine-local (`RAUXA_COPILOT_BIN`, then PATH),
not committed TOML. Operational settings are excluded from the dataset fingerprint.

In automatic mode, do not answer request files or run a second controller alongside the
dispatcher. It launches fresh, tools-disabled CLI sessions, validates their JSON, writes
responses atomically, records permanent failures separately, and resumes the pipeline.
Exit 3 means the call budget needs increasing; exit 130 means interrupted. Resume with the
same dataset config and, if necessary, a larger operational cap. Effective settings and
their origins are recorded in `dispatcher/events.jsonl` and the manifest.

The instructions below describe the **manual** workflow. Set `enabled = false` in `[runners.copilot]`
to choose it when the config enables automatic dispatch. Legacy manual runs remain manually resumable.

---

## Instructions for the agent

You are going to run a data-generation pipeline. When the pipeline needs an LLM, it writes a
request file and stops; you answer the request files, then restart it. Repeat until it
finishes. Work through these steps in order.

### 1. Set up

- Work from the repository root. Python 3.11 or newer is required.
- The commands below use `rauxa`. Your shell may not keep an activated environment between
  commands, so call it by path: if `.venv/bin/rauxa` exists, use `.venv/bin/rauxa`. If there
  is no `.venv`, either create it (`python3 -m venv .venv && .venv/bin/pip install -e .`) or,
  if that fails, use `PYTHONPATH=src python3 -m rauxa` instead.
- LanguageTool: check whether a server answers at `http://localhost:8081/v2/languages`. If
  not, and Java 17+ and a LanguageTool download are available, start it:
  `java -cp languagetool-server.jar org.languagetool.server.HTTPServer --port 8081`.
  If that isn't possible, set `enabled = false` under `[checks.languagetool]` in
  `configs/castellanismes_agent.toml` and tell the user the LanguageTool check was skipped.
  Don't install system packages without asking the user.
- In `configs/castellanismes_agent.toml`, set `model` under `[backends.writer]` and
  `[backends.judge]` to the model you are running on. If the user says the judge will be
  answered by a different tool or model, use that model's name for the judge.

Do these before the first run: the run refuses to resume if the config changes afterwards.

### 2. Run, answer, resume

Start the run:

```bash
rauxa run configs/castellanismes_agent.toml
```

The exit code tells you what to do next:

| Exit code | Meaning | Next step |
|---|---|---|
| 3 | Waiting for you | Answer the requests (below), then run again with `--resume` |
| 0 | Complete | Go to step 4 |
| 2 | Finished but incomplete (too few pairs, or a minimum share missed) | Go to step 4 |
| 1 | Error | Read the message, fix the cause if you can, otherwise report it to the user |

To continue after answering:

```bash
rauxa run configs/castellanismes_agent.toml --resume
```

A full run usually takes three to five rounds of waiting.

### 3. Answer the requests

Pending requests are in `out/runs/castellanismes_agent/agent_exchange/requests/`. Each one is a
Markdown file, `<id>.md` (ignore the `.json` copy next to it and the `done/` folder).

For each `<id>.md`:

1. Read it. It has a **System** section (your role) and a **User** section (the task).
2. Do exactly what the User section asks, in the role the System section describes.
3. Save your answer to `out/runs/castellanismes_agent/agent_exchange/responses/<id>.json`.
   The file must contain a single valid JSON object and nothing else: no Markdown fences, no
   explanation.

There are three kinds of request:

- **Write** (asks for "new pairs"): return exactly the number of items asked for. Each pair is
  a fully correct Catalan sentence and the same sentence with only the stated error. Vary topics
  and structures; don't reuse the example sentences.
- **Fix** (says the item "did not pass our checks"): rewrite the item so the listed problems go
  away while keeping the requirements.
- **Judge** (the System section starts "You are a strict reviewer"): give your honest verdict
  on each question. You are checking work, possibly your own: don't default to `true`. If a
  sentence is unnatural, has a second error, or the "correct" one isn't normative, say so.

Rules that keep the run meaningful:

- Answer each request from its own text. Don't read the run's other files (`candidates.jsonl`,
  `checks.jsonl`, other responses, the manifest) or the source code to decide an answer.
- Only write files in `responses/`. Never edit `state.json`, the `.jsonl` files or the config
  during a run.
- Write Catalan with correct accents and the straight apostrophe (`'`).
- If you can, answer the judge requests in a fresh session, or have the user answer them with
  a different tool or model. A model judging its own sentences tends to approve them.

### 4. Report

When the run finishes, read `out/runs/castellanismes_agent/manifest.json` and tell the user:

- the status and the dataset path (`dataset.jsonl` or `dataset.incomplete.jsonl`);
- how many pairs were made, kept and rejected, and how many came from you (`by_source_type`);
- the judge stats (`judged`, `passed`, `failed_questions`);
- every entry in `warnings`.

### Troubleshooting

- **A response isn't valid JSON**: the pipeline records that call as failed (a judged pair is
  then rejected). Answer the remaining requests carefully; don't edit the run's records.
- **"config changed since this run started"**: the config was edited after the first run.
  Either undo the edit, or change `name` under `[run]` to start a fresh run.
- **The same requests keep appearing**: check the response file name matches the request id
  exactly and that it is in `responses/`, not `requests/`.
