# Getting started with rauxa

This walks you from a fresh clone to a finished dataset with the pipeline. You need:

- **Python 3.11 or newer**, with `venv` and `pip`. rauxa itself has no dependencies.
- **Java 17 or newer**, for the LanguageTool check. It's recommended but optional: see step 2.

Run every command from the repository root.

## 0. Install

Create a virtual environment and install rauxa into it:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

This gives you the `rauxa` command used below. `-e` (editable) means your changes to the code
take effect without reinstalling. In each new terminal, activate the environment again with
`source .venv/bin/activate`.

On Debian or Ubuntu, if `python3 -m venv` fails, install the missing pieces with
`sudo apt install python3-venv python3-pip`.

**Without pip:** skip the install, run `export PYTHONPATH=src` in each terminal, and write
`python3 -m rauxa` wherever this guide says `rauxa`.

## 1. Check that everything works

```bash
python3 -m unittest discover -s tests
```

It should end with `OK` after a couple of seconds. The tests need neither Java nor LanguageTool.

## 2. Start LanguageTool

[LanguageTool](https://languagetool.org) is a free grammar checker that runs on your machine.
The pipeline uses it to make sure the "correct" sentences really are correct. Download it once
(about 250 MB):

```bash
mkdir -p ~/tools/languagetool && cd ~/tools/languagetool
curl -L -o LanguageTool-stable.zip https://languagetool.org/download/LanguageTool-stable.zip
python3 -m zipfile -e LanguageTool-stable.zip .
```

Then start the
server in a separate terminal and leave it open:

```bash
cd ~/tools/languagetool/LanguageTool-*/
java -cp languagetool-server.jar org.languagetool.server.HTTPServer --port 8081
```

To check it's up, `curl -s http://localhost:8081/v2/languages` should print a list of
languages.

**No Java?** Set `enabled = false` under `[checks.languagetool]` in the config. The other
checks still run, but mistakes in the templates are less likely to be caught.

## 3. Run the pipeline

Back in the repository:

```bash
rauxa run configs/castellanismes_procedural.toml
```

This config uses templates only, no LLM. It makes 20,000 pairs in about 2–3 minutes and prints
one line per round, ending like this:

```text
round 1: candidates=24000 checked=24000 kept=24000/24000 share_shortfalls={}
complete: 20000/20000 pairs → out/runs/castellanismes_procedural/dataset.jsonl
warning: no judge used: pairs were kept on the automatic checks alone
```

The warning is expected here: this config uses no LLM judge. The exit code tells you how the
run ended:

| Exit code | Meaning |
|---|---|
| 0 | Complete: `dataset.jsonl` has the target size and every minimum share is met |
| 2 | Incomplete: written to `dataset.incomplete.jsonl`; the summary says what fell short |
| 3 | Waiting for a coding agent to answer requests (agent mode only, see step 7) |
| 1 | Error, e.g. LanguageTool isn't running; the message says what to do |

## 4. Look at the results

```bash
rauxa summary out/runs/castellanismes_procedural
```

This shows the status, how many pairs were made, kept and rejected, how often LanguageTool
caught each error, the share of each phenomenon and format, any warnings, and five random
pairs. The dataset itself is a JSONL file with one item per line; see
[Procedural generator](README.md#procedural-generator) for the fields and
[Run directory](README.md#run-directory) for the other files in the run folder.

## 5. Run it again

Each run is stored in `out/runs/<name>/`, where `<name>` is `name` under `[run]`. Running the
same config a second time stops with "already holds a run":

- If the run was interrupted (Ctrl+C, a crash), continue it with `--resume`.
- For a fresh run, delete the folder or change `name`.

A run refuses to resume if its dataset config has changed since it started, so the results
match the config recorded in `manifest.json`. Operational `[runners.copilot]` settings are excluded:
workers, call caps, and executable selection may change on resume without resetting usage.

## 6. Change what gets generated

Copy a config and edit the copy. The settings you are most likely to change:

| Setting | Section | What it does |
|---|---|---|
| `name` | `[run]` | Folder name under `out/runs/` |
| `target_size` | `[run]` | Number of pairs in the dataset |
| `seed` | `[run]` | Same seed, same pairs; change it for different pairs |
| `tenses`, `formats`, `phenomena` | `[sources.params]` | Limit what the templates produce |
| one key per value | `[min_share.phenomenon]`, `[min_share.format]` | Minimum share of each value; the run is incomplete if one can't be met |
| `enabled` | `[checks.languagetool]` | Turn the LanguageTool check off |

For a quick experiment, set `name = "test"` and `target_size = 500`.

Every key, with defaults and how they interact, is in [CONFIG.md](CONFIG.md).

## 7. Add LLMs (optional)

An LLM writer adds more varied sentences than the templates, and an LLM judge reviews them.
There are four ways to use them:

- **An API or a local model server.** Use
  [configs/castellanismes_full.toml](configs/castellanismes_full.toml) and set `base_url` and
  `model` under `[backends.writer]` and `[backends.judge]`. For a hosted API, export the key
  named by `api_key_env` (e.g. `export OPENAI_API_KEY=...`). Local servers such as Ollama
  (`base_url = "http://localhost:11434/v1"`) or vLLM need no key. Then run it as in step 3.
- **Claude.** Install Anthropic's SDK and set your key, then run
  [configs/castellanismes_claude.toml](configs/castellanismes_claude.toml), which uses
  Claude Sonnet 5.5 as the writer and Claude Opus 5.5 as the judge (about $2–4 per run):

  ```bash
  pip install -e ".[anthropic]"
  export ANTHROPIC_API_KEY=your-key
  rauxa run configs/castellanismes_claude.toml
  ```

- **A coding agent instead of an API** (Claude Code, Codex...), at no API cost. Open the repo
  in your agent and ask it to follow [AGENT_PIPELINE.md](AGENT_PIPELINE.md).
- **Automated Copilot CLI workers inside WSL/Linux.** Use the setup below. Copilot plan usage
  and service limits still apply; this does not make generation free.

Judging only LLM-written pairs (`source_types = ["llm"]` under `[judge]`) keeps the cost low:
template pairs are correct by construction. Use models from different families for the writer
and the judge when you can; a model judging its own sentences tends to approve them.

### Automated Copilot CLI in WSL

Run Python, Copilot CLI, and the pipeline **inside the same WSL distribution**, from the
repository root. No Windows-to-WSL launcher or path translation is involved.

Install GitHub Copilot CLI yourself using its
[installation guide](https://docs.github.com/en/copilot/how-tos/copilot-cli/set-up-copilot-cli/install-copilot-cli).
Check `copilot --version` and `copilot --help` in the shell that will launch rauxa.
Rauxa checks the selected binary for non-interactive model selection, silent text output,
disabled tools/MCP/custom instructions, disabled questions/updates/BASH_ENV, and disabled
remote sessions/export. Unsupported CLI builds fail preflight rather than weakening isolation.

Authenticate with a supported token environment variable (`COPILOT_GITHUB_TOKEN`, `GH_TOKEN`,
or `GITHUB_TOKEN`; see the
[CLI authentication reference](https://docs.github.com/en/copilot/reference/copilot-cli-reference/cli-command-reference#copilot-login-options)).
Do not put tokens in TOML or commit them. Token-based runs get an empty temporary Copilot
profile for each request.

Alternatively, use a **login-only Copilot profile** for normal CLI/keychain authentication:

```bash
export COPILOT_HOME="$HOME/.copilot-rauxa"
copilot login
```

Do not add settings, hooks, plugins, instructions, MCP servers, or other customizations to
this profile. Without a token environment variable, rauxa refuses customized profiles
instead of loading potentially executable hooks or plugins. Your normal profile is never
modified. Each request uses a fresh conversation in a temporary working directory, with no
model tools; this is context/permission isolation, **not an OS sandbox**.

For an executable not on PATH, set `RAUXA_COPILOT_BIN` to its executable name or path.
Selection order is the environment variable, then `copilot` on PATH.
This machine-local setting intentionally has no TOML key; an invalid explicit choice fails.

Start LanguageTool as in step 2, or explicitly disable it in your config before the first run.
Copy [the Copilot config](configs/castellanismes_copilot.toml), choose models available to
your account, and review its `[runners.copilot]` section:

```toml
[runners.copilot]
enabled = true
workers = 4
max_calls = 500
```

Then run:

```bash
rauxa run configs/castellanismes_copilot.toml
rauxa summary out/runs/castellanismes_copilot
```

Automation requires explicit `enabled = true` because CLI calls can incur costs. Without the
section or the `enabled` key, automation stays off. `enabled = false` keeps the settings but
disables it. Older configs must rename `[copilot]` to `[runners.copilot]` and add
`enabled = true` to keep using automation.

The dispatcher answers the pipeline's writer, repair, and judge requests and resumes
automatically. `workers` caps concurrent CLI calls across all roles. `max_calls` caps
cumulative launch reservations, including retries, across the whole run.
When the cap is reached, exit 3 means the remaining work is waiting, not failed. Raise
`max_calls` in TOML (and change `workers` there if needed), then resume:

```bash
rauxa run configs/castellanismes_copilot.toml --resume
```

Copilot settings are configured only in TOML; there are no Copilot-specific CLI flags.
Workers, caps, and executable choices can change on resume without changing the dataset
fingerprint. Models, source budgets, and backend timeout/retry settings cannot.
Set `enabled = false` in `[runners.copilot]` to select the manual file workflow.

Transport/JSON failures are retried up to each backend's `max_retries` total attempts,
with its `timeout` in seconds per attempt. Exhausted requests are recorded and processing
continues; judge errors follow `judge.on_error`. Authentication/model setup failures stop
the run. Ctrl+C/SIGTERM stops owned workers and preserves work (exit 130). A guard process
keeps the run locked and enforces the timeout even if its coordinator dies unexpectedly.
Do not delete `.run.lock` or run two controllers against one output directory.

`dispatcher/events.jsonl` records reservations, outcomes, worker guard PIDs, and each
invocation's effective settings and value origins. `dispatcher/status.json` and the final
manifest report usage/failures; `rauxa summary` also works as a status report before a final
manifest exists. Complete JSON answers are published atomically. A launch whose completion
is unknown after a crash still consumes its reserved budget; exact once-only billing is
not guaranteed.

For a larger run, increase `sources.count` (LLM candidates requested), not just
`run.target_size` (final pairs selected). Writers batch pairs, filter judges use one call
per eligible pair, repairs add calls, and retries add invocations. A call cap is not an
exact credit/token/currency limit. Start with a small run before scaling toward 10,000 LLM
candidates, and keep writer/judge model families distinct when possible.

Manual agent runs from the legacy exchange protocol remain manually resumable, but cannot
be automatically adopted safely; use a fresh run name for Copilot dispatch.

## 8. Use the dataset

- **Benchmark a model** with
  [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness):

  ```bash
  rauxa export-lm-eval out/runs/castellanismes_procedural/dataset.jsonl out/lm_eval --name rauxa_cv
  lm_eval --model hf --model_args pretrained=BSC-LT/salamandra-7b --include_path out/lm_eval --tasks rauxa_cv_pairs,rauxa_cv_mcq
  ```

- **Have a native speaker check a sample.** `rauxa review` shows pairs one at a time and saves
  the answers; `rauxa calibrate` then reports how many pairs they rejected and, if you used a
  judge, how often it agreed with them:

  ```bash
  rauxa review out/runs/castellanismes_procedural/dataset.jsonl out/labels.jsonl --n 100 --reviewer your-name
  rauxa calibrate out/labels.jsonl out/runs/castellanismes_procedural
  ```

  This step is optional: nothing in the pipeline depends on it, and you can skip it for
  experiments. Do it before you publish the dataset or report results on it. The automatic
  checks only catch the errors they were built for, and a native speaker reading 50–100 pairs
  is the only way to estimate how many others got through.

- **Have an LLM check a sample.** The judge can review a finished dataset without dropping
  anything: add this to a copy of the config, plus a `[backends.judge]` section (an API, a local
  server, or `type = "agent"` for a coding agent, see step 7), and run it under a new `name`:

  ```toml
  [judge]
  enabled = true
  backend = "judge"
  mode = "audit"        # judges a sample and drops nothing
  sample_rate = 0.02    # about 2% of pairs
  ```

  `rauxa summary` then shows how many sampled pairs the judge approved. This is cheap and
  catches obvious problems, but it doesn't replace a native speaker: the human review is what
  tells you whether the judge itself can be trusted. A good routine is to let the judge audit a
  large sample, have a person label a small one, and run `rauxa calibrate` to see how often the
  two agree.
