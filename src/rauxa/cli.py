"""Command-line interface:
``rauxa list | generate | run | summary | qc | review | calibrate | export-lm-eval``."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from random import Random

from . import DATASETS, create_dataset
from .export import export_lm_eval, read_jsonl, write_jsonl
from .qc.languagetool import DEFAULT_URL, LanguageToolClient, check_pair


def _csv(value: str) -> tuple[str, ...]:
    return tuple(v.strip() for v in value.split(",") if v.strip())


def cmd_list(_: argparse.Namespace) -> int:
    for name, entry in sorted(DATASETS.items()):
        doc = (sys.modules[entry.dataset_cls.__module__].__doc__ or "").strip().splitlines()
        print(f"{name}\t{doc[0] if doc else ''}")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    kwargs = {"size": args.size, "seed": args.seed}
    for key in ("phenomena", "tenses", "formats"):
        value = getattr(args, key)
        if value:
            kwargs[key] = value
    if args.norm:
        kwargs["norm"] = args.norm
    dataset = create_dataset(args.task, **kwargs)
    if args.output:
        n = write_jsonl(dataset, Path(args.output))
        print(f"wrote {n} items to {args.output}", file=sys.stderr)
    else:
        for entry in dataset:
            print(json.dumps(entry, ensure_ascii=False))
    return 0


def cmd_qc(args: argparse.Namespace) -> int:
    entries = read_jsonl(Path(args.input))
    client = LanguageToolClient(url=args.lt_url, language=args.language, disabled_rules=args.disable_rules)
    controls = [e for e in entries if e["metadata"].get("sentence_bad") is None]
    if controls:
        # A control has no incorrect sentence to flag; the pipeline's languagetool check covers it.
        print(f"skipping {len(controls)} control items (no incorrect sentence)", file=sys.stderr)
        entries = [e for e in entries if e["metadata"].get("sentence_bad") is not None]
    kept, stats, rules = [], Counter(), Counter()
    for entry in entries:
        meta = entry["metadata"]
        result = check_pair(client, meta["sentence_good"], meta["sentence_bad"])
        meta["qc_languagetool"] = result
        stats["good_clean"] += result["good_clean"]
        stats["bad_flagged"] += result["bad_flagged"]
        for m in result["good_matches"]:
            rules[("good", m["rule_id"])] += 1
        for m in result["bad_matches"]:
            rules[("bad", m["rule_id"])] += 1
        if result["good_clean"] and (result["bad_flagged"] or not args.require_bad_flagged):
            kept.append(entry)

    n = len(entries)
    print(f"items: {n}", file=sys.stderr)
    print(f"good sentence clean: {stats['good_clean']}/{n}", file=sys.stderr)
    print(f"bad sentence flagged: {stats['bad_flagged']}/{n}", file=sys.stderr)
    print(f"kept: {len(kept)}/{n}", file=sys.stderr)
    for (side, rule), count in rules.most_common(args.top_rules):
        print(f"  {side:4} {rule}: {count}", file=sys.stderr)
    if args.output:
        write_jsonl(kept, Path(args.output))
    if args.report:
        write_jsonl(entries, Path(args.report))
    return 0


def cmd_export_lm_eval(args: argparse.Namespace) -> int:
    entries = read_jsonl(Path(args.input))
    for path in export_lm_eval(entries, Path(args.out_dir), args.name):
        print(path)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from .pipeline import Pipeline, PipelineError
    from .pipeline.config import load_settings, resolve_copilot
    from .pipeline.dispatch import Dispatcher
    from .pipeline.llm import BackendUnavailable
    from .pipeline.storage import RunLocked

    try:
        config, copilot = load_settings(args.config)
        settings = resolve_copilot(copilot)
        if settings.enabled:
            result = Dispatcher(config, settings, resume=args.resume).run()
        else:
            with Pipeline(config, resume=args.resume) as pipeline:
                if copilot.section_present or (config.run_dir() / "dispatcher").exists():
                    Dispatcher(config, settings).record_disabled()
                result = pipeline.run()
    except (KeyboardInterrupt, InterruptedError):
        print("interrupted: completed work is saved; continue with the same config and --resume", file=sys.stderr)
        return 130
    except BackendUnavailable as e:
        print(f"error: {e}", file=sys.stderr)
        print(f"Fix this, then continue with: rauxa run {args.config} --resume "
              "(everything done so far is kept)", file=sys.stderr)
        return 1
    except (PipelineError, RunLocked, OSError, ValueError, TypeError, AssertionError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(result.path)
    return {"complete": 0, "incomplete": 2, "waiting": 3}[result.status]


def _dispatcher_summary(dispatcher: dict) -> None:
    settings = dispatcher["settings"]
    print(f"Copilot:  {dispatcher['status']}; {dispatcher['calls_used']}/{settings['max_calls']} calls reserved, "
          f"{dispatcher['completed']} responses, {dispatcher['failed']} failed requests")
    print(f"CLI:      {dispatcher['executable'] or 'not invoked'} ({dispatcher['cli_version'] or 'preflight not completed'})")
    for key in ("enabled", "concurrency", "max_calls", "command"):
        # Older runs stored the concurrency limit as "workers".
        stored_key = "workers" if key == "concurrency" and key not in settings else key
        print(f"  {key}: {settings[stored_key]} ({settings['origins'][stored_key]})")


def cmd_summary(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    manifest_path = run_dir / "manifest.json"
    dispatcher_path = run_dir / "dispatcher" / "status.json"
    dispatcher = json.loads(dispatcher_path.read_text(encoding="utf-8")) if dispatcher_path.exists() else None
    if not manifest_path.exists():
        if dispatcher:
            _dispatcher_summary(dispatcher)
        if (run_dir / "state.json").exists():
            print(f"{run_dir}: the run hasn't finished (it was interrupted, or it is waiting for an "
                  "agent). Continue it with: rauxa run CONFIG --resume", file=sys.stderr)
        else:
            print(f"{run_dir}: no run found here", file=sys.stderr)
        return 1
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    if dispatcher or m.get("dispatcher"):
        _dispatcher_summary(dispatcher or m["dispatcher"])
    c = m["counts"]
    dataset = run_dir / ("dataset.jsonl" if m["status"] == "complete" else "dataset.incomplete.jsonl")
    print(f"Status:   {m['status']}  ({m['started_at']} → {m['finished_at']})")
    print(f"Dataset:  {dataset}  ({c['selected']} of {m['config']['target_size']} pairs)")
    print(f"Pairs:    {c['candidates']} made, {c['kept']} kept, {c['rejected']} rejected, "
          f"{c['duplicates_skipped']} duplicates skipped")
    print(f"Sources:  " + ", ".join(f"{k} {v}" for k, v in c["by_source_type"].items()))
    if m["check_failures"]:
        print("Failed checks: " + ", ".join(f"{k} {v}" for k, v in m["check_failures"].items()))
    if m["languagetool"]:
        print("LanguageTool flagged the incorrect sentence in:")
        width = max(18, *(len(p) for p in m["languagetool"]))
        for phenomenon, lt in sorted(m["languagetool"].items()):
            print(f"  {phenomenon:<{width}} {lt['flag_rate']:>6.1%} of {lt['checked']}")
    for dim, rows in m["shares"]["fields"].items():
        print(f"Share by {dim}: " + ", ".join(f"{v} {r['share']:.1%}" for v, r in rows.items()))
    j = m["judge"]
    if j["enabled"]:
        print(f"Judge:    {j['judged']} judged, {j['passed']} passed, {j['errors']} errors ({j['model'] or 'model not set'})")
    for warning in m["warnings"]:
        print(f"Warning:  {warning}")
    if args.samples and dataset.exists():
        items = read_jsonl(dataset)
        n = min(args.samples, len(items))
        print(f"\n{n} random pair{'s' if n != 1 else ''}:")
        for entry in Random(args.seed).sample(items, n):
            meta = entry["metadata"]
            kind = meta.get("kind", "pair")
            print(f"\n  [{meta['phenomenon']}, {meta['format']}{'' if kind == 'pair' else ', ' + kind}]")
            if meta.get("context"):
                print(f"  context: {meta['context']}")
            print(f"  ✓ {meta['sentence_good']}")
            if meta.get("sentence_bad") is not None:
                print(f"  {'≈' if kind == 'preference' else '✗'} {meta['sentence_bad']}")
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    from .pipeline.calibration import review

    n = review(Path(args.items), Path(args.labels), n=args.n, seed=args.seed, reviewer=args.reviewer)
    print(f"{n} items labelled → {args.labels}", file=sys.stderr)
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    from .pipeline.calibration import calibrate

    print(json.dumps(calibrate(Path(args.labels), Path(args.run_dir)), ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rauxa", description="Procedural generator of verifiable Catalan tasks.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list", help="list registered tasks")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("generate", help="generate items as JSONL")
    p.add_argument("task")
    p.add_argument("--size", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--phenomena", type=_csv, help="comma-separated subset of phenomena")
    p.add_argument("--tenses", type=_csv, help="comma-separated subset of tenses")
    p.add_argument("--formats", type=_csv, help="comma-separated: mcq,judgement,gec")
    p.add_argument("--norm", help="normative standard (only 'iec' for now)")
    p.add_argument("-o", "--output", help="output JSONL (default: stdout)")
    p.set_defaults(func=cmd_generate)

    p = sub.add_parser("qc", help="check sentence pairs with a LanguageTool server")
    p.add_argument("input")
    p.add_argument("--lt-url", default=DEFAULT_URL)
    p.add_argument("--language", default="ca-ES")
    p.add_argument("--disable-rules", type=_csv, default=())
    p.add_argument("--require-bad-flagged", action="store_true",
                   help="also drop items whose incorrect sentence LanguageTool does not flag")
    p.add_argument("--top-rules", type=int, default=15)
    p.add_argument("-o", "--output", help="write the items that pass to this JSONL")
    p.add_argument("--report", help="write all items annotated with QC results to this JSONL")
    p.set_defaults(func=cmd_qc)

    p = sub.add_parser("run", help="run the generation pipeline from a TOML config")
    p.add_argument("config")
    p.add_argument("--resume", action="store_true", help="continue an interrupted run with the same config")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("summary", help="show the results of a pipeline run, with sample pairs")
    p.add_argument("run_dir", help="e.g. out/runs/castellanismes_procedural")
    p.add_argument("--samples", type=int, default=5, help="random pairs to show (0 for none)")
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_summary)

    p = sub.add_parser("review", help="label pairs as a native speaker, to measure the judge and checks")
    p.add_argument("items", help="JSONL of items, e.g. a run's dataset.jsonl")
    p.add_argument("labels", help="JSONL file to append labels to")
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--reviewer", default="anonymous")
    p.set_defaults(func=cmd_review)

    p = sub.add_parser("calibrate", help="compare human labels with the judge's verdicts and the automatic checks")
    p.add_argument("labels")
    p.add_argument("run_dir")
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("export-lm-eval", help="write lm-evaluation-harness task YAMLs and data")
    p.add_argument("input")
    p.add_argument("out_dir")
    p.add_argument("--name", default="rauxa")
    p.set_defaults(func=cmd_export_lm_eval)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
