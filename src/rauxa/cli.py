"""Command-line interface: ``rauxa list | generate | run | qc | review | calibrate | export-lm-eval``."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

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
    from .pipeline import Pipeline, PipelineError, load_config

    config = load_config(args.config)
    try:
        result = Pipeline(config, resume=args.resume).run()
    except PipelineError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(result.path)
    return {"complete": 0, "incomplete": 2, "waiting": 3}[result.status]


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
