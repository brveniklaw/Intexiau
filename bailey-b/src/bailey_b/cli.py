"""Command-line entry point: ``bailey-b <command>``."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from bailey_b.config import get_settings
from bailey_b.evals import MIN_EVAL_QUESTIONS, load_eval_set, run_eval


def _config(_: argparse.Namespace) -> int:
    print(get_settings().model_dump_json(indent=2))
    return 0


def _eval_validate(args: argparse.Namespace) -> int:
    try:
        questions = load_eval_set(args.path)
    except ValueError as e:
        print(f"invalid: {e}", file=sys.stderr)
        return 1
    areas = Counter(q.practice_area.value for q in questions)
    print(f"{len(questions)} questions OK")
    for area, n in sorted(areas.items()):
        print(f"  {area}: {n}")
    if len(questions) < MIN_EVAL_QUESTIONS:
        print(f"warning: fewer than {MIN_EVAL_QUESTIONS} questions", file=sys.stderr)
    return 0


def _eval_run(args: argparse.Namespace) -> int:
    from bailey_b.search import retrieve

    questions = load_eval_set(args.path)
    k = args.k or get_settings().search.top_n
    report = run_eval(questions, lambda q: retrieve(q, top_n=k), k=k)
    print(json.dumps(report.model_dump(mode="json"), indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bailey-b")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("config", help="print resolved settings").set_defaults(func=_config)

    ev = sub.add_parser("eval", help="retrieval eval set").add_subparsers(
        dest="eval_command", required=True
    )
    v = ev.add_parser("validate", help="check an eval JSONL file")
    v.add_argument("path", type=Path)
    v.set_defaults(func=_eval_validate)
    r = ev.add_parser("run", help="score retrieval against an eval JSONL file")
    r.add_argument("path", type=Path)
    r.add_argument("-k", type=int, default=None, help="cutoff (default: search.top_n)")
    r.set_defaults(func=_eval_run)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
