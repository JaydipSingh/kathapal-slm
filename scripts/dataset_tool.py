#!/usr/bin/env python3
"""
Dataset authoring tool: add / lint / merge records safely.

Every added record is schema- AND safety-validated before it is written, and
ids are auto-assigned to avoid collisions. Invalid records are reported, never
silently saved.

Examples:
    # Add one record from a JSON payload file (auto-assigns id):
    uv run python scripts/dataset_tool.py add \
        --dataset data/samples/starter_dataset.jsonl --from-json new_story.json

    # Add interactively (prompts for each field):
    uv run python scripts/dataset_tool.py add \
        --dataset data/samples/starter_dataset.jsonl --interactive

    # Lint a dataset (validation report + health summary):
    uv run python scripts/dataset_tool.py lint --dataset data/samples/starter_dataset.jsonl

    # Merge several files into one (dedup by id):
    uv run python scripts/dataset_tool.py merge --out data/processed/all.jsonl \
        --sources data/samples/starter_dataset.jsonl data/samples/more.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kathapal.data.authoring import add_from_dict, merge_files  # noqa: E402
from kathapal.data.schema import ALLOWED_TOPICS  # noqa: E402
from kathapal.data.validation import validate_dataset  # noqa: E402
from kathapal.logging_config import configure_logging  # noqa: E402

log = configure_logging()


_INTERACTIVE_FIELDS = [
    ("topic", f"topic {sorted(ALLOWED_TOPICS)}"),
    ("instruction", "instruction (the request the model should answer)"),
    ("story_title", "story title"),
    ("story", "story text (20-250 words)"),
    ("question", "comprehension question (end with '?')"),
    ("answer", "answer"),
    ("learning_goal", "learning goal"),
]


def _cmd_add(args: argparse.Namespace) -> int:
    if args.from_json:
        payload = json.loads(Path(args.from_json).read_text(encoding="utf-8"))
    elif args.interactive:
        payload = {}
        print("Enter record fields (Ctrl-C to abort):")
        for key, desc in _INTERACTIVE_FIELDS:
            payload[key] = input(f"  {desc}: ").strip()
        age = input("  age_min-age_max [3-5]: ").strip() or "3-5"
        lo, hi = age.split("-")
        payload["age_min"], payload["age_max"] = int(lo), int(hi)
        payload["source"] = input("  source [original-author]: ").strip() or "original-author"
        payload["license"] = input("  license [CC-BY-4.0]: ").strip() or "CC-BY-4.0"
    else:
        log.error("provide --from-json PATH or --interactive")
        return 2

    result = add_from_dict(args.dataset, payload)
    if result.ok:
        log.info("Added record '%s' to %s", result.record.id, args.dataset)
        return 0
    log.error("Record rejected (not saved):")
    for err in result.errors:
        log.error("  - %s", err)
    return 1


def _cmd_lint(args: argparse.Namespace) -> int:
    report = validate_dataset(args.dataset)
    log.info("records=%d languages=%s topics=%s", report.total_records,
             report.by_language, report.by_topic)
    if report.load_errors:
        log.error("load errors: %d", len(report.load_errors))
    if report.duplicate_ids:
        log.error("duplicate ids: %s", report.duplicate_ids)
    if report.safety_failures:
        log.error("safety failures: %s", report.safety_failures)
    if report.missing_provenance_ids:
        log.error("missing provenance: %s", report.missing_provenance_ids)
    # Simple health hint on dataset size.
    n = report.total_records
    if n < 50:
        log.warning("dataset has %d records: smoke-test scale only", n)
    elif n < 200:
        log.info("dataset has %d records: early-experiment scale", n)
    else:
        log.info("dataset has %d records: reasonable experiment scale", n)
    log.info("overall OK: %s", report.ok)
    return 0 if report.ok else 1


def _cmd_merge(args: argparse.Namespace) -> int:
    summary = merge_files(args.sources, args.out)
    log.info("merged: %s", summary)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="KathaPal dataset authoring tool.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("add", help="add and validate one record")
    a.add_argument("--dataset", required=True)
    a.add_argument("--from-json", default=None)
    a.add_argument("--interactive", action="store_true")
    a.set_defaults(func=_cmd_add)

    ln = sub.add_parser("lint", help="validate a dataset and summarize health")
    ln.add_argument("--dataset", required=True)
    ln.set_defaults(func=_cmd_lint)

    m = sub.add_parser("merge", help="merge several files (dedup by id)")
    m.add_argument("--out", required=True)
    m.add_argument("--sources", nargs="+", required=True)
    m.set_defaults(func=_cmd_merge)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
