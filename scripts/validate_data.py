#!/usr/bin/env python3
"""
Validate the dataset and print a report.

Usage:
    uv run python scripts/validate_data.py --config configs/training.yaml
    uv run python scripts/validate_data.py --data data/samples/starter_dataset.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make `src` importable when run directly.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kathapal.config import load_training_config  # noqa: E402
from kathapal.data.validation import validate_dataset  # noqa: E402
from kathapal.logging_config import configure_logging  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Validate the KathaPal dataset.")
    ap.add_argument("--config", type=str, default=None, help="Path to training.yaml")
    ap.add_argument("--data", type=str, default=None, help="Path to a JSONL dataset")
    ap.add_argument("--json", action="store_true", help="Print the report as JSON")
    args = ap.parse_args()

    log = configure_logging()

    if args.data:
        data_path = args.data
    elif args.config:
        data_path = load_training_config(args.config).dataset_path
    else:
        ap.error("provide --config or --data")
        return 2

    report = validate_dataset(data_path)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        log.info("Dataset: %s", data_path)
        log.info("Total records: %d", report.total_records)
        log.info("By language: %s", report.by_language)
        log.info("By topic: %s", report.by_topic)
        log.info("By age range: %s", report.by_age_range)
        if report.load_errors:
            log.error("Load errors: %d", len(report.load_errors))
            for e in report.load_errors:
                log.error("  line %d: %s", e.line_number, e.error)
        if report.duplicate_ids:
            log.error("Duplicate IDs: %s", report.duplicate_ids)
        if report.near_duplicate_instructions:
            log.warning("Near-duplicate instructions: %s", report.near_duplicate_instructions)
        if report.excessive_length_ids:
            log.error("Excessive length: %s", report.excessive_length_ids)
        if report.safety_failures:
            log.error("Safety failures: %s", report.safety_failures)
        if report.missing_provenance_ids:
            log.error("Missing provenance/licence: %s", report.missing_provenance_ids)
        log.info("Overall OK: %s", report.ok)

    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
