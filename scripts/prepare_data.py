#!/usr/bin/env python3
"""
Prepare (normalize + de-duplicate) the raw dataset into a processed JSONL,
then validate the processed output.

Usage:
    uv run python scripts/prepare_data.py --config configs/training.yaml
    uv run python scripts/prepare_data.py --raw data/samples/starter_dataset.jsonl \
        --out data/processed/dataset.jsonl
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kathapal.config import load_training_config  # noqa: E402
from kathapal.data.preparation import prepare_dataset  # noqa: E402
from kathapal.data.validation import validate_dataset  # noqa: E402
from kathapal.logging_config import configure_logging  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare the KathaPal dataset.")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--raw", type=str, default=None)
    ap.add_argument("--out", type=str, default="data/processed/dataset.jsonl")
    args = ap.parse_args()

    log = configure_logging()

    if args.raw:
        raw_path = args.raw
    elif args.config:
        raw_path = load_training_config(args.config).dataset_path
    else:
        ap.error("provide --config or --raw")
        return 2

    summary = prepare_dataset(raw_path, args.out)
    log.info("Prepared dataset: %s", summary)

    report = validate_dataset(args.out)
    log.info("Processed validation OK: %s (records=%d)", report.ok, report.total_records)
    if not report.ok:
        log.error("Processed dataset has problems; see validate_data.py for detail.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
