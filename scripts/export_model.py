#!/usr/bin/env python3
"""
Export a trained run (adapter + tokenizer + configs + model card + manifest).

Usage:
    uv run python scripts/export_model.py --adapter outputs/adapters --out outputs/export
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kathapal.logging_config import configure_logging  # noqa: E402
from kathapal.models.export import export_run  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Export a trained KathaPal run.")
    ap.add_argument("--adapter", type=str, default="outputs/adapters")
    ap.add_argument("--out", type=str, default="outputs/export")
    ap.add_argument("--model-card", type=str, default=str(ROOT / "docs" / "model_card.md"))
    ap.add_argument("--eval-report", type=str, default="outputs/reports/evaluation_report.md")
    args = ap.parse_args()

    log = configure_logging()
    extras = [p for p in [args.model_card, args.eval_report] if p]
    result = export_run(args.adapter, args.out, extra_files=extras)
    log.info("Export complete: %s", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
