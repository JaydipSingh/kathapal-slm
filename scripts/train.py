#!/usr/bin/env python3
"""
Train KathaPal-SLM with LoRA. RUN ON MAC (requires the `train` extras).

Usage:
    uv run python scripts/train.py --config configs/training.yaml --smoke-test
    uv run python scripts/train.py --config configs/training.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kathapal.config import load_training_config  # noqa: E402
from kathapal.logging_config import configure_logging  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Train KathaPal-SLM (LoRA).")
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--smoke-test", action="store_true", help="tiny run to verify wiring")
    args = ap.parse_args()

    log = configure_logging()
    cfg = load_training_config(args.config)
    if args.smoke_test:
        cfg = cfg.model_copy(update={"smoke_test": True})

    if not cfg.model.license_verified:
        log.warning(
            "Base model licence is NOT marked verified (%s). Verify its licence "
            "and acceptable-use terms before any commercial use.", cfg.model.model_id
        )

    try:
        from kathapal.training.trainer import train
    except ImportError as exc:
        log.error("Training requires the 'train' extras (torch, transformers, peft, datasets).")
        log.error("Install with: uv sync --extra train   (import error: %s)", exc)
        return 2

    summary = train(cfg)
    log.info("Done. Run inference with: uv run python scripts/generate.py --topic animals --age 4")
    log.info("Summary: %s", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
