#!/usr/bin/env python3
"""
Evaluate the model against the template baseline (and the base model on Mac).

Builds a fixed request set across allowed topics, runs each arm, and writes a
Markdown + JSON report to outputs/reports/.

Usage:
    uv run python scripts/evaluate.py --config configs/generation.yaml
    uv run python scripts/evaluate.py --stub   # offline demo with stub model
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kathapal.config import load_generation_config  # noqa: E402
from kathapal.evaluation.evaluator import evaluate_adapter, evaluate_template_baseline  # noqa: E402
from kathapal.evaluation.report import write_reports  # noqa: E402
from kathapal.logging_config import configure_logging  # noqa: E402
from kathapal.models.base import GenerationParams  # noqa: E402
from kathapal.models.prompting import StoryRequest  # noqa: E402

_EVAL_REQUESTS = [
    StoryRequest(age=4, language="en", topic="animals", learning_goal="kindness"),
    StoryRequest(age=5, language="en", topic="friendship", learning_goal="helping"),
    StoryRequest(age=3, language="en", topic="nature", learning_goal="curiosity"),
    StoryRequest(age=4, language="en", topic="kindness", learning_goal="sharing"),
    StoryRequest(age=5, language="en", topic="emotions", learning_goal="calm"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate KathaPal-SLM.")
    ap.add_argument("--config", type=str, default=str(ROOT / "configs" / "generation.yaml"))
    ap.add_argument("--adapter", type=str, default=None)
    ap.add_argument("--stub", action="store_true")
    ap.add_argument("--out", type=str, default=str(ROOT / "outputs" / "reports"))
    args = ap.parse_args()

    log = configure_logging()
    cfg = load_generation_config(args.config)
    params = GenerationParams(max_new_tokens=cfg.max_new_tokens, deterministic=True)

    arms = []

    # Model arm (real on Mac, stub offline).
    adapter = None
    if not args.stub:
        try:
            from kathapal.models.loader import HFModelAdapter
            adapter = HFModelAdapter(cfg.model)
        except Exception as exc:  # noqa: BLE001
            log.warning("Real model unavailable (%s); evaluating stub only.", exc)
    if adapter is None:
        from kathapal.models.base import StubModelAdapter
        adapter = StubModelAdapter()
        log.info("Using STUB model arm (offline demo).")

    arms.append(evaluate_adapter("kathapal-model", adapter, _EVAL_REQUESTS, params))
    arms.append(evaluate_template_baseline(_EVAL_REQUESTS))

    meta = {"model": adapter.name, "n_requests": len(_EVAL_REQUESTS)}
    paths = write_reports(arms, args.out, meta=meta)
    log.info("Reports written: %s", paths)
    for arm in arms:
        log.info("Arm %s: %s", arm.name, arm.aggregate())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
