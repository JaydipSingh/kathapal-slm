#!/usr/bin/env python3
"""
Generate a story from a structured request.

Uses the real HF model (RUN ON MAC) when the `train` extras + adapter are
available; otherwise falls back to a stub adapter so the pipeline is
demonstrable offline (clearly logged). Always returns validated, safe output.

Usage:
    uv run python scripts/generate.py --topic animals --age 4 --language en
    uv run python scripts/generate.py --topic kindness --age 5 --adapter outputs/adapters
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kathapal.config import load_generation_config  # noqa: E402
from kathapal.logging_config import configure_logging  # noqa: E402
from kathapal.models.base import GenerationParams  # noqa: E402
from kathapal.models.generation import generate_story  # noqa: E402
from kathapal.models.prompting import StoryRequest  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate a KathaPal story.")
    ap.add_argument("--config", type=str, default=str(ROOT / "configs" / "generation.yaml"))
    ap.add_argument("--topic", type=str, required=True)
    ap.add_argument("--age", type=int, required=True)
    ap.add_argument("--language", type=str, default="en")
    ap.add_argument("--learning-goal", type=str, default=None)
    ap.add_argument("--adapter", type=str, default=None, help="LoRA adapter dir (Mac)")
    ap.add_argument("--stub", action="store_true", help="force offline stub adapter")
    args = ap.parse_args()

    log = configure_logging()
    cfg = load_generation_config(args.config)
    params = GenerationParams(
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        repetition_penalty=cfg.repetition_penalty,
        deterministic=cfg.deterministic,
    )
    req = StoryRequest(
        age=args.age, language=args.language, topic=args.topic, learning_goal=args.learning_goal
    )

    adapter = None
    if not args.stub:
        try:
            from kathapal.models.loader import HFModelAdapter
            adapter = HFModelAdapter(cfg.model)
            if args.adapter:
                # Load LoRA adapter on top of the base model.
                from peft import PeftModel  # noqa: F401
                adapter.model.load_adapter(args.adapter) if hasattr(adapter.model, "load_adapter") else None
        except Exception as exc:  # noqa: BLE001
            log.warning("Real model unavailable (%s); using offline stub adapter.", exc)

    if adapter is None:
        from kathapal.models.base import StubModelAdapter
        adapter = StubModelAdapter()
        log.info("Using STUB adapter (offline demo; not the trained model).")

    outcome = generate_story(adapter, req, params, max_retries=cfg.max_retries)
    r = outcome.response
    print("=" * 60)
    print(f"TITLE: {r.story_title}")
    print(f"TOPIC: {r.topic} | AGE: {r.age_range} | LANG: {r.language}")
    print("-" * 60)
    print(r.story)
    print("-" * 60)
    print(f"Q: {r.question}")
    print(f"A: {r.answer}")
    print(f"LEARNING GOAL: {r.learning_goal}")
    print("=" * 60)
    if outcome.used_fallback:
        log.warning("Used safe fallback (reason: %s)", outcome.reason)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
