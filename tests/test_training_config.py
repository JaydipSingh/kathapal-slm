"""Tests for training config validation and the pure-Python training helpers."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kathapal.config import load_training_config  # noqa: E402
from kathapal.data.loader import load_records  # noqa: E402
from kathapal.training.checkpointing import RunMetadata, dataset_hash  # noqa: E402
from kathapal.training.trainer import build_examples_text, compute_label_mask  # noqa: E402

STARTER = ROOT / "data" / "samples" / "starter_dataset.jsonl"


def test_training_config_loads():
    cfg = load_training_config(ROOT / "configs" / "training.yaml")
    assert cfg.lora.rank == 8
    assert abs(cfg.split_train + cfg.split_val + cfg.split_test - 1.0) < 1e-6


def test_config_rejects_bad_splits(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        "dataset_path: d.jsonl\nsplit_train: 0.5\nsplit_val: 0.3\nsplit_test: 0.3\n"
        "model:\n  model_id: m\n",
        encoding="utf-8",
    )
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        load_training_config(bad)


def test_compute_label_mask():
    mask = compute_label_mask(prompt_len=3, total_len=5)
    assert mask == [-100, -100, -100, 1, 1]


def test_compute_label_mask_rejects_bad_lengths():
    with pytest.raises(ValueError):
        compute_label_mask(prompt_len=5, total_len=3)


def test_build_examples_text_masks_prompt_prefix():
    records, _ = load_records(STARTER)
    pairs = build_examples_text(records[:3])
    assert len(pairs) == 3
    for prompt_text, full_text in pairs:
        # full_text must start with the prompt_text (so prefix masking is valid)
        assert full_text.startswith(prompt_text)
        # the response JSON must be the suffix
        assert full_text[len(prompt_text):].strip().startswith("{")


def test_dataset_hash_stable():
    h1 = dataset_hash(STARTER)
    h2 = dataset_hash(STARTER)
    assert h1 == h2 and len(h1) == 64


def test_run_metadata_serializes():
    meta = RunMetadata(
        model_id="m", seed=42, dataset_path=str(STARTER),
        dataset_sha256=dataset_hash(STARTER), config={"a": 1},
    )
    js = meta.to_json()
    assert '"seed": 42' in js and '"model_id": "m"' in js
