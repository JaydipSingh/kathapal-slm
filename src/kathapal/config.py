"""
Configuration loading and validation.

All tunable values live in YAML (``configs/*.yaml``); this module parses them
into validated Pydantic models so that a bad config fails fast and clearly
before any training or generation begins.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------


class ModelConfig(BaseModel):
    model_config = {"extra": "forbid"}

    model_id: str = Field(..., description="HF model identifier for the base LM.")
    revision: str | None = None
    tokenizer_id: str | None = None
    license_notes: str = "UNVERIFIED - verify licence before any commercial use"
    license_verified: bool = False
    max_context_length: int = Field(1024, ge=128, le=8192)
    trust_remote_code: bool = False
    device_preference: str = Field("auto", pattern="^(auto|mps|cuda|cpu)$")
    local_cache_dir: str | None = None

    @field_validator("trust_remote_code")
    @classmethod
    def _warn_trust(cls, v: bool) -> bool:
        # Allowed but off by default; the loader logs a warning if enabled.
        return v


# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------


class LoRAConfig(BaseModel):
    model_config = {"extra": "forbid"}

    rank: int = Field(8, ge=1, le=256)
    alpha: int = Field(16, ge=1, le=512)
    dropout: float = Field(0.05, ge=0.0, le=0.9)
    target_modules: list[str] = Field(default_factory=lambda: ["q_proj", "v_proj"])


class TrainingConfig(BaseModel):
    model_config = {"extra": "forbid"}

    dataset_path: str
    output_dir: str = "outputs"
    seed: int = 42
    split_train: float = 0.8
    split_val: float = 0.1
    split_test: float = 0.1
    sequence_length: int = Field(512, ge=64, le=4096)
    batch_size: int = Field(1, ge=1, le=64)
    gradient_accumulation: int = Field(8, ge=1, le=256)
    learning_rate: float = Field(2e-4, gt=0, le=1e-1)
    epochs: int = Field(3, ge=1, le=100)
    weight_decay: float = Field(0.01, ge=0.0, le=1.0)
    warmup_ratio: float = Field(0.05, ge=0.0, le=0.5)
    checkpoint_every_steps: int = Field(50, ge=1)
    eval_every_steps: int = Field(50, ge=1)
    gradient_checkpointing: bool = False
    smoke_test: bool = False
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    model: ModelConfig

    @field_validator("split_test")
    @classmethod
    def _splits_sum_to_one(cls, v: float, info: Any) -> float:
        data = info.data
        tr = data.get("split_train", 0.0)
        va = data.get("split_val", 0.0)
        total = tr + va + v
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split_train + split_val + split_test must equal 1.0 (got {total})")
        return v


# ---------------------------------------------------------------------------
# Generation config
# ---------------------------------------------------------------------------


class GenerationConfig(BaseModel):
    model_config = {"extra": "forbid"}

    default_language: str = "en"
    default_topic: str = "animals"
    max_new_tokens: int = Field(400, ge=16, le=2048)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0)
    max_retries: int = Field(2, ge=0, le=10)
    timeout_seconds: float = Field(30.0, gt=0, le=600)
    deterministic: bool = False
    model: ModelConfig


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------


def _load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Config {p} must parse to a mapping, got {type(data).__name__}")
    return data


def load_training_config(path: str | Path) -> TrainingConfig:
    return TrainingConfig.model_validate(_load_yaml(path))


def load_generation_config(path: str | Path) -> GenerationConfig:
    return GenerationConfig.model_validate(_load_yaml(path))


def load_model_config(path: str | Path) -> ModelConfig:
    return ModelConfig.model_validate(_load_yaml(path))
