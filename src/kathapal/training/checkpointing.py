"""
Reproducibility metadata and checkpoint path helpers.

These are pure-Python (no torch) so they are offline-testable. The trainer uses
them to record exactly how a run was produced (seed, versions, dataset hash,
config) and to locate adapter/checkpoint directories.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def dataset_hash(path: str | Path) -> str:
    """SHA-256 of the dataset file, for provenance."""
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _optional_version(module: str) -> str | None:
    try:
        mod = __import__(module)
        return getattr(mod, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        return None


@dataclass
class RunMetadata:
    model_id: str
    seed: int
    dataset_path: str
    dataset_sha256: str
    config: dict[str, Any]
    device: str = "unknown"
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    platform: str = field(default_factory=platform.platform)
    torch_version: str | None = field(default_factory=lambda: _optional_version("torch"))
    transformers_version: str | None = field(
        default_factory=lambda: _optional_version("transformers")
    )
    peft_version: str | None = field(default_factory=lambda: _optional_version("peft"))

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json(), encoding="utf-8")


def adapter_dir(output_dir: str | Path) -> Path:
    return Path(output_dir) / "adapters"


def checkpoint_dir(output_dir: str | Path) -> Path:
    return Path(output_dir) / "checkpoints"
