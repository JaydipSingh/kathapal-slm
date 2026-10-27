"""
Model export: package a trained run for distribution.

Collects the LoRA adapter, tokenizer, configs, model card, and a manifest
(versions + file hashes) into an export directory. Merging LoRA into the base
model is offered as a documented, optional step (RUN ON MAC; requires peft).
Adapter-based inference remains the default until a merged model is tested.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

from kathapal.logging_config import get_logger

log = get_logger()


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(export_dir: str | Path) -> dict:
    """Hash every file in the export dir into a manifest."""
    d = Path(export_dir)
    files = {}
    for p in sorted(d.rglob("*")):
        if p.is_file() and p.name != "manifest.json":
            files[str(p.relative_to(d))] = _file_sha256(p)
    return {"files": files, "count": len(files)}


def export_run(adapter_dir: str | Path, export_dir: str | Path, extra_files: list[str] | None = None) -> dict:
    """
    Copy the adapter directory and any extra files (model card, eval report) into
    a clean export directory and write a manifest. Pure file IO -- offline-safe.
    """
    src = Path(adapter_dir)
    dst = Path(export_dir)
    if not src.exists():
        raise FileNotFoundError(f"adapter dir not found: {src}")
    dst.mkdir(parents=True, exist_ok=True)

    for item in src.iterdir():
        target = dst / item.name
        if item.is_file():
            shutil.copy2(item, target)
        elif item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)

    for extra in extra_files or []:
        ep = Path(extra)
        if ep.exists():
            shutil.copy2(ep, dst / ep.name)
        else:
            log.warning("export extra file missing, skipping: %s", extra)

    manifest = build_manifest(dst)
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log.info("Exported %d files to %s", manifest["count"], dst)
    return {"export_dir": str(dst), "file_count": manifest["count"]}


def merge_lora_into_base(adapter_dir: str | Path, out_dir: str | Path, base_model_id: str) -> str:
    """
    Optional: merge LoRA weights into the base model for standalone inference.
    RUN ON MAC (requires torch + peft + transformers). Returns the output path.
    """
    import torch  # deferred
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base = AutoModelForCausalLM.from_pretrained(base_model_id, torch_dtype=torch.float32)
    merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    merged.save_pretrained(str(out_dir))
    AutoTokenizer.from_pretrained(str(adapter_dir)).save_pretrained(str(out_dir))
    log.info("Merged model saved to %s", out_dir)
    return str(out_dir)
