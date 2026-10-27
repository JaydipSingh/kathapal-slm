"""
Training callbacks.

Early stopping is delegated to the Transformers ``EarlyStoppingCallback`` when a
validation set and a compatible eval strategy are available, rather than
reimplementing it. This helper builds that callback on demand (RUN ON MAC;
deferred import). Kept separate so the trainer stays readable.
"""
from __future__ import annotations


def build_early_stopping(patience: int = 2):
    """Return a Transformers EarlyStoppingCallback. RUN ON MAC."""
    from transformers import EarlyStoppingCallback

    return EarlyStoppingCallback(early_stopping_patience=patience)
