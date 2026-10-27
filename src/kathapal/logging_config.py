"""
Structured logging setup.

Deliberately simple and side-effect-free on import: call ``configure_logging()``
explicitly from entry points. We never log children's personal data; callers
must not pass raw end-user content into log messages at INFO or above.
"""
from __future__ import annotations

import logging
import sys


def configure_logging(level: str = "INFO") -> logging.Logger:
    """Configure and return the root 'kathapal' logger."""
    logger = logging.getLogger("kathapal")
    if logger.handlers:  # idempotent: don't add duplicate handlers
        return logger
    handler = logging.StreamHandler(stream=sys.stdout)
    fmt = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    logger.setLevel(level.upper())
    logger.propagate = False
    return logger


def get_logger(name: str = "kathapal") -> logging.Logger:
    return logging.getLogger(name)
