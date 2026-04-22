"""Shared utilities: seeding, logging setup, run-id generation."""
from __future__ import annotations

import logging
import random
import uuid
from datetime import datetime, timezone


def seed_everything(seed: int) -> None:
    """Seed all RNG sources before any component is instantiated."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    logging.getLogger(__name__).debug(f"All RNGs seeded with seed={seed}")


def make_run_id(experiment_id: str, config_hash: str) -> str:
    """
    Deterministic-ish run ID: experiment name + timestamp + hash prefix.
    Timestamp ensures uniqueness when the same config is re-run.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{experiment_id}__{ts}__{config_hash[:8]}"


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
