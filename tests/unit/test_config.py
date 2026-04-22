"""Tests for src/config.py."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.config import ExperimentConfig, compute_config_hash, load_config


def test_load_config_sets_id(minimal_config_yaml: Path) -> None:
    cfg = load_config(minimal_config_yaml)
    assert cfg.id == "test_experiment"


def test_load_config_applies_defaults(minimal_config_yaml: Path) -> None:
    cfg = load_config(minimal_config_yaml)
    assert cfg.agent.llm.temperature == 0.0
    assert cfg.evaluation.validator == "fuzzy_match"
    assert cfg.logging.save_traces is True


def test_load_config_overrides_defaults(minimal_config_yaml: Path) -> None:
    cfg = load_config(minimal_config_yaml)
    assert cfg.agent.llm.model == "gpt-4o"   # set in fixture, not base
    assert cfg.seed == 42
    assert cfg.n_trials == 2                  # overrides base default of 5


def test_config_hash_is_deterministic(minimal_config_yaml: Path) -> None:
    cfg = load_config(minimal_config_yaml)
    assert compute_config_hash(cfg) == compute_config_hash(cfg)


def test_config_hash_is_hex_64(minimal_config_yaml: Path) -> None:
    cfg = load_config(minimal_config_yaml)
    h = compute_config_hash(cfg)
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_config_hash_changes_with_seed(minimal_config_yaml: Path) -> None:
    cfg1 = load_config(minimal_config_yaml)
    cfg2 = cfg1.model_copy(update={"seed": 99})
    assert compute_config_hash(cfg1) != compute_config_hash(cfg2)


def test_missing_id_raises(tmp_path: Path) -> None:
    config_file = tmp_path / "bad.yaml"
    config_file.write_text("experiment:\n  seed: 1\n")
    with pytest.raises(Exception):   # Pydantic ValidationError
        load_config(config_file)


def test_invalid_strategy_raises(tmp_path: Path) -> None:
    config_file = tmp_path / "bad.yaml"
    config_file.write_text(
        "experiment:\n  id: x\nagent:\n  strategy: nonexistent\n"
    )
    with pytest.raises(Exception):
        load_config(config_file)
