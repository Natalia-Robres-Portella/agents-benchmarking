"""
Integration smoke test — full pipeline with mocked LLM.

Verifies that:
  1. ExperimentOrchestrator.run() executes without errors
  2. metrics.json and report.md are written to results/{run_id}/
  3. trajectories.jsonl contains the expected number of entries
  4. No real OpenAI API calls are made
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from src.config import load_config
from src.llm.base import LLMResponse
from src.orchestrator import ExperimentOrchestrator


# ---------------------------------------------------------------------------
# Fixture — minimal experiment config using offline fixture dataset
# ---------------------------------------------------------------------------

@pytest.fixture
def experiment_yaml(tmp_path: Path) -> Path:
    content = textwrap.dedent(f"""\
        experiment:
          id: "smoke_test"
          seed: 0
          n_trials: 2
          max_steps: 3
          output_dir: "{tmp_path}"
          tags: ["smoke"]

        agent:
          strategy: "react"
          llm:
            provider: "openai"
            model: "gpt-4o"
            temperature: 0.0
          memory:
            type: "no_memory"
          tools:
            - name: "finish"
            - name: "search"

        tasks:
          dataset: "hotpotqa"
          split: "validation"
          n_samples: 2

        evaluation:
          metrics:
            - "success_rate"
            - "pass_at_k"
            - "tokens_per_task"
          validator: "fuzzy_match"
          pass_k_values: [1, 2]

        logging:
          level: "WARNING"
          save_traces: true
    """)
    p = tmp_path / "smoke.yaml"
    p.write_text(content)
    return p


# ---------------------------------------------------------------------------
# Mock LLM — always calls finish with a deterministic answer
# ---------------------------------------------------------------------------

def _fake_complete(self, prompt: str, stop=None, **kwargs) -> LLMResponse:
    """Simulates ReAct agent immediately calling finish."""
    return LLMResponse(
        content=(
            "Thought: I know the answer directly.\n"
            "Action: finish\n"
            'Action Input: {"answer": "Paris"}'
        ),
        tokens_in=60,
        tokens_out=25,
        model="gpt-4o",
        latency_ms=50.0,
    )


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------

def test_full_pipeline(
    experiment_yaml: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.llm.openai_backend import OpenAIBackend
    monkeypatch.setattr(OpenAIBackend, "complete", _fake_complete)

    cfg = load_config(experiment_yaml)
    orch = ExperimentOrchestrator(cfg, config_path=str(experiment_yaml))
    orch.run()

    run_dir = Path(cfg.output_dir) / orch.run_id

    # ── output files exist ────────────────────────────────────────────
    assert (run_dir / "metrics.json").exists(), "metrics.json missing"
    assert (run_dir / "report.md").exists(), "report.md missing"
    assert (run_dir / "config.yaml").exists(), "config.yaml snapshot missing"
    assert (run_dir / "trajectories.jsonl").exists(), "trajectories.jsonl missing"

    # ── metrics.json is valid and contains expected keys ──────────────
    metrics = json.loads((run_dir / "metrics.json").read_text())
    assert "success_rate" in metrics["metrics"]
    assert "pass_at_k" in metrics["metrics"]
    assert "tokens_per_task" in metrics["metrics"]

    # ── trajectories: n_samples × n_trials lines ─────────────────────
    traj_lines = (run_dir / "trajectories.jsonl").read_text().strip().splitlines()
    expected = cfg.tasks.n_samples * cfg.n_trials
    assert len(traj_lines) == expected, (
        f"Expected {expected} trajectories, got {len(traj_lines)}"
    )

    # ── every trajectory terminated with success ──────────────────────
    for line in traj_lines:
        obj = json.loads(line)
        assert obj["termination"] == "success", f"Unexpected termination: {obj}"
        assert obj["final_answer"] == "Paris"

    # ── report.md is valid markdown with a table ─────────────────────
    report = (run_dir / "report.md").read_text()
    assert "success_rate" in report
    assert "smoke_test" in report
