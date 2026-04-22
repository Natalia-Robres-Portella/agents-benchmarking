"""ReportGenerator — writes metrics.json and report.md after an experiment."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from src.schema import MetricResult


class ReportGenerator:
    """
    Writes two files into results/{run_id}/:
      metrics.json  — machine-readable metric vector
      report.md     — human-readable markdown summary (renders on GitHub)
    """

    def emit(
        self,
        run_dir: Path,
        metrics: Dict[str, MetricResult],
        config_snapshot: dict,
    ) -> None:
        self._write_json(run_dir, metrics, config_snapshot)
        self._write_markdown(run_dir, metrics, config_snapshot)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _write_json(
        run_dir: Path,
        metrics: Dict[str, MetricResult],
        config_snapshot: dict,
    ) -> None:
        agent_cfg = config_snapshot.get("agent", {})
        agent_id = (
            f"{agent_cfg.get('strategy', '?')}"
            f"__{agent_cfg.get('memory', {}).get('type', '?')}"
            f"__{agent_cfg.get('llm', {}).get('model', '?')}"
        )
        payload = {
            "run_id": run_dir.name,
            "agent_id": agent_id,
            "config": config_snapshot,
            "metrics": {
                name: {
                    "value": round(r.value, 4),
                    "breakdown": {k: round(v, 4) for k, v in r.breakdown.items()},
                }
                for name, r in metrics.items()
            },
        }
        (run_dir / "metrics.json").write_text(json.dumps(payload, indent=2))

    @staticmethod
    def _write_markdown(
        run_dir: Path,
        metrics: Dict[str, MetricResult],
        config_snapshot: dict,
    ) -> None:
        agent_cfg = config_snapshot.get("agent", {})
        exp_id = config_snapshot.get("id", run_dir.name)
        strategy = agent_cfg.get("strategy", "?")
        model = agent_cfg.get("llm", {}).get("model", "?")
        memory = agent_cfg.get("memory", {}).get("type", "?")

        lines = [
            f"# Experiment: {exp_id}",
            "",
            f"**Strategy:** {strategy} | **Model:** {model} | **Memory:** {memory}",
            "",
            "## Metrics",
            "",
            "| Metric | Value | Breakdown |",
            "|--------|-------|-----------|",
        ]
        for name, result in metrics.items():
            breakdown_str = (
                ", ".join(f"{k}={v:.3f}" for k, v in result.breakdown.items())
                if result.breakdown
                else "—"
            )
            lines.append(f"| {name} | {result.value:.4f} | {breakdown_str} |")

        lines += ["", "---", f"*Run directory: `{run_dir}`*", ""]
        (run_dir / "report.md").write_text("\n".join(lines))
