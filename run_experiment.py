#!/usr/bin/env python3
"""
CLI entry point.

Usage:
    python run_experiment.py --config configs/experiments/react_hotpotqa.yaml
    python run_experiment.py --config configs/experiments/react_hotpotqa.yaml --dry-run
"""
from __future__ import annotations

import sys

import click
from rich.console import Console
from rich.panel import Panel

console = Console()


@click.command()
@click.option(
    "--config",
    required=True,
    type=click.Path(exists=True),
    help="Path to experiment YAML config.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Validate config and print summary without running.",
)
def main(config: str, dry_run: bool) -> None:
    """Run an agent benchmark experiment."""
    from src.config import compute_config_hash, load_config

    cfg = load_config(config)
    h = compute_config_hash(cfg)

    console.print(
        Panel(
            f"[bold]{cfg.id}[/bold]\n"
            f"strategy=[cyan]{cfg.agent.strategy}[/] | "
            f"model=[cyan]{cfg.agent.llm.model}[/] | "
            f"memory=[cyan]{cfg.agent.memory.type}[/]\n"
            f"tasks=[cyan]{cfg.tasks.dataset}[/] n={cfg.tasks.n_samples} | "
            f"trials=[cyan]{cfg.n_trials}[/] | seed=[cyan]{cfg.seed}[/]\n"
            f"config_hash=[dim]{h[:16]}…[/]",
            title="[bold green]Agent Benchmark Suite[/]",
        )
    )

    if dry_run:
        console.print("[yellow]Dry run — config valid, exiting.[/]")
        sys.exit(0)

    from src.orchestrator import ExperimentOrchestrator

    orchestrator = ExperimentOrchestrator(cfg)
    orchestrator.run()


if __name__ == "__main__":
    main()
