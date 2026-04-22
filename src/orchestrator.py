"""
ExperimentOrchestrator — top-level coordinator.

Lifecycle:
  1. Parse + validate ExperimentConfig
  2. seed_everything(config.seed)
  3. Create results/{run_id}/, snapshot config.yaml
  4. Instantiate all components via factories
  5. For each task: engine.run(task)  →  trajectories
  6. EvaluationModule.compute_all(all_trajectories, tasks)  →  metrics
  7. ReportGenerator.emit(run_dir, metrics, config_snapshot)
  8. TraceLogger.close()
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import List, Optional

import yaml

from src.config import ExperimentConfig, compute_config_hash
from src.utils import configure_logging, make_run_id, seed_everything

logger = logging.getLogger(__name__)


class ExperimentOrchestrator:

    def __init__(
        self,
        config: ExperimentConfig,
        config_path: Optional[str] = None,
    ) -> None:
        self.config = config
        self.config_path = config_path
        self.config_hash = compute_config_hash(config)
        self.run_id = make_run_id(config.id, self.config_hash)

    def run(self) -> None:
        configure_logging(self.config.logging.level)
        seed_everything(self.config.seed)
        logger.info("Run %s starting (seed=%s)", self.run_id, self.config.seed)

        # ── output directory ──────────────────────────────────────────
        run_dir = Path(self.config.output_dir) / self.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        self._snapshot_config(run_dir)

        # ── lazy-import loaders so they register with TASK_REGISTRY ──
        import src.tasks.loaders  # noqa: F401

        # ── instantiate components ────────────────────────────────────
        from src.agents.factory import build_agent
        from src.evaluation.module import EvaluationModule
        from src.reporting.aggregator import ResultAggregator
        from src.reporting.report_generator import ReportGenerator
        from src.tasks.base import TASK_REGISTRY
        from src.trace_logger import TraceLogger
        from src.execution_engine import ExecutionEngine

        agent = build_agent(self.config.agent)
        task_loader = TASK_REGISTRY.get(self.config.tasks.dataset)
        tasks = task_loader.load(
            split=self.config.tasks.split,
            n_samples=self.config.tasks.n_samples,
            seed=self.config.seed,
            filter_kwargs=self.config.tasks.filter.model_dump()
            if self.config.tasks.filter
            else None,
        )
        trace_logger = TraceLogger(
            run_dir, save_traces=self.config.logging.save_traces
        )
        engine = ExecutionEngine(
            config=self.config,
            agent=agent,
            trace_logger=trace_logger,
            tools=agent.tools,
            run_id=self.run_id,
            config_hash=self.config_hash,
        )
        evaluator = EvaluationModule(self.config.evaluation)
        aggregator = ResultAggregator()
        reporter = ReportGenerator()

        # ── run all tasks ─────────────────────────────────────────────
        all_trajectories = []
        for i, task in enumerate(tasks):
            logger.info(
                "[%d/%d] task=%s", i + 1, len(tasks), task.task_id
            )
            trajs = engine.run(task)
            all_trajectories.extend(trajs)

            # Reset between tasks: EpisodicMemory.hard_reset() if applicable
            mem = agent.memory
            if hasattr(mem, "hard_reset"):
                mem.hard_reset()
            else:
                mem.reset()

        trace_logger.close()

        # ── evaluate ─────────────────────────────────────────────────
        metrics = evaluator.compute_all(all_trajectories, tasks)
        aggregator.aggregate(all_trajectories)  # groups for future use

        # ── report ───────────────────────────────────────────────────
        reporter.emit(run_dir, metrics, self.config.model_dump())

        logger.info(
            "Run %s complete — results in %s", self.run_id, run_dir
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _snapshot_config(self, run_dir: Path) -> None:
        """Save the exact config that produced this run for reproducibility."""
        if self.config_path and Path(self.config_path).exists():
            shutil.copy(self.config_path, run_dir / "config.yaml")
        else:
            # Serialise from the Pydantic model if no original file is available
            (run_dir / "config.yaml").write_text(
                yaml.dump(self.config.model_dump(), default_flow_style=False)
            )
