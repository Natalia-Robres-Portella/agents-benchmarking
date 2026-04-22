"""
Trace logger — streams trajectory data to JSONL files (crash-safe).

Two output files per run:
  traces.jsonl       — one line per Step (written and flushed on every log_step)
  trajectories.jsonl — one line per completed Trajectory (written on close)

Because each line is independently valid JSON, a run that crashes mid-way
still produces a complete record of all steps up to the crash point.

SQLite index is an advanced-tier feature (Step 7A in the plan).
"""
from __future__ import annotations

import json
from io import TextIOWrapper
from pathlib import Path
from typing import IO, Dict, List, Optional

from src.schema import Step, Trajectory


class TraceLogger:
    """
    Streaming JSONL logger.

    Usage:
        logger = TraceLogger(run_dir, save_traces=True)
        logger.open_trajectory(run_id, task_id, agent_id, trial_num, seed, config_hash)
        logger.log_step(step)   # called after every env.step()
        traj = logger.close_trajectory(final_answer, termination)
    """

    def __init__(self, run_dir: Path, save_traces: bool = True) -> None:
        self._run_dir = run_dir
        self._save = save_traces
        self._trace_fh: Optional[IO[str]] = None
        self._meta: Dict = {}
        self._current_steps: List[Step] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def open_trajectory(
        self,
        run_id: str,
        task_id: str,
        agent_id: str,
        trial_num: int,
        seed: int,
        config_hash: str,
    ) -> None:
        self._meta = dict(
            run_id=run_id,
            task_id=task_id,
            agent_id=agent_id,
            trial_num=trial_num,
            seed=seed,
            config_hash=config_hash,
        )
        self._current_steps = []
        if self._save and self._trace_fh is None:
            self._trace_fh = open(self._run_dir / "traces.jsonl", "a")

    def log_step(self, step: Step) -> None:
        self._current_steps.append(step)
        if not self._save or self._trace_fh is None:
            return
        record = {**self._meta, **step.model_dump()}
        self._trace_fh.write(json.dumps(record) + "\n")
        self._trace_fh.flush()  # crash-safe: every step is durable immediately

    def close_trajectory(
        self, final_answer: Optional[str], termination: str
    ) -> Trajectory:
        traj = Trajectory(
            **self._meta,
            steps=list(self._current_steps),
            termination=termination,
            final_answer=final_answer,
            total_tokens=sum(
                s.tokens_in + s.tokens_out for s in self._current_steps
            ),
            total_latency_ms=sum(s.latency_ms for s in self._current_steps),
        )
        if self._save:
            traj_path = self._run_dir / "trajectories.jsonl"
            with open(traj_path, "a") as fh:
                fh.write(traj.model_dump_json() + "\n")
        return traj

    def load_trajectories(self, run_dir: str) -> List[Trajectory]:
        path = Path(run_dir) / "trajectories.jsonl"
        if not path.exists():
            return []
        return [
            Trajectory.model_validate_json(line)
            for line in path.read_text().splitlines()
            if line.strip()
        ]

    def close(self) -> None:
        """Flush and close the open trace file handle."""
        if self._trace_fh is not None:
            self._trace_fh.flush()
            self._trace_fh.close()
            self._trace_fh = None
