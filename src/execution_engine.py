"""
ExecutionEngine — the step loop.

Runs an agent on a single task for N trials.  Each trial:
  1. Call strategy.post_episode_hook() on the PREVIOUS trajectory so
     Reflexion can write reflections before memory is reset.
  2. agent.reset(trial_seed)
  3. Step loop: act → dispatch tool → observe → log → repeat
  4. logger.close_trajectory() → Trajectory

Termination reasons written to Trajectory.termination:
  "success"     — agent called the finish tool (or action_type="final_answer")
  "max_steps"   — loop exhausted config.max_steps
  "parse_error" — strategy returned action_type="abort"
  "llm_error"   — LLM backend raised an uncaught exception
"""
from __future__ import annotations

import logging
import time
from typing import List, Optional

from src.agents.base import Agent
from src.config import ExperimentConfig
from src.schema import Action, AgentState, Observation, Step, TaskInstance, Trajectory
from src.tools.base import ToolRegistry
from src.trace_logger import TraceLogger

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """Runs an agent on a task for n_trials, returning all Trajectory objects."""

    def __init__(
        self,
        config: ExperimentConfig,
        agent: Agent,
        trace_logger: TraceLogger,
        tools: ToolRegistry,
        run_id: str,
        config_hash: str,
    ) -> None:
        self._config = config
        self._agent = agent
        self._logger = trace_logger
        self._tools = tools
        self._run_id = run_id
        self._config_hash = config_hash

    def run(self, task: TaskInstance) -> List[Trajectory]:
        trajectories: List[Trajectory] = []
        prev_traj: Optional[Trajectory] = None

        for trial in range(self._config.n_trials):
            trial_seed = self._config.seed + trial

            # Reflexion: write reflection from previous trial before memory reset
            if prev_traj is not None:
                self._agent.strategy.post_episode_hook(prev_traj, self._agent)

            self._agent.reset(trial_seed)

            self._logger.open_trajectory(
                run_id=self._run_id,
                task_id=task.task_id,
                agent_id=self._agent.agent_id,
                trial_num=trial,
                seed=trial_seed,
                config_hash=self._config_hash,
            )

            obs = Observation(
                content=task.input, source="environment", is_terminal=False
            )
            steps_so_far: List[Step] = []
            termination = "max_steps"
            final_answer: Optional[str] = None

            for step_num in range(self._config.max_steps):
                state = AgentState(
                    task=task,
                    history=steps_so_far,
                    observation=obs,
                    step_num=step_num,
                )

                t0 = time.monotonic()
                try:
                    action = self._agent.act(state)
                except Exception as exc:
                    logger.warning("LLM error at step %d: %s", step_num, exc)
                    action = Action(action_type="abort", raw_llm_out=str(exc))
                    termination = "llm_error"

                latency_ms = (time.monotonic() - t0) * 1000

                # ---- abort ----
                if action.action_type == "abort":
                    if termination != "llm_error":
                        termination = "parse_error"
                    obs = Observation(
                        content="Parse error — aborting.", source="system", is_terminal=True
                    )
                    step_rec = self._make_step(step_num, action, obs, latency_ms)
                    steps_so_far.append(step_rec)
                    self._logger.log_step(step_rec)
                    break

                # ---- final_answer (direct strategy) ----
                if action.action_type == "final_answer":
                    final_answer = action.final_answer
                    termination = "success"
                    obs = Observation(
                        content=final_answer or "",
                        source="agent",
                        is_terminal=True,
                    )
                    step_rec = self._make_step(step_num, action, obs, latency_ms)
                    steps_so_far.append(step_rec)
                    self._logger.log_step(step_rec)
                    break

                # ---- tool_call ----
                tool_result = self._tools.validate_and_execute(
                    action.tool_name or "", action.tool_args or {}
                )
                obs_content = tool_result.output or tool_result.error or ""
                obs = Observation(
                    content=obs_content,
                    source=action.tool_name or "unknown_tool",
                    is_terminal=bool(tool_result.metadata.get("terminal")),
                    error=tool_result.error,
                )

                # FinishTool sets metadata["terminal"] = True
                if obs.is_terminal:
                    final_answer = tool_result.output
                    termination = "success"

                self._agent.observe(obs)

                step_rec = self._make_step(
                    step_num, action, obs, latency_ms,
                    tool_error=tool_result.error,
                    arg_valid=tool_result.arg_valid,
                )
                steps_so_far.append(step_rec)
                self._logger.log_step(step_rec)

                if obs.is_terminal:
                    break

            traj = self._logger.close_trajectory(final_answer, termination)
            trajectories.append(traj)
            prev_traj = traj

        return trajectories

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_step(
        step_id: int,
        action: Action,
        obs: Observation,
        latency_ms: float,
        tool_error: Optional[str] = None,
        arg_valid: bool = True,
    ) -> Step:
        return Step(
            step_id=step_id,
            thought=action.thought,
            action=action,
            observation=obs,
            tokens_in=action.tokens_in,
            tokens_out=action.tokens_out,
            latency_ms=latency_ms,
            tool_error=tool_error,
            arg_valid=arg_valid,
        )
