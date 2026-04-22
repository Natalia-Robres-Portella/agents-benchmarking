"""
Central data model definitions for the agent benchmark suite.

Import rule: this module imports from nothing inside src/.
All other modules import from here — never the reverse.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Primitive types
# ---------------------------------------------------------------------------

ActionType = Literal["tool_call", "final_answer", "reflect", "abort"]


# ---------------------------------------------------------------------------
# Tool layer
# ---------------------------------------------------------------------------

class ToolResult(BaseModel):
    """Returned by every tool execution, successful or not."""
    output: str
    error: Optional[str] = None
    arg_valid: bool = True          # False when JSON-Schema validation fails
    metadata: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Environment / observation layer
# ---------------------------------------------------------------------------

class Observation(BaseModel):
    """What the agent perceives after an action or at episode start."""
    content: str
    source: str                     # tool name, "environment", or "system"
    is_terminal: bool = False
    error: Optional[str] = None


class StepResult(BaseModel):
    """Raw result returned by Environment.step()."""
    observation: Observation
    is_terminal: bool
    tool_error: Optional[str] = None


# ---------------------------------------------------------------------------
# Action layer
# ---------------------------------------------------------------------------

class Action(BaseModel):
    """Everything the agent decided in a single step."""
    action_type: ActionType
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None
    final_answer: Optional[str] = None
    thought: Optional[str] = None   # extracted CoT / reasoning trace
    raw_llm_out: str = ""           # full unparsed LLM response (for logs)
    token_count: int = 0
    tokens_in: int = 0              # prompt tokens (set by BaseAgent.act)
    tokens_out: int = 0             # completion tokens (set by BaseAgent.act)


# ---------------------------------------------------------------------------
# Trajectory layer
# ---------------------------------------------------------------------------

class Step(BaseModel):
    """A single (action, observation) transition within an episode."""
    step_id: int
    thought: Optional[str] = None
    action: Action
    observation: Observation
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: float = 0.0
    tool_error: Optional[str] = None
    arg_valid: bool = True


class Trajectory(BaseModel):
    """Complete record of one agent trial on one task."""
    run_id: str
    task_id: str
    agent_id: str                   # strategy__memory__model fingerprint
    trial_num: int
    seed: int
    config_hash: str                # SHA-256 of the experiment config
    steps: List[Step] = Field(default_factory=list)
    termination: str = "unknown"    # success | max_steps | llm_error | ...
    final_answer: Optional[str] = None
    total_tokens: int = 0
    total_latency_ms: float = 0.0
    success: bool = False
    score: float = 0.0


# ---------------------------------------------------------------------------
# Task layer
# ---------------------------------------------------------------------------

class TaskInstance(BaseModel):
    """A single benchmark task, fully serialisable."""
    task_id: str
    input: str                          # the question / instruction shown to agent
    gold: Any                           # ground-truth answer (type depends on task)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    env_config: Optional[Dict[str, Any]] = None  # embodied env params (ALFWorld etc.)


class AgentState(BaseModel):
    """Snapshot of everything the agent can see when act() is called."""
    task: TaskInstance
    history: List[Step] = Field(default_factory=list)
    observation: Observation            # most recent observation
    step_num: int = 0
    token_budget: Optional[int] = None  # remaining tokens, if enforced


# ---------------------------------------------------------------------------
# Evaluation layer
# ---------------------------------------------------------------------------

class MetricResult(BaseModel):
    """Output of a single metric computation over a set of trajectories."""
    name: str
    value: float
    breakdown: Dict[str, float] = Field(default_factory=dict)  # per-task scores
    metadata: Dict[str, Any] = Field(default_factory=dict)
