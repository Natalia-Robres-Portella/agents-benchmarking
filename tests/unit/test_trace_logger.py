"""Unit tests for TraceLogger — JSONL output, crash-safety, round-trip."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.schema import Action, Observation, Step, Trajectory
from src.trace_logger import TraceLogger


def _make_step(step_id: int) -> Step:
    return Step(
        step_id=step_id,
        action=Action(action_type="tool_call", tool_name="search", tool_args={"query": "x"}, raw_llm_out=""),
        observation=Observation(content=f"obs_{step_id}", source="search"),
        tokens_in=10,
        tokens_out=5,
        latency_ms=100.0,
    )


@pytest.fixture
def logger_in_tmp(tmp_path: Path) -> TraceLogger:
    return TraceLogger(tmp_path, save_traces=True)


def test_trace_file_created(logger_in_tmp: TraceLogger, tmp_path: Path) -> None:
    logger_in_tmp.open_trajectory("r1", "t1", "react__no__gpt-4o", 0, 42, "a" * 64)
    logger_in_tmp.log_step(_make_step(0))
    logger_in_tmp.close_trajectory("answer", "success")
    assert (tmp_path / "traces.jsonl").exists()


def test_traces_jsonl_line_count(logger_in_tmp: TraceLogger, tmp_path: Path) -> None:
    logger_in_tmp.open_trajectory("r1", "t1", "react__no__gpt-4o", 0, 42, "a" * 64)
    for i in range(3):
        logger_in_tmp.log_step(_make_step(i))
    logger_in_tmp.close_trajectory("answer", "success")

    lines = (tmp_path / "traces.jsonl").read_text().strip().splitlines()
    assert len(lines) == 3
    for line in lines:
        obj = json.loads(line)
        assert "step_id" in obj
        assert obj["task_id"] == "t1"


def test_trajectories_jsonl_written(logger_in_tmp: TraceLogger, tmp_path: Path) -> None:
    logger_in_tmp.open_trajectory("r1", "t1", "react__no__gpt-4o", 0, 42, "a" * 64)
    logger_in_tmp.log_step(_make_step(0))
    logger_in_tmp.close_trajectory("Paris", "success")

    traj_lines = (tmp_path / "trajectories.jsonl").read_text().strip().splitlines()
    assert len(traj_lines) == 1
    obj = json.loads(traj_lines[0])
    assert obj["final_answer"] == "Paris"
    assert obj["termination"] == "success"


def test_round_trip_load_trajectories(logger_in_tmp: TraceLogger, tmp_path: Path) -> None:
    logger_in_tmp.open_trajectory("r1", "t1", "a", 0, 0, "x" * 64)
    logger_in_tmp.log_step(_make_step(0))
    traj = logger_in_tmp.close_trajectory("42", "success")

    loaded = logger_in_tmp.load_trajectories(str(tmp_path))
    assert len(loaded) == 1
    assert loaded[0].final_answer == "42"
    assert loaded[0].total_tokens == traj.total_tokens


def test_no_write_when_save_traces_false(tmp_path: Path) -> None:
    log = TraceLogger(tmp_path, save_traces=False)
    log.open_trajectory("r1", "t1", "a", 0, 0, "x" * 64)
    log.log_step(_make_step(0))
    log.close_trajectory("x", "success")
    assert not (tmp_path / "traces.jsonl").exists()
    assert not (tmp_path / "trajectories.jsonl").exists()


def test_multiple_trajectories_append(logger_in_tmp: TraceLogger, tmp_path: Path) -> None:
    for trial in range(3):
        logger_in_tmp.open_trajectory("r1", "t1", "a", trial, trial, "x" * 64)
        logger_in_tmp.log_step(_make_step(0))
        logger_in_tmp.close_trajectory("ans", "success")

    loaded = logger_in_tmp.load_trajectories(str(tmp_path))
    assert len(loaded) == 3
    assert [t.trial_num for t in loaded] == [0, 1, 2]
