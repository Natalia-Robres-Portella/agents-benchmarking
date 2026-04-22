# Agent Benchmark Suite

A research framework for evaluating and comparing LLM agents in a rigorous, reproducible, and extensible way.

---

## Table of contents

1. [What is it for?](#1-what-is-it-for)
2. [Why does it exist?](#2-why-does-it-exist)
3. [What does it compare?](#3-what-does-it-compare)
4. [How does it work internally?](#4-how-does-it-work-internally)
5. [Where does the data come from?](#5-where-does-the-data-come-from)
6. [How is evaluation done?](#6-how-is-evaluation-done)
7. [Folder structure](#7-folder-structure)
8. [Installation](#8-installation)
9. [How to run an experiment](#9-how-to-run-an-experiment)
10. [Configuration checklist](#10-configuration-checklist)
11. [Full YAML configuration reference](#11-full-yaml-configuration-reference)
12. [What is saved in the results](#12-what-is-saved-in-the-results)
13. [How to extend the codebase](#13-how-to-extend-the-codebase)
14. [Tests](#14-tests)
15. [Reference papers](#15-reference-papers)

---

## 1. What is it for?

This project lets you **run controlled experiments on LLM agents** and measure objectively how they behave on reasoning and information-seeking tasks.

An “LLM agent” is a system that uses a language model (GPT-4o, Claude, etc.) not only to generate text, but to **reason, choose tools, execute them, observe outcomes, and repeat** until it produces an answer. This framework helps answer questions such as:

- Is ReAct better than a tool-free agent on HotPotQA?
- How many steps does the agent need on average to answer correctly?
- What fraction of tool calls have valid arguments?
- How many tokens does each strategy consume per task?
- Can the agent recover when a tool fails?

---

## 2. Why does it exist?

Most LLM benchmarks evaluate the model in isolation (how well does it answer a question?). This project evaluates the **full agent system**: reasoning strategy + tools + memory + model.

### Problems it addresses

**Reproducibility.** Without fixing the random seed, the same experiment can yield different results each run. Every run has a `config_hash` (SHA-256 of the configuration YAML) embedded in each trajectory, so two runs with the same hash are comparable.

**Fair comparison.** To compare ReAct vs. Reflexion vs. Direct, all runs must use the same task set (same seed), same model, and same metrics. The framework enforces that by design.

**Full traceability.** Every agent step—what it thought, which tool it called, with which arguments, what it observed, token usage, latency—is logged as JSONL. Failures can be replayed and inspected.

**Academic grounding.** Design choices map to papers:

- ReAct (Yao et al., ICLR 2023)
- Reflexion (Shinn et al., NeurIPS 2023)
- Pass@k (Chen et al., 2021 / Shinn et al., 2023)
- LLM-as-judge bias mitigation (Zheng et al., 2023)
- HELM multi-metric evaluation (Liang et al., 2022)

---

## 3. What does it compare?

The framework compares agents along **four independent axes** you can mix freely:

### Axis 1 — Planning strategy

| Strategy | Status | Description |
|---|---|---|
| `direct` | Implemented | No tools. The model answers directly. Baseline. |
| `react` | Implemented | Thought → Action → Observation loop (Yao et al., 2023). |
| `reflexion` | Planned | ReAct + verbal reflection across trials (Shinn et al., 2023). |
| `plan_execute` | Planned | Full plan before execution (Wang et al., 2023). |
| `tot` | Planned | Tree of Thoughts: multiple reasoning branches (Yao et al., 2023). |

### Axis 2 — Memory type

| Type | Description |
|---|---|
| `no_memory` | No memory. Each trial starts from scratch. |
| `window_buffer` | Keeps the last N steps in context. |
| `episodic` | Buffer of verbal reflections (for Reflexion). Persists across trials on the same task; cleared between tasks. |
| `vector_store` | (Advanced) Embedding retrieval (requires ChromaDB). |

### Axis 3 — LLM model

Any OpenAI model (GPT-4o, GPT-4o-mini, …) or Anthropic (Claude Opus, Sonnet, …). Configured in YAML.

### Axis 4 — Available tools

| Tool | Description |
|---|---|
| `search` | Search. Official benchmark runs use **MockSearchTool** (local JSON fixture, reproducible). `LiveSearchTool` (DuckDuckGo) exists for exploratory use only. |
| `calculator` | Safe math expression evaluator (operator allowlist). |
| `finish` | Episode termination. The agent calls it when it has the final answer. |

---

## 4. How does it work internally?

End-to-end experiment flow:

```
run_experiment.py
    └── ExperimentOrchestrator.run()
            │
            ├── 1. seed_everything(seed)          ← reproducibility
            ├── 2. Creates results/{run_id}/
            ├── 3. Snapshot config.yaml
            │
            ├── 4. Builds components
            │       ├── Agent (strategy + memory + LLM + tools)
            │       ├── TaskLoader (loads dataset tasks)
            │       ├── TraceLogger (writes JSONL per step)
            │       ├── ExecutionEngine (step loop)
            │       ├── EvaluationModule (metrics)
            │       └── ReportGenerator (writes outputs)
            │
            ├── 5. For each task:
            │       └── ExecutionEngine.run(task)
            │               │
            │               └── For each trial (n_trials times):
            │                       ├── post_episode_hook() ← Reflexion writes here
            │                       ├── agent.reset(seed)
            │                       └── Loop until max_steps:
            │                               ├── agent.act(state)   → Action
            │                               ├── tools.execute()    → ToolResult
            │                               ├── agent.observe(obs)
            │                               └── logger.log_step()
            │
            ├── 6. EvaluationModule.compute_all()
            │       ├── Stage 1: validator.validate() → score per trajectory
            │       └── Stage 2: metric.compute()     → MetricResult per metric
            │
            └── 7. ReportGenerator.emit()
                    ├── metrics.json
                    └── report.md
```

### Agent and strategy

`PlanningStrategy` is **stateless**—a pure transform from (state, memory, tools) → prompt → action. All reasoning-format logic lives here.

`BaseAgent` is **stateful**—it holds memory and wires strategy to LLM and tools. Its `agent_id` is a readable fingerprint, e.g. `"react__no_memory__gpt-4o"`.

### The ReAct loop

Each step:

```
Thought: I need to search for who wrote Hamlet.
Action: search
Action Input: {"query": "who wrote Hamlet"}
                    ↓ (engine runs the tool)
Observation: Hamlet was written by William Shakespeare...
                    ↓ (next iteration)
Thought: I have the answer.
Action: finish
Action Input: {"answer": "William Shakespeare"}
```

The stop sequence `"\nObservation:"` prevents the model from hallucinating its own observation—the engine supplies the real tool output.

---

## 5. Where does the data come from?

### HotPotQA (primary implemented dataset)

HotPotQA is a multi-hop reasoning QA benchmark: answering often requires combining evidence from multiple sources. Example:

> *“In what year was the founder of the company that made the first iPhone born?”*

The loader downloads from Hugging Face Hub when online:

```python
load_dataset("hotpot_qa", "fullwiki", split="validation")
```

**Without internet** (CI, offline), it automatically falls back to the local fixture `fixtures/hotpotqa_sample.json` (20 example questions shipped in the repo).

Each loaded task has this internal shape:

```python
TaskInstance(
    task_id="hotpotqa_5abc123",
    input="What is the capital of France?",   # question shown to the agent
    gold="Paris",                             # reference answer for scoring
    metadata={"type": "bridge", "level": "easy"}
)
```

### Adding other datasets

Implement a `TaskLoader` under `src/tasks/loaders/` and register it in `src/tasks/loaders/__init__.py`. See [13. How to extend the codebase](#13-how-to-extend-the-codebase).

---

## 6. How is evaluation done?

Evaluation has two stages.

### Stage 1 — Per-trajectory scoring (Validator)

Compares the agent’s final answer to the reference (`gold`):

| Validator | When to use | How it works |
|---|---|---|
| `exact_match` | Very strict string answers | Case-insensitive string equality. 1.0 or 0.0. |
| `fuzzy_match` | Open-ended QA (HotPotQA, etc.) | Token-level F1 with stop words (SQuAD-style). Allows “William Shakespeare” vs “Shakespeare” with partial credit; **`success` is True only when the F1 score is exactly 1.0.** |
| `llm_judge` | Long-form answers, many valid phrasings | External LLM scores 0–10 with bias mitigations (below). |

**Outcome:** each `Trajectory` gets a `score` ∈ [0,1] and a `success` flag (True iff score ≥ 1.0 with the current implementation).

### Stage 2 — Aggregate metrics (Metrics)

Computed over all trajectories in the run:

| Metric | Main value | Breakdown |
|---|---|---|
| `success_rate` | Fraction of tasks where at least one trial succeeded | By difficulty (easy/medium/hard) |
| `pass_at_k` | P(at least one of k random trials succeeds)—exact combinatorial formula | pass@1, pass@3, pass@5 |
| `tokens_per_task` | Mean total tokens per trajectory | — |
| `step_count` | Mean ReAct steps per trajectory | success vs. failure |
| `tool_accuracy` | Fraction of tool calls with valid arguments | error_rate, calls_per_episode, per tool |
| `failure_recovery` | Fraction of episodes with errors that still succeeded | episodes_with_errors, recovered |
| `latency` | Total latency per episode (ms) | p50, p95, mean (when registered under the name your config uses) |

### LLM judge (`llm_judge`) and bias mitigation

With `validator: "llm_judge"`, each prediction is scored with **4 prompt templates × n_samples independent calls**:

- **Two perspectives:** factual accuracy + semantic equivalence  
- **Two orderings:** prediction-first / gold-first → reduces positional bias  
- **temperature > 0:** measures variance across calls  
- **Aggregation:** mean score; if std_dev > 0.15, a low-confidence WARNING is logged  

See [docs/llm_judge_notes.md](docs/llm_judge_notes.md) for biases and mitigations in detail.

---

## 7. Folder structure

```
agents_benchmarking/
│
├── run_experiment.py          ← ENTRY POINT. CLI to launch experiments.
├── requirements.txt           ← MVP dependencies (install first).
├── requirements-advanced.txt  ← Optional deps (Anthropic, ChromaDB, etc.).
├── pyproject.toml             ← pytest and ruff config.
├── .env.example               ← Environment variable template (API keys).
│
├── configs/                   ← EXPERIMENT CONFIGURATION
│   ├── base_config.yaml       ← Defaults. Every experiment inherits from this.
│   └── experiments/
│       └── react_hotpotqa.yaml ← Concrete experiment. Overrides defaults.
│
├── fixtures/                  ← OFFLINE DATA for CI and development without internet
│   ├── hotpotqa_sample.json   ← 20 HotPotQA-style items. Automatic fallback.
│   └── search_responses.json  ← Simulated search hits (MockSearchTool).
│
├── results/                   ← EXPERIMENT OUTPUT (created automatically)
│   └── {run_id}/              ← One folder per run: "{id}__{timestamp}__{hash}"
│       ├── config.yaml        ← Exact config snapshot.
│       ├── metrics.json       ← Machine-readable metrics.
│       ├── report.md          ← Markdown metric table (GitHub-friendly).
│       ├── trajectories.jsonl ← One JSON line per completed trajectory.
│       └── traces.jsonl       ← One JSON line per agent step (if save_traces=true).
│
├── docs/
│   └── llm_judge_notes.md     ← LLM judge limitations and mitigations.
│
├── src/                       ← ALL SOURCE CODE
│   │
│   ├── schema.py              ← Core Pydantic models. IMPORTED EVERYWHERE.
│   │                            Defines: Action, Observation, Step, Trajectory,
│   │                            TaskInstance, AgentState, MetricResult, ToolResult.
│   │
│   ├── config.py              ← Config models + load_config() + config_hash.
│   │                            Defines: ExperimentConfig, AgentConfig, LLMConfig,
│   │                            MemoryConfig, ToolConfig, EvaluationConfig, etc.
│   │
│   ├── utils.py               ← Shared helpers: seed_everything(), make_run_id(),
│   │                            configure_logging().
│   │
│   ├── orchestrator.py        ← ExperimentOrchestrator. Full lifecycle coordinator.
│   │
│   ├── execution_engine.py    ← ExecutionEngine. act→observe→log step loop.
│   │
│   ├── trace_logger.py        ← TraceLogger. Writes traces.jsonl and trajectories.jsonl.
│   │                            Flush after each step → crash-safe.
│   │
│   ├── agents/
│   │   ├── base.py            ← Agent (ABC) + BaseAgent (concrete).
│   │   │                        BaseAgent wires: strategy + memory + LLM + tools.
│   │   └── factory.py         ← build_agent(config) → Agent.
│   │
│   ├── strategies/
│   │   ├── base.py            ← PlanningStrategy (ABC). Contract: build_prompt + parse_response.
│   │   ├── direct.py          ← Direct answer, no tools. Baseline.
│   │   ├── react.py          ← Full ReAct with Thought/Action/Action Input parser.
│   │   └── factory.py         ← build_strategy(name) → PlanningStrategy.
│   │
│   ├── memory/
│   │   ├── base.py            ← MemoryModule (ABC). Contract: read/write/reset.
│   │   ├── no_memory.py       ← No memory. read() → []. No-op on write/reset.
│   │   ├── window_buffer.py   ← Last N steps in a deque.
│   │   ├── episodic.py        ← Verbal reflection buffer for Reflexion.
│   │   │                        reset() is no-op. hard_reset() clears between tasks.
│   │   └── factory.py         ← build_memory(config) → MemoryModule.
│   │
│   ├── llm/
│   │   ├── base.py            ← LLMBackend (ABC) + LLMResponse.
│   │   ├── openai_backend.py  ← OpenAIBackend with exponential retries (tenacity).
│   │   └── factory.py         ← build_llm_backend(config) → LLMBackend.
│   │
│   ├── tools/
│   │   ├── base.py            ← BaseTool (ABC) + ToolRegistry.
│   │   │                        ToolRegistry.validate_and_execute() is the single
│   │   │                        entry point—errors become ToolResult, not exceptions.
│   │   ├── finish.py          ← FinishTool. Episode termination signal.
│   │   ├── calculator.py      ← Safe calculator with operator allowlist.
│   │   ├── search.py          ← MockSearchTool (offline) + LiveSearchTool (DuckDuckGo).
│   │   └── factory.py         ← build_tool_registry(tool_configs) → ToolRegistry.
│   │
│   ├── tasks/
│   │   ├── base.py            ← TaskLoader (ABC) + TaskValidator (ABC) + TaskRegistry.
│   │   └── loaders/
│   │       └── hotpotqa.py    ← HotPotQALoader. Hugging Face Hub + fixture fallback.
│   │
│   ├── evaluation/
│   │   ├── module.py          ← EvaluationModule. Orchestrates Stage 1 and Stage 2.
│   │   ├── validators/
│   │   │   ├── base.py        ← ValidatorRegistry.
│   │   │   ├── exact_match.py ← Case-insensitive exact equality.
│   │   │   ├── fuzzy_match.py ← Token F1 with stop words (SQuAD-style).
│   │   │   └── llm_judge.py   ← LLMJudgeValidator with bias mitigations.
│   │   └── metrics/
│   │       ├── base.py        ← Metric (ABC) + MetricRegistry.
│   │       ├── success_rate.py
│   │       ├── pass_at_k.py
│   │       ├── tokens.py
│   │       ├── steps.py
│   │       ├── tool_accuracy.py
│   │       ├── failure_recovery.py
│   │       └── latency.py
│   │
│   └── reporting/
│       ├── aggregator.py      ← ResultAggregator. Groups trajectories by task_id.
│       └── report_generator.py ← Writes metrics.json and report.md.
│
└── tests/
    ├── conftest.py
    ├── unit/                  ← Unit tests (no real LLM, no network)
    │   ├── test_schema.py
    │   ├── test_config.py
    │   ├── test_agent.py
    │   ├── test_react_strategy.py
    │   ├── test_tools.py
    │   ├── test_tools_base.py
    │   ├── test_execution_engine.py
    │   ├── test_trace_logger.py
    │   ├── test_evaluation.py
    │   ├── test_metrics_phase5.py
    │   └── test_llm_judge.py
    └── integration/
        └── test_react_smoke.py ← Full pipeline with mocked LLM.
```

---

## 8. Installation

### Prerequisites

- Python 3.11+
- An OpenAI API key (minimum) or Anthropic (optional)

### Steps

```bash
# 1. Clone the repository
git clone <repo-url>
cd agents_benchmarking

# 2. Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows

# 3. Install MVP dependencies
pip install -r requirements.txt

# 4. (Optional) Advanced deps: Anthropic, ChromaDB, ALFWorld...
pip install -r requirements-advanced.txt

# 5. Configure API keys
cp .env.example .env
# Edit .env and add real keys
```

### `.env` contents

```bash
OPENAI_API_KEY=sk-...       # Required for OpenAI-backed experiments
ANTHROPIC_API_KEY=sk-ant-... # Only if you use provider: "anthropic"
```

**Important:** `.env` is in `.gitignore`. Never commit it.

---

## 9. How to run an experiment

### Real experiment (requires API key)

```bash
python run_experiment.py --config configs/experiments/react_hotpotqa.yaml
```

### Dry run (validates config without calling any API)

```bash
python run_experiment.py --config configs/experiments/react_hotpotqa.yaml --dry-run
```

Prints a summary and exits:

```
╭─────────────── Agent Benchmark Suite ───────────────╮
│ react_hotpotqa_gpt4o_v1                              │
│ strategy=react | model=gpt-4o | memory=no_memory     │
│ tasks=hotpotqa n=100 | trials=5 | seed=42            │
│ config_hash=3f8a1c2b…                                │
╰──────────────────────────────────────────────────────╯
Dry run — config valid, exiting.
```

### Tests (no API key, fully offline)

```bash
# All tests
python3 -m pytest

# Unit tests only
python3 -m pytest tests/unit/

# Integration smoke test only
python3 -m pytest tests/integration/

# Verbose
python3 -m pytest -v

# Single module
python3 -m pytest tests/unit/test_llm_judge.py -v
```

### Where results appear

After a real run:

```
results/
└── react_hotpotqa_gpt4o_v1__20260422T143022__3f8a1c2b/
    ├── config.yaml        ← exact YAML snapshot
    ├── metrics.json       ← all metrics as JSON
    ├── report.md          ← Markdown table for GitHub
    ├── trajectories.jsonl ← one line per trajectory
    └── traces.jsonl       ← one line per agent step
```

---

## 10. Configuration checklist

### Required for real experiments

| What | Where | Example |
|---|---|---|
| OpenAI API key | `.env` | `OPENAI_API_KEY=sk-proj-...` |
| Experiment ID | YAML config | `id: "my_experiment_v1"` |

### Creating a new experiment

1. Copy `configs/experiments/react_hotpotqa.yaml`
2. Give it a unique `id` (used under `results/`)
3. Change what you want to compare (strategy, model, n_samples, …)
4. Run with `--config your_new_config.yaml`

### Comparing two strategies

Create two YAMLs that differ only in `agent.strategy`. Keep everything else identical (seed, tasks, metrics). Example:

```yaml
# react_baseline.yaml
experiment:
  id: "baseline_react"
  seed: 42
agent:
  strategy: "react"
  ...

# direct_baseline.yaml
experiment:
  id: "baseline_direct"
  seed: 42
agent:
  strategy: "direct"
  ...
```

Run both and compare `metrics.json`.

### Using the LLM judge

Add to YAML:

```yaml
evaluation:
  validator: "llm_judge"
  llm_judge_model: "claude-opus-4-7"   # model DIFFERENT from the one being evaluated
```

Ensure `ANTHROPIC_API_KEY` is in `.env` if the judge is Claude.

---

## 11. Full YAML configuration reference

Experiment YAMLs are merged with `configs/base_config.yaml`. Experiment fields override defaults.

```yaml
experiment:
  id: "unique_name"           # REQUIRED. Run identifier. Avoid spaces.
  seed: 42                    # Global seed. Affects task shuffle and LLM sampling.
  n_trials: 5                 # Repeats per task. Needed for pass@k.
  max_steps: 25               # Max ReAct steps per trial. Prevents infinite loops.
  output_dir: "results/"      # Where to write outputs.
  tags: ["react", "v1"]       # Free-form tags for organization.

agent:
  strategy: "react"           # "direct" | "react" | "reflexion" | "plan_execute" | "tot"
  llm:
    provider: "openai"        # "openai" | "anthropic" | "local"
    model: "gpt-4o"           # Model name (gpt-4o, gpt-4o-mini, claude-opus-4-7...)
    temperature: 0.0          # 0.0 for reproducibility; > 0 for stochastic sampling.
    max_tokens: 1024          # Max output tokens per call.
  memory:
    type: "no_memory"         # "no_memory" | "window_buffer" | "episodic" | "vector_store"
    window_size: 10           # window_buffer only: how many steps to retain.
    top_k: 5                  # vector_store only: nearest neighbors.
  tools:
    # Benchmark runs register `search` as MockSearchTool (fixtures/search_responses.json).
    # Extra keys in config are ignored; use fixture_path to override the JSON file.
    - name: "search"
      config: {}
    - name: "calculator"
    - name: "finish"          # Always include finish for ReAct.

tasks:
  dataset: "hotpotqa"         # Must exist in TASK_REGISTRY.
  split: "validation"         # "train" | "validation" | "test"
  n_samples: 100              # Number of tasks. More = more cost, more stable stats.
  filter:                     # (Optional) Filter by task metadata.
    difficulty: ["hard"]      # Hard tasks only.

evaluation:
  validator: "fuzzy_match"    # "exact_match" | "fuzzy_match" | "llm_judge"
  llm_judge_model: null       # Only if validator="llm_judge". Judge model id.
  metrics:                    # Metrics to compute.
    - "success_rate"
    - "pass_at_k"
    - "tokens_per_task"
    - "step_count"
    - "tool_accuracy"
    - "failure_recovery"
    - "latency"
  pass_k_values: [1, 3, 5]    # k values for pass@k.

logging:
  level: "INFO"               # "DEBUG" | "INFO" | "WARNING"
  save_traces: true           # If false, skips traces.jsonl (faster, less disk).
  trace_format: "jsonl"       # Only "jsonl" supported for now.
```

---

## 12. What is saved in the results

### `trajectories.jsonl`

One JSON object per line, one line per completed trajectory:

```json
{
  "run_id": "react_hotpotqa_v1__20260422T143022__3f8a1c2b",
  "task_id": "hotpotqa_5abc123",
  "agent_id": "react__no_memory__gpt-4o",
  "trial_num": 0,
  "seed": 42,
  "config_hash": "3f8a1c2b...",   // SHA-256 of full config
  "steps": [...],                  // Step list
  "termination": "success",        // "success" | "max_steps" | "parse_error" | "llm_error"
  "final_answer": "William Shakespeare",
  "total_tokens": 1247,
  "total_latency_ms": 3421.5,
  "success": true,
  "score": 1.0
}
```

### `traces.jsonl`

One JSON line per **agent step** (finer granularity). Only if `save_traces: true`. Useful for debugging and behavior analysis.

### `metrics.json`

```json
{
  "run_id": "react_hotpotqa_v1__20260422T143022__3f8a1c2b",
  "agent_id": "react__no_memory__gpt-4o",
  "config": { ... },
  "metrics": {
    "success_rate": {
      "value": 0.62,
      "breakdown": {"easy": 0.85, "medium": 0.55, "hard": 0.30}
    },
    "pass_at_k": {
      "value": 0.71,
      "breakdown": {"pass@1": 0.62, "pass@3": 0.71, "pass@5": 0.78}
    },
    "step_count": {
      "value": 4.3,
      "breakdown": {"success": 3.1, "failure": 6.8}
    }
  }
}
```

### `report.md`

GitHub-ready Markdown table. Example:

```markdown
# Experiment: react_hotpotqa_gpt4o_v1

**Strategy:** react | **Model:** gpt-4o | **Memory:** no_memory

## Metrics

| Metric | Value | Breakdown |
|--------|-------|-----------|
| success_rate | 0.6200 | easy=0.850, medium=0.550, hard=0.300 |
| pass_at_k | 0.7100 | pass@1=0.620, pass@3=0.710, pass@5=0.780 |
```

---

## 13. How to extend the codebase

### Add a dataset

1. Create `src/tasks/loaders/my_dataset.py`:

```python
from src.tasks.base import TaskLoader
from src.schema import TaskInstance

class MyDatasetLoader(TaskLoader):
    @property
    def name(self) -> str:
        return "my_dataset"

    def load(self, split, n_samples, seed, filter_kwargs=None):
        # Load data and return List[TaskInstance]
        ...
```

2. Register in `src/tasks/loaders/__init__.py`:

```python
from src.tasks.loaders.my_dataset import MyDatasetLoader
TASK_REGISTRY.register("my_dataset", MyDatasetLoader)
```

3. Use in YAML: `dataset: "my_dataset"`.

### Add a strategy

1. Create `src/strategies/my_strategy.py` implementing `PlanningStrategy`:
   - `build_prompt(state, memory_context, tool_descriptions) -> str`
   - `parse_response(raw, state) -> Action` — **must not raise** (handle errors in-band)
   - `name` property

2. Register in `src/strategies/factory.py`.

3. Add the name to the `Literal` in `src/config.py` → `AgentConfig.strategy`.

### Add a tool

1. Create `src/tools/my_tool.py` subclassing `BaseTool`.

2. Register in `src/tools/factory.py`.

3. Use in YAML: `tools: [{name: "my_tool"}]`.

### Add a metric

1. Create `src/evaluation/metrics/my_metric.py` implementing `Metric`:
   - `name` property
   - `compute(trajectories, tasks) -> MetricResult`

2. Register in `src/evaluation/metrics/__init__.py`:

```python
from src.evaluation.metrics.my_metric import MyMetric
METRIC_REGISTRY.register(MyMetric)
```

3. Use in YAML: `metrics: ["my_metric"]`.

---

## 14. Tests

The suite has **152** collected tests—all runnable offline; none call real APIs.

```bash
python3 -m pytest                    # all
python3 -m pytest tests/unit/        # unit only
python3 -m pytest tests/integration/ # integration only
python3 -m pytest -v --tb=short      # verbose, short traceback
```

Coverage by test file:

| Test file | What it covers |
|---|---|
| `test_schema.py` | Pydantic model serialization |
| `test_config.py` | load_config, deep merge, deterministic config_hash |
| `test_agent.py` | BaseAgent: act(), observe(), memory wiring, agent_id |
| `test_react_strategy.py` | Thought/Action/Action Input parser, edge cases |
| `test_tools.py` | search (offline fixture), calculator (allowlist), finish |
| `test_tools_base.py` | ToolRegistry: validate_and_execute, unknown tool |
| `test_execution_engine.py` | Step loop, termination on max_steps/abort/success |
| `test_trace_logger.py` | JSONL output, crash-safety, round-trip load |
| `test_evaluation.py` | Validators (exact/fuzzy), PassAtK, SuccessRate, EvaluationModule |
| `test_metrics_phase5.py` | StepCount, ToolAccuracy, FailureRecovery, Latency |
| `test_llm_judge.py` | Score parsing, bias mitigation, failure robustness |
| `test_react_smoke.py` | End-to-end pipeline with mocked LLM |

---

## 15. Reference papers

Design choices map to published work:

| Paper | Citation | Relevance in code |
|---|---|---|
| ReAct | Yao et al., ICLR 2023. arXiv:2210.03629 | `src/strategies/react.py` |
| Reflexion | Shinn et al., NeurIPS 2023. arXiv:2303.11366 | `src/memory/episodic.py`, `EpisodicMemory.reset()` no-op |
| Plan-and-Solve | Wang et al., ACL 2023. arXiv:2305.04091 | `src/strategies/` (planned) |
| Tree of Thoughts | Yao et al., NeurIPS 2023. arXiv:2305.10601 | `src/strategies/` (planned) |
| Toolformer | Schick et al., NeurIPS 2023. arXiv:2302.04761 | `src/tools/base.py`, ToolRegistry |
| HotPotQA | Yang et al., EMNLP 2018 | `src/tasks/loaders/hotpotqa.py` |
| SQuAD token F1 | Rajpurkar et al., 2016 | `src/evaluation/validators/fuzzy_match.py` |
| Pass@k formula | Chen et al., 2021 / Shinn et al., 2023 | `src/evaluation/metrics/pass_at_k.py` |
| HELM | Liang et al., 2022. arXiv:2211.09110 | Multi-metric design, `EvaluationModule` |
| AgentBench | Liu et al., ICLR 2024. arXiv:2308.03688 | Overall benchmark architecture |
| WebArena | Zhou et al., 2023. arXiv:2307.13854 | Reproducibility-first: config_hash, seed |
| MT-Bench / LLM Judge | Zheng et al., NeurIPS 2023. arXiv:2306.05685 | `src/evaluation/validators/llm_judge.py` |
| Positional bias | Wang et al., 2023. arXiv:2309.03882 | Prompt ordering in LLMJudgeValidator |
| Verbosity bias | Dubois et al., 2024. arXiv:2404.04475 | LLM judge templates |
