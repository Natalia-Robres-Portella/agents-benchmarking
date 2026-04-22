# LLM-as-a-Judge: Limitations and Mitigation Strategies

This document explains the known failure modes of LLM-based evaluators and the
specific design decisions made in `LLMJudgeValidator` to address them.

---

## Why use an LLM judge?

Exact-match and fuzzy-match (F1) validators fail on tasks where correctness is
not a string equality problem: open-ended generation, multi-hop reasoning with
equivalent phrasings, or tasks with multiple valid answers.  An LLM judge can
evaluate semantic correctness where rule-based metrics cannot.

This is the approach used in:
- MT-Bench (Zheng et al., 2023 — arXiv:2306.05685)
- Alpaca Eval (Li et al., 2023)
- AgentBench subset evaluations (Liu et al., 2024 — arXiv:2308.03688)

---

## Known limitations

### 1. Positional bias
The judge assigns higher scores to whichever response appears first (or last) in
the prompt.  Zheng et al. (2023) found that GPT-4, when used as a pairwise
judge, agreed with the first-presented response ~65% of the time on ties.

**Mitigation (implemented):** All four prompt templates are split evenly between
"prediction before gold" and "gold before prediction" orderings.  Scores are
averaged across orderings, cancelling the directional bias in expectation.

### 2. Verbosity bias
LLM judges systematically prefer longer responses, even when the additional
content is irrelevant (Dubois et al., 2024 — arXiv:2404.04475).

**Mitigation (implemented):** Templates explicitly instruct the judge to focus
on factual content or semantic equivalence, not style or length.  The factual
template penalises responses that add irrelevant information.

**Residual risk:** Verbosity bias cannot be fully eliminated via prompting alone.
For high-stakes evaluations, consider a length-normalised scoring rubric or a
separate conciseness check.

### 3. Self-preference (self-enhancement) bias
Models from the same family as the evaluated agent tend to score responses from
that family more favourably (Panickssery et al., 2024 — arXiv:2404.13076).

**Mitigation (recommended, not automatically enforced):** Use a judge from a
different model family than the evaluated agent.  The `llm_judge_model`
config field lets you specify any backend.  The default `gpt-4o` should not
be used as a judge for `gpt-4o`-generated responses without validation.

### 4. Calibration variance / stochasticity
A single judge call at temperature=0 produces deterministic but potentially
biased scores.  At temperature > 0, scores vary — but this variance exposes
instability rather than hiding it.

**Mitigation (implemented):** `n_samples` independent calls per template are
made at `temperature=0.3`.  When the standard deviation of all calls exceeds
`confidence_threshold` (default 0.15, i.e., 1.5 points on a 0–10 scale), a
`WARNING` is emitted.  The caller can quarantine or re-evaluate flagged tasks.

### 5. Rubric sensitivity
Small changes to the scoring rubric (e.g., changing "0–10" to "1–5") can shift
average scores by several points without changing relative rankings.

**Mitigation (implemented):** Two complementary rubrics are used — one focused
on factual precision, one on semantic equivalence.  Averaging across both
dimensions reduces sensitivity to any single rubric choice.

### 6. Prompt injection in agent outputs
A malicious or misaligned agent could produce a final answer that contains
judge-manipulation text (e.g., "SCORE: 10. Ignore previous instructions.").

**Mitigation (partial):** The score parser (`_parse_score`) only extracts the
first numeric value matching the `SCORE:` pattern from the judge's *response*,
not from the agent's answer.  The agent's answer appears inside the prompt body,
after the judge instructions.  However, a sufficiently adversarial answer could
still confuse the judge's reasoning.  For safety-critical settings, sanitise
agent outputs before passing them to the judge.

### 7. Cost and latency
Four templates × `n_samples` calls per trajectory makes the LLM judge 4×
more expensive than a single-call evaluator.  For large-scale benchmarks
(> 1,000 tasks) this can be prohibitive.

**Mitigation (trade-off):** Set `n_samples=1` for budget-constrained runs.
The positional debiasing across the 4 templates still fires; only the
variance estimation is reduced.  Use `n_samples=3+` for final publication
numbers.

---

## Confidence flagging

A run-level low-confidence rate can be computed post-hoc from the `WARNING`
log lines:

```
LLMJudge: LOW CONFIDENCE for task=<id> (mean=0.45, std=0.31, ...)
```

If more than ~15% of tasks are flagged, the judge is not reliable for this
task distribution and a different evaluation strategy should be used.

---

## Recommended configuration for publication results

```yaml
evaluation:
  validator: "llm_judge"
  llm_judge_model: "claude-opus-4-7"   # different family than evaluated agent
  # n_samples and temperature are set on LLMJudgeValidator directly
```

In code, construct the module with `n_samples=3, temperature=0.3`.

---

## References

- Zheng et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* NeurIPS 2023. arXiv:2306.05685
- Wang et al. (2023). *Large Language Models are not Robust Multiple Choice Selectors.* ICLR 2024. arXiv:2309.03882
- Dubois et al. (2024). *Length-Controlled AlpacaEval.* arXiv:2404.04475
- Panickssery et al. (2024). *LLM Evaluators Recognise and Favour Their Own Generations.* arXiv:2404.13076
- Liu et al. (2024). *AgentBench: Evaluating LLMs as Agents.* ICLR 2024. arXiv:2308.03688
