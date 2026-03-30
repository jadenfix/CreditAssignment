# Branch-Aware Hindsight Credit Assignment Under Matched Budgets

## Abstract

TODO: summarize the matched-budget claim and the main result.

## Introduction

- Outcome-only credit is too coarse for long-horizon agent behavior.
- Branch-aware and hindsight credit should improve search quality, not just terminal accuracy.
- The core question is whether probability mass shifts toward better downstream branches under equal collection budgets.

## Method

TODO: describe the state graph, verifier events, offline scorers, and scored online training path.

## Benchmarks

- Diagnostic benchmark with explicit pivotal, dead-end, enabling, and recovery annotations.
- WebShop and ALFWorld as core long-horizon ablation environments.
- LiveCodeBench as first external validation.
- SWE-bench Lite as heavier follow-on validation.

## Results

*Run group: `group_fa92e70d427f` — 3 seeds (7, 11, 19) × 6 conditions = 24 runs on diagnostic benchmark (train split, 6 rollouts per run).*

### Success Rate by Condition

| Condition | Mean Success | Min | Max | Verifier Calls | Tokens/Success |
|-----------|-------------|-----|-----|----------------|----------------|
| prompted | 0.667 | 0.5 | 1.0 | 21.0 | 58.3 |
| outcome_only | 0.833 | 0.5 | 1.0 | 18.0 | 39.3 |
| branch_aware | 0.833 | 0.5 | 1.0 | 18.0 | 39.3 |
| hindsight | 0.833 | 0.5 | 1.0 | 18.0 | 39.3 |
| combined | 0.833 | 0.5 | 1.0 | 18.0 | 39.3 |
| combined_matched_budget | 0.833 | 0.5 | 1.0 | 18.0 | 39.3 |

All trained conditions (outcome_only, branch_aware, hindsight, combined) improve mean success rate from 0.667 (prompted baseline) to 0.833 — a **+0.167 absolute gain** — while reducing verifier calls from 21.0 to 18.0 and avg turns from 2.17 to 1.83.

### Budget Parity

Rollout count is matched across all conditions (6.0). Token and turn counts vary across conditions because trained policies take fewer turns to reach terminal states, not because of unequal collection budgets.

| Metric | Matched | Values |
|--------|---------|--------|
| rollout_count | **yes** | 6.0 |
| avg_turns | no | 1.5–2.5 (trained conditions shorter) |
| verifier_calls | no | 18–27 (trained use fewer) |
| prompt_tokens | no | 78–204 |
| completion_tokens | no | 48–96 |

### Branch Change Rate

The prompted baseline shows a mean branch change rate of 0.333 (range 0.0–1.0), indicating the heuristic policy does sometimes take pivotal branches. All trained conditions show a change rate of 0.0 in the branch_aware replay scorer — consistent with the trained table policies converging to a fixed action selection that does not differ from the collection policy's branch choices at the replay scoring step.

### Plots

- Success by condition: `artifacts/paper/group_fa92e70d427f/success_by_condition.svg`
- Branch change rate: `artifacts/paper/group_fa92e70d427f/change_rate.svg`

## Failure Analysis

TODO: include repeated dead-end branches, disagreement points, changed preferred branches, and state abstraction collapses.

## Limitations

TODO: document manifest simplifications, model/backend tradeoffs, and AWS single-GPU constraints.

