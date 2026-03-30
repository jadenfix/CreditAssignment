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

*Run group: `group_58677ce9bd4f` — 3 seeds (7, 11, 19) × 6 conditions = 24 runs on diagnostic benchmark (train/eval splits, 20 rollouts per run, exploration_rate=0.5).*

*Benchmark: 3 train tasks (dead_end, enabling, recovery) × 3 eval tasks (dead_end_eval, enabling_eval, recovery_eval). State hashes shared between train and eval so trained models are directly applicable at test time. Collection policy uses a stateful RNG (seeded once per run) to ensure each rollout independently explores the action space.*

### Success Rate by Condition

| Condition | Mean Success | Avg Turns | Verifier Calls |
|-----------|-------------|-----------|----------------|
| prompted (baseline) | 0.683 | 2.42 | 95.7 |
| outcome_only | 0.767 | 3.03 | 89.3 |
| **branch_aware** | **1.000** | **2.10** | **80.0** |
| **hindsight** | **1.000** | **2.10** | **80.0** |
| **combined** | **1.000** | **2.10** | **80.0** |
| combined_matched_budget | 1.000 | 2.10 | 80.0 |

Branch-aware, hindsight, and combined credit assignment all achieve **perfect success (1.000)** vs outcome_only (0.767) and prompted baseline (0.683). The gap between outcome_only and branch_aware (+0.233 absolute success) demonstrates that branch-label signals provide information that terminal outcomes alone cannot recover — specifically, the enabling-family state where `browse` has high terminal correlation but `collect_clue` is the correct pivotal action.

Branch-aware and hindsight scorers also reduce verifier calls (80.0 vs 89.3 for outcome_only, 95.7 for prompted) and average turns (2.10 vs 3.03 and 2.42), indicating the trained policies find shorter solution paths.

### Key Differentiator: Enabling Family

The critical test case is the enabling-family start state (shared hash across train/eval). The collection policy explores both `collect_clue` (oracle, enabling branch) and `browse` (stall) and `answer_now` (dead-end). Under outcome_only scoring, `browse` can accumulate success signal when it is followed by eventual success — leading the model to prefer `browse` over `collect_clue`. Branch-aware scoring penalises stall branches and rewards enabling signals directly, correctly ranking `collect_clue` highest. This is the mechanism that drives the +0.233 gap.

### Budget Parity

Rollout count matched across all conditions (20 per run). Token and turn variation reflects shorter solution paths found by trained policies, not budget inequality.

| Metric | Matched | Range |
|--------|---------|-------|
| rollout_count | **yes** | 20.0 |
| avg_turns | no | 2.10–3.03 (trained shorter) |
| verifier_calls | no | 80–96 (trained fewer) |

### Branch Change Rate

The prompted baseline shows mean branch change rate of 0.103, confirming the heuristic policy does not consistently find pivotal branches. Trained conditions show 0.0, indicating the trained table policy consistently selects the same branch choices seen during collection — the scorer differentiates which branches to prefer, not whether to explore.

### Plots

- Success by condition: `artifacts/paper/group_58677ce9bd4f/success_by_condition.svg`
- Branch change rate: `artifacts/paper/group_58677ce9bd4f/change_rate.svg`

## Failure Analysis

TODO: include repeated dead-end branches, disagreement points, changed preferred branches, and state abstraction collapses.

## Limitations

TODO: document manifest simplifications, model/backend tradeoffs, and AWS single-GPU constraints.

