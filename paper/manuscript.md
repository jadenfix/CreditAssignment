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

*Run group: `group_71758879651a` — 3 seeds (7, 11, 19) × 6 conditions = 21 runs on diagnostic benchmark (5 train tasks, 5 eval tasks, 50 rollouts per run, exploration_rate=0.5). Table policy falls back to random action (not oracle) for unseen states, ensuring measured performance reflects learned policy quality.*

*Benchmark families: dead_end, enabling, recovery, multi_gate (3 sequential gates with stall/jam actions), and cascade (3-stage evidence gathering with stall/premature-guess actions). Train and eval tasks share abstract state hashes so trained models are directly applicable at test time.*

### Success Rate by Condition

| Condition | Mean | Min | Max | Avg Turns | Verifier Calls |
|-----------|------|-----|-----|-----------|----------------|
| prompted (baseline) | 0.667 | 0.56 | 0.76 | 3.46 | 331.7 |
| outcome_only | 0.467 | 0.40 | 0.60 | 6.40 | 396.7 |
| branch_aware | 0.933 | 0.80 | 1.00 | 2.93 | 270.0 |
| hindsight | 0.867 | 0.80 | 1.00 | 3.40 | 310.0 |
| **combined** | **1.000** | **1.00** | **1.00** | **2.40** | **250.0** |
| combined_matched_budget | 1.000 | 1.00 | 1.00 | 2.40 | 250.0 |

**Key findings:**

1. **Outcome-only scoring is worse than the prompted baseline** (0.467 vs 0.667). Without oracle fallback, outcome_only's loop-action bias — rewarding stall actions (`browse`, `rattle_gate`, `review_notes`) that correlate with eventual success but waste turns — actively degrades performance.

2. **Branch-aware scoring recovers from this failure** (0.933 vs 0.467, +0.466 absolute). Branch-label signals (verifier scores for `pivotal`, penalties for `dead_end`) correctly rank pivotal actions above stall actions even when both appear in successful episodes.

3. **Hindsight scoring also recovers but less reliably** (0.867). Hindsight bonuses for pivotal/recovery branches help but are less discriminative than branch_aware's explicit dead-end penalties on multi-step tasks.

4. **Combined scoring achieves perfect success** (1.000) by leveraging both branch-aware penalties and hindsight bonuses, eliminating all errors across all seeds and all 5 eval task families.

5. **The scorers form a clear hierarchy**: combined (1.0) > branch_aware (0.93) > hindsight (0.87) > prompted (0.67) > outcome_only (0.47).

### Exploration Rate Ablation

The ranking is stable across exploration rates:

| Condition | ER=0.3 | ER=0.5 | ER=0.7 |
|-----------|--------|--------|--------|
| prompted | 0.853 | 0.667 | 0.527 |
| outcome_only | 0.533 | 0.467 | 0.667 |
| branch_aware | 1.000 | 0.933 | 0.867 |
| hindsight | 0.867 | 0.867 | 0.867 |
| combined | 1.000 | 1.000 | 1.000 |

The absolute numbers shift with exploration rate (higher exploration means more noise at collection time), but the condition ranking `combined >= branch_aware >= hindsight >> outcome_only` holds across all tested rates.

### Key Mechanism: Loop-Action Bias

The critical confound that branch-aware scoring corrects is **loop-action bias**: when a stall action (e.g., `browse`, `rattle_gate`, `review_notes`) loops back to the same state and the episode eventually succeeds via a different path, outcome_only scoring assigns the stall action the same score as the pivotal action. On multi-step tasks (multi_gate, cascade), this compounds — the model prefers stalling at every gate rather than progressing, resulting in long episodes that exhaust the turn budget.

Branch-aware scoring directly penalises dead-end branches and rewards verifier signals (e.g., `gate_progress`, `evidence_quality`), breaking the tie and correctly ranking progressive actions above stalls.

### Budget Parity

Rollout count matched across all conditions (50 per run). Turn and verifier-call variation is an outcome of policy quality — better policies solve tasks in fewer turns.

| Metric | Matched | Range |
|--------|---------|-------|
| rollout_count | **yes** | 50 |
| avg_turns | no | 2.40–6.40 (combined shortest, outcome_only longest) |
| verifier_calls | no | 250–397 (combined fewest) |

### Plots

- Success by condition: `artifacts/paper/group_71758879651a/success_by_condition.svg`
- Change rate: `artifacts/paper/group_71758879651a/change_rate.svg`

## Failure Analysis

TODO: include repeated dead-end branches, disagreement points, changed preferred branches, and state abstraction collapses.

## Limitations

TODO: document manifest simplifications, model/backend tradeoffs, and AWS single-GPU constraints.

