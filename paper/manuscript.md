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

*Run group: `group_f410ed95cc83` — 7 seeds × 6 conditions = 49 runs on diagnostic benchmark (5 train tasks, 5 eval tasks, 50 rollouts per run, exploration_rate=0.5). Table policy falls back to random action (not oracle) for unseen states, ensuring measured performance reflects learned policy quality.*

*Benchmark families: dead_end, enabling, recovery, multi_gate (3 sequential gates with stall/jam actions), and cascade (3-stage evidence gathering with stall/premature-guess actions). Train and eval tasks share abstract state hashes so trained models are directly applicable at test time.*

### Success Rate by Condition

| Condition | Mean | Min | Max | n | Avg Turns | Verifier Calls |
|-----------|------|-----|-----|---|-----------|----------------|
| prompted (baseline) | 0.726 | 0.56 | 0.78 | 7 | 3.38 | 327.1 |
| outcome_only | 0.514 | 0.20 | 1.00 | 7 | 6.06 | 394.3 |
| branch_aware | 0.971 | 0.80 | 1.00 | 7 | 2.63 | 258.6 |
| hindsight | 0.857 | 0.80 | 1.00 | 7 | 3.46 | 315.7 |
| **combined** | **1.000** | **1.00** | **1.00** | **7** | **2.40** | **250.0** |
| combined_matched_budget | 1.000 | 1.00 | 1.00 | 7 | 2.40 | 250.0 |

**Key findings:**

1. **Outcome-only scoring is worse than the prompted baseline** (0.514 vs 0.726). Without oracle fallback, outcome_only's loop-action bias — rewarding stall actions (`browse`, `rattle_gate`, `review_notes`) that correlate with eventual success but waste turns — actively degrades performance.

2. **Branch-aware scoring recovers from this failure** (0.971 vs 0.514, +0.457 absolute). Branch-label signals (verifier scores for `pivotal`, penalties for `dead_end`) correctly rank pivotal actions above stall actions even when both appear in successful episodes.

3. **Hindsight scoring also recovers but less reliably** (0.857). Hindsight bonuses for pivotal/recovery branches help but are less discriminative than branch_aware's explicit dead-end penalties on multi-step tasks.

4. **Combined scoring achieves perfect success** (1.000) across all 7 seeds by leveraging both branch-aware penalties and hindsight bonuses.

5. **The scorers form a clear hierarchy**: combined (1.00) > branch_aware (0.97) > hindsight (0.86) > prompted (0.73) > outcome_only (0.51).

### Per-Family Breakdown

| Condition | cascade | dead_end | enabling | multi_gate | recovery |
|-----------|---------|----------|----------|------------|----------|
| prompted | 0.657 | 0.871 | 0.629 | 0.800 | 0.671 |
| outcome_only | 0.143 | 0.571 | 0.714 | 0.143 | 1.000 |
| branch_aware | 1.000 | 1.000 | 0.857 | 1.000 | 1.000 |
| hindsight | 0.286 | 1.000 | 1.000 | 1.000 | 1.000 |
| **combined** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |

**Family-level analysis:**

- **cascade and multi_gate are the hardest families.** Outcome_only collapses to 0.143 on both — the loop-action bias compounds across 3+ sequential decision points, making the model prefer stalling at every gate/stage.
- **Hindsight fails on cascade (0.286)** but succeeds on multi_gate (1.000). Hindsight bonuses reward pivotal branches but do not penalise stalls; on cascade, where the stall action (`review_notes`) has no dead-end label, hindsight cannot distinguish it from progress.
- **Branch_aware's one weakness is enabling (0.857)** — on a minority of seeds, the `browse` stall action accumulates enough branch_aware signal to compete with `collect_clue`. Combined scoring eliminates this by adding hindsight's family-specific enabling bonus.
- **Combined scoring achieves 1.000 on every family** because branch_aware penalties handle dead-ends/stalls while hindsight bonuses handle enabling-specific discrimination.

### Exploration Rate Ablation

The ranking is stable across exploration rates (3 seeds each):

| Condition | ER=0.3 | ER=0.5 | ER=0.7 |
|-----------|--------|--------|--------|
| prompted | 0.853 | 0.667 | 0.527 |
| outcome_only | 0.533 | 0.467 | 0.667 |
| branch_aware | 1.000 | 0.933 | 0.867 |
| hindsight | 0.867 | 0.867 | 0.867 |
| combined | 1.000 | 1.000 | 1.000 |

The condition ranking `combined >= branch_aware >= hindsight >> outcome_only` holds across all tested rates.

### Key Mechanism: Loop-Action Bias

The critical confound that branch-aware scoring corrects is **loop-action bias**: when a stall action (e.g., `browse`, `rattle_gate`, `review_notes`) loops back to the same state and the episode eventually succeeds via a different path, outcome_only scoring assigns the stall action the same score as the pivotal action. On multi-step tasks (multi_gate, cascade), this compounds — the model prefers stalling at every gate rather than progressing, resulting in long episodes that exhaust the turn budget.

Branch-aware scoring directly penalises dead-end branches and rewards verifier signals (e.g., `gate_progress`, `evidence_quality`), breaking the tie and correctly ranking progressive actions above stalls.

### Budget Parity

Rollout count matched across all conditions (50 per run). Turn and verifier-call variation is an outcome of policy quality — better policies solve tasks in fewer turns.

| Metric | Matched | Range |
|--------|---------|-------|
| rollout_count | **yes** | 50 |
| avg_turns | no | 2.40–6.06 (combined shortest, outcome_only longest) |
| verifier_calls | no | 250–394 (combined fewest) |

### Plots

- Success by condition: `artifacts/paper/group_f410ed95cc83/success_by_condition.svg`
- Change rate: `artifacts/paper/group_f410ed95cc83/change_rate.svg`

## Failure Analysis

TODO: include repeated dead-end branches, disagreement points, changed preferred branches, and state abstraction collapses.

## Limitations

TODO: document manifest simplifications, model/backend tradeoffs, and AWS single-GPU constraints.

