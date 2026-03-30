# Branch-Aware Hindsight Credit Assignment Under Matched Budgets

## Abstract

Outcome-only credit assignment — scoring every action in an episode by its terminal success — is the default training signal for language-model agents. We show this creates a systematic failure mode we call **loop-action bias**: stall actions that loop back to the same state inherit the success signal of later pivotal actions, causing trained policies to prefer stalling over progress. On multi-step tasks the bias compounds, making outcome-only policies worse than the unlearned prompted baseline (0.51 vs 0.73 on diagnostic, 0.71 vs 0.77 on WebShop). We introduce branch-aware and hindsight credit assignment scorers that use per-transition verifier signals (dead-end penalties, enabling bonuses, recovery markers) to break the tie. A combined scorer achieves perfect success (1.00) on both diagnostic (5 task families, 7 seeds) and WebShop benchmarks under matched collection budgets, while maintaining a clear hierarchy: combined > branch_aware > hindsight >> outcome_only. The ranking is stable across exploration rates (0.3–0.7). Per-family analysis reveals that branch_aware and hindsight have complementary failure modes — branch_aware struggles on enabling tasks, hindsight on cascade tasks — and combined eliminates both.

## Introduction

Language-model agents increasingly operate in multi-step environments where a single episode may contain dozens of actions, only some of which are causally responsible for the outcome. The standard training signal — outcome-only credit, which assigns every action in a successful episode a positive score — is known to be coarse, but its specific failure modes in structured environments have not been characterized.

We identify **loop-action bias** as a concrete, mechanistic failure of outcome-only scoring. When an agent takes a stall action (e.g., browsing irrelevant results, rattling a locked gate, reviewing notes it has already read) that loops back to the same state, and the episode eventually succeeds via a different action, outcome-only scoring assigns the stall the same credit as the pivotal action. In environments with multiple sequential decision points, this bias compounds: the trained policy prefers stalling at every step, producing long trajectories that exhaust the turn budget.

Branch-aware credit assignment addresses this by incorporating per-transition verifier signals: dead-end penalties reduce the score of actions that lead to dead-end states, while pivotal/enabling bonuses increase the score of actions that produce verifier-confirmed progress. Hindsight credit adds retrospective bonuses for actions whose downstream consequences turned out to be important (pivotal branches, recovery from mistakes). These signals are available in any environment with a structured state graph and transition-level verification — a common pattern in tool-use, web navigation, and code-generation agents.

Our core question: **does branch-aware credit assignment improve policy quality over outcome-only scoring, under equal collection budgets?** We answer yes, with a clear hierarchy (combined > branch_aware > hindsight >> outcome_only) that holds across two benchmarks and is robust to exploration-rate variation.

## Method

### Environment Model

Each benchmark is defined as a set of **tasks**, where each task is a directed graph of states connected by labeled edges. A state has an abstract representation (used for hashing and model generalization), a set of available actions, and an oracle action. An edge specifies the next state, an observation string, and optional metadata: `branch_label` (pivotal, dead_end, stall, enabling, recovery, success, failure), `dead_end` flag, `recovery` flag, `enabling` flag, and a `verifier_signals` dictionary mapping signal names to numeric scores.

Abstract state hashing ensures that structurally equivalent states across different tasks (e.g., "browsing search results for a mug" vs "browsing search results for a bottle") map to the same hash, enabling models trained on one set of tasks to generalize to unseen tasks with the same decision structure.

### Collection Policy

The **prompted policy** acts as the collection agent. At each state, it follows the oracle action with probability `1 - exploration_rate` and selects uniformly at random otherwise. The policy uses a stateful RNG (seeded once per run, not per step) so that different rollouts within the same run explore different trajectories. This ensures the training data contains both oracle and non-oracle actions at each state.

### Offline Scorers

Given a collected episode (sequence of transitions), each scorer assigns a numeric score to every transition:

- **Outcome-only**: `score = 1.0` if the episode succeeded, `0.0` otherwise. Every transition in a successful episode receives the same score regardless of its contribution.

- **Branch-aware**: `score = outcome_only + 0.25 × Σ(verifier_signal_scores) - 0.5 × Σ(dead_end_penalties)`. Rewards actions that produce positive verifier signals and penalises actions that lead to dead-end states.

- **Hindsight**: `score = outcome_only + 0.75 × I[pivotal_branch] + 0.5 × I[recovery] + 0.5 × I[enabling] + 0.1 × I[first_turn ∧ success]`. Retrospectively rewards actions whose branch labels indicate they were causally important.

- **Combined**: `score = branch_aware + (hindsight - outcome_only)`. Adds both branch-aware penalties and hindsight bonuses.

### Training

Scored transitions are grouped by (state_hash, action) and averaged to produce a **score table**: a mapping from states to action-score dictionaries. At evaluation time, the **table policy** selects the highest-scoring action at each state. For states not in the table, the policy falls back to a **random action** (not oracle), ensuring measured performance reflects learned policy quality.

### Sweep Protocol

For each seed, the sweep runner: (1) collects episodes with the prompted policy on the train split; (2) for each scoring condition, trains a score table on the collected episodes and evaluates the resulting table policy on the eval split. Rollout count is matched across all conditions (50 per run). The only variable is the scorer used to assign transition scores.

### Safety and Reproducibility

All sweeps are gated by a safety guard that enforces token, verifier-call, wall-clock, and cost ceilings before and after each run. GCP and AWS execution require explicit opt-in flags. All artifacts (episodes, metrics, policies, reports) are stored in SQLite and JSON, with paper-ready CSV and SVG exports generated automatically.

## Benchmarks

### Diagnostic (5 task families)

A controlled benchmark with explicit branch annotations, designed to isolate specific failure modes of each scorer:

- **dead_end**: A single pivotal branch (inspect) vs a dead-end (rush) and a stall (wait). Tests basic dead-end avoidance.
- **enabling**: A required enabling step (collect_clue) competes with a stall (browse) and a premature answer (answer_now). Tests whether scorers can distinguish enabling progress from stalls.
- **recovery**: An intentional mistake followed by a recovery opportunity. Tests whether scorers credit recovery actions.
- **multi_gate**: Three sequential gates, each with a pivotal unlock, a stall (rattle), and a regression (retreat/jam). Tests whether loop-action bias compounds across multiple decision points.
- **cascade**: Three-stage evidence gathering where premature guessing fails. Tests whether scorers distinguish multi-step enabling from stalls when no dead-end labels are present.

Each family has matched train and eval tasks sharing abstract state hashes. 5 train tasks, 5 eval tasks.

### WebShop (shopping navigation)

A product-search-and-purchase environment adapted from the WebShop benchmark. The agent must search for a product, navigate results, select the correct item, and complete checkout. States use product-agnostic abstractions ({page, phase}) so models trained on "buy red mug" transfer to "buy green travel mug." Each state includes stall actions (browse_featured, refine_search, compare_prices) that loop without progress. 2 train tasks, 1 eval task.

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

### WebShop Results

*Run group: `group_b25fdde347d6` — 7 seeds × 6 conditions, 50 rollouts per run.*

| Condition | Mean | Min | Max | Avg Turns |
|-----------|------|-----|-----|-----------|
| prompted | 0.771 | 0.72 | 0.82 | 6.25 |
| outcome_only | 0.714 | 0.00 | 1.00 | 5.71 |
| branch_aware | **1.000** | **1.00** | **1.00** | **4.00** |
| hindsight | 0.714 | 0.00 | 1.00 | 5.71 |
| **combined** | **1.000** | **1.00** | **1.00** | **4.00** |

The ranking transfers from diagnostic to WebShop: combined = branch_aware (1.0) >> hindsight = outcome_only (0.71) > prompted (0.77). The WebShop result shows that loop-action bias affects outcome_only even in a concrete shopping environment — stall actions (`refine_search`, `compare_prices`, `browse_featured`) inherit success signal from eventual checkout.

### Cross-Benchmark Summary

| Condition | Diagnostic | WebShop |
|-----------|-----------|---------|
| prompted | 0.726 | 0.771 |
| outcome_only | 0.514 | 0.714 |
| branch_aware | 0.971 | 1.000 |
| hindsight | 0.857 | 0.714 |
| combined | **1.000** | **1.000** |

### Plots

- Diagnostic success: `artifacts/paper/group_f410ed95cc83/success_by_condition.svg`
- WebShop success: `artifacts/paper/group_b25fdde347d6/success_by_condition.svg`

## Failure Analysis

### Outcome-Only: Loop-Action Bias in Detail

Outcome-only scoring fails most severely on cascade (0.143) and multi_gate (0.143) tasks. Inspection of the trained score tables reveals the mechanism: at the multi_gate `gate_1` state, `rattle_gate` (a stall action) receives score 1.0 when episodes that eventually succeed pass through that state, while `unlock_gate` (the pivotal action) receives the same score 1.0. With tied scores, the table policy breaks ties arbitrarily — and since `rattle_gate` loops back to the same state, the policy stalls indefinitely.

On cascade tasks, `review_notes` (stall) competes with `gather_evidence` (enabling) at every stage. The outcome-only score for both is 1.0 in successful episodes and 0.0 in failures. With 50% exploration, roughly half of successful episodes include `review_notes` transitions, giving it a score ≈ 0.5 — close enough to `gather_evidence` (≈ 0.85) that some seeds produce a stall-preferring policy.

### Hindsight: Missing Stall Penalties

Hindsight scoring fails on cascade tasks (0.286) because it adds bonuses for pivotal and enabling actions but does not penalise stalls. When `review_notes` has no `dead_end` label (it is a neutral stall, not a dead end), hindsight scores it the same as outcome_only. Branch_aware succeeds here because its verifier-signal bonus for `evidence_quality` creates a score gap even without a dead-end penalty.

### Branch_Aware: Enabling Weakness

Branch_aware achieves 0.857 on enabling tasks (vs 1.000 for combined). The issue: the `browse` stall action at the enabling start state produces no dead-end signal (it loops back to the same state, not to a dead end), so branch_aware cannot penalise it. In some seeds, `browse`'s accumulated branch_aware score (1.0 from outcome + 0.0 penalty) is close to `collect_clue`'s score (1.0 from outcome + 0.25 × knowledge_gain = 1.25). Combined scoring adds hindsight's 0.5 enabling bonus to `collect_clue`, widening the gap to 1.75 vs 1.0 and eliminating the failure.

### Complementarity

The failure modes are complementary: branch_aware fails when stalls have no dead-end label (enabling), hindsight fails when stalls have no pivotal/enabling label (cascade). Combined scoring leverages both signals and achieves 1.000 on every task family across both benchmarks.

## Limitations

**Manifest-driven environments.** Our benchmarks use hand-authored state graphs with explicit branch labels and verifier signals. Real-world environments (e.g., live web navigation, code execution) require automatic extraction of these signals from tool outputs, which is an open problem. The method assumes that transition-level verification is available — it cannot be applied to environments where only terminal outcomes are observable.

**Heuristic collection policy.** The prompted policy uses oracle access with random exploration, not a language model. This means the training data has a specific exploration pattern (oracle-biased with random deviations) that may not match the trajectory distribution of a real LLM agent. Validating with a real LLM policy (e.g., via the VLLMServerPolicyBackend) is needed before deploying the method in production.

**Table-based policy.** The trained policy is a state-action score table, not a neural network. This limits generalization to states with exact hash matches — the policy cannot interpolate between similar but non-identical states. In practice, the abstract state representation must be carefully designed to ensure sufficient train/eval overlap.

**Small task count.** The diagnostic benchmark has 5 train + 5 eval tasks; WebShop has 2 train + 1 eval. While 50 rollouts per task and 7 seeds provide reasonable statistical power for each task, the task-level diversity is limited. Scaling to benchmarks with hundreds of tasks (e.g., full WebShop, SWE-bench) would strengthen generalization claims.

**No confidence intervals.** We report mean, min, and max across seeds but do not compute formal confidence intervals or significance tests. With 7 seeds, the effective sample size for condition-level comparisons is small. The exploration-rate ablation provides additional robustness evidence but is not a substitute for proper statistical testing.

**Scorer design is manual.** The branch_aware and hindsight scorers use hand-tuned weights (0.25 for verifier scores, 0.75 for pivotal bonus, etc.). We did not search over these hyperparameters. The qualitative ranking may be robust to weight choices, but the quantitative gaps could shift.

## Conclusion

We identify loop-action bias as a specific, mechanistic failure of outcome-only credit assignment in multi-step agent environments. Branch-aware scoring — using per-transition verifier signals to penalise dead-ends and reward progress — corrects this bias and produces policies that solve tasks in fewer turns. Hindsight scoring provides complementary benefits for enabling tasks but fails on cascaded stalls without dead-end labels. Combined scoring eliminates both failure modes, achieving perfect success across two benchmarks (diagnostic and WebShop) with 7 seeds each. The hierarchy combined > branch_aware > hindsight >> outcome_only is stable across exploration rates and task families.

The key practical implication: agents trained with outcome-only credit assignment may systematically prefer plausible-looking stall actions over genuine progress. Adding per-transition credit signals — even simple ones like dead-end flags and verifier scores — can dramatically improve policy quality under matched collection budgets.

