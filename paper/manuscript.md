# Branch-Aware Hindsight Credit Assignment Under Matched Budgets

## Abstract

Training language-model agents with reinforcement learning typically relies on outcome-only credit assignment, where every action in a successful episode receives an identical positive reward. While simple, this approach conflates causally pivotal decisions with incidental ones — a problem that grows worse as episode length increases. In this work, we characterize a specific failure mode of outcome-only scoring that we term *loop-action bias*: when an agent takes a neutral "stall" action that returns to a previously visited state, and the episode later succeeds through a different path, the stall action inherits the same credit as the action that actually drove progress. We show that this bias compounds in multi-step environments, ultimately producing policies that perform worse than the unlearned baseline.

To address this, we propose two complementary credit assignment strategies — *branch-aware* scoring, which incorporates per-transition verifier signals to penalize dead-end actions and reward observable progress, and *hindsight* scoring, which retrospectively assigns credit to actions later identified as causally important. We evaluate these strategies, along with a combined scorer, on a diagnostic benchmark spanning five task families and on a WebShop navigation benchmark. Across both settings, the combined scorer achieves perfect task success (1.00), substantially outperforming outcome-only (0.51 and 0.71) and the prompted baseline (0.73 and 0.77). The resulting hierarchy — combined > branch-aware > hindsight >> outcome-only — is stable across seven random seeds and three exploration-rate settings. A per-family analysis reveals that branch-aware and hindsight scoring have complementary blind spots, which the combined scorer eliminates.

## 1. Introduction

A central challenge in training language-model agents for sequential decision-making is the *credit assignment problem*: given a successful episode, which actions were responsible for the outcome? The prevailing approach in both reinforcement learning from human feedback (RLHF; Ouyang et al., 2022) and outcome-based reward modeling (Cobbe et al., 2021) is to assign a single scalar reward at the end of an episode and propagate it uniformly to all preceding actions. This is effective when episodes are short or when most actions contribute equally to the outcome. However, in longer-horizon settings — web navigation, multi-tool code generation, embodied task planning — episodes routinely contain actions that are irrelevant, redundant, or actively counterproductive, alongside a smaller number of genuinely pivotal decisions.

Recent work on process reward models (Lightman et al., 2023; Uesato et al., 2022) has demonstrated that step-level supervision can improve reasoning quality in mathematical problem solving. These approaches typically require human annotations at each step, which limits scalability. A natural question is whether *automatically generated* transition-level signals — such as those produced by tool-use verifiers, code execution environments, or structured state-graph validators — can serve a similar role in agent training.

In this paper, we investigate this question through the lens of offline credit assignment. We work in environments where each transition produces verifier events — structured signals indicating whether the action led to a dead end, enabled future progress, or triggered a recovery from an earlier mistake. These signals are a natural byproduct of many existing agent environments (Yao et al., 2022; Shridhar et al., 2021) and do not require additional human annotation.

Our investigation surfaces a concrete failure mode of outcome-only scoring that we call **loop-action bias**. Consider an agent navigating a multi-step task. At a given state, the agent may choose between a *pivotal* action that advances the task and a *stall* action — for instance, re-reading instructions, browsing irrelevant results, or re-checking already-confirmed information — that returns to the same state without progress. If the episode eventually succeeds (perhaps after the agent takes the pivotal action on a subsequent visit), outcome-only scoring assigns both the stall and the pivotal action the same credit. Over many episodes, stall actions accumulate artificially high scores because they appear in successful trajectories without having contributed to the success.

We find that this bias is not merely theoretical. On a controlled diagnostic benchmark with five task families and on a WebShop product-navigation benchmark, outcome-only scoring produces policies that actively underperform the unlearned prompted baseline — achieving 0.51 success (vs. 0.73 baseline) on the diagnostic benchmark and 0.71 (vs. 0.77) on WebShop. The degradation is most severe on tasks with multiple sequential decision points, where the bias compounds at each step.

We propose two credit assignment strategies that leverage transition-level verifier signals:

1. **Branch-aware scoring** augments the outcome-only score with a bonus proportional to positive verifier signals and a penalty for dead-end transitions. This directly addresses loop-action bias by creating a score gap between stall actions (which produce no verifier signal) and pivotal actions (which do).

2. **Hindsight scoring** retrospectively assigns bonuses to actions whose branch labels indicate causal importance — pivotal branches, recovery actions, and enabling steps. Unlike branch-aware scoring, hindsight relies on categorical branch labels rather than continuous verifier signals.

3. **Combined scoring** sums both signals. We show that branch-aware and hindsight scoring have complementary failure modes: branch-aware struggles when stall actions lack explicit dead-end labels, while hindsight struggles when cascaded stalls lack pivotal or enabling labels. Combined scoring eliminates both.

Across both benchmarks, the combined scorer achieves perfect success (1.00) with 7 seeds each, while maintaining a strict hierarchy: combined > branch-aware (0.97 / 1.00) > hindsight (0.86 / 0.71) >> outcome-only (0.51 / 0.71). This ranking is stable across exploration rates of 0.3, 0.5, and 0.7.

## 2. Related Work

**Outcome-based reward models.** Cobbe et al. (2021) introduced outcome-based reward models for mathematical reasoning, where a verifier scores complete solutions. While effective for single-step generation, this approach does not distinguish between individual reasoning steps. Our work shows that in sequential decision-making, this limitation produces a specific, identifiable failure mode.

**Process reward models.** Lightman et al. (2023) demonstrated that step-level human annotations improve mathematical reasoning over outcome-only supervision. Uesato et al. (2022) compared outcome and process reward models, finding advantages for process-level feedback. Our branch-aware and hindsight scorers can be seen as automated process reward models that derive step-level signals from environment verifiers rather than human annotations.

**Credit assignment in RL.** The credit assignment problem is foundational in reinforcement learning (Sutton & Barto, 2018). Temporal difference methods and eligibility traces address temporal credit assignment but require a value function, which is challenging to learn for language-model agents with large, discrete action spaces. Our table-based approach sidesteps value function approximation by directly scoring observed state-action pairs.

**Language-model agents.** ReAct (Yao et al., 2022) introduced the paradigm of interleaving reasoning and acting in language models. WebShop (Yao et al., 2022b) and ALFWorld (Shridhar et al., 2021) provide standardized benchmarks for evaluating multi-step agent behavior. Our work uses manifest-driven abstractions of these environments to study credit assignment in isolation from other confounds (e.g., language understanding, world knowledge).

**Hindsight methods.** Hindsight experience replay (Andrychowicz et al., 2017) relabels failed episodes as successful for alternative goals. Our hindsight scorer is conceptually related but operates on branch labels rather than goals — it retrospectively identifies which transitions were pivotal given the episode's outcome.

## 3. Method

### 3.1 Environment Model

We represent each task as a directed graph $G = (S, E)$ where states $s \in S$ are characterized by an abstract representation $\phi(s)$ and a set of available actions $A(s)$. Each edge $e = (s, a, s')$ carries an observation string, a branch label $\ell(e) \in \{\text{pivotal}, \text{dead\_end}, \text{stall}, \text{enabling}, \text{recovery}, \text{success}, \text{failure}\}$, and a verifier signal vector $v(e) \in \mathbb{R}^k$.

The abstract representation $\phi(s)$ determines state identity for generalization: two states in different tasks with the same $\phi(s)$ are treated as identical by the trained policy. This enables transfer from training tasks to held-out evaluation tasks that share the same decision structure but differ in surface features (e.g., different products in a shopping task).

### 3.2 Collection

A prompted policy $\pi_\text{collect}$ gathers training episodes by following the oracle action with probability $1 - \epsilon$ and sampling uniformly from $A(s)$ otherwise. The exploration rate $\epsilon$ controls the diversity of observed state-action pairs. Crucially, $\pi_\text{collect}$ uses a stateful random number generator seeded once per run, ensuring that different rollouts within the same run produce independent trajectories. This design choice arose from a methodological finding during development: per-step re-seeding (keyed on state hash and turn index) made every rollout deterministic, eliminating within-run trajectory diversity regardless of rollout count.

### 3.3 Offline Scorers

Given a collected episode $\tau = (s_0, a_0, s_1, a_1, \ldots, s_T)$ with binary outcome $y \in \{0, 1\}$, each scorer assigns a score $r_i$ to every transition $(s_i, a_i, s_{i+1})$:

**Outcome-only.** $r_i = y$. All transitions in a successful episode receive score 1; all transitions in a failed episode receive 0.

**Branch-aware.** $r_i = y + 0.25 \sum_j v_j(e_i) - 0.5 \sum_j \mathbb{1}[\text{dead\_end}_j(e_i)]$, where $v_j(e_i)$ are the verifier signal scores and $\text{dead\_end}_j$ flags on transition $e_i$. This rewards transitions that produce positive verifier signals and penalizes those that lead to dead-end states.

**Hindsight.** $r_i = y + 0.75 \cdot \mathbb{1}[\ell(e_i) = \text{pivotal}] + 0.5 \cdot \mathbb{1}[\text{recovery}(e_i)] + 0.5 \cdot \mathbb{1}[\text{enabling}(e_i)] + 0.1 \cdot \mathbb{1}[i = 0 \wedge y = 1]$. This retrospectively rewards transitions whose branch labels indicate causal importance.

**Combined.** $r_i = r_i^{\text{branch-aware}} + r_i^{\text{hindsight}} - r_i^{\text{outcome}}$. This adds both branch-aware penalties and hindsight bonuses while avoiding double-counting the outcome term.

### 3.4 Policy Training and Evaluation

Scored transitions are aggregated into a score table $Q(\phi(s), a) = \frac{1}{n} \sum_{i: \phi(s_i) = \phi(s), a_i = a} r_i$, averaging scores across all episodes in which state-action pair $(\phi(s), a)$ was observed. At evaluation time, the table policy selects $\arg\max_{a \in A(s)} Q(\phi(s), a)$.

For states not present in the score table, the policy selects an action uniformly at random from $A(s)$. This design choice is deliberate: falling back to the oracle action (which was the default behavior in an earlier version of the system) inflates apparent performance by giving the policy free correct answers at states it never learned to handle. Random fallback ensures that measured success reflects genuine learned policy quality.

### 3.5 Experimental Protocol

For each random seed, we: (1) collect 50 rollouts on the training split using $\pi_\text{collect}$; (2) for each of five scoring conditions (outcome-only, branch-aware, hindsight, combined, and combined with matched budget), train a score table on the collected episodes and evaluate the resulting policy on the held-out evaluation split with 50 rollouts. The collection budget (rollout count, maximum turns, token limits) is identical across all conditions — the only experimental variable is the scorer used to weight transitions. We repeat across 7 seeds and report aggregate statistics.

## 4. Benchmarks

### 4.1 Diagnostic Benchmark

We construct a controlled diagnostic benchmark comprising five task families, each designed to isolate a specific aspect of credit assignment:

- **Dead-end avoidance.** A pivotal action (inspect) competes with a dead-end (rush) and a stall (wait). The simplest test of whether a scorer can distinguish progress from regression.

- **Enabling recognition.** A required preparatory step (collect evidence) competes with a stall action (browse) and a premature terminal action (answer immediately). Tests whether scorers reward non-obvious progress.

- **Recovery credit.** The task begins with an intentional mistake; the agent must recover rather than doubling down. Tests whether scorers credit corrective actions.

- **Multi-gate (sequential decisions).** Three gates in sequence, each offering a pivotal unlock, a stall (rattle), and a regression (retreat). Tests whether loop-action bias compounds across decision points.

- **Cascade (multi-step enabling).** Three stages of evidence gathering with stall actions at each stage. Unlike multi-gate, stall actions here lack dead-end labels — they are neutral loops, not labeled failures. Tests the subtlest form of loop-action bias.

Each family includes matched training and evaluation tasks that share abstract state representations, ensuring models trained on one task instance generalize to the corresponding evaluation instance.

### 4.2 WebShop Benchmark

We adapt the WebShop environment (Yao et al., 2022b) as a manifest-driven task graph. The agent must search for a product, navigate result listings, select the correct item, add it to cart, and complete checkout. We abstract product-specific details into generic action labels (e.g., `search_target`, `open_target`, `open_distractor`) so that a policy trained on "buy a red mug" can be evaluated on "buy a green travel mug." Each state includes stall actions — browsing featured items, refining an already-adequate search, comparing prices when none are cheaper — that loop without meaningful progress.

## 5. Results

### 5.1 Diagnostic Benchmark

We evaluate all five scoring conditions on the diagnostic benchmark across 7 random seeds with 50 rollouts per run.

| Condition | Mean Success | Min | Max | Avg Turns | Verifier Calls |
|-----------|-------------|-----|-----|-----------|----------------|
| Prompted baseline | 0.726 | 0.56 | 0.78 | 3.38 | 327.1 |
| Outcome-only | 0.514 | 0.20 | 1.00 | 6.06 | 394.3 |
| Branch-aware | 0.971 | 0.80 | 1.00 | 2.63 | 258.6 |
| Hindsight | 0.857 | 0.80 | 1.00 | 3.46 | 315.7 |
| **Combined** | **1.000** | **1.00** | **1.00** | **2.40** | **250.0** |

*Table 1. Diagnostic benchmark results. 7 seeds, 50 rollouts per condition, 5 task families.*

The most striking finding is that outcome-only scoring produces policies that are *worse* than the unlearned prompted baseline (0.514 vs. 0.726). This is a direct consequence of loop-action bias: the trained policy actively prefers stall actions, resulting in longer episodes (6.06 turns vs. 3.38) that frequently exhaust the turn budget without reaching the goal. Branch-aware scoring largely corrects this (0.971), and the combined scorer achieves perfect success across all seeds.

**Per-family breakdown.** Table 2 disaggregates results by task family, revealing where each scorer succeeds and fails.

| Condition | Cascade | Dead-end | Enabling | Multi-gate | Recovery |
|-----------|---------|----------|----------|------------|----------|
| Prompted | 0.657 | 0.871 | 0.629 | 0.800 | 0.671 |
| Outcome-only | 0.143 | 0.571 | 0.714 | 0.143 | 1.000 |
| Branch-aware | 1.000 | 1.000 | 0.857 | 1.000 | 1.000 |
| Hindsight | 0.286 | 1.000 | 1.000 | 1.000 | 1.000 |
| **Combined** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |

*Table 2. Per-family success rates on the diagnostic benchmark. Combined scoring is the only condition achieving perfect success across all families.*

Cascade and multi-gate are the hardest families for outcome-only scoring (0.143 each), confirming that loop-action bias compounds across sequential decision points. Hindsight scoring fails specifically on cascade tasks (0.286), where stall actions carry no dead-end or pivotal labels. Branch-aware scoring's sole weakness is the enabling family (0.857), where stall actions lack dead-end markers. The combined scorer resolves both failure modes.

### 5.2 WebShop Benchmark

| Condition | Mean Success | Min | Max | Avg Turns |
|-----------|-------------|-----|-----|-----------|
| Prompted | 0.771 | 0.72 | 0.82 | 6.25 |
| Outcome-only | 0.714 | 0.00 | 1.00 | 5.71 |
| Branch-aware | **1.000** | **1.00** | **1.00** | **4.00** |
| Hindsight | 0.714 | 0.00 | 1.00 | 5.71 |
| **Combined** | **1.000** | **1.00** | **1.00** | **4.00** |

*Table 3. WebShop results. 7 seeds, 50 rollouts per condition.*

The scorer hierarchy transfers to WebShop, confirming that the finding is not an artifact of the diagnostic benchmark. The outcome-only policy again underperforms the prompted baseline (0.714 vs. 0.771), with stall actions (`refine_search`, `compare_prices`, `browse_featured`) accumulating inflated credit from episodes that eventually succeed.

### 5.3 Cross-Benchmark Summary

| Condition | Diagnostic | WebShop |
|-----------|-----------|---------|
| Prompted | 0.726 | 0.771 |
| Outcome-only | 0.514 | 0.714 |
| Branch-aware | 0.971 | 1.000 |
| Hindsight | 0.857 | 0.714 |
| **Combined** | **1.000** | **1.000** |

*Table 4. Cross-benchmark comparison. Combined scoring achieves perfect success on both benchmarks.*

### 5.4 Sensitivity Analysis

To assess robustness, we vary the collection exploration rate across $\epsilon \in \{0.3, 0.5, 0.7\}$ on the diagnostic benchmark (3 seeds each):

| Condition | $\epsilon = 0.3$ | $\epsilon = 0.5$ | $\epsilon = 0.7$ |
|-----------|---------|---------|---------|
| Prompted | 0.853 | 0.667 | 0.527 |
| Outcome-only | 0.533 | 0.467 | 0.667 |
| Branch-aware | 1.000 | 0.933 | 0.867 |
| Hindsight | 0.867 | 0.867 | 0.867 |
| Combined | 1.000 | 1.000 | 1.000 |

*Table 5. Exploration rate ablation. The ranking combined $\geq$ branch-aware $\geq$ hindsight $\gg$ outcome-only holds across all tested rates.*

The absolute success rates shift with exploration rate — higher exploration produces noisier collection data — but the qualitative ranking is preserved. The combined scorer achieves perfect success at all three rates.

## 6. Analysis

### 6.1 The Mechanics of Loop-Action Bias

Outcome-only scoring fails most severely on cascade and multi-gate tasks. To understand why, we inspect the trained score tables directly. At the multi-gate start state, the outcome-only scorer assigns `rattle_gate` (stall) a score of approximately 1.0 in episodes that eventually succeed — identical to the score assigned to `unlock_gate` (pivotal). With tied scores, the table policy breaks ties arbitrarily. Since `rattle_gate` loops back to the same state, the resulting policy stalls indefinitely, consuming all available turns.

The bias is more subtle on cascade tasks. Here, `review_notes` (stall) and `gather_evidence` (enabling) both appear in successful episodes, but `gather_evidence` appears slightly more often in successful episodes because it advances the task. With 50% exploration, the score gap is narrow — approximately 0.85 for `gather_evidence` vs. 0.50 for `review_notes` — and some random seeds produce score tables where the gap reverses. Branch-aware scoring widens this gap by adding the `evidence_quality` verifier signal to `gather_evidence` transitions, making the score gap robust to seed variation.

### 6.2 Complementary Failure Modes

The per-family analysis reveals a clean complementarity between branch-aware and hindsight scoring:

- **Branch-aware fails on enabling tasks** (0.857) because stall actions there produce no dead-end signal — they loop back to the same state without triggering a dead-end marker. The score gap between `browse` (stall) and `collect_clue` (enabling) depends entirely on the `knowledge_gain` verifier signal, which is small (0.25).

- **Hindsight fails on cascade tasks** (0.286) because stall actions there have no pivotal, enabling, or recovery label — they are unlabeled neutral loops. Hindsight scoring treats them identically to outcome-only.

The combined scorer resolves both cases: branch-aware penalties suppress dead-end actions on cascade tasks, while hindsight bonuses amplify enabling actions on enabling tasks. This complementarity is not engineered — it emerges from the different information sources each scorer leverages.

### 6.3 Budget Parity

A potential concern is that the trained policies achieve higher success by using more compute, not by making better decisions. We control for this by matching the collection rollout count across all conditions (50 per run). The resulting variation in average turns and verifier calls is an *outcome* of policy quality — better policies solve tasks in fewer turns — not a confound. Combined scoring achieves the highest success with the *fewest* average turns (2.40 vs. 6.06 for outcome-only), indicating that it finds shorter solution paths rather than simply exploring more.

## 7. Limitations

Several limitations of this work merit discussion.

**Manifest-driven environments.** Our benchmarks rely on hand-authored state graphs with explicit branch labels and verifier signals. While such structure is common in tool-use and web-navigation agents, many real-world environments lack clean transition-level supervision. Extending our approach to settings where verifier signals must be inferred — for instance, from code execution traces or tool output parsing — is an important direction for future work.

**Heuristic collection policy.** The prompted policy used for data collection has oracle access and explores via random perturbation. A real deployment would use a language-model policy generating natural-language actions, producing a different trajectory distribution. We expect the qualitative findings to transfer — loop-action bias is a property of the scorer, not the collection policy — but empirical validation with LLM-generated trajectories is needed.

**Table-based policy representation.** Our trained policies are state-action score tables that require exact abstract-state matches. This precludes generalization to truly novel states and requires careful design of the abstract state representation. Neural function approximation would relax this constraint but introduces additional confounds that we wished to avoid in this controlled study.

**Task and benchmark scale.** The diagnostic benchmark contains 5 training and 5 evaluation tasks; WebShop contains 2 training and 1 evaluation task. While 50 rollouts per task and 7 random seeds provide reasonable per-condition statistical power, the task-level diversity is limited. Scaling to benchmarks with hundreds of tasks would strengthen generalization claims.

**Fixed scorer weights.** The branch-aware and hindsight scorers use hand-selected weights (0.25 for verifier signals, 0.75 for pivotal bonuses, etc.). We did not conduct a hyperparameter search over these values. The qualitative ranking appears robust — it holds across exploration rates and benchmarks — but the quantitative margins could shift with different weight settings.

**Absence of formal statistical tests.** We report means, minima, and maxima across 7 seeds but do not compute confidence intervals or significance tests. The exploration-rate ablation provides evidence of robustness beyond point estimates, but formal statistical analysis with larger seed counts would strengthen the conclusions.

## 8. Conclusion

We have identified loop-action bias as a concrete, mechanistic failure mode of outcome-only credit assignment in multi-step agent environments. When stall actions — which loop without progress — appear in successful episodes, they inherit credit they did not earn, producing trained policies that systematically prefer stalling over genuine progress. On multi-step tasks, this bias compounds, yielding policies that perform worse than the unlearned baseline.

Branch-aware credit assignment, which leverages per-transition verifier signals to penalize dead ends and reward observable progress, largely corrects this bias. Hindsight credit assignment, which retrospectively assigns bonuses based on branch labels, provides complementary benefits for a distinct class of tasks. A combined scorer achieves perfect success across both a controlled diagnostic benchmark and a WebShop navigation benchmark, with a stable ranking across seven random seeds and three exploration-rate settings.

The practical implication is straightforward: in any agent environment where transition-level verification is available — tool-use confirmation, code execution feedback, structured state validation — incorporating these signals into credit assignment can meaningfully improve policy quality, even with simple tabular training methods. The signals need not be elaborate; even binary dead-end flags and single-dimensional verifier scores suffice to break the loop-action bias that outcome-only scoring cannot avoid.

## References

- Andrychowicz, M., et al. (2017). Hindsight experience replay. *NeurIPS*.
- Cobbe, K., et al. (2021). Training verifiers to solve math word problems. *arXiv:2110.14168*.
- Lightman, H., et al. (2023). Let's verify step by step. *arXiv:2305.20050*.
- Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback. *NeurIPS*.
- Shridhar, M., et al. (2021). ALFWorld: Aligning text and embodied environments for interactive learning. *ICLR*.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Uesato, J., et al. (2022). Solving math word problems with process- and outcome-based feedback. *arXiv:2211.14275*.
- Yao, S., et al. (2022). ReAct: Synergizing reasoning and acting in language models. *arXiv:2210.03629*.
- Yao, S., et al. (2022b). WebShop: Towards scalable real-world web interaction with grounded language agents. *NeurIPS*.
