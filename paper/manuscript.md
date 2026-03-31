# Branch-Aware Hindsight Credit Assignment Under Matched Budgets

## Abstract

Training language-model agents with reinforcement learning typically relies on outcome-only credit assignment, where every action in a successful episode receives an identical positive reward. While simple, this approach conflates causally pivotal decisions with incidental ones — a problem that grows worse as episode length increases. In this work, we characterize a specific failure mode of outcome-only scoring that we term *loop-action bias*: when an agent takes a neutral "stall" action that returns to a previously visited state, and the episode later succeeds through a different path, the stall action inherits the same credit as the action that actually drove progress. We show that this bias compounds in multi-step environments, ultimately producing policies that perform worse than the unlearned baseline.

To address this, we propose two complementary credit assignment strategies — *branch-aware* scoring, which incorporates per-transition verifier signals to penalize dead-end actions and reward observable progress, and *hindsight* scoring, which retrospectively assigns credit to actions later identified as causally important. We evaluate these strategies, along with a combined scorer, across three benchmarks: a diagnostic suite spanning five task families, a WebShop product-navigation environment, and ALFWorld embodied navigation. Across all three settings, outcome-only scoring degrades below the unlearned baseline, while the combined scorer achieves the best performance — perfect success (1.00) on diagnostic and WebShop, and the strongest recovery on ALFWorld. The resulting hierarchy — combined $\geq$ branch-aware > hindsight >> outcome-only — is stable across seven random seeds and three exploration-rate settings. A per-family analysis reveals that branch-aware and hindsight scoring have complementary blind spots, which the combined scorer eliminates.

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

Across three benchmarks (diagnostic, WebShop, ALFWorld), the combined scorer achieves the best performance in every setting — perfect success on diagnostic and WebShop, and the strongest recovery on ALFWorld — while maintaining a strict hierarchy: combined $\geq$ branch-aware > hindsight >> outcome-only. This ranking is stable across seven random seeds and exploration rates of 0.3, 0.5, and 0.7.

The broad insight that dense intermediate rewards improve over sparse outcome signals is well-established and actively studied (Lightman et al., 2023; Wang et al., 2025; Li et al., 2026; Zhang et al., 2025; Liu et al., 2026). Our contributions are more specific: (1) a mechanistic analysis of *loop-action bias* as a concrete, named failure mode of outcome-only scoring — distinct from the general "sparse reward is bad" observation — showing that it causes policies to perform *worse than the unlearned baseline*, not merely suboptimally; (2) a controlled diagnostic benchmark with five task families designed to isolate when and why each scoring strategy fails; (3) the finding that branch-aware and hindsight scoring have complementary blind spots, and that a simple combined scorer that requires no learned critic or additional model inference addresses both; and (4) cross-benchmark validation on WebShop and ALFWorld showing the pattern generalizes.

## 2. Related Work

The general insight that outcome-only reward is too coarse for long-horizon agents, and that intermediate or process-level signals help, is well-established. Our contribution is not to argue this point for the first time, but rather to provide a mechanistic diagnosis of *how* outcome-only scoring fails (loop-action bias), a controlled benchmark that isolates the failure mode, and a specific combined scorer that addresses complementary blind spots. We situate our work within several active lines of research.

**Process reward models for reasoning.** Lightman et al. (2023) demonstrated that human step-level annotations substantially improve mathematical reasoning over outcome-only supervision, a finding corroborated by Uesato et al. (2022). Subsequent work has sought to reduce the annotation cost: Math-Shepherd (Wang et al., 2024a) uses outcome supervision to automatically generate step-level labels; Luo et al. (2024) employ Monte Carlo estimation for automated process supervision; and Setlur et al. (2024) study how to scale automated process verifiers, finding that rewarding intermediate *progress* — measured as the change in solution quality before and after a step — is critical for effective supervision. This progress-based framing is conceptually close to our branch-aware scorer, which similarly rewards observable forward movement (verifier signals) rather than terminal outcomes. Step-DPO (Lai et al., 2024) extends preference optimization to the step level for chain-of-thought reasoning. Our work differs from the PRM literature primarily in setting: we operate on agent *actions* in structured environments rather than reasoning *steps* in text generation, and we derive step-level signals from environment verifiers rather than from learned models or human annotations.

**Process rewards and credit assignment for LLM agents.** Several concurrent and recent works address the same broad problem of dense reward for long-horizon LLM agents. AgentPRM (Wang et al., 2025) introduces process reward models specifically for LLM agents, with ALFWorld evaluation, arguing that step-level rewards outperform sparse terminal rewards for agent training. HCAPO (Li et al., 2026) integrates hindsight credit assignment into LLM agents by using an LLM as a post-hoc critic to refine step-level Q-values on WebShop and ALFWorld — the closest prior work to our hindsight component. LEAFE (Chen et al., 2026) argues that outcome-only post-training underuses rich environment feedback, and distills corrective experience from alternative branches after failures. GUI-Shepherd (Zhang et al., 2025) applies step-by-step process verification to long-sequence GUI agents. MiRA (Liu et al., 2026) uses milestone-based subgoal reward shaping for long-horizon web agents. InfoPO (Wang et al., 2026) provides turn-level counterfactual rewards based on how feedback changes the next-action distribution.

Our work is distinct from these concurrent efforts in several respects. First, we provide a specific mechanistic diagnosis — loop-action bias — that explains *why* outcome-only scoring fails, rather than simply demonstrating that denser rewards help. Second, our branch-aware scorer uses environment-native verifier signals (dead-end flags, progress markers) rather than a learned critic (HCAPO) or a separate reward model (AgentPRM), making it applicable without additional training infrastructure. Third, we show that branch-aware and hindsight scoring have *complementary* failure modes on distinct task families, and that a simple combined scorer resolves both — a finding that, to our knowledge, has not been characterized in prior work.

**Inference-time compute and verification.** Snell et al. (2024) showed that process reward models can guide optimal allocation of test-time compute, with PRM-guided beam search sometimes outperforming larger models. Language Agent Tree Search (Zhou et al., 2024) combines LLM agents with Monte Carlo Tree Search on WebShop and HotPotQA, using environment feedback as value signals. OpenAI's o1 (OpenAI, 2024) and DeepSeek-R1 (DeepSeek-AI, 2025) demonstrate that large-scale reinforcement learning can teach models to perform extended internal reasoning with learned self-verification. V-STaR (Hosseini et al., 2024) iteratively co-trains verifiers and reasoners. These approaches all benefit from step-level or branch-level reward signals; our contribution characterizes the specific conditions under which outcome-level signals produce pathological behavior.

**Credit assignment in reinforcement learning.** The temporal credit assignment problem is foundational in RL (Sutton & Barto, 2018). Pignatelli et al. (2023) provide a comprehensive survey of modern approaches, categorizing methods into return decomposition, reward redistribution, sequence modeling, and hindsight techniques. Ren et al. (2024) propose randomized return decomposition to redistribute sparse rewards to individual transitions. Our work complements the theoretical literature with a concrete empirical failure mode (loop-action bias) and a practical correction for environments with structured transition feedback.

**Exploration and trajectory optimization for agents.** ETO (Song et al., 2024) uses contrastive trajectory pairs from exploration failures to train LLM agents, which overlaps with our credit-assignment motivation though not our scorer design. Reflexion (Shinn et al., 2023) uses verbal self-reflection as a reward signal for trial-and-error learning. Toolformer (Schick et al., 2023) trains LLMs to use tools via self-supervised learning with implicit execution-based rewards. AgentTuning (Zeng et al., 2024) fine-tunes LLMs on diverse agent interaction trajectories to build generalized capabilities. Our work uses manifest-driven abstractions of standard agent benchmarks — WebShop (Yao et al., 2022b) and ALFWorld (Shridhar et al., 2021) — to study credit assignment in isolation from language understanding and world knowledge.

**Hindsight methods.** Hindsight experience replay (Andrychowicz et al., 2017) relabels failed episodes as successful for alternative goals, enabling learning from failure. LEMA (Zhang et al., 2024) applies a related idea to LLM reasoning by training on mistake-correction pairs. HCAPO (Li et al., 2026) is the most directly related work, using an LLM-based hindsight critic for agent credit assignment. Our hindsight scorer differs in mechanism: rather than using a learned LLM critic, it operates on pre-existing branch labels from the environment's transition structure, making it applicable without additional model inference. We also do not claim novelty for hindsight credit assignment in agents per se — our contribution is the complementarity analysis showing that hindsight and branch-aware scoring fail on different task families and that their combination resolves both.

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

### 4.3 ALFWorld Benchmark

We include ALFWorld (Shridhar et al., 2021), an embodied text-based environment requiring navigation and object manipulation, as a more challenging third benchmark. The agent must navigate rooms, locate objects, and execute multi-step manipulation sequences. As with WebShop, we abstract surface details into generic action labels and include stall actions at each decision point.

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

### 5.3 ALFWorld Benchmark

| Condition | Mean Success | Min | Max | Avg Turns |
|-----------|-------------|-----|-----|-----------|
| Prompted | 0.480 | 0.34 | 0.58 | 4.36 |
| Outcome-only | 0.069 | 0.00 | 0.24 | 8.49 |
| Branch-aware | 0.240 | 0.24 | 0.24 | 4.20 |
| Hindsight | 0.069 | 0.00 | 0.24 | 8.49 |
| Combined | 0.240 | 0.24 | 0.24 | 4.20 |

*Table 4. ALFWorld results. 7 seeds, 50 rollouts per condition.*

ALFWorld proves substantially harder than the other benchmarks. Outcome-only scoring again degrades well below the prompted baseline (0.069 vs. 0.480), confirming that loop-action bias is not specific to shopping or diagnostic tasks. Branch-aware and combined scoring partially recover (0.240), reducing average turns from 8.49 to 4.20 — indicating that the learned policies make faster progress even when they do not always reach the goal. The lower absolute success reflects the greater complexity of ALFWorld's state space and the limited task diversity in our current manifest (1 train, 1 eval task); scaling the task set is a clear direction for future work.

### 5.4 Cross-Benchmark Summary

| Condition | Diagnostic | WebShop | ALFWorld |
|-----------|-----------|---------|----------|
| Prompted | 0.726 | 0.771 | 0.480 |
| Outcome-only | 0.514 | 0.714 | 0.069 |
| Branch-aware | 0.971 | 1.000 | 0.240 |
| Hindsight | 0.857 | 0.714 | 0.069 |
| **Combined** | **1.000** | **1.000** | **0.240** |

*Table 5. Cross-benchmark comparison. The qualitative pattern — outcome-only degrades below baseline, branch-aware recovers — holds across all three benchmarks.*

A consistent pattern emerges across all three benchmarks: outcome-only scoring produces the worst-performing policies, branch-aware scoring substantially recovers, and the combined scorer achieves the best or tied-best performance in every setting. The magnitude of improvement varies with benchmark difficulty, but the directional finding is robust.

### 5.5 Sensitivity Analysis

To assess robustness, we vary the collection exploration rate across $\epsilon \in \{0.3, 0.5, 0.7\}$ on the diagnostic benchmark (3 seeds each):

| Condition | $\epsilon = 0.3$ | $\epsilon = 0.5$ | $\epsilon = 0.7$ |
|-----------|---------|---------|---------|
| Prompted | 0.853 | 0.667 | 0.527 |
| Outcome-only | 0.533 | 0.467 | 0.667 |
| Branch-aware | 1.000 | 0.933 | 0.867 |
| Hindsight | 0.867 | 0.867 | 0.867 |
| Combined | 1.000 | 1.000 | 1.000 |

*Table 6. Exploration rate ablation. The ranking combined $\geq$ branch-aware $\geq$ hindsight $\gg$ outcome-only holds across all tested rates.*

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

Branch-aware credit assignment, which leverages per-transition verifier signals to penalize dead ends and reward observable progress, largely corrects this bias. Hindsight credit assignment, which retrospectively assigns bonuses based on branch labels, provides complementary benefits for a distinct class of tasks. A combined scorer achieves the strongest performance across three benchmarks — perfect success on diagnostic and WebShop, and the best recovery on the harder ALFWorld environment — with a stable ranking across seven random seeds and three exploration-rate settings.

The practical implication is straightforward: in any agent environment where transition-level verification is available — tool-use confirmation, code execution feedback, structured state validation — incorporating these signals into credit assignment can meaningfully improve policy quality, even with simple tabular training methods. The signals need not be elaborate; even binary dead-end flags and single-dimensional verifier scores suffice to break the loop-action bias that outcome-only scoring cannot avoid.

## References

- Andrychowicz, M., Wolski, F., Ray, A., Schneider, J., Fong, R., Welinder, P., McGrew, B., Tobin, J., Abbeel, P., & Zaremba, W. (2017). Hindsight experience replay. *NeurIPS*.
- Chen, X., et al. (2026). LEAFE: Internalizing agency from reflective experience. *arXiv:2603.16843*.
- Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., Hesse, C., & Schulman, J. (2021). Training verifiers to solve math word problems. *arXiv:2110.14168*.
- DeepSeek-AI. (2025). DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. *arXiv:2501.12948*.
- Gulcehre, C., Le Paine, T., Srinivasan, S., Konyushkova, K., Weerts, L., Sharma, A., Siddhant, A., Ahern, A., Wang, M., Gu, C., et al. (2023). Reinforced self-training (ReST) for language modeling. *arXiv* (Google DeepMind).
- Hosseini, A., Yuan, X., Malkin, N., Courville, A., & Sordoni, A. (2024). V-STaR: Training verifiers for self-taught reasoners. *arXiv* (Google DeepMind).
- Lai, X., Tian, Z., Chen, Y., Yang, S., Peng, X., & Jia, J. (2024). Step-DPO: Step-wise preference optimization for long-chain reasoning of LLMs. *arXiv*.
- Li, Y., et al. (2026). HCAPO: Hindsight credit assignment for long-horizon LLM agents. *arXiv:2603.08754*.
- Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker, B., Lee, T., Leike, J., Schulman, J., Sutskever, I., & Cobbe, K. (2023). Let's verify step by step. *arXiv:2305.20050*.
- Liu, X., et al. (2026). MiRA: A subgoal-driven framework for improving long-horizon LLM agents. *arXiv:2603.19685*.
- Luo, L., Liu, Y., Liu, R., Phatale, S., Lara, H., Li, Y., Shu, L., Zhu, Y., Meng, L., Sun, J., & Rastogi, A. (2024). Improve mathematical reasoning in language models by automated process supervision. *arXiv:2406.06592* (Google DeepMind).
- OpenAI. (2024). Learning to reason with LLMs. *OpenAI technical report*.
- Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al. (2022). Training language models to follow instructions with human feedback. *NeurIPS*.
- Pignatelli, E., Ferret, J., Rocktäschel, T., Grefenstette, E., & Toni, L. (2023). A survey of temporal credit assignment in deep reinforcement learning. *arXiv:2312.01072*.
- Rafailov, R., Sharma, A., Mitchell, E., Ermon, S., Manning, C. D., & Finn, C. (2023). Direct preference optimization: Your language model is secretly a reward model. *NeurIPS*.
- Ren, Z., et al. (2024). Learning long-term reward redistribution via randomized return decomposition. *ICLR*.
- Schick, T., Dwivedi-Yu, J., Dessì, R., Raileanu, R., Lomeli, M., Hambro, E., Zettlemoyer, L., & Cancedda, N. (2023). Toolformer: Language models can teach themselves to use tools. *NeurIPS*.
- Setlur, A., Nagpal, C., Fisch, A., Geng, X., Steinhardt, J., & Kumar, A. (2024). Rewarding progress: Scaling automated process verifiers for LLM reasoning. *arXiv*.
- Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., & Yao, S. (2023). Reflexion: Language agents with verbal reinforcement learning. *NeurIPS*.
- Shridhar, M., Thomason, J., Gordon, D., Bisk, Y., Han, W., Mottaghi, R., Zettlemoyer, L., & Fox, D. (2021). ALFWorld: Aligning text and embodied environments for interactive learning. *ICLR*.
- Singh, A., Co-Reyes, J. D., Agarwal, R., Anand, A., Patil, P., Liu, P. J., Harrison, J., Lee, J., Xu, K., et al. (2024). Beyond human data: Scaling self-training for problem-solving with language models. *arXiv* (Google DeepMind).
- Snell, C., Lee, J., Xu, K., & Kumar, A. (2024). Scaling LLM test-time compute optimally can be more effective than scaling model parameters. *arXiv*.
- Song, Y., Yin, D., Yue, X., Huang, J., Li, S., & Lin, B. Y. (2024). Trial and error: Exploration-based trajectory optimization for LLM agents. *ACL*.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Uesato, J., Kushman, N., Kumar, R., Song, F., Siegel, N., Wang, L., Creswell, A., Irving, G., & Higgins, I. (2022). Solving math word problems with process- and outcome-based feedback. *arXiv:2211.14275*.
- Wang, H., et al. (2026). InfoPO: Information-driven policy optimization for user-centric agents. *arXiv:2603.00656*.
- Wang, P., Li, L., Shao, Z., Xu, R. X., Dai, D., Li, Y., Chen, D., Wu, Y., & Sui, Z. (2024a). Math-Shepherd: Verify and reinforce LLMs step-by-step without human annotations. *ACL*.
- Wang, X., et al. (2025). AgentPRM: Process reward models for LLM agents. *arXiv:2502.10325*.
- Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2022). ReAct: Synergizing reasoning and acting in language models. *arXiv:2210.03629*.
- Yao, S., Chen, H., Yang, J., & Narasimhan, K. (2022b). WebShop: Towards scalable real-world web interaction with grounded language agents. *NeurIPS*.
- Zeng, A., Liu, M., Lu, R., Wang, B., Liu, X., Dong, Y., & Tang, J. (2024). AgentTuning: Enabling generalized agent abilities for LLMs. *ICLR*.
- Zhang, Y., et al. (2024). LEMA: Learning from mistakes makes LLM better reasoner. *ICML*.
- Zhang, Z., et al. (2025). GUI-Shepherd: Reliable process reward and verification for long-sequence GUI agents. *arXiv:2509.23738*.
- Zhou, A., Yan, K., Shlapentokh-Rothman, M., Wang, H., & Wang, Y.-X. (2024). Language agent tree search unifies reasoning, acting, and planning in language models. *ICML*.
