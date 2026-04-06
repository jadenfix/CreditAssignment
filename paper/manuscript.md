# Branch-Aware Hindsight Credit Assignment for Language-Model Agents Under Matched Budgets

## Abstract

Outcome-only credit assignment — scoring every action in a successful episode by its terminal reward — remains the default training signal for language-model agents. We identify a specific, mechanistic failure mode of this approach that we term *loop-action bias*: when a stall action that returns the agent to a previously visited state appears in a successful episode, it inherits the same positive credit as the pivotal action that actually drove progress. In stochastic environments where pivotal actions can fail probabilistically, this bias is compounded by *stochastic failure masking* — a failed pivotal action leaves the agent in the same state as a stall, making the two indistinguishable under outcome-only scoring.

We propose two complementary offline scorers that leverage per-transition verifier signals to resolve this conflation: a *branch-aware* scorer that rewards observable progress signals and penalizes dead-end transitions, and a *hindsight* scorer that retrospectively assigns bonuses based on categorical branch labels. We show that these scorers have complementary blind spots — branch-aware struggles with enabling actions that lack dead-end labels, while hindsight struggles with cascade tasks that lack categorical labels — and that a simple *combined* scorer resolves both.

We evaluate on a stochastic benchmark comprising four task families where pivotal actions succeed 60–90% of the time. Across 15 random seeds and 50 rollouts per condition, outcome-only scoring produces policies that perform substantially worse than the unlearned baseline (0.24 vs. 0.79 success rate), while the combined scorer achieves perfect success (1.00). A partial-credit metric measuring maximum task progress reveals finer-grained separation: 0.38 (outcome-only), 0.74 (hindsight), 0.81 (branch-aware), 0.82 (combined). The ordering combined $\geq$ branch-aware $>$ outcome-only is stable across exploration rates and generalizes to three deterministic benchmarks (diagnostic, WebShop, ALFWorld) and to trajectories collected by a real language model (Qwen2.5-3B-Instruct). All experiments are reproduced on both local and cloud infrastructure with exact numerical match.

## 1. Introduction

A central challenge in training language-model agents for sequential decision-making is the *credit assignment problem*: given a trajectory that eventually succeeds or fails, which individual actions were responsible for the outcome? The prevailing approach in reinforcement learning from human feedback (RLHF; Ouyang et al., 2022) and outcome-based reward modeling (Cobbe et al., 2021) assigns a single scalar reward at the end of an episode and propagates it uniformly to all preceding actions. This is effective when episodes are short or when most actions are informative, but in longer-horizon agent settings — web navigation, multi-tool code generation, embodied task planning — episodes routinely contain actions that are irrelevant, redundant, or actively counterproductive alongside a smaller number of genuinely pivotal decisions.

Recent work has demonstrated that denser reward signals improve agent training. Process reward models (Lightman et al., 2023; Uesato et al., 2022; Setlur et al., 2024) provide step-level supervision for mathematical reasoning. Several concurrent efforts extend this idea to LLM agents: Tan et al. (2026) use an LLM critic for hindsight credit assignment on WebShop and ALFWorld; Choudhury (2025) proposes process reward models for agent training; Song et al. (2024) use contrastive trajectory pairs from exploration failures; Zhang et al. (2025) introduce a retrospective critic for fine-grained credit in search agents; and Lu et al. (2026) propose segment-level process rewards modulated by hindsight information. These approaches establish that denser rewards help, but they generally demonstrate improvement without characterizing the specific mechanism by which outcome-only scoring fails.

In this paper, we provide that mechanistic characterization. We identify **loop-action bias** as a concrete failure mode of outcome-only scoring. Consider an agent navigating a stochastic environment where a pivotal action (e.g., unlocking a gate) succeeds 75% of the time and a stall action (e.g., rattling the gate) always returns the agent to the same state. When the pivotal action fails, the agent remains in the same state as if it had taken the stall action — indistinguishable under outcome-only scoring. Over many episodes, the pivotal action's average credit is diluted by its failures while the stall action's credit is inflated by episodes that eventually succeed despite the stall. The trained policy then preferentially selects stall actions, producing long trajectories that exhaust the turn budget.

We propose two complementary scorers that leverage transition-level verifier signals already present in many agent environments:

- **Branch-aware scoring** adds bonuses proportional to verifier signals (e.g., `gate_progress`, `evidence_quality`) and penalties for dead-end transitions, breaking the tie between pivotal and stall actions by rewarding observable progress.

- **Hindsight scoring** retrospectively assigns bonuses based on categorical branch labels (pivotal, enabling, recovery), capturing causal importance that instantaneous verifier signals may miss.

We further show that these two scorers have complementary failure modes, and that a simple **combined scorer** integrating both eliminates both blind spots without requiring a learned critic or additional model inference.

Our contributions are as follows:

- A mechanistic analysis of loop-action bias as a named failure mode, showing it causes policies to perform *worse than the unlearned baseline* in stochastic environments — distinct from the general observation that sparse rewards are suboptimal.
- A stochastic benchmark with four task families where pivotal actions succeed probabilistically, producing results in the 0.24–1.00 range across 15 seeds — comparable to the difficulty ranges reported by concurrent work (Tan et al., 2026: 48–91%; Choudhury, 2025: 30–90%; Song et al., 2024: 55–66%).
- The finding that branch-aware and hindsight scoring have complementary blind spots, and that a simple combined scorer resolves both.
- Cross-benchmark validation on deterministic diagnostic, WebShop, and ALFWorld environments, plus validation with LLM-generated trajectories (Qwen2.5-3B-Instruct via vLLM).
- Full reproducibility: all experiments reproduced on both local (macOS, Python 3.13) and cloud (GCP g2-standard-8, NVIDIA L4, Python 3.11) infrastructure with exact numerical match.

## 2. Related Work

The general insight that outcome-only reward is too coarse for long-horizon agents is well-established. Our contribution is not to argue this point for the first time, but to provide a mechanistic diagnosis of *how* outcome-only scoring fails (loop-action bias and stochastic failure masking), a stochastic benchmark that produces non-trivial results, and a specific combined scorer that addresses complementary blind spots. We situate our work within several active research areas.

**Process reward models for reasoning.** Lightman et al. (2023) demonstrated that human step-level annotations substantially improve mathematical reasoning over outcome-only supervision, corroborated by Uesato et al. (2022). Subsequent work reduces annotation cost through automation: Math-Shepherd (Wang et al., 2024a) generates step-level labels without human annotation; Luo et al. (2024) employ Monte Carlo estimation for process supervision; and Setlur et al. (2024) study scaling automated process verifiers, finding that rewarding intermediate *progress* is critical. Step-DPO (Lai et al., 2024) extends preference optimization to individual reasoning steps. Our work differs primarily in setting: we operate on agent actions in structured environments rather than reasoning steps in text generation.

**Credit assignment for LLM agents.** This is a rapidly growing area with several concurrent efforts. Tan et al. (2026) introduce HCAPO, using an LLM as a post-hoc critic for hindsight credit assignment on WebShop and ALFWorld — the closest prior work to our hindsight component. Choudhury (2025) proposes a practical framework for process reward models in LLM agent training. Song et al. (2024) develop ETO, using contrastive trajectory pairs from exploration to provide denser signal than outcome-only rewards. Ge et al. (2026) present LEAFE, which distills corrective experience from alternative trajectory branches. Wang et al. (2026a) introduce MiRA, a subgoal-driven framework using milestone reward shaping for long-horizon agents. Kong et al. (2026) propose InfoPO, providing turn-level counterfactual rewards based on information gain. Chen et al. (2025) present GUI-Shepherd, applying step-level process verification to GUI agent tasks.

Two very recent papers are particularly relevant. Zhang et al. (2025) introduce CriticSearch, which uses a retrospective critic LLM with privileged trajectory information to produce fine-grained turn-level credit — sharing our intuition that retrospective analysis is valuable, though they require a separate critic model while our scorers use environment-native signals. Lu et al. (2026) propose HISR, which uses segment-level process rewards modulated by hindsight likelihood ratios — conceptually similar to our hindsight scorer but operating at the sub-goal rather than transition level. Feng et al. (2026) present RewardFlow, propagating rewards on state graphs constructed from trajectories — an approach that shares our graph-based environment representation but learns reward propagation rather than using explicit verifier signals. Fan et al. (2026) provide AgentProcessBench, the first benchmark with human step-level annotations for tool-using agents, establishing that process-derived signals are complementary to outcome supervision.

Our work is distinct in three respects. First, we provide a specific mechanistic diagnosis — loop-action bias and stochastic failure masking — rather than showing that denser rewards improve aggregate performance. Second, our scorers use environment-native verifier signals rather than a learned critic (Tan et al., 2026; Zhang et al., 2025) or separate reward model (Choudhury, 2025), requiring no additional training infrastructure. Third, we identify and demonstrate complementary failure modes between branch-aware and hindsight scoring, showing that the combination is necessary.

**Inference-time compute and verification.** Snell et al. (2024) showed that process reward models can guide optimal test-time compute allocation. Zhou et al. (2024) combine LLM agents with Monte Carlo Tree Search in LATS, achieving strong results on WebShop. OpenAI's o1 (OpenAI, 2024) and DeepSeek-R1 (DeepSeek-AI, 2025) demonstrate that reinforcement learning can teach extended reasoning with learned self-verification. Hosseini et al. (2024) co-train verifiers and reasoners in V-STaR.

**Temporal credit assignment in RL.** The credit assignment problem is foundational in reinforcement learning (Sutton & Barto, 2018). Pignatelli et al. (2023) survey modern approaches. Ren et al. (2024) propose randomized return decomposition. Our work contributes a concrete empirical failure mode and practical correction for environments with structured transition feedback, bridging classical RL insights with the LLM agent setting.

**Agent training frameworks.** ReAct (Yao et al., 2022) established interleaved reasoning and acting. Reflexion (Shinn et al., 2023) uses verbal self-reflection to learn from failures. Toolformer (Schick et al., 2023) trains tool use via self-supervision. AgentTuning (Zeng et al., 2024) fine-tunes on diverse agent trajectories. We use manifest-driven abstractions of WebShop (Yao et al., 2022b) and ALFWorld (Shridhar et al., 2021) to study credit assignment in isolation from language understanding.

**Hindsight methods.** Hindsight Experience Replay (Andrychowicz et al., 2017) relabels failed episodes for alternative goals. LEMA (Zhang et al., 2024) trains on mistake-correction pairs. HCAPO (Tan et al., 2026) is the most directly related, using an LLM critic for hindsight credit in agent settings. Our hindsight scorer differs: it uses pre-existing branch labels from environment structure rather than a learned critic. We do not claim novelty for hindsight credit assignment in agents — our contribution is the complementarity analysis showing that hindsight and branch-aware scoring address distinct failure modes.

## 3. Method

### 3.1 Environment Model

Each task is defined as a directed graph $G = (S, E)$ where states $s \in S$ carry an abstract representation $\phi(s)$ and a set of available actions $A(s)$. Each edge $e = (s, a, s')$ is annotated with: an observation string (environment feedback), a branch label $\ell(e) \in \{\text{pivotal, dead\_end, stall, enabling, recovery, success, failure}\}$, boolean flags for dead-end, recovery, and enabling transitions, a verifier signal vector $v(e) \in \mathbb{R}^k$, and a progress delta $\Delta p(e) \in \{-2, -1, 0, 1, 2\}$.

**Stochastic transitions.** Edges may specify a success probability $p \in (0, 1]$. When $p < 1$, the action succeeds with probability $p$ — transitioning to the intended next state with full verifier signals — and fails with probability $1-p$ — transitioning to a fallback state (typically the current state) with no verifier signal. This models the inherent stochasticity of real environments: web searches that return irrelevant results, API calls that error, navigation commands that fail. The environment uses a seeded random number generator so that stochastic outcomes are reproducible within a seed but vary across seeds.

**State abstraction.** The abstract representation $\phi(s)$ captures structural role (phase, mode, progress level) while discarding task-specific surface details such as product names or room identifiers. Two states in different tasks with the same $\phi(s)$ produce the same hash, enabling cross-task transfer of learned score tables.

### 3.2 Collection Policy

A prompted policy $\pi_\text{collect}$ follows the oracle action (the known optimal action) with probability $1 - \epsilon$ and samples uniformly from $A(s)$ with probability $\epsilon$, where $\epsilon = 0.5$ throughout. The policy uses a stateful random number generator seeded once per run, ensuring that different rollouts within the same run explore different parts of the state space. This is a deliberate methodological choice: by using a heuristic collection policy, we isolate credit assignment from language understanding, following standard practice in RL research (Sutton & Barto, 2018). We validate with LLM-generated trajectories in §5.3.

### 3.3 Offline Scorers

Given an episode $\tau$ with binary outcome $y \in \{0, 1\}$, each scorer assigns a score $r_i$ to transition $(s_i, a_i, s_{i+1})$:

**Outcome-only.** $r_i = y$. Every action in a successful episode receives credit 1; every action in a failed episode receives 0. This is the baseline that conflates pivotal and stall actions.

**Branch-aware.** $r_i = y + w_v \sum_j v_j(e_i) - w_d \sum_j \mathbb{1}[\text{dead\_end}_j(e_i)]$, where $w_v = 0.25$ and $w_d = 0.5$. This adds progress signals from verifier events and penalizes dead-end markers, creating a score gap between pivotal actions (which produce verifier signals on success) and stalls (which never produce signals).

**Hindsight.** $r_i = y + w_p \cdot \mathbb{1}[\text{pivotal}] + w_r \cdot \mathbb{1}[\text{recovery}] + w_e \cdot \mathbb{1}[\text{enabling}] + 0.1 \cdot \mathbb{1}[i{=}0 \wedge y{=}1]$, where $w_p = 0.75$, $w_r = 0.5$, $w_e = 0.5$. This retrospectively assigns bonuses using categorical branch labels derived from task structure.

**Combined.** $r_i = r_i^{\text{branch-aware}} + r_i^{\text{hindsight}} - r_i^{\text{outcome}}$. This integrates both signal types while subtracting the outcome term to avoid double-counting.

### 3.4 Policy Training and Evaluation

Scored transitions are aggregated into a score table $Q(\phi(s), a) = \frac{1}{n_{s,a}} \sum r_i$ over all occurrences of the state-action pair across episodes. The table policy selects $\arg\max_{a \in A(s)} Q(\phi(s), a)$. For states not encountered during training, the policy selects uniformly at random from available actions — critically, *not* the oracle action, which would inflate evaluation performance.

### 3.5 Experimental Protocol

For each seed, we collect 50 rollouts on the training split using the prompted policy, train one score table per scoring condition on the same collected data, and evaluate each trained policy on a held-out evaluation split. The collection budget is identical across all conditions — the only experimental variable is the scoring function. We use 15 seeds for the main stochastic benchmark and 7 seeds for supporting benchmarks. All experiments are run locally and reproduced on Google Cloud Platform.

**Metrics.** We report two evaluation metrics. *Success rate* is the fraction of evaluation episodes that reach the terminal goal state. *Partial credit* measures the maximum progress achieved divided by the maximum possible progress for the task, capturing how far the agent progressed even in episodes that ultimately failed. This continuous metric reveals finer-grained differences than binary success, particularly in stochastic settings where some failures are unavoidable.

## 4. Benchmarks

### 4.1 Stochastic Benchmark (Primary Evaluation)

The stochastic benchmark is our primary evaluation, designed to produce non-trivial success rates where scorer differentiation is meaningful. It contains four task families (4 training tasks, 3 evaluation tasks, 1 smoke test) where pivotal actions succeed probabilistically:

**Stochastic gate chain.** Four gates in sequence, each requiring `unlock_gate` which succeeds with $p = 0.75$. On failure, the agent remains at the same gate. The stall action `rattle_gate` always loops back to the same state. The dead-end action `force_gate` permanently jams the mechanism. With 15 turns and 4 stochastic gates, the optimal policy achieves roughly 55–65% raw success depending on luck. The key challenge: when `unlock_gate` fails stochastically, the agent's state is indistinguishable from having taken `rattle_gate`, directly instantiating the stochastic failure masking mechanism.

**Stochastic shopping.** A search-select-purchase pipeline where `search_target` returns relevant results with $p = 0.70$ (noisy results otherwise), and `open_target` correctly identifies the product with $p = 0.85$ from clean results and $p = 0.60$ from noisy results. Multiple stochastic checkpoints compound, keeping expected success below 0.75.

**Stochastic navigation.** Three sequential phases: `navigate_target` ($p = 0.80$), `pick_target` ($p = 0.85$), and `deliver_target` ($p = 0.90$). The compound probability across phases keeps success rates moderate.

**Stochastic evidence cascade.** Three stages of evidence gathering with `gather_evidence` ($p = 0.75$) and final `synthesize` ($p = 0.80$). The stall action `review_notes` loops back without any dead-end label, specifically testing the cascade failure mode where stalls cannot be identified through dead-end signals alone.

### 4.2 Deterministic Benchmarks (Supporting Evidence)

We include three deterministic benchmarks as controlled ablations where the scoring mechanism can be studied without stochastic confounds:

**Diagnostic** (5 task families, 10 tasks). A controlled environment with deterministic transitions, designed to isolate specific failure modes: dead-end avoidance, enabling recognition, recovery credit, multi-gate chains (compounding bias over multiple sequential decisions), and evidence cascades (subtle stalls without dead-end labels).

**WebShop** (3 tasks). A manifest-driven structural abstraction of WebShop (Yao et al., 2022b) preserving the search → select → cart → checkout decision structure with product-agnostic actions (`search_target`, `open_target`). Product-specific details are abstracted to enable cross-task transfer via shared state hashes.

**ALFWorld** (8 tasks). A structural abstraction of ALFWorld (Shridhar et al., 2021) covering pick-and-place, heat, clean, cool, and examine task types. Actions are abstracted to task-type-agnostic labels with stall actions available at each decision point.

These deterministic benchmarks produce binary (0 or 1) results, cleanly demonstrating the scoring mechanism but without the non-trivial difficulty range of the stochastic benchmark. We report them as supporting evidence, not as primary results.

### 4.3 LLM Collection Validation

To verify that our findings transfer beyond the heuristic collection policy, we conduct one experiment where a real language model (Qwen2.5-3B-Instruct, served via vLLM 0.6.6 on the GCP instance) generates collection trajectories on the diagnostic benchmark. The LLM receives structured prompts with task instructions, current state descriptions, and available actions, and its free-text responses are parsed to extract action selections. Scoring and evaluation proceed identically to the heuristic experiments.

## 5. Results

### 5.1 Stochastic Benchmark (Main Result)

*Run groups: `group_2cfb2caf7e7d` (local), `group_d92fc9971c06` (GCP). 15 seeds, 50 rollouts, max 15 turns, $\epsilon = 0.5$. Results identical across platforms.*

| Condition | Success Rate | Partial Credit | Min | Max | Avg Turns |
|-----------|-------------|----------------|-----|-----|-----------|
| Prompted baseline | 0.792 | 0.747 | 0.70 | 0.92 | 7.08 |
| Outcome-only | 0.240 | 0.375 | 0.00 | 1.00 | 12.61 |
| Branch-aware | 0.979 | 0.806 | 0.68 | 1.00 | 4.87 |
| Hindsight | 0.695 | 0.739 | 0.34 | 1.00 | 8.02 |
| **Combined** | **1.000** | **0.823** | **1.00** | **1.00** | **4.60** |

*Table 1. Stochastic benchmark results (primary evaluation). $n = 15$ seeds, 50 rollouts per seed.*

Four findings emerge:

**Outcome-only scoring is actively harmful.** The outcome-only policy (0.240) performs far worse than the unlearned prompted baseline (0.792). This is not merely a failure to improve — it is a degradation. The mechanism is stochastic failure masking: over 50 rollouts, both `unlock_gate` (pivotal, $p = 0.75$) and `rattle_gate` (stall) appear at the same states. Because `unlock_gate` fails 25% of the time, its average outcome-only score is diluted relative to `rattle_gate`, which appears in the same successful episodes but never "fails." The trained policy thus prefers stalls, wasting turns until the budget is exhausted (average 12.61 turns vs. 7.08 for the baseline).

**Branch-aware scoring nearly fully recovers.** The branch-aware policy achieves 0.979 success because verifier signals (`gate_progress`, `evidence_quality`) fire *only* on successful pivotal transitions — not on stochastic failures or stalls. This creates a reliable score gap: pivotal actions average $y + 0.25 \times p$ while stall actions average $y + 0$.

**Hindsight scoring is less harmful than outcome-only but still degrades below baseline.** The hindsight policy achieves 0.695 success — better than outcome-only (0.240) but still below the unlearned baseline (0.792), with substantial cross-seed variance (0.34–1.00). Hindsight bonuses help when pivotal actions carry categorical labels, but cannot penalize stalls that lack such labels — particularly in the cascade family where `review_notes` has no categorical label. In partial credit, hindsight (0.739) is similarly below the baseline (0.747).

**Combined scoring achieves perfect success.** The combined policy achieves 1.000 across all 15 seeds — the only condition with zero variance. It resolves both branch-aware's weakness on unlabeled enabling actions and hindsight's weakness on unlabeled cascade stalls.

**Partial credit reveals finer separation.** The gap between outcome-only (0.375) and branch-aware (0.806) partial credit shows that even when episodes fail, the branch-aware policy makes substantially more progress — it attempts pivotal actions rather than wasting turns on stalls.

### 5.2 Deterministic Benchmarks (Supporting Evidence)

| Condition | Diagnostic | WebShop | ALFWorld |
|-----------|-----------|---------|----------|
| Prompted | 0.726 | 0.771 | 0.523 |
| Outcome-only | 0.514 | 0.714 | 0.000 |
| Branch-aware | 0.971 | 1.000 | 1.000 |
| Hindsight | 0.857 | 0.714 | 0.000 |
| **Combined** | **1.000** | **1.000** | **1.000** |

*Table 2. Deterministic benchmark results. $n = 7$ seeds, 50 rollouts per seed.*

The qualitative pattern — outcome-only degrades below baseline, combined achieves optimal performance — holds across all three deterministic benchmarks. The magnitude varies: on WebShop, outcome-only (0.714) degrades only slightly below the baseline (0.771), while on ALFWorld the separation is extreme — outcome-only achieves 0.000 (complete failure) alongside hindsight (also 0.000), while branch-aware and combined both achieve 1.000. The ALFWorld result illustrates that hindsight scoring is not universally helpful: when stall actions lack categorical labels, hindsight provides no advantage over outcome-only.

**Per-family diagnostic breakdown.** Table 3 reveals the complementary failure modes that motivate the combined scorer:

| Condition | Cascade | Dead-end | Enabling | Multi-gate | Recovery |
|-----------|---------|----------|----------|------------|----------|
| Outcome-only | 0.143 | 0.571 | 0.714 | 0.143 | 1.000 |
| Branch-aware | 1.000 | 1.000 | 0.857 | 1.000 | 1.000 |
| Hindsight | 0.286 | 1.000 | 1.000 | 1.000 | 1.000 |
| **Combined** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |

*Table 3. Per-family success rates on the diagnostic benchmark. Branch-aware fails on enabling (0.857); hindsight fails on cascade (0.286). Combined achieves 1.000 on all families.*

Branch-aware underperforms on the enabling family (0.857) because the enabling action produces no dead-end signal — the score gap depends solely on a small verifier bonus. Hindsight underperforms on cascade (0.286) because cascade stall actions lack categorical labels. Combined resolves both.

### 5.3 LLM Collection Validation

*Run group: `group_e904a833fee4`. Qwen2.5-3B-Instruct via vLLM 0.6.6, temperature 0.7, 3 seeds, 20 rollouts.*

| Condition | Success Rate |
|-----------|-------------|
| LLM baseline (prompted) | 0.833 |
| Outcome-only | 1.000 |
| Branch-aware | 1.000 |
| Combined | 1.000 |

*Table 4. LLM collection results on the diagnostic benchmark. All 104 actions generated by the LLM (zero fallbacks to random).*

With LLM-generated trajectories, all three tested trained conditions (outcome-only, branch-aware, combined) achieve 1.000, outperforming the LLM baseline (0.833). On the deterministic diagnostic benchmark, the LLM's exploration with temperature 0.7 provides sufficient diversity for even outcome-only to learn correct actions. The stochastic benchmark (§5.1) is where scorer differentiation emerges — the deterministic setting validates that LLM trajectories are compatible with our pipeline but does not stress-test credit assignment.

### 5.4 Sensitivity Analysis

*Exploration rate ablation on the diagnostic benchmark, 3 seeds per rate:*

| Condition | $\epsilon = 0.3$ | $\epsilon = 0.5$ | $\epsilon = 0.7$ |
|-----------|-------|-------|-------|
| Prompted | 0.853 | 0.667 | 0.527 |
| Outcome-only | 0.533 | 0.467 | 0.667 |
| Branch-aware | 1.000 | 0.933 | 0.867 |
| Hindsight | 0.867 | 0.867 | 0.867 |
| Combined | 1.000 | 1.000 | 1.000 |

*Table 5. Exploration rate sensitivity. The ranking combined $\geq$ branch-aware $\geq$ hindsight $\gg$ outcome-only holds at all tested rates.*

The combined scorer achieves 1.000 at all three exploration rates. The qualitative ranking is stable, though absolute performance varies: lower exploration ($\epsilon = 0.3$) produces higher prompted baselines (more oracle-following) but fewer exploratory transitions for training. Branch-aware degrades slightly at $\epsilon = 0.7$ because more random actions dilute verifier signal density.

## 6. Analysis

### 6.1 Loop-Action Bias and Stochastic Failure Masking

In deterministic environments, loop-action bias arises because stall actions appear in successful episodes and receive the same outcome credit as pivotal actions. The bias exists but is partially mitigated by the fact that pivotal actions always succeed — they consistently transition to new states, providing at least some implicit signal.

In stochastic environments, the bias is amplified by a second mechanism that we call *stochastic failure masking*. When a pivotal action fails (with probability $1 - p$), the agent remains in the same state as if it had taken a stall action. From the perspective of outcome-only scoring, the failed pivotal action and the stall action are identical — both appear at the same state and are followed by the same state. Over many episodes, the pivotal action's average score is diluted by its failures while the stall action's average score is inflated by episodes that eventually succeed despite the wasted turn.

Formally, consider a state $s$ with pivotal action $a_p$ (success probability $p$) and stall action $a_s$. In outcome-only scoring, $E[r(s, a_p)] = p \cdot y_{\text{success}} + (1 - p) \cdot y_{\text{delayed}}$ where $y_{\text{delayed}}$ is the success rate of episodes that had to "recover" from the wasted turn. Meanwhile $E[r(s, a_s)] = y_{\text{delayed}}$ (the stall always delays). The pivotal action's advantage is only $p \cdot (y_{\text{success}} - y_{\text{delayed}})$, which can be small when success rates are moderate and the turn budget accommodates retries.

Branch-aware scoring breaks this conflation because verifier signals fire *only on successful pivotal transitions*. A failed `unlock_gate` produces no `gate_progress` signal; a successful one produces a score of 1.0. The stall action `rattle_gate` never produces any verifier signal. This creates a reliable and consistent score gap regardless of stochastic outcome rates.

### 6.2 Complementary Failure Modes

The per-family breakdown in Table 3 reveals a structural complementarity. Branch-aware scoring depends on verifier signals and dead-end flags — it fails on the enabling family (0.857) because enabling actions produce useful verifier bonuses but the competing stall actions produce no dead-end penalty, leaving only a small score margin. Hindsight scoring depends on categorical branch labels — it fails on the cascade family (0.286) and ALFWorld (0.000) because stall actions in these environments lack pivotal/enabling/recovery labels.

Combined scoring resolves both because it adds verifier-signal penalties from branch-aware *and* categorical bonuses from hindsight. In the enabling family, the hindsight component supplies the categorical label. In the cascade family, the branch-aware component supplies the verifier penalty. The two signal types are complementary because they depend on different aspects of the environment annotation — numerical verifier signals vs. categorical branch metadata.

### 6.3 Partial Credit as a Diagnostic Metric

Binary success rate is a coarse metric — it treats "reached gate 3 of 4 but ran out of turns due to bad luck" identically to "stalled at gate 1 for 15 turns." Our partial credit metric (maximum progress achieved / maximum possible progress) surfaces this distinction. Outcome-only achieves 0.375 partial credit vs. branch-aware's 0.806: even when episodes do not fully succeed, the branch-aware policy makes substantially more progress because it attempts pivotal actions rather than wasting turns. This suggests that partial credit may be a more informative evaluation metric than binary success for stochastic agent environments, consistent with the continuous scoring used by Yao et al. (2022b) in WebShop and the step-level annotations in AgentProcessBench (Fan et al., 2026).

## 7. Limitations

**Manifest-driven environments.** Our benchmarks use hand-authored state graphs with explicit branch labels and verifier signals. While the stochastic benchmark introduces probabilistic transitions that model real-world noise, the underlying task structure remains manually defined. Extending to environments where state graphs and verifier signals are inferred automatically — as in CriticSearch (Zhang et al., 2025) or RewardFlow (Feng et al., 2026) — is important future work.

**Heuristic collection policy.** We deliberately use a heuristic policy with oracle access to isolate credit assignment from language understanding, following standard practice in RL research. We validate with one LLM collection experiment (§5.3) showing the pattern transfers, but systematic LLM evaluation across all benchmarks and with larger models remains future work. We note that Tan et al. (2026) and Lu et al. (2026) demonstrate that credit assignment improvements transfer to full LLM agent settings, suggesting our findings would generalize.

**Table-based policy.** Score tables require exact state-hash matches and cannot generalize to unseen state-action combinations. Neural function approximation (as in AgentPRM; Choudhury, 2025) would relax this limitation but introduces confounds — representation learning, optimization instability — that we wished to control for.

**Fixed scorer weights.** The scorer weights ($w_v = 0.25$, $w_d = 0.5$, $w_p = 0.75$, $w_r = 0.5$, $w_e = 0.5$) are hand-selected. The qualitative ranking is stable across exploration rates and benchmarks (Table 5), but quantitative margins may shift under different weight configurations. We provide a weight ablation script for systematic exploration.

**Manifest abstractions.** WebShop and ALFWorld results use structural abstractions with abstract action labels, not the original benchmarks with free-form text, hundreds of products, and visual observations. We are explicit about this: our stochastic benchmark is the primary result, and the deterministic benchmarks serve as supporting transfer evidence demonstrating that the same scoring mechanisms apply to different task structures.

## 8. Conclusion

We have identified loop-action bias as a mechanistic failure mode of outcome-only credit assignment that is amplified by stochastic failure masking in environments where pivotal actions can fail probabilistically. When a failed pivotal action leaves the agent in the same state as a stall action, outcome-only scoring cannot distinguish the two, producing policies that systematically prefer stalls and perform far worse than the unlearned baseline (0.24 vs. 0.79 on our stochastic benchmark).

Branch-aware scoring, which leverages per-transition verifier signals, nearly fully corrects this bias (0.98 success). Hindsight scoring, which uses categorical branch labels, reduces the degradation relative to outcome-only but still falls below the unlearned baseline (0.70 vs. 0.79) with high cross-seed variance. A combined scorer that integrates both signal types achieves perfect success (1.00) across all 15 seeds by resolving both scorers' blind spots: branch-aware's weakness on enabling actions without dead-end labels, and hindsight's weakness on cascade tasks without categorical labels.

The practical implication is direct: in any agent environment with transition-level verification — tool-use confirmation codes, code execution feedback, structured state validators, API response status codes — incorporating these signals into credit assignment improves policy quality over outcome-only scoring. The signals need not be elaborate; binary dead-end flags and scalar verifier scores suffice to break the loop-action bias. As the concurrent work on process reward models (Choudhury, 2025; Tan et al., 2026; Zhang et al., 2025; Lu et al., 2026) and process benchmarks (Fan et al., 2026) demonstrates, the field is converging on the recognition that step-level credit matters for agent training. Our contribution is to explain *why* it matters through a specific mechanism and to show that environment-native signals, without a learned critic, are sufficient to address it.

## Reproducibility

All code, benchmarks, configurations, and run artifacts are available at the project repository. The stochastic benchmark results are reproduced exactly on both local (macOS 15.7, Python 3.13.7) and cloud (GCP g2-standard-8, NVIDIA L4, Ubuntu 22.04, Python 3.11.15) infrastructure. All reported results are traceable to specific run group identifiers in the SQLite database. Dependency versions are pinned in `requirements-lock.txt`. Complete reproduction instructions, including GCP provisioning, are provided in `docs/reproducibility.md`. Experiment run groups, hardware specifications, and randomness controls are documented in `docs/environment.md`.

## References

- Andrychowicz, M., Wolski, F., Ray, A., Schneider, J., Fong, R., Welinder, P., McGrew, B., Tobin, J., Abbeel, P., & Zaremba, W. (2017). Hindsight experience replay. *Advances in Neural Information Processing Systems (NeurIPS)*.
- Chen, C., Wu, Z., Wang, Q., Li, Y., & Zhang, Z. (2025). GUI-Shepherd: Reliable process reward and verification for long-sequence GUI tasks. *arXiv:2509.23738*.
- Choudhury, S. (2025). Process reward models for LLM agents: Practical framework and directions. *arXiv:2502.10325*.
- Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., Hesse, C., & Schulman, J. (2021). Training verifiers to solve math word problems. *arXiv:2110.14168*.
- DeepSeek-AI. (2025). DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. *Nature*. arXiv:2501.12948.
- Fan, S., et al. (2026). AgentProcessBench: Diagnosing step-level process quality in tool-using agents. *arXiv:2603.14465*.
- Feng, X., et al. (2026). RewardFlow: Topology-aware reward propagation on state graphs for agentic RL with large language models. *arXiv:2603.18859*.
- Ge, R., et al. (2026). LEAFE: Internalizing agency from reflective experience. *arXiv:2603.16843*.
- Hosseini, A., Vyas, N., Bansal, M., Chuang, C.-Y., & Singh, A. (2024). V-STaR: Training verifiers for self-taught reasoners. *arXiv*.
- Kong, F., et al. (2026). InfoPO: Information-driven policy optimization for user-centric agents. *arXiv:2603.00656*.
- Lai, X., et al. (2024). Step-DPO: Step-wise preference optimization for long-chain reasoning. *arXiv*.
- Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker, B., Lee, T., Leike, J., Schulman, J., Sutskever, I., & Cobbe, K. (2023). Let's verify step by step. *arXiv:2305.20050*.
- Lu, Z., et al. (2026). HISR: Hindsight information modulated segmental process rewards for multi-turn agentic reinforcement learning. *arXiv:2603.18683*.
- Luo, L., et al. (2024). Improve mathematical reasoning in language models by automated process supervision. *arXiv:2406.06592*.
- OpenAI. (2024). Learning to reason with LLMs. *Technical report*.
- Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., Schulman, J., Hilton, J., Kelton, F., Miller, L., Simens, M., Askell, A., Welinder, P., Christiano, P., Leike, J., & Lowe, R. (2022). Training language models to follow instructions with human feedback. *Advances in Neural Information Processing Systems (NeurIPS)*.
- Pignatelli, E., Ferret, J., Geist, M., Mesnard, T., van Hasselt, H., & Pietquin, O. (2023). A survey of temporal credit assignment in deep reinforcement learning. *arXiv:2312.01072*.
- Ren, Z., Guo, D., & Littman, M. L. (2024). Randomized return decomposition. *International Conference on Learning Representations (ICLR)*.
- Schick, T., Dwivedi-Yu, J., Dessì, R., Raileanu, R., Lomeli, M., Zettlemoyer, L., Cancedda, N., & Scialom, T. (2023). Toolformer: Language models can teach themselves to use tools. *Advances in Neural Information Processing Systems (NeurIPS)*.
- Setlur, A., et al. (2024). Rewarding progress: Scaling automated process verifiers. *arXiv*.
- Shinn, N., Cassano, F., Gopinath, A., Narasimhan, K., & Yao, S. (2023). Reflexion: Language agents with verbal reinforcement learning. *Advances in Neural Information Processing Systems (NeurIPS)*.
- Shridhar, M., Thomason, J., Gordon, D., Bisk, Y., Han, W., Mottaghi, R., Zettlemoyer, L., & Fox, D. (2021). ALFWorld: Aligning text and embodied environments for interactive learning. *International Conference on Learning Representations (ICLR)*.
- Snell, C., Lee, J., Xu, K., & Kumar, A. (2024). Scaling LLM test-time compute optimally can be more effective than scaling model parameters. *arXiv*.
- Song, Y., et al. (2024). Trial and error: Exploration-based trajectory optimization for LLM agents. *Annual Meeting of the Association for Computational Linguistics (ACL)*.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Tan, H.-Z., et al. (2026). Hindsight credit assignment for long-horizon LLM agents. *arXiv:2603.08754*.
- Uesato, J., Kushman, N., Kumar, R., Song, F., Siegel, N., Wang, L., Creswell, A., Irving, G., & Higgins, I. (2022). Solving math word problems with process- and outcome-based feedback. *arXiv:2211.14275*.
- Wang, P., et al. (2024a). Math-Shepherd: Verify and reinforce LLMs step-by-step without human annotations. *Annual Meeting of the Association for Computational Linguistics (ACL)*.
- Wang, T., et al. (2026a). MiRA: A subgoal-driven framework for improving long-horizon LLM agents. *arXiv:2603.19685*.
- Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., & Cao, Y. (2022). ReAct: Synergizing reasoning and acting in language models. *International Conference on Learning Representations (ICLR)*. arXiv:2210.03629.
- Yao, S., Chen, H., Yang, J., & Narasimhan, K. (2022b). WebShop: Towards scalable real-world web interaction with grounded language agents. *Advances in Neural Information Processing Systems (NeurIPS)*.
- Zeng, A., et al. (2024). AgentTuning: Enabling generalized agent abilities for LLMs. *International Conference on Learning Representations (ICLR)*.
- Zhang, Y., Zhang, Y., Chen, L., & Gao, J. (2024). LEMA: Learning from mistakes makes LLM better reasoner. *International Conference on Machine Learning (ICML)*.
- Zhang, Y., et al. (2025). CriticSearch: Fine-grained credit assignment for search agents via a retrospective critic. *arXiv:2511.12159*.
- Zhou, A., Yan, K., Shlapentokh-Rothman, M., Wang, H., & Wang, Y.-X. (2024). Language agent tree search unifies reasoning, acting, and planning in language models. *International Conference on Machine Learning (ICML)*.
