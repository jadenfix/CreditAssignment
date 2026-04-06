# Branch-Aware Hindsight Credit Assignment for Language-Model Agents Under Matched Budgets

## Abstract

Outcome-only credit assignment — scoring every action in a successful episode by its terminal reward — remains the default training signal for language-model agents. We identify a specific, mechanistic failure mode of this approach that we term *loop-action bias*: when a stall action that returns the agent to a previously visited state appears in a successful episode, it inherits the same positive credit as the pivotal action that actually drove progress. In stochastic environments where pivotal actions can fail probabilistically, this bias is compounded by *stochastic failure masking* — a failed pivotal action leaves the agent in the same state as a stall, making the two indistinguishable under outcome-only scoring.

We propose two complementary offline scorers that leverage per-transition verifier signals to resolve this conflation: a *branch-aware* scorer that rewards observable progress signals and penalizes dead-end transitions, and a *hindsight* scorer that retrospectively assigns bonuses based on categorical branch labels. We show that these scorers have complementary blind spots and that a simple *combined* scorer resolves both.

We evaluate on a stochastic benchmark where pivotal actions succeed 60–90% of the time (15 seeds, 50 rollouts), finding that outcome-only scoring produces policies far worse than the unlearned baseline (0.24 vs. 0.79), while the combined scorer achieves perfect success (1.00). We then measure training signal quality on the real ALFWorld benchmark (Shridhar et al., 2021) with two language models — Claude Haiku 4.5 (30.0% $\pm$ 2.3% success) and Qwen2.5-7B-Instruct (16.7% $\pm$ 3.5% success) — using automatically extracted verifier signals from observation text diffs. The branch-aware scorer differentiates 2.6–9.4x more states than outcome-only, with the advantage largest for the weaker model where outcome-only has almost no signal. All controlled experiments are reproduced on both local and cloud infrastructure.

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
- Validation on the real ALFWorld benchmark (Shridhar et al., 2021) with two LLMs of different capability levels, using automatically extracted verifier signals from observation text diffs — showing that branch-aware scoring produces substantially more informative training signal than outcome-only, with the advantage being *larger* for weaker models.
- The finding that branch-aware and hindsight scoring address structurally distinct failure modes: branch-aware uses numerical verifier signals, hindsight uses categorical labels. On controlled benchmarks where both signal types are available, a combined scorer resolves both; on real environments where only numerical signals are extractable, branch-aware alone provides the benefit.
- Full reproducibility: all experiments reproduced on both local (macOS, Python 3.13) and cloud (GCP g2-standard-8, NVIDIA L4, Python 3.11) infrastructure.

## 2. Related Work

The general insight that outcome-only reward is too coarse for long-horizon agents is well-established. Our contribution is not to argue this point for the first time, but to provide a mechanistic diagnosis of *how* outcome-only scoring fails (loop-action bias and stochastic failure masking), a stochastic benchmark that produces non-trivial results, automatic verifier signal extraction from environment observations, and a specific combined scorer that addresses complementary blind spots. We situate our work within several active research areas.

**Process reward models for reasoning.** Lightman et al. (2023) demonstrated that human step-level annotations substantially improve mathematical reasoning over outcome-only supervision, corroborated by Uesato et al. (2022). Subsequent work reduces annotation cost through automation: Math-Shepherd (Wang et al., 2024a) generates step-level labels without human annotation; Luo et al. (2024) employ Monte Carlo estimation for process supervision; and Setlur et al. (2024) study scaling automated process verifiers, finding that rewarding intermediate *progress* is critical. Step-DPO (Lai et al., 2024) extends preference optimization to individual reasoning steps. Our work differs primarily in setting: we operate on agent actions in structured environments rather than reasoning steps in text generation.

**Credit assignment for LLM agents.** This is a rapidly growing area with several concurrent efforts. Tan et al. (2026) introduce HCAPO, using an LLM as a post-hoc critic for hindsight credit assignment on WebShop and ALFWorld — the closest prior work to our hindsight component. Choudhury (2025) proposes a practical framework for process reward models in LLM agent training. Song et al. (2024) develop ETO, using contrastive trajectory pairs from exploration to provide denser signal than outcome-only rewards. Ge et al. (2026) present LEAFE, which distills corrective experience from alternative trajectory branches. Wang et al. (2026a) introduce MiRA, a subgoal-driven framework using milestone reward shaping for long-horizon agents. Kong et al. (2026) propose InfoPO, providing turn-level counterfactual rewards based on information gain. Chen et al. (2025) present GUI-Shepherd, applying step-level process verification to GUI agent tasks.

Two very recent papers are particularly relevant. Zhang et al. (2025) introduce CriticSearch, which uses a retrospective critic LLM with privileged trajectory information to produce fine-grained turn-level credit — sharing our intuition that retrospective analysis is valuable, though they require a separate critic model while our scorers use environment-native signals. Lu et al. (2026) propose HISR, which uses segment-level process rewards modulated by hindsight likelihood ratios — conceptually similar to our hindsight scorer but operating at the sub-goal rather than transition level. Feng et al. (2026) present RewardFlow, propagating rewards on state graphs constructed from trajectories — an approach that shares our graph-based environment representation but learns reward propagation rather than using explicit verifier signals. Fan et al. (2026) provide AgentProcessBench, the first benchmark with human step-level annotations for tool-using agents, establishing that process-derived signals are complementary to outcome supervision.

Our work is distinct in three respects. First, we provide a specific mechanistic diagnosis — loop-action bias and stochastic failure masking — rather than showing that denser rewards improve aggregate performance. Second, our scorers use environment-native verifier signals rather than a learned critic (Tan et al., 2026; Zhang et al., 2025) or separate reward model (Choudhury, 2025), and we show these signals can be automatically extracted from observation text diffs without any annotation. Third, we identify and demonstrate complementary failure modes between branch-aware and hindsight scoring, showing that the combination is necessary.

**Inference-time compute and verification.** Snell et al. (2024) showed that process reward models can guide optimal test-time compute allocation. Zhou et al. (2024) combine LLM agents with Monte Carlo Tree Search in LATS, achieving strong results on WebShop. OpenAI's o1 (OpenAI, 2024) and DeepSeek-R1 (DeepSeek-AI, 2025) demonstrate that reinforcement learning can teach extended reasoning with learned self-verification. Hosseini et al. (2024) co-train verifiers and reasoners in V-STaR.

**Temporal credit assignment in RL.** The credit assignment problem is foundational in reinforcement learning (Sutton & Barto, 2018). Pignatelli et al. (2023) survey modern approaches. Ren et al. (2024) propose randomized return decomposition. Our work contributes a concrete empirical failure mode and practical correction for environments with structured transition feedback, bridging classical RL insights with the LLM agent setting.

**Agent training frameworks.** ReAct (Yao et al., 2022) established interleaved reasoning and acting. Reflexion (Shinn et al., 2023) uses verbal self-reflection to learn from failures. Toolformer (Schick et al., 2023) trains tool use via self-supervision. AgentTuning (Zeng et al., 2024) fine-tunes on diverse agent trajectories. We validate on the actual ALFWorld environment (Shridhar et al., 2021) and additionally use manifest-driven abstractions of WebShop (Yao et al., 2022b) for controlled ablation.

**Hindsight methods.** Hindsight Experience Replay (Andrychowicz et al., 2017) relabels failed episodes for alternative goals. LEMA (Zhang et al., 2024) trains on mistake-correction pairs. HCAPO (Tan et al., 2026) is the most directly related, using an LLM critic for hindsight credit in agent settings. Our hindsight scorer differs: it uses pre-existing branch labels from environment structure rather than a learned critic. We do not claim novelty for hindsight credit assignment in agents — our contribution is the complementarity analysis showing that hindsight and branch-aware scoring address distinct failure modes.

## 3. Method

### 3.1 Environment Model

Each task is defined as a directed graph $G = (S, E)$ where states $s \in S$ carry an abstract representation $\phi(s)$ and a set of available actions $A(s)$. Each edge $e = (s, a, s')$ is annotated with: an observation string (environment feedback), a branch label $\ell(e) \in \{\text{pivotal, dead\_end, stall, enabling, recovery, success, failure}\}$, boolean flags for dead-end, recovery, and enabling transitions, a verifier signal vector $v(e) \in \mathbb{R}^k$, and a progress delta $\Delta p(e) \in \{-2, -1, 0, 1, 2\}$.

**Stochastic transitions.** Edges may specify a success probability $p \in (0, 1]$. When $p < 1$, the action succeeds with probability $p$ — transitioning to the intended next state with full verifier signals — and fails with probability $1-p$ — transitioning to a fallback state (typically the current state) with no verifier signal. This models the inherent stochasticity of real environments: web searches that return irrelevant results, API calls that error, navigation commands that fail. The environment uses a seeded random number generator so that stochastic outcomes are reproducible within a seed but vary across seeds.

**State abstraction.** The abstract representation $\phi(s)$ captures structural role (phase, mode, progress level) while discarding task-specific surface details such as product names or room identifiers. Two states in different tasks with the same $\phi(s)$ produce the same hash, enabling cross-task transfer of learned score tables.

### 3.2 Automatic Verifier Signal Extraction

For real environments where hand-authored branch labels are unavailable, we extract verifier signals automatically from observation text:

1. **State-diff detection.** We compare consecutive observations using text similarity (SequenceMatcher). High similarity ($> 0.9$) combined with error patterns ("Nothing happens," "You can't") emits a dead-end signal ($-1.0$).
2. **Progress heuristics.** Room navigation (detected via "You arrive at" patterns) emits a progress signal ($+0.5$). Inventory changes (pick up or put down objects) emit a stronger progress signal ($+1.0$).
3. **Repetition detection.** If the same observation appears three or more times within an episode, we emit a repetition penalty ($-0.5$). This directly targets loop-action bias by penalizing the stall pattern.

These heuristics require no privileged environment information — only the observation text that the agent itself receives. We show in §5.4 that even these noisy signals are sufficient to break loop-action bias on a real benchmark.

### 3.3 Collection Policy

A prompted policy $\pi_\text{collect}$ follows the oracle action (the known optimal action) with probability $1 - \epsilon$ and samples uniformly from $A(s)$ with probability $\epsilon$, where $\epsilon = 0.5$ throughout. For real-benchmark experiments (§5.4), collection is performed by the LLM itself, with no oracle access.

### 3.4 Offline Scorers

Given an episode $\tau$ with binary outcome $y \in \{0, 1\}$, each scorer assigns a score $r_i$ to transition $(s_i, a_i, s_{i+1})$:

**Outcome-only.** $r_i = y$. Every action in a successful episode receives credit 1; every action in a failed episode receives 0. This is the baseline that conflates pivotal and stall actions.

**Branch-aware.** $r_i = y + w_v \sum_j v_j(e_i) - w_d \sum_j \mathbb{1}[\text{dead\_end}_j(e_i)]$, where $w_v = 0.25$ and $w_d = 0.5$. This adds progress signals from verifier events and penalizes dead-end markers, creating a score gap between pivotal actions (which produce verifier signals on success) and stalls (which never produce signals).

**Hindsight.** $r_i = y + w_p \cdot \mathbb{1}[\text{pivotal}] + w_r \cdot \mathbb{1}[\text{recovery}] + w_e \cdot \mathbb{1}[\text{enabling}] + 0.1 \cdot \mathbb{1}[i{=}0 \wedge y{=}1]$, where $w_p = 0.75$, $w_r = 0.5$, $w_e = 0.5$. This retrospectively assigns bonuses using categorical branch labels derived from task structure.

**Combined.** $r_i = r_i^{\text{branch-aware}} + r_i^{\text{hindsight}} - r_i^{\text{outcome}}$. This integrates both signal types while subtracting the outcome term to avoid double-counting.

### 3.5 Policy Training and Evaluation

Scored transitions are aggregated into a score table $Q(\phi(s), a) = \frac{1}{n_{s,a}} \sum r_i$ over all occurrences of the state-action pair across episodes. The table policy selects $\arg\max_{a \in A(s)} Q(\phi(s), a)$. For states not encountered during training, the policy selects uniformly at random from available actions — critically, *not* the oracle action, which would inflate evaluation performance.

### 3.6 Experimental Protocol

For each seed, we collect 50 rollouts on the training split using the prompted policy, train one score table per scoring condition on the same collected data, and evaluate each trained policy on a held-out evaluation split. The collection budget is identical across all conditions — the only experimental variable is the scoring function. We use 15 seeds for the main stochastic benchmark, 7 seeds for supporting benchmarks, and 3 seeds for real-benchmark LLM experiments. All controlled experiments are run locally and reproduced on GCP.

**Metrics.** We report binary *success rate* (did the episode reach the goal?), *partial credit* (maximum progress / maximum possible), and *scorer differentiation* (fraction of states where the scorer assigns different preferred actions than a uniform baseline). The last metric directly measures how much training signal each scorer provides.

## 4. Benchmarks

### 4.1 Stochastic Benchmark (Controlled Evaluation)

The stochastic benchmark is our primary controlled evaluation, designed to produce non-trivial success rates where scorer differentiation is meaningful. It contains four task families (4 training tasks, 3 evaluation tasks, 1 smoke test) where pivotal actions succeed probabilistically:

**Stochastic gate chain.** Four gates in sequence, each requiring `unlock_gate` which succeeds with $p = 0.75$. On failure, the agent remains at the same gate. The stall action `rattle_gate` always loops back to the same state. The dead-end action `force_gate` permanently jams the mechanism. The key challenge: when `unlock_gate` fails stochastically, the agent's state is indistinguishable from having taken `rattle_gate`, directly instantiating the stochastic failure masking mechanism.

**Stochastic shopping.** `search_target` returns relevant results with $p = 0.70$, and `open_target` correctly identifies the product with $p = 0.85$ from clean results and $p = 0.60$ from noisy results. Multiple stochastic checkpoints compound, keeping expected success below 0.75.

**Stochastic navigation.** Three sequential phases: `navigate_target` ($p = 0.80$), `pick_target` ($p = 0.85$), and `deliver_target` ($p = 0.90$).

**Stochastic evidence cascade.** Three stages of evidence gathering with `gather_evidence` ($p = 0.75$) and final `synthesize` ($p = 0.80$). The stall action `review_notes` loops back without any dead-end label, specifically testing the cascade failure mode.

### 4.2 Real ALFWorld Benchmark

We validate on the actual ALFWorld environment (Shridhar et al., 2021) in TextWorld mode. Unlike the manifest-driven benchmarks above, this uses the real ALFWorld environment with natural language observations, hundreds of distinct game configurations, and free-text action selection from admissible action lists.

We collect trajectories with two LLMs of different capability levels:

- **Claude Haiku 4.5** (Anthropic, 2025): A commercial API model. 3 seeds × 50 episodes, 30% mean success rate.
- **Qwen2.5-7B-Instruct** (Qwen Team, 2024): An open-weight model self-hosted via vLLM on the GCP instance. 3 seeds × 50 episodes, 17% mean success rate.

Verifier signals are extracted automatically from observation text diffs (§3.2) — no hand-authored labels or privileged environment information.

### 4.3 Deterministic Benchmarks (Supporting Evidence)

We include three deterministic manifest-driven benchmarks as controlled ablations:

**Diagnostic** (5 task families, 10 tasks). Deterministic transitions isolating specific failure modes: dead-end avoidance, enabling recognition, recovery credit, multi-gate chains, and evidence cascades.

**WebShop** (3 tasks). A structural abstraction of WebShop (Yao et al., 2022b) preserving the search → select → cart → checkout decision structure with product-agnostic actions.

**ALFWorld-manifest** (8 tasks). A structural abstraction of ALFWorld covering pick-and-place, heat, clean, cool, and examine task types with abstract action labels.

These produce binary (0 or 1) results, cleanly demonstrating the scoring mechanism in isolation.

## 5. Results

### 5.1 Stochastic Benchmark (Controlled Result)

*Run groups: `group_2cfb2caf7e7d` (local), `group_d92fc9971c06` (GCP). 15 seeds, 50 rollouts, max 15 turns, $\epsilon = 0.5$. Results identical across platforms.*

| Condition | Success Rate | Partial Credit | Min | Max | Avg Turns |
|-----------|-------------|----------------|-----|-----|-----------|
| Prompted baseline | 0.792 | 0.747 | 0.70 | 0.92 | 7.08 |
| Outcome-only | 0.240 | 0.375 | 0.00 | 1.00 | 12.61 |
| Branch-aware | 0.979 | 0.806 | 0.68 | 1.00 | 4.87 |
| Hindsight | 0.695 | 0.739 | 0.34 | 1.00 | 8.02 |
| **Combined** | **1.000** | **0.823** | **1.00** | **1.00** | **4.60** |

*Table 1. Stochastic benchmark results (controlled evaluation). $n = 15$ seeds, 50 rollouts per seed.*

**Outcome-only scoring is actively harmful.** The outcome-only policy (0.240) performs far worse than the unlearned prompted baseline (0.792). The mechanism is stochastic failure masking: both `unlock_gate` (pivotal, $p = 0.75$) and `rattle_gate` (stall) leave the agent at the same state when the pivotal action fails. The trained policy prefers stalls, averaging 12.61 turns vs. 7.08 for the baseline.

**Branch-aware scoring nearly fully recovers** (0.979) because verifier signals fire only on successful pivotal transitions. **Hindsight scoring is less harmful than outcome-only but still degrades below baseline** (0.695 vs. 0.792) with high variance (0.34–1.00). **Combined scoring achieves perfect success** (1.000) across all 15 seeds — the only condition with zero variance.

### 5.2 Deterministic Benchmarks (Supporting Evidence)

| Condition | Diagnostic | WebShop | ALFWorld-manifest |
|-----------|-----------|---------|-------------------|
| Prompted | 0.726 | 0.771 | 0.523 |
| Outcome-only | 0.514 | 0.714 | 0.000 |
| Branch-aware | 0.971 | 1.000 | 1.000 |
| Hindsight | 0.857 | 0.714 | 0.000 |
| **Combined** | **1.000** | **1.000** | **1.000** |

*Table 2. Deterministic benchmark results. $n = 7$ seeds, 50 rollouts per seed.*

The qualitative pattern holds across all three deterministic benchmarks. ALFWorld-manifest shows the most extreme separation: outcome-only and hindsight both achieve 0.000, while branch-aware and combined achieve 1.000.

**Per-family diagnostic breakdown.** Table 3 reveals the complementary failure modes:

| Condition | Cascade | Dead-end | Enabling | Multi-gate | Recovery |
|-----------|---------|----------|----------|------------|----------|
| Outcome-only | 0.143 | 0.571 | 0.714 | 0.143 | 1.000 |
| Branch-aware | 1.000 | 1.000 | 0.857 | 1.000 | 1.000 |
| Hindsight | 0.286 | 1.000 | 1.000 | 1.000 | 1.000 |
| **Combined** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |

*Table 3. Per-family success rates on the diagnostic benchmark. Branch-aware fails on enabling (0.857); hindsight fails on cascade (0.286). Combined achieves 1.000 on all families.*

### 5.3 LLM Collection on Manifest Benchmark

*Run group: `group_e904a833fee4`. Qwen2.5-3B-Instruct via vLLM 0.6.6, temperature 0.7, 3 seeds, 20 rollouts.*

| Condition | Success Rate |
|-----------|-------------|
| LLM baseline (prompted) | 0.833 |
| Outcome-only | 1.000 |
| Branch-aware | 1.000 |
| Combined | 1.000 |

*Table 4. LLM collection on the deterministic diagnostic benchmark. All 104 actions generated by the LLM.*

On the deterministic diagnostic, the LLM's exploration provides sufficient diversity for even outcome-only to learn correct actions. This validates pipeline compatibility but does not stress-test credit assignment.

### 5.4 Real ALFWorld: Training Signal Quality

*This experiment measures the informativeness of training signal — how many states each scorer can differentiate — rather than downstream policy success. We use real ALFWorld (Shridhar et al., 2021) with two LLMs and auto-extracted verifier signals (§3.2). Claude Haiku 4.5: runs `run_2e6990b98cd4`, `run_6b56b8e0571f`, `run_c37097d4b935`. Qwen2.5-7B: runs `run_00767781c915`, `run_22c894f94f13`, `run_74a5173ab1c8`. 3 seeds each, 50 episodes per seed, max 30 turns, temperature 0.7.*

| Model | Success Rate | 95% CI | Episodes | Transitions |
|-------|-------------|--------|----------|-------------|
| Claude Haiku 4.5 | 30.0% | $\pm$ 2.3% | 150 | 3,737 |
| Qwen2.5-7B-Instruct | 16.7% | $\pm$ 3.5% | 150 | 3,962 |

*Table 5. Raw collection performance on real ALFWorld. CI computed across 3 seeds.*

| Scorer | Haiku: States Diff. | Haiku: % | 7B: States Diff. | 7B: % |
|--------|-------|------|------|------|
| Outcome-only | 143 / 1,078 | 13.3% | 41 / 1,212 | 3.4% |
| Branch-aware | 375 / 1,078 | 34.8% | 385 / 1,212 | 31.8% |

*Table 6. Scorer differentiation on real ALFWorld. "States Diff." = states where the scorer assigns different preferred actions. Branch-aware differentiates 2.6x (Haiku) to 9.4x (7B) more states than outcome-only. Note: hindsight scoring equals outcome-only because auto-extracted signals produce numerical progress scores but not categorical branch labels. Combined scoring equals branch-aware for the same reason — the hindsight component contributes zero additional signal when categorical labels are absent (see §6.2).*

We define an action as *task-relevant* if it matches one of ALFWorld's progress-making verb patterns: `go to`, `take`, `put`, `open`, `close`, `use`, `clean`, `heat`, `cool`, `toggle`. All other actions (`examine`, `look`, `inventory`, `help`) are classified as *observational*. This classification follows ALFWorld's own action semantics: progress-making actions change the environment state, while observational actions only query it.

| Scorer | Haiku: Task-Relevant % | 7B: Task-Relevant % |
|--------|---------------------|-------------------|
| Outcome-only | 69% | 54% |
| Branch-aware | **80%** | **82%** |

*Table 7. Fraction of scorer preferences favoring task-relevant actions over observational actions at differentiated states.*

**Key findings:**

1. **Branch-aware provides substantially more training signal than outcome-only.** With Qwen2.5-7B (16.7% success), outcome-only differentiates just 3.4% of states — almost no usable training signal. Branch-aware differentiates 31.8%, a **9.4x improvement**. For Haiku (30% success), the ratio is 2.6x. Outcome-only scoring requires episode success to assign *any* positive credit; auto verifier signals fire on every transition regardless of episode outcome.

2. **Auto-extracted signals are sufficient for branch-aware scoring.** No hand-authored labels, no PDDL goal checker, no privileged information — only observation text diffs and repetition detection.

3. **Hindsight and combined add no benefit when categorical labels are unavailable.** On real ALFWorld with auto-extracted signals, hindsight = outcome-only and combined = branch-aware. The complementarity between branch-aware and hindsight (demonstrated on controlled benchmarks in §5.1–5.2) requires that *both* signal types are present. On real environments where only numerical signals are extractable, branch-aware alone captures the full benefit. This is a structural property of the scorers, not a limitation of the auto-extraction approach.

4. **Branch-aware prefers the right actions.** At 80–82% of differentiated states, branch-aware prefers task-relevant actions over observational ones. Outcome-only's preferences are weaker (54–69%) because it can only distinguish actions from the few successful episodes.

**Limitation.** This experiment measures training signal informativeness, not downstream policy performance. The score-table policy requires exact state-hash matches between collection and evaluation episodes, which limits evaluation on real environments where state spaces are large. Evaluating with neural policies that generalize across states is important future work.

### 5.5 Sensitivity Analysis

*Exploration rate ablation on the diagnostic benchmark, 3 seeds per rate:*

| Condition | $\epsilon = 0.3$ | $\epsilon = 0.5$ | $\epsilon = 0.7$ |
|-----------|-------|-------|-------|
| Prompted | 0.853 | 0.667 | 0.527 |
| Outcome-only | 0.533 | 0.467 | 0.667 |
| Branch-aware | 1.000 | 0.933 | 0.867 |
| Hindsight | 0.867 | 0.867 | 0.867 |
| Combined | 1.000 | 1.000 | 1.000 |

*Table 8. Exploration rate sensitivity. The ranking combined $\geq$ branch-aware $\geq$ hindsight $\gg$ outcome-only holds at all tested rates.*

## 6. Analysis

### 6.1 Loop-Action Bias and Stochastic Failure Masking

In deterministic environments, loop-action bias arises because stall actions appear in successful episodes and receive the same outcome credit as pivotal actions. In stochastic environments, the bias is amplified by *stochastic failure masking*: when a pivotal action fails (probability $1-p$), the agent remains in the same state as a stall, and from outcome-only scoring's perspective the two are identical.

Formally, consider a state $s$ with pivotal action $a_p$ (success probability $p$) and stall action $a_s$. In outcome-only scoring, $E[r(s, a_p)] = p \cdot y_{\text{success}} + (1 - p) \cdot y_{\text{delayed}}$ where $y_{\text{delayed}}$ is the success rate of episodes recovering from the wasted turn. Meanwhile $E[r(s, a_s)] = y_{\text{delayed}}$. The pivotal action's advantage is only $p \cdot (y_{\text{success}} - y_{\text{delayed}})$, which can be small when success rates are moderate.

Branch-aware scoring breaks this conflation because verifier signals fire *only on successful pivotal transitions*. This creates a reliable score gap regardless of stochastic outcome rates.

### 6.2 Complementary Failure Modes

The per-family breakdown in Table 3 reveals a structural complementarity. Branch-aware fails on enabling (0.857) because stall actions produce no dead-end penalty. Hindsight fails on cascade (0.286) and ALFWorld (0.000) because stall actions lack categorical labels. Combined resolves both by combining verifier-signal penalties with categorical bonuses.

The real ALFWorld experiment (§5.4) provides an additional data point: hindsight equals outcome-only because auto-extracted signals produce numerical progress scores but not categorical labels. This confirms that the two signal types are structurally independent and that environments determine which is available.

### 6.3 Signal Quality and Model Capability

The real ALFWorld results reveal an important interaction between signal quality and model capability. For Haiku (30% success), outcome-only has moderate signal (13.3% differentiation) — enough episodes succeed that some credit is assigned. For Qwen2.5-7B (17% success), outcome-only has almost no signal (3.4%). But combined scoring maintains consistent differentiation (31.8–34.8%) regardless of success rate, because auto verifier signals fire on every transition.

This suggests that the combined scorer's practical value is *inversely correlated* with model capability: it matters most precisely when the model is weakest and outcome-only has the least signal. As models improve and success rates rise, outcome-only's signal improves and the gap narrows — but for the frontier of tasks that challenge current models, the combined scorer provides critical additional information.

### 6.4 Partial Credit as a Diagnostic Metric

Binary success rate is a coarse metric — it treats "reached gate 3 of 4 but ran out of turns due to bad luck" identically to "stalled at gate 1 for 15 turns." Our partial credit metric (maximum progress / maximum possible progress) surfaces this distinction. Outcome-only achieves 0.375 partial credit vs. branch-aware's 0.806: even when episodes fail, the branch-aware policy makes substantially more progress.

## 7. Limitations

**Auto-signal noise.** Our automatic verifier signal extraction uses simple text heuristics (room detection, inventory tracking, repetition counting). More sophisticated extraction — for example, training a small classifier on a few hundred labeled transitions, or using an LLM-as-judge — could improve signal quality. We leave this exploration to future work and note that even simple heuristics suffice to demonstrate the mechanism.

**Table-based policy.** Score tables require exact state-hash matches and cannot generalize to unseen state-action combinations. Neural function approximation (as in AgentPRM; Choudhury, 2025) or LoRA fine-tuning would relax this limitation. The scorer differentiation results in §5.4 measure the *quality of the training signal* rather than downstream policy performance, and we expect the signal-quality advantage to transfer to neural training.

**Fixed scorer weights.** The scorer weights ($w_v = 0.25$, $w_d = 0.5$, $w_p = 0.75$, $w_r = 0.5$, $w_e = 0.5$) are hand-selected. The qualitative ranking is stable across exploration rates and benchmarks (Table 8), but quantitative margins may shift.

**Manifest abstractions.** WebShop and diagnostic results use structural abstractions, not the original benchmarks. The real ALFWorld experiment (§5.4) addresses this limitation for one benchmark; extending to real WebShop is future work.

**Evaluation episodes.** The real ALFWorld experiment measures scorer differentiation (training signal quality) rather than downstream policy success rate, because the table-based policy requires state-hash overlap between collection and evaluation. Future work should evaluate with neural policies that can generalize across states.

## 8. Conclusion

We have identified loop-action bias as a mechanistic failure mode of outcome-only credit assignment that is amplified by stochastic failure masking. On controlled benchmarks with hand-authored verifier signals, outcome-only scoring produces policies far worse than the unlearned baseline (0.24 vs. 0.79 on the stochastic benchmark), while the combined scorer — integrating both branch-aware and hindsight components — achieves perfect success (1.00) across all 15 seeds. The complementarity between branch-aware and hindsight scoring is a structural property: branch-aware depends on numerical verifier signals, hindsight depends on categorical branch labels, and the combined scorer is needed only when both signal types are present.

On the real ALFWorld benchmark, we demonstrate that automatically extracted verifier signals — observation text diffs and repetition detection, requiring no hand-authored labels — produce substantially more informative training signal than outcome-only scoring. Branch-aware scoring differentiates 2.6–9.4x more states than outcome-only, with the advantage *largest for weaker models* (9.4x for Qwen2.5-7B at 17% success vs. 2.6x for Haiku at 30% success). In this setting, where only numerical signals are available, the hindsight component provides no additional benefit and combined reduces to branch-aware — confirming that the relevant contribution on real environments is the branch-aware verifier signal mechanism rather than the combined scorer per se.

The practical implication: in any agent environment with observable transition feedback — tool-use confirmation codes, code execution output, state descriptions, API response codes — incorporating these signals into credit assignment produces more informative training signal than outcome-only scoring. The signals need not be elaborate or hand-authored; simple observation-diff heuristics suffice. Translating this signal advantage into downstream policy improvement via neural training (LoRA fine-tuning, DPO) is the natural next step, which we leave to future work.

## Reproducibility

All code, benchmarks, configurations, and run artifacts are available at the project repository. Controlled benchmark results are reproduced exactly on both local (macOS 15.7, Python 3.13.7) and cloud (GCP g2-standard-8, NVIDIA L4, Ubuntu 22.04, Python 3.11.15) infrastructure. Real ALFWorld experiments are documented with exact run IDs, commands, and SQL queries in `docs/real_alfworld_experiment.md`. All reported results are traceable to specific run identifiers in the SQLite database. Dependency versions are pinned in `requirements-lock.txt`. Complete reproduction instructions are provided in `docs/reproducibility.md` and `docs/environment.md`.

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
- Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback. *Advances in Neural Information Processing Systems (NeurIPS)*.
- Pignatelli, E., Ferret, J., Geist, M., Mesnard, T., van Hasselt, H., & Pietquin, O. (2023). A survey of temporal credit assignment in deep reinforcement learning. *arXiv:2312.01072*.
- Ren, Z., Guo, D., & Littman, M. L. (2024). Randomized return decomposition. *International Conference on Learning Representations (ICLR)*.
- Schick, T., et al. (2023). Toolformer: Language models can teach themselves to use tools. *Advances in Neural Information Processing Systems (NeurIPS)*.
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
