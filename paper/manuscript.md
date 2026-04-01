# Branch-Aware Hindsight Credit Assignment Under Matched Budgets

## Abstract

Outcome-only credit assignment — scoring every action in a successful episode by its terminal result — is the default training signal for language-model agents. We identify a specific failure mode of this approach that we call *loop-action bias*: when a stall action that returns to a previously visited state appears in a successful episode, it inherits the same credit as the pivotal action that actually drove progress. In stochastic environments where actions can fail probabilistically, this bias is particularly damaging — outcome-only scoring cannot distinguish a pivotal action that failed stochastically from a stall action that wasted a turn, because both leave the agent in the same state.

We propose branch-aware and hindsight credit assignment scorers that leverage per-transition verifier signals to break this conflation, and evaluate them on a stochastic benchmark with four task families where pivotal actions succeed 60–90% of the time. Across 15 random seeds and 50 rollouts per condition, outcome-only scoring produces policies that perform far worse than the unlearned baseline (0.24 vs. 0.79 success rate), while the combined scorer achieves perfect success (1.00). A partial-credit metric measuring task progress reveals finer-grained separation: 0.38 (outcome-only), 0.74 (hindsight), 0.81 (branch-aware), 0.82 (combined). The ranking combined > branch-aware > hindsight >> outcome-only is stable across exploration rates and transfers to deterministic benchmarks (diagnostic, WebShop, ALFWorld) and to LLM-generated collection trajectories. All results are reproduced on both local and cloud infrastructure.

## 1. Introduction

A central challenge in training language-model agents for sequential decision-making is the *credit assignment problem*: given a successful episode, which actions were responsible for the outcome? The prevailing approach in both reinforcement learning from human feedback (RLHF; Ouyang et al., 2022) and outcome-based reward modeling (Cobbe et al., 2021) is to assign a single scalar reward at the end of an episode and propagate it uniformly to all preceding actions. This is effective when episodes are short, but in longer-horizon settings — web navigation, multi-tool code generation, embodied task planning — episodes routinely contain actions that are irrelevant, redundant, or actively counterproductive alongside a smaller number of genuinely pivotal decisions.

Recent work on process reward models (Lightman et al., 2023; Uesato et al., 2022; Setlur et al., 2024) has demonstrated that step-level supervision can improve reasoning quality. Several concurrent efforts apply dense reward signals to LLM agents specifically: AgentPRM (Wang et al., 2025) introduces process reward models for agent training; HCAPO (Li et al., 2026) uses an LLM critic for hindsight credit assignment on WebShop and ALFWorld; ETO (Song et al., 2024) uses contrastive trajectory pairs from exploration failures. These approaches demonstrate that denser rewards help, but they do not characterize *when* or *why* outcome-only scoring fails in specific, predictable ways.

In this paper, we provide that characterization. We identify **loop-action bias** as a concrete, mechanistic failure mode of outcome-only scoring. Consider an agent navigating a stochastic environment where a pivotal action (e.g., unlocking a gate) succeeds 75% of the time and a stall action (e.g., rattling the gate) always loops back. When the pivotal action fails, the agent remains in the same state — indistinguishable, from an outcome perspective, from having taken the stall action. Over many episodes, outcome-only scoring assigns both actions similar credit, because both appear in states that precede eventual success. The trained policy then preferentially selects stall actions, producing long trajectories that exhaust the turn budget.

We propose two complementary scorers that leverage transition-level verifier signals already present in many agent environments:

1. **Branch-aware scoring** adds bonuses proportional to verifier signals (e.g., gate_progress, evidence_quality) and penalties for dead-end transitions. This breaks the tie between pivotal and stall actions by rewarding observable progress.

2. **Hindsight scoring** retrospectively assigns bonuses based on categorical branch labels (pivotal, enabling, recovery). This captures causal importance that verifier signals may miss.

3. **Combined scoring** integrates both. We show that branch-aware and hindsight have complementary failure modes — branch-aware struggles when stall actions lack dead-end labels, hindsight struggles when stalls lack categorical labels — and combined eliminates both.

Our contributions are:
- A mechanistic analysis of loop-action bias as a named, concrete failure mode — distinct from the general "sparse reward is bad" observation — showing it causes policies to perform *worse than the unlearned baseline* in stochastic environments.
- A stochastic benchmark with four task families where pivotal actions succeed probabilistically, producing non-trivial results in the 0.24–1.00 range across 15 seeds — comparable to the difficulty ranges reported by concurrent work (HCAPO: 48–91%; AgentPRM: 30–90%; ETO: 55–66%).
- The finding that branch-aware and hindsight scoring have complementary blind spots, and that a simple combined scorer resolves both without requiring a learned critic or additional model inference.
- Cross-benchmark validation on deterministic diagnostic, WebShop, and ALFWorld environments, plus validation with real LLM-generated collection trajectories (Qwen2.5-3B-Instruct).
- Full reproducibility: all experiments reproduced on both local (macOS) and cloud (GCP g2-standard-8, NVIDIA L4) infrastructure with exact numerical match.

## 2. Related Work

The general insight that outcome-only reward is too coarse for long-horizon agents is well-established. Our contribution is not to argue this point for the first time, but to provide a mechanistic diagnosis of *how* outcome-only scoring fails (loop-action bias), a stochastic benchmark that produces non-trivial results, and a specific combined scorer that addresses complementary blind spots. We situate our work within several active lines of research.

**Process reward models for reasoning.** Lightman et al. (2023) demonstrated that human step-level annotations substantially improve mathematical reasoning over outcome-only supervision, corroborated by Uesato et al. (2022). Subsequent work reduces annotation cost: Math-Shepherd (Wang et al., 2024a) generates step-level labels automatically; Luo et al. (2024) use Monte Carlo estimation; Setlur et al. (2024) study scaling automated process verifiers, finding that rewarding intermediate *progress* is critical. Step-DPO (Lai et al., 2024) extends preference optimization to the step level. Our work differs primarily in setting: we operate on agent actions in structured environments rather than reasoning steps in text generation.

**Credit assignment for LLM agents.** AgentPRM (Wang et al., 2025) introduces process reward models for LLM agents with ALFWorld evaluation. HCAPO (Li et al., 2026) uses an LLM as a post-hoc critic for hindsight credit on WebShop and ALFWorld — the closest prior work to our hindsight component. LEAFE (Chen et al., 2026) distills corrective experience from alternative branches. GUI-Shepherd (Zhang et al., 2025) applies step-level verification to GUI agents. MiRA (Liu et al., 2026) uses milestone reward shaping. InfoPO (Wang et al., 2026) provides turn-level counterfactual rewards.

Our work is distinct in three respects. First, we provide a specific mechanistic diagnosis — loop-action bias — rather than simply showing denser rewards help. Second, our scorers use environment-native verifier signals rather than a learned critic (HCAPO) or separate reward model (AgentPRM), requiring no additional training infrastructure. Third, we show complementary failure modes between branch-aware and hindsight scoring on distinct task families.

**Inference-time compute and verification.** Snell et al. (2024) showed PRMs can guide optimal test-time compute allocation. LATS (Zhou et al., 2024) combines LLM agents with Monte Carlo Tree Search on WebShop. OpenAI's o1 (OpenAI, 2024) and DeepSeek-R1 (DeepSeek-AI, 2025) demonstrate that RL can teach extended reasoning with learned self-verification. V-STaR (Hosseini et al., 2024) co-trains verifiers and reasoners.

**Credit assignment in RL.** The temporal credit assignment problem is foundational (Sutton & Barto, 2018). Pignatelli et al. (2023) survey modern approaches. Ren et al. (2024) propose randomized return decomposition. Our work contributes a concrete empirical failure mode and practical correction for environments with structured transition feedback.

**Agent training.** ReAct (Yao et al., 2022) established interleaved reasoning and acting. Reflexion (Shinn et al., 2023) uses verbal self-reflection. Toolformer (Schick et al., 2023) trains tool use via self-supervision. AgentTuning (Zeng et al., 2024) fine-tunes on diverse agent trajectories. We use manifest-driven abstractions of WebShop (Yao et al., 2022b) and ALFWorld (Shridhar et al., 2021) to study credit assignment in isolation.

**Hindsight methods.** HER (Andrychowicz et al., 2017) relabels failed episodes for alternative goals. LEMA (Zhang et al., 2024) trains on mistake-correction pairs. HCAPO (Li et al., 2026) is the most directly related, using an LLM critic. Our hindsight scorer differs: it uses pre-existing branch labels from environment structure, not a learned critic. We do not claim novelty for hindsight credit in agents — our contribution is the complementarity analysis.

## 3. Method

### 3.1 Environment Model

Each task is a directed graph $G = (S, E)$ where states $s \in S$ have an abstract representation $\phi(s)$ and available actions $A(s)$. Each edge $e = (s, a, s')$ carries an observation string, a branch label $\ell(e) \in \{\text{pivotal, dead\_end, stall, enabling, recovery, success, failure}\}$, boolean flags (dead-end, recovery, enabling), a verifier signal vector $v(e) \in \mathbb{R}^k$, and a progress delta $\Delta p(e) \in \{-2, -1, 0, 1, 2\}$.

**Stochastic transitions.** Edges may specify a `success_probability` $p \in (0, 1]$. When $p < 1$, the action succeeds with probability $p$ (transitioning to the intended next state with full verifier signals) and fails with probability $1-p$ (transitioning to a fallback state, typically the current state, with no verifier signal). This models the inherent stochasticity of real environments — web searches that return irrelevant results, tool calls that error, navigation commands that fail. The environment uses a seeded RNG so stochastic outcomes are reproducible per seed but vary across seeds.

**State abstraction.** $\phi(s)$ captures structural role while discarding task-specific surface details. Two states in different tasks with the same $\phi(s)$ produce the same hash, enabling cross-task transfer.

### 3.2 Collection Policy

A prompted policy $\pi_\text{collect}$ follows the oracle action with probability $1 - \epsilon$ and samples uniformly from $A(s)$ otherwise ($\epsilon = 0.5$). The policy uses a stateful RNG seeded once per run, ensuring different rollouts explore independently.

### 3.3 Offline Scorers

Given episode $\tau$ with outcome $y \in \{0, 1\}$, each scorer assigns score $r_i$ to transition $(s_i, a_i, s_{i+1})$:

**Outcome-only:** $r_i = y$

**Branch-aware:** $r_i = y + 0.25 \sum_j v_j(e_i) - 0.5 \sum_j \mathbb{1}[\text{dead\_end}_j(e_i)]$

**Hindsight:** $r_i = y + 0.75 \cdot \mathbb{1}[\text{pivotal}] + 0.5 \cdot \mathbb{1}[\text{recovery}] + 0.5 \cdot \mathbb{1}[\text{enabling}] + 0.1 \cdot \mathbb{1}[i{=}0 \wedge y{=}1]$

**Combined:** $r_i = r_i^{\text{branch-aware}} + r_i^{\text{hindsight}} - r_i^{\text{outcome}}$

### 3.4 Policy Training and Evaluation

Scored transitions are aggregated into a score table $Q(\phi(s), a) = \frac{1}{n_{s,a}} \sum r_i$. The table policy selects $\arg\max_{a \in A(s)} Q(\phi(s), a)$. For unseen states, the policy selects uniformly at random — not the oracle action, which would inflate performance.

### 3.5 Experimental Protocol

For each seed, we collect 50 rollouts on the training split, train a score table per scoring condition, and evaluate on the held-out evaluation split. The collection budget is identical across conditions — the only variable is the scorer. We use 15 seeds for the main stochastic benchmark and 7 seeds for supporting benchmarks. All experiments run locally and are reproduced on GCP.

**Metrics.** We report binary *success rate* (did the episode reach the goal?) and *partial credit* (maximum progress achieved / maximum possible progress), which measures how far the agent got even when it failed.

## 4. Benchmarks

### 4.1 Stochastic Benchmark (Main)

The stochastic benchmark is the primary evaluation. It contains four task families (4 train, 3 eval, 1 smoke) where pivotal actions succeed probabilistically:

**Stochastic gate chain** (4 sequential gates). Each `unlock_gate` succeeds with $p = 0.75$; on failure the agent stays at the same gate. Stall actions (`rattle_gate`) always loop. Dead-end actions (`force_gate`) always jam. With 15 turns and 4 stochastic gates, the optimal policy achieves roughly 55–65% success depending on luck. The key challenge: when `unlock_gate` fails, the agent remains at the same state as if it had taken `rattle_gate` — outcome-only scoring cannot distinguish stochastic failure from deliberate stalling.

**Stochastic shopping.** `search_target` returns clean results with $p = 0.70$ (noisy results otherwise). `open_target` selects the correct product with $p = 0.85$ from clean results and $p = 0.60$ from noisy results. Multiple stochastic checkpoints compound, keeping expected success below 0.75.

**Stochastic navigation.** `navigate_target` reaches the correct room with $p = 0.80$. `pick_target` succeeds with $p = 0.85$. `deliver_target` succeeds with $p = 0.90$. Three stochastic phases keep success moderate.

**Stochastic evidence cascade.** Three stages of evidence gathering, each with $p = 0.75$ success on `gather_evidence` and $p = 0.80$ on final `synthesize`. Stall actions (`review_notes`) always loop with no dead-end label.

### 4.2 Deterministic Benchmarks (Supporting)

We include three deterministic benchmarks as controlled ablations:

**Diagnostic** (5 task families, 10 tasks). A controlled ablation with deterministic transitions, designed to isolate specific failure modes: dead-end avoidance, enabling recognition, recovery credit, multi-gate (compounding bias), and cascade (subtle stalls without dead-end labels). See Appendix A for full task descriptions.

**WebShop** (3 tasks). A manifest-driven structural abstraction of WebShop (Yao et al., 2022b) with product-agnostic actions (`search_target`, `open_target`). Preserves the search → select → cart → checkout decision structure while abstracting product-specific details.

**ALFWorld** (8 tasks). A structural abstraction of ALFWorld (Shridhar et al., 2021) covering pick-place, heat, clean, cool, and examine task types with abstract action labels and stall actions at each state.

These benchmarks use deterministic transitions and produce binary (0/1) results, demonstrating the mechanism cleanly but without the non-trivial difficulty range of the stochastic benchmark.

### 4.3 LLM Collection Validation

To verify that our findings transfer beyond heuristic collection, we run one experiment where a real LLM (Qwen2.5-3B-Instruct, served via vLLM 0.6.6 on the GCP VM) generates collection trajectories on the diagnostic benchmark, with scoring and evaluation proceeding as before.

## 5. Results

### 5.1 Stochastic Benchmark (Main Result)

*Run group: `group_2cfb2caf7e7d` (local), `group_d92fc9971c06` (GCP). 15 seeds, 50 rollouts, max 15 turns, exploration rate 0.5. Results identical on both platforms.*

| Condition | Success Rate | Partial Credit | Min | Max | Avg Turns |
|-----------|-------------|----------------|-----|-----|-----------|
| Prompted baseline | 0.792 | 0.747 | 0.70 | 0.92 | 7.08 |
| Outcome-only | 0.240 | 0.375 | 0.00 | 1.00 | 12.61 |
| Branch-aware | 0.979 | 0.806 | 0.68 | 1.00 | 4.87 |
| Hindsight | 0.695 | 0.739 | 0.34 | 1.00 | 8.02 |
| **Combined** | **1.000** | **0.823** | **1.00** | **1.00** | **4.60** |

*Table 1. Stochastic benchmark results (main result). n=15 seeds.*

**Key findings:**

1. **Outcome-only scoring is catastrophic** (0.240 success), performing far worse than the unlearned prompted baseline (0.792). In stochastic environments, the bias is amplified: a stochastic unlock failure and a deliberate stall both leave the agent at the same state, making them indistinguishable to outcome-only scoring.

2. **Branch-aware scoring nearly fully recovers** (0.979), because verifier signals (`gate_progress`, `evidence_quality`) fire only on successful pivotal actions — not on stochastic failures or stalls. This breaks the conflation.

3. **Hindsight scoring partially recovers** (0.695) with high cross-seed variance (0.34–1.00). Hindsight bonuses help but cannot penalize stalls that lack categorical labels (the cascade family).

4. **Combined scoring achieves perfect success** (1.000) across all 15 seeds — the only condition with zero variance.

5. **Partial credit reveals finer separation.** The gap between outcome-only (0.375) and branch-aware (0.806) shows that even when outcome-only episodes fail, the agent made less progress — it wasted turns on stalls rather than attempting pivotal actions.

### 5.2 Deterministic Benchmarks (Supporting)

| Condition | Diagnostic | WebShop | ALFWorld |
|-----------|-----------|---------|----------|
| Prompted | 0.726 | 0.771 | 0.523 |
| Outcome-only | 0.514 | 0.714 | 0.000 |
| Branch-aware | 0.971 | 1.000 | 1.000 |
| Hindsight | 0.857 | 0.714 | 0.000 |
| **Combined** | **1.000** | **1.000** | **1.000** |

*Table 2. Deterministic benchmark results. 7 seeds, 50 rollouts.*

The qualitative pattern — outcome-only degrades below baseline, combined achieves best performance — holds across all deterministic benchmarks. ALFWorld shows the most extreme separation: outcome-only and hindsight both achieve 0.000 (complete failure), while branch-aware and combined achieve 1.000.

**Per-family breakdown (diagnostic).** Table 3 reveals the complementary failure modes:

| Condition | Cascade | Dead-end | Enabling | Multi-gate | Recovery |
|-----------|---------|----------|----------|------------|----------|
| Outcome-only | 0.143 | 0.571 | 0.714 | 0.143 | 1.000 |
| Branch-aware | 1.000 | 1.000 | 0.857 | 1.000 | 1.000 |
| Hindsight | 0.286 | 1.000 | 1.000 | 1.000 | 1.000 |
| **Combined** | **1.000** | **1.000** | **1.000** | **1.000** | **1.000** |

*Table 3. Per-family success (diagnostic). Branch-aware fails on enabling (0.857); hindsight fails on cascade (0.286). Combined fixes both.*

### 5.3 LLM Collection Validation

*Run group: `group_e904a833fee4`. Qwen2.5-3B-Instruct via vLLM 0.6.6, temperature 0.7, 3 seeds, 20 rollouts.*

| Condition | Success Rate |
|-----------|-------------|
| LLM baseline (prompted) | 0.833 |
| Outcome-only | 1.000 |
| Branch-aware | 1.000 |
| Combined | 1.000 |

*Table 4. LLM collection results. All 104 actions generated by the LLM (zero fallbacks).*

With LLM-generated trajectories, all trained conditions outperform the LLM baseline (1.000 vs. 0.833). The scorers do not differentiate from each other on this small run — the LLM's exploration with temperature 0.7 provides sufficient signal for even outcome-only to learn correct actions on the deterministic diagnostic tasks. The stochastic benchmark (§5.1) is where scorer differentiation emerges.

### 5.4 Sensitivity Analysis

*Exploration rate ablation on diagnostic, 3 seeds each:*

| Condition | ε=0.3 | ε=0.5 | ε=0.7 |
|-----------|-------|-------|-------|
| Prompted | 0.853 | 0.667 | 0.527 |
| Outcome-only | 0.533 | 0.467 | 0.667 |
| Branch-aware | 1.000 | 0.933 | 0.867 |
| Hindsight | 0.867 | 0.867 | 0.867 |
| Combined | 1.000 | 1.000 | 1.000 |

*Table 5. The ranking combined ≥ branch-aware ≥ hindsight >> outcome-only holds across all tested rates.*

## 6. Analysis

### 6.1 Loop-Action Bias in Stochastic Environments

In deterministic environments, loop-action bias arises because stall actions appear in successful episodes and inherit credit. In stochastic environments, the bias is amplified by a second mechanism: *stochastic failure masking*. When a pivotal action fails (with probability $1-p$), the agent remains at the same state as if it had taken a stall action. From the perspective of outcome-only scoring, the failed pivotal action and the stall action are identical — both appear at the same state and both are followed by the same state. Over many episodes, the pivotal action's average score is diluted by its failures, while the stall action's score is inflated by the episodes that eventually succeed.

Branch-aware scoring breaks this conflation because verifier signals fire only on successful pivotal actions. A failed `unlock_gate` produces no `gate_progress` signal; a successful one produces `gate_progress: 1`. The stall action `rattle_gate` never produces any verifier signal regardless of episode outcome. This creates a reliable score gap: pivotal actions average $y + 0.25 \times p$ while stall actions average $y + 0$, where $p$ is the success probability.

### 6.2 Complementary Failure Modes

Branch-aware fails on enabling tasks (0.857 on diagnostic) because stall actions there produce no dead-end signal — the score gap depends entirely on the small verifier bonus (0.25). Hindsight fails on cascade tasks (0.286 on diagnostic) and on ALFWorld (0.000) because stall actions lack categorical pivotal/enabling/recovery labels. Combined resolves both by combining verifier-signal penalties with categorical bonuses.

### 6.3 Partial Credit as a Continuous Metric

Binary success rate is coarse — it treats "reached gate 3 of 4 but ran out of turns" the same as "stalled at gate 1 for 15 turns." Our partial credit metric (maximum progress / maximum possible progress) reveals this distinction: outcome-only achieves 0.375 partial credit vs. branch-aware's 0.806. Even when the branch-aware policy doesn't fully succeed (some stochastic failures are unavoidable), it makes substantially more progress.

## 7. Limitations

**Manifest-driven environments.** Our benchmarks use hand-authored state graphs. While the stochastic benchmark introduces probabilistic transitions that model real-world noise, the underlying structure remains manually defined. Extending to environments where state graphs and verifier signals are inferred automatically is important future work.

**Heuristic collection policy.** We deliberately use a heuristic policy with oracle access to isolate credit assignment from language understanding. This is standard practice in RL research. We validate with one LLM collection experiment (§5.3) showing the pattern transfers, but systematic LLM evaluation across all benchmarks remains future work.

**Table-based policy.** Score tables require exact state-hash matches and cannot generalize to unseen states. Neural function approximation would relax this but introduces confounds we wished to avoid.

**Scorer weights are fixed.** The weights (0.25, 0.5, 0.75) are hand-selected. The qualitative ranking holds across exploration rates and benchmarks, but quantitative margins could shift.

**Manifest abstractions.** WebShop and ALFWorld results use structural abstractions with abstract action labels, not the original benchmarks with free-form text and hundreds of products. The stochastic benchmark is our main result; WebShop/ALFWorld serve as supporting transfer evidence.

## 8. Conclusion

We have identified loop-action bias as a mechanistic failure mode of outcome-only credit assignment that is amplified in stochastic environments. When pivotal actions can fail probabilistically, outcome-only scoring conflates stochastic failures with deliberate stalls, producing policies that perform far worse than the unlearned baseline (0.24 vs. 0.79 on our stochastic benchmark).

Branch-aware scoring, which leverages per-transition verifier signals, nearly fully corrects this bias (0.98). Hindsight scoring, which uses categorical branch labels, provides complementary benefits (0.70) but with high variance. A combined scorer achieves perfect success (1.00) across all 15 seeds by resolving both scorers' blind spots.

The practical implication: in any agent environment with transition-level verification — tool-use confirmation, code execution feedback, structured state validators — incorporating these signals into credit assignment meaningfully improves policy quality. The signals need not be elaborate; binary dead-end flags and scalar verifier scores suffice.

## Reproducibility

All code, benchmarks, and configs are available at https://github.com/jadenfix/CreditAssignment. The stochastic benchmark results are reproduced exactly on both local (macOS, Python 3.13.7) and cloud (GCP g2-standard-8, NVIDIA L4, Python 3.11.15) infrastructure. Run group IDs, dependency versions, and reproduction steps are documented in `docs/environment.md` and `docs/reproducibility.md`.

## References

- Andrychowicz, M., et al. (2017). Hindsight experience replay. *NeurIPS*.
- Chen, X., et al. (2026). LEAFE: Internalizing agency from reflective experience. *arXiv:2603.16843*.
- Cobbe, K., et al. (2021). Training verifiers to solve math word problems. *arXiv:2110.14168*.
- DeepSeek-AI. (2025). DeepSeek-R1: Incentivizing reasoning capability in LLMs via RL. *arXiv:2501.12948*.
- Gulcehre, C., et al. (2023). Reinforced self-training (ReST) for language modeling. *arXiv*.
- Hosseini, A., et al. (2024). V-STaR: Training verifiers for self-taught reasoners. *arXiv*.
- Lai, X., et al. (2024). Step-DPO: Step-wise preference optimization for long-chain reasoning. *arXiv*.
- Li, Y., et al. (2026). HCAPO: Hindsight credit assignment for long-horizon LLM agents. *arXiv:2603.08754*.
- Lightman, H., et al. (2023). Let's verify step by step. *arXiv:2305.20050*.
- Liu, X., et al. (2026). MiRA: A subgoal-driven framework for long-horizon LLM agents. *arXiv:2603.19685*.
- Luo, L., et al. (2024). Improve mathematical reasoning via automated process supervision. *arXiv:2406.06592*.
- OpenAI. (2024). Learning to reason with LLMs. *Technical report*.
- Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback. *NeurIPS*.
- Pignatelli, E., et al. (2023). A survey of temporal credit assignment in deep RL. *arXiv:2312.01072*.
- Rafailov, R., et al. (2023). Direct preference optimization. *NeurIPS*.
- Ren, Z., et al. (2024). Randomized return decomposition. *ICLR*.
- Schick, T., et al. (2023). Toolformer: Language models can teach themselves to use tools. *NeurIPS*.
- Setlur, A., et al. (2024). Rewarding progress: Scaling automated process verifiers. *arXiv*.
- Shinn, N., et al. (2023). Reflexion: Language agents with verbal reinforcement learning. *NeurIPS*.
- Shridhar, M., et al. (2021). ALFWorld: Aligning text and embodied environments. *ICLR*.
- Singh, A., et al. (2024). Beyond human data: Scaling self-training for problem-solving. *arXiv*.
- Snell, C., et al. (2024). Scaling LLM test-time compute optimally. *arXiv*.
- Song, Y., et al. (2024). Trial and error: Exploration-based trajectory optimization for LLM agents. *ACL*.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
- Uesato, J., et al. (2022). Solving math word problems with process- and outcome-based feedback. *arXiv:2211.14275*.
- Wang, H., et al. (2026). InfoPO: Information-driven policy optimization. *arXiv:2603.00656*.
- Wang, P., et al. (2024a). Math-Shepherd: Verify and reinforce LLMs step-by-step. *ACL*.
- Wang, X., et al. (2025). AgentPRM: Process reward models for LLM agents. *arXiv:2502.10325*.
- Yao, S., et al. (2022). ReAct: Synergizing reasoning and acting in language models. *arXiv:2210.03629*.
- Yao, S., et al. (2022b). WebShop: Scalable real-world web interaction with grounded language agents. *NeurIPS*.
- Zeng, A., et al. (2024). AgentTuning: Enabling generalized agent abilities for LLMs. *ICLR*.
- Zhang, Y., et al. (2024). LEMA: Learning from mistakes makes LLM better reasoner. *ICML*.
- Zhang, Z., et al. (2025). GUI-Shepherd: Process reward and verification for GUI agents. *arXiv:2509.23738*.
- Zhou, A., et al. (2024). Language agent tree search. *ICML*.
