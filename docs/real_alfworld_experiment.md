# Real ALFWorld Experiment Log

Exact steps, run IDs, and results for reproducing the real ALFWorld benchmark experiments.

## Infrastructure

| Component | Version |
|-----------|---------|
| GCP Instance | g2-standard-8, us-central1-a |
| GPU | NVIDIA L4 (23,034 MiB) |
| OS | Ubuntu 22.04.5 LTS |
| Python | 3.11.15 |
| ALFWorld | 0.4.2 |
| vLLM | 0.6.6 |
| TextWorld | 1.7.0 |
| ALFWORLD_DATA | ~/.cache/alfworld |

## Experiment Design

Two LLM backends, each with 3 seeds x 50 episodes on real ALFWorld (text mode):

| Backend | Model | Seeds | Episodes/Seed | Max Turns | Temperature |
|---------|-------|-------|---------------|-----------|-------------|
| Anthropic API | claude-haiku-4-5-20251001 | 7, 11, 19 | 50 | 30 | 0.7 |
| vLLM (self-hosted) | Qwen/Qwen2.5-7B-Instruct | 7, 11, 19 | 50 | 30 | 0.7 |

Verifier signal mode: `auto` (observation text diffs, no oracle/privileged information).

## Commands Used

### Haiku Collection

```bash
export ALFWORLD_DATA=~/.cache/alfworld
export ANTHROPIC_API_KEY=<redacted>

# Seed 7
python -u scripts/collect_alfworld_real.py \
  --backend anthropic --model claude-haiku-4-5-20251001 \
  --num-episodes 50 --max-turns 30 --seed 7 --temperature 0.7 --task-limit 80

# Seed 11
python -u scripts/collect_alfworld_real.py \
  --backend anthropic --model claude-haiku-4-5-20251001 \
  --num-episodes 50 --max-turns 30 --seed 11 --temperature 0.7 --task-limit 80

# Seed 19
python -u scripts/collect_alfworld_real.py \
  --backend anthropic --model claude-haiku-4-5-20251001 \
  --num-episodes 50 --max-turns 30 --seed 19 --temperature 0.7 --task-limit 80
```

### Qwen2.5-7B Collection

```bash
# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct --max-model-len 2048 --dtype float16 --port 8000

# Each seed
python -u scripts/collect_alfworld_real.py \
  --backend vllm --model Qwen/Qwen2.5-7B-Instruct \
  --num-episodes 50 --max-turns 30 --seed 7 --temperature 0.7 --task-limit 80
# (repeat for seeds 11, 19)
```

## Run IDs and Results

### Claude Haiku 4.5

| Seed | Run ID | Success | Rate | Avg Turns |
|------|--------|---------|------|-----------|
| 7 | `run_2e6990b98cd4` | 14/50 | 28.0% | 25.2 |
| 11 | `run_6b56b8e0571f` | 15/50 | 30.0% | 24.8 |
| 19 | `run_c37097d4b935` | 16/50 | 32.0% | 24.7 |
| **Mean** | | **15/50** | **30.0%** | **24.9** |

### Qwen2.5-7B-Instruct

| Seed | Run ID | Success | Rate | Avg Turns |
|------|--------|---------|------|-----------|
| 7 | `run_00767781c915` | 7/50 | 14.0% | 26.7 |
| 11 | `run_22c894f94f13` | 10/50 | 20.0% | 25.6 |
| 19 | `run_74a5173ab1c8` | 8/50 | 16.0% | 26.9 |
| **Mean** | | **8.3/50** | **16.7%** | **26.4** |

### Previous: Qwen2.5-3B-Instruct (reference only)

| Seed | Run ID | Success | Rate | Avg Turns |
|------|--------|---------|------|-----------|
| 42 | `run_7ce244778ebe` | 1/50 | 2.0% | 29.9 |
| 7 | `run_841ef1846fa4` | 0/50 | 0.0% | 30.0 |

## Scorer Analysis (Haiku, 150 episodes)

| Scorer | Targets | Mean Score | States Differentiated | % | Action Verbs |
|--------|---------|------------|----------------------|---|-------------|
| outcome_only | 3737 | 0.157 | 143/1078 | 13.3% | 98 (69%) |
| branch_aware | 3737 | 0.165 | 375/1078 | 34.8% | 299 (80%) |
| hindsight | 3737 | 0.158 | 143/1078 | 13.3% | 98 (69%) |
| **combined** | 3737 | 0.166 | **375/1078** | **34.8%** | **299 (80%)** |

**Key finding**: Combined/branch-aware differentiate 2.6x more states than outcome-only (375 vs 143), and 80% of their preferences are for action verbs vs 69% for outcome-only.

## Auto Verifier Signal Distribution (Haiku)

| Signal | Positive | Negative | Neutral | Total |
|--------|----------|----------|---------|-------|
| progress | 1743 | 0 | 1994 | 3737 |
| repetition | 0 | 1665 | 0 | 1665 |

Signals are extracted automatically from observation text diffs:
- `progress (+0.5)`: room navigation detected (new "You arrive at" pattern)
- `progress (+1.0)`: inventory change (pick up / put down object)
- `repetition (-0.5)`: same observation seen 3+ times in episode
- `progress (0.0)`: no detectable state change

## ALFWorld Environment Details

- Total games in train split: 3,553 (from 8,810 game files)
- Task limit per experiment: 80 (sampled with seeded RNG)
- Task types: pick_and_place_simple, look_at_obj_in_light, pick_clean_then_place_in_recep, pick_heat_then_place_in_recep, pick_cool_then_place_in_recep, pick_two_obj_and_place
- Environment: AlfredTWEnv (TextWorld mode, no rendering)
- Config: domain_randomization=False, max_nb_steps_per_episode=50

## Data Storage

All episodes stored in `runs/creditlab.sqlite`:
```sql
-- List Haiku episodes
SELECT run_id, COUNT(*) FROM episodes
WHERE run_id IN ('run_2e6990b98cd4', 'run_6b56b8e0571f', 'run_c37097d4b935')
GROUP BY run_id;

-- List Qwen 7B episodes
SELECT run_id, COUNT(*) FROM episodes
WHERE run_id LIKE 'run_%'
AND run_id NOT IN ('run_2e6990b98cd4', 'run_6b56b8e0571f', 'run_c37097d4b935')
GROUP BY run_id ORDER BY rowid DESC LIMIT 3;
```

## Scorer Analysis (Qwen2.5-7B, 150 episodes)

| Scorer | Targets | Mean Score | States Differentiated | % | Action Verbs |
|--------|---------|------------|----------------------|---|-------------|
| outcome_only | 3962 | 0.054 | 41/1212 | 3.4% | 22 (54%) |
| branch_aware | 3962 | 0.032 | 385/1212 | 31.8% | 314 (82%) |
| hindsight | 3962 | 0.054 | 41/1212 | 3.4% | 22 (54%) |
| **combined** | 3962 | 0.032 | **385/1212** | **31.8%** | **314 (82%)** |

## Cross-Model Comparison

| Metric | Haiku | Qwen2.5-7B |
|--------|-------|-----------|
| Raw success rate | 30.0% | 16.7% |
| OO states differentiated | 143 (13.3%) | 41 (3.4%) |
| Combined states differentiated | 375 (34.8%) | 385 (31.8%) |
| **Differentiation ratio (CB/OO)** | **2.6x** | **9.4x** |
| OO action verb preference | 69% | 54% |
| Combined action verb preference | 80% | 82% |

**Key finding**: The combined scorer's advantage is even larger with a weaker model (9.4x vs 2.6x). With only 16.7% success, outcome-only has almost no signal (differentiates 3.4% of states), while combined still identifies good actions at 31.8% of states using auto-extracted verifier signals from observation diffs.

## Cost

| Experiment | Cost |
|------------|------|
| Haiku (150 episodes, ~3737 API calls) | ~$0.50 (Haiku pricing) |
| Qwen2.5-7B (150 episodes, ~3962 vLLM calls) | ~$2-3 GCP compute |
| Total | < $5 |
