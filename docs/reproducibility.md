# Reproducing the Paper Results

Step-by-step guide to reproduce all results from "Branch-Aware Hindsight Credit Assignment for Language-Model Agents Under Matched Budgets."

## Prerequisites

- Python 3.11+ (tested on 3.11.15 and 3.13.7)
- ~500MB disk space for artifacts
- No GPU required for local runs
- Optional: GCP account for cloud reproduction (g2-standard-8 with NVIDIA L4)

## Step 1: Clone and Install

```bash
git clone https://github.com/jadenfix/CreditAssignment.git
cd CreditAssignment

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
pip install --no-deps -e ".[dev]"
```

Verify the install:

```bash
creditlab --help
```

## Step 2: Run Tests

```bash
make test
```

Expected: `11 passed`. If any test fails, the environment is not correctly set up.

## Step 3: Smoke Test (< 1 minute)

```bash
make smoke
```

This runs `creditlab validate-config` and `creditlab collect` on a minimal task (1 seed, 4 rollouts). Produces a `run_id` — confirms the pipeline works end-to-end.

## Step 4: Reproduce Stochastic Benchmark (Table 1 — Main Result)

```bash
creditlab sweep --config configs/stochastic_paper_local.yaml
```

**Configuration:** 15 seeds (7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61), 50 rollouts per run, exploration_rate=0.5, max_turns=15, 4 train tasks, 3 eval tasks across 4 stochastic families.

**Runtime:** ~10-20 minutes on a modern laptop.

**Expected output:** A JSON blob with `run_group_id` and aggregate results. Look for:

| Condition | Expected Mean Success |
|-----------|----------------------|
| prompted | ~0.79 |
| outcome_only | ~0.24 |
| branch_aware | ~0.98 |
| hindsight | ~0.70 |
| combined | 1.00 |

**Key verification:** Outcome-only should be *below* the prompted baseline — this is the loop-action bias effect. Combined should be 1.00 across all seeds.

To export paper artifacts:

```bash
creditlab aggregate --config configs/stochastic_paper_local.yaml --run-group-id <GROUP_ID>
creditlab export-results --config configs/stochastic_paper_local.yaml --run-group-id <GROUP_ID>
creditlab render-paper-assets --config configs/stochastic_paper_local.yaml --run-group-id <GROUP_ID>
```

This produces CSVs, SVGs, and markdown in `artifacts/paper/<GROUP_ID>/`.

## Step 5: Reproduce Diagnostic Results (Table 2, Table 3)

```bash
creditlab sweep --config configs/diagnostic_paper_local.yaml
```

**Configuration:** 7 seeds (7, 11, 19, 23, 31, 37, 41), 50 rollouts per run, exploration_rate=0.5, max_turns=10, 5 train tasks, 5 eval tasks across 5 families.

**Runtime:** ~5-10 minutes.

**Expected output:**

| Condition | Expected Mean Success |
|-----------|----------------------|
| prompted | ~0.73 |
| outcome_only | ~0.51 |
| branch_aware | ~0.97 |
| hindsight | ~0.86 |
| combined | 1.00 |

**Note:** Exact values may differ slightly due to the stateful RNG, but the ranking (combined > branch_aware > hindsight >> outcome_only) should be consistent.

## Step 6: Reproduce WebShop Results (Table 2)

```bash
creditlab sweep --config configs/webshop_paper_local.yaml
```

**Configuration:** 7 seeds, 50 rollouts, exploration_rate=0.5. 2 train tasks, 1 eval task. Actions are abstracted (search_target, open_target, open_distractor) for cross-task transfer.

**Runtime:** ~3-5 minutes.

**Expected output:**

| Condition | Expected Mean Success |
|-----------|----------------------|
| prompted | ~0.77 |
| outcome_only | ~0.71 |
| branch_aware | 1.00 |
| hindsight | ~0.71 |
| combined | 1.00 |

## Step 7: Reproduce ALFWorld Results (Table 2)

```bash
creditlab sweep --config configs/alfworld_paper_local.yaml
```

**Configuration:** 7 seeds, 50 rollouts, exploration_rate=0.5. 4 train tasks (pick_place, clean_place, cool_place, examine), 3 eval tasks (heat_food, clean_place, pick_place_far). Abstract actions shared across all task types.

**Runtime:** ~5-10 minutes.

**Expected output:**

| Condition | Expected Mean Success |
|-----------|----------------------|
| prompted | ~0.52 |
| outcome_only | 0.00 |
| branch_aware | 1.00 |
| hindsight | 0.00 |
| combined | 1.00 |

ALFWorld produces the most extreme separation: outcome_only and hindsight achieve 0.00, while branch_aware and combined achieve 1.00.

## Step 8: Reproduce Exploration Rate Ablation (Table 8)

```bash
for rate in 0.3 0.7; do
  sed "s/exploration_rate: 0.5/exploration_rate: $rate/" \
    configs/diagnostic_paper_local.yaml > /tmp/diag_er_${rate}.yaml
  creditlab sweep --config /tmp/diag_er_${rate}.yaml
done
```

The ranking (combined >= branch_aware >= hindsight >> outcome_only) should hold at all three rates.

## Step 9: Generate Publication Figures

```bash
pip install matplotlib  # or: pip install -e ".[paper]"
python scripts/render_paper_figures.py --output artifacts/paper/figures
```

This generates three figures in both PDF (300 DPI) and SVG:
- `cross_benchmark_success.pdf` — Grouped bar chart with bootstrap 95% CIs
- `per_family_heatmap.pdf` — Conditions × families heatmap (diagnostic)
- `ablation_exploration_rate.pdf` — Sensitivity line chart

## Step 10: Inspect Trained Policies

After any sweep, inspect the learned score tables:

```bash
python3 -c "
import json, glob
for f in sorted(glob.glob('artifacts/policies/*.json'))[-5:]:
    model = json.loads(open(f).read())
    print(f'{f}:')
    for state, actions in model.items():
        best = max(actions, key=actions.get)
        print(f'  {state[:12]}: {actions} -> {best}')
    print()
"
```

You should see:
- **outcome_only** preferring stall actions (`rattle_gate`, `browse`, `review_notes`) at some states
- **branch_aware** and **combined** consistently preferring pivotal actions

## Step 11: Query the Run Database

All episodes, transitions, metrics, and verifier events are stored in `runs/creditlab.sqlite`:

```bash
# List all runs in a sweep group
sqlite3 runs/creditlab.sqlite \
  "SELECT run_id, condition_name, seed FROM runs WHERE run_group_id='<GROUP_ID>' ORDER BY condition_name, seed;"

# Count episodes per run
sqlite3 runs/creditlab.sqlite \
  "SELECT r.condition_name, r.seed, COUNT(e.episode_id) FROM runs r JOIN episodes e ON r.run_id = e.run_id WHERE r.run_group_id='<GROUP_ID>' GROUP BY r.condition_name, r.seed;"

# Check partial credit scores
sqlite3 runs/creditlab.sqlite \
  "SELECT r.condition_name, AVG(m.value) FROM runs r JOIN metrics m ON r.run_id = m.run_id WHERE r.run_group_id='<GROUP_ID>' AND m.name='partial_credit' GROUP BY r.condition_name;"
```

## Step 12: Reproduce on GCP (Cloud Verification)

For cloud reproduction matching the paper's GCP results:

```bash
# 1. Provision VM
gcloud compute instances create creditlab-gpu \
  --machine-type=g2-standard-8 \
  --accelerator=type=nvidia-l4,count=1 \
  --image-family=ubuntu-2204-lts \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-ssd \
  --zone=us-central1-a

# 2. Copy repo (excluding generated files)
gcloud compute config-ssh
rsync -av \
  --exclude='.venv/' --exclude='__pycache__/' --exclude='*.pyc' \
  --exclude='.pytest_cache/' --exclude='*.egg-info/' --exclude='.git/' \
  --exclude='artifacts/runs/' --exclude='artifacts/policies/' --exclude='artifacts/paper/' \
  . creditlab-gpu.us-central1-a.<PROJECT_ID>:/workspace/creditlab/

# 3. Bootstrap the VM
gcloud compute ssh creditlab-gpu --command "bash /workspace/creditlab/scripts/gcp/bootstrap_ubuntu22_gpu_gcp.sh"
gcloud compute ssh creditlab-gpu --command "
  sudo apt-get install -y python3.11 python3.11-venv
  python3.11 -m venv /workspace/venv
  cd /workspace/creditlab
  /workspace/venv/bin/pip install -r requirements-lock.txt
  /workspace/venv/bin/pip install --no-deps .
"

# 4. Validate GPU setup
gcloud compute ssh creditlab-gpu --command "/workspace/venv/bin/creditlab host-check"

# 5. Run stochastic benchmark (main result)
gcloud compute ssh creditlab-gpu --command "
  cd /workspace/creditlab
  nohup /workspace/venv/bin/creditlab sweep --config configs/stochastic_paper_gcp.yaml > /tmp/stochastic_sweep.log 2>&1 &
"
# Check progress:
gcloud compute ssh creditlab-gpu --command "tail -5 /tmp/stochastic_sweep.log"

# 6. Run supporting benchmarks
gcloud compute ssh creditlab-gpu --command "
  cd /workspace/creditlab
  /workspace/venv/bin/creditlab sweep --config configs/diagnostic_paper_gcp.yaml
  /workspace/venv/bin/creditlab sweep --config configs/webshop_paper_gcp.yaml
  /workspace/venv/bin/creditlab sweep --config configs/alfworld_paper_gcp.yaml
"
```

**Tip:** For long-running sweeps, use `nohup` to prevent SSH disconnects from killing the process.

## Step 13: Reproduce LLM Collection (Table 4, GCP Only)

This requires the GCP VM with GPU for vLLM inference:

```bash
# 1. Start vLLM server on the GCP VM
gcloud compute ssh creditlab-gpu --command "
  /workspace/venv/bin/python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-3B-Instruct \
    --max-model-len 2048 \
    --dtype float16 \
    --port 8000 &
"

# 2. Wait for the server to be ready (~60 seconds)
gcloud compute ssh creditlab-gpu --command "
  until curl -s http://localhost:8000/v1/models > /dev/null 2>&1; do
    echo 'Waiting for vLLM...'
    sleep 5
  done
  echo 'vLLM ready'
"

# 3. Run LLM collection sweep
gcloud compute ssh creditlab-gpu --command "
  cd /workspace/creditlab
  /workspace/venv/bin/creditlab sweep --config configs/diagnostic_llm_gcp.yaml
"
```

**Expected:** All trained conditions achieve 1.000 success; LLM baseline achieves ~0.833.

## Step 14: Run Weight Ablation (Optional)

```bash
python scripts/run_weight_ablation.py
```

This runs a grid search over scorer weights (27 configurations × 3 seeds). The qualitative ranking (combined > branch_aware > hindsight >> outcome_only) should hold across all weight configurations.

## Verifying Reproducibility

After running both local and GCP sweeps, compare results:

```bash
# Local stochastic results
sqlite3 runs/creditlab.sqlite \
  "SELECT condition_name, AVG(success_rate) FROM run_summaries WHERE run_group_id='<LOCAL_GROUP_ID>' GROUP BY condition_name;"

# GCP stochastic results (copy the GCP database locally first)
gcloud compute scp creditlab-gpu:/workspace/creditlab/runs/creditlab.sqlite /tmp/gcp_creditlab.sqlite
sqlite3 /tmp/gcp_creditlab.sqlite \
  "SELECT condition_name, AVG(success_rate) FROM run_summaries WHERE run_group_id='<GCP_GROUP_ID>' GROUP BY condition_name;"
```

Results should match exactly across platforms for heuristic experiments (same seeds, same manifest, same RNG). LLM experiments are statistically reproducible (same ranking) but not bit-reproducible due to vLLM inference stochasticity.

## File Layout

After running all steps, your artifact directory should contain:

```
artifacts/
  runs/<run_id>/             # Per-run: report.json, metrics.jsonl
  policies/<scorer>_*.json   # Trained score tables
  paper/<group_id>/          # CSVs, SVGs, markdown per sweep
  paper/figures/             # Publication PDFs + SVGs
    cross_benchmark_success.pdf
    per_family_heatmap.pdf
    ablation_exploration_rate.pdf
runs/
  creditlab.sqlite           # All episodes, transitions, metrics
scripts/
  render_paper_figures.py    # Publication figure generator
  run_weight_ablation.py     # Weight sensitivity analysis
```

## Key Configuration Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Seeds (stochastic) | 15 primes: 7–61 | Statistical power for main result |
| Seeds (deterministic) | 7 primes: 7–41 | Sufficient for binary outcomes |
| Rollout count | 50 | State-action coverage |
| Max turns (stochastic) | 15 | Extra turns for stochastic retries |
| Max turns (deterministic) | 10 | Accommodates deepest tasks |
| Exploration rate | 0.5 | Balances oracle following and exploration |
| Fallback policy | Random | Prevents oracle inflation at unseen states |

## Experiment Run Groups (Paper)

All reported results trace to specific run group IDs:

| Table | Experiment | Run Group / Run ID | Platform |
|-------|------------|-----------|----------|
| Table 1 | Stochastic (controlled) | `group_2cfb2caf7e7d` | Local |
| Table 1 | Stochastic (GCP reproduction) | `group_d92fc9971c06` | GCP |
| Table 2 | Diagnostic | `group_f410ed95cc83` | Local |
| Table 2 | Diagnostic (GCP) | `group_a0acb246112e` | GCP |
| Table 2 | WebShop | `group_b25fdde347d6` | Local |
| Table 2 | ALFWorld-manifest | `group_24296be59b90` | Local |
| Table 4 | LLM collection (manifest) | `group_e904a833fee4` | GCP |
| Tables 5-7 | Real ALFWorld (Haiku seed 7) | `run_2e6990b98cd4` | GCP |
| Tables 5-7 | Real ALFWorld (Haiku seed 11) | `run_6b56b8e0571f` | GCP |
| Tables 5-7 | Real ALFWorld (Haiku seed 19) | `run_c37097d4b935` | GCP |
| Tables 5-7 | Real ALFWorld (Qwen-7B seed 7) | `run_00767781c915` | GCP |
| Tables 5-7 | Real ALFWorld (Qwen-7B seed 11) | `run_22c894f94f13` | GCP |
| Tables 5-7 | Real ALFWorld (Qwen-7B seed 19) | `run_74a5173ab1c8` | GCP |
| Table 8 | Ablation ε=0.3 | per-run | Local |
| Table 8 | Ablation ε=0.7 | per-run | Local |

## Troubleshooting

**"deployment.platform=gcp_gpu is disabled"** — You're using a cloud config locally. Use the `*_local.yaml` variant.

**Sweep halts with "max_failed_runs"** — Check that the benchmark manifest has `start` as the initial state name for all tasks. Custom node names require setting `initial_state` in the YAML.

**All trained conditions score 0.0** — Check state hash overlap between train and eval. If overlap is 0, the abstract_fields in your manifest need alignment.

**Identical scores across all trained conditions** — Check that the collection policy is non-deterministic (stateful RNG, not per-step re-seeding). Inspect trained score tables — if each state has only one action, the scorers cannot differentiate.

**vLLM fails to start** — Ensure vLLM 0.6.6 is installed (`pip install vllm==0.6.6`). Later versions may have breaking changes. The Qwen2.5-3B-Instruct model requires ~6GB VRAM.

**Local vs GCP results differ** — Verify both use the same config seeds and manifest files. Differences indicate a code or dependency version mismatch. Check `pip freeze` on both platforms against `requirements-lock.txt`.
