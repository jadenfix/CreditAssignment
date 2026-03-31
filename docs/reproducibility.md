# Reproducing the Paper Results

Step-by-step guide to reproduce all results from "Branch-Aware Hindsight Credit Assignment Under Matched Budgets."

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

This runs `creditlab validate-config` and `creditlab collect` on a minimal diagnostic task (1 seed, 4 rollouts). Produces a `run_id` — confirms the pipeline works end-to-end.

## Step 4: Reproduce Diagnostic Results (Table 1, Table 2)

```bash
creditlab sweep --config configs/diagnostic_paper_local.yaml
```

**Configuration:** 7 seeds (7, 11, 19, 23, 31, 37, 41), 50 rollouts per run, exploration_rate=0.5, max_turns=10, 5 train tasks, 5 eval tasks across 5 families.

**Runtime:** ~5-10 minutes on a modern laptop.

**Expected output:** A JSON blob with `run_group_id` and aggregate results. Look for:

| Condition | Expected Mean Success |
|-----------|----------------------|
| prompted | ~0.73 |
| outcome_only | ~0.51 |
| branch_aware | ~0.97 |
| hindsight | ~0.86 |
| combined | 1.00 |

**Note:** Exact values may differ slightly due to the stateful RNG, but the ranking (combined > branch_aware > hindsight >> outcome_only) should be consistent.

To export paper artifacts:

```bash
creditlab aggregate --config configs/diagnostic_paper_local.yaml --run-group-id <GROUP_ID>
creditlab export-results --config configs/diagnostic_paper_local.yaml --run-group-id <GROUP_ID>
creditlab render-paper-assets --config configs/diagnostic_paper_local.yaml --run-group-id <GROUP_ID>
```

This produces CSVs, SVGs, and markdown in `artifacts/paper/<GROUP_ID>/`.

## Step 5: Reproduce WebShop Results (Table 3)

```bash
creditlab sweep --config configs/webshop_paper_local.yaml
```

**Configuration:** Same 7 seeds, 50 rollouts, exploration_rate=0.5. 2 train tasks (red_mug, blue_bottle), 1 eval task (travel_mug). Actions are abstracted (search_target, open_target, open_distractor) for cross-task transfer.

**Runtime:** ~3-5 minutes.

**Expected output:**

| Condition | Expected Mean Success |
|-----------|----------------------|
| prompted | ~0.77 |
| outcome_only | ~0.71 |
| branch_aware | 1.00 |
| hindsight | ~0.71 |
| combined | 1.00 |

## Step 6: Reproduce ALFWorld Results (Table 4)

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

ALFWorld produces the most dramatic separation: outcome_only and hindsight achieve complete failure (0.00), while branch_aware and combined achieve perfect success.

## Step 7: Generate Publication Figures

```bash
pip install matplotlib  # or: pip install -e ".[paper]"
python scripts/render_paper_figures.py --output artifacts/paper/figures
```

This generates three figures in both PDF (300 DPI) and SVG:
- `cross_benchmark_success.pdf` — Grouped bar chart with bootstrap 95% CIs
- `per_family_heatmap.pdf` — Conditions × families heatmap (diagnostic)
- `ablation_exploration_rate.pdf` — Sensitivity line chart

The script also prints bootstrap 95% confidence intervals for all conditions.

## Step 8: Reproduce Exploration Rate Ablation (Table 6)

Create temporary configs with different exploration rates:

```bash
for rate in 0.3 0.7; do
  sed "s/exploration_rate: 0.5/exploration_rate: $rate/" \
    configs/diagnostic_paper_local.yaml > /tmp/diag_er_${rate}.yaml
  creditlab sweep --config /tmp/diag_er_${rate}.yaml
done
```

The ranking (combined >= branch_aware >= hindsight >> outcome_only) should hold at all three rates.

## Step 9: Inspect Trained Policies

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

This shows what action each scorer's model prefers at each state. You should see:
- **outcome_only** preferring stall actions (e.g., `rattle_gate`, `browse`, `review_notes`) at some states
- **branch_aware** and **combined** consistently preferring pivotal actions

## Step 10: Query the Run Database

All episodes, transitions, metrics, and verifier events are stored in `runs/creditlab.sqlite`:

```bash
# List all runs in a sweep group
sqlite3 runs/creditlab.sqlite \
  "SELECT run_id, condition_name, seed FROM runs WHERE run_group_id='<GROUP_ID>' ORDER BY condition_name, seed;"

# Count episodes per run
sqlite3 runs/creditlab.sqlite \
  "SELECT r.condition_name, r.seed, COUNT(e.episode_id) FROM runs r JOIN episodes e ON r.run_id = e.run_id WHERE r.run_group_id='<GROUP_ID>' GROUP BY r.condition_name, r.seed;"
```

## Step 11: Reproduce on GCP (Optional)

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

# 2. Copy repo (excluding venv and generated files)
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

# 5. Run all three sweeps
gcloud compute ssh creditlab-gpu --command "
  cd /workspace/creditlab
  /workspace/venv/bin/creditlab sweep --config configs/diagnostic_paper_gcp.yaml
  /workspace/venv/bin/creditlab sweep --config configs/webshop_paper_gcp.yaml
  /workspace/venv/bin/creditlab sweep --config configs/alfworld_paper_gcp.yaml
"
```

**Tip:** For long-running sweeps, use `nohup` to prevent SSH disconnects from killing the process:

```bash
gcloud compute scp scripts/run_all_sweeps.sh creditlab-gpu:/tmp/
gcloud compute ssh creditlab-gpu --command "nohup bash /tmp/run_all_sweeps.sh > /tmp/all_sweeps.log 2>&1 &"
# Check progress:
gcloud compute ssh creditlab-gpu --command "tail -5 /tmp/all_sweeps.log"
```

## File Layout

After running all steps, your artifact directory should contain:

```
artifacts/
  runs/<run_id>/             # Per-run: report.json, metrics.jsonl
  policies/<scorer>_*.json   # Trained score tables
  paper/<group_id>/          # CSVs, SVGs, markdown per sweep
  paper/figures/             # Publication PDFs + SVGs (from render_paper_figures.py)
    cross_benchmark_success.pdf
    per_family_heatmap.pdf
    ablation_exploration_rate.pdf
runs/
  creditlab.sqlite           # All episodes, transitions, metrics
scripts/
  render_paper_figures.py    # Publication figure generator
```

## Key Configuration Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| seeds | 7, 11, 19, 23, 31, 37, 41 | 7 seeds for statistical power |
| rollout_count | 50 | Sufficient state-action coverage |
| max_turns | 10 | Accommodates deepest tasks (multi_gate, cascade) |
| exploration_rate | 0.5 | Balances oracle following and exploration |
| fallback policy | random | Prevents oracle inflation at unseen states |
| scoring conditions | 6 | prompted, outcome_only, branch_aware, hindsight, combined, combined_matched_budget |

## Troubleshooting

**"deployment.platform=aws_gpu is disabled"** — You're using a cloud config locally. Use the `*_local.yaml` variant instead.

**Sweep halts with "max_failed_runs"** — Check that the benchmark manifest has `start` as the initial state name for all tasks. Custom node names require setting `initial_state` in the YAML.

**All trained conditions score 0.0** — Check state hash overlap between train and eval. Run the state-hash comparison from Step 8. If overlap is 0, the abstract_fields in your manifest need alignment.

**Identical scores across all trained conditions** — Check that the collection policy is non-deterministic (stateful RNG, not per-step re-seeding). Inspect the trained score tables — if each state has only one action, the scorers cannot differentiate.
