# Branch-Aware Hindsight Credit Assignment for Language-Model Agents

**Paper**: [paper/main.pdf](paper/main.pdf)

We identify *loop-action bias* as a mechanistic failure mode of outcome-only credit assignment for LLM agents and propose branch-aware and hindsight scorers that leverage per-transition verifier signals. Validated on controlled stochastic benchmarks and the real ALFWorld environment with two language models using automatically extracted verifier signals.

## Key Results

| Experiment | Outcome-only | Branch-aware | Combined |
|------------|-------------|-------------|----------|
| Stochastic benchmark (15 seeds) | 0.240 | 0.979 | **1.000** |
| Real ALFWorld, Haiku (scorer diff.) | 13.3% | **34.8%** | = branch-aware |
| Real ALFWorld, Qwen-7B (scorer diff.) | 3.4% | **31.8%** | = branch-aware |

## Reproducing the Results

### Prerequisites

- Python 3.11+
- No GPU required for controlled benchmarks (local runs)
- GCP account with `g2-standard-8` instance for real ALFWorld experiments (NVIDIA L4, ~$0.70/hr)
- Anthropic API key for Claude Haiku collection (or use the included Qwen-7B path via vLLM)

### Step 1: Install

```bash
git clone https://github.com/jadenfix/CreditAssignment.git
cd CreditAssignment

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
pip install -e ".[dev]"
```

Verify:
```bash
creditlab --help
make test        # all tests should pass
```

### Step 2: Controlled Benchmarks (Tables 1-3, local, ~30 min)

**Stochastic benchmark** (Table 1, main controlled result):
```bash
creditlab sweep --config configs/stochastic_paper_local.yaml
```
Expected: outcome-only ~0.24, combined = 1.00 across 15 seeds.

**Diagnostic benchmark** (Tables 2-3):
```bash
creditlab sweep --config configs/diagnostic_paper_local.yaml
```

**WebShop and ALFWorld-manifest** (Table 2, supporting):
```bash
creditlab sweep --config configs/webshop_paper_local.yaml
creditlab sweep --config configs/alfworld_paper_local.yaml
```

### Step 3: Exploration Rate Ablation (Table 6, local, ~45 min)

```bash
for rate in 0.3 0.5 0.7; do
  sed "s/exploration_rate: 0.5/exploration_rate: $rate/" \
    configs/diagnostic_paper_local.yaml > /tmp/diag_er_${rate}.yaml
  creditlab sweep --config /tmp/diag_er_${rate}.yaml
done
```

### Step 4: Real ALFWorld Experiment (Tables 4-5, GCP)

This requires a GCP VM with NVIDIA L4 GPU and either an Anthropic API key or self-hosted vLLM.

**4a. Provision GCP VM:**
```bash
gcloud compute instances create creditlab-gpu \
  --machine-type=g2-standard-8 \
  --accelerator=type=nvidia-l4,count=1 \
  --image-family=ubuntu-2204-lts \
  --boot-disk-size=200GB --boot-disk-type=pd-ssd \
  --zone=us-central1-a
```

**4b. Setup on the VM:**
```bash
# Copy repo to VM
gcloud compute config-ssh
rsync -av --exclude='.venv/' --exclude='__pycache__/' --exclude='.git/' \
  . creditlab-gpu.us-central1-a.<PROJECT>:/workspace/creditlab/

# SSH in and install
gcloud compute ssh creditlab-gpu --zone=us-central1-a
cd /workspace/creditlab
python3.11 -m venv /workspace/venv
/workspace/venv/bin/pip install -r requirements-lock.txt
/workspace/venv/bin/pip install --no-deps .
/workspace/venv/bin/pip install alfworld==0.4.2

# Download ALFWorld data
alfworld-download
```

**4c. Collect with Claude Haiku** (requires Anthropic API key, ~$0.50):
```bash
export ANTHROPIC_API_KEY=<your-key>
export ALFWORLD_DATA=~/.cache/alfworld

for seed in 7 11 19; do
  python -u scripts/collect_alfworld_real.py \
    --backend anthropic --model claude-haiku-4-5-20251001 \
    --num-episodes 50 --max-turns 30 --seed $seed
done
```

**4d. Collect with Qwen2.5-7B** (self-hosted, no API key needed):
```bash
# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct --max-model-len 2048 --dtype float16 --port 8000 &

# Wait for server, then collect
for seed in 7 11 19; do
  python -u scripts/collect_alfworld_real.py \
    --backend vllm --model Qwen/Qwen2.5-7B-Instruct \
    --num-episodes 50 --max-turns 30 --seed $seed
done
```

**4e. Analyze results:**
```bash
python -c "
from creditlab.store.sqlite_store import SqliteTrajectoryStore
from creditlab.verifiers.scorers import build_training_targets
from collections import defaultdict

store = SqliteTrajectoryStore(sqlite_path='runs/creditlab.sqlite', artifact_root='artifacts')
# Replace with your actual run IDs from the collection output
run_ids = ['run_XXXXX', 'run_YYYYY', 'run_ZZZZZ']
episodes = []
for rid in run_ids:
    episodes.extend(store.load_episodes(rid))

for scorer in ['outcome_only', 'branch_aware']:
    targets = build_training_targets(episodes, scorer)
    prefs = defaultdict(lambda: defaultdict(list))
    for t in targets:
        prefs[t.state_hash][t.action].append(t.score)
    diff = sum(1 for sh, acts in prefs.items() if len(acts) > 1
               and max(sum(s)/len(s) for s in acts.values()) - min(sum(s)/len(s) for s in acts.values()) > 0.01)
    print(f'{scorer}: {diff}/{len(prefs)} states differentiated ({100*diff/len(prefs):.1f}%)')
"
```

### Step 5: Generate Figures

```bash
pip install matplotlib
python scripts/generate_paper_figures.py
```
Output: `paper/figures/*.pdf` and `paper/figures/*.png`

### Step 6: Compile Paper

```bash
# Install tectonic (standalone LaTeX compiler, no root needed)
brew install tectonic   # macOS
# or: cargo install tectonic  # any platform

cd paper && tectonic main.tex
```
Output: `paper/main.pdf`

## Run IDs for Paper Results

All results are traceable to specific run identifiers in `runs/creditlab.sqlite`:

| Table | Experiment | Run ID / Group |
|-------|------------|----------------|
| 1 | Stochastic (local) | `group_2cfb2caf7e7d` |
| 1 | Stochastic (GCP reproduction) | `group_d92fc9971c06` |
| 2 | Diagnostic | `group_f410ed95cc83` |
| 2 | WebShop | `group_b25fdde347d6` |
| 2 | ALFWorld-manifest | `group_24296be59b90` |
| 4 | Real ALFWorld, Haiku seed 7 | `run_2e6990b98cd4` |
| 4 | Real ALFWorld, Haiku seed 11 | `run_6b56b8e0571f` |
| 4 | Real ALFWorld, Haiku seed 19 | `run_c37097d4b935` |
| 4 | Real ALFWorld, Qwen-7B seed 7 | `run_00767781c915` |
| 4 | Real ALFWorld, Qwen-7B seed 11 | `run_22c894f94f13` |
| 4 | Real ALFWorld, Qwen-7B seed 19 | `run_74a5173ab1c8` |
| 6 | Ablation, epsilon=0.3 | `group_51f1c6b520ac` |
| 6 | Ablation, epsilon=0.5 | `group_2cd95b33ebf2` |
| 6 | Ablation, epsilon=0.7 | `group_8206d1e63749` |

## Environment

| Component | Version |
|-----------|---------|
| Python (local) | 3.13.7 |
| Python (GCP) | 3.11.15 |
| GCP instance | g2-standard-8, NVIDIA L4 (24 GB) |
| ALFWorld | 0.4.2 |
| vLLM | 0.6.6 |
| Qwen model | Qwen/Qwen2.5-7B-Instruct |
| Anthropic model | claude-haiku-4-5-20251001 |

See [docs/environment.md](docs/environment.md) for full dependency versions and hardware specs.

## Repository Structure

```
paper/              Manuscript (LaTeX + PDF + figures)
src/creditlab/      Experiment system
  envs/             Environment adapters (manifest + real ALFWorld)
  verifiers/        Scorer implementations
  policies/         Collection policies (prompted, vLLM, table)
  trainers/         Score table trainers
  analysis/         Sweep orchestration, reporting
benchmarks/         Versioned task manifests (stochastic, diagnostic, webshop, alfworld)
configs/            Experiment configurations
scripts/            Collection and figure generation scripts
docs/               Environment spec, reproducibility guide, experiment log
tests/              Test suite
```

## License

MIT. See [LICENSE](LICENSE).

## Citation

```bibtex
@article{fix2026branch,
  title={Branch-Aware Hindsight Credit Assignment for Language-Model Agents Under Matched Budgets},
  author={Fix, Jaden},
  year={2026}
}
```
