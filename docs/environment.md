# Environment Specification

Exact software and hardware versions used to produce all results in the paper.
All experiments were run on both platforms to verify cross-platform reproducibility.

## Local Development Machine

| Component | Version |
|-----------|---------|
| OS | macOS 15.7.3 (Darwin 24.6.0 arm64) |
| Python | 3.13.7 (local venv) |
| pip | 25.2 |

## GCP Compute Instance

| Component | Version |
|-----------|---------|
| Instance type | g2-standard-8 |
| OS | Ubuntu 22.04.5 LTS |
| Kernel | 6.8.0-1052-gcp |
| GPU | NVIDIA L4 (23,034 MiB VRAM) |
| GPU Driver | 570.211.01 |
| CUDA (PyTorch) | 12.4 |
| Compute Capability | 8.9 |
| Python | 3.11.15 |
| Zone | us-central1-a |
| Project | creditlab-491502 |

## Python Dependencies (pinned)

All experiments use these exact versions, recorded in `requirements-lock.txt`:

| Package | Version | Role |
|---------|---------|------|
| pydantic | 2.12.5 | Config validation, data models |
| sqlalchemy | 2.0.48 | SQLite trajectory store |
| pyarrow | 21.0.0 | Data serialization |
| polars | 1.39.3 | Data analysis |
| typer | 0.24.1 | CLI framework |
| PyYAML | 6.0.3 | Benchmark manifest parsing |
| pytest | 8.4.2 | Testing (dev only) |
| matplotlib | 3.10.1 | Publication figures (paper only) |

## LLM Inference Stack (GCP only)

| Component | Version | Notes |
|-----------|---------|-------|
| vLLM | 0.6.6 | OpenAI-compatible API server |
| PyTorch | 2.5.1+cu124 | GPU inference backend |
| Model | Qwen/Qwen2.5-3B-Instruct | 3B parameter, FP16, ~6GB VRAM |
| Max model length | 2,048 tokens | Sufficient for task prompts |
| Temperature | 0.7 | For trajectory diversity |
| Endpoint | http://localhost:8000/v1/chat/completions | OpenAI chat format |

## Docker Images (for containerized runs)

| Image | Base | Contents |
|-------|------|----------|
| creditlab-gpu:latest | nvidia/cuda:12.3.2-runtime-ubuntu22.04 | CreditLab + dependencies |
| creditlab-cpu:latest | python:3.11-slim | CreditLab (analysis only) |

## Reproducibility Checksums

To verify you have the exact same code:

```bash
# Check the git commit used for paper results
git log --oneline -1

# Verify dependency versions match
pip show pydantic sqlalchemy pyarrow polars typer pyyaml pytest | grep -E "^(Name|Version)"

# Verify benchmark manifests are unchanged
sha256sum benchmarks/stochastic/v1/tasks.yaml
sha256sum benchmarks/diagnostic/v1/tasks.yaml
sha256sum benchmarks/webshop/v1/tasks.yaml
sha256sum benchmarks/alfworld/v1/tasks.yaml
```

## Experiment Run Groups

All reported results are traceable to specific run group IDs in `runs/creditlab.sqlite`:

| Experiment | Run Group | Seeds | Rollouts | Platform | Key Result |
|------------|-----------|-------|----------|----------|------------|
| **Stochastic (main result)** | group_2cfb2caf7e7d | 15 | 50 | Local | combined=1.00, branch_aware=0.98, outcome_only=0.24 |
| Stochastic (GCP reproduction) | group_d92fc9971c06 | 15 | 50 | GCP | Exact match with local |
| Diagnostic (deterministic) | group_f410ed95cc83 | 7 | 50 | Local | combined=1.00, branch_aware=0.97, outcome_only=0.51 |
| Diagnostic (GCP) | group_a0acb246112e | 7 | 50 | GCP | Exact match with local |
| WebShop (deterministic) | group_b25fdde347d6 | 7 | 50 | Local | combined=1.00, branch_aware=1.00, outcome_only=0.71 |
| ALFWorld (deterministic) | group_24296be59b90 | 7 | 50 | Local | combined=1.00, branch_aware=1.00, outcome_only=0.00 |
| LLM collection (Qwen2.5-3B) | group_e904a833fee4 | 3 | 20 | GCP | All trained=1.00, LLM baseline=0.83 |
| Exploration ablation ε=0.3 | per-run | 3 | 50 | Local | Ranking stable |
| Exploration ablation ε=0.7 | per-run | 3 | 50 | Local | Ranking stable |

## Randomness and Reproducibility

- **Collection policy RNG**: Seeded once per run (not per step). Different rollouts within the same run produce independent trajectories.
- **Table policy RNG**: Seeded once per policy instance for random fallback at unseen states.
- **Sweep seeds**: Explicit in config files. Each seed produces a complete train→score→eval pipeline.
- **vLLM inference**: Temperature=0.7 introduces stochasticity. LLM experiments are not bit-reproducible across runs but are statistically reproducible (same ranking across seeds).

## Cost

| Experiment | Approx. cost |
|------------|-------------|
| Full 3-benchmark sweep (heuristic, 7 seeds) | ~$0.50/hr × 0.5hr = $0.25 |
| LLM sweep (3 seeds, 20 rollouts) | ~$0.70/hr × 0.3hr = $0.21 |
| Weight ablation (27 configs × 3 seeds) | ~$0.70/hr × 2hr = $1.40 |
| Total GCP cost for all paper experiments | < $5.00 |
