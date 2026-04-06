# CreditLab

Research scaffold for branch-aware hindsight credit assignment experiments under matched budgets.

## Core Claim

CreditLab tests whether branch-aware plus hindsight credit assignment shifts probability mass toward better downstream branches, improving policy quality over outcome-only training under equal collection budgets.

## What Exists Now

- Versioned benchmark manifests for `diagnostic`, `webshop`, `alfworld`, and `stochastic`.
- Real ALFWorld environment adapter (`alfworld_real`) for live benchmark evaluation.
- Manifest-driven env adapters with shared verifier-first state transitions.
- SQLite + Parquet experiment store with run-group, condition, seed, benchmark, and deployment metadata.
- Scored table trainers for `outcome_only`, `branch_aware`, `hindsight`, and `combined`.
- Automatic verifier signal extraction from environment observations.
- Sweep execution for the paper conditions with safety rails.
- Aggregate reporting, budget-parity checks, failure analysis, and paper-asset export.
- GCP GPU lane with NVIDIA L4 support, GCS sync, and bootstrap scripts.
- LLM collection via vLLM (Qwen2.5-3B-Instruct) with OpenAI-compatible API.

## Setup

```bash
pip install -e ".[dev]"   # install creditlab + pytest (editable)
make test                 # verify: all tests should pass
```

Optional dependencies:
```bash
pip install -e ".[paper]"      # matplotlib for publication figures
pip install -e ".[alfworld]"   # real ALFWorld benchmark
pip install -e ".[training]"   # LoRA fine-tuning (transformers, peft, trl)
```

## Main Commands

```bash
creditlab validate-config --config configs/diagnostic_paper_local.yaml
creditlab collect --config configs/diagnostic_smoke.yaml
creditlab sweep --config configs/stochastic_paper_local.yaml
creditlab aggregate --config configs/stochastic_paper_local.yaml --run-group-id <group_id>
creditlab export-results --config configs/stochastic_paper_local.yaml --run-group-id <group_id>
creditlab render-paper-assets --config configs/stochastic_paper_local.yaml --run-group-id <group_id>
creditlab host-check --output artifacts/host_report.json
```

Quick workflow:

```bash
make smoke       # validate config + collect one run (seed=7, local)
make run-paper   # sweep all paper conditions (local)
make clean-data  # archive + wipe runs/ and artifacts/ for a fresh slate
```

## Benchmarks

- `stochastic`: **primary evaluation** — 4 task families with probabilistic transitions (60-90% success probability).
- `diagnostic`: controlled ablation with 5 task families isolating specific failure modes.
- `webshop`: shopping-task structural abstraction for cross-benchmark transfer.
- `alfworld`: embodied task structural abstraction for cross-benchmark transfer.
- `alfworld_real`: live ALFWorld TextWorld environment (requires `pip install alfworld`).

## GCP Path (Recommended)

- Host: `g2-standard-8` with 1x NVIDIA L4 (~$0.70/hr on-demand)
- Zone: `us-central1-a`
- Setup guide: [docs/gcp_gpu_setup.md](docs/gcp_gpu_setup.md)
- Bootstrap scripts: `scripts/gcp/`
- Compose stack: `docker/docker-compose.gcp.yml`

## Safety Rails

- GCP-backed sweep commands fail closed unless `sweep.safety.allow_gcp_execution` is `true`.
- Sweep execution enforces run-count, token, verifier-call, wall-clock, failure-count, and estimated-cost ceilings.
- `vllm_server` retries transient failures with bounded backoff and falls back to the prompted policy.
- Run `creditlab host-check` on the GCP host before enabling cloud configs.

## Repo Layout

```text
benchmarks/     versioned benchmark manifests (diagnostic, webshop, alfworld, stochastic)
configs/        smoke, benchmark, and paper configs
docker/         cpu/gpu images and compose files
docs/           architecture, environment spec, reproducibility guide
paper/          manuscript
scripts/gcp/    GCP host bootstrap and sync helpers
scripts/aws/    AWS host bootstrap (legacy, retained for reproducibility)
src/creditlab/  experiment system
tests/          local verification
```

## Reproducibility

All paper results are reproduced on both local (macOS) and cloud (GCP) infrastructure with exact numerical match. See [docs/reproducibility.md](docs/reproducibility.md) for step-by-step instructions and [docs/environment.md](docs/environment.md) for exact software/hardware versions.
