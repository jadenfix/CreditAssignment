# CreditLab

Cornerstone research repo for branch-aware hindsight credit assignment experiments under matched budgets.

## Core Claim

CreditLab is built to test whether branch-aware plus hindsight credit shifts probability mass toward better downstream branches, improves search quality, and improves downstream task success over outcome-only training under equal collection budgets.

## What Exists Now

- Versioned benchmark manifests for `diagnostic`, `webshop`, `alfworld`, `livecodebench`, and `swebench_lite`.
- Manifest-driven env adapters with shared verifier-first state transitions.
- SQLite + Parquet experiment store with run-group, condition, seed, benchmark, and deployment metadata.
- Scored table trainers for `outcome_only`, `branch_aware`, `hindsight`, and `combined`.
- Sweep execution for the six paper conditions.
- Aggregate reporting, budget-parity checks, failure analysis, and paper-asset export.
- AWS GPU Docker lane with host-check tooling, bootstrap scripts, CPU/GPU Dockerfiles, and compose wiring.

## Setup

```bash
pip install -e ".[dev]"   # install creditlab + pytest (editable)
make test                 # verify: 11/11 tests should pass
```

Or use `./setup.sh` for a guided install with Python version check.

## Main Commands

```bash
creditlab validate-config --config configs/diagnostic_paper.yaml
creditlab collect --config configs/diagnostic_smoke.yaml
creditlab train --config configs/diagnostic_outcome_only.yaml --scorer outcome_only
creditlab replay --run-id <run_id> --config configs/diagnostic_smoke.yaml --scorer branch_aware
creditlab report --run-id <run_id> --config configs/diagnostic_smoke.yaml
creditlab sweep --config configs/diagnostic_paper.yaml
creditlab aggregate --config configs/diagnostic_paper.yaml --run-group-id <group_id>
creditlab export-results --config configs/diagnostic_paper.yaml --run-group-id <group_id>
creditlab render-paper-assets --config configs/diagnostic_paper.yaml --run-group-id <group_id>
creditlab host-check --output artifacts/host_report.json
```

Quick workflow:

```bash
make smoke       # validate config + collect one run (seed=7, local)
make run-paper   # sweep all 6 paper conditions (local, diagnostic)
make clean-data  # archive + wipe runs/ and artifacts/ for a fresh slate
```

## Benchmarks

- `diagnostic`: explicit pivotal, dead-end, enabling, and recovery tasks for method debugging.
- `webshop`: shopping-task benchmark manifest for branchy long-horizon decisions.
- `alfworld`: text embodied tasks for navigation, enabling steps, and recovery.
- `livecodebench`: first evaluation-only code transfer benchmark.
- `swebench_lite`: heavier code benchmark after inner-loop results are stable.

## AWS Path

- Recommended host: `Ubuntu 22.04` on `g6e.4xlarge`
- Budget fallback: `g6.4xlarge`
- Escalation only if needed: `p5.4xlarge`
- Setup guide: [docs/aws_gpu_setup.md](/Users/jadenfix/RLLL/docs/aws_gpu_setup.md)
- Bootstrap scripts: `scripts/aws/`
- Compose stack: `docker/docker-compose.aws.yml`

## Safety Rails

- AWS-backed `collect`, `evaluate`, and `sweep` commands fail closed unless `sweep.safety.allow_aws_execution` is explicitly set to `true` in the config you are using.
- Sweep execution enforces run-count, token, verifier-call, wall-clock, failure-count, and estimated-cost ceilings before launching the next run.
- `vllm_server` retries transient request failures with bounded backoff and falls back to the local prompted policy if the server stays unhealthy or returns an invalid action.
- Run `uv run creditlab host-check` on the target AWS host before enabling the AWS configs; the default paper configs require a healthy Docker + NVIDIA + CUDA container probe.
- Keep the paper configs conservative first, then raise ceilings intentionally after a successful smoke sweep on the target machine.

## Repo Layout

```text
benchmarks/     versioned benchmark manifests
configs/        smoke, benchmark, and paper configs
docker/         cpu/gpu images and aws compose file
docs/           architecture and aws setup
paper/          manuscript scaffold
scripts/aws/    host bootstrap and sync helpers
src/creditlab/  experiment system
tests/          local verification
```

## Notes

- The benchmark adapters are manifest-driven so the repo stays reproducible even when the heavyweight public benchmarks are integrated in stages.
- The full experiment path is Linux GPU first; macOS remains a smoke-run and analysis machine.
- Local structured artifacts in `artifacts/` are still the source of truth even when W&B is enabled.
