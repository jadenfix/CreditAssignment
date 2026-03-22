# CreditLab

Local-first baseline scaffold for branch-aware credit assignment research on long-horizon agents.

## Research Objective

v0 objective: determine whether action-level branch-aware offline rescoring on replayed trajectories produces better matched-prefix branch selection than outcome-only scoring under equal collection budgets.

## What Is Implemented

- Typed experiment configs backed by YAML.
- Two runnable environments: `diagnostic_branch_lab` and a local `webshop` smoke adapter.
- Verifier-first rollout collection with state-based node deduplication into replayable trajectory graphs.
- SQLite metadata storage plus Parquet artifacts for episodes and replay outputs.
- Outcome-only training baseline.
- Offline `branch_aware` and `hindsight` rescoring.
- Required summary and failure-analysis reporting.
- CLI commands for `collect`, `train`, `replay`, `report`, and `validate-config`.

## Layout

```text
src/creditlab/
  analysis/    report + replay analysis
  cli/         Typer CLI
  core/        configs, models, interfaces, registry
  envs/        diagnostic, webshop, code-agent contracts
  policies/    prompted + table policy backends
  rollout/     collection engine
  store/       SQLite + Parquet trajectory store
  tracking/    local + optional W&B tracker
  trainers/    outcome-only trainer
  verifiers/   offline rescorers
configs/       example experiments
docs/          architecture notes
tests/         unit and integration coverage
```

## Quickstart

```bash
uv run creditlab validate-config --config configs/diagnostic_smoke.yaml
uv run creditlab collect --config configs/diagnostic_smoke.yaml
uv run creditlab train --config configs/diagnostic_outcome_only.yaml
uv run creditlab replay --run-id <run_id> --config configs/diagnostic_smoke.yaml --scorer branch_aware
uv run creditlab report --run-id <run_id> --config configs/diagnostic_smoke.yaml
```

## Notes

- `webshop` is a self-contained smoke adapter, not the full public benchmark integration yet.
- Code-agent benchmarks are interface-only in v0 by design.
- Local structured artifacts in `artifacts/` are the source of truth; W&B is optional.
