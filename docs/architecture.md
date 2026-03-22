# Architecture

## Design Rules

- Local-first execution on Python 3.11 with `uv`.
- Replayable trajectory graphs with node deduplication by state abstraction.
- Verifier outputs are first-class data, not incidental logs.
- Budget accounting is mandatory and included in every report.
- v0 online learning is outcome-only; branch-aware and hindsight logic start offline.

## Core Flow

1. Load a YAML experiment config.
2. Instantiate the env, policy, store, and tracker from the registry.
3. Collect rollouts into episode graphs with typed nodes, transitions, and verifier events.
4. Persist metadata to SQLite and large artifacts to Parquet.
5. Generate matched-budget reports automatically after collection.
6. Replay stored runs under alternate scorers for offline branch comparison.
7. Train an outcome-only action table from successful trajectories if requested.

## Reporting Guarantees

Each report emits:

- rollout count
- average turns
- prompt tokens
- completion tokens
- verifier calls
- wall-clock time
- success
- tokens per success

Each failure-analysis section emits:

- top repeated dead-end branches
- branch points with highest disagreement
- examples where offline rescoring changes the preferred branch
- state abstraction collapse cases

