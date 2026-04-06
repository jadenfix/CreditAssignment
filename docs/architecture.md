# Architecture

## Paper Claim

CreditLab is built to test whether branch-aware plus hindsight credit assignment improves matched-prefix branch selection and downstream success over outcome-only training under equal collection budgets.

## Lanes

- Benchmark lane: `stochastic` (primary), `diagnostic`, `webshop`, `alfworld` (manifest-driven), and `alfworld_real` (live environment) all run through the same EnvAdapter/verifier interface.
- Experiment lane: sweep execution tracks run groups, conditions, seeds, budgets, and deployment metadata.
- Training lane: offline scorers define the truth surface. Scored table trainers provide the baseline path for `outcome_only`, `branch_aware`, `hindsight`, and `combined`. LoRA SFT trainer provides the neural training path.
- Deployment lane: local CPU smoke plus GCP Linux GPU deployment (NVIDIA L4).
- Paper lane: aggregate results, export tables, render publication figures, and maintain the manuscript in-repo.

## Key Guarantees

- State abstraction and verifier versions are stored with the benchmark metadata.
- Reports always emit rollout count, average turns, prompt tokens, completion tokens, verifier calls, wall-clock time, success, and tokens per success.
- Sweep aggregates always emit budget-parity checks across compared conditions.
- Failure analysis always includes repeated dead-end branches, disagreement branch points, changed preferred branches under rescoring, and state abstraction collapse cases.

