# Architecture

## Paper Claim

CreditLab is built to test whether branch-aware plus hindsight credit assignment improves matched-prefix branch selection and downstream success over outcome-only training under equal collection budgets.

## Lanes

- Benchmark lane: `diagnostic`, `webshop`, `alfworld`, `livecodebench`, and `swebench_lite` all run through versioned task manifests and the same graph/verifier interface.
- Experiment lane: sweep execution tracks run groups, conditions, seeds, budgets, and deployment metadata.
- Training lane: offline scorers define the truth surface, and scored table trainers provide the first trainable online path for `outcome_only`, `branch_aware`, `hindsight`, and `combined`.
- Deployment lane: local CPU smoke plus AWS Linux GPU Docker deployment.
- Paper lane: aggregate results, export tables, render simple plots, and maintain a manuscript scaffold in-repo.

## Key Guarantees

- State abstraction and verifier versions are stored with the benchmark metadata.
- Reports always emit rollout count, average turns, prompt tokens, completion tokens, verifier calls, wall-clock time, success, and tokens per success.
- Sweep aggregates always emit budget-parity checks across compared conditions.
- Failure analysis always includes repeated dead-end branches, disagreement branch points, changed preferred branches under rescoring, and state abstraction collapse cases.

