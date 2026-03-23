# Branch-Aware Hindsight Credit Assignment Under Matched Budgets

## Abstract

TODO: summarize the matched-budget claim and the main result.

## Introduction

- Outcome-only credit is too coarse for long-horizon agent behavior.
- Branch-aware and hindsight credit should improve search quality, not just terminal accuracy.
- The core question is whether probability mass shifts toward better downstream branches under equal collection budgets.

## Method

TODO: describe the state graph, verifier events, offline scorers, and scored online training path.

## Benchmarks

- Diagnostic benchmark with explicit pivotal, dead-end, enabling, and recovery annotations.
- WebShop and ALFWorld as core long-horizon ablation environments.
- LiveCodeBench as first external validation.
- SWE-bench Lite as heavier follow-on validation.

## Results

TODO: include exported tables, budget summaries, success plots, and matched-prefix change-rate plots.

## Failure Analysis

TODO: include repeated dead-end branches, disagreement points, changed preferred branches, and state abstraction collapses.

## Limitations

TODO: document manifest simplifications, model/backend tradeoffs, and AWS single-GPU constraints.

