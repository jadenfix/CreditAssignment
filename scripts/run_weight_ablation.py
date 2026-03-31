#!/usr/bin/env python3
"""Run scorer weight sensitivity analysis on the diagnostic benchmark.

Usage (on GCP VM):
    python scripts/run_weight_ablation.py --config configs/diagnostic_paper_local.yaml

Varies branch_aware verifier_weight, dead_end_weight, and hindsight pivotal_bonus
across a small grid. Runs a 3-seed sweep for each combination and reports results.
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from creditlab.analysis.sweep import SweepRunner
from creditlab.core.config import ExperimentConfig
from creditlab.verifiers.scorers import SCORER_WEIGHTS, set_scorer_weights


def run_ablation(config_path: str, output_path: str):
    verifier_weights = [0.1, 0.25, 0.5]
    dead_end_weights = [0.25, 0.5, 1.0]
    pivotal_bonuses = [0.5, 0.75, 1.0]

    results = []
    total = len(verifier_weights) * len(dead_end_weights) * len(pivotal_bonuses)
    count = 0

    for vw, dw, pb in product(verifier_weights, dead_end_weights, pivotal_bonuses):
        count += 1
        print(f"[{count}/{total}] verifier={vw}, dead_end={dw}, pivotal={pb}")

        set_scorer_weights(verifier_weight=vw, dead_end_weight=dw, pivotal_bonus=pb)

        with open(config_path) as f:
            import yaml
            raw = yaml.safe_load(f)
        # Use only 3 seeds for ablation speed
        raw["sweep"]["seeds"] = [7, 11, 19]
        raw["experiment"]["seeds"] = [7, 11, 19]
        config = ExperimentConfig(**raw)

        try:
            payload = SweepRunner(config).run()
            summaries = payload["aggregate"]["condition_summaries"]
            row = {"verifier_weight": vw, "dead_end_weight": dw, "pivotal_bonus": pb}
            for s in summaries:
                if s["metric_name"] == "success" and "replay" not in s["metadata"].get("scope", ""):
                    row[f"{s['condition_name']}_success"] = round(s["mean"], 3)
            results.append(row)
            print(f"  -> outcome_only={row.get('outcome_only_success','?')} branch_aware={row.get('branch_aware_success','?')} combined={row.get('combined_success','?')}")
        except Exception as e:
            print(f"  -> FAILED: {e}")
            results.append({"verifier_weight": vw, "dead_end_weight": dw, "pivotal_bonus": pb, "error": str(e)})

    # Reset to defaults
    set_scorer_weights(verifier_weight=0.25, dead_end_weight=0.5, pivotal_bonus=0.75)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    print(f"\nResults written to {output}")

    # Summary: check if ranking holds across all configs
    ranking_holds = 0
    for r in results:
        if "error" in r:
            continue
        ba = r.get("branch_aware_success", 0)
        oo = r.get("outcome_only_success", 0)
        co = r.get("combined_success", 0)
        if co >= ba >= oo:
            ranking_holds += 1
    valid = sum(1 for r in results if "error" not in r)
    print(f"Ranking (combined >= branch_aware >= outcome_only) holds in {ranking_holds}/{valid} configurations")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/diagnostic_paper_local.yaml")
    parser.add_argument("--output", default="artifacts/paper/weight_ablation.json")
    args = parser.parse_args()
    run_ablation(args.config, args.output)
