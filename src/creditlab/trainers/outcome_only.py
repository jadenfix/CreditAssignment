from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from creditlab.core.interfaces import TrainerBackend
from creditlab.core.models import BudgetConfig
from creditlab.store.sqlite_store import SqliteTrajectoryStore
from creditlab.verifiers.scorers import build_training_targets


class ScoredTableTrainer(TrainerBackend):
    trainer_name = "scored_table"
    default_scorer = "outcome_only"

    def __init__(self, store: SqliteTrajectoryStore, artifact_root: Path) -> None:
        self.store = store
        self.artifact_root = artifact_root

    def fit(self, run_ids: Sequence[str], scorer_name: str, budget: BudgetConfig) -> dict[str, Any]:
        scorer = scorer_name or self.default_scorer
        table: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        all_targets = []
        for run_id in run_ids:
            targets = build_training_targets(self.store.load_episodes(run_id), scorer)
            all_targets.extend(targets)
            for target in targets:
                table[target.state_hash][target.action] += target.score
                counts[target.state_hash][target.action] += 1
        self.store.write_training_targets(all_targets)
        normalized = {
            state_hash: {
                action: score / counts[state_hash][action]
                for action, score in action_scores.items()
            }
            for state_hash, action_scores in table.items()
        }
        policy_dir = self.artifact_root / "policies"
        policy_dir.mkdir(parents=True, exist_ok=True)
        model_path = policy_dir / f"latest_{scorer}.json"
        with model_path.open("w", encoding="utf-8") as handle:
            json.dump(normalized, handle, indent=2, sort_keys=True)
        return {
            "trainer_name": self.trainer_name,
            "scorer_name": scorer,
            "model_path": str(model_path),
            "states_trained": len(normalized),
            "budget_rollout_count": budget.rollout_count,
            "target_count": len(all_targets),
        }


class OutcomeOnlyTrainer(ScoredTableTrainer):
    trainer_name = "outcome_only"
    default_scorer = "outcome_only"


class BranchAwareTrainer(ScoredTableTrainer):
    trainer_name = "branch_aware"
    default_scorer = "branch_aware"


class HindsightTrainer(ScoredTableTrainer):
    trainer_name = "hindsight"
    default_scorer = "hindsight"


class CombinedTrainer(ScoredTableTrainer):
    trainer_name = "combined"
    default_scorer = "combined"
