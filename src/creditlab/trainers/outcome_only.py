from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

from creditlab.core.interfaces import TrainerBackend
from creditlab.core.models import BudgetConfig
from creditlab.store.sqlite_store import SqliteTrajectoryStore


class OutcomeOnlyTrainer(TrainerBackend):
    trainer_name = "outcome_only"

    def __init__(self, store: SqliteTrajectoryStore, artifact_root: Path) -> None:
        self.store = store
        self.artifact_root = artifact_root

    def fit(self, run_ids: Sequence[str], scorer_name: str, budget: BudgetConfig) -> dict[str, Any]:
        if scorer_name != "outcome_only":
            raise ValueError("v0 training only supports the outcome_only scorer")
        table: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for run_id in run_ids:
            for episode in self.store.load_episodes(run_id):
                reward = 1.0 if episode.success else 0.0
                node_index = {node.node_id: node for node in episode.nodes}
                for transition in episode.transitions:
                    state_hash = node_index[transition.parent_node_id].state.state_hash
                    action = transition.action.content
                    table[state_hash][action] += reward
                    counts[state_hash][action] += 1
        normalized = {
            state_hash: {
                action: score / counts[state_hash][action]
                for action, score in action_scores.items()
            }
            for state_hash, action_scores in table.items()
        }
        policy_dir = self.artifact_root / "policies"
        policy_dir.mkdir(parents=True, exist_ok=True)
        model_path = policy_dir / "latest_outcome_only.json"
        with model_path.open("w", encoding="utf-8") as handle:
            json.dump(normalized, handle, indent=2, sort_keys=True)
        return {
            "trainer_name": self.trainer_name,
            "scorer_name": scorer_name,
            "model_path": str(model_path),
            "states_trained": len(normalized),
            "budget_rollout_count": budget.rollout_count,
        }

