from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from creditlab.core.interfaces import PolicyBackend
from creditlab.core.models import ActionEvent, BudgetConfig, StateSnapshot, TaskSpec
from creditlab.core.utils import new_id


class TablePolicyBackend(PolicyBackend):
    policy_name = "table"

    def __init__(self, config: dict[str, Any] | None, artifact_root: Path) -> None:
        self.config = config or {}
        self.artifact_root = artifact_root
        self.fallback = self.config.get("fallback_action")
        model_path = Path(self.config.get("model_path", artifact_root / "policies" / "latest_outcome_only.json"))
        self.model = self._load(model_path)

    def _load(self, path: Path) -> dict[str, dict[str, float]]:
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    def act(
        self,
        history: Sequence[Any],
        state: StateSnapshot,
        budget: BudgetConfig,
        task: TaskSpec,
    ) -> ActionEvent:
        available = list(state.payload.get("available_actions", []))
        state_scores = self.model.get(state.state_hash, {})
        if not available:
            choice = "noop"
        elif state_scores:
            choice = max(available, key=lambda action: state_scores.get(action, float("-inf")))
        else:
            choice = self.fallback or state.payload.get("oracle_action") or available[0]
        return ActionEvent(
            action_id=new_id("action"),
            action_type="tool",
            content=choice,
            prompt_tokens=min(8 + len(history) * 2, budget.max_prompt_tokens),
            completion_tokens=min(8, budget.max_completion_tokens),
            metadata={"policy": self.policy_name, "state_hash": state.state_hash},
        )
