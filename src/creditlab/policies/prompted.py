from __future__ import annotations

import random
from typing import Any, Sequence

from creditlab.core.interfaces import PolicyBackend
from creditlab.core.models import ActionEvent, BudgetConfig, StateSnapshot, TaskSpec
from creditlab.core.utils import new_id


class PromptedPolicyBackend(PolicyBackend):
    policy_name = "prompted"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.exploration_rate = float(self.config.get("exploration_rate", 0.35))
        self.seed = int(self.config.get("seed", 7))

    def act(
        self,
        history: Sequence[Any],
        state: StateSnapshot,
        budget: BudgetConfig,
        task: TaskSpec,
    ) -> ActionEvent:
        choice = self._choose_action(history, state, task)
        prompt_tokens = min(len(task.instruction.split()) + len(history) * 4, budget.max_prompt_tokens)
        completion_tokens = min(max(4, len(choice.split("_")) * 2), budget.max_completion_tokens)
        return ActionEvent(
            action_id=new_id("action"),
            action_type="tool",
            content=choice,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            metadata={"policy": self.policy_name},
        )

    def _choose_action(self, history: Sequence[Any], state: StateSnapshot, task: TaskSpec) -> str:
        available = list(state.payload.get("available_actions", []))
        if not available:
            return "noop"
        rng = random.Random(f"{self.seed}:{task.task_id}:{state.state_hash}:{len(history)}")
        if task.env_name == "diagnostic_branch_lab":
            preferred = self._diagnostic_preference(task, state)
        else:
            preferred = self._webshop_preference(task, state)
        if preferred in available and rng.random() >= self.exploration_rate:
            return preferred
        return available[rng.randrange(len(available))]

    def _diagnostic_preference(self, task: TaskSpec, state: StateSnapshot) -> str:
        family = state.abstract_state["family"]
        mode = state.abstract_state["mode"]
        available = list(state.payload["available_actions"])
        pivotal = task.metadata["pivotal_branch"]
        if family == "dead_end":
            if mode == "start":
                return pivotal
            if mode == "prepared":
                return next(action for action in available if action.startswith("commit_fix"))
            if mode == "dead_end":
                return next(action for action in available if action.startswith("backtrack"))
        if family == "enabling":
            if mode == "start":
                return pivotal
            if mode == "ready":
                return next(action for action in available if action.startswith("use_clue"))
        if family == "recovery":
            if mode == "start":
                return next(action for action in available if action.startswith("wrong_tool"))
            if mode == "mistake":
                return pivotal
            if mode == "recovered":
                return next(action for action in available if action.startswith("finish_cleanly"))
        return available[0]

    def _webshop_preference(self, task: TaskSpec, state: StateSnapshot) -> str:
        constraints = task.metadata["constraints"]
        page = state.payload["page"]
        if page == "search":
            if constraints["material"] == "ceramic":
                return "search:red mug"
            if constraints["color"] == "blue":
                return "search:blue bottle"
            if constraints["color"] == "green":
                return "search:travel mug"
            return "search:red bottle"
        if page == "results":
            target = {
                ("red", "ceramic"): "open:red_mug",
                ("red", "metal"): "open:red_bottle",
                ("blue", "metal"): "open:blue_bottle",
                ("green", "steel"): "open:green_mug",
            }[(constraints["color"], constraints["material"])]
            return target
        if page == "product":
            return "add_to_cart"
        if page == "cart":
            return "checkout"
        return state.payload["available_actions"][0]

