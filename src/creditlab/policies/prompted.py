from __future__ import annotations

import json
import random
import time
import urllib.error
import urllib.request
from typing import Any, Sequence

from creditlab.core.interfaces import PolicyBackend
from creditlab.core.models import ActionEvent, BudgetConfig, StateSnapshot, TaskSpec
from creditlab.core.utils import new_id


class PromptedPolicyBackend(PolicyBackend):
    policy_name = "prompted"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.exploration_rate = float(self.config.get("exploration_rate", 0.2))
        self.seed = int(self.config.get("seed", 7))
        self._rng = random.Random(self.seed)

    def act(
        self,
        history: Sequence[Any],
        state: StateSnapshot,
        budget: BudgetConfig,
        task: TaskSpec,
    ) -> ActionEvent:
        available = list(state.payload.get("available_actions", []))
        oracle = state.payload.get("oracle_action")
        rng = self._rng
        if not available:
            choice = "noop"
        elif oracle in available and rng.random() >= self.exploration_rate:
            choice = str(oracle)
        else:
            choice = available[rng.randrange(len(available))]
        return ActionEvent(
            action_id=new_id("action"),
            action_type="tool",
            content=choice,
            prompt_tokens=min(len(task.instruction.split()) + len(history) * 4, budget.max_prompt_tokens),
            completion_tokens=min(max(4, len(choice.split("_")) * 2), budget.max_completion_tokens),
            metadata={"policy": self.policy_name},
        )


class VLLMServerPolicyBackend(PolicyBackend):
    policy_name = "vllm_server"

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.endpoint = self.config.get("endpoint", "http://localhost:8000/generate")
        self.model = self.config.get("model", "default")
        self.api_format = self.config.get("api_format", "custom")
        self.timeout_seconds = float(self.config.get("timeout_seconds", 20))
        self.max_retries = int(self.config.get("max_retries", 2))
        self.retry_backoff_seconds = float(self.config.get("retry_backoff_seconds", 0.5))
        self.fallback = PromptedPolicyBackend(config={"exploration_rate": 0.0, **self.config})

    def _build_openai_payload(
        self,
        task: TaskSpec,
        state: StateSnapshot,
        history: Sequence[Any],
        budget: BudgetConfig,
    ) -> dict[str, Any]:
        available = state.payload.get("available_actions", [])
        history_text = ", ".join(
            getattr(getattr(item, "action", None), "content", "?") for item in history
        ) or "none"
        user_msg = (
            f"Task: {task.instruction}\n"
            f"Current state: {state.abstract_state}\n"
            f"Actions taken so far: {history_text}\n"
            f"Available actions: {available}\n\n"
            f"Choose exactly one action from the available list. "
            f"Respond with only the action name, nothing else."
        )
        return {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "You are an agent that selects actions to complete tasks. Respond with exactly one action name from the available list, nothing else."},
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": 32,
            "temperature": 0.0,
        }

    def _build_custom_payload(
        self,
        task: TaskSpec,
        state: StateSnapshot,
        history: Sequence[Any],
        budget: BudgetConfig,
    ) -> dict[str, Any]:
        return {
            "instruction": task.instruction,
            "available_actions": state.payload.get("available_actions", []),
            "history": [getattr(getattr(item, "action", None), "content", None) for item in history],
            "state": state.abstract_state,
            "max_tokens": budget.max_completion_tokens,
        }

    def act(
        self,
        history: Sequence[Any],
        state: StateSnapshot,
        budget: BudgetConfig,
        task: TaskSpec,
    ) -> ActionEvent:
        if self.api_format == "openai":
            payload = self._build_openai_payload(task, state, history, budget)
        else:
            payload = self._build_custom_payload(task, state, history, budget)
        request = urllib.request.Request(
            self.endpoint,
            method="POST",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        last_error = "unknown_error"
        for attempt in range(1, self.max_retries + 2):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    body = json.loads(response.read().decode("utf-8"))
                choice = self._validate_choice(body, state)
                return self._build_action(
                    choice=choice,
                    history=history,
                    task=task,
                    budget=budget,
                    metadata={
                        "policy": self.policy_name,
                        "endpoint": self.endpoint,
                        "model": self.model,
                        "api_format": self.api_format,
                        "attempts": attempt,
                        "fallback_used": False,
                    },
                )
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if attempt >= self.max_retries + 1:
                    break
                time.sleep(self.retry_backoff_seconds * attempt)
        fallback_action = self.fallback.act(history, state, budget, task)
        return fallback_action.model_copy(
            update={
                "metadata": {
                    **fallback_action.metadata,
                    "policy": self.policy_name,
                    "endpoint": self.endpoint,
                    "model": self.model,
                    "attempts": self.max_retries + 1,
                    "fallback_used": True,
                    "fallback_policy": self.fallback.policy_name,
                    "fallback_reason": last_error,
                }
            }
        )

    def _parse_action_from_text(self, text: str, available: list[str]) -> str | None:
        cleaned = text.strip().strip("\"'`.").lower()
        for action in available:
            if action.lower() == cleaned:
                return action
        for action in available:
            if action.lower() in cleaned:
                return action
        return None

    def _validate_choice(self, body: dict[str, Any], state: StateSnapshot) -> str:
        available_actions = list(state.payload.get("available_actions", []))
        if self.api_format == "openai":
            choices = body.get("choices", [])
            if not choices:
                raise ValueError("OpenAI response has no choices")
            text = choices[0].get("message", {}).get("content", "") or choices[0].get("text", "")
            choice = self._parse_action_from_text(text, available_actions)
            if choice:
                return choice
            raise ValueError(f"could not parse action from LLM response: {text!r}")
        choice = body.get("action")
        if choice in available_actions:
            return str(choice)
        if choice is None:
            raise ValueError("response body did not include an action")
        raise ValueError(f"response action {choice!r} is not valid for this state")

    def _build_action(
        self,
        choice: str,
        history: Sequence[Any],
        task: TaskSpec,
        budget: BudgetConfig,
        metadata: dict[str, Any],
    ) -> ActionEvent:
        return ActionEvent(
            action_id=new_id("action"),
            action_type="tool",
            content=str(choice),
            prompt_tokens=min(len(task.instruction.split()) + len(history) * 4, budget.max_prompt_tokens),
            completion_tokens=min(max(4, len(str(choice).split("_")) * 2), budget.max_completion_tokens),
            metadata=metadata,
        )
