from __future__ import annotations

from typing import Any

from creditlab.core.interfaces import EnvAdapter
from creditlab.core.models import ActionEvent, StateDelta, StateSnapshot, TaskSpec, VerifierEvent, VerifierStatus
from creditlab.core.utils import stable_hash


CATALOG = {
    "red_mug": {"title": "Red Ceramic Mug", "color": "red", "material": "ceramic", "price": 12},
    "blue_bottle": {"title": "Blue Metal Bottle", "color": "blue", "material": "metal", "price": 24},
    "green_mug": {"title": "Green Travel Mug", "color": "green", "material": "steel", "price": 19},
    "red_bottle": {"title": "Red Hiking Bottle", "color": "red", "material": "metal", "price": 22},
}


class WebShopSmokeEnv(EnvAdapter):
    env_name = "webshop"

    def __init__(self) -> None:
        self._tasks = self._build_tasks()

    def _build_tasks(self) -> list[TaskSpec]:
        smoke_specs = [
            ("webshop_smoke_01", "Find a red ceramic mug under $15.", {"color": "red", "material": "ceramic", "max_price": 15}),
            ("webshop_smoke_02", "Buy a red metal bottle under $25.", {"color": "red", "material": "metal", "max_price": 25}),
            ("webshop_smoke_03", "Pick the blue metal bottle.", {"color": "blue", "material": "metal", "max_price": 30}),
            ("webshop_smoke_04", "Choose a travel mug costing less than $20.", {"color": "green", "material": "steel", "max_price": 20}),
        ]
        return [
            TaskSpec(
                task_id=task_id,
                env_name=self.env_name,
                instruction=instruction,
                split="smoke",
                metadata={"constraints": constraints},
            )
            for task_id, instruction, constraints in smoke_specs
        ]

    def list_tasks(self, split: str) -> list[TaskSpec]:
        if split == "smoke":
            return self._tasks
        return self._tasks

    def reset(self, task: TaskSpec) -> StateSnapshot:
        payload = {
            "page": "search",
            "selected_product": None,
            "cart_product": None,
            "query": None,
            "history": [],
            "available_actions": self._available_actions("search", None),
        }
        abstract_state = self.abstract_state_from_payload(payload, task)
        return StateSnapshot(state_hash=stable_hash(abstract_state), abstract_state=abstract_state, payload=payload)

    def step(self, task: TaskSpec, state: StateSnapshot, action: ActionEvent) -> tuple[StateSnapshot, str, bool]:
        payload = dict(state.payload)
        payload["history"] = list(payload["history"]) + [action.content]
        observation = "No change."
        terminal = False
        if action.content.startswith("search:"):
            payload["page"] = "results"
            payload["query"] = action.content.split(":", 1)[1]
            observation = f"Search results for '{payload['query']}' show {', '.join(CATALOG)}."
        elif action.content.startswith("open:"):
            product_id = action.content.split(":", 1)[1]
            payload["page"] = "product"
            payload["selected_product"] = product_id
            observation = f"Opened {CATALOG[product_id]['title']}."
        elif action.content == "add_to_cart" and payload["selected_product"]:
            payload["page"] = "cart"
            payload["cart_product"] = payload["selected_product"]
            observation = f"Added {payload['cart_product']} to cart."
        elif action.content == "checkout" and payload["cart_product"]:
            payload["page"] = "checkout"
            observation = "Checkout completed."
            terminal = True
        elif action.content == "back":
            payload["page"] = "results"
            observation = "Went back to the results page."
        else:
            observation = "That action is invalid on this page."
        payload["available_actions"] = self._available_actions(payload["page"], payload["selected_product"])
        abstract_state = self.abstract_state_from_payload(payload, task)
        next_state = StateSnapshot(
            state_hash=stable_hash(abstract_state),
            abstract_state=abstract_state,
            payload=payload,
        )
        return next_state, observation, terminal

    def snapshot(self, state: StateSnapshot) -> StateSnapshot:
        return state

    def abstract_state(self, state: StateSnapshot) -> dict[str, Any]:
        return state.abstract_state

    def abstract_state_from_payload(self, payload: dict[str, Any], task: TaskSpec) -> dict[str, Any]:
        constraints = task.metadata["constraints"]
        selected = CATALOG.get(payload["selected_product"] or "", {})
        matched_attributes = sum(
            [
                selected.get("color") == constraints.get("color"),
                selected.get("material") == constraints.get("material"),
                selected.get("price", 10**9) <= constraints.get("max_price", 10**9),
            ]
        )
        return {
            "page": payload["page"],
            "has_selection": bool(payload["selected_product"]),
            "has_cart": bool(payload["cart_product"]),
            "matched_attributes": matched_attributes,
        }

    def _available_actions(self, page: str, selected_product: str | None) -> list[str]:
        if page == "search":
            return ["search:red mug", "search:red bottle", "search:blue bottle", "search:travel mug"]
        if page == "results":
            return [f"open:{product_id}" for product_id in sorted(CATALOG)]
        if page == "product":
            return ["add_to_cart", "back"]
        if page == "cart":
            return ["checkout", "back"]
        return ["back"]

    def verify_transition(
        self,
        task: TaskSpec,
        before: StateSnapshot,
        action: ActionEvent,
        after: StateSnapshot,
        observation_text: str,
    ) -> list[VerifierEvent]:
        constraints = task.metadata["constraints"]
        selected = CATALOG.get(after.payload["selected_product"] or "", {})
        matched_attributes = after.abstract_state["matched_attributes"]
        events = [
            VerifierEvent(
                verifier_id=f"{task.task_id}:progress",
                status=VerifierStatus.PASS if matched_attributes >= before.abstract_state["matched_attributes"] else VerifierStatus.WARN,
                signal_name="attribute_match",
                score=float(matched_attributes),
                state_delta=(
                    StateDelta.IMPROVED
                    if matched_attributes > before.abstract_state["matched_attributes"]
                    else StateDelta.NEUTRAL
                ),
                details={"matched_attributes": matched_attributes, "constraints": constraints},
            )
        ]
        if after.payload["page"] == "checkout":
            success = matched_attributes == 3 and after.payload["cart_product"] is not None
            events.append(
                VerifierEvent(
                    verifier_id=f"{task.task_id}:success",
                    status=VerifierStatus.PASS if success else VerifierStatus.FAIL,
                    signal_name="success",
                    score=1.0 if success else 0.0,
                    state_delta=StateDelta.IMPROVED if success else StateDelta.REGRESSED,
                    details={"selected_product": after.payload["cart_product"]},
                )
            )
        if "invalid" in observation_text.lower():
            events.append(
                VerifierEvent(
                    verifier_id=f"{task.task_id}:validity",
                    status=VerifierStatus.FAIL,
                    signal_name="action_validity",
                    score=-1.0,
                    state_delta=StateDelta.REGRESSED,
                    details={"page": after.payload["page"]},
                )
            )
        if after.payload["selected_product"] and matched_attributes < 2:
            events.append(
                VerifierEvent(
                    verifier_id=f"{task.task_id}:dead_end",
                    status=VerifierStatus.WARN,
                    signal_name="dead_end",
                    score=-0.5,
                    state_delta=StateDelta.REGRESSED,
                    details={"selected_product": selected},
                )
            )
        return events
