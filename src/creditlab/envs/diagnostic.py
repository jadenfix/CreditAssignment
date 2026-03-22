from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from creditlab.core.interfaces import EnvAdapter
from creditlab.core.models import ActionEvent, StateDelta, StateSnapshot, TaskSpec, VerifierEvent, VerifierStatus
from creditlab.core.utils import stable_hash


@dataclass(frozen=True)
class EdgeSpec:
    next_node: str
    observation: str
    terminal: bool
    success: bool
    progress_delta: int
    state_delta: str
    branch_label: str
    dead_end: bool = False
    enabling: bool = False
    recovery: bool = False
    invalid: bool = False


def _node_payload(
    family: str,
    phase: str,
    mode: str,
    raw_node: str,
    progress: int,
    history: list[str],
    available_actions: list[str],
) -> dict[str, Any]:
    return {
        "family": family,
        "phase": phase,
        "mode": mode,
        "raw_node": raw_node,
        "progress": progress,
        "history": history,
        "available_actions": available_actions,
    }


class DiagnosticBranchLabEnv(EnvAdapter):
    env_name = "diagnostic_branch_lab"

    def __init__(self) -> None:
        self.task_definitions = self._build_task_definitions()

    def _build_task_definitions(self) -> dict[str, dict[str, Any]]:
        definitions: dict[str, dict[str, Any]] = {}
        for family in ("dead_end", "enabling", "recovery"):
            for index in range(12):
                task_id = f"{family}_{index:02d}"
                definitions[task_id] = self._family_task(family, index)
        return definitions

    def _family_task(self, family: str, index: int) -> dict[str, Any]:
        if family == "dead_end":
            return {
                "family": family,
                "instruction": f"Resolve a branchy dead-end scenario #{index}.",
                "pivotal_branch": f"inspect_{index}",
                "states": {
                    "start": {
                        "phase": "branch",
                        "mode": "start",
                        "progress": 0,
                        "edges": {
                            f"inspect_{index}": EdgeSpec(
                                next_node="prepared",
                                observation="You inspected the state and found the correct path.",
                                terminal=False,
                                success=False,
                                progress_delta=1,
                                state_delta="improved",
                                branch_label="pivotal",
                            ),
                            f"rush_{index}": EdgeSpec(
                                next_node="dead_end",
                                observation="You rushed ahead and hit a dead end.",
                                terminal=False,
                                success=False,
                                progress_delta=-1,
                                state_delta="regressed",
                                branch_label="dead_end",
                                dead_end=True,
                            ),
                            f"wait_{index}": EdgeSpec(
                                next_node="start",
                                observation="You waited and learned nothing new.",
                                terminal=False,
                                success=False,
                                progress_delta=0,
                                state_delta="neutral",
                                branch_label="stall",
                            ),
                        },
                    },
                    "prepared": {
                        "phase": "commit",
                        "mode": "prepared",
                        "progress": 1,
                        "edges": {
                            f"commit_fix_{index}": EdgeSpec(
                                next_node="success",
                                observation="The prepared fix resolves the task.",
                                terminal=True,
                                success=True,
                                progress_delta=2,
                                state_delta="improved",
                                branch_label="success",
                            ),
                            f"wander_{index}": EdgeSpec(
                                next_node="dead_end",
                                observation="You wandered off and lost the prepared state.",
                                terminal=False,
                                success=False,
                                progress_delta=-1,
                                state_delta="regressed",
                                branch_label="dead_end",
                                dead_end=True,
                            ),
                        },
                    },
                    "dead_end": {
                        "phase": "trap",
                        "mode": "dead_end",
                        "progress": -1,
                        "edges": {
                            f"backtrack_{index}": EdgeSpec(
                                next_node="prepared",
                                observation="You backtracked and recovered partial progress.",
                                terminal=False,
                                success=False,
                                progress_delta=1,
                                state_delta="improved",
                                branch_label="recovery",
                                recovery=True,
                            ),
                            f"double_down_{index}": EdgeSpec(
                                next_node="failure",
                                observation="Doubling down locks in the dead end.",
                                terminal=True,
                                success=False,
                                progress_delta=-1,
                                state_delta="regressed",
                                branch_label="failure",
                                dead_end=True,
                            ),
                        },
                    },
                    "success": {"phase": "done", "mode": "success", "progress": 3, "edges": {}},
                    "failure": {"phase": "done", "mode": "failure", "progress": -2, "edges": {}},
                },
            }
        if family == "enabling":
            return {
                "family": family,
                "instruction": f"Solve an enabling-step scenario #{index}.",
                "pivotal_branch": f"collect_clue_{index}",
                "states": {
                    "start": {
                        "phase": "setup",
                        "mode": "start",
                        "progress": 0,
                        "edges": {
                            f"collect_clue_{index}": EdgeSpec(
                                next_node="ready",
                                observation="You collected a clue. No immediate reward, but the path is now unlocked.",
                                terminal=False,
                                success=False,
                                progress_delta=0,
                                state_delta="neutral",
                                branch_label="enabler",
                                enabling=True,
                            ),
                            f"answer_now_{index}": EdgeSpec(
                                next_node="failure",
                                observation="You answered too early and failed the task.",
                                terminal=True,
                                success=False,
                                progress_delta=-1,
                                state_delta="regressed",
                                branch_label="premature",
                                dead_end=True,
                            ),
                            f"browse_{index}": EdgeSpec(
                                next_node="start",
                                observation="You browse around but do not uncover anything useful.",
                                terminal=False,
                                success=False,
                                progress_delta=0,
                                state_delta="neutral",
                                branch_label="stall",
                            ),
                        },
                    },
                    "ready": {
                        "phase": "use_clue",
                        "mode": "ready",
                        "progress": 1,
                        "edges": {
                            f"use_clue_{index}": EdgeSpec(
                                next_node="success",
                                observation="The clue leads directly to the correct answer.",
                                terminal=True,
                                success=True,
                                progress_delta=2,
                                state_delta="improved",
                                branch_label="success",
                            ),
                            f"ignore_clue_{index}": EdgeSpec(
                                next_node="failure",
                                observation="Ignoring the clue wastes the setup and fails the task.",
                                terminal=True,
                                success=False,
                                progress_delta=-1,
                                state_delta="regressed",
                                branch_label="failure",
                            ),
                        },
                    },
                    "success": {"phase": "done", "mode": "success", "progress": 3, "edges": {}},
                    "failure": {"phase": "done", "mode": "failure", "progress": -1, "edges": {}},
                },
            }
        return {
            "family": family,
            "instruction": f"Recover from an early mistake in scenario #{index}.",
            "pivotal_branch": f"recover_{index}",
            "states": {
                "start": {
                    "phase": "attempt",
                    "mode": "start",
                    "progress": 0,
                    "edges": {
                        f"wrong_tool_{index}": EdgeSpec(
                            next_node="mistake",
                            observation="You used the wrong tool and introduced an error.",
                            terminal=False,
                            success=False,
                            progress_delta=-1,
                            state_delta="regressed",
                            branch_label="mistake",
                            dead_end=True,
                        ),
                        f"safe_probe_{index}": EdgeSpec(
                            next_node="recovered",
                            observation="You probed safely and left yourself a clean recovery path.",
                            terminal=False,
                            success=False,
                            progress_delta=1,
                            state_delta="improved",
                            branch_label="safe",
                        ),
                    },
                },
                "mistake": {
                    "phase": "recovery",
                    "mode": "mistake",
                    "progress": -1,
                    "edges": {
                        f"recover_{index}": EdgeSpec(
                            next_node="recovered",
                            observation="You recover from the error and restore a viable path.",
                            terminal=False,
                            success=False,
                            progress_delta=2,
                            state_delta="improved",
                            branch_label="recovery",
                            recovery=True,
                        ),
                        f"double_down_{index}": EdgeSpec(
                            next_node="failure",
                            observation="Doubling down makes the error unrecoverable.",
                            terminal=True,
                            success=False,
                            progress_delta=-1,
                            state_delta="regressed",
                            branch_label="failure",
                            dead_end=True,
                        ),
                    },
                },
                "recovered": {
                    "phase": "finish",
                    "mode": "recovered",
                    "progress": 1,
                    "edges": {
                        f"finish_cleanly_{index}": EdgeSpec(
                            next_node="success",
                            observation="You finish the task cleanly after recovery.",
                            terminal=True,
                            success=True,
                            progress_delta=2,
                            state_delta="improved",
                            branch_label="success",
                        ),
                        f"panic_{index}": EdgeSpec(
                            next_node="failure",
                            observation="You panic and give up the recovered path.",
                            terminal=True,
                            success=False,
                            progress_delta=-1,
                            state_delta="regressed",
                            branch_label="failure",
                        ),
                    },
                },
                "success": {"phase": "done", "mode": "success", "progress": 3, "edges": {}},
                "failure": {"phase": "done", "mode": "failure", "progress": -2, "edges": {}},
            },
        }

    def list_tasks(self, split: str) -> list[TaskSpec]:
        task_ids = sorted(self.task_definitions)
        if split == "smoke":
            task_ids = task_ids[:6]
        elif split == "train":
            task_ids = task_ids[:24]
        elif split == "eval":
            task_ids = task_ids[24:]
        return [
            TaskSpec(
                task_id=task_id,
                env_name=self.env_name,
                instruction=self.task_definitions[task_id]["instruction"],
                split=split,
                metadata={
                    "family": self.task_definitions[task_id]["family"],
                    "pivotal_branch": self.task_definitions[task_id]["pivotal_branch"],
                },
            )
            for task_id in task_ids
        ]

    def reset(self, task: TaskSpec) -> StateSnapshot:
        definition = self.task_definitions[task.task_id]
        payload = _node_payload(
            family=definition["family"],
            phase=definition["states"]["start"]["phase"],
            mode=definition["states"]["start"]["mode"],
            raw_node="start",
            progress=definition["states"]["start"]["progress"],
            history=[],
            available_actions=sorted(definition["states"]["start"]["edges"]),
        )
        abstract_state = {
            "family": payload["family"],
            "phase": payload["phase"],
            "mode": payload["mode"],
            "progress": payload["progress"],
        }
        return StateSnapshot(state_hash=stable_hash(abstract_state), abstract_state=abstract_state, payload=payload)

    def step(self, task: TaskSpec, state: StateSnapshot, action: ActionEvent) -> tuple[StateSnapshot, str, bool]:
        definition = self.task_definitions[task.task_id]
        current_node = state.payload["raw_node"]
        current = definition["states"][current_node]
        edge = current["edges"].get(action.content)
        if edge is None:
            payload = _node_payload(
                family=definition["family"],
                phase=current["phase"],
                mode="invalid",
                raw_node=current_node,
                progress=current["progress"] - 1,
                history=state.payload["history"] + [action.content],
                available_actions=sorted(current["edges"]),
            )
            abstract_state = {
                "family": payload["family"],
                "phase": payload["phase"],
                "mode": payload["mode"],
                "progress": payload["progress"],
            }
            return (
                StateSnapshot(state_hash=stable_hash(abstract_state), abstract_state=abstract_state, payload=payload),
                "That action is invalid from the current state.",
                False,
            )
        next_state = definition["states"][edge.next_node]
        payload = _node_payload(
            family=definition["family"],
            phase=next_state["phase"],
            mode=next_state["mode"],
            raw_node=edge.next_node,
            progress=next_state["progress"],
            history=state.payload["history"] + [action.content],
            available_actions=sorted(next_state["edges"]),
        )
        abstract_state = {
            "family": payload["family"],
            "phase": payload["phase"],
            "mode": payload["mode"],
            "progress": payload["progress"],
        }
        return (
            StateSnapshot(state_hash=stable_hash(abstract_state), abstract_state=abstract_state, payload=payload),
            edge.observation,
            edge.terminal,
        )

    def snapshot(self, state: StateSnapshot) -> StateSnapshot:
        return state

    def abstract_state(self, state: StateSnapshot) -> dict[str, Any]:
        return state.abstract_state

    def verify_transition(
        self,
        task: TaskSpec,
        before: StateSnapshot,
        action: ActionEvent,
        after: StateSnapshot,
        observation_text: str,
    ) -> list[VerifierEvent]:
        definition = self.task_definitions[task.task_id]
        current_node = before.payload["raw_node"]
        edge = definition["states"][current_node]["edges"].get(action.content)
        if edge is None:
            return [
                VerifierEvent(
                    verifier_id=f"{task.task_id}:validity",
                    status=VerifierStatus.FAIL,
                    signal_name="action_validity",
                    score=-1.0,
                    state_delta=StateDelta.REGRESSED,
                    details={"reason": "invalid_action"},
                ),
            ]
        progress_score = float(edge.progress_delta)
        events = [
            VerifierEvent(
                verifier_id=f"{task.task_id}:progress",
                status=VerifierStatus.PASS if progress_score >= 0 else VerifierStatus.WARN,
                signal_name="progress",
                score=progress_score,
                state_delta=StateDelta(edge.state_delta),
                details={
                    "branch_label": edge.branch_label,
                    "pivotal_action": task.metadata["pivotal_branch"],
                    "dead_end": edge.dead_end,
                    "enabling": edge.enabling,
                    "recovery": edge.recovery,
                },
            )
        ]
        if edge.terminal:
            events.append(
                VerifierEvent(
                    verifier_id=f"{task.task_id}:terminal",
                    status=VerifierStatus.PASS if edge.success else VerifierStatus.FAIL,
                    signal_name="success",
                    score=1.0 if edge.success else 0.0,
                    state_delta=StateDelta.IMPROVED if edge.success else StateDelta.REGRESSED,
                    details={"terminal": True},
                )
            )
        if edge.dead_end:
            events.append(
                VerifierEvent(
                    verifier_id=f"{task.task_id}:dead_end",
                    status=VerifierStatus.WARN,
                    signal_name="dead_end",
                    score=-1.0,
                    state_delta=StateDelta.REGRESSED,
                    details={"observation": observation_text},
                )
            )
        return events
