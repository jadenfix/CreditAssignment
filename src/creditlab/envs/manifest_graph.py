from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import yaml

from creditlab.core.interfaces import EnvAdapter
from creditlab.core.manifest_models import BenchmarkManifest, EdgeManifest
from creditlab.core.models import ActionEvent, StateDelta, StateSnapshot, TaskSpec, VerifierEvent, VerifierStatus
from creditlab.core.utils import stable_hash


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


class GraphManifestEnv(EnvAdapter):
    env_name = "graph_manifest"
    benchmark_name = "graph_manifest"
    default_manifest = Path("benchmarks/diagnostic/v1/tasks.yaml")

    def __init__(self, manifest_path: str | Path | None = None, task_limit: int | None = None, seed: int = 0) -> None:
        path = Path(manifest_path) if manifest_path else repo_root() / self.default_manifest
        with path.open("r", encoding="utf-8") as handle:
            self.manifest = BenchmarkManifest.model_validate(yaml.safe_load(handle))
        self.task_limit = task_limit
        self.task_definitions = {task.task_id: task for task in self.manifest.tasks}
        self._rng = random.Random(seed)

    def list_tasks(self, split: str) -> list[TaskSpec]:
        tasks = [
            TaskSpec(
                task_id=task.task_id,
                env_name=self.env_name,
                instruction=task.instruction,
                split=split,
                metadata={
                    **task.metadata,
                    "benchmark_name": self.manifest.benchmark_name,
                    "benchmark_revision": self.manifest.revision,
                    "state_abstraction_version": self.manifest.state_abstraction_version,
                    "verifier_version": self.manifest.verifier_version,
                },
            )
            for task in self.manifest.tasks
            if task.split == split
        ]
        if self.task_limit is not None:
            tasks = tasks[: self.task_limit]
        return tasks

    def reset(self, task: TaskSpec) -> StateSnapshot:
        definition = self.task_definitions[task.task_id]
        return self._snapshot_from_node(task, definition.initial_state, [])

    def step(self, task: TaskSpec, state: StateSnapshot, action: ActionEvent) -> tuple[StateSnapshot, str, bool]:
        definition = self.task_definitions[task.task_id]
        current = definition.states[state.payload["raw_node"]]
        edge = current.edges.get(action.content)
        if edge is None:
            invalid = StateSnapshot(
                state_hash=stable_hash({**state.abstract_state, "mode": "invalid"}),
                abstract_state={**state.abstract_state, "mode": "invalid"},
                payload={
                    **state.payload,
                    "history": list(state.payload["history"]) + [action.content],
                    "invalid_action": action.content,
                },
            )
            return invalid, "That action is invalid from the current state.", False
        updated_history = list(state.payload["history"]) + [action.content]
        if edge.success_probability < 1.0 and self._rng.random() > edge.success_probability:
            fail_node = edge.failure_node or state.payload["raw_node"]
            next_state = self._snapshot_from_node(task, fail_node, updated_history)
            return next_state, edge.failure_observation, False
        next_state = self._snapshot_from_node(task, edge.next_node, updated_history)
        return next_state, edge.observation, edge.terminal

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
        current = definition.states[before.payload["raw_node"]]
        edge = current.edges.get(action.content)
        if edge is None:
            return [
                VerifierEvent(
                    verifier_id=f"{task.task_id}:validity",
                    status=VerifierStatus.FAIL,
                    signal_name="action_validity",
                    score=-1.0,
                    state_delta=StateDelta.REGRESSED,
                    details={"invalid_action": action.content},
                )
            ]
        events = [
            VerifierEvent(
                verifier_id=f"{task.task_id}:progress",
                status=VerifierStatus.PASS if edge.progress_delta >= 0 else VerifierStatus.WARN,
                signal_name="progress",
                score=float(edge.progress_delta),
                state_delta=StateDelta(edge.state_delta),
                details={
                    "branch_label": edge.branch_label,
                    "dead_end": edge.dead_end,
                    "enabling": edge.enabling,
                    "recovery": edge.recovery,
                },
            )
        ]
        if edge.success or edge.terminal:
            events.append(
                VerifierEvent(
                    verifier_id=f"{task.task_id}:success",
                    status=VerifierStatus.PASS if edge.success else VerifierStatus.FAIL,
                    signal_name="success",
                    score=1.0 if edge.success else 0.0,
                    state_delta=StateDelta.IMPROVED if edge.success else StateDelta.REGRESSED,
                    details={"terminal": edge.terminal},
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
        for signal_name, score in edge.verifier_signals.items():
            events.append(
                VerifierEvent(
                    verifier_id=f"{task.task_id}:{signal_name}",
                    status=VerifierStatus.INFO if score == 0 else VerifierStatus.PASS if score > 0 else VerifierStatus.WARN,
                    signal_name=signal_name,
                    score=float(score),
                    state_delta=StateDelta.NEUTRAL if score == 0 else StateDelta.IMPROVED if score > 0 else StateDelta.REGRESSED,
                    details={"edge": action.content},
                )
            )
        return events

    def _snapshot_from_node(self, task: TaskSpec, node_name: str, history: list[str]) -> StateSnapshot:
        definition = self.task_definitions[task.task_id]
        node = definition.states[node_name]
        abstract_state = {
            "phase": node.phase,
            "mode": node.mode,
            "progress": node.progress,
            **node.abstract_fields,
        }
        payload = {
            "raw_node": node_name,
            "history": history,
            "available_actions": sorted(node.edges),
            "oracle_action": node.oracle_action,
            "phase": node.phase,
            "mode": node.mode,
            "progress": node.progress,
            "benchmark_name": self.manifest.benchmark_name,
        }
        return StateSnapshot(
            state_hash=stable_hash(abstract_state),
            abstract_state=abstract_state,
            payload=payload,
        )


def _default_manifest_path(rel_path: str) -> Path:
    return repo_root() / rel_path


class DiagnosticBranchLabEnv(GraphManifestEnv):
    env_name = "diagnostic_branch_lab"
    benchmark_name = "diagnostic"
    default_manifest = Path("benchmarks/diagnostic/v1/tasks.yaml")


class WebShopEnv(GraphManifestEnv):
    env_name = "webshop"
    benchmark_name = "webshop"
    default_manifest = Path("benchmarks/webshop/v1/tasks.yaml")


class ALFWorldEnv(GraphManifestEnv):
    env_name = "alfworld"
    benchmark_name = "alfworld"
    default_manifest = Path("benchmarks/alfworld/v1/tasks.yaml")


class StochasticEnv(GraphManifestEnv):
    env_name = "stochastic"
    benchmark_name = "stochastic"
    default_manifest = Path("benchmarks/stochastic/v1/tasks.yaml")



