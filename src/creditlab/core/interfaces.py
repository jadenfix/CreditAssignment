from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence

from creditlab.core.models import (
    ActionEvent,
    BudgetConfig,
    EpisodeRun,
    MetricRow,
    ReplayResult,
    StateSnapshot,
    TaskSpec,
    VerifierEvent,
)


class EnvAdapter(ABC):
    env_name: str

    @abstractmethod
    def list_tasks(self, split: str) -> list[TaskSpec]:
        raise NotImplementedError

    @abstractmethod
    def reset(self, task: TaskSpec) -> StateSnapshot:
        raise NotImplementedError

    @abstractmethod
    def step(self, task: TaskSpec, state: StateSnapshot, action: ActionEvent) -> tuple[StateSnapshot, str, bool]:
        raise NotImplementedError

    @abstractmethod
    def snapshot(self, state: StateSnapshot) -> StateSnapshot:
        raise NotImplementedError

    @abstractmethod
    def abstract_state(self, state: StateSnapshot) -> dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def verify_transition(
        self,
        task: TaskSpec,
        before: StateSnapshot,
        action: ActionEvent,
        after: StateSnapshot,
        observation_text: str,
    ) -> list[VerifierEvent]:
        raise NotImplementedError


class PolicyBackend(ABC):
    policy_name: str

    @abstractmethod
    def act(
        self,
        history: Sequence[Any],
        state: StateSnapshot,
        budget: BudgetConfig,
        task: TaskSpec,
    ) -> ActionEvent:
        raise NotImplementedError


class TrainerBackend(ABC):
    trainer_name: str

    @abstractmethod
    def fit(self, run_ids: Sequence[str], scorer_name: str, budget: BudgetConfig) -> dict[str, Any]:
        raise NotImplementedError


class TrajectoryStore(ABC):
    @abstractmethod
    def create_run(self, config: dict[str, Any], run_id: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def write_episode(self, episode: EpisodeRun) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_episodes(self, run_id: str) -> list[EpisodeRun]:
        raise NotImplementedError

    @abstractmethod
    def list_run_ids(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def write_metrics(self, rows: Sequence[MetricRow]) -> None:
        raise NotImplementedError

    @abstractmethod
    def load_metrics(self, run_id: str) -> list[MetricRow]:
        raise NotImplementedError

    @abstractmethod
    def write_replay(self, result: ReplayResult) -> Path:
        raise NotImplementedError


class TrackerBackend(ABC):
    tracker_name: str

    @abstractmethod
    def log_metrics(self, run_id: str, rows: Sequence[MetricRow]) -> None:
        raise NotImplementedError

    @abstractmethod
    def log_artifact(self, run_id: str, name: str, payload: dict[str, Any]) -> Path:
        raise NotImplementedError

