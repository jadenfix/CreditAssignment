from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


def utc_now() -> datetime:
    return datetime.now(tz=UTC)


class VerifierStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    INFO = "info"


class StateDelta(str, Enum):
    IMPROVED = "improved"
    REGRESSED = "regressed"
    NEUTRAL = "neutral"


class ArtifactRef(BaseModel):
    path: str
    media_type: str | None = None


class BudgetConfig(BaseModel):
    rollout_count: int
    max_turns: int
    max_prompt_tokens: int
    max_completion_tokens: int
    max_verifier_calls: int
    wall_clock_seconds: float | None = None

    @field_validator(
        "rollout_count",
        "max_turns",
        "max_prompt_tokens",
        "max_completion_tokens",
        "max_verifier_calls",
    )
    @classmethod
    def _positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("budget values must be positive")
        return value

    @field_validator("wall_clock_seconds")
    @classmethod
    def _positive_wall_clock(cls, value: float | None) -> float | None:
        if value is not None and value <= 0:
            raise ValueError("wall_clock_seconds must be positive when provided")
        return value


class TaskSpec(BaseModel):
    task_id: str
    env_name: str
    instruction: str
    split: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class StateSnapshot(BaseModel):
    state_hash: str
    abstract_state: dict[str, Any]
    payload: dict[str, Any]


class ActionEvent(BaseModel):
    action_id: str
    action_type: str
    content: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ObservationEvent(BaseModel):
    text: str
    latency_ms: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class VerifierEvent(BaseModel):
    verifier_id: str
    status: VerifierStatus
    signal_name: str
    score: float
    state_delta: StateDelta
    latency_ms: float = 0.0
    cost_tokens: int = 0
    details: dict[str, Any] = Field(default_factory=dict)
    artifact_ref: ArtifactRef | None = None


class TreeNode(BaseModel):
    node_id: str
    episode_id: str
    task_id: str
    turn_index: int
    parent_node_id: str | None = None
    state: StateSnapshot
    metadata: dict[str, Any] = Field(default_factory=dict)


class Transition(BaseModel):
    transition_id: str
    episode_id: str
    parent_node_id: str
    child_node_id: str
    turn_index: int
    action: ActionEvent
    observation: ObservationEvent
    verifier_events: list[VerifierEvent] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EpisodeRun(BaseModel):
    run_id: str
    episode_id: str
    experiment_name: str
    env_name: str
    policy_name: str
    task: TaskSpec
    budget: BudgetConfig
    root_node_id: str
    nodes: list[TreeNode]
    transitions: list[Transition]
    success: bool
    outcome: str
    started_at: datetime = Field(default_factory=utc_now)
    completed_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_root(self) -> "EpisodeRun":
        node_ids = {node.node_id for node in self.nodes}
        if self.root_node_id not in node_ids:
            raise ValueError("root_node_id must exist in nodes")
        return self


class MetricRow(BaseModel):
    run_id: str
    scope: str
    name: str
    value: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class ReplayPreferenceShift(BaseModel):
    state_hash: str
    abstract_state: dict[str, Any]
    outcome_only_action: str
    replay_action: str
    changed: bool
    margin: float
    support: int


class ReplayResult(BaseModel):
    run_id: str
    scorer_name: str
    preference_shifts: list[ReplayPreferenceShift]
    metrics: list[MetricRow]
    changed_examples: list[dict[str, Any]]
    summary: dict[str, Any] = Field(default_factory=dict)

