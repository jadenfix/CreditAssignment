from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EdgeManifest(BaseModel):
    next_node: str
    observation: str
    terminal: bool = False
    success: bool = False
    progress_delta: float = 0.0
    state_delta: str = "neutral"
    branch_label: str = "neutral"
    dead_end: bool = False
    enabling: bool = False
    recovery: bool = False
    verifier_signals: dict[str, float] = Field(default_factory=dict)
    success_probability: float = 1.0
    failure_node: str | None = None
    failure_observation: str = "The action did not succeed this time."


class StateManifest(BaseModel):
    phase: str
    mode: str
    progress: float = 0.0
    abstract_fields: dict[str, Any] = Field(default_factory=dict)
    oracle_action: str | None = None
    edges: dict[str, EdgeManifest] = Field(default_factory=dict)


class GraphTaskManifest(BaseModel):
    task_id: str
    split: str
    instruction: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    initial_state: str = "start"
    states: dict[str, StateManifest]


class BenchmarkManifest(BaseModel):
    benchmark_name: str
    revision: str = "v1"
    state_abstraction_version: str = "v1"
    verifier_version: str = "v1"
    tasks: list[GraphTaskManifest]
