from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

from creditlab.core.models import BudgetConfig


class StoreConfig(BaseModel):
    sqlite_path: Path = Path("runs/creditlab.sqlite")
    artifact_root: Path = Path("artifacts")


class ComponentConfig(BaseModel):
    name: str
    config: dict[str, Any] = Field(default_factory=dict)


class EnvConfig(ComponentConfig):
    split: str = "smoke"


class TrackingConfig(BaseModel):
    backend: str = "local"
    config: dict[str, Any] = Field(default_factory=dict)


class BenchmarkConfig(BaseModel):
    name: str | None = None
    revision: str = "v1"
    manifest_path: Path | None = None
    collect_split: str | None = None
    eval_split: str | None = None
    task_limit: int | None = None
    state_abstraction_version: str = "v1"
    verifier_version: str = "v1"


class BackendConfig(BaseModel):
    name: str
    config: dict[str, Any] = Field(default_factory=dict)


class DeploymentConfig(BaseModel):
    profile: str = "local_smoke"
    platform: str = "local"
    container_image: str | None = None
    workspace_dir: str | None = None
    data_dir: str | None = None
    aws_instance_profile: str | None = None
    s3_sync_uri: str | None = None


class PaperExportConfig(BaseModel):
    output_dir: Path = Path("artifacts/paper")
    manuscript_path: Path = Path("paper/manuscript.md")
    bundle_name: str = "results_bundle"


class SweepConditionConfig(BaseModel):
    name: str
    scorer: str | None = None
    trainer_name: str | None = None
    matched_budget: bool = False


class SweepSafetyConfig(BaseModel):
    allow_aws_execution: bool = False
    require_healthy_gpu_for_aws: bool = True
    max_total_runs: int = 32
    max_failed_runs: int = 2
    max_retry_attempts: int = 2
    retry_backoff_seconds: float = 1.0
    stop_on_first_failure: bool = False
    max_total_prompt_tokens: int = 250_000
    max_total_completion_tokens: int = 150_000
    max_total_verifier_calls: int = 50_000
    max_total_wall_clock_seconds: float = 14_400.0
    max_estimated_cost_usd: float = 25.0
    estimated_prompt_cost_per_1k: float = 0.0
    estimated_completion_cost_per_1k: float = 0.0
    estimated_instance_cost_per_hour: float = 0.0

    @field_validator("max_total_runs", "max_retry_attempts")
    @classmethod
    def _positive_ints(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("value must be positive")
        return value

    @field_validator("max_failed_runs")
    @classmethod
    def _non_negative_failures(cls, value: int) -> int:
        if value < 0:
            raise ValueError("max_failed_runs must be non-negative")
        return value

    @field_validator(
        "retry_backoff_seconds",
        "max_total_wall_clock_seconds",
        "max_estimated_cost_usd",
        "estimated_prompt_cost_per_1k",
        "estimated_completion_cost_per_1k",
        "estimated_instance_cost_per_hour",
    )
    @classmethod
    def _non_negative_floats(cls, value: float) -> float:
        if value < 0:
            raise ValueError("value must be non-negative")
        return value

    @field_validator(
        "max_total_prompt_tokens",
        "max_total_completion_tokens",
        "max_total_verifier_calls",
    )
    @classmethod
    def _positive_budget_limits(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("budget ceilings must be positive")
        return value


class SweepConfig(BaseModel):
    enabled: bool = False
    run_group_id: str | None = None
    seeds: list[int] = Field(default_factory=list)
    resume: bool = True
    conditions: list[SweepConditionConfig] = Field(default_factory=list)
    safety: SweepSafetyConfig = Field(default_factory=SweepSafetyConfig)


class ExperimentControl(BaseModel):
    seeds: list[int] = Field(default_factory=lambda: [7])
    budgets: BudgetConfig


class ExperimentConfig(BaseModel):
    name: str
    research_objective: str
    env: EnvConfig
    policy: ComponentConfig
    trainer: ComponentConfig | None = None
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    store: StoreConfig = Field(default_factory=StoreConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    model_backend: BackendConfig = Field(default_factory=lambda: BackendConfig(name="local_heuristic"))
    training_backend: BackendConfig = Field(default_factory=lambda: BackendConfig(name="local_table"))
    deployment: DeploymentConfig = Field(default_factory=DeploymentConfig)
    paper_export: PaperExportConfig = Field(default_factory=PaperExportConfig)
    sweep: SweepConfig = Field(default_factory=SweepConfig)
    experiment: ExperimentControl

    @model_validator(mode="after")
    def _fill_defaults(self) -> "ExperimentConfig":
        if self.benchmark.name is None:
            self.benchmark.name = self.env.name
        if self.benchmark.collect_split is None:
            self.benchmark.collect_split = self.env.split
        if self.benchmark.eval_split is None:
            self.benchmark.eval_split = self.env.split
        if not self.sweep.conditions:
            self.sweep.conditions = [
                SweepConditionConfig(name="prompted"),
                SweepConditionConfig(name="outcome_only", scorer="outcome_only", trainer_name="outcome_only"),
                SweepConditionConfig(name="branch_aware", scorer="branch_aware", trainer_name="branch_aware"),
                SweepConditionConfig(name="hindsight", scorer="hindsight", trainer_name="hindsight"),
                SweepConditionConfig(name="combined", scorer="combined", trainer_name="combined"),
                SweepConditionConfig(
                    name="combined_matched_budget",
                    scorer="combined",
                    trainer_name="combined",
                    matched_budget=True,
                ),
            ]
        return self


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return ExperimentConfig.model_validate(payload)
