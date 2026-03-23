from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

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


class SweepConfig(BaseModel):
    enabled: bool = False
    run_group_id: str | None = None
    seeds: list[int] = Field(default_factory=list)
    resume: bool = True
    conditions: list[SweepConditionConfig] = Field(default_factory=list)


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
