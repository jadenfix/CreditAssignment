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
    experiment: ExperimentControl

    @model_validator(mode="after")
    def _require_budgets(self) -> "ExperimentConfig":
        _ = self.experiment.budgets
        return self


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return ExperimentConfig.model_validate(payload)
