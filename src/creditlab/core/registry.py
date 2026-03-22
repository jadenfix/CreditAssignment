from __future__ import annotations

from pathlib import Path

from creditlab.core.config import ExperimentConfig
from creditlab.envs.diagnostic import DiagnosticBranchLabEnv
from creditlab.envs.webshop import WebShopSmokeEnv
from creditlab.policies.prompted import PromptedPolicyBackend
from creditlab.policies.table import TablePolicyBackend
from creditlab.store.sqlite_store import SqliteTrajectoryStore
from creditlab.tracking.local import LocalTracker
from creditlab.tracking.wandb_adapter import OptionalWandbTracker
from creditlab.trainers.outcome_only import OutcomeOnlyTrainer


def create_env(name: str):
    if name == "diagnostic_branch_lab":
        return DiagnosticBranchLabEnv()
    if name == "webshop":
        return WebShopSmokeEnv()
    raise ValueError(f"unknown environment: {name}")


def create_policy(name: str, config: dict, artifact_root: Path):
    if name == "prompted":
        return PromptedPolicyBackend(config=config)
    if name == "table":
        return TablePolicyBackend(config=config, artifact_root=artifact_root)
    raise ValueError(f"unknown policy: {name}")


def create_tracker(backend: str, config: dict, artifact_root: Path):
    if backend == "local":
        return LocalTracker(artifact_root=artifact_root)
    if backend == "wandb":
        return OptionalWandbTracker(config=config, artifact_root=artifact_root)
    raise ValueError(f"unknown tracking backend: {backend}")


def create_store(config: ExperimentConfig) -> SqliteTrajectoryStore:
    return SqliteTrajectoryStore(
        sqlite_path=config.store.sqlite_path,
        artifact_root=config.store.artifact_root,
    )


def create_trainer(name: str, store: SqliteTrajectoryStore, artifact_root: Path):
    if name == "outcome_only":
        return OutcomeOnlyTrainer(store=store, artifact_root=artifact_root)
    raise ValueError(f"unknown trainer: {name}")

