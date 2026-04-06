from __future__ import annotations

from pathlib import Path

from creditlab.core.config import ExperimentConfig
from creditlab.envs.alfworld import ALFWorldEnv
from creditlab.envs.diagnostic import DiagnosticBranchLabEnv
from creditlab.envs.stochastic import StochasticEnv
from creditlab.envs.webshop import WebShopEnv
from creditlab.policies.prompted import PromptedPolicyBackend, VLLMServerPolicyBackend
from creditlab.policies.table import TablePolicyBackend
from creditlab.store.sqlite_store import SqliteTrajectoryStore
from creditlab.tracking.local import LocalTracker
from creditlab.tracking.wandb_adapter import OptionalWandbTracker
from creditlab.trainers.outcome_only import (
    BranchAwareTrainer,
    CombinedTrainer,
    HindsightTrainer,
    OutcomeOnlyTrainer,
)


def create_env(config: ExperimentConfig):
    manifest = config.benchmark.manifest_path
    task_limit = config.benchmark.task_limit
    if config.env.name == "diagnostic_branch_lab":
        return DiagnosticBranchLabEnv(manifest_path=manifest, task_limit=task_limit)
    if config.env.name == "webshop":
        return WebShopEnv(manifest_path=manifest, task_limit=task_limit)
    if config.env.name == "alfworld":
        return ALFWorldEnv(manifest_path=manifest, task_limit=task_limit)
    if config.env.name == "stochastic":
        return StochasticEnv(manifest_path=manifest, task_limit=task_limit)
    if config.env.name == "alfworld_real":
        from creditlab.envs.alfworld_real import RealALFWorldEnv
        return RealALFWorldEnv(
            config=config.env.config,
            task_limit=task_limit,
        )
    raise ValueError(f"unknown environment: {config.env.name}")


def create_policy(name: str, config: dict, artifact_root: Path):
    if name == "prompted":
        return PromptedPolicyBackend(config=config)
    if name == "table":
        return TablePolicyBackend(config=config, artifact_root=artifact_root)
    if name == "vllm_server":
        return VLLMServerPolicyBackend(config=config)
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
    if name == "branch_aware":
        return BranchAwareTrainer(store=store, artifact_root=artifact_root)
    if name == "hindsight":
        return HindsightTrainer(store=store, artifact_root=artifact_root)
    if name == "combined":
        return CombinedTrainer(store=store, artifact_root=artifact_root)
    raise ValueError(f"unknown trainer: {name}")
