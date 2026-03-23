from __future__ import annotations

import urllib.error
from pathlib import Path

import pytest
import yaml

from creditlab.analysis.safety import SafetyViolation, SweepExecutionGuard
from creditlab.analysis.sweep import SweepRunner
from creditlab.core.config import load_experiment_config
from creditlab.core.models import BudgetConfig, StateSnapshot, TaskSpec
from creditlab.policies.prompted import VLLMServerPolicyBackend
from creditlab.store.sqlite_store import SqliteTrajectoryStore


def _base_payload(tmp_path: Path) -> dict:
    return {
        "name": "diagnostic_safety_test",
        "research_objective": "safety test",
        "env": {"name": "diagnostic_branch_lab", "split": "smoke", "config": {}},
        "policy": {"name": "prompted", "config": {"exploration_rate": 0.1, "seed": 7}},
        "trainer": {"name": "outcome_only", "config": {}},
        "tracking": {"backend": "local", "config": {}},
        "store": {
            "sqlite_path": str(tmp_path / "runs" / "creditlab.sqlite"),
            "artifact_root": str(tmp_path / "artifacts"),
        },
        "benchmark": {
            "name": "diagnostic",
            "revision": "v1",
            "manifest_path": str(Path.cwd() / "benchmarks" / "diagnostic" / "v1" / "tasks.yaml"),
            "collect_split": "smoke",
            "eval_split": "smoke",
        },
        "deployment": {"profile": "local_smoke", "platform": "local"},
        "sweep": {
            "enabled": True,
            "seeds": [7],
            "safety": {
                "max_total_runs": 8,
                "max_failed_runs": 1,
                "max_retry_attempts": 1,
                "retry_backoff_seconds": 0.0,
                "max_total_prompt_tokens": 10_000,
                "max_total_completion_tokens": 5_000,
                "max_total_verifier_calls": 500,
                "max_total_wall_clock_seconds": 300,
                "max_estimated_cost_usd": 1.0,
            },
        },
        "experiment": {
            "seeds": [7],
            "budgets": {
                "rollout_count": 2,
                "max_turns": 3,
                "max_prompt_tokens": 32,
                "max_completion_tokens": 16,
                "max_verifier_calls": 8,
                "wall_clock_seconds": 10,
            },
        },
    }


def test_sweep_fails_closed_for_aws_without_explicit_opt_in(tmp_path: Path):
    payload = _base_payload(tmp_path)
    payload["deployment"] = {
        "profile": "aws_gpu_dev",
        "platform": "aws_gpu",
        "container_image": "creditlab-gpu:latest",
        "aws_instance_profile": "g6e.4xlarge",
    }
    payload["sweep"]["safety"]["allow_aws_execution"] = False
    config_path = tmp_path / "aws_disabled.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    loaded = load_experiment_config(config_path)

    with pytest.raises(SafetyViolation, match="allow_aws_execution"):
        SweepRunner(loaded).run()


def test_sweep_guard_blocks_projected_budget_overrun(tmp_path: Path):
    payload = _base_payload(tmp_path)
    payload["sweep"]["safety"]["max_total_prompt_tokens"] = 100
    config_path = tmp_path / "budget_guard.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    loaded = load_experiment_config(config_path)
    store = SqliteTrajectoryStore(loaded.store.sqlite_path, loaded.store.artifact_root)
    guard = SweepExecutionGuard(loaded, store)
    guard.preflight(planned_runs=1)

    with pytest.raises(SafetyViolation, match="prompt tokens"):
        guard.run_with_retries(
            "seed=7/collection/prompted_base",
            loaded.experiment.budgets,
            lambda: "run_unused",
        )


def test_vllm_policy_retries_then_falls_back(monkeypatch: pytest.MonkeyPatch):
    calls = {"count": 0}

    def _always_fail(*args, **kwargs):
        calls["count"] += 1
        raise urllib.error.URLError("offline")

    monkeypatch.setattr("urllib.request.urlopen", _always_fail)
    policy = VLLMServerPolicyBackend(config={"max_retries": 2, "retry_backoff_seconds": 0.0})
    task = TaskSpec(task_id="task_1", env_name="diagnostic", instruction="Inspect then fix", split="smoke")
    state = StateSnapshot(
        state_hash="state_1",
        abstract_state={"mode": "start"},
        payload={"available_actions": ["inspect", "noop"], "oracle_action": "inspect"},
    )
    budget = BudgetConfig(
        rollout_count=1,
        max_turns=2,
        max_prompt_tokens=32,
        max_completion_tokens=16,
        max_verifier_calls=4,
        wall_clock_seconds=5,
    )

    action = policy.act(history=[], state=state, budget=budget, task=task)

    assert calls["count"] == 3
    assert action.content == "inspect"
    assert action.metadata["fallback_used"] is True
    assert action.metadata["attempts"] == 3
    assert "URLError" in action.metadata["fallback_reason"]


def test_local_smoke_sweep_runs_with_safety_defaults(tmp_path: Path):
    payload = _base_payload(tmp_path)
    config_path = tmp_path / "local_sweep.yaml"
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    loaded = load_experiment_config(config_path)

    result = SweepRunner(loaded).run()

    assert result["run_group_id"].startswith("group_")
    assert result["safety"]["halted"] is False
    assert result["safety"]["totals"]["completed_runs"] >= 2
