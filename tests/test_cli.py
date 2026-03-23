from pathlib import Path

import yaml
from typer.testing import CliRunner

from creditlab.cli.main import app


def test_cli_validate_collect_and_host_check(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    payload = {
        "name": "diagnostic_smoke_cli",
        "research_objective": "cli test",
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
        "experiment": {
            "seeds": [7],
            "budgets": {
                "rollout_count": 4,
                "max_turns": 4,
                "max_prompt_tokens": 64,
                "max_completion_tokens": 32,
                "max_verifier_calls": 16,
                "wall_clock_seconds": 30,
            },
        },
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    runner = CliRunner()

    validate = runner.invoke(app, ["validate-config", "--config", str(config_path)])
    collect = runner.invoke(app, ["collect", "--config", str(config_path)])
    host = runner.invoke(app, ["host-check"])

    assert validate.exit_code == 0
    assert collect.exit_code == 0
    assert host.exit_code == 0
    run_id = collect.stdout.strip()
    assert run_id.startswith("run_")
    assert (tmp_path / "artifacts" / "runs" / run_id / "report.json").exists()


def test_cli_collect_fails_closed_for_aws_without_opt_in(tmp_path: Path):
    config_path = tmp_path / "aws_config.yaml"
    payload = {
        "name": "diagnostic_aws_cli",
        "research_objective": "cli aws safety test",
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
        "deployment": {"profile": "aws_gpu_dev", "platform": "aws_gpu", "aws_instance_profile": "g6e.4xlarge"},
        "sweep": {
            "safety": {
                "allow_aws_execution": False,
                "max_total_runs": 8,
                "max_failed_runs": 1,
                "max_retry_attempts": 1,
                "retry_backoff_seconds": 0.0,
                "max_total_prompt_tokens": 10_000,
                "max_total_completion_tokens": 5_000,
                "max_total_verifier_calls": 500,
                "max_total_wall_clock_seconds": 300,
                "max_estimated_cost_usd": 1.0,
            }
        },
        "experiment": {
            "seeds": [7],
            "budgets": {
                "rollout_count": 4,
                "max_turns": 4,
                "max_prompt_tokens": 64,
                "max_completion_tokens": 32,
                "max_verifier_calls": 16,
                "wall_clock_seconds": 30,
            },
        },
    }
    config_path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    runner = CliRunner()

    collect = runner.invoke(app, ["collect", "--config", str(config_path)])

    assert collect.exit_code == 1
    assert "allow_aws_execution" in collect.output
