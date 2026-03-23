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

