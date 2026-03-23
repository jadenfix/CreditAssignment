from pathlib import Path

from creditlab.analysis.reporting import build_replay_result, build_run_report
from creditlab.core.config import ExperimentConfig
from creditlab.core.registry import create_env, create_policy, create_store, create_tracker
from creditlab.rollout.engine import RolloutEngine


def make_config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "name": "diagnostic_smoke_test",
            "research_objective": "test replayable branch-aware rescoring",
            "env": {"name": "diagnostic_branch_lab", "split": "smoke", "config": {}},
            "policy": {"name": "prompted", "config": {"exploration_rate": 0.15, "seed": 7}},
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
                "seeds": [7, 11],
                "budgets": {
                    "rollout_count": 4,
                    "max_turns": 4,
                    "max_prompt_tokens": 128,
                    "max_completion_tokens": 64,
                    "max_verifier_calls": 32,
                    "wall_clock_seconds": 30,
                },
            },
        }
    )


def test_collect_replay_and_report_end_to_end(tmp_path: Path):
    config = make_config(tmp_path)
    store = create_store(config)
    tracker = create_tracker(config.tracking.backend, config.tracking.config, config.store.artifact_root)
    env = create_env(config)
    policy = create_policy(config.policy.name, config.policy.config, config.store.artifact_root)
    run_id = RolloutEngine(env=env, policy=policy, store=store, tracker=tracker).collect(config)

    episodes = store.load_episodes(run_id)
    report = build_run_report(run_id, store)
    replay = build_replay_result(run_id, store, "branch_aware")

    assert len(episodes) == 4
    assert report["summary"]["rollout_count"] == 4
    assert "top_repeated_dead_end_branches" in report["failure_analysis"]
    assert replay.summary["states_compared"] >= 0
    assert "change_rate" in replay.summary
    assert (tmp_path / "artifacts" / "runs" / run_id / "report.json").exists()
