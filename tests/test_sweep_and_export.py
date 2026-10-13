from pathlib import Path

import yaml

from creditlab.analysis.exporters import export_results_bundle, render_paper_assets
from creditlab.analysis.sweep import SweepRunner
from creditlab.core.config import ExperimentConfig
from creditlab.core.models import ActionEvent
from creditlab.core.registry import create_env, create_store


def make_config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "name": "diagnostic_sweep_test",
            "research_objective": "exercise sweep and paper export",
            "env": {"name": "diagnostic_branch_lab", "split": "train", "config": {}},
            "policy": {"name": "prompted", "config": {"exploration_rate": 0.2, "seed": 7}},
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
                "collect_split": "train",
                "eval_split": "eval",
            },
            "paper_export": {
                "output_dir": str(tmp_path / "paper_out"),
                "manuscript_path": str(Path.cwd() / "paper" / "manuscript.md"),
            },
            "deployment": {"profile": "local_smoke", "platform": "local"},
            "sweep": {"enabled": True, "seeds": [7]},
            "experiment": {
                "seeds": [7],
                "budgets": {
                    "rollout_count": 3,
                    "max_turns": 4,
                    "max_prompt_tokens": 64,
                    "max_completion_tokens": 32,
                    "max_verifier_calls": 16,
                    "wall_clock_seconds": 30,
                },
            },
        }
    )


def test_sweep_aggregate_and_export(tmp_path: Path):
    config = make_config(tmp_path)
    sweep = SweepRunner(config).run()
    run_group_id = sweep["run_group_id"]
    store = create_store(config)
    bundle = export_results_bundle(run_group_id, store, config.paper_export.output_dir)
    rendered = render_paper_assets(run_group_id, store, config.paper_export.manuscript_path, config.paper_export.output_dir)

    assert sweep["aggregate"]["condition_summaries"]
    assert Path(bundle["files"][0]).exists()
    assert Path(rendered["generated_results"]).exists()


def test_manifest_environment_seed_controls_stochastic_transitions(tmp_path: Path):
    manifest_path = tmp_path / "stochastic.yaml"
    manifest_path.write_text(
        yaml.safe_dump(
            {
                "benchmark_name": "seed_probe",
                "revision": "v1",
                "state_abstraction_version": "v1",
                "verifier_version": "v1",
                "tasks": [
                    {
                        "task_id": "seed_probe",
                        "split": "smoke",
                        "instruction": "Take the stochastic edge.",
                        "initial_state": "start",
                        "states": {
                            "start": {
                                "phase": "start",
                                "mode": "ready",
                                "progress": 0,
                                "oracle_action": "advance",
                                "edges": {
                                    "advance": {
                                        "next_node": "success",
                                        "failure_node": "failure",
                                        "success_probability": 0.5,
                                        "observation": "advanced",
                                        "failure_observation": "blocked",
                                        "terminal": True,
                                        "success": True,
                                        "progress_delta": 1,
                                        "state_delta": "improved",
                                    }
                                },
                            },
                            "success": {
                                "phase": "done",
                                "mode": "success",
                                "progress": 1,
                                "edges": {},
                            },
                            "failure": {
                                "phase": "done",
                                "mode": "failure",
                                "progress": 0,
                                "edges": {},
                            },
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    config = make_config(tmp_path)
    config.env.name = "stochastic"
    config.env.split = "smoke"
    config.benchmark.manifest_path = manifest_path

    def destination(seed: int) -> str:
        seeded = config.model_copy(deep=True)
        seeded.env.config["seed"] = seed
        env = create_env(seeded)
        task = env.list_tasks("smoke")[0]
        state = env.reset(task)
        action = ActionEvent(action_id="advance", action_type="tool", content="advance")
        next_state, _, _ = env.step(task, state, action)
        return next_state.payload["raw_node"]

    assert destination(0) == destination(0) == "failure"
    assert destination(1) == destination(1) == "success"


def test_sweep_propagates_seed_to_policy_and_environment(monkeypatch, tmp_path: Path):
    config = make_config(tmp_path)
    runner = SweepRunner(config)
    captured: dict[str, int] = {}

    class FakeTrainer:
        def fit(self, run_ids, scorer_name, budget):
            return {"model_path": str(tmp_path / "model.json")}

    class FakeEngine:
        def __init__(self, env, policy, store, tracker):
            captured["environment"] = env
            captured["policy"] = policy

        def collect(self, config, run_context, split_override):
            captured["env_seed"] = config.env.config["seed"]
            captured["policy_seed"] = config.policy.config["seed"]
            return "run_seeded"

    monkeypatch.setattr("creditlab.analysis.sweep.create_trainer", lambda *args, **kwargs: FakeTrainer())
    monkeypatch.setattr("creditlab.analysis.sweep.create_env", lambda config: config.env.config["seed"])
    monkeypatch.setattr("creditlab.analysis.sweep.create_policy", lambda name, config, root: config["seed"])
    monkeypatch.setattr("creditlab.analysis.sweep.RolloutEngine", FakeEngine)

    run_id = runner._train_and_evaluate(
        run_group_id="group_seeded",
        seed=29,
        source_run="run_source",
        condition_name="branch_aware",
        scorer_name="branch_aware",
        trainer_name="branch_aware",
    )

    assert run_id == "run_seeded"
    assert captured == {"environment": 29, "policy": 29, "env_seed": 29, "policy_seed": 29}
