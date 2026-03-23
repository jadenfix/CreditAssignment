from pathlib import Path

from creditlab.analysis.exporters import export_results_bundle, render_paper_assets
from creditlab.analysis.sweep import SweepRunner
from creditlab.core.config import ExperimentConfig
from creditlab.core.registry import create_store


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
