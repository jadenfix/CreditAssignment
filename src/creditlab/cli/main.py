from __future__ import annotations

import json
from pathlib import Path

import typer

from creditlab.analysis.reporting import build_replay_result, build_run_report, report_to_metric_rows
from creditlab.core.config import load_experiment_config
from creditlab.core.registry import create_env, create_policy, create_store, create_tracker, create_trainer
from creditlab.rollout.engine import RolloutEngine

app = typer.Typer(no_args_is_help=True)


@app.command()
def validate_config(config: str = typer.Option(..., "--config")) -> None:
    loaded = load_experiment_config(config)
    typer.echo(json.dumps(loaded.model_dump(mode="json"), indent=2, sort_keys=True))


@app.command()
def collect(config: str = typer.Option(..., "--config")) -> None:
    loaded = load_experiment_config(config)
    store = create_store(loaded)
    tracker = create_tracker(loaded.tracking.backend, loaded.tracking.config, loaded.store.artifact_root)
    env = create_env(loaded.env.name)
    policy = create_policy(loaded.policy.name, loaded.policy.config, loaded.store.artifact_root)
    run_id = RolloutEngine(env=env, policy=policy, store=store, tracker=tracker).collect(loaded)
    typer.echo(run_id)


@app.command()
def train(
    config: str = typer.Option(..., "--config"),
    run_id: list[str] = typer.Option([], "--run-id"),
) -> None:
    loaded = load_experiment_config(config)
    if loaded.trainer is None:
        raise typer.BadParameter("trainer config is required for training")
    store = create_store(loaded)
    trainer = create_trainer(loaded.trainer.name, store, loaded.store.artifact_root)
    source_runs = run_id or loaded.trainer.config.get("source_run_ids") or store.list_run_ids()[-1:]
    result = trainer.fit(source_runs, scorer_name="outcome_only", budget=loaded.experiment.budgets)
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@app.command()
def replay(
    run_id: str = typer.Option(..., "--run-id"),
    config: str = typer.Option(..., "--config"),
    scorer: str = typer.Option("branch_aware", "--scorer"),
) -> None:
    loaded = load_experiment_config(config)
    store = create_store(loaded)
    result = build_replay_result(run_id, store, scorer)
    typer.echo(json.dumps(result.model_dump(mode="json"), indent=2, sort_keys=True))


@app.command()
def report(
    run_id: str = typer.Option(..., "--run-id"),
    config: str = typer.Option(..., "--config"),
) -> None:
    loaded = load_experiment_config(config)
    store = create_store(loaded)
    tracker = create_tracker(loaded.tracking.backend, loaded.tracking.config, loaded.store.artifact_root)
    report_payload = build_run_report(run_id, store)
    rows = report_to_metric_rows(run_id, report_payload)
    tracker.log_metrics(run_id, rows)
    tracker.log_artifact(run_id, "report", report_payload)
    typer.echo(json.dumps(report_payload, indent=2, sort_keys=True))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
