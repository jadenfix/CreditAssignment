from __future__ import annotations

import json
from pathlib import Path

import typer

from creditlab.analysis.exporters import export_results_bundle, render_paper_assets
from creditlab.analysis.reporting import aggregate_run_group, build_replay_result, build_run_report, report_to_metric_rows
from creditlab.analysis.safety import SafetyViolation, SweepExecutionGuard
from creditlab.analysis.sweep import SweepRunner
from creditlab.core.config import load_experiment_config
from creditlab.core.models import RunContext
from creditlab.core.registry import create_env, create_policy, create_store, create_tracker, create_trainer
from creditlab.deployment.host_check import build_host_report, write_host_report
from creditlab.rollout.engine import RolloutEngine

app = typer.Typer(no_args_is_help=True)


@app.command()
def validate_config(config: str = typer.Option(..., "--config")) -> None:
    loaded = load_experiment_config(config)
    typer.echo(json.dumps(loaded.model_dump(mode="json"), indent=2, sort_keys=True))


@app.command()
def collect(config: str = typer.Option(..., "--config")) -> None:
    loaded = load_experiment_config(config)
    if loaded.deployment.platform == "aws_gpu":
        try:
            SweepExecutionGuard(loaded, store=None).preflight(planned_runs=1)
        except SafetyViolation as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
    store = create_store(loaded)
    tracker = create_tracker(loaded.tracking.backend, loaded.tracking.config, loaded.store.artifact_root)
    env = create_env(loaded)
    policy = create_policy(loaded.policy.name, loaded.policy.config, loaded.store.artifact_root)
    context = RunContext(
        benchmark_name=loaded.benchmark.name,
        benchmark_revision=loaded.benchmark.revision,
        model_backend=loaded.model_backend.name,
        training_backend=loaded.training_backend.name,
        deployment_profile=loaded.deployment.profile,
        container_image_tag=loaded.deployment.container_image,
        aws_instance_profile=loaded.deployment.aws_instance_profile,
    )
    run_id = RolloutEngine(env=env, policy=policy, store=store, tracker=tracker).collect(loaded, run_context=context)
    typer.echo(run_id)


@app.command()
def train(
    config: str = typer.Option(..., "--config"),
    run_id: list[str] = typer.Option([], "--run-id"),
    scorer: str = typer.Option("outcome_only", "--scorer"),
) -> None:
    loaded = load_experiment_config(config)
    if loaded.trainer is None:
        raise typer.BadParameter("trainer config is required for training")
    store = create_store(loaded)
    trainer = create_trainer(loaded.trainer.name, store, loaded.store.artifact_root)
    source_runs = run_id or loaded.trainer.config.get("source_run_ids") or store.list_run_ids()[-1:]
    result = trainer.fit(source_runs, scorer_name=scorer, budget=loaded.experiment.budgets)
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


@app.command()
def evaluate(config: str = typer.Option(..., "--config")) -> None:
    loaded = load_experiment_config(config)
    if loaded.deployment.platform == "aws_gpu":
        try:
            SweepExecutionGuard(loaded, store=None).preflight(planned_runs=1)
        except SafetyViolation as exc:
            typer.echo(str(exc), err=True)
            raise typer.Exit(code=1) from exc
    store = create_store(loaded)
    tracker = create_tracker(loaded.tracking.backend, loaded.tracking.config, loaded.store.artifact_root)
    env = create_env(loaded)
    policy = create_policy(loaded.policy.name, loaded.policy.config, loaded.store.artifact_root)
    context = RunContext(
        condition_name=loaded.policy.name,
        run_type="evaluation",
        benchmark_name=loaded.benchmark.name,
        benchmark_revision=loaded.benchmark.revision,
        model_backend=loaded.model_backend.name,
        training_backend=loaded.training_backend.name,
        deployment_profile=loaded.deployment.profile,
        container_image_tag=loaded.deployment.container_image,
        aws_instance_profile=loaded.deployment.aws_instance_profile,
    )
    run_id = RolloutEngine(env=env, policy=policy, store=store, tracker=tracker).collect(
        loaded,
        run_context=context,
        split_override=loaded.benchmark.eval_split,
    )
    typer.echo(run_id)


@app.command()
def sweep(config: str = typer.Option(..., "--config")) -> None:
    loaded = load_experiment_config(config)
    try:
        result = SweepRunner(loaded).run()
    except SafetyViolation as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(json.dumps(result, indent=2, sort_keys=True))


@app.command()
def aggregate(
    config: str = typer.Option(..., "--config"),
    run_group_id: str = typer.Option(..., "--run-group-id"),
) -> None:
    loaded = load_experiment_config(config)
    store = create_store(loaded)
    payload = aggregate_run_group(run_group_id, store)
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command(name="export-results")
def export_results(
    config: str = typer.Option(..., "--config"),
    run_group_id: str = typer.Option(..., "--run-group-id"),
) -> None:
    loaded = load_experiment_config(config)
    store = create_store(loaded)
    payload = export_results_bundle(run_group_id, store, loaded.paper_export.output_dir)
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command(name="render-paper-assets")
def render_assets(
    config: str = typer.Option(..., "--config"),
    run_group_id: str = typer.Option(..., "--run-group-id"),
) -> None:
    loaded = load_experiment_config(config)
    store = create_store(loaded)
    payload = render_paper_assets(
        run_group_id,
        store,
        manuscript_path=loaded.paper_export.manuscript_path,
        output_dir=loaded.paper_export.output_dir,
    )
    typer.echo(json.dumps(payload, indent=2, sort_keys=True))


@app.command(name="host-check")
def host_check(output: str | None = typer.Option(None, "--output")) -> None:
    report = build_host_report()
    if output:
        write_host_report(output)
    typer.echo(json.dumps(report, indent=2, sort_keys=True))


def main() -> None:
    app()


if __name__ == "__main__":
    main()
