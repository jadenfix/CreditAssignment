from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import (
    Boolean,
    Column,
    Float,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    create_engine,
    select,
    text,
)

from creditlab.core.interfaces import TrajectoryStore
from creditlab.core.models import (
    EpisodeRun,
    MetricRow,
    ReplayResult,
    RunContext,
    RunRecord,
    TrainingTarget,
    Transition,
    TreeNode,
    VerifierEvent,
)


class SqliteTrajectoryStore(TrajectoryStore):
    def __init__(self, sqlite_path: Path, artifact_root: Path) -> None:
        self.sqlite_path = Path(sqlite_path)
        self.artifact_root = Path(artifact_root)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{self.sqlite_path}")
        self.metadata = MetaData()
        self.runs = Table(
            "runs",
            self.metadata,
            Column("run_id", String, primary_key=True),
            Column("run_group_id", String),
            Column("condition_name", String),
            Column("run_type", String, nullable=False, default="collect"),
            Column("seed", Integer),
            Column("experiment_name", String, nullable=False),
            Column("env_name", String, nullable=False),
            Column("policy_name", String, nullable=False),
            Column("benchmark_name", String),
            Column("benchmark_revision", String),
            Column("split_name", String),
            Column("scorer_name", String),
            Column("trainer_name", String),
            Column("model_backend", String),
            Column("training_backend", String),
            Column("deployment_profile", String),
            Column("container_image_tag", String),
            Column("aws_instance_profile", String),
            Column("machine_profile_json", Text, nullable=False, default="{}"),
            Column("config_json", Text, nullable=False),
            Column("metadata_json", Text, nullable=False, default="{}"),
        )
        self.episodes = Table(
            "episodes",
            self.metadata,
            Column("episode_id", String, primary_key=True),
            Column("run_id", String, nullable=False),
            Column("experiment_name", String, nullable=False),
            Column("env_name", String, nullable=False),
            Column("policy_name", String, nullable=False),
            Column("task_json", Text, nullable=False),
            Column("budget_json", Text, nullable=False),
            Column("root_node_id", String, nullable=False),
            Column("success", Boolean, nullable=False),
            Column("outcome", String, nullable=False),
            Column("started_at", String, nullable=False),
            Column("completed_at", String, nullable=False),
            Column("metadata_json", Text, nullable=False),
            Column("artifact_path", String, nullable=False),
        )
        self.nodes = Table(
            "nodes",
            self.metadata,
            Column("node_id", String, primary_key=True),
            Column("episode_id", String, nullable=False),
            Column("task_id", String, nullable=False),
            Column("turn_index", Integer, nullable=False),
            Column("parent_node_id", String),
            Column("state_hash", String, nullable=False),
            Column("abstract_state_json", Text, nullable=False),
            Column("payload_json", Text, nullable=False),
            Column("metadata_json", Text, nullable=False),
        )
        self.edges = Table(
            "edges",
            self.metadata,
            Column("transition_id", String, primary_key=True),
            Column("episode_id", String, nullable=False),
            Column("parent_node_id", String, nullable=False),
            Column("child_node_id", String, nullable=False),
            Column("turn_index", Integer, nullable=False),
            Column("action_json", Text, nullable=False),
            Column("observation_json", Text, nullable=False),
            Column("metadata_json", Text, nullable=False),
        )
        self.verifier_events = Table(
            "verifier_events",
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("transition_id", String, nullable=False),
            Column("verifier_json", Text, nullable=False),
        )
        self.metrics = Table(
            "metrics",
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("run_id", String, nullable=False),
            Column("scope", String, nullable=False),
            Column("name", String, nullable=False),
            Column("value", Float, nullable=False),
            Column("metadata_json", Text, nullable=False),
        )
        self.replays = Table(
            "replays",
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("run_id", String, nullable=False),
            Column("scorer_name", String, nullable=False),
            Column("artifact_path", String, nullable=False),
            Column("summary_json", Text, nullable=False),
        )
        self.training_targets = Table(
            "training_targets",
            self.metadata,
            Column("id", Integer, primary_key=True, autoincrement=True),
            Column("run_id", String, nullable=False),
            Column("episode_id", String, nullable=False),
            Column("transition_id", String, nullable=False),
            Column("state_hash", String, nullable=False),
            Column("action", String, nullable=False),
            Column("scorer_name", String, nullable=False),
            Column("score", Float, nullable=False),
            Column("metadata_json", Text, nullable=False),
        )
        self.metadata.create_all(self.engine)
        self._migrate_schema()

    def _migrate_schema(self) -> None:
        expected_columns = {
            "runs": {
                "run_group_id": "TEXT",
                "condition_name": "TEXT",
                "run_type": "TEXT DEFAULT 'collect'",
                "seed": "INTEGER",
                "benchmark_name": "TEXT",
                "benchmark_revision": "TEXT",
                "split_name": "TEXT",
                "scorer_name": "TEXT",
                "trainer_name": "TEXT",
                "model_backend": "TEXT",
                "training_backend": "TEXT",
                "deployment_profile": "TEXT",
                "container_image_tag": "TEXT",
                "aws_instance_profile": "TEXT",
                "machine_profile_json": "TEXT DEFAULT '{}'",
                "metadata_json": "TEXT DEFAULT '{}'",
            }
        }
        with self.engine.begin() as connection:
            for table_name, columns in expected_columns.items():
                existing = {
                    row[1]
                    for row in connection.execute(text(f"PRAGMA table_info({table_name})")).fetchall()
                }
                for column_name, column_type in columns.items():
                    if column_name not in existing:
                        connection.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"))

    def create_run(self, config: dict[str, Any], run_id: str, context: RunContext) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                self.runs.insert().values(
                    run_id=run_id,
                    run_group_id=context.run_group_id,
                    condition_name=context.condition_name,
                    run_type=context.run_type,
                    seed=context.seed,
                    experiment_name=config["name"],
                    env_name=config["env"]["name"],
                    policy_name=config["policy"]["name"],
                    benchmark_name=context.benchmark_name,
                    benchmark_revision=context.benchmark_revision,
                    split_name=context.split_name,
                    scorer_name=context.scorer_name,
                    trainer_name=context.trainer_name,
                    model_backend=context.model_backend,
                    training_backend=context.training_backend,
                    deployment_profile=context.deployment_profile,
                    container_image_tag=context.container_image_tag,
                    aws_instance_profile=context.aws_instance_profile,
                    machine_profile_json=context.machine_profile.model_dump_json(),
                    config_json=json.dumps(config, sort_keys=True),
                    metadata_json=json.dumps(context.metadata, sort_keys=True),
                )
            )

    def write_episode(self, episode: EpisodeRun) -> None:
        episode_dir = self.artifact_root / "runs" / episode.run_id / "episodes"
        episode_dir.mkdir(parents=True, exist_ok=True)
        artifact_path = episode_dir / f"{episode.episode_id}.parquet"
        self._write_episode_artifact(artifact_path, episode)
        with self.engine.begin() as connection:
            connection.execute(
                self.episodes.insert().values(
                    episode_id=episode.episode_id,
                    run_id=episode.run_id,
                    experiment_name=episode.experiment_name,
                    env_name=episode.env_name,
                    policy_name=episode.policy_name,
                    task_json=episode.task.model_dump_json(),
                    budget_json=episode.budget.model_dump_json(),
                    root_node_id=episode.root_node_id,
                    success=episode.success,
                    outcome=episode.outcome,
                    started_at=episode.started_at.isoformat(),
                    completed_at=episode.completed_at.isoformat(),
                    metadata_json=json.dumps(episode.metadata, sort_keys=True),
                    artifact_path=str(artifact_path),
                )
            )
            connection.execute(
                self.nodes.insert(),
                [
                    {
                        "node_id": node.node_id,
                        "episode_id": node.episode_id,
                        "task_id": node.task_id,
                        "turn_index": node.turn_index,
                        "parent_node_id": node.parent_node_id,
                        "state_hash": node.state.state_hash,
                        "abstract_state_json": json.dumps(node.state.abstract_state, sort_keys=True),
                        "payload_json": json.dumps(node.state.payload, sort_keys=True),
                        "metadata_json": json.dumps(node.metadata, sort_keys=True),
                    }
                    for node in episode.nodes
                ],
            )
            edge_rows = [
                {
                    "transition_id": transition.transition_id,
                    "episode_id": transition.episode_id,
                    "parent_node_id": transition.parent_node_id,
                    "child_node_id": transition.child_node_id,
                    "turn_index": transition.turn_index,
                    "action_json": transition.action.model_dump_json(),
                    "observation_json": transition.observation.model_dump_json(),
                    "metadata_json": json.dumps(transition.metadata, sort_keys=True),
                }
                for transition in episode.transitions
            ]
            if edge_rows:
                connection.execute(self.edges.insert(), edge_rows)
            verifier_rows = []
            for transition in episode.transitions:
                for event in transition.verifier_events:
                    verifier_rows.append(
                        {
                            "transition_id": transition.transition_id,
                            "verifier_json": event.model_dump_json(),
                        }
                    )
            if verifier_rows:
                connection.execute(self.verifier_events.insert(), verifier_rows)

    def load_episodes(self, run_id: str) -> list[EpisodeRun]:
        episodes: list[EpisodeRun] = []
        with self.engine.begin() as connection:
            episode_rows = list(
                connection.execute(
                    select(self.episodes).where(self.episodes.c.run_id == run_id).order_by(self.episodes.c.episode_id)
                ).mappings()
            )
            for episode_row in episode_rows:
                node_rows = list(
                    connection.execute(
                        select(self.nodes).where(self.nodes.c.episode_id == episode_row["episode_id"]).order_by(self.nodes.c.turn_index)
                    ).mappings()
                )
                edge_rows = list(
                    connection.execute(
                        select(self.edges).where(self.edges.c.episode_id == episode_row["episode_id"]).order_by(self.edges.c.turn_index)
                    ).mappings()
                )
                transition_ids = [row["transition_id"] for row in edge_rows]
                verifier_rows = (
                    list(
                        connection.execute(
                            select(self.verifier_events).where(self.verifier_events.c.transition_id.in_(transition_ids))
                        ).mappings()
                    )
                    if transition_ids
                    else []
                )
                verifier_by_transition: dict[str, list[VerifierEvent]] = {}
                for row in verifier_rows:
                    verifier_by_transition.setdefault(row["transition_id"], []).append(
                        VerifierEvent.model_validate_json(row["verifier_json"])
                    )
                nodes = [
                    TreeNode(
                        node_id=row["node_id"],
                        episode_id=row["episode_id"],
                        task_id=row["task_id"],
                        turn_index=row["turn_index"],
                        parent_node_id=row["parent_node_id"],
                        state={
                            "state_hash": row["state_hash"],
                            "abstract_state": json.loads(row["abstract_state_json"]),
                            "payload": json.loads(row["payload_json"]),
                        },
                        metadata=json.loads(row["metadata_json"]),
                    )
                    for row in node_rows
                ]
                transitions = [
                    Transition(
                        transition_id=row["transition_id"],
                        episode_id=row["episode_id"],
                        parent_node_id=row["parent_node_id"],
                        child_node_id=row["child_node_id"],
                        turn_index=row["turn_index"],
                        action=json.loads(row["action_json"]),
                        observation=json.loads(row["observation_json"]),
                        verifier_events=verifier_by_transition.get(row["transition_id"], []),
                        metadata=json.loads(row["metadata_json"]),
                    )
                    for row in edge_rows
                ]
                episodes.append(
                    EpisodeRun(
                        run_id=episode_row["run_id"],
                        episode_id=episode_row["episode_id"],
                        experiment_name=episode_row["experiment_name"],
                        env_name=episode_row["env_name"],
                        policy_name=episode_row["policy_name"],
                        task=json.loads(episode_row["task_json"]),
                        budget=json.loads(episode_row["budget_json"]),
                        root_node_id=episode_row["root_node_id"],
                        nodes=nodes,
                        transitions=transitions,
                        success=episode_row["success"],
                        outcome=episode_row["outcome"],
                        started_at=datetime.fromisoformat(episode_row["started_at"]),
                        completed_at=datetime.fromisoformat(episode_row["completed_at"]),
                        metadata=json.loads(episode_row["metadata_json"]),
                    )
                )
        return episodes

    def list_run_ids(self) -> list[str]:
        with self.engine.begin() as connection:
            rows = connection.execute(select(self.runs.c.run_id).order_by(self.runs.c.run_id)).all()
        return [row[0] for row in rows]

    def list_runs(self, run_group_id: str | None = None, run_type: str | None = None) -> list[RunRecord]:
        query = select(self.runs).order_by(self.runs.c.run_id)
        if run_group_id is not None:
            query = query.where(self.runs.c.run_group_id == run_group_id)
        if run_type is not None:
            query = query.where(self.runs.c.run_type == run_type)
        with self.engine.begin() as connection:
            rows = list(connection.execute(query).mappings())
        return [self._run_record_from_row(row) for row in rows]

    def load_run(self, run_id: str) -> RunRecord | None:
        with self.engine.begin() as connection:
            row = connection.execute(select(self.runs).where(self.runs.c.run_id == run_id)).mappings().first()
        return self._run_record_from_row(row) if row else None

    def write_metrics(self, rows: Sequence[MetricRow]) -> None:
        if not rows:
            return
        with self.engine.begin() as connection:
            connection.execute(
                self.metrics.insert(),
                [
                    {
                        "run_id": row.run_id,
                        "scope": row.scope,
                        "name": row.name,
                        "value": row.value,
                        "metadata_json": json.dumps(row.metadata, sort_keys=True),
                    }
                    for row in rows
                ],
            )

    def load_metrics(self, run_id: str) -> list[MetricRow]:
        with self.engine.begin() as connection:
            rows = list(
                connection.execute(
                    select(self.metrics).where(self.metrics.c.run_id == run_id).order_by(self.metrics.c.id)
                ).mappings()
            )
        return [
            MetricRow(
                run_id=row["run_id"],
                scope=row["scope"],
                name=row["name"],
                value=row["value"],
                metadata=json.loads(row["metadata_json"]),
            )
            for row in rows
        ]

    def write_replay(self, result: ReplayResult) -> Path:
        replay_dir = self.artifact_root / "runs" / result.run_id / "replays"
        replay_dir.mkdir(parents=True, exist_ok=True)
        path = replay_dir / f"{result.scorer_name}.parquet"
        table = pa.Table.from_pylist(
            [
                {
                    "state_hash": shift.state_hash,
                    "abstract_state_json": json.dumps(shift.abstract_state, sort_keys=True),
                    "outcome_only_action": shift.outcome_only_action,
                    "replay_action": shift.replay_action,
                    "changed": shift.changed,
                    "margin": shift.margin,
                    "support": shift.support,
                }
                for shift in result.preference_shifts
            ]
        )
        pq.write_table(table, path)
        with self.engine.begin() as connection:
            connection.execute(
                self.replays.insert().values(
                    run_id=result.run_id,
                    scorer_name=result.scorer_name,
                    artifact_path=str(path),
                    summary_json=json.dumps(result.summary, sort_keys=True),
                )
            )
        return path

    def write_training_targets(self, targets: Sequence[TrainingTarget]) -> Path | None:
        if not targets:
            return None
        run_id = targets[0].run_id
        scorer_name = targets[0].scorer_name
        target_dir = self.artifact_root / "runs" / run_id / "targets"
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"{scorer_name}.parquet"
        pq.write_table(
            pa.Table.from_pylist([target.model_dump(mode="json") for target in targets]),
            path,
        )
        with self.engine.begin() as connection:
            connection.execute(
                self.training_targets.insert(),
                [
                    {
                        "run_id": target.run_id,
                        "episode_id": target.episode_id,
                        "transition_id": target.transition_id,
                        "state_hash": target.state_hash,
                        "action": target.action,
                        "scorer_name": target.scorer_name,
                        "score": target.score,
                        "metadata_json": json.dumps(target.metadata, sort_keys=True),
                    }
                    for target in targets
                ],
            )
        return path

    def load_training_targets(self, run_id: str, scorer_name: str | None = None) -> list[TrainingTarget]:
        query = select(self.training_targets).where(self.training_targets.c.run_id == run_id)
        if scorer_name is not None:
            query = query.where(self.training_targets.c.scorer_name == scorer_name)
        with self.engine.begin() as connection:
            rows = list(connection.execute(query).mappings())
        return [
            TrainingTarget(
                run_id=row["run_id"],
                episode_id=row["episode_id"],
                transition_id=row["transition_id"],
                state_hash=row["state_hash"],
                action=row["action"],
                scorer_name=row["scorer_name"],
                score=row["score"],
                metadata=json.loads(row["metadata_json"]),
            )
            for row in rows
        ]

    def _run_record_from_row(self, row: Any) -> RunRecord:
        return RunRecord(
            run_id=row["run_id"],
            run_group_id=row.get("run_group_id"),
            condition_name=row.get("condition_name"),
            run_type=row.get("run_type") or "collect",
            seed=row.get("seed"),
            experiment_name=row["experiment_name"],
            env_name=row["env_name"],
            policy_name=row["policy_name"],
            benchmark_name=row.get("benchmark_name"),
            benchmark_revision=row.get("benchmark_revision"),
            split_name=row.get("split_name"),
            scorer_name=row.get("scorer_name"),
            trainer_name=row.get("trainer_name"),
            model_backend=row.get("model_backend"),
            training_backend=row.get("training_backend"),
            deployment_profile=row.get("deployment_profile"),
            container_image_tag=row.get("container_image_tag"),
            aws_instance_profile=row.get("aws_instance_profile"),
            machine_profile=json.loads(row.get("machine_profile_json") or "{}"),
            config=json.loads(row["config_json"]),
            metadata=json.loads(row.get("metadata_json") or "{}"),
        )

    def _write_episode_artifact(self, path: Path, episode: EpisodeRun) -> None:
        node_index = {node.node_id: node for node in episode.nodes}
        rows = []
        for transition in episode.transitions:
            rows.append(
                {
                    "episode_id": episode.episode_id,
                    "transition_id": transition.transition_id,
                    "turn_index": transition.turn_index,
                    "parent_node_id": transition.parent_node_id,
                    "child_node_id": transition.child_node_id,
                    "parent_state_hash": node_index[transition.parent_node_id].state.state_hash,
                    "child_state_hash": node_index[transition.child_node_id].state.state_hash,
                    "parent_abstract_state_json": json.dumps(
                        node_index[transition.parent_node_id].state.abstract_state, sort_keys=True
                    ),
                    "child_abstract_state_json": json.dumps(
                        node_index[transition.child_node_id].state.abstract_state, sort_keys=True
                    ),
                    "parent_payload_json": json.dumps(node_index[transition.parent_node_id].state.payload, sort_keys=True),
                    "child_payload_json": json.dumps(node_index[transition.child_node_id].state.payload, sort_keys=True),
                    "action_json": transition.action.model_dump_json(),
                    "observation_json": transition.observation.model_dump_json(),
                    "verifier_events_json": json.dumps(
                        [event.model_dump(mode="json") for event in transition.verifier_events],
                        sort_keys=True,
                    ),
                }
            )
        if not rows:
            rows = [{"episode_id": episode.episode_id, "transition_id": ""}]
        pq.write_table(pa.Table.from_pylist(rows), path)
