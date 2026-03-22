from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
from sqlalchemy import Boolean, Column, Float, Integer, MetaData, String, Table, Text, create_engine, select

from creditlab.core.interfaces import TrajectoryStore
from creditlab.core.models import EpisodeRun, MetricRow, ReplayPreferenceShift, ReplayResult, Transition, TreeNode, VerifierEvent


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
            Column("experiment_name", String, nullable=False),
            Column("env_name", String, nullable=False),
            Column("policy_name", String, nullable=False),
            Column("config_json", Text, nullable=False),
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
        self.metadata.create_all(self.engine)

    def create_run(self, config: dict[str, Any], run_id: str) -> None:
        with self.engine.begin() as connection:
            connection.execute(
                self.runs.insert().values(
                    run_id=run_id,
                    experiment_name=config["name"],
                    env_name=config["env"]["name"],
                    policy_name=config["policy"]["name"],
                    config_json=json.dumps(config, sort_keys=True),
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
                verifier_rows = list(
                    connection.execute(
                        select(self.verifier_events).where(self.verifier_events.c.transition_id.in_(transition_ids))
                    ).mappings()
                ) if transition_ids else []
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
                        started_at=episode_row["started_at"],
                        completed_at=episode_row["completed_at"],
                        metadata=json.loads(episode_row["metadata_json"]),
                    )
                )
        return episodes

    def list_run_ids(self) -> list[str]:
        with self.engine.begin() as connection:
            rows = connection.execute(select(self.runs.c.run_id).order_by(self.runs.c.run_id)).all()
        return [row[0] for row in rows]

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
            rows = connection.execute(
                select(self.metrics).where(self.metrics.c.run_id == run_id).order_by(self.metrics.c.id)
            ).mappings()
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
        rows = rows or [{"episode_id": episode.episode_id, "transition_id": ""}]
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, path)
