from __future__ import annotations

from datetime import UTC, datetime, timedelta

from creditlab.analysis.reporting import build_replay_result, build_run_report, report_to_metric_rows
from creditlab.core.config import ExperimentConfig
from creditlab.core.interfaces import EnvAdapter, PolicyBackend, TrackerBackend, TrajectoryStore
from creditlab.core.models import EpisodeRun, ObservationEvent, RunContext, StateSnapshot, Transition, TreeNode
from creditlab.core.utils import collect_machine_profile, new_id


class RolloutEngine:
    def __init__(
        self,
        env: EnvAdapter,
        policy: PolicyBackend,
        store: TrajectoryStore,
        tracker: TrackerBackend,
    ) -> None:
        self.env = env
        self.policy = policy
        self.store = store
        self.tracker = tracker

    def collect(
        self,
        config: ExperimentConfig,
        run_id: str | None = None,
        run_context: RunContext | None = None,
        split_override: str | None = None,
    ) -> str:
        run_id = run_id or new_id("run")
        split_name = split_override or config.benchmark.collect_split or config.env.split
        context = run_context or RunContext()
        context.split_name = split_name
        context.benchmark_name = context.benchmark_name or config.benchmark.name
        context.benchmark_revision = context.benchmark_revision or config.benchmark.revision
        context.model_backend = context.model_backend or config.model_backend.name
        context.training_backend = context.training_backend or config.training_backend.name
        context.deployment_profile = context.deployment_profile or config.deployment.profile
        context.container_image_tag = context.container_image_tag or config.deployment.container_image
        context.aws_instance_profile = context.aws_instance_profile or config.deployment.aws_instance_profile
        context.machine_profile = collect_machine_profile()
        self.store.create_run(config.model_dump(mode="json"), run_id, context)
        tasks = self.env.list_tasks(split_name)
        budget = config.experiment.budgets
        run_started_at = datetime.now(tz=UTC)
        deadline_at = None
        if budget.wall_clock_seconds is not None:
            deadline_at = run_started_at + timedelta(seconds=budget.wall_clock_seconds)
        for rollout_index in range(budget.rollout_count):
            if deadline_at is not None and datetime.now(tz=UTC) >= deadline_at:
                break
            task = tasks[rollout_index % len(tasks)]
            episode = self._collect_episode(
                config=config,
                run_id=run_id,
                task=task,
                rollout_index=rollout_index,
                run_context=context,
                deadline_at=deadline_at,
            )
            self.store.write_episode(episode)
        replay_result = build_replay_result(run_id, self.store, "branch_aware", persist=True)
        report = build_run_report(run_id, self.store)
        rows = report_to_metric_rows(run_id, report)
        self.store.write_metrics(rows)
        self.tracker.log_metrics(run_id, rows)
        self.tracker.log_metrics(run_id, replay_result.metrics)
        self.tracker.log_artifact(run_id, "report", report)
        return run_id

    def _collect_episode(
        self,
        config: ExperimentConfig,
        run_id: str,
        task,
        rollout_index: int,
        run_context: RunContext,
        deadline_at: datetime | None = None,
    ) -> EpisodeRun:
        budget = config.experiment.budgets
        episode_id = new_id("episode")
        started_at = datetime.now(tz=UTC)
        state = self.env.reset(task)
        nodes: list[TreeNode] = []
        transitions: list[Transition] = []
        node_lookup: dict[str, str] = {}
        root_node = self._ensure_node(
            node_lookup=node_lookup,
            nodes=nodes,
            state=state,
            episode_id=episode_id,
            task_id=task.task_id,
            turn_index=0,
            parent_node_id=None,
        )
        verifier_calls = 0
        success = False
        outcome = "max_turns"
        for turn_index in range(budget.max_turns):
            if deadline_at is not None and datetime.now(tz=UTC) >= deadline_at:
                outcome = "wall_clock_budget_exhausted"
                break
            action = self.policy.act(transitions, state, budget, task)
            next_state, observation_text, terminal = self.env.step(task, state, action)
            observation = ObservationEvent(text=observation_text)
            verifier_events = self.env.verify_transition(task, state, action, next_state, observation_text)
            remaining_budget = budget.max_verifier_calls - verifier_calls
            if remaining_budget <= 0:
                outcome = "verifier_budget_exhausted"
                break
            verifier_events = verifier_events[:remaining_budget]
            verifier_calls += len(verifier_events)
            parent_node_id = root_node.node_id if turn_index == 0 else transitions[-1].child_node_id
            child_node = self._ensure_node(
                node_lookup=node_lookup,
                nodes=nodes,
                state=next_state,
                episode_id=episode_id,
                task_id=task.task_id,
                turn_index=turn_index + 1,
                parent_node_id=parent_node_id,
            )
            transition = Transition(
                transition_id=new_id("transition"),
                episode_id=episode_id,
                parent_node_id=parent_node_id,
                child_node_id=child_node.node_id,
                turn_index=turn_index,
                action=action,
                observation=observation,
                verifier_events=verifier_events,
                metadata={"rollout_index": rollout_index},
            )
            transitions.append(transition)
            state = next_state
            terminal_success = any(event.signal_name == "success" and event.score > 0 for event in verifier_events)
            if terminal_success:
                success = True
                outcome = "success"
            elif terminal:
                outcome = "terminal_failure"
            if terminal or verifier_calls >= budget.max_verifier_calls:
                if verifier_calls >= budget.max_verifier_calls and not terminal:
                    outcome = "verifier_budget_exhausted"
                break
        completed_at = datetime.now(tz=UTC)
        return EpisodeRun(
            run_id=run_id,
            episode_id=episode_id,
            experiment_name=config.name,
            env_name=self.env.env_name,
            policy_name=self.policy.policy_name,
            task=task,
            budget=budget,
            root_node_id=root_node.node_id,
            nodes=nodes,
            transitions=transitions,
            success=success,
            outcome=outcome,
            started_at=started_at,
            completed_at=completed_at,
            metadata={
                "research_objective": config.research_objective,
                "seed_space": config.experiment.seeds,
                "run_group_id": run_context.run_group_id,
                "condition_name": run_context.condition_name,
                "run_type": run_context.run_type,
                "benchmark_name": run_context.benchmark_name,
                "benchmark_revision": run_context.benchmark_revision,
                "split_name": run_context.split_name,
                "seed": run_context.seed,
                "deployment_profile": run_context.deployment_profile,
            },
        )

    def _ensure_node(
        self,
        node_lookup: dict[str, str],
        nodes: list[TreeNode],
        state: StateSnapshot,
        episode_id: str,
        task_id: str,
        turn_index: int,
        parent_node_id: str | None,
    ) -> TreeNode:
        existing = node_lookup.get(state.state_hash)
        if existing:
            return next(node for node in nodes if node.node_id == existing)
        node = TreeNode(
            node_id=new_id("node"),
            episode_id=episode_id,
            task_id=task_id,
            turn_index=turn_index,
            parent_node_id=parent_node_id,
            state=state,
            metadata={"dedup_key": state.state_hash},
        )
        node_lookup[state.state_hash] = node.node_id
        nodes.append(node)
        return node
