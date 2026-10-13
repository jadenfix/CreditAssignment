from __future__ import annotations

from creditlab.analysis.reporting import aggregate_run_group
from creditlab.analysis.safety import SafetyViolation, SweepExecutionGuard
from creditlab.core.config import ExperimentConfig
from creditlab.core.models import RunContext
from creditlab.core.registry import create_env, create_policy, create_store, create_tracker, create_trainer
from creditlab.core.utils import new_id
from creditlab.rollout.engine import RolloutEngine


class SweepRunner:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.store = create_store(config)
        self.tracker = create_tracker(config.tracking.backend, config.tracking.config, config.store.artifact_root)

    def run(self) -> dict:
        run_group_id = self.config.sweep.run_group_id or new_id("group")
        seeds = self.config.sweep.seeds or self.config.experiment.seeds
        run_index: dict[str, dict[str, dict[str, str]]] = {}
        guard = SweepExecutionGuard(self.config, self.store)
        planned_runs = len(seeds) * (1 + len(self.config.sweep.conditions))
        guard.preflight(planned_runs)
        for seed in seeds:
            if guard.should_halt():
                break
            run_index[str(seed)] = {"collection": {}, "evaluation": {}}
            base_collect_run: str | None = None
            try:
                base_collect_run = self._execute_run_step(
                    guard=guard,
                    label=f"seed={seed}/collection/prompted_base",
                    operation=lambda seed=seed: self._collect_base_prompted_run(run_group_id, seed),
                )
                run_index[str(seed)]["collection"]["prompted_base"] = base_collect_run
            except Exception:
                run_index[str(seed)]["collection"]["prompted_base"] = None
            for condition in self.config.sweep.conditions:
                if guard.should_halt():
                    break
                if condition.name != "prompted" and base_collect_run is None:
                    run_index[str(seed)]["evaluation"][condition.name] = None
                    continue
                try:
                    if condition.name == "prompted":
                        eval_run = self._execute_run_step(
                            guard=guard,
                            label=f"seed={seed}/evaluation/prompted",
                            operation=lambda seed=seed: self._evaluate_prompted(run_group_id, seed),
                        )
                    else:
                        assert base_collect_run is not None
                        eval_run = self._execute_run_step(
                            guard=guard,
                            label=f"seed={seed}/evaluation/{condition.name}",
                            operation=lambda seed=seed, base_collect_run=base_collect_run, condition=condition: self._train_and_evaluate(
                                run_group_id,
                                seed,
                                base_collect_run,
                                condition.name,
                                condition.scorer or condition.name,
                                condition.trainer_name or condition.name,
                            ),
                        )
                    run_index[str(seed)]["evaluation"][condition.name] = eval_run
                except Exception:
                    run_index[str(seed)]["evaluation"][condition.name] = None
                    if guard.should_halt():
                        break
        aggregate = aggregate_run_group(run_group_id, self.store)
        payload = {
            "run_group_id": run_group_id,
            "runs": run_index,
            "aggregate": aggregate,
            "safety": guard.summary(),
        }
        self.tracker.log_artifact(run_group_id, "sweep", payload)
        return payload

    def _execute_run_step(
        self,
        guard: SweepExecutionGuard,
        label: str,
        operation,
    ) -> str:
        try:
            run_id = guard.run_with_retries(label, self.config.experiment.budgets, operation)
        except SafetyViolation as exc:
            guard.halt_reason = str(exc)
            raise
        try:
            guard.record_run(run_id, label)
        except SafetyViolation as exc:
            guard.halt_reason = str(exc)
        return run_id

    def _collect_base_prompted_run(self, run_group_id: str, seed: int) -> str:
        collect_config = self.config.model_copy(deep=True)
        if collect_config.policy.name not in ("vllm_server",):
            collect_config.policy.name = "prompted"
        collect_config.policy.config["seed"] = seed
        collect_config.env.config["seed"] = seed
        env = create_env(collect_config)
        policy = create_policy(collect_config.policy.name, collect_config.policy.config, collect_config.store.artifact_root)
        context = RunContext(
            run_group_id=run_group_id,
            condition_name="prompted_base",
            run_type="collection",
            seed=seed,
            benchmark_name=collect_config.benchmark.name,
            benchmark_revision=collect_config.benchmark.revision,
            scorer_name="outcome_only",
            trainer_name=None,
            model_backend=collect_config.model_backend.name,
            training_backend=collect_config.training_backend.name,
            deployment_profile=collect_config.deployment.profile,
            container_image_tag=collect_config.deployment.container_image,
            aws_instance_profile=collect_config.deployment.aws_instance_profile,
        )
        return RolloutEngine(env=env, policy=policy, store=self.store, tracker=self.tracker).collect(
            collect_config,
            run_context=context,
            split_override=collect_config.benchmark.collect_split,
        )

    def _evaluate_prompted(self, run_group_id: str, seed: int) -> str:
        eval_config = self.config.model_copy(deep=True)
        eval_config.policy.name = "prompted"
        eval_config.policy.config["seed"] = seed
        eval_config.env.config["seed"] = seed
        env = create_env(eval_config)
        policy = create_policy(eval_config.policy.name, eval_config.policy.config, eval_config.store.artifact_root)
        context = RunContext(
            run_group_id=run_group_id,
            condition_name="prompted",
            run_type="evaluation",
            seed=seed,
            benchmark_name=eval_config.benchmark.name,
            benchmark_revision=eval_config.benchmark.revision,
            scorer_name="outcome_only",
            trainer_name=None,
            model_backend=eval_config.model_backend.name,
            training_backend=eval_config.training_backend.name,
            deployment_profile=eval_config.deployment.profile,
            container_image_tag=eval_config.deployment.container_image,
            aws_instance_profile=eval_config.deployment.aws_instance_profile,
        )
        return RolloutEngine(env=env, policy=policy, store=self.store, tracker=self.tracker).collect(
            eval_config,
            run_context=context,
            split_override=eval_config.benchmark.eval_split,
        )

    def _train_and_evaluate(
        self,
        run_group_id: str,
        seed: int,
        source_run: str,
        condition_name: str,
        scorer_name: str,
        trainer_name: str,
    ) -> str:
        train_config = self.config.model_copy(deep=True)
        trainer = create_trainer(trainer_name, self.store, train_config.store.artifact_root)
        train_result = trainer.fit([source_run], scorer_name=scorer_name, budget=train_config.experiment.budgets)
        eval_config = self.config.model_copy(deep=True)
        eval_config.policy.name = "table"
        eval_config.policy.config = {"model_path": train_result["model_path"], "seed": seed}
        eval_config.env.config["seed"] = seed
        env = create_env(eval_config)
        policy = create_policy(eval_config.policy.name, eval_config.policy.config, eval_config.store.artifact_root)
        context = RunContext(
            run_group_id=run_group_id,
            condition_name=condition_name,
            run_type="evaluation",
            seed=seed,
            benchmark_name=eval_config.benchmark.name,
            benchmark_revision=eval_config.benchmark.revision,
            scorer_name=scorer_name,
            trainer_name=trainer_name,
            model_backend=eval_config.model_backend.name,
            training_backend=eval_config.training_backend.name,
            deployment_profile=eval_config.deployment.profile,
            container_image_tag=eval_config.deployment.container_image,
            aws_instance_profile=eval_config.deployment.aws_instance_profile,
            metadata={"source_run_id": source_run, "model_path": train_result["model_path"]},
        )
        eval_run = RolloutEngine(env=env, policy=policy, store=self.store, tracker=self.tracker).collect(
            eval_config,
            run_context=context,
            split_override=eval_config.benchmark.eval_split,
        )
        return eval_run
