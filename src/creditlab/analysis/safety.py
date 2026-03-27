from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from creditlab.analysis.reporting import build_run_report
from creditlab.core.config import ExperimentConfig, SweepSafetyConfig
from creditlab.core.models import BudgetConfig
from creditlab.deployment.host_check import build_host_report


class SafetyViolation(RuntimeError):
    """Raised when sweep execution would violate a configured safety constraint."""


@dataclass
class SweepTotals:
    launched_runs: int = 0
    completed_runs: int = 0
    failed_runs: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    verifier_calls: int = 0
    successful_wall_clock_seconds: float = 0.0
    failed_wall_clock_seconds: float = 0.0
    successful_runs: list[dict[str, Any]] = field(default_factory=list)

    @property
    def wall_clock_seconds(self) -> float:
        return self.successful_wall_clock_seconds + self.failed_wall_clock_seconds


class SweepExecutionGuard:
    def __init__(self, config: ExperimentConfig, store) -> None:
        self.config = config
        self.store = store
        self.safety: SweepSafetyConfig = config.sweep.safety
        self.totals = SweepTotals()
        self.failures: list[dict[str, Any]] = []
        self.host_report: dict[str, Any] | None = None
        self.planned_runs = 0
        self.halt_reason: str | None = None
        self.started_at = time.monotonic()

    def preflight(self, planned_runs: int) -> dict[str, Any]:
        self.planned_runs = planned_runs
        if planned_runs > self.safety.max_total_runs:
            raise SafetyViolation(
                f"planned sweep launches {planned_runs} runs, which exceeds max_total_runs={self.safety.max_total_runs}"
            )
        if self.config.deployment.platform == "aws_gpu":
            if not self.safety.allow_aws_execution:
                raise SafetyViolation(
                    "deployment.platform=aws_gpu is disabled by sweep.safety.allow_aws_execution=false"
                )
            if self.safety.require_healthy_gpu_for_aws:
                self.host_report = build_host_report()
                self._validate_aws_host(self.host_report)
        elif self.config.deployment.platform == "gcp_gpu":
            if not self.safety.allow_gcp_execution:
                raise SafetyViolation(
                    "deployment.platform=gcp_gpu is disabled by sweep.safety.allow_gcp_execution=false"
                )
            if self.safety.require_healthy_gpu_for_gcp:
                self.host_report = build_host_report()
                self._validate_gcp_host(self.host_report)
        return self.summary()

    def run_with_retries(
        self,
        label: str,
        budget: BudgetConfig,
        operation: Callable[[], str],
    ) -> str:
        self._ensure_can_launch(label, budget)
        self.totals.launched_runs += 1
        attempts_allowed = self.safety.max_retry_attempts + 1
        started_at = time.monotonic()
        last_error: Exception | None = None
        for attempt in range(1, attempts_allowed + 1):
            try:
                return operation()
            except Exception as exc:  # pragma: no cover - failure paths exercised in tests
                last_error = exc
                if attempt >= attempts_allowed:
                    break
                time.sleep(self.safety.retry_backoff_seconds * attempt)
        elapsed = time.monotonic() - started_at
        self._record_failure(label=label, attempts=attempts_allowed, error=last_error, elapsed_seconds=elapsed)
        assert last_error is not None
        raise last_error

    def record_run(self, run_id: str, label: str) -> dict[str, float]:
        summary = self._load_run_summary(run_id)
        self.totals.completed_runs += 1
        self.totals.prompt_tokens += int(summary.get("prompt_tokens", 0))
        self.totals.completion_tokens += int(summary.get("completion_tokens", 0))
        self.totals.verifier_calls += int(summary.get("verifier_calls", 0))
        self.totals.successful_wall_clock_seconds += float(summary.get("wall_clock_time", 0.0))
        self.totals.successful_runs.append({"label": label, "run_id": run_id, "summary": summary})
        self._enforce_current_limits(label)
        return summary

    def should_halt(self) -> bool:
        return self.halt_reason is not None

    def summary(self) -> dict[str, Any]:
        estimated_cost = self._estimate_cost(
            prompt_tokens=self.totals.prompt_tokens,
            completion_tokens=self.totals.completion_tokens,
            wall_clock_seconds=self.totals.wall_clock_seconds,
        )
        elapsed_seconds = time.monotonic() - self.started_at
        return {
            "planned_runs": self.planned_runs,
            "halted": self.should_halt(),
            "halt_reason": self.halt_reason,
            "totals": {
                "launched_runs": self.totals.launched_runs,
                "completed_runs": self.totals.completed_runs,
                "failed_runs": self.totals.failed_runs,
                "prompt_tokens": self.totals.prompt_tokens,
                "completion_tokens": self.totals.completion_tokens,
                "verifier_calls": self.totals.verifier_calls,
                "successful_wall_clock_seconds": round(self.totals.successful_wall_clock_seconds, 4),
                "failed_wall_clock_seconds": round(self.totals.failed_wall_clock_seconds, 4),
                "wall_clock_seconds": round(self.totals.wall_clock_seconds, 4),
                "estimated_cost_usd": round(estimated_cost, 6),
                "guard_elapsed_seconds": round(elapsed_seconds, 4),
            },
            "limits": {
                "max_total_runs": self.safety.max_total_runs,
                "max_failed_runs": self.safety.max_failed_runs,
                "max_total_prompt_tokens": self.safety.max_total_prompt_tokens,
                "max_total_completion_tokens": self.safety.max_total_completion_tokens,
                "max_total_verifier_calls": self.safety.max_total_verifier_calls,
                "max_total_wall_clock_seconds": self.safety.max_total_wall_clock_seconds,
                "max_estimated_cost_usd": self.safety.max_estimated_cost_usd,
            },
            "failures": self.failures,
            "host_preflight": self.host_report,
        }

    def _ensure_can_launch(self, label: str, budget: BudgetConfig) -> None:
        if self.halt_reason is not None:
            raise SafetyViolation(self.halt_reason)
        if self.totals.launched_runs + 1 > self.safety.max_total_runs:
            raise SafetyViolation(
                f"refusing to launch {label}: next run would exceed max_total_runs={self.safety.max_total_runs}"
            )
        projected = self._projected_totals_for_next_run(budget)
        self._check_limits(projected, label, projected_mode=True)

    def _projected_totals_for_next_run(self, budget: BudgetConfig) -> dict[str, float]:
        prompt_tokens = self.totals.prompt_tokens + (budget.rollout_count * budget.max_turns * budget.max_prompt_tokens)
        completion_tokens = self.totals.completion_tokens + (
            budget.rollout_count * budget.max_turns * budget.max_completion_tokens
        )
        verifier_calls = self.totals.verifier_calls + (budget.rollout_count * budget.max_verifier_calls)
        wall_clock_seconds = self.totals.wall_clock_seconds + float(budget.wall_clock_seconds or 0.0)
        return {
            "prompt_tokens": float(prompt_tokens),
            "completion_tokens": float(completion_tokens),
            "verifier_calls": float(verifier_calls),
            "wall_clock_seconds": float(wall_clock_seconds),
            "estimated_cost_usd": self._estimate_cost(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                wall_clock_seconds=wall_clock_seconds,
            ),
        }

    def _load_run_summary(self, run_id: str) -> dict[str, float]:
        metric_rows = self.store.load_metrics(run_id)
        summary = {row.name: row.value for row in metric_rows if row.scope == "report"}
        if not summary:
            summary = build_run_report(run_id, self.store)["summary"]
        return summary

    def _record_failure(
        self,
        label: str,
        attempts: int,
        error: Exception | None,
        elapsed_seconds: float,
    ) -> None:
        self.totals.failed_runs += 1
        self.totals.failed_wall_clock_seconds += max(elapsed_seconds, 0.0)
        message = str(error) if error is not None else "unknown failure"
        self.failures.append(
            {
                "label": label,
                "attempts": attempts,
                "error_type": type(error).__name__ if error is not None else "UnknownError",
                "message": message,
                "elapsed_seconds": round(max(elapsed_seconds, 0.0), 4),
            }
        )
        if self.safety.stop_on_first_failure:
            self.halt_reason = f"stopped after first failure in {label}: {message}"
            return
        if self.totals.failed_runs >= self.safety.max_failed_runs:
            self.halt_reason = (
                f"stopped after reaching max_failed_runs={self.safety.max_failed_runs}; "
                f"latest failure was in {label}: {message}"
            )

    def _enforce_current_limits(self, label: str) -> None:
        current = {
            "prompt_tokens": float(self.totals.prompt_tokens),
            "completion_tokens": float(self.totals.completion_tokens),
            "verifier_calls": float(self.totals.verifier_calls),
            "wall_clock_seconds": float(self.totals.wall_clock_seconds),
            "estimated_cost_usd": self._estimate_cost(
                prompt_tokens=self.totals.prompt_tokens,
                completion_tokens=self.totals.completion_tokens,
                wall_clock_seconds=self.totals.wall_clock_seconds,
            ),
        }
        self._check_limits(current, label, projected_mode=False)

    def _check_limits(self, totals: dict[str, float], label: str, projected_mode: bool) -> None:
        prefix = "refusing to launch" if projected_mode else "stopping sweep after"
        if totals["prompt_tokens"] > self.safety.max_total_prompt_tokens:
            raise SafetyViolation(
                f"{prefix} {label}: prompt tokens {int(totals['prompt_tokens'])} exceed "
                f"max_total_prompt_tokens={self.safety.max_total_prompt_tokens}"
            )
        if totals["completion_tokens"] > self.safety.max_total_completion_tokens:
            raise SafetyViolation(
                f"{prefix} {label}: completion tokens {int(totals['completion_tokens'])} exceed "
                f"max_total_completion_tokens={self.safety.max_total_completion_tokens}"
            )
        if totals["verifier_calls"] > self.safety.max_total_verifier_calls:
            raise SafetyViolation(
                f"{prefix} {label}: verifier calls {int(totals['verifier_calls'])} exceed "
                f"max_total_verifier_calls={self.safety.max_total_verifier_calls}"
            )
        if totals["wall_clock_seconds"] > self.safety.max_total_wall_clock_seconds:
            raise SafetyViolation(
                f"{prefix} {label}: wall clock {totals['wall_clock_seconds']:.2f}s exceeds "
                f"max_total_wall_clock_seconds={self.safety.max_total_wall_clock_seconds}"
            )
        if totals["estimated_cost_usd"] > self.safety.max_estimated_cost_usd:
            raise SafetyViolation(
                f"{prefix} {label}: estimated cost ${totals['estimated_cost_usd']:.4f} exceeds "
                f"max_estimated_cost_usd=${self.safety.max_estimated_cost_usd:.4f}"
            )

    def _estimate_cost(
        self,
        prompt_tokens: float,
        completion_tokens: float,
        wall_clock_seconds: float,
    ) -> float:
        prompt_cost = (prompt_tokens / 1000.0) * self.safety.estimated_prompt_cost_per_1k
        completion_cost = (completion_tokens / 1000.0) * self.safety.estimated_completion_cost_per_1k
        instance_cost = (wall_clock_seconds / 3600.0) * self.safety.estimated_instance_cost_per_hour
        return float(prompt_cost + completion_cost + instance_cost)

    def _validate_gcp_host(self, report: dict[str, Any]) -> None:
        profile = report.get("machine_profile", {})
        cuda_probe = report.get("docker_cuda_probe", {})
        failures: list[str] = []
        if not profile.get("docker_available"):
            failures.append("docker is unavailable")
        if not profile.get("gpu_available"):
            failures.append("nvidia-smi is unavailable")
        if not profile.get("nvidia_ctk_available"):
            failures.append("nvidia-ctk is unavailable")
        if not cuda_probe.get("ok"):
            failures.append("docker cuda probe failed")
        if failures:
            raise SafetyViolation(
                "GCP GPU preflight failed: "
                + "; ".join(failures)
                + ". Run `creditlab host-check` or scripts/gcp/validate_cuda_container.sh on the target host."
            )

    def _validate_aws_host(self, report: dict[str, Any]) -> None:
        profile = report.get("machine_profile", {})
        cuda_probe = report.get("docker_cuda_probe", {})
        failures: list[str] = []
        if not profile.get("docker_available"):
            failures.append("docker is unavailable")
        if not profile.get("gpu_available"):
            failures.append("nvidia-smi is unavailable")
        if not profile.get("nvidia_ctk_available"):
            failures.append("nvidia-ctk is unavailable")
        if not cuda_probe.get("ok"):
            failures.append("docker cuda probe failed")
        if failures:
            raise SafetyViolation(
                "AWS GPU preflight failed: "
                + "; ".join(failures)
                + ". Run `creditlab host-check` or scripts/aws/validate_cuda_container.sh on the target host."
            )
