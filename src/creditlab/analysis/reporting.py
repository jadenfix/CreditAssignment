from __future__ import annotations

import json
from collections import Counter, defaultdict
from typing import Any

import polars as pl

from creditlab.core.models import MetricRow, ReplayPreferenceShift, ReplayResult
from creditlab.core.utils import action_label
from creditlab.verifiers.scorers import aggregate_branch_preferences


def build_run_report(run_id: str, store) -> dict[str, Any]:
    episodes = store.load_episodes(run_id)
    transition_rows: list[dict[str, Any]] = []
    state_payloads: dict[str, set[str]] = defaultdict(set)
    for episode in episodes:
        node_index = {node.node_id: node for node in episode.nodes}
        for transition in episode.transitions:
            parent_state = node_index[transition.parent_node_id].state
            state_payloads[parent_state.state_hash].add(json.dumps(parent_state.payload, sort_keys=True))
            transition_rows.append(
                {
                    "episode_id": episode.episode_id,
                    "state_hash": parent_state.state_hash,
                    "action": action_label(transition.action.content),
                    "prompt_tokens": transition.action.prompt_tokens,
                    "completion_tokens": transition.action.completion_tokens,
                    "verifier_calls": len(transition.verifier_events),
                    "turn_index": transition.turn_index,
                    "dead_end": any(event.signal_name == "dead_end" for event in transition.verifier_events),
                    "success": episode.success,
                    "duration_seconds": max(
                        (episode.completed_at - episode.started_at).total_seconds(),
                        0.0,
                    ),
                }
            )
    frame = pl.DataFrame(transition_rows) if transition_rows else pl.DataFrame(schema={"episode_id": pl.String})
    rollout_count = len(episodes)
    success_count = sum(1 for episode in episodes if episode.success)
    total_prompt_tokens = int(frame["prompt_tokens"].sum()) if "prompt_tokens" in frame.columns else 0
    total_completion_tokens = int(frame["completion_tokens"].sum()) if "completion_tokens" in frame.columns else 0
    total_verifier_calls = int(frame["verifier_calls"].sum()) if "verifier_calls" in frame.columns else 0
    avg_turns = (len(transition_rows) / rollout_count) if rollout_count else 0.0
    wall_clock = sum(
        max((episode.completed_at - episode.started_at).total_seconds(), 0.0)
        for episode in episodes
    )
    summary = {
        "rollout_count": rollout_count,
        "avg_turns": avg_turns,
        "prompt_tokens": total_prompt_tokens,
        "completion_tokens": total_completion_tokens,
        "verifier_calls": total_verifier_calls,
        "wall_clock_time": wall_clock,
        "success": success_count / rollout_count if rollout_count else 0.0,
        "tokens_per_success": (
            (total_prompt_tokens + total_completion_tokens) / success_count
            if success_count
            else float(total_prompt_tokens + total_completion_tokens)
        ),
    }
    replay_branch = build_replay_result(run_id, store, "branch_aware")
    repeated_dead_ends = Counter(row["action"] for row in transition_rows if row["dead_end"]).most_common(5)
    disagreement = sorted(
        replay_branch.preference_shifts,
        key=lambda shift: abs(shift.margin),
        reverse=True,
    )[:5]
    state_collapses = [
        {"state_hash": state_hash, "distinct_payloads": len(payloads)}
        for state_hash, payloads in state_payloads.items()
        if len(payloads) > 1
    ]
    return {
        "run_id": run_id,
        "summary": summary,
        "failure_analysis": {
            "top_repeated_dead_end_branches": [
                {"action": action, "count": count}
                for action, count in repeated_dead_ends
            ],
            "highest_disagreement_branch_points": [
                shift.model_dump(mode="json")
                for shift in disagreement
            ],
            "rescoring_changed_preferred_branch_examples": replay_branch.changed_examples[:5],
            "state_abstraction_collapses": state_collapses[:5],
        },
    }


def build_replay_result(run_id: str, store, scorer_name: str) -> ReplayResult:
    episodes = store.load_episodes(run_id)
    outcome_preferences = aggregate_branch_preferences(episodes, "outcome_only")
    replay_preferences = aggregate_branch_preferences(episodes, scorer_name)
    shifts: list[ReplayPreferenceShift] = []
    changed_examples: list[dict[str, Any]] = []
    changed_count = 0
    total_states = 0
    for state_hash, outcome_entry in outcome_preferences.items():
        replay_entry = replay_preferences.get(state_hash)
        if not replay_entry or len(outcome_entry["action_scores"]) < 2:
            continue
        total_states += 1
        outcome_action, outcome_score = max(
            outcome_entry["action_scores"].items(),
            key=lambda item: item[1],
        )
        replay_action, replay_score = max(
            replay_entry["action_scores"].items(),
            key=lambda item: item[1],
        )
        changed = outcome_action != replay_action
        if changed:
            changed_count += 1
            changed_examples.append(
                {
                    "state_hash": state_hash,
                    "abstract_state": replay_entry["abstract_state"],
                    "outcome_only_action": outcome_action,
                    "replay_action": replay_action,
                    "outcome_only_scores": outcome_entry["action_scores"],
                    "replay_scores": replay_entry["action_scores"],
                }
            )
        shifts.append(
            ReplayPreferenceShift(
                state_hash=state_hash,
                abstract_state=replay_entry["abstract_state"],
                outcome_only_action=outcome_action,
                replay_action=replay_action,
                changed=changed,
                margin=float(replay_score - outcome_score),
                support=int(replay_entry["support"]),
            )
        )
    summary = {
        "states_compared": total_states,
        "changed_preferences": changed_count,
        "change_rate": changed_count / total_states if total_states else 0.0,
    }
    metrics = [
        MetricRow(
            run_id=run_id,
            scope=f"replay/{scorer_name}",
            name="change_rate",
            value=summary["change_rate"],
            metadata={"states_compared": total_states},
        )
    ]
    result = ReplayResult(
        run_id=run_id,
        scorer_name=scorer_name,
        preference_shifts=shifts,
        metrics=metrics,
        changed_examples=changed_examples,
        summary=summary,
    )
    store.write_replay(result)
    store.write_metrics(metrics)
    return result


def report_to_metric_rows(run_id: str, report: dict[str, Any]) -> list[MetricRow]:
    summary = report["summary"]
    return [
        MetricRow(run_id=run_id, scope="report", name=name, value=float(value))
        for name, value in summary.items()
    ]
