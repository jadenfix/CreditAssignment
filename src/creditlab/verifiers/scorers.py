from __future__ import annotations

from collections import defaultdict
from typing import Any

from creditlab.core.models import EpisodeRun, TrainingTarget
from creditlab.core.utils import action_label

# Default scorer weights — can be overridden via SCORER_WEIGHTS global
SCORER_WEIGHTS: dict[str, float] = {
    "verifier_weight": 0.25,
    "dead_end_weight": 0.5,
    "pivotal_bonus": 0.75,
    "recovery_bonus": 0.5,
    "enabling_bonus": 0.5,
    "first_turn_bonus": 0.1,
}


def set_scorer_weights(**kwargs: float) -> None:
    """Override default scorer weights. Used by weight ablation experiments."""
    SCORER_WEIGHTS.update(kwargs)


def outcome_only_transition_score(episode: EpisodeRun, transition_index: int) -> float:
    return 1.0 if episode.success else 0.0


def _local_branch_bonus(episode: EpisodeRun, transition_index: int) -> float:
    transition = episode.transitions[transition_index]
    verifier_score = sum(event.score for event in transition.verifier_events if event.signal_name != "success")
    dead_end_penalty = sum(1.0 for event in transition.verifier_events if event.signal_name == "dead_end")
    return SCORER_WEIGHTS["verifier_weight"] * verifier_score - SCORER_WEIGHTS["dead_end_weight"] * dead_end_penalty


def branch_aware_transition_score(episode: EpisodeRun, transition_index: int) -> float:
    return outcome_only_transition_score(episode, transition_index) + _local_branch_bonus(episode, transition_index)


def hindsight_transition_score(episode: EpisodeRun, transition_index: int) -> float:
    transition = episode.transitions[transition_index]
    transition_family = episode.task.metadata.get("family")
    bonus = 0.0
    if transition.action.content == episode.task.metadata.get("pivotal_branch"):
        bonus += SCORER_WEIGHTS["pivotal_bonus"]
    if transition_family == "recovery" and any(event.details.get("recovery") for event in transition.verifier_events):
        bonus += SCORER_WEIGHTS["recovery_bonus"]
    if transition_family == "enabling" and any(event.details.get("enabling") for event in transition.verifier_events):
        bonus += SCORER_WEIGHTS["enabling_bonus"]
    if episode.success and transition_index == 0:
        bonus += SCORER_WEIGHTS["first_turn_bonus"]
    return outcome_only_transition_score(episode, transition_index) + bonus


def combined_transition_score(episode: EpisodeRun, transition_index: int) -> float:
    return branch_aware_transition_score(episode, transition_index) + (
        hindsight_transition_score(episode, transition_index) - outcome_only_transition_score(episode, transition_index)
    )


SCORERS = {
    "outcome_only": outcome_only_transition_score,
    "branch_aware": branch_aware_transition_score,
    "hindsight": hindsight_transition_score,
    "combined": combined_transition_score,
}


def transition_score(episode: EpisodeRun, scorer_name: str, transition_index: int) -> float:
    if scorer_name not in SCORERS:
        raise ValueError(f"unknown scorer: {scorer_name}")
    return SCORERS[scorer_name](episode, transition_index)


def build_training_targets(episodes: list[EpisodeRun], scorer_name: str) -> list[TrainingTarget]:
    targets: list[TrainingTarget] = []
    for episode in episodes:
        node_index = {node.node_id: node for node in episode.nodes}
        for idx, transition in enumerate(episode.transitions):
            parent_state = node_index[transition.parent_node_id].state
            targets.append(
                TrainingTarget(
                    run_id=episode.run_id,
                    episode_id=episode.episode_id,
                    transition_id=transition.transition_id,
                    state_hash=parent_state.state_hash,
                    action=transition.action.content,
                    scorer_name=scorer_name,
                    score=transition_score(episode, scorer_name, idx),
                    metadata={
                        "abstract_state": parent_state.abstract_state,
                        "action_label": action_label(transition.action.content),
                        "success": episode.success,
                    },
                )
            )
    return targets


def aggregate_branch_preferences(episodes: list[EpisodeRun], scorer_name: str) -> dict[str, dict[str, Any]]:
    grouped_scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    grouped_states: dict[str, dict[str, Any]] = {}
    for target in build_training_targets(episodes, scorer_name):
        grouped_scores[target.state_hash][action_label(target.action)].append(target.score)
        grouped_states[target.state_hash] = target.metadata["abstract_state"]
    return {
        state_hash: {
            "abstract_state": grouped_states[state_hash],
            "action_scores": {
                action: sum(values) / len(values)
                for action, values in action_scores.items()
            },
            "support": sum(len(values) for values in action_scores.values()),
        }
        for state_hash, action_scores in grouped_scores.items()
    }
