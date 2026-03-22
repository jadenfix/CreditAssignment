from __future__ import annotations

from collections import defaultdict
from typing import Any

from creditlab.core.models import EpisodeRun
from creditlab.core.utils import action_label


def outcome_only_transition_score(episode: EpisodeRun) -> float:
    return 1.0 if episode.success else 0.0


def branch_aware_transition_score(episode: EpisodeRun, transition_index: int) -> float:
    base = outcome_only_transition_score(episode)
    transition = episode.transitions[transition_index]
    verifier_score = sum(event.score for event in transition.verifier_events)
    dead_end_penalty = sum(1.0 for event in transition.verifier_events if event.signal_name == "dead_end")
    return base + 0.25 * verifier_score - 0.5 * dead_end_penalty


def hindsight_transition_score(episode: EpisodeRun, transition_index: int) -> float:
    score = branch_aware_transition_score(episode, transition_index)
    transition = episode.transitions[transition_index]
    family = episode.task.metadata.get("family")
    action = transition.action.content
    pivotal = episode.task.metadata.get("pivotal_branch")
    if pivotal and action == pivotal:
        score += 0.75
    if family == "recovery" and any(event.details.get("recovery") for event in transition.verifier_events):
        score += 0.5
    if family == "enabling" and any(event.details.get("enabling") for event in transition.verifier_events):
        score += 0.5
    return score


SCORERS = {
    "outcome_only": outcome_only_transition_score,
    "branch_aware": branch_aware_transition_score,
    "hindsight": hindsight_transition_score,
}


def aggregate_branch_preferences(episodes: list[EpisodeRun], scorer_name: str) -> dict[str, dict[str, Any]]:
    if scorer_name not in SCORERS:
        raise ValueError(f"unknown scorer: {scorer_name}")
    grouped_scores: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    grouped_states: dict[str, dict[str, Any]] = {}
    for episode in episodes:
        node_index = {node.node_id: node for node in episode.nodes}
        for idx, transition in enumerate(episode.transitions):
            state = node_index[transition.parent_node_id].state
            state_hash = state.state_hash
            grouped_states[state_hash] = state.abstract_state
            scorer = SCORERS[scorer_name]
            if scorer_name == "outcome_only":
                score = scorer(episode)
            else:
                score = scorer(episode, idx)
            grouped_scores[state_hash][action_label(transition.action.content)].append(score)
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
