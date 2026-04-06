"""Real ALFWorld environment adapter using the alfworld TextWorld interface.

Wraps the actual ALFWorld gym environment (text mode) to produce StateSnapshot,
VerifierEvent, etc. compatible with the CreditLab pipeline. Supports both oracle
verifier signals (from ALFWorld's score changes) and automatic signal extraction
from observation text diffs.
"""

from __future__ import annotations

import difflib
import os
import re
from typing import Any

import yaml

from creditlab.core.interfaces import EnvAdapter
from creditlab.core.models import (
    ActionEvent,
    StateDelta,
    StateSnapshot,
    TaskSpec,
    VerifierEvent,
    VerifierStatus,
)
from creditlab.core.utils import stable_hash

# Task type categories in ALFWorld
TASK_TYPES = [
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
]

# Patterns for parsing ALFWorld observations
_ROOM_PATTERN = re.compile(r"You arrive at[:\s]+(.+?)[\.\n]", re.IGNORECASE)
_INVENTORY_PATTERN = re.compile(r"You (?:pick up|take) the (.+?)(?:\s+from|\.|$)", re.IGNORECASE)
_PUT_PATTERN = re.compile(r"You put the (.+?) (?:in|on) ", re.IGNORECASE)
_NOTHING_PATTERN = re.compile(r"Nothing happens|You can't|doesn't appear", re.IGNORECASE)


def _observation_similarity(before: str, after: str) -> float:
    """Compute text similarity between two observations."""
    return difflib.SequenceMatcher(None, before, after).ratio()


def _parse_room(observation: str) -> str | None:
    """Extract location from ALFWorld observation text."""
    match = _ROOM_PATTERN.search(observation)
    return match.group(1).strip() if match else None


def _default_alfworld_config() -> dict:
    """Build a minimal ALFWorld config dict."""
    data_path = os.environ.get("ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld"))
    return {
        "env": {
            "type": "AlfredTWEnv",
            "domain_randomization": False,
            "expert_type": "handcoded",
            "regen": {"type": "AlfredTWEnv", "num_per_task": 1},
            "task_types": {
                1: "pick_and_place_simple-SofaChair-1-LightSwitch-102",
                2: "look_at_obj_in_light-Lettuce-None-DeskLamp-308",
                3: "pick_clean_then_place_in_recep-Spatula-None-Drawer-513",
                4: "pick_heat_then_place_in_recep-Egg-None-CounterTop-19",
                5: "pick_cool_then_place_in_recep-Lettuce-None-DiningTable-423",
                6: "pick_two_obj_and_place-SoapBottle-None-Cart-419",
            },
            "goal_desc_human_anns_prob": 0.0,
            "expert": {"type": "HandCodedTWAgent", "max_steps": 50, "timeout": 10},
            "prefixes": {
                "pick_and_place": "put",
                "pick_clean_then_place": "clean",
                "pick_heat_then_place": "heat",
                "pick_cool_then_place": "cool",
                "look_at_obj_in_light": "examine",
                "pick_two_obj_and_place": "puttwo",
            },
        },
        "dataset": {
            "data_path": f"{data_path}/json_2.1.1/train",
            "eval_paths": [
                f"{data_path}/json_2.1.1/valid_seen",
                f"{data_path}/json_2.1.1/valid_unseen",
            ],
            "num_train_games": -1,
            "num_eval_games": -1,
        },
        "logic": {
            "domain": f"{data_path}/logic/alfred.pddl",
            "grammar": f"{data_path}/logic/alfred.twl2",
        },
        "general": {
            "random_seed": 42,
            "use_cuda": False,
            "training_method": "dagger",
            "save_path": "/tmp/alfworld_results",
        },
        "dagger": {"training": {"max_nb_steps_per_episode": 50}},
    }


class RealALFWorldEnv(EnvAdapter):
    """Adapter for the actual ALFWorld TextWorld environment.

    Requires: pip install alfworld
    """

    env_name = "alfworld_real"

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        task_limit: int | None = None,
        signal_mode: str = "auto",
    ) -> None:
        self._config = config or {}
        self.task_limit = task_limit
        self.signal_mode = self._config.get("signal_mode", signal_mode)

        self._tw_env = None  # AlfredTWEnv wrapper
        self._gym_env = None  # actual gym env from init_env
        self._game_files: list[str] = []
        self._task_list_cache: dict[str, list[TaskSpec]] = {}

        # Episode state tracking
        self._current_obs: str = ""
        self._obs_history: list[str] = []
        self._room: str | None = None
        self._inventory: list[str] = []
        self._turn_count: int = 0
        self._current_admissible: list[str] = []
        self._current_score: float = 0.0

    def _ensure_env(self) -> Any:
        """Lazily import and initialize alfworld."""
        if self._gym_env is not None:
            return self._gym_env

        try:
            from alfworld.agents.environment import get_environment
        except ImportError as e:
            raise ImportError(
                "alfworld is required for RealALFWorldEnv. "
                "Install with: pip install 'creditlab[alfworld]'"
            ) from e

        alfworld_config = self._config.get("alfworld_config")
        if alfworld_config and os.path.isfile(alfworld_config):
            with open(alfworld_config) as f:
                config = yaml.safe_load(f)
        else:
            config = _default_alfworld_config()

        # Override num_train_games if task_limit is set
        if self.task_limit:
            config["dataset"]["num_train_games"] = self.task_limit

        EnvClass = get_environment(config["env"]["type"])
        self._tw_env = EnvClass(config, train_eval="train")
        self._game_files = list(self._tw_env.game_files) if hasattr(self._tw_env, "game_files") else []
        self._gym_env = self._tw_env.init_env(batch_size=1)
        return self._gym_env

    def list_tasks(self, split: str) -> list[TaskSpec]:
        """List available ALFWorld tasks for the given split."""
        if split in self._task_list_cache:
            return self._task_list_cache[split]

        self._ensure_env()

        tasks = []
        n_games = len(self._game_files)
        # 80% train, 20% eval
        split_idx = int(n_games * 0.8)

        for idx, game_file in enumerate(self._game_files):
            task_split = "train" if idx < split_idx else "eval"
            if task_split != split:
                continue

            task_type = "unknown"
            for tt in TASK_TYPES:
                if tt in str(game_file):
                    task_type = tt
                    break

            tasks.append(
                TaskSpec(
                    task_id=f"alfworld_real_{idx}",
                    env_name=self.env_name,
                    instruction="",
                    split=split,
                    metadata={
                        "benchmark_name": "alfworld_real",
                        "benchmark_revision": "v1",
                        "task_type": task_type,
                        "game_file": str(game_file),
                        "game_index": idx,
                    },
                )
            )

        if self.task_limit is not None:
            tasks = tasks[: self.task_limit]

        self._task_list_cache[split] = tasks
        return tasks

    def reset(self, task: TaskSpec) -> StateSnapshot:
        """Reset to a new ALFWorld episode."""
        env = self._ensure_env()

        obs, infos = env.reset()
        if isinstance(obs, (list, tuple)):
            obs_text = str(obs[0])
        else:
            obs_text = str(obs)
        self._current_obs = obs_text
        self._current_admissible = self._extract_admissible(infos)
        self._obs_history = [obs_text]
        self._room = _parse_room(obs_text) or "start"
        self._inventory = []
        self._turn_count = 0
        self._current_score = 0.0

        instruction = self._extract_instruction(obs_text)
        abstract = self._build_abstract_state()

        return StateSnapshot(
            state_hash=stable_hash(abstract),
            abstract_state=abstract,
            payload={
                "observation_text": obs_text,
                "admissible_actions": self._current_admissible,
                "available_actions": self._current_admissible,
                "instruction": instruction,
                "room": self._room,
                "inventory": list(self._inventory),
                "turn": self._turn_count,
                "score": self._current_score,
                "oracle_action": None,
                "history": [],
                "benchmark_name": "alfworld_real",
            },
        )

    def step(
        self, task: TaskSpec, state: StateSnapshot, action: ActionEvent
    ) -> tuple[StateSnapshot, str, bool]:
        """Execute an action in ALFWorld."""
        env = self._ensure_env()

        obs, scores, dones, infos = env.step([action.content])
        obs_text = str(obs[0]) if isinstance(obs, (list, tuple)) else str(obs)
        done = bool(dones[0]) if isinstance(dones, (list, tuple)) else bool(dones)
        score = float(scores[0]) if isinstance(scores, (list, tuple)) else float(scores)

        self._turn_count += 1
        prev_obs = self._current_obs
        self._current_obs = obs_text
        self._current_admissible = self._extract_admissible(infos)
        self._obs_history.append(obs_text)
        prev_score = self._current_score
        self._current_score = score

        # Track room and inventory
        new_room = _parse_room(obs_text)
        if new_room:
            self._room = new_room
        inv_match = _INVENTORY_PATTERN.search(obs_text)
        if inv_match:
            self._inventory.append(inv_match.group(1).strip())
        put_match = _PUT_PATTERN.search(obs_text)
        if put_match:
            item = put_match.group(1).strip()
            if item in self._inventory:
                self._inventory.remove(item)

        abstract = self._build_abstract_state()
        history = list(state.payload.get("history", [])) + [action.content]

        new_state = StateSnapshot(
            state_hash=stable_hash(abstract),
            abstract_state=abstract,
            payload={
                "observation_text": obs_text,
                "prev_observation_text": prev_obs,
                "admissible_actions": self._current_admissible,
                "available_actions": self._current_admissible,
                "instruction": state.payload.get("instruction", ""),
                "room": self._room,
                "inventory": list(self._inventory),
                "turn": self._turn_count,
                "score": score,
                "prev_score": prev_score,
                "oracle_action": None,
                "history": history,
                "benchmark_name": "alfworld_real",
            },
        )

        return new_state, obs_text, done

    def snapshot(self, state: StateSnapshot) -> StateSnapshot:
        return state

    def abstract_state(self, state: StateSnapshot) -> dict[str, Any]:
        return state.abstract_state

    def verify_transition(
        self,
        task: TaskSpec,
        before: StateSnapshot,
        action: ActionEvent,
        after: StateSnapshot,
        observation_text: str,
    ) -> list[VerifierEvent]:
        """Generate verifier events. Supports 'oracle' and 'auto' modes."""
        if self.signal_mode == "oracle":
            return self._verify_oracle(task, before, action, after, observation_text)
        return self._verify_auto(task, before, action, after, observation_text)

    def _verify_oracle(
        self, task: TaskSpec, before: StateSnapshot, action: ActionEvent,
        after: StateSnapshot, observation_text: str,
    ) -> list[VerifierEvent]:
        """Oracle verification using ALFWorld's internal score delta."""
        events = []
        score_delta = after.payload.get("score", 0) - before.payload.get("score", 0)

        if score_delta > 0:
            delta, status = StateDelta.IMPROVED, VerifierStatus.PASS
        elif score_delta < 0:
            delta, status = StateDelta.REGRESSED, VerifierStatus.WARN
        else:
            delta, status = StateDelta.NEUTRAL, VerifierStatus.INFO

        events.append(VerifierEvent(
            verifier_id=f"{task.task_id}:progress",
            status=status, signal_name="progress", score=score_delta,
            state_delta=delta,
            details={"branch_label": "pivotal" if score_delta > 0 else "stall",
                      "dead_end": False, "enabling": False, "recovery": False},
        ))

        if _NOTHING_PATTERN.search(observation_text):
            events.append(VerifierEvent(
                verifier_id=f"{task.task_id}:dead_end",
                status=VerifierStatus.WARN, signal_name="dead_end", score=-1.0,
                state_delta=StateDelta.REGRESSED,
                details={"observation": observation_text[:200]},
            ))

        before_adm = set(before.payload.get("admissible_actions", []))
        after_adm = set(after.payload.get("admissible_actions", []))
        if after_adm != before_adm and score_delta >= 0:
            events.append(VerifierEvent(
                verifier_id=f"{task.task_id}:state_change",
                status=VerifierStatus.PASS, signal_name="state_change", score=1.0,
                state_delta=StateDelta.IMPROVED,
                details={"new_actions": len(after_adm - before_adm)},
            ))

        return events

    def _verify_auto(
        self, task: TaskSpec, before: StateSnapshot, action: ActionEvent,
        after: StateSnapshot, observation_text: str,
    ) -> list[VerifierEvent]:
        """Automatic verification from observation text diffs only."""
        events = []
        before_obs = before.payload.get("observation_text", "")

        # 1. Dead-end detection
        similarity = _observation_similarity(before_obs, observation_text)
        is_error = bool(_NOTHING_PATTERN.search(observation_text))

        if similarity > 0.9 and is_error:
            events.append(VerifierEvent(
                verifier_id=f"{task.task_id}:dead_end",
                status=VerifierStatus.WARN, signal_name="dead_end", score=-1.0,
                state_delta=StateDelta.REGRESSED,
                details={"similarity": similarity, "error_detected": True},
            ))

        # 2. Progress detection
        room_changed = (_parse_room(observation_text) is not None
                        and _parse_room(observation_text) != before.payload.get("room"))
        inventory_changed = bool(_INVENTORY_PATTERN.search(observation_text)
                                 or _PUT_PATTERN.search(observation_text))

        progress_score = 0.0
        if room_changed:
            progress_score += 0.5
        if inventory_changed:
            progress_score += 1.0

        branch_label = "pivotal" if progress_score > 0 else "stall"
        events.append(VerifierEvent(
            verifier_id=f"{task.task_id}:progress",
            status=VerifierStatus.PASS if progress_score > 0 else VerifierStatus.INFO,
            signal_name="progress", score=progress_score,
            state_delta=StateDelta.IMPROVED if progress_score > 0 else StateDelta.NEUTRAL,
            details={"branch_label": branch_label, "dead_end": False,
                      "enabling": room_changed and not inventory_changed,
                      "recovery": False},
        ))

        # 3. Repetition detection
        obs_count = sum(1 for h in self._obs_history
                        if _observation_similarity(h, observation_text) > 0.95)
        if obs_count >= 3:
            events.append(VerifierEvent(
                verifier_id=f"{task.task_id}:repetition",
                status=VerifierStatus.WARN, signal_name="repetition", score=-0.5,
                state_delta=StateDelta.REGRESSED,
                details={"repeat_count": obs_count},
            ))

        return events

    def _build_abstract_state(self) -> dict[str, Any]:
        """Build abstract state for hashing.

        Uses room + inventory status (not turn count) so that the same
        physical situation produces the same hash across episodes,
        enabling cross-episode score aggregation.
        """
        return {
            "phase": self._room or "unknown",
            "mode": "carrying" if self._inventory else "empty",
            "inventory_count": len(self._inventory),
        }

    def _extract_admissible(self, infos: Any) -> list[str]:
        """Extract admissible actions from ALFWorld infos."""
        if isinstance(infos, dict):
            adm = infos.get("admissible_commands", [[]])
            if isinstance(adm, list) and adm:
                if isinstance(adm[0], list):
                    return sorted(adm[0])
                return sorted(adm)
        return []

    def _extract_instruction(self, observation: str) -> str:
        """Extract task instruction from initial observation."""
        for line in observation.split("\n"):
            if "task is to" in line.lower():
                return line.strip()
        lines = [l.strip() for l in observation.split("\n") if l.strip()]
        return lines[0] if lines else "Complete the ALFWorld task."
