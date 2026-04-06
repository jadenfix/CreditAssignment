#!/usr/bin/env python3
"""Collect LLM trajectories on real ALFWorld and store in CreditLab format.

Usage (vLLM / open model):
    python scripts/collect_alfworld_real.py \
        --backend vllm --model Qwen/Qwen2.5-7B-Instruct \
        --num-episodes 50 --seed 7

Usage (Anthropic API):
    ANTHROPIC_API_KEY=sk-ant-... python scripts/collect_alfworld_real.py \
        --backend anthropic --model claude-haiku-4-5-20251001 \
        --num-episodes 50 --seed 7

Requires: alfworld installed; vLLM running at localhost:8000 (for vllm backend)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime

os.environ.setdefault("ALFWORLD_DATA", os.path.expanduser("~/.cache/alfworld"))

from creditlab.core.models import (
    ActionEvent,
    BudgetConfig,
    EpisodeRun,
    ObservationEvent,
    TreeNode,
    Transition,
)
from creditlab.core.utils import new_id
from creditlab.envs.alfworld_real import RealALFWorldEnv
from creditlab.store.sqlite_store import SqliteTrajectoryStore


# ---------------------------------------------------------------------------
# LLM backends
# ---------------------------------------------------------------------------

def _call_vllm(prompt: str, admissible: list[str], model: str, temperature: float) -> tuple[str, str]:
    """Call local vLLM server. Returns (chosen_action, raw_text)."""
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an agent completing household tasks in a text environment. Choose the single best action from the admissible list to make progress toward your goal. Respond with ONLY the exact action text, nothing else."},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": 64,
        "temperature": temperature,
    }
    req = urllib.request.Request(
        "http://localhost:8000/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read())
        raw = body["choices"][0]["message"]["content"].strip()
    except Exception:
        return admissible[0], "(fallback)"
    return _match_action(raw, admissible), raw


def _call_anthropic(prompt: str, admissible: list[str], model: str, temperature: float) -> tuple[str, str]:
    """Call Anthropic Messages API. Returns (chosen_action, raw_text)."""
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    payload = {
        "model": model,
        "max_tokens": 64,
        "temperature": temperature,
        "messages": [
            {"role": "user", "content": (
                "You are an agent completing a household task in a text environment.\n\n"
                + prompt +
                "\n\nRespond with ONLY the exact action text from the admissible list, nothing else."
            )},
        ],
    }
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(payload).encode(),
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read())
        raw = body["content"][0]["text"].strip()
    except Exception as e:
        print(f"    API error: {e}")
        return admissible[0], "(fallback)"
    return _match_action(raw, admissible), raw


def _match_action(raw: str, admissible: list[str]) -> str:
    """Match LLM output to an admissible action."""
    cleaned = raw.strip().strip("\"'`.")
    # Exact match
    for a in admissible:
        if a.lower() == cleaned.lower():
            return a
    # Substring match
    for a in admissible:
        if a.lower() in cleaned.lower():
            return a
    for a in admissible:
        if cleaned.lower() in a.lower():
            return a
    return admissible[0]


# ---------------------------------------------------------------------------
# Episode collection
# ---------------------------------------------------------------------------

def collect_episode(
    env: RealALFWorldEnv,
    task,
    max_turns: int,
    run_id: str,
    backend: str,
    model: str,
    temperature: float,
) -> EpisodeRun:
    """Collect a single episode."""
    call_fn = _call_anthropic if backend == "anthropic" else _call_vllm

    episode_id = new_id("episode")
    started_at = datetime.now(tz=UTC)
    state = env.reset(task)
    instruction = state.payload.get("instruction", task.instruction)

    nodes, transitions = [], []
    root = TreeNode(node_id=new_id("node"), episode_id=episode_id,
                    task_id=task.task_id, turn_index=0, state=state)
    nodes.append(root)
    current_node = root
    done = False

    for turn in range(max_turns):
        admissible = state.payload.get("available_actions", [])
        if not admissible or done:
            break

        obs = state.payload.get("observation_text", "")
        history = state.payload.get("history", [])
        history_str = " -> ".join(history[-5:]) if history else "(start)"
        prompt = (
            f"Task: {instruction}\n\n"
            f"Recent actions: {history_str}\n\n"
            f"Current observation: {obs}\n\n"
            f"Admissible actions: {', '.join(admissible)}"
        )

        chosen, raw_text = call_fn(prompt, admissible, model, temperature)
        action = ActionEvent(
            action_id=new_id("action"), action_type="tool", content=chosen,
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(chosen.split()),
            metadata={"policy": backend, "model": model, "raw_response": raw_text[:200]},
        )

        new_state, obs_text, done = env.step(task, state, action)
        events = env.verify_transition(task, state, action, new_state, obs_text)

        child = TreeNode(node_id=new_id("node"), episode_id=episode_id,
                         task_id=task.task_id, turn_index=turn + 1,
                         parent_node_id=current_node.node_id, state=new_state)
        nodes.append(child)
        transitions.append(Transition(
            transition_id=new_id("trans"), episode_id=episode_id,
            parent_node_id=current_node.node_id, child_node_id=child.node_id,
            turn_index=turn, action=action,
            observation=ObservationEvent(text=obs_text[:500]),
            verifier_events=events,
        ))

        if done:
            break
        current_node = child
        state = new_state

    success = done and new_state.payload.get("score", 0) > 0 if done else False
    return EpisodeRun(
        run_id=run_id, episode_id=episode_id,
        experiment_name=f"alfworld_real_{backend}",
        env_name="alfworld_real", policy_name=backend,
        task=task,
        budget=BudgetConfig(rollout_count=1, max_turns=max_turns,
                            max_prompt_tokens=4096, max_completion_tokens=128,
                            max_verifier_calls=1000),
        root_node_id=root.node_id, nodes=nodes, transitions=transitions,
        success=success, outcome="success" if success else "failure",
        started_at=started_at, completed_at=datetime.now(tz=UTC),
        metadata={"signal_mode": env.signal_mode, "backend": backend, "model": model,
                  "task_type": task.metadata.get("task_type", "unknown")},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["vllm", "anthropic"], default="vllm")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--max-turns", type=int, default=30)
    parser.add_argument("--signal-mode", choices=["auto", "oracle"], default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--task-limit", type=int, default=80)
    parser.add_argument("--sqlite-path", default="runs/creditlab.sqlite")
    parser.add_argument("--artifact-root", default="artifacts")
    args = parser.parse_args()

    run_id = new_id("run")
    print(f"Run ID: {run_id}")
    print(f"Backend: {args.backend}, Model: {args.model}")
    print(f"Episodes: {args.num_episodes}, Max turns: {args.max_turns}, Seed: {args.seed}")

    env = RealALFWorldEnv(config={"signal_mode": args.signal_mode}, task_limit=args.task_limit)
    tasks = env.list_tasks("train")
    print(f"Tasks available: {len(tasks)}")

    store = SqliteTrajectoryStore(sqlite_path=args.sqlite_path, artifact_root=args.artifact_root)
    rng = random.Random(args.seed)

    successes, total_turns = 0, 0
    for ep_idx in range(args.num_episodes):
        task = tasks[rng.randrange(len(tasks))]
        t0 = time.time()
        episode = collect_episode(env, task, args.max_turns, run_id,
                                  args.backend, args.model, args.temperature)
        elapsed = time.time() - t0
        store.write_episode(episode)

        turns = len(episode.transitions)
        total_turns += turns
        if episode.success:
            successes += 1
        status = "SUCCESS" if episode.success else "fail"
        print(f"  [{ep_idx+1}/{args.num_episodes}] {task.metadata.get('task_type', '?')[:20]} "
              f"turns={turns} {status} ({elapsed:.1f}s)")

    rate = 100 * successes / args.num_episodes if args.num_episodes else 0
    print(f"\nDone. {successes}/{args.num_episodes} ({rate:.1f}%), "
          f"avg turns={total_turns/args.num_episodes:.1f}")
    print(f"Run ID: {run_id}")


if __name__ == "__main__":
    main()
