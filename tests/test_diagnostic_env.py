from creditlab.core.models import ActionEvent
from creditlab.envs.diagnostic import DiagnosticBranchLabEnv


def test_diagnostic_dead_end_path_can_succeed_after_pivotal_branch():
    env = DiagnosticBranchLabEnv()
    task = env.list_tasks("smoke")[0]
    state = env.reset(task)
    next_state, _, terminal = env.step(
        task,
        state,
        ActionEvent(action_id="a1", action_type="tool", content="inspect_0"),
    )
    assert terminal is False
    assert next_state.abstract_state["mode"] == "prepared"
    final_state, _, terminal = env.step(
        task,
        next_state,
        ActionEvent(action_id="a2", action_type="tool", content="commit_fix_0"),
    )
    events = env.verify_transition(
        task,
        next_state,
        ActionEvent(action_id="a2", action_type="tool", content="commit_fix_0"),
        final_state,
        "The prepared fix resolves the task.",
    )
    assert terminal is True
    assert any(event.signal_name == "success" and event.score == 1.0 for event in events)

