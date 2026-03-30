from creditlab.core.models import ActionEvent
from creditlab.envs.webshop import WebShopEnv


def test_webshop_smoke_path_reaches_success():
    env = WebShopEnv()
    task = env.list_tasks("smoke")[0]
    state = env.reset(task)
    state, _, _ = env.step(task, state, ActionEvent(action_id="a1", action_type="tool", content="search_target"))
    state, _, _ = env.step(task, state, ActionEvent(action_id="a2", action_type="tool", content="open_target"))
    state, _, _ = env.step(task, state, ActionEvent(action_id="a3", action_type="tool", content="add_to_cart"))
    final_state, observation, terminal = env.step(
        task,
        state,
        ActionEvent(action_id="a4", action_type="tool", content="checkout"),
    )
    events = env.verify_transition(
        task,
        state,
        ActionEvent(action_id="a4", action_type="tool", content="checkout"),
        final_state,
        observation,
    )
    assert terminal is True
    assert any(event.signal_name == "success" and event.score == 1.0 for event in events)

