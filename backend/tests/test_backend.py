"""
test_backend.py  —  Day 1 Backend Unit Tests
Run with:  python tests/test_backend.py
(No Minecraft server or Ollama required.)
"""

import sys, os, json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from main import simple_planner, ObservationState, PlayerInfo, Environment

PASSED = 0
FAILED = 0


def test(name, fn):
    global PASSED, FAILED
    try:
        fn()
        print(f"  [PASS] {name}")
        PASSED += 1
    except (AssertionError, Exception) as e:
        print(f"  [FAIL] {name}\n      -> {e}")
        FAILED += 1


def make_state(**kwargs) -> ObservationState:
    defaults = dict(
        player=PlayerInfo(username="TestBot", health=20, food=20, saturation=5),
        inventory={},
        nearby_entities=[],
        nearby_blocks=[],
        environment=Environment(time_of_day="noon", weather="clear", dimension="overworld"),
        goal=None,
        timestamp="2026-05-15T17:00:00+00:00",
    )
    defaults.update(kwargs)
    return ObservationState(**defaults)


# ── Tests ─────────────────────────────────────────────────────────

print("\nMineAgent Day-1 Backend Tests\n")


def t01_returns_action_response():
    r = simple_planner(make_state())
    assert hasattr(r, "action"), "Response must have .action"
    assert hasattr(r, "params"), "Response must have .params"
test("simple_planner — returns ActionResponse object", t01_returns_action_response)


def t02_idle_when_healthy():
    r = simple_planner(make_state())
    assert r.action == "IDLE", f"Expected IDLE, got {r.action}"
test("simple_planner — IDLE when healthy and fed", t02_idle_when_healthy)


def t03_chat_when_low_hp():
    s = make_state(player=PlayerInfo(health=5, food=20))
    r = simple_planner(s)
    assert r.action == "CHAT", f"Expected CHAT, got {r.action}"
test("simple_planner — CHAT when hp < 8", t03_chat_when_low_hp)


def t04_seek_food_when_hungry():
    s = make_state(player=PlayerInfo(health=20, food=4))
    r = simple_planner(s)
    assert r.action == "SEEK", f"Expected SEEK, got {r.action}"
test("simple_planner — SEEK food when food < 6", t04_seek_food_when_hungry)


def t05_hp_beats_hunger():
    s = make_state(player=PlayerInfo(health=3, food=2))
    r = simple_planner(s)
    assert r.action == "CHAT", f"Health should take priority, got {r.action}"
test("simple_planner — health priority over hunger", t05_hp_beats_hunger)


def t06_reasoning_is_string():
    s = make_state(player=PlayerInfo(health=5, food=20))
    r = simple_planner(s)
    assert isinstance(r.reasoning, str) and len(r.reasoning) > 0, "reasoning must be non-empty str"
test("simple_planner — reasoning is a non-empty string", t06_reasoning_is_string)


def t07_params_is_dict():
    r = simple_planner(make_state())
    assert isinstance(r.params, dict), f"params must be dict, got {type(r.params)}"
test("simple_planner — params is a dict", t07_params_is_dict)


def t08_json_serialisable():
    r = simple_planner(make_state())
    dumped = r.model_dump()
    raw = json.dumps(dumped)
    assert isinstance(raw, str) and len(raw) > 10
test("ActionResponse — serialises to JSON", t08_json_serialisable)


def t09_default_inventory_empty():
    s = make_state()
    assert s.inventory == {}, "Default inventory should be {}"
test("ObservationState — default inventory is empty dict", t09_default_inventory_empty)


def t10_goal_preserved():
    s = make_state(goal="gather wood")
    assert s.goal == "gather wood"
test("ObservationState — goal field passes through", t10_goal_preserved)


def t11_mine_when_no_wood_and_log_nearby():
    s = make_state(
        player=PlayerInfo(health=20, food=20),
        inventory={},
        nearby_blocks=[{"name": "oak_log", "distance": 3.2, "position": {"x": 101, "y": 64, "z": -22}}],
        goal="build shelter",
    )
    r = simple_planner(s)
    assert r.action == "MINE", f"Expected MINE when no wood + log nearby, got {r.action}"
test("simple_planner — MINE when no wood and oak_log nearby (build goal)", t11_mine_when_no_wood_and_log_nearby)


def t12_environment_defaults():
    s = make_state()
    assert s.environment.time_of_day == "noon"
    assert s.environment.weather     == "clear"
test("ObservationState — environment defaults are set", t12_environment_defaults)


# ── Summary ───────────────────────────────────────────────────────

print(f"\nResults: {PASSED} passed, {FAILED} failed\n")
if FAILED > 0:
    sys.exit(1)
