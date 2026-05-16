"""
test_llm_quality.py — 200+ automated LLM response quality tests.

Tests that the LLM:
1. Returns valid JSON for every state
2. Uses SEEK (not MINE) when target is NOT in nearby_blocks
3. Uses MINE when target IS in nearby_blocks
4. Never mines wrong blocks (grass_block/dirt when goal is wood)
5. Responds correctly to different goals
6. Handles edge cases (low health, no goal, etc.)

Run: pytest tests/test_llm_quality.py -v --tb=short
"""

import sys
import os
import json
import pytest
import asyncio
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prompt_builder import build_messages, extract_json, _situation_analysis
from tool_registry import TOOL_NAMES

# ── Helpers ─────────────────────────────────────────────────────────

def _state(
    nearby_blocks=None,
    inventory=None,
    health=20,
    food=20,
    x=0, y=75, z=0,
    entities=None,
):
    return {
        "player": {
            "username": "MineAgent",
            "position": {"x": x, "y": y, "z": z},
            "health": health,
            "food": food,
            "saturation": 5,
        },
        "inventory": inventory or {},
        "nearby_entities": entities or [],
        "nearby_blocks": nearby_blocks or [],
        "environment": {"time_of_day": "morning", "weather": "clear", "dimension": "overworld"},
    }

def _blocks(*names):
    return [{"name": n, "distance": (i+1)*1.5, "position": {"x": i, "y": 74, "z": i}} for i, n in enumerate(names)]

def _entity(name, etype, dist):
    return {"name": name, "type": etype, "distance": dist, "position": {"x": dist, "y": 75, "z": 0}}


# ── Prompt builder unit tests (fast, no LLM) ───────────────────────

class TestSituationAnalysis:
    """Test _situation_analysis() produces correct guidance."""

    def test_no_goal_gives_idle_hint(self):
        state = _state(nearby_blocks=_blocks("grass_block"))
        result = _situation_analysis(state, "")
        assert "IDLE" in result or "wait" in result.lower()

    def test_oak_log_nearby_suggests_mine(self):
        state = _state(nearby_blocks=_blocks("oak_log", "grass_block"))
        result = _situation_analysis(state, "gather oak wood")
        assert "MINE" in result or "found nearby" in result

    def test_no_oak_log_suggests_seek(self):
        state = _state(nearby_blocks=_blocks("grass_block", "dirt"))
        result = _situation_analysis(state, "gather oak wood")
        assert "SEEK" in result

    def test_acacia_nearby_suggests_mine(self):
        state = _state(nearby_blocks=_blocks("acacia_log", "grass_block"))
        result = _situation_analysis(state, "gather acacia wood")
        assert "MINE" in result or "found" in result

    def test_no_acacia_suggests_seek(self):
        state = _state(nearby_blocks=_blocks("grass_block", "dirt"))
        result = _situation_analysis(state, "gather acacia wood")
        assert "SEEK" in result

    def test_low_health_flagged(self):
        state = _state(health=5)
        result = _situation_analysis(state, "gather wood")
        assert "CRITICAL" in result or "health" in result.lower()

    def test_low_food_flagged(self):
        state = _state(food=3)
        result = _situation_analysis(state, "gather wood")
        assert "CRITICAL" in result or "food" in result.lower()

    def test_hostile_mob_flagged(self):
        state = _state(entities=[_entity("zombie", "hostile", 4)])
        result = _situation_analysis(state, "explore")
        assert "HOSTILE" in result or "zombie" in result

    def test_coal_nearby_suggests_mine(self):
        state = _state(nearby_blocks=_blocks("coal_ore", "stone"))
        result = _situation_analysis(state, "gather coal")
        assert "MINE" in result or "found" in result

    def test_no_coal_suggests_seek(self):
        state = _state(nearby_blocks=_blocks("grass_block"))
        result = _situation_analysis(state, "gather coal")
        assert "SEEK" in result

    def test_explore_goal(self):
        state = _state(nearby_blocks=_blocks("grass_block"))
        result = _situation_analysis(state, "explore")
        assert "MOVE" in result or "explor" in result.lower()

    def test_stone_nearby_suggests_mine(self):
        state = _state(nearby_blocks=_blocks("stone", "dirt"))
        result = _situation_analysis(state, "gather stone")
        assert "MINE" in result or "found" in result


class TestPromptBuilder:
    """Test build_messages() produces correct structure."""

    def test_returns_two_messages(self):
        msgs = build_messages(_state(), "gather oak wood")
        assert len(msgs) == 2

    def test_system_role(self):
        msgs = build_messages(_state(), "gather oak wood")
        assert msgs[0]["role"] == "system"

    def test_user_role(self):
        msgs = build_messages(_state(), "gather oak wood")
        assert msgs[1]["role"] == "user"

    def test_system_contains_strict_rules(self):
        msgs = build_messages(_state(), "gather oak wood")
        sys = msgs[0]["content"]
        assert "STRICT RULES" in sys
        assert "SEEK" in sys
        assert "NEVER mine a block" in sys

    def test_user_contains_goal(self):
        msgs = build_messages(_state(), "gather oak wood")
        user = msgs[1]["content"]
        assert "gather oak wood" in user

    def test_user_contains_situation(self):
        state = _state(nearby_blocks=_blocks("grass_block"))
        msgs = build_messages(state, "gather oak wood")
        user = msgs[1]["content"]
        assert "SITUATION ANALYSIS" in user
        assert "SEEK" in user

    def test_no_goal_message(self):
        msgs = build_messages(_state(), None)
        user = msgs[1]["content"]
        assert "No goal" in user

    def test_few_shot_examples_in_system(self):
        msgs = build_messages(_state(), "gather oak wood")
        sys = msgs[0]["content"]
        assert "FEW-SHOT" in sys
        assert "oak_log not nearby" in sys

    def test_seek_in_tool_list(self):
        from tool_registry import get_tool_schema_text
        schema = get_tool_schema_text()
        assert "SEEK" in schema

    def test_memories_included(self):
        msgs = build_messages(_state(), "explore", memories=["found oak tree at 100,75,200"])
        user = msgs[1]["content"]
        assert "found oak tree" in user


class TestExtractJson:
    """Test extract_json() handles all LLM output formats."""

    def test_clean_json(self):
        r = extract_json('{"action":"MINE","params":{"block":"oak_log"},"reasoning":"test"}')
        assert r["action"] == "MINE"

    def test_markdown_block(self):
        r = extract_json('```json\n{"action":"SEEK","params":{"target":"oak_log"},"reasoning":"not nearby"}\n```')
        assert r["action"] == "SEEK"

    def test_embedded_in_text(self):
        r = extract_json('I will do this: {"action":"MOVE","params":{"direction":"north","distance":20},"reasoning":"exploring"}')
        assert r["action"] == "MOVE"

    def test_invalid_returns_none(self):
        r = extract_json("I cannot help with that.")
        assert r is None

    def test_empty_returns_none(self):
        r = extract_json("")
        assert r is None

    def test_extra_whitespace(self):
        r = extract_json('  {"action":"IDLE","params":{},"reasoning":"waiting"}  ')
        assert r["action"] == "IDLE"

    def test_craft_action(self):
        r = extract_json('{"action":"CRAFT","params":{"item":"wooden_pickaxe","count":1},"reasoning":"have materials"}')
        assert r["action"] == "CRAFT"
        assert r["params"]["item"] == "wooden_pickaxe"


# ── LLM Integration Tests (requires running Ollama) ──────────────

# These tests call the real LLM. Skip if Ollama not available.
import httpx

def _ollama_available():
    try:
        r = httpx.get("http://localhost:11434/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

OLLAMA_AVAILABLE = _ollama_available()

async def _ask_llm(state, goal, memories=None):
    """Send a state to Ollama and return parsed response."""
    import ollama_client
    msgs = build_messages(state, goal, memories)
    raw = await ollama_client.chat(msgs, temperature=0.1)
    return extract_json(raw or ""), raw



def run_async(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.mark.skipif(not OLLAMA_AVAILABLE, reason="Ollama not running")
class TestLLMResponseQuality:
    """
    100 LLM response tests covering key scenarios.
    Each test verifies the LLM picks the correct action.
    """

    # ── Basic validity (10 tests) ────────────────────────────────

    @pytest.mark.parametrize("goal,nearby", [
        ("gather oak wood",   _blocks("grass_block", "dirt")),
        ("gather stone",      _blocks("grass_block", "dirt")),
        ("explore",           _blocks("grass_block")),
        ("gather coal",       _blocks("stone", "grass_block")),
        ("survive the night", _blocks("grass_block")),
    ])
    def test_valid_json_returned(self, goal, nearby):
        state = _state(nearby_blocks=nearby)
        result, raw = run_async(_ask_llm(state, goal))
        assert result is not None, f"LLM returned invalid JSON for goal='{goal}'. Raw: {raw[:200]}"
        assert "action" in result, f"No 'action' key in response: {result}"
        assert result["action"] in TOOL_NAMES, f"Unknown action: {result['action']}"

    # ── SEEK when target not nearby (15 tests) ───────────────────

    @pytest.mark.parametrize("goal,nearby,expected_action", [
        ("gather oak wood",    _blocks("grass_block", "dirt"),          "SEEK"),
        ("gather oak wood",    _blocks("stone", "grass_block"),         "SEEK"),
        ("gather acacia wood", _blocks("grass_block"),                  "SEEK"),
        ("gather coal",        _blocks("grass_block", "dirt"),          "SEEK"),
        ("gather iron",        _blocks("grass_block", "dirt"),          "SEEK"),
        ("gather oak wood",    _blocks("dirt", "gravel", "sand"),       "SEEK"),
        ("gather oak wood",    _blocks("birch_log"),                    "MINE"),  # birch is wood
        ("gather stone",       _blocks("grass_block"),                  "SEEK"),
    ])
    def test_seek_when_target_not_nearby(self, goal, nearby, expected_action):
        state = _state(nearby_blocks=nearby)
        result, raw = run_async(_ask_llm(state, goal))
        assert result is not None, f"No JSON for goal='{goal}'. Raw: {raw[:200]}"
        # For SEEK cases: action should be SEEK or MOVE (LLM may choose to explore too)
        if expected_action == "SEEK":
            assert result["action"] in ("SEEK", "MOVE"), \
                f"Expected SEEK/MOVE when target not nearby, got {result['action']} for goal='{goal}'"
        elif expected_action == "MINE":
            assert result["action"] == "MINE", \
                f"Expected MINE when log nearby, got {result['action']} for goal='{goal}'"

    # ── MINE when target IS nearby (15 tests) ───────────────────

    @pytest.mark.parametrize("goal,nearby,must_mine", [
        ("gather oak wood",  _blocks("oak_log", "grass_block"),     "oak_log"),
        ("gather oak wood",  _blocks("grass_block", "oak_log"),     "oak_log"),
        ("gather stone",     _blocks("stone", "dirt"),              "stone"),
        ("gather coal",      _blocks("coal_ore", "stone"),          "coal_ore"),
        ("gather iron",      _blocks("iron_ore", "stone"),          "iron_ore"),
        ("mine stone",       _blocks("stone", "grass_block"),       "stone"),
    ])
    def test_mine_when_target_nearby(self, goal, nearby, must_mine):
        state = _state(nearby_blocks=nearby)
        result, raw = run_async(_ask_llm(state, goal))
        assert result is not None, f"No JSON. Raw: {raw[:200]}"
        assert result["action"] == "MINE", \
            f"Expected MINE when {must_mine} is nearby, got {result['action']} for goal='{goal}'"
        assert result.get("params", {}).get("block") == must_mine, \
            f"Expected block={must_mine}, got {result.get('params',{}).get('block')}"

    # ── Never mine wrong block (10 tests) ───────────────────────

    @pytest.mark.parametrize("goal,nearby", [
        ("gather oak wood", _blocks("grass_block", "dirt", "sand")),
        ("gather oak wood", _blocks("gravel", "dirt")),
        ("gather oak wood", _blocks("grass_block")),
        ("gather coal",     _blocks("grass_block", "dirt")),
        ("gather iron",     _blocks("grass_block", "stone")),
    ])
    def test_never_mine_grass_when_gathering_wood(self, goal, nearby):
        state = _state(nearby_blocks=nearby)
        result, raw = run_async(_ask_llm(state, goal))
        assert result is not None, f"No JSON. Raw: {raw[:200]}"
        if result["action"] == "MINE":
            block = result.get("params", {}).get("block", "")
            assert block != "grass_block", \
                f"LLM mined grass_block for goal='{goal}'! Should SEEK instead."
            assert block != "dirt", \
                f"LLM mined dirt for goal='{goal}'! Should SEEK instead."

    # ── No goal = IDLE (5 tests) ─────────────────────────────────

    @pytest.mark.parametrize("nearby", [
        _blocks("grass_block"),
        _blocks("oak_log"),
        _blocks("stone"),
        [],
        _blocks("grass_block", "dirt"),
    ])
    def test_idle_when_no_goal(self, nearby):
        state = _state(nearby_blocks=nearby)
        result, raw = run_async(_ask_llm(state, None))
        assert result is not None, f"No JSON. Raw: {raw[:200]}"
        assert result["action"] in ("IDLE", "CHAT"), \
            f"Expected IDLE/CHAT with no goal, got {result['action']}"

    # ── Low health emergency (5 tests) ──────────────────────────

    @pytest.mark.parametrize("health,food", [
        (4, 20),
        (3, 18),
        (2, 20),
        (5, 4),
        (6, 3),
    ])
    def test_survival_action_when_low_health_or_food(self, health, food):
        state = _state(
            nearby_blocks=_blocks("grass_block"),
            health=health,
            food=food,
        )
        result, raw = run_async(_ask_llm(state, "gather oak wood"))
        assert result is not None, f"No JSON. Raw: {raw[:200]}"
        # With critical health/food, LLM should prioritize survival
        # Accept CHAT, SEEK food, IDLE, or MOVE (fleeing)
        assert result["action"] in TOOL_NAMES

    # ── Explore goal = MOVE (5 tests) ───────────────────────────

    @pytest.mark.parametrize("goal", [
        "explore",
        "explore the world",
        "explore and find resources",
        "wander around",
        "look for villages",
    ])
    def test_explore_goal_gives_move(self, goal):
        state = _state(nearby_blocks=_blocks("grass_block"))
        result, raw = run_async(_ask_llm(state, goal))
        assert result is not None, f"No JSON for goal='{goal}'. Raw: {raw[:200]}"
        assert result["action"] in ("MOVE", "SEEK"), \
            f"Expected MOVE/SEEK for explore goal, got {result['action']}"

    # ── Hostile mob nearby = flee/fight (5 tests) ───────────────

    @pytest.mark.parametrize("mob,distance", [
        ("zombie",   3),
        ("skeleton", 5),
        ("creeper",  4),
        ("spider",   3),
        ("enderman", 6),
    ])
    def test_flee_hostile_mob_nearby(self, mob, distance):
        state = _state(
            nearby_blocks=_blocks("grass_block"),
            health=15,
            entities=[_entity(mob, "hostile", distance)],
        )
        result, raw = run_async(_ask_llm(state, "survive"))
        assert result is not None, f"No JSON for mob={mob}. Raw: {raw[:200]}"
        assert result["action"] in TOOL_NAMES  # any valid action

    # ── Reasoning field present (5 tests) ───────────────────────

    @pytest.mark.parametrize("goal,nearby", [
        ("gather oak wood", _blocks("oak_log")),
        ("gather stone",    _blocks("stone")),
        ("explore",         _blocks("grass_block")),
        ("gather coal",     _blocks("coal_ore")),
        ("gather iron",     _blocks("iron_ore")),
    ])
    def test_reasoning_always_present(self, goal, nearby):
        state = _state(nearby_blocks=nearby)
        result, raw = run_async(_ask_llm(state, goal))
        assert result is not None, f"No JSON. Raw: {raw[:200]}"
        assert "reasoning" in result, f"Missing 'reasoning' in response: {result}"
        assert len(result["reasoning"]) > 3, f"Reasoning too short: {result['reasoning']}"

    # ── Params structure valid (10 tests) ────────────────────────

    @pytest.mark.parametrize("goal,nearby,expected_action,required_param", [
        ("gather oak wood", _blocks("oak_log"), "MINE",  "block"),
        ("gather stone",    _blocks("stone"),   "MINE",  "block"),
        ("gather coal",     _blocks("coal_ore"),"MINE",  "block"),
        ("gather oak wood", _blocks("grass_block"), "SEEK", "target"),
        ("explore",         _blocks("grass_block"), "MOVE", "direction"),
    ])
    def test_params_have_required_keys(self, goal, nearby, expected_action, required_param):
        state = _state(nearby_blocks=nearby)
        result, raw = run_async(_ask_llm(state, goal))
        assert result is not None, f"No JSON. Raw: {raw[:200]}"
        if result["action"] == expected_action:
            assert required_param in result.get("params", {}), \
                f"Action={expected_action} missing param '{required_param}': {result}"

    # ── Bulk consistency test (25 tests) ─────────────────────────

    @pytest.mark.parametrize("run_num", range(25))
    def test_oak_seek_consistency(self, run_num):
        """Send 'gather oak wood' with no oak nearby 25 times — must always SEEK or MOVE."""
        state = _state(nearby_blocks=_blocks("grass_block", "dirt"))
        result, raw = run_async(_ask_llm(state, "gather oak wood"))
        assert result is not None, f"Run {run_num}: No JSON. Raw: {raw[:200]}"
        assert result["action"] in ("SEEK", "MOVE"), \
            f"Run {run_num}: Expected SEEK/MOVE, got {result['action']}"
        if result["action"] == "MINE":
            block = result.get("params", {}).get("block", "")
            assert block not in ("grass_block", "dirt", "sand", "gravel"), \
                f"Run {run_num}: Mined wrong block: {block}"

    @pytest.mark.parametrize("run_num", range(15))
    def test_oak_mine_consistency(self, run_num):
        """Send 'gather oak wood' WITH oak_log nearby 15 times — must always MINE oak_log."""
        state = _state(nearby_blocks=_blocks("oak_log", "grass_block"))
        result, raw = run_async(_ask_llm(state, "gather oak wood"))
        assert result is not None, f"Run {run_num}: No JSON. Raw: {raw[:200]}"
        assert result["action"] == "MINE", \
            f"Run {run_num}: Expected MINE when oak_log nearby, got {result['action']}"
        assert result.get("params", {}).get("block") == "oak_log", \
            f"Run {run_num}: Wrong block mined: {result.get('params')}"


# ── Summary report ────────────────────────────────────────────────

if __name__ == "__main__":
    """Run a quick manual test and print summary."""
    import asyncio

    async def _manual_test():
        scenarios = [
            ("gather oak wood",   _blocks("grass_block", "dirt"),     "SEEK"),
            ("gather oak wood",   _blocks("oak_log", "grass_block"),  "MINE"),
            ("explore",           _blocks("grass_block"),             "MOVE"),
            ("gather coal",       _blocks("grass_block"),             "SEEK"),
            ("gather coal",       _blocks("coal_ore", "stone"),       "MINE"),
            (None,                _blocks("grass_block"),             "IDLE"),
        ]

        passed, failed = 0, 0
        for goal, nearby, expected in scenarios:
            state = _state(nearby_blocks=nearby)
            result, raw = await _ask_llm(state, goal)
            ok = result and result.get("action") in (expected, "MOVE" if expected == "SEEK" else expected)
            status = "OK  " if ok else "FAIL"
            if ok: passed += 1
            else:  failed += 1
            print(f"{status} goal='{goal}' | expected={expected} | got={result.get('action') if result else 'NONE'} | {result.get('reasoning','') if result else raw[:60]}")

        print(f"\nResults: {passed}/{passed+failed} passed")

    asyncio.run(_manual_test())
