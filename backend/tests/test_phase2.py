"""
test_phase2.py — Phase 2 unit tests (no Ollama or ChromaDB required).
Run: python tests/test_phase2.py
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prompt_builder import build_messages, extract_json, build_correction_messages
from tool_registry  import get_tool_schema_text, TOOL_NAMES, TOOLS
from main           import simple_planner, ObservationState, PlayerInfo, Environment

PASSED = FAILED = 0

def assert_(cond, msg="assertion failed"):
    if not cond: raise AssertionError(msg)

def test(name, fn):
    global PASSED, FAILED
    try:
        fn()
        print(f"  [PASS] {name}")
        PASSED += 1
    except Exception as e:
        print(f"  [FAIL] {name}\n      -> {e}")
        FAILED += 1

def state(**kwargs):
    defaults = dict(
        player=PlayerInfo(health=20, food=20),
        inventory={}, nearby_entities=[], nearby_blocks=[],
        environment=Environment(time_of_day="noon", weather="clear"),
        goal=None, timestamp="2026-05-15T18:00:00Z",
    )
    defaults.update(kwargs)
    return ObservationState(**defaults)

print("\nPhase 2 Unit Tests\n")

# ── tool_registry ─────────────────────────────────────────────────
test("TOOL_NAMES contains all expected actions", lambda: (
    [assert_(t in TOOL_NAMES, f"{t} missing") for t in
     ["MOVE","MINE","CRAFT","EAT","CHAT","FOLLOW","GOTO","STOP","IDLE","COLLECT"]]
))
test("get_tool_schema_text returns non-empty string", lambda:
    assert_(len(get_tool_schema_text()) > 50)
)
test("Every tool has name+description+params", lambda:
    [assert_("name" in t and "description" in t and "params" in t) for t in TOOLS]
)

# ── prompt_builder — extract_json ─────────────────────────────────
test("extract_json — plain JSON", lambda:
    assert_(extract_json('{"action":"IDLE","params":{}}') == {"action":"IDLE","params":{}})
)
test("extract_json — markdown block", lambda:
    assert_(extract_json('```json\n{"action":"MINE","params":{"block":"oak_log"}}\n```') is not None)
)
test("extract_json — embedded in prose", lambda:
    assert_(extract_json('I think you should {"action":"STOP","params":{}} now') is not None)
)
test("extract_json — returns None for garbage", lambda:
    assert_(extract_json("hello world no json here") is None)
)
test("extract_json — action field preserved", lambda: (
    r := extract_json('{"action":"CHAT","params":{"message":"hi"},"reasoning":"test"}'),
    assert_(r["action"] == "CHAT")
))

# ── prompt_builder — build_messages ──────────────────────────────
def _test_build_messages():
    msgs = build_messages(
        {"player":{"health":18},"inventory":{"oak_log":3},"nearby_blocks":[],"nearby_entities":[],"environment":{}},
        "gather wood",
        memories=["Past: mined oak_log at (100,64,-22)"]
    )
    assert len(msgs) == 2, "Should have system + user message"
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "gather wood" in msgs[1]["content"]
    assert "oak_log" in msgs[1]["content"]
    assert "Past:" in msgs[1]["content"]
test("build_messages — returns [system, user] with goal and memories", _test_build_messages)

def _test_build_no_memory():
    msgs = build_messages({"player":{},"inventory":{},"nearby_blocks":[],"nearby_entities":[],"environment":{}}, "survive")
    assert len(msgs) == 2
    assert "survive" in msgs[1]["content"]
test("build_messages — works without memories", _test_build_no_memory)

def _test_correction_messages():
    orig = [{"role":"system","content":"sys"},{"role":"user","content":"usr"}]
    corrected = build_correction_messages(orig, "bad response")
    assert len(corrected) == 4
    assert corrected[2]["role"] == "assistant"
    assert corrected[3]["role"] == "user"
test("build_correction_messages — appends bad response + correction", _test_correction_messages)

test("system prompt contains tool names", lambda: (
    msgs := build_messages({"player":{},"inventory":{},"nearby_blocks":[],"nearby_entities":[],"environment":{}}, "test"),
    assert_("MINE" in msgs[0]["content"] and "IDLE" in msgs[0]["content"])
))

# ── simple_planner (fallback) ─────────────────────────────────────
test("fallback — IDLE when healthy", lambda:
    assert_(simple_planner(state()).action == "IDLE")
)
test("fallback — CHAT when hp < 8", lambda:
    assert_(simple_planner(state(player=PlayerInfo(health=5,food=20))).action == "CHAT")
)
test("fallback — SEEK when food < 6", lambda:
    assert_(simple_planner(state(player=PlayerInfo(health=20,food=3))).action == "SEEK")
)
test("fallback — source is 'fallback'", lambda:
    assert_(simple_planner(state()).source == "fallback")
)
test("fallback — MINE when no wood and log nearby", lambda: (
    s := state(
        inventory={},
        nearby_blocks=[{"name":"oak_log","distance":3,"position":{"x":1,"y":64,"z":1}}],
        goal="build shelter"
    ),
    assert_(simple_planner(s).action == "MINE")
))

# ── JSON round-trip ───────────────────────────────────────────────
test("extract_json round-trip with all tool names", lambda:
    [assert_(extract_json(f'{{"action":"{t}","params":{{}}}}') is not None, t)
     for t in TOOL_NAMES]
)

# ── Summary ───────────────────────────────────────────────────────
print(f"\nResults: {PASSED} passed, {FAILED} failed\n")
if FAILED > 0:
    sys.exit(1)


