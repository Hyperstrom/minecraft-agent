"""
test_phase2_complete.py — Full Phase 2 test suite (no live services needed)
Run: python tests/test_phase2_complete.py
"""

import sys, os, json, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prompt_builder import build_messages, extract_json, build_correction_messages
from tool_registry  import get_tool_schema_text, TOOL_NAMES
from main           import simple_planner, ObservationState, PlayerInfo, Environment
import session_logger as slog

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

def make_state(**kw):
    d = dict(player=PlayerInfo(health=20, food=20), inventory={},
             nearby_entities=[], nearby_blocks=[],
             environment=Environment(time_of_day="noon", weather="clear"),
             goal=None, timestamp="2026-05-16T00:00:00Z")
    d.update(kw)
    return ObservationState(**d)

print("\nPhase 2 Complete Test Suite\n")

# ── Tool Registry ─────────────────────────────────────────────────
test("All 10 tools present", lambda: [
    assert_(t in TOOL_NAMES) for t in
    ["MOVE","MINE","COLLECT","CRAFT","EAT","CHAT","FOLLOW","GOTO","STOP","IDLE"]
])
test("Tool schema text is non-empty", lambda: assert_(len(get_tool_schema_text()) > 100))

# ── Prompt Builder — Few-Shot Examples ───────────────────────────
def t_fewshot():
    msgs = build_messages({"player":{"health":20},"inventory":{},"nearby_blocks":[],
                           "nearby_entities":[],"environment":{}}, "gather wood")
    assert "FEW-SHOT" in msgs[0]["content"], "System prompt should contain few-shot examples"
    assert "MINE" in msgs[0]["content"]
test("System prompt has few-shot examples", t_fewshot)

def t_prompt_structure():
    msgs = build_messages({"player":{"health":18,"food":16},"inventory":{"oak_log":3},
                           "nearby_blocks":[],"nearby_entities":[],"environment":{}},
                          "build shelter", memories=["Build shelter from planks"])
    assert len(msgs) == 2
    assert "build shelter" in msgs[1]["content"]
    assert "Build shelter" in msgs[1]["content"]
test("Prompt contains goal and memories", t_prompt_structure)

# ── JSON Extractor ────────────────────────────────────────────────
test("extract_json — plain", lambda:
    assert_(extract_json('{"action":"IDLE","params":{}}') == {"action":"IDLE","params":{}}))
test("extract_json — markdown block", lambda:
    assert_(extract_json('```json\n{"action":"MINE","params":{"block":"oak_log"}}\n```') is not None))
test("extract_json — embedded prose", lambda:
    assert_(extract_json('I suggest {"action":"STOP","params":{}} now') is not None))
test("extract_json — None for garbage", lambda:
    assert_(extract_json("no json here at all lol") is None))
test("extract_json — all tool names round-trip", lambda:
    [assert_(extract_json(f'{{"action":"{t}","params":{{}}}}') is not None, t) for t in TOOL_NAMES])

def t_nested_params():
    result = extract_json('{"action":"MOVE","params":{"direction":"north","distance":10},"reasoning":"test"}')
    assert result is not None
    assert result["params"]["direction"] == "north"
    assert result["params"]["distance"] == 10
test("extract_json — nested params preserved", t_nested_params)

def t_correction_messages():
    orig = [{"role":"system","content":"sys"},{"role":"user","content":"usr"}]
    corrected = build_correction_messages(orig, "bad output")
    assert len(corrected) == 4
    assert corrected[2]["role"] == "assistant"
    assert corrected[2]["content"] == "bad output"
    assert corrected[3]["role"] == "user"
    assert "JSON" in corrected[3]["content"]
test("Correction messages structure", t_correction_messages)

# ── Fallback Planner ──────────────────────────────────────────────
test("Fallback — IDLE when healthy", lambda:
    assert_(simple_planner(make_state()).action == "IDLE"))
test("Fallback — CHAT when hp=5", lambda:
    assert_(simple_planner(make_state(player=PlayerInfo(health=5,food=20))).action == "CHAT"))
test("Fallback — SEEK when food=3", lambda:
    assert_(simple_planner(make_state(player=PlayerInfo(health=20,food=3))).action == "SEEK"))
test("Fallback — HP beats food priority", lambda:
    assert_(simple_planner(make_state(player=PlayerInfo(health=4,food=2))).action == "CHAT"))
test("Fallback — MINE when no wood + log nearby", lambda:
    assert_(simple_planner(make_state(
        inventory={},
        nearby_blocks=[{"name":"oak_log","distance":3,"position":{"x":1,"y":64,"z":1}}],
        goal="build"
    )).action == "MINE"))
test("Fallback — source is 'fallback'", lambda:
    assert_(simple_planner(make_state()).source == "fallback"))
test("Fallback — ActionResponse JSON serialisable", lambda:
    assert_(isinstance(json.dumps(simple_planner(make_state()).model_dump()), str)))

# ── Session Logger ────────────────────────────────────────────────
def t_session_log():
    with tempfile.TemporaryDirectory() as tmpdir:
        orig_log = slog.LOG_FILE
        slog.LOG_FILE = os.path.join(tmpdir, "test_session.jsonl")

        slog.log_interaction(
            state={"player":{"health":20,"food":20,"position":{"x":0,"y":64,"z":0}},
                   "inventory":{"oak_log":3},"nearby_blocks":[]},
            goal="gather wood",
            memories=["mine oak_log for wood"],
            raw_response='{"action":"MINE","params":{"block":"oak_log"},"reasoning":"need wood"}',
            action={"action":"MINE","params":{"block":"oak_log"},"reasoning":"need wood"},
            source="llm",
        )

        stats = slog.get_stats()
        assert stats["total"] == 1
        assert stats["by_source"]["llm"] == 1
        assert stats["by_action"]["MINE"] == 1

        slog.LOG_FILE = orig_log
test("Session logger — writes and reads stats", t_session_log)

def t_session_log_multiple():
    with tempfile.TemporaryDirectory() as tmpdir:
        slog.LOG_FILE = os.path.join(tmpdir, "multi.jsonl")
        base_state = {"player":{"health":20,"food":20,"position":None},"inventory":{},"nearby_blocks":[]}

        for action_name in ["IDLE","MINE","CHAT"]:
            slog.log_interaction(base_state, "test", [], None,
                                 {"action":action_name,"params":{}},
                                 "llm" if action_name != "CHAT" else "fallback")

        stats = slog.get_stats()
        assert stats["total"] == 3
        assert stats["by_action"]["IDLE"] == 1
        assert stats["by_source"]["fallback"] == 1
        slog.LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                      "data","sessions","session_log.jsonl")
test("Session logger — multiple entries with correct stats", t_session_log_multiple)

# ── Knowledge Seeder (dry-run, no ChromaDB needed) ────────────────
def t_seeder_data():
    from knowledge_seeder import RECIPES, SURVIVAL_TIPS, COMBAT_TIPS, BIOME_TIPS
    assert len(RECIPES) >= 10, "Should have at least 10 recipes"
    assert len(SURVIVAL_TIPS) >= 10, "Should have at least 10 survival tips"
    for topic, text in RECIPES:
        assert len(topic) > 0 and len(text) > 10, f"Bad recipe entry: {topic}"
test("Knowledge seeder — data entries are valid", t_seeder_data)

# ── Summary ───────────────────────────────────────────────────────
print(f"\nResults: {PASSED} passed, {FAILED} failed\n")
if FAILED > 0:
    sys.exit(1)
