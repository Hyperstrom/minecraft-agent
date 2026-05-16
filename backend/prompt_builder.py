"""
prompt_builder.py — Builds Ollama messages and parses JSON from LLM responses.
"""

import json
import re
import logging
from typing import Any, Dict, List, Optional

from tool_registry import get_tool_schema_text
from recipe_advisor import get_craftable_hint

logger = logging.getLogger("mineagent.prompt")

# ── System prompt ────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are MineAgent, an autonomous Minecraft AI. You MUST follow ALL rules below.

STRICT RULES (never break these):
1. Respond with ONLY a valid JSON object — no markdown, no prose outside JSON.
2. Format: {{"action": "NAME", "params": {{}}, "reasoning": "short reason"}}
3. Survival priority: health > 8 FIRST, food > 6 SECOND, then goal.
4. NEVER mine a block that is NOT in "nearby_blocks" list.
5. If your target resource is NOT in "nearby_blocks", use SEEK to find it.
6. NEVER mine grass_block, dirt, sand, or gravel unless that IS the explicit goal.
7. Use block names EXACTLY as shown in nearby_blocks (e.g. oak_log, not wood).

RESOURCE GATHERING LOGIC:
- Goal contains "oak wood"     → MINE oak_log ONLY. If no oak_log nearby → SEEK oak_log
- Goal contains "stone"        → MINE stone ONLY. If no stone nearby → SEEK stone
- Goal contains "coal"         → MINE coal_ore. If not nearby → SEEK coal_ore
- Goal contains "explore"      → MOVE in a direction
- Goal contains "craft"        → check craftable items, then CRAFT
- No goal set                  → IDLE and wait for instructions

Available actions:
{tools}

FEW-SHOT EXAMPLES (study these carefully):
State: goal=gather oak wood, nearby=[grass_block,dirt] → {{"action":"SEEK","params":{{"target":"oak_log"}},"reasoning":"oak_log not nearby, searching for trees"}}
State: goal=gather oak wood, nearby=[oak_log(d=3),dirt] → {{"action":"MINE","params":{{"block":"oak_log"}},"reasoning":"oak_log found nearby, mining it"}}
State: goal=gather oak wood, nearby=[birch_log,grass_block] → {{"action":"MINE","params":{{"block":"birch_log"}},"reasoning":"birch_log is wood, acceptable for goal"}}
State: goal=explore, nearby=[grass_block,oak_log] → {{"action":"MOVE","params":{{"direction":"north","distance":20}},"reasoning":"exploring as instructed"}}
State: health=5, food=18 → {{"action":"CHAT","params":{{"message":"Low health!"}},"reasoning":"health critical, alerting player"}}
State: food=3, health=18 → {{"action":"SEEK","params":{{"target":"food"}},"reasoning":"food low, searching for food"}}
State: no goal set → {{"action":"IDLE","params":{{}},"reasoning":"waiting for goal from player"}}
State: goal=craft wooden_pickaxe, craftable=[wooden_pickaxe] → {{"action":"CRAFT","params":{{"item":"wooden_pickaxe"}},"reasoning":"have materials, crafting pickaxe"}}
State: goal=craft pickaxe, craftable=[], nearby=[oak_log] → {{"action":"MINE","params":{{"block":"oak_log"}},"reasoning":"need wood first to craft"}}
State: zombie at d=3, health=15 → {{"action":"MOVE","params":{{"direction":"south","distance":15}},"reasoning":"fleeing zombie"}}
"""

USER_TEMPLATE = """\
=== CURRENT GAME STATE ===
{state_json}

=== SITUATION ANALYSIS ===
{situation}

=== GOAL PROGRESS ===
{goal_progress}

=== RECENT ACTIONS (last {recent_count}) ===
{recent_actions}

=== WHAT YOU CAN CRAFT NOW ===
{craftable_hint}

=== RELEVANT MEMORIES ===
{memory_block}

=== YOUR CURRENT GOAL ===
{goal}

Based on the situation analysis above, choose ONE action (JSON only):"""

CORRECTION_PROMPT = """\
Your last response was not valid JSON. You MUST reply with ONLY a JSON object:
{{"action": "ACTION_NAME", "params": {{}}, "reasoning": "reason"}}
No other text. Just the JSON."""


def _situation_analysis(state: Dict[str, Any], goal: str) -> str:
    """Generate a natural-language situation summary to guide LLM decisions."""
    lines = []

    # Health/food
    hp   = state.get("player", {}).get("health", 20)
    food = state.get("player", {}).get("food", 20)
    if hp < 8:   lines.append(f"⚠️  CRITICAL: Health is {hp}/20 — must address immediately!")
    if food < 6: lines.append(f"⚠️  CRITICAL: Food is {food}/20 — must eat soon!")

    # Nearby blocks
    nearby = state.get("nearby_blocks", [])
    block_names = [b["name"] for b in nearby]

    if nearby:
        lines.append(f"Nearby blocks: {', '.join(block_names[:8])}")
    else:
        lines.append("Nearby blocks: none detected")

    # Goal-specific analysis
    goal_lower = goal.lower() if goal else ""
    if not goal or goal.strip() == "":
        lines.append("No goal set → use IDLE and wait for player instructions.")
    elif any(w in goal_lower for w in ["oak", "wood", "log", "tree"]):
        wood_blocks = [b for b in block_names if b.endswith("_log")]
        if wood_blocks:
            lines.append(f"✅ Wood found nearby: {wood_blocks[0]} — use MINE")
        else:
            lines.append("❌ No wood/logs in nearby_blocks — use SEEK with target='oak_log'")
    elif "acacia" in goal_lower:
        acacia = [b for b in block_names if "acacia" in b]
        if acacia:
            lines.append(f"✅ Acacia found: {acacia[0]} — use MINE")
        else:
            lines.append("❌ No acacia_log nearby — use SEEK with target='acacia_log'")
    elif "stone" in goal_lower or "cobblestone" in goal_lower:
        stone = [b for b in block_names if "stone" in b or "cobblestone" in b]
        if stone:
            lines.append(f"✅ Stone found: {stone[0]} — use MINE")
        else:
            lines.append("❌ No stone nearby — use SEEK with target='stone'")
    elif "coal" in goal_lower:
        coal = [b for b in block_names if "coal" in b]
        if coal:
            lines.append(f"✅ Coal found: {coal[0]} — use MINE")
        else:
            lines.append("❌ No coal_ore nearby — use SEEK with target='coal_ore'")
    elif "explore" in goal_lower:
        lines.append("Goal is exploration → use MOVE in any direction.")
    elif "craft" in goal_lower:
        lines.append("Craft goal → check craftable items above. If craftable, CRAFT. If not, mine missing ingredients.")
    elif "survive" in goal_lower:
        lines.append("Survival goal → check health/food first, then explore if safe.")

    # Entities
    entities = state.get("nearby_entities", [])
    hostile = [e for e in entities if e.get("type") == "hostile"]
    if hostile:
        names = [e["name"] for e in hostile[:3]]
        lines.append(f"⚠️  HOSTILE MOBS nearby: {', '.join(names)} — consider fleeing!")

    return "\n".join(lines)


def build_messages(
    state: Dict[str, Any],
    goal: str,
    memories: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """Return [system, user] message list ready for Ollama /api/chat."""

    system = SYSTEM_PROMPT.format(tools=get_tool_schema_text())

    lean_state = {
        "player":          state.get("player", {}),
        "inventory":       state.get("inventory", {}),
        "nearby_entities": state.get("nearby_entities", [])[:5],
        "nearby_blocks":   state.get("nearby_blocks", [])[:8],
        "environment":     state.get("environment", {}),
    }

    # Goal progress (sent by bot's goal_tracker)
    goal_progress   = state.get("goal_progress", "No structured goal — open-ended.")

    # Recent action history (last N actions the bot took)
    recent_raw      = state.get("recent_actions", [])
    recent_actions  = "\n".join(f"  {i+1}. {a}" for i, a in enumerate(recent_raw)) or "  None yet"

    situation    = _situation_analysis(state, goal or "")
    craftable_h  = get_craftable_hint(state.get("inventory", {}))
    mem_block    = "\n".join(f"- {m}" for m in (memories or [])[:3]) or "None"

    user = USER_TEMPLATE.format(
        state_json=json.dumps(lean_state, indent=2),
        situation=situation,
        goal_progress=goal_progress,
        recent_actions=recent_actions,
        recent_count=len(recent_raw),
        craftable_hint=craftable_h,
        memory_block=mem_block,
        goal=goal or "No goal set — wait for player to set a goal using !goal",
    )

    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]


def build_correction_messages(original_messages: List[Dict], bad_response: str) -> List[Dict]:
    return original_messages + [
        {"role": "assistant", "content": bad_response},
        {"role": "user",      "content": CORRECTION_PROMPT},
    ]


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract JSON from LLM response — handles raw, markdown block, and embedded."""
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    logger.warning("Could not extract JSON from: %s", text[:200])
    return None
