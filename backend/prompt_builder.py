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

# ── Prompt templates ──────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are MineAgent, an autonomous AI playing Minecraft.

RULES:
- Respond with ONLY a valid JSON object — no markdown, no explanation outside the JSON.
- Format: {{"action": "NAME", "params": {{}}, "reasoning": "brief reason"}}
- Prioritise survival: health (must stay > 8) > food (must stay > 6) > goal
- Use block names exactly as they appear in nearby_blocks (e.g. oak_log, iron_ore)

Available actions:
{tools}

FEW-SHOT EXAMPLES:
State: health=4, food=18 → {{"action":"CHAT","params":{{"message":"Health critical!"}},"reasoning":"HP below 8, alerting player"}}
State: health=20, food=3 → {{"action":"SEEK","params":{{"target":"food"}},"reasoning":"Food dangerously low"}}
State: inventory has no wood, oak_log nearby, goal=build → {{"action":"MINE","params":{{"block":"oak_log"}},"reasoning":"Need wood to start building"}}
State: zombie at distance 3, health=15 → {{"action":"MOVE","params":{{"direction":"south","distance":10}},"reasoning":"Fleeing zombie before it attacks"}}
State: health=20, food=18, goal=explore → {{"action":"MOVE","params":{{"direction":"north","distance":20}},"reasoning":"Exploring as instructed"}}
"""

USER_TEMPLATE = """\
Game state:
{state_json}
{craftable_hint}
{memory_block}
Goal: {goal}

Choose one action (JSON only):"""

MEMORY_BLOCK = """\

Relevant past experiences:
{memories}
"""

CORRECTION_PROMPT = """\
Your last response was not valid JSON. Reply again with ONLY a JSON object in this format:
{{"action": "ACTION_NAME", "params": {{}}, "reasoning": "reason"}}
"""


def build_messages(
    state: Dict[str, Any],
    goal: str,
    memories: Optional[List[str]] = None,
) -> List[Dict[str, str]]:
    """Return [system, user] message list ready for Ollama /api/chat."""

    system = SYSTEM_PROMPT.format(tools=get_tool_schema_text())

    # Trim state for token efficiency
    lean_state = {
        "player":          state.get("player", {}),
        "inventory":       state.get("inventory", {}),
        "nearby_entities": state.get("nearby_entities", [])[:5],
        "nearby_blocks":   state.get("nearby_blocks", [])[:8],
        "environment":     state.get("environment", {}),
    }

    # Inventory-aware: tell LLM what it can craft right now
    craftable_hint = get_craftable_hint(state.get("inventory", {}))

    mem_block = ""
    if memories:
        mem_lines = "\n".join(f"  - {m}" for m in memories[:3])
        mem_block = MEMORY_BLOCK.format(memories=mem_lines)

    user = USER_TEMPLATE.format(
        state_json=json.dumps(lean_state, indent=2),
        craftable_hint=craftable_hint,
        memory_block=mem_block,
        goal=goal or "survive and explore",
    )

    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]


def build_correction_messages(original_messages: List[Dict], bad_response: str) -> List[Dict]:
    """Append bad response + correction prompt to force valid JSON."""
    return original_messages + [
        {"role": "assistant", "content": bad_response},
        {"role": "user",      "content": CORRECTION_PROMPT},
    ]


def extract_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract a JSON object from LLM response text.
    Handles: raw JSON, ```json blocks, JSON embedded in prose.
    """
    text = text.strip()

    # 1. Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. ```json ... ``` block
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 3. First balanced { ... } in text (handles nested objects like "params":{})
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    logger.warning("Could not extract JSON from: %s", text[:200])
    return None
