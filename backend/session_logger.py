"""
session_logger.py
Logs every LLM interaction to a JSONL file for:
  - Debugging response quality
  - Building fine-tuning dataset later (Phase 3)
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("mineagent.session")

LOG_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "sessions")
LOG_FILE = os.path.join(LOG_DIR, "session_log.jsonl")


def log_interaction(
    state:        Dict[str, Any],
    goal:         str,
    memories:     List[str],
    raw_response: Optional[str],
    action:       Dict[str, Any],
    source:       str,   # "llm" | "fallback" | "correction"
) -> None:
    """Append one planning interaction to the JSONL session log."""
    try:
        os.makedirs(LOG_DIR, exist_ok=True)

        entry = {
            "timestamp":    datetime.now(timezone.utc).isoformat(),
            "goal":         goal,
            "source":       source,
            "player": {
                "health":   state.get("player", {}).get("health"),
                "food":     state.get("player", {}).get("food"),
                "position": state.get("player", {}).get("position"),
            },
            "inventory":    state.get("inventory", {}),
            "nearby_blocks": [b["name"] for b in state.get("nearby_blocks", [])[:5]],
            "memories_used": memories[:3],
            "raw_llm_response": raw_response,
            "action": action,
        }

        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    except Exception as e:
        logger.error("Session log write failed: %s", e)


def get_stats() -> Dict[str, Any]:
    """Return basic stats about the session log."""
    if not os.path.exists(LOG_FILE):
        return {"total": 0, "log_file": LOG_FILE}

    total = 0
    sources: Dict[str, int] = {}
    actions: Dict[str, int] = {}

    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                total += 1
                s = entry.get("source", "unknown")
                sources[s] = sources.get(s, 0) + 1
                a = entry.get("action", {}).get("action", "?")
                actions[a] = actions.get(a, 0) + 1
            except json.JSONDecodeError:
                pass

    return {
        "total":      total,
        "by_source":  sources,
        "by_action":  actions,
        "log_file":   LOG_FILE,
    }
