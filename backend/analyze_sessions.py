"""
analyze_sessions.py
Analyzes session_log.jsonl to identify:
  - LLM success rate vs fallback rate
  - Most common actions
  - Common failure patterns (invalid JSON, fallbacks)
  - Goal coverage

Run:  python analyze_sessions.py
      python analyze_sessions.py --tail 50   (last 50 entries only)
"""

import argparse
import json
import os
from collections import Counter
from typing import List, Dict, Any

LOG_FILE = os.path.join(os.path.dirname(__file__), "..", "data", "sessions", "session_log.jsonl")


def load_entries(tail: int = 0) -> List[Dict[str, Any]]:
    if not os.path.exists(LOG_FILE):
        print(f"No session log found at: {LOG_FILE}")
        return []
    entries = []
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return entries[-tail:] if tail > 0 else entries


def analyze(entries: List[Dict]) -> None:
    if not entries:
        print("No entries to analyze.")
        return

    total   = len(entries)
    sources = Counter(e.get("source", "unknown") for e in entries)
    actions = Counter(e.get("action", {}).get("action", "?") for e in entries)
    goals   = Counter(e.get("goal", "none") for e in entries)

    # HP distribution
    healths = [e["player"]["health"] for e in entries if e.get("player", {}).get("health") is not None]
    foods   = [e["player"]["food"]   for e in entries if e.get("player", {}).get("food")   is not None]

    # Fallback patterns
    fallback_actions = Counter(
        e.get("action", {}).get("action", "?")
        for e in entries if e.get("source") == "fallback"
    )

    # LLM response quality
    no_raw   = sum(1 for e in entries if not e.get("raw_llm_response"))
    with_mem = sum(1 for e in entries if e.get("memories_used"))

    print("\n" + "═" * 55)
    print("  MineAgent Session Log Analysis")
    print("═" * 55)
    print(f"\n  Total interactions:  {total}")
    print(f"  Log file:            {LOG_FILE}\n")

    print("─ Source breakdown ─────────────────────────────────")
    for src, cnt in sources.most_common():
        pct = 100 * cnt / total
        bar = "█" * int(pct / 5)
        print(f"  {src:<12} {cnt:>4}  ({pct:5.1f}%)  {bar}")

    print("\n─ Top actions ───────────────────────────────────────")
    for act, cnt in actions.most_common(8):
        pct = 100 * cnt / total
        print(f"  {act:<14} {cnt:>4}  ({pct:5.1f}%)")

    print("\n─ Top goals ─────────────────────────────────────────")
    for goal, cnt in goals.most_common(5):
        short = goal[:40] + "..." if len(goal) > 40 else goal
        print(f"  {short:<43} {cnt:>4}")

    print("\n─ Health/food stats ─────────────────────────────────")
    if healths:
        print(f"  Avg HP:    {sum(healths)/len(healths):.1f}  "
              f"Min: {min(healths):.0f}  Max: {max(healths):.0f}")
    if foods:
        print(f"  Avg Food:  {sum(foods)/len(foods):.1f}  "
              f"Min: {min(foods):.0f}  Max: {max(foods):.0f}")

    print("\n─ Fallback patterns ─────────────────────────────────")
    if fallback_actions:
        for act, cnt in fallback_actions.most_common(5):
            print(f"  {act:<14} {cnt:>4}")
    else:
        print("  No fallbacks recorded")

    print("\n─ Quality signals ───────────────────────────────────")
    llm_rate = 100 * sources.get("llm", 0) / total
    print(f"  LLM success rate:   {llm_rate:.1f}%")
    print(f"  Fallback rate:      {100 - llm_rate:.1f}%")
    print(f"  Calls with no RAW:  {no_raw} (Ollama was down)")
    print(f"  Calls using memory: {with_mem} ({100*with_mem/total:.1f}%)")
    print("\n" + "═" * 55 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze MineAgent session logs")
    parser.add_argument("--tail", type=int, default=0, help="Analyse only last N entries (0=all)")
    args = parser.parse_args()

    entries = load_entries(args.tail)
    analyze(entries)
