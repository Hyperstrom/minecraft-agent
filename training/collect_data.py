"""
MineAgent — Data Collection for Kaggle
Collects from: Wiki + Reddit + YouTube + bot sessions
Run: python collect_data.py
Output: data/stage1.jsonl, data/stage2.jsonl
"""
import json, time, os, requests
from pathlib import Path

Path("data").mkdir(exist_ok=True)

SYSTEM_KNOWLEDGE = (
    "You are MineAgent, an expert Minecraft AI. "
    "Answer accurately about Minecraft mechanics, crafting, and survival."
)
SYSTEM_ACTION = (
    "You are MineAgent, an autonomous Minecraft AI. "
    "Given the game state and goal, output ONLY valid JSON: "
    '{"action":"NAME","params":{},"reasoning":"short reason"}'
)

# ── Shared helpers ────────────────────────────────────────────────

def qa_to_chat(instruction, output, system=SYSTEM_KNOWLEDGE):
    return {"messages": [
        {"role": "system",    "content": system},
        {"role": "user",      "content": instruction},
        {"role": "assistant", "content": output},
    ]}

def write_jsonl(path, items):
    with open(path, "w", encoding="utf-8") as f:
        for x in items:
            f.write(json.dumps(x) + "\n")
    print(f"Saved {len(items)} → {path}")

# ── Source 1: Minecraft Wiki ──────────────────────────────────────

def collect_wiki(max_pages=300):
    API = "https://minecraft.wiki/api.php"
    CATEGORIES = ["Blocks","Items","Crafting","Mobs","Biomes",
                  "Food","Tools","Weapons","Armor","Mechanics"]
    pairs = []

    def get_titles(cat):
        r = requests.get(API, params={"action":"query","list":"categorymembers",
            "cmtitle":f"Category:{cat}","cmlimit":"50","format":"json"}, timeout=10)
        return [p["title"] for p in r.json()["query"]["categorymembers"]]

    def get_text(title):
        r = requests.get(API, params={"action":"query","titles":title,
            "prop":"extracts","explaintext":True,"format":"json"}, timeout=10)
        pages = r.json()["query"]["pages"]
        return list(pages.values())[0].get("extract","")[:1000]

    seen = set()
    for cat in CATEGORIES:
        if len(seen) >= max_pages: break
        for title in get_titles(cat):
            if title in seen or len(seen) >= max_pages: continue
            seen.add(title)
            text = get_text(title)
            if len(text) < 100: continue
            pairs.append(qa_to_chat(f"What is {title} in Minecraft?", text))
            pairs.append(qa_to_chat(f"How do you use {title} in Minecraft?", text[:500]))
            time.sleep(0.3)

    print(f"Wiki: {len(pairs)} pairs from {len(seen)} pages")
    return pairs

# ── Source 2: MineDojo Reddit (streaming) ────────────────────────

def collect_reddit(limit=15000):
    try:
        from minedojo.data import RedditDataset
        ds = RedditDataset(full=False, download=True)
    except Exception as e:
        print(f"Reddit skip: {e}")
        return []

    KEYWORDS = ["craft","mine","build","survive","wood","stone","coal",
                "iron","diamond","pickaxe","hunger","health","recipe"]
    pairs = []
    for item in ds:
        if len(pairs) >= limit: break
        title = item.get("title","")
        body  = item.get("selftext","") or item.get("body","")
        if len(body) < 40: continue
        if not any(k in (title+body).lower() for k in KEYWORDS): continue
        pairs.append(qa_to_chat(title, body[:500]))

    print(f"Reddit: {len(pairs)} pairs")
    return pairs

# ── Source 3: MineDojo YouTube (streaming) ───────────────────────

def collect_youtube(limit=10000):
    try:
        from minedojo.data import YouTubeDataset
        ds = YouTubeDataset(full=False, download=True)
    except Exception as e:
        print(f"YouTube skip: {e}")
        return []

    TASK_KW = ["wood","mine","craft","survive","build","explore","cave","diamond"]
    pairs   = []
    for item in ds:
        if len(pairs) >= limit: break
        title      = item.get("title","")
        transcript = item.get("transcript","")
        if not transcript or len(transcript) < 100: continue
        if not any(k in title.lower() for k in TASK_KW): continue
        words  = transcript.split()
        chunks = [" ".join(words[i:i+150]) for i in range(0,min(len(words),600),150)]
        for chunk in chunks[:4]:
            if len(chunk) < 60: continue
            pairs.append(qa_to_chat(
                f"Minecraft gameplay: {title}",
                chunk.strip()
            ))

    print(f"YouTube: {len(pairs)} pairs")
    return pairs

# ── Source 4: Bot sessions (upload session_log.jsonl to Kaggle) ──

def collect_sessions(path="data/session_log.jsonl"):
    if not Path(path).exists():
        print("No session log found — skip")
        return []

    # Minimal state→action format
    VALID_ACTIONS = {"SEEK","MINE","MOVE","CRAFT","EAT","CHAT","FOLLOW","GOTO","IDLE","STOP"}
    pairs = []
    for line in open(path):
        item = json.loads(line)
        if item.get("source") != "llm": continue
        action = item.get("action",{})
        if action.get("action") not in VALID_ACTIONS: continue
        if action.get("action") == "IDLE": continue

        state_summary = (
            f"HP:{item.get('player',{}).get('health',20)} "
            f"Food:{item.get('player',{}).get('food',20)} "
            f"Inventory:{json.dumps(item.get('inventory',{}))[:100]} "
            f"Nearby:{[b['name'] for b in item.get('nearby_blocks',[])[:5]]} "
            f"Progress:{item.get('goal_progress','')}"
        )
        instruction = f"Goal: {item.get('goal','')}. State: {state_summary}"
        output = json.dumps({
            "action":    action["action"],
            "params":    action.get("params",{}),
            "reasoning": action.get("reasoning",""),
        })
        pairs.append(qa_to_chat(instruction, output, system=SYSTEM_ACTION))

    print(f"Sessions: {len(pairs)} pairs")
    return pairs

# ── Build final datasets ──────────────────────────────────────────

import random

if __name__ == "__main__":
    print("=== Collecting data ===")
    stage1 = []
    stage1 += collect_wiki(max_pages=300)
    stage1 += collect_reddit(limit=15000)
    stage1 += collect_youtube(limit=10000)

    stage2 = []
    stage2 += collect_sessions("data/session_log.jsonl")

    random.shuffle(stage1)
    random.shuffle(stage2)

    write_jsonl("data/stage1.jsonl", stage1)
    write_jsonl("data/stage2.jsonl", stage2)
    print(f"\nTotal — Stage1: {len(stage1)} | Stage2: {len(stage2)}")
    print("Done. Upload data/ folder to Kaggle Dataset.")
