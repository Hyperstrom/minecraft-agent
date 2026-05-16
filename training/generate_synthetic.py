"""
generate_synthetic.py
Generates 1600 unique synthetic Minecraft training examples.
Each covers a different goal, state, inventory, and action combination.

Run: python training/generate_synthetic.py
Output: training/synthetic_stage2.jsonl
"""

import json, random
from pathlib import Path
from itertools import product

random.seed(42)

SYS_ACT = (
    'You are MineAgent, an autonomous Minecraft AI. '
    'Given the game state and goal, output ONLY valid JSON: '
    '{"action":"NAME","params":{},"reasoning":"short reason"}'
)

def chat(q, a):
    return {"messages": [
        {"role": "system",    "content": SYS_ACT},
        {"role": "user",      "content": q},
        {"role": "assistant", "content": a},
    ]}

# ── Building blocks ───────────────────────────────────────────────

LOG_TYPES  = ["oak_log","birch_log","spruce_log","jungle_log","acacia_log","dark_oak_log"]
ORE_TYPES  = ["coal_ore","iron_ore","gold_ore","diamond_ore","deepslate_coal_ore","deepslate_iron_ore"]
STONE_TYPES= ["stone","cobblestone","granite","diorite","andesite","deepslate"]
FOOD_ITEMS = ["bread","apple","cooked_beef","cooked_porkchop","carrot","baked_potato"]
TOOLS      = ["wooden_pickaxe","stone_pickaxe","iron_pickaxe","wooden_axe","stone_axe","wooden_sword"]
DIRS       = ["north","south","east","west"]
BIOMES     = ["forest","plains","desert","taiga","jungle","savanna","mountains","swamp"]
WEATHERS   = ["clear","rain","thunder"]
TIMES      = ["day","noon","afternoon","dusk","night","midnight","dawn"]
HOSTILES   = ["zombie","skeleton","creeper","spider","enderman","witch","pillager"]

def rand_pos():
    x,y,z = random.randint(-500,500), random.randint(60,80), random.randint(-500,500)
    return f"x={x},y={y},z={z}"

def rand_hp():   return random.choice([4,5,6,8,10,12,14,16,18,20])
def rand_food(): return random.choice([2,4,6,8,10,12,14,16,18,20])
def rand_count(lo=1,hi=20): return random.randint(lo,hi)

def progress_bar(n, total):
    pct = min(100, int(n/total*100))
    filled = pct // 10
    bar = "█"*filled + "░"*(10-filled)
    return f"{n}/{total} [{bar}] {pct}%"

examples = []

# ═══════════════════════════════════════════════════════════════════
# CATEGORY 1: WOOD COLLECTION (200 examples)
# ═══════════════════════════════════════════════════════════════════

for _ in range(200):
    log   = random.choice(LOG_TYPES)
    total = random.choice([5,10,15,20,32])
    have  = random.randint(0, total-1)
    prog  = progress_bar(have, total)
    hp    = rand_hp()
    food  = rand_food()
    pos   = rand_pos()
    biome = random.choice(BIOMES)
    time_ = random.choice(TIMES)
    inv   = {log: have} if have > 0 else {}
    inv_str = json.dumps(inv)

    goal = f"collect {total} {log.replace('_',' ')}"

    nearby_has_log = random.random() > 0.45
    if nearby_has_log:
        dist = round(random.uniform(1.0, 12.0), 1)
        nearby = [log, random.choice(["grass_block","dirt","leaves"])]*random.randint(1,3)
        nearby = list(set(nearby))
        q = (f"Goal:{goal}. HP:{hp} Food:{food} Pos:{pos} "
             f"Inv:{inv_str} Nearby:{nearby} Biome:{biome} Time:{time_} "
             f"Progress:{prog}")
        a = json.dumps({"action":"MINE","params":{"block":log},
                        "reasoning":f"{log} found nearby at distance {dist}, mining it"})
    else:
        nearby = [random.choice(["grass_block","dirt","sand","gravel","flower","tall_grass"])
                  for _ in range(random.randint(2,5))]
        q = (f"Goal:{goal}. HP:{hp} Food:{food} Pos:{pos} "
             f"Inv:{inv_str} Nearby:{nearby} Biome:{biome} Time:{time_} "
             f"Progress:{prog}")
        a = json.dumps({"action":"SEEK","params":{"target":log},
                        "reasoning":f"no {log} in nearby blocks, searching the world"})

    examples.append(chat(q, a))

# ═══════════════════════════════════════════════════════════════════
# CATEGORY 2: STONE / ORE MINING (200 examples)
# ═══════════════════════════════════════════════════════════════════

for _ in range(200):
    stone = random.choice(STONE_TYPES + ORE_TYPES)
    total = random.choice([5,10,15,20])
    have  = random.randint(0, total-1)
    prog  = progress_bar(have, total)
    hp    = rand_hp()
    food  = rand_food()
    pos   = rand_pos()
    inv   = {"cobblestone": have, "wooden_pickaxe": 1}
    inv_str = json.dumps(inv)
    goal  = f"mine {total} {stone.replace('_',' ')}"

    has_stone = random.random() > 0.4
    if has_stone:
        nearby = [stone] + [random.choice(["cobblestone","dirt","gravel"]) for _ in range(2)]
        q = (f"Goal:{goal}. HP:{hp} Food:{food} Pos:{pos} "
             f"Inv:{inv_str} Nearby:{nearby} Progress:{prog}")
        a = json.dumps({"action":"MINE","params":{"block":stone},
                        "reasoning":f"{stone} visible nearby, mining now"})
    else:
        nearby = [random.choice(["grass_block","dirt","oak_log","sand"]) for _ in range(3)]
        q = (f"Goal:{goal}. HP:{hp} Food:{food} Pos:{pos} "
             f"Inv:{inv_str} Nearby:{nearby} Progress:{prog}")
        a = json.dumps({"action":"SEEK","params":{"target":stone},
                        "reasoning":f"no {stone} visible, need to find it"})

    examples.append(chat(q, a))

# ═══════════════════════════════════════════════════════════════════
# CATEGORY 3: CRAFTING (200 examples)
# ═══════════════════════════════════════════════════════════════════

CRAFT_RECIPES = [
    ("wooden_pickaxe",  {"oak_log":3},   {"oak_planks":3, "stick":2}),
    ("stone_pickaxe",   {"cobblestone":3, "oak_log":1}, {"cobblestone":3, "stick":2}),
    ("crafting_table",  {"oak_log":1},   {"oak_planks":4}),
    ("torch",           {"coal":1, "oak_log":1}, {"coal":1, "stick":1}),
    ("chest",           {"oak_log":2},   {"oak_planks":8}),
    ("wooden_sword",    {"oak_log":2},   {"oak_planks":2, "stick":1}),
    ("oak_planks",      {"oak_log":1},   {"oak_planks":4}),
    ("stick",           {"oak_log":1},   {"oak_planks":2}),
    ("bread",           {},              {"wheat":3}),
    ("iron_pickaxe",    {"iron_ingot":3, "stick":2}, {"iron_ingot":3, "stick":2}),
]

for _ in range(200):
    item, prereq, recipe = random.choice(CRAFT_RECIPES)
    hp   = rand_hp()
    food = rand_food()
    pos  = rand_pos()
    goal = f"craft a {item.replace('_',' ')}"

    has_materials = random.random() > 0.4
    has_table     = random.random() > 0.3
    nearby_table  = ["crafting_table"] if has_table else []
    nearby = nearby_table + [random.choice(["grass_block","oak_log","dirt"]) for _ in range(2)]

    if has_materials:
        inv = {**recipe, "wooden_pickaxe":1}
        inv_str = json.dumps(inv)
        q = f"Goal:{goal}. HP:{hp} Food:{food} Pos:{pos} Inv:{inv_str} Nearby:{nearby}"
        a = json.dumps({"action":"CRAFT","params":{"item":item,"count":1},
                        "reasoning":f"have all materials for {item}, crafting now"})
    else:
        missing = list(recipe.keys())[0] if recipe else "oak_log"
        inv = {"wooden_pickaxe":1}
        inv_str = json.dumps(inv)
        q = f"Goal:{goal}. HP:{hp} Food:{food} Pos:{pos} Inv:{inv_str} Nearby:{nearby}"
        if "log" in missing or "wood" in missing:
            a = json.dumps({"action":"SEEK","params":{"target":missing},
                            "reasoning":f"need {missing} to craft {item}"})
        else:
            a = json.dumps({"action":"MINE","params":{"block":missing},
                            "reasoning":f"missing {missing} required for {item}"})

    examples.append(chat(q, a))

# ═══════════════════════════════════════════════════════════════════
# CATEGORY 4: SURVIVAL (low HP / food) (200 examples)
# ═══════════════════════════════════════════════════════════════════

for _ in range(200):
    scenario = random.choice(["low_hp","low_food","both","hostile"])
    pos  = rand_pos()
    goal = random.choice(["survive","survive and explore","stay alive"])

    if scenario == "low_hp":
        hp, food = random.randint(2,7), random.randint(10,20)
        inv = {random.choice(FOOD_ITEMS): random.randint(1,3)}
        nearby = [random.choice(["grass_block","oak_log","dirt"]) for _ in range(3)]
        q = f"Goal:{goal}. HP:{hp} Food:{food} Pos:{pos} Inv:{json.dumps(inv)} Nearby:{nearby}"
        a = json.dumps({"action":"CHAT","params":{"message":f"Health critical at {hp}/20!"},
                        "reasoning":"HP dangerously low, alerting player"})

    elif scenario == "low_food":
        hp, food = random.randint(15,20), random.randint(1,5)
        inv = {}
        nearby = [random.choice(["grass_block","wheat","carrot","oak_log"]) for _ in range(4)]
        has_food = any(f in nearby for f in ["wheat","carrot","bread"])
        q = f"Goal:{goal}. HP:{hp} Food:{food} Pos:{pos} Inv:{json.dumps(inv)} Nearby:{nearby}"
        a = json.dumps({"action":"SEEK","params":{"target":"food"},
                        "reasoning":f"food level {food}/20, need to find food urgently"})

    elif scenario == "both":
        hp   = random.randint(3,6)
        food = random.randint(2,5)
        inv  = {}
        nearby = [random.choice(["grass_block","dirt"]) for _ in range(3)]
        q = f"Goal:{goal}. HP:{hp} Food:{food} Pos:{pos} Inv:{json.dumps(inv)} Nearby:{nearby}"
        a = json.dumps({"action":"CHAT","params":{"message":f"CRITICAL: HP={hp} Food={food}!"},
                        "reasoning":"both health and food critical"})

    else:  # hostile
        hp   = rand_hp()
        food = rand_food()
        mob  = random.choice(HOSTILES)
        dist = round(random.uniform(2,8), 1)
        inv  = {"wooden_sword":1}
        nearby = [mob, "grass_block","dirt"]
        dir_ = random.choice(DIRS)
        q = f"Goal:{goal}. HP:{hp} Food:{food} Pos:{pos} Inv:{json.dumps(inv)} Nearby:{nearby} HostileMob:{mob}@{dist}blocks"
        a = json.dumps({"action":"MOVE","params":{"direction":dir_,"distance":15},
                        "reasoning":f"fleeing {mob} at {dist} blocks"})

    examples.append(chat(q, a))

# ═══════════════════════════════════════════════════════════════════
# CATEGORY 5: EXPLORATION (150 examples)
# ═══════════════════════════════════════════════════════════════════

for _ in range(150):
    hp   = random.randint(15,20)
    food = random.randint(12,20)
    pos  = rand_pos()
    dir_ = random.choice(DIRS)
    dist = random.choice([10,15,20,25,30,50])
    biome= random.choice(BIOMES)
    time_= random.choice(TIMES)
    goal = random.choice(["explore","survive and explore","find a village","find diamonds",
                           "discover new biomes","map the area"])
    nearby = [random.choice(["grass_block","oak_log","dirt","sand","snow"]) for _ in range(4)]
    inv  = {"wooden_pickaxe":1, "bread": random.randint(0,5)}
    q = (f"Goal:{goal}. HP:{hp} Food:{food} Pos:{pos} Inv:{json.dumps(inv)} "
         f"Nearby:{nearby} Biome:{biome} Time:{time_}")
    a = json.dumps({"action":"MOVE","params":{"direction":dir_,"distance":dist},
                    "reasoning":f"exploring {biome} biome, moving {dir_}"})
    examples.append(chat(q, a))

# ═══════════════════════════════════════════════════════════════════
# CATEGORY 6: COAL / IRON / DIAMOND GATHERING (150 examples)
# ═══════════════════════════════════════════════════════════════════

ORE_GOALS = [
    ("coal", "coal_ore", "deepslate_coal_ore", 16),
    ("iron", "iron_ore", "deepslate_iron_ore",  8),
    ("gold", "gold_ore", "deepslate_gold_ore",  4),
    ("diamond", "diamond_ore", "deepslate_diamond_ore", 3),
]

for _ in range(150):
    name, ore, deep_ore, typical_count = random.choice(ORE_GOALS)
    total = random.choice([typical_count, typical_count*2])
    have  = random.randint(0, total-1)
    prog  = progress_bar(have, total)
    hp    = rand_hp()
    food  = rand_food()
    pos   = rand_pos()
    goal  = f"gather {total} {name}"
    inv   = {name: have, "stone_pickaxe":1}

    has_ore = random.random() > 0.45
    target_ore = random.choice([ore, deep_ore])
    if has_ore:
        nearby = [target_ore, "cobblestone","stone","gravel"]
        q = (f"Goal:{goal}. HP:{hp} Food:{food} Pos:{pos} "
             f"Inv:{json.dumps(inv)} Nearby:{nearby} Progress:{prog}")
        a = json.dumps({"action":"MINE","params":{"block":target_ore},
                        "reasoning":f"{target_ore} found, mining for {name}"})
    else:
        nearby = ["stone","cobblestone","dirt","granite"]
        q = (f"Goal:{goal}. HP:{hp} Food:{food} Pos:{pos} "
             f"Inv:{json.dumps(inv)} Nearby:{nearby} Progress:{prog}")
        a = json.dumps({"action":"SEEK","params":{"target":ore},
                        "reasoning":f"no {name} ore visible, searching deeper"})

    examples.append(chat(q, a))

# ═══════════════════════════════════════════════════════════════════
# CATEGORY 7: GOAL COMPLETE → IDLE (100 examples)
# ═══════════════════════════════════════════════════════════════════

for _ in range(100):
    log   = random.choice(LOG_TYPES)
    total = random.choice([5,10,15,20])
    have  = total  # complete
    prog  = progress_bar(have, total)
    hp    = random.randint(15,20)
    food  = random.randint(14,20)
    inv   = {log: have}
    q = (f"Goal:collect {total} {log.replace('_',' ')}. "
         f"HP:{hp} Food:{food} Inv:{json.dumps(inv)} Progress:{prog} GOAL COMPLETE")
    a = json.dumps({"action":"IDLE","params":{},
                    "reasoning":f"goal complete, collected {have}/{total} {log}"})
    examples.append(chat(q, a))

# ═══════════════════════════════════════════════════════════════════
# CATEGORY 8: NIGHT / TIME-BASED DECISIONS (100 examples)
# ═══════════════════════════════════════════════════════════════════

for _ in range(100):
    is_night = random.random() > 0.5
    time_    = random.choice(["night","midnight"]) if is_night else random.choice(["day","noon"])
    hp       = rand_hp()
    food     = rand_food()
    pos      = rand_pos()
    goal     = random.choice(["survive","explore","collect wood"])
    nearby   = [random.choice(["grass_block","oak_log","dirt","cobblestone"]) for _ in range(3)]
    inv      = {"wooden_sword":1, "torch": random.randint(0,8)}

    if is_night:
        dir_ = random.choice(DIRS)
        q = (f"Goal:{goal}. HP:{hp} Food:{food} Pos:{pos} Time:{time_} "
             f"Inv:{json.dumps(inv)} Nearby:{nearby}")
        a = json.dumps({"action":"SEEK","params":{"target":"shelter"},
                        "reasoning":"nighttime — hostile mobs spawn, seeking shelter"})
    else:
        q = (f"Goal:{goal}. HP:{hp} Food:{food} Pos:{pos} Time:{time_} "
             f"Inv:{json.dumps(inv)} Nearby:{nearby}")
        a = json.dumps({"action":"MOVE","params":{"direction":random.choice(DIRS),"distance":20},
                        "reasoning":"daytime — safe to explore and gather resources"})

    examples.append(chat(q, a))

# ═══════════════════════════════════════════════════════════════════
# CATEGORY 9: MULTI-STEP SEQUENCES (100 examples)
# ═══════════════════════════════════════════════════════════════════

SEQUENCES = [
    # (goal, inv, nearby, expected_action, reason)
    ("craft iron pickaxe", {"oak_log":0}, ["oak_log","dirt"],
     "MINE", {"block":"oak_log"}, "need oak_log to make planks for sticks"),
    ("craft iron pickaxe", {"oak_log":4,"oak_planks":0}, ["crafting_table","dirt"],
     "CRAFT", {"item":"oak_planks"}, "have logs, make planks first"),
    ("craft iron pickaxe", {"oak_planks":8,"stick":0}, ["crafting_table"],
     "CRAFT", {"item":"stick"}, "have planks, craft sticks next"),
    ("craft iron pickaxe", {"stick":4,"iron_ingot":0}, ["stone","iron_ore"],
     "MINE", {"block":"iron_ore"}, "have sticks, need iron for pickaxe"),
    ("craft iron pickaxe", {"stick":4,"iron_ingot":3}, ["crafting_table"],
     "CRAFT", {"item":"iron_pickaxe"}, "have all materials, crafting now"),
    ("build a shelter", {"oak_log":0}, ["oak_log","grass_block"],
     "MINE", {"block":"oak_log"}, "need wood for shelter walls"),
    ("build a shelter", {"oak_planks":20}, ["grass_block","dirt"],
     "GOTO", {"x":0,"y":65,"z":0}, "have materials, going to build location"),
    ("farm wheat", {"wheat_seeds":0}, ["grass_block","dirt"],
     "SEEK", {"target":"grass"}, "need to find grass to get wheat seeds"),
    ("brew a potion", {"glass_bottle":0}, ["sand","dirt"],
     "SEEK", {"target":"sand"}, "need sand to make glass for bottles"),
    ("enchant a sword", {"wooden_sword":1,"oak_log":0}, ["oak_log"],
     "MINE", {"block":"oak_log"}, "need wood for bookshelf construction"),
]

for _ in range(100):
    goal, inv, nearby, action, params, reason = random.choice(SEQUENCES)
    hp   = random.randint(14,20)
    food = random.randint(12,20)
    pos  = rand_pos()
    # Add some noise to inventory
    inv_noisy = {**inv, "stick": random.randint(0,4)}
    q = (f"Goal:{goal}. HP:{hp} Food:{food} Pos:{pos} "
         f"Inv:{json.dumps(inv_noisy)} Nearby:{nearby}")
    a = json.dumps({"action":action,"params":params,"reasoning":reason})
    examples.append(chat(q, a))

# ═══════════════════════════════════════════════════════════════════
# CATEGORY 10: EDGE CASES (100 examples)
# ═══════════════════════════════════════════════════════════════════

EDGE_CASES = [
    # Empty inventory, no goal
    ({"inventory":{},"nearby":["grass_block"],"goal":"","hp":20,"food":20},
     "IDLE", {}, "no goal set, waiting for player instructions"),
    # Has food, low on food bar
    ({"inventory":{"bread":3},"nearby":["grass_block"],"goal":"survive","hp":20,"food":3},
     "EAT", {"item":"bread"}, "food low at 3/20, eating bread from inventory"),
    # Zombie close, has sword
    ({"inventory":{"stone_sword":1},"nearby":["zombie","grass_block"],"goal":"survive","hp":18,"food":16},
     "MOVE", {"direction":"south","distance":10}, "zombie nearby, retreating to safety"),
    # Full inventory
    ({"inventory":{"oak_log":64,"cobblestone":64},"nearby":["oak_log"],"goal":"collect wood","hp":20,"food":20},
     "IDLE", {}, "inventory appears full, cannot collect more"),
    # Already at goal location
    ({"inventory":{"oak_log":10},"nearby":["crafting_table"],"goal":"craft wooden pickaxe","hp":20,"food":20},
     "CRAFT", {"item":"wooden_pickaxe"}, "crafting table nearby and have materials"),
    # Rain / thunder
    ({"inventory":{},"nearby":["grass_block","dirt"],"goal":"explore","hp":20,"food":20},
     "SEEK", {"target":"shelter"}, "thunder storm — dangerous to be outside"),
    # Creeper very close
    ({"inventory":{"wooden_sword":1},"nearby":["creeper","dirt"],"goal":"collect wood","hp":15,"food":18},
     "MOVE", {"direction":"north","distance":20}, "creeper detected very close, fleeing immediately"),
    # No nearby blocks
    ({"inventory":{},"nearby":[],"goal":"collect wood","hp":20,"food":20},
     "SEEK", {"target":"oak_log"}, "no blocks detected, scanning wider area"),
    # Just crafted, continuing goal
    ({"inventory":{"wooden_pickaxe":1,"oak_log":3},"nearby":["stone","cobblestone"],"goal":"mine 10 stone","hp":20,"food":20},
     "MINE", {"block":"stone"}, "have pickaxe, stone visible, mining now"),
    # Player following
    ({"inventory":{},"nearby":["grass_block"],"goal":"follow player","hp":20,"food":20},
     "FOLLOW", {"player":"aniket_0011"}, "goal is to follow player"),
]

for _ in range(100):
    case, action, params, reason = random.choice(EDGE_CASES)
    hp     = case.get("hp", rand_hp())
    food   = case.get("food", rand_food())
    inv    = case.get("inventory", {})
    nearby = case.get("nearby", [])
    goal   = case.get("goal", "survive")
    pos    = rand_pos()
    q = f"Goal:{goal}. HP:{hp} Food:{food} Pos:{pos} Inv:{json.dumps(inv)} Nearby:{nearby}"
    a = json.dumps({"action":action,"params":params,"reasoning":reason})
    examples.append(chat(q, a))

# ═══════════════════════════════════════════════════════════════════
# Finalize: shuffle, cap at 1600, save
# ═══════════════════════════════════════════════════════════════════

random.shuffle(examples)
examples = examples[:1600]

out = Path("training/synthetic_stage2.jsonl")
out.parent.mkdir(exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    for ex in examples:
        f.write(json.dumps(ex) + "\n")

print(f"Generated {len(examples)} unique examples -> {out}")

# Quick stats
from collections import Counter
actions = Counter(
    json.loads(ex["messages"][2]["content"])["action"]
    for ex in examples
)
print("\nAction distribution:")
for act, cnt in actions.most_common():
    bar = "#" * (cnt // 10)
    print(f"  {act:<12} {cnt:>4}  {bar}")
