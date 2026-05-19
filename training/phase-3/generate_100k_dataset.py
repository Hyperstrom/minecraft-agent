"""
generate_100k_dataset.py
========================
Generates 100,000 pure JSON state-action pairs for Minecraft LLM training.
Run this LOCALLY (no GPU needed). Output: data/stage1_100k.jsonl

Usage:
    python generate_100k_dataset.py --output ./data/stage1_100k.jsonl --count 100000
"""

import json
import random
import argparse
import math
from pathlib import Path
from itertools import product

random.seed(42)

# ── Minecraft Data Registry ────────────────────────────────────────────────────

CRAFTING_RECIPES = {
    "wooden_planks":     {"ingredients": {"oak_log": 1},         "count": 4},
    "stick":             {"ingredients": {"wooden_planks": 2},    "count": 4},
    "crafting_table":    {"ingredients": {"wooden_planks": 4},    "count": 1},
    "wooden_pickaxe":    {"ingredients": {"wooden_planks": 3, "stick": 2}, "count": 1},
    "wooden_axe":        {"ingredients": {"wooden_planks": 3, "stick": 2}, "count": 1},
    "wooden_shovel":     {"ingredients": {"wooden_planks": 1, "stick": 2}, "count": 1},
    "wooden_sword":      {"ingredients": {"wooden_planks": 2, "stick": 1}, "count": 1},
    "stone_pickaxe":     {"ingredients": {"cobblestone": 3, "stick": 2},   "count": 1},
    "stone_axe":         {"ingredients": {"cobblestone": 3, "stick": 2},   "count": 1},
    "stone_shovel":      {"ingredients": {"cobblestone": 1, "stick": 2},   "count": 1},
    "stone_sword":       {"ingredients": {"cobblestone": 2, "stick": 1},   "count": 1},
    "iron_pickaxe":      {"ingredients": {"iron_ingot": 3, "stick": 2},    "count": 1},
    "iron_axe":          {"ingredients": {"iron_ingot": 3, "stick": 2},    "count": 1},
    "iron_sword":        {"ingredients": {"iron_ingot": 2, "stick": 1},    "count": 1},
    "iron_shovel":       {"ingredients": {"iron_ingot": 1, "stick": 2},    "count": 1},
    "iron_helmet":       {"ingredients": {"iron_ingot": 5},                "count": 1},
    "iron_chestplate":   {"ingredients": {"iron_ingot": 8},                "count": 1},
    "iron_leggings":     {"ingredients": {"iron_ingot": 7},                "count": 1},
    "iron_boots":        {"ingredients": {"iron_ingot": 4},                "count": 1},
    "furnace":           {"ingredients": {"cobblestone": 8},               "count": 1},
    "chest":             {"ingredients": {"wooden_planks": 8},             "count": 1},
    "torch":             {"ingredients": {"coal": 1, "stick": 1},          "count": 4},
    "bread":             {"ingredients": {"wheat": 3},                     "count": 1},
    "bowl":              {"ingredients": {"wooden_planks": 3},             "count": 4},
    "ladder":            {"ingredients": {"stick": 7},                     "count": 3},
    "iron_ingot":        {"ingredients": {"iron_ore": 1},                  "count": 1, "smelt": True},
    "cooked_beef":       {"ingredients": {"beef": 1},                      "count": 1, "smelt": True},
    "charcoal":          {"ingredients": {"oak_log": 1},                   "count": 1, "smelt": True},
}

BLOCKS = [
    "oak_log", "birch_log", "spruce_log", "cobblestone", "stone",
    "dirt", "grass_block", "sand", "gravel", "coal_ore", "iron_ore",
    "gold_ore", "diamond_ore", "crafting_table", "furnace", "chest",
    "water", "lava", "bedrock", "obsidian", "netherrack",
]

MOBS = [
    {"name": "zombie",         "hostile": True,  "hp": 20, "damage": 3},
    {"name": "skeleton",       "hostile": True,  "hp": 20, "damage": 4},
    {"name": "creeper",        "hostile": True,  "hp": 20, "damage": 0, "explodes": True},
    {"name": "spider",         "hostile": True,  "hp": 16, "damage": 2},
    {"name": "enderman",       "hostile": False, "hp": 40, "damage": 7},
    {"name": "witch",          "hostile": True,  "hp": 26, "damage": 6},
    {"name": "cow",            "hostile": False, "hp": 10, "damage": 0},
    {"name": "pig",            "hostile": False, "hp": 10, "damage": 0},
    {"name": "sheep",          "hostile": False, "hp": 8,  "damage": 0},
    {"name": "chicken",        "hostile": False, "hp": 4,  "damage": 0},
]

TOOLS = ["wooden_pickaxe", "stone_pickaxe", "iron_pickaxe", "diamond_pickaxe",
         "wooden_axe", "stone_axe", "iron_axe",
         "wooden_sword", "stone_sword", "iron_sword", "diamond_sword",
         "bow", "shield"]

# ── Helper Functions ───────────────────────────────────────────────────────────

def rand_inv(min_items=1, max_items=8):
    items = random.sample(list(CRAFTING_RECIPES.keys()) + BLOCKS, k=random.randint(min_items, max_items))
    return {item: random.randint(1, 64) for item in items}

def rand_pos():
    return {"x": random.randint(-500, 500), "y": random.randint(60, 80), "z": random.randint(-500, 500)}

def rand_nearby_blocks(count=3):
    blocks = random.sample(BLOCKS, k=count)
    return [{"name": b, "dist": round(random.uniform(1.0, 8.0), 1)} for b in blocks]

def rand_nearby_mobs(count=None):
    if count is None:
        count = random.randint(0, 3)
    selected = random.sample(MOBS, k=min(count, len(MOBS)))
    return [{"name": m["name"], "dist": round(random.uniform(2.0, 16.0), 1),
             "hp": random.randint(1, m["hp"]), "hostile": m["hostile"]} for m in selected]

def make_state(hp=None, food=None, inv=None, nearby_blocks=None, nearby_mobs=None,
               goal=None, pos=None, time_of_day=None, is_raining=None):
    return {
        "hp":         hp          or random.randint(1, 20),
        "food":       food        or random.randint(0, 20),
        "pos":        pos         or rand_pos(),
        "inv":        inv         or rand_inv(),
        "nearby":     nearby_blocks if nearby_blocks is not None else rand_nearby_blocks(),
        "mobs":       nearby_mobs  if nearby_mobs is not None else rand_nearby_mobs(0),
        "time":       time_of_day or random.randint(0, 24000),
        "raining":    is_raining  if is_raining is not None else random.choice([True, False]),
        "goal":       goal        or "survive",
    }

def make_pair(state, action):
    return {
        "messages": [
            {"role": "user",      "content": json.dumps(state,  separators=(',', ':'))},
            {"role": "assistant", "content": json.dumps(action, separators=(',', ':'))},
        ]
    }

# ── Generator Functions (each produces one category of examples) ───────────────

def gen_crafting_examples(n):
    """Pure crafting logic: given inventory + goal, output CRAFT action."""
    examples = []
    recipe_list = [(name, data) for name, data in CRAFTING_RECIPES.items() if not data.get("smelt")]
    for _ in range(n):
        item_name, recipe = random.choice(recipe_list)
        # Build inventory that has exactly enough ingredients
        inv = dict(recipe["ingredients"])
        # Add some random extra items
        extras = random.sample(BLOCKS, k=random.randint(0, 5))
        for e in extras:
            inv[e] = random.randint(1, 32)

        nearby = [{"name": "crafting_table", "dist": round(random.uniform(0.5, 2.0), 1)}]
        state = make_state(inv=inv, nearby_blocks=nearby, goal=f"craft_{item_name}")
        action = {"action": "CRAFT", "target": item_name, "quantity": 1}
        examples.append(make_pair(state, action))
    return examples


def gen_smelting_examples(n):
    """Furnace smelting: given ore/food + fuel, output SMELT action."""
    examples = []
    smeltable = [(name, data) for name, data in CRAFTING_RECIPES.items() if data.get("smelt")]
    fuels = ["coal", "charcoal", "oak_log", "wooden_planks"]
    for _ in range(n):
        item_name, recipe = random.choice(smeltable)
        raw = list(recipe["ingredients"].keys())[0]
        fuel = random.choice(fuels)
        inv = {raw: random.randint(1, 32), fuel: random.randint(1, 16)}
        nearby = [{"name": "furnace", "dist": round(random.uniform(0.5, 2.0), 1)}]
        state = make_state(inv=inv, nearby_blocks=nearby, goal=f"smelt_{item_name}")
        action = {"action": "SMELT", "target": item_name, "fuel": fuel, "quantity": inv[raw]}
        examples.append(make_pair(state, action))
    return examples


def gen_navigation_examples(n):
    """A* navigation: given current pos and target, output MOVE action."""
    examples = []
    for _ in range(n):
        pos = rand_pos()
        target = rand_pos()
        dx = target["x"] - pos["x"]
        dz = target["z"] - pos["z"]
        dist = math.sqrt(dx*dx + dz*dz)
        # Determine direction
        if abs(dx) > abs(dz):
            direction = "east" if dx > 0 else "west"
        else:
            direction = "south" if dz > 0 else "north"

        state = make_state(pos=pos, goal=f"navigate_to_{target['x']}_{target['z']}")
        action = {
            "action": "MOVE",
            "direction": direction,
            "target_pos": target,
            "distance": round(dist, 1),
        }
        examples.append(make_pair(state, action))
    return examples


def gen_mining_examples(n):
    """Mining: given nearby blocks and tool, output MINE action."""
    examples = []
    tool_map = {
        "oak_log":      ["wooden_axe", "stone_axe", "iron_axe"],
        "birch_log":    ["wooden_axe", "stone_axe", "iron_axe"],
        "cobblestone":  ["wooden_pickaxe", "stone_pickaxe", "iron_pickaxe"],
        "stone":        ["wooden_pickaxe", "stone_pickaxe", "iron_pickaxe"],
        "coal_ore":     ["wooden_pickaxe", "stone_pickaxe", "iron_pickaxe"],
        "iron_ore":     ["stone_pickaxe", "iron_pickaxe"],
        "gold_ore":     ["iron_pickaxe"],
        "diamond_ore":  ["iron_pickaxe"],
        "dirt":         ["wooden_shovel", "stone_shovel", "iron_shovel"],
        "sand":         ["wooden_shovel", "stone_shovel", "iron_shovel"],
        "gravel":       ["wooden_shovel", "stone_shovel", "iron_shovel"],
    }
    mineable = list(tool_map.keys())
    for _ in range(n):
        target_block = random.choice(mineable)
        valid_tools = tool_map[target_block]
        tool = random.choice(valid_tools)
        inv = {tool: 1}
        inv.update({b: random.randint(1, 32) for b in random.sample(BLOCKS, 3)})
        nearby = [{"name": target_block, "dist": round(random.uniform(1.0, 3.5), 1)}]
        state = make_state(inv=inv, nearby_blocks=nearby, goal=f"mine_{target_block}")
        action = {"action": "MINE", "target": target_block, "tool": tool}
        examples.append(make_pair(state, action))
    return examples


def gen_combat_examples(n):
    """Combat AI: hostile mob nearby, decide to attack, retreat, or use ranged."""
    examples = []
    for _ in range(n):
        mob = random.choice([m for m in MOBS if m["hostile"]])
        dist = round(random.uniform(1.5, 20.0), 1)
        hp = random.randint(1, 20)
        food = random.randint(0, 20)
        has_sword = random.choice([True, False])
        has_bow = random.choice([True, False])
        inv = {}
        if has_sword:
            inv["iron_sword"] = 1
        if has_bow:
            inv["bow"] = 1
            inv["arrow"] = random.randint(1, 32)

        mob_hp = random.randint(1, mob["hp"])
        mobs = [{"name": mob["name"], "dist": dist, "hp": mob_hp, "hostile": True}]
        state = make_state(hp=hp, food=food, inv=inv, nearby_mobs=mobs,
                           goal=f"defeat_{mob['name']}")

        # Decision logic
        if mob["name"] == "creeper" and dist < 4.0:
            action = {"action": "MOVE", "direction": "away", "reason": "creeper_explosion_range"}
        elif hp < 6 and dist > 5.0:
            action = {"action": "MOVE", "direction": "away", "reason": "low_hp_retreat"}
        elif has_bow and dist > 6.0:
            action = {"action": "SHOOT", "target": mob["name"], "weapon": "bow"}
        elif has_sword and dist <= 3.0:
            action = {"action": "ATTACK", "target": mob["name"], "weapon": "iron_sword"}
        elif dist > 3.0:
            action = {"action": "MOVE", "direction": "toward", "target": mob["name"]}
        else:
            action = {"action": "ATTACK", "target": mob["name"], "weapon": "fist"}
        examples.append(make_pair(state, action))
    return examples


def gen_survival_examples(n):
    """Survival decisions: hunger, night time, shelter, health."""
    examples = []
    for _ in range(n):
        hp = random.randint(1, 20)
        food = random.randint(0, 20)
        time = random.randint(0, 24000)
        is_night = time > 13000
        inv = rand_inv()

        if food <= 6 and any(f in inv for f in ["bread", "cooked_beef", "apple", "cooked_chicken"]):
            food_item = next(f for f in ["bread", "cooked_beef", "apple"] if f in inv)
            action = {"action": "EAT", "target": food_item, "reason": "hunger_critical"}
        elif is_night and not any(b.get("name") == "chest" for b in rand_nearby_blocks()):
            action = {"action": "PLACE", "target": "torch", "reason": "night_lighting"}
        elif hp < 10 and food >= 18:
            action = {"action": "WAIT", "reason": "regenerating_health", "duration_ticks": 40}
        else:
            goal = random.choice(["gather_wood", "find_shelter", "explore", "mine_stone"])
            action = {"action": "MOVE", "direction": random.choice(["north","south","east","west"]),
                      "reason": goal}

        state = make_state(hp=hp, food=food, time_of_day=time, inv=inv,
                           goal="survive_night" if is_night else "survive")
        examples.append(make_pair(state, action))
    return examples


def gen_inventory_management_examples(n):
    """Inventory decisions: drop, organize, prioritize items."""
    examples = []
    for _ in range(n):
        # Overfull inventory scenario
        all_items = BLOCKS + list(CRAFTING_RECIPES.keys())
        inv = {item: random.randint(1, 64) for item in random.sample(all_items, 36)}  # over 36 = full

        # Decision: drop the least valuable item
        valuable = ["diamond", "iron_ingot", "gold_ingot", "iron_pickaxe", "diamond_pickaxe",
                    "iron_sword", "diamond_sword", "bow", "cooked_beef"]
        junk = [k for k in inv.keys() if k not in valuable and k not in ["oak_log", "cobblestone"]]
        drop_item = random.choice(junk) if junk else random.choice(list(inv.keys()))

        state = make_state(inv=inv, goal="manage_inventory")
        action = {"action": "DROP", "target": drop_item, "quantity": inv[drop_item],
                  "reason": "inventory_full"}
        examples.append(make_pair(state, action))
    return examples


def gen_block_placement_examples(n):
    """Placement: build shelter, place torches, create paths."""
    examples = []
    placements = [
        ("torch",        "place_torch",   "lighting"),
        ("oak_log",      "build_wall",    "shelter"),
        ("cobblestone",  "build_floor",   "building"),
        ("crafting_table","place_crafting_table", "crafting_setup"),
        ("furnace",      "place_furnace", "smelting_setup"),
        ("chest",        "place_chest",   "storage"),
        ("ladder",       "place_ladder",  "climbing"),
    ]
    for _ in range(n):
        item, goal_name, reason = random.choice(placements)
        inv = {item: random.randint(1, 16)}
        inv.update(rand_inv(0, 4))
        pos = rand_pos()
        state = make_state(inv=inv, pos=pos, goal=goal_name)
        action = {"action": "PLACE", "target": item,
                  "pos": {"x": pos["x"], "y": pos["y"], "z": pos["z"] + 1},
                  "reason": reason}
        examples.append(make_pair(state, action))
    return examples


def gen_goal_planning_examples(n):
    """Multi-step planning: given a high-level goal, output next immediate action."""
    examples = []
    goal_trees = [
        {
            "goal": "craft_iron_pickaxe",
            "steps": [
                ({"oak_log": 0},    {"action": "MINE", "target": "oak_log"}),
                ({"oak_log": 3},    {"action": "CRAFT", "target": "wooden_planks"}),
                ({"wooden_planks": 4}, {"action": "CRAFT", "target": "crafting_table"}),
                ({"crafting_table": 1, "wooden_planks": 3, "stick": 0},
                 {"action": "CRAFT", "target": "wooden_pickaxe"}),
                ({"wooden_pickaxe": 1, "cobblestone": 0},
                 {"action": "MINE", "target": "stone"}),
                ({"cobblestone": 8, "iron_ore": 0},
                 {"action": "MINE", "target": "iron_ore"}),
                ({"iron_ore": 3, "furnace": 0},
                 {"action": "CRAFT", "target": "furnace"}),
                ({"iron_ore": 3, "furnace": 1, "coal": 1, "iron_ingot": 0},
                 {"action": "SMELT", "target": "iron_ingot", "fuel": "coal", "quantity": 3}),
                ({"iron_ingot": 3, "stick": 2},
                 {"action": "CRAFT", "target": "iron_pickaxe"}),
            ]
        },
    ]
    for _ in range(n):
        tree = random.choice(goal_trees)
        step_idx = random.randint(0, len(tree["steps"]) - 1)
        inv_base, action = tree["steps"][step_idx]
        # Build inv from step requirements
        inv = dict(inv_base)
        inv.update({b: random.randint(1, 10) for b in random.sample(BLOCKS, 3)})
        state = make_state(inv=inv, goal=tree["goal"])
        examples.append(make_pair(state, action))
    return examples


# ── Main Generator ─────────────────────────────────────────────────────────────

def generate_dataset(total=100000, output_path="./data/stage1_100k.jsonl"):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Distribution of example types
    distribution = {
        "crafting":    int(total * 0.18),   # 18k
        "smelting":    int(total * 0.07),   # 7k
        "navigation":  int(total * 0.20),   # 20k
        "mining":      int(total * 0.20),   # 20k
        "combat":      int(total * 0.15),   # 15k
        "survival":    int(total * 0.08),   # 8k
        "inventory":   int(total * 0.05),   # 5k
        "placement":   int(total * 0.04),   # 4k
        "planning":    int(total * 0.03),   # 3k
    }

    generators = {
        "crafting":   gen_crafting_examples,
        "smelting":   gen_smelting_examples,
        "navigation": gen_navigation_examples,
        "mining":     gen_mining_examples,
        "combat":     gen_combat_examples,
        "survival":   gen_survival_examples,
        "inventory":  gen_inventory_management_examples,
        "placement":  gen_block_placement_examples,
        "planning":   gen_goal_planning_examples,
    }

    all_examples = []
    for category, count in distribution.items():
        print(f"  Generating {count:>6,} {category} examples...")
        examples = generators[category](count)
        all_examples.extend(examples)

    # Shuffle to avoid category clustering
    random.shuffle(all_examples)

    # Write JSONL
    with open(output_path, "w") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"\n✅ Dataset generated: {len(all_examples):,} examples → {output_path}")
    print(f"   File size: {Path(output_path).stat().st_size / 1024 / 1024:.1f} MB")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="./data/stage1_100k.jsonl")
    parser.add_argument("--count",  type=int, default=100000)
    args = parser.parse_args()

    print(f"🎮 Generating {args.count:,} Minecraft state-action pairs...")
    generate_dataset(args.count, args.output)
