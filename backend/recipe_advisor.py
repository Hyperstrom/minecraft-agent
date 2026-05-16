"""
recipe_advisor.py
Computes what the bot CAN craft from its current inventory.
Injected into the prompt so Mistral makes inventory-aware decisions.
"""

from typing import Dict, List

# ── Recipe definitions ────────────────────────────────────────────
# Each entry: (output_item, {ingredient: count_needed, ...}, needs_crafting_table)

RECIPES: List[tuple] = [
    # No crafting table needed (2x2 inventory grid)
    ("oak_planks",       {"oak_log": 1},                              False),
    ("birch_planks",     {"birch_log": 1},                            False),
    ("spruce_planks",    {"spruce_log": 1},                           False),
    ("crafting_table",   {"oak_planks": 4},                           False),
    ("stick",            {"oak_planks": 2},                           False),
    ("torch",            {"stick": 1, "coal": 1},                     False),
    ("torch_charcoal",   {"stick": 1, "charcoal": 1},                 False),

    # Crafting table needed (3x3 grid)
    ("wooden_pickaxe",   {"oak_planks": 3, "stick": 2},               True),
    ("wooden_axe",       {"oak_planks": 3, "stick": 2},               True),
    ("wooden_sword",     {"oak_planks": 2, "stick": 1},               True),
    ("wooden_shovel",    {"oak_planks": 1, "stick": 2},               True),
    ("stone_pickaxe",    {"cobblestone": 3, "stick": 2},              True),
    ("stone_axe",        {"cobblestone": 3, "stick": 2},              True),
    ("stone_sword",      {"cobblestone": 2, "stick": 1},              True),
    ("iron_pickaxe",     {"iron_ingot": 3, "stick": 2},               True),
    ("iron_sword",       {"iron_ingot": 2, "stick": 1},               True),
    ("iron_axe",         {"iron_ingot": 3, "stick": 2},               True),
    ("furnace",          {"cobblestone": 8},                          True),
    ("chest",            {"oak_planks": 8},                           True),
    ("bread",            {"wheat": 3},                                True),
    ("bowl",             {"oak_planks": 3},                           True),
    ("ladder",           {"stick": 7},                                True),
    ("oak_door",         {"oak_planks": 6},                           True),
    ("fence",            {"oak_planks": 4, "stick": 2},               True),
    ("oak_stairs",       {"oak_planks": 6},                           True),
    ("oak_slab",         {"oak_planks": 3},                           True),
    ("bed",              {"oak_planks": 3, "white_wool": 3},          True),
    ("bow",              {"stick": 3, "string": 3},                   True),
    ("arrow",            {"flint": 1, "stick": 1, "feather": 1},      True),
    ("bucket",           {"iron_ingot": 3},                           True),
    ("compass",          {"iron_ingot": 4, "redstone": 1},            True),
]

WOOD_TYPES = ["oak_log","birch_log","spruce_log","jungle_log","dark_oak_log",
              "oak_planks","birch_planks","spruce_planks","jungle_planks"]

def get_craftable(inventory: Dict[str, int]) -> List[str]:
    """Return list of items the bot can craft from its current inventory."""
    inv   = {k.lower(): v for k, v in inventory.items()}
    has_table = inv.get("crafting_table", 0) > 0

    craftable = []
    for (item, ingredients, needs_table) in RECIPES:
        if needs_table and not has_table:
            continue
        if all(inv.get(ing, 0) >= qty for ing, qty in ingredients.items()):
            craftable.append(item)

    return craftable


def get_craftable_hint(inventory: Dict[str, int]) -> str:
    """Return a short text hint about what can be crafted — for the LLM prompt."""
    craftable = get_craftable(inventory)
    if not craftable:
        return "Craftable: nothing yet (need more materials)"
    if len(craftable) > 8:
        craftable = craftable[:8]  # keep prompt short
    return f"Craftable right now: {', '.join(craftable)}"


def has_wood(inventory: Dict[str, int]) -> bool:
    inv = {k.lower(): v for k, v in inventory.items()}
    return any(inv.get(w, 0) > 0 for w in WOOD_TYPES)
