"""
knowledge_seeder.py
Seeds ChromaDB with Minecraft domain knowledge:
  - Crafting recipes
  - Survival tips
  - Combat strategies
  - Biome info

Run once:  python knowledge_seeder.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import memory as mem

RECIPES = [
    ("craft oak_planks",       "To make oak planks: place 1 oak_log anywhere in crafting grid → 4 planks"),
    ("craft crafting_table",   "To make crafting_table: place 4 oak_planks in 2x2 grid in inventory → 1 crafting_table"),
    ("craft wooden_pickaxe",   "To make wooden_pickaxe: 3 planks on top row + 2 sticks in center column on crafting table"),
    ("craft wooden_axe",       "To make wooden_axe: 2 planks top-left+middle-left, 1 plank top-center, 2 sticks center+bottom-center"),
    ("craft wooden_sword",     "To make wooden_sword: 2 planks stacked in center column + 1 stick below on crafting table"),
    ("craft stick",            "To make sticks: 2 planks stacked vertically → 4 sticks"),
    ("craft stone_pickaxe",    "To make stone_pickaxe: 3 cobblestone on top row + 2 sticks center column on crafting table"),
    ("craft iron_pickaxe",     "To make iron_pickaxe: 3 iron_ingot on top row + 2 sticks below on crafting table. Need furnace to smelt iron_ore first"),
    ("craft furnace",          "To make furnace: 8 cobblestone in outer ring of 3x3 crafting grid (leave center empty)"),
    ("craft torch",            "To make torch: 1 coal or charcoal + 1 stick below → 4 torches"),
    ("craft chest",            "To make chest: 8 planks in outer ring of 3x3 crafting grid (leave center empty)"),
    ("craft bread",            "To make bread: 3 wheat in a horizontal row on crafting table"),
    ("smelt iron_ingot",       "To smelt iron: put iron_ore in top slot of furnace + any fuel (wood/coal) in bottom → iron_ingot"),
    ("smelt charcoal",         "To make charcoal: smelt any log in furnace with wood fuel → charcoal (alternative to coal)"),
]

SURVIVAL_TIPS = [
    ("low health", "When health is below 8, eat food immediately. Cooked meat restores most hunger. Bread restores 5 hunger points."),
    ("first night", "On your first day, gather wood → make crafting table → make wooden pickaxe → mine stone → make shelter before night"),
    ("hunger system", "When food bar is below 6, you cannot sprint and will not regenerate health. Eat before it gets critical."),
    ("night time", "At night (timeOfDay > 13000), hostile mobs spawn. Stay inside shelter or light up the area with torches."),
    ("zombie behavior", "Zombies are slow but numerous. Run away if outnumbered. They burn in sunlight (time < 12000)."),
    ("creeper danger", "Creepers are silent and explode. If you see one within 5 blocks, run away immediately before it explodes."),
    ("skeleton danger", "Skeletons shoot arrows from range. Use trees as cover and approach diagonally to melee attack."),
    ("wood gathering", "Oak trees drop oak_log. Punch trunk from bottom. Leaves sometimes drop saplings and apples."),
    ("stone mining", "Stone is found at any depth underground or in exposed cliff faces. Mine with wooden_pickaxe or better."),
    ("coal usage", "Coal is found embedded in stone (coal_ore). Use it as fuel in furnace or to make torches."),
    ("iron ore location", "Iron_ore generates at y=15 to y=60. Mine with stone_pickaxe or better. Smelt to get iron_ingot."),
    ("shelter basics", "A basic shelter needs 4 walls + roof + door. Can be made entirely of wooden planks or dirt in emergency."),
    ("sleep at night", "Sleep in a bed to skip the night. Craft bed from 3 wool + 3 planks. Wool comes from sheep."),
    ("crafting table needed", "A crafting table is required for most recipes. Always carry one or place one near your base."),
    ("inventory full", "When inventory is full, drop less valuable items like dirt or cobblestone to make room for important items."),
]

COMBAT_TIPS = [
    ("sword vs mobs", "Use a sword for combat. Attack with full charge (wait for attack cooldown to fill) for maximum damage."),
    ("shield blocking", "Hold shield to block attacks. Right-click to raise shield. Reduces damage from projectiles and melee."),
    ("hit and run", "Against multiple enemies, hit one then back away to reset combat. Never let them surround you."),
    ("torch placement", "Place torches on walls and floors to prevent mob spawning. Light level must be above 7."),
]

BIOME_TIPS = [
    ("plains biome", "Plains biome has flat terrain with grass, flowers, occasional oak trees, and villages. Good for first base."),
    ("forest biome", "Forest biome has dense trees (oak, birch). Great for wood early game. Watch for mob ambushes at night."),
    ("desert biome", "Desert has sand, sandstone, cacti. Few trees. Villages common. No rain. Hot during day."),
    ("taiga biome", "Taiga has spruce trees and ferns. Good spruce_log source. Cold biome, wolves spawn here."),
]


def seed_all():
    print("Seeding ChromaDB with Minecraft knowledge...")

    if not mem.is_available():
        print("ERROR: ChromaDB not available. Run: pip install chromadb")
        return False

    count = 0
    categories = [
        ("recipe",   RECIPES),
        ("survival", SURVIVAL_TIPS),
        ("combat",   COMBAT_TIPS),
        ("biome",    BIOME_TIPS),
    ]

    col = mem._init()
    if col is None:
        print("ERROR: Could not initialise ChromaDB collection.")
        return False

    for category, entries in categories:
        for i, (topic, text) in enumerate(entries):
            doc_id = f"seed_{category}_{i:03d}"

            # Skip if already exists
            try:
                existing = col.get(ids=[doc_id])
                if existing["ids"]:
                    continue
            except Exception:
                pass

            try:
                col.add(
                    documents=[f"[{category.upper()}] {topic}: {text}"],
                    metadatas=[{"category": category, "topic": topic, "source": "seed"}],
                    ids=[doc_id],
                )
                count += 1
            except Exception as e:
                print(f"  Warning: could not add {doc_id}: {e}")

    total = mem.count()
    print(f"Done! Added {count} entries. Total memories: {total}")
    return True


if __name__ == "__main__":
    seed_all()
