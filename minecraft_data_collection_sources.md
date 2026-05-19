# Minecraft Data Collection Strategy

To achieve `< 0.001` loss and create a pure Minecraft gameplay LLM, we need at least **100,000 highly structured data points**. Because we need strict JSON formatting (not conversational English), we cannot just scrape text. We must source and format data carefully.

Here is the complete guide on where to get this data and how to collect it.

---

## 1. Programmatic Generation (The "Self-Play" Engine)
**The absolute best source of data for a deterministic AI is *another* deterministic AI.** 
We can write a script using your existing `mineflayer` Node.js bot to generate tens of thousands of perfect gameplay scenarios overnight.

### How it works:
1.  **Set up a local server**: Run a local Minecraft server in creative mode.
2.  **Use `mineflayer-pathfinder`**: Write a script where the bot is given a task (e.g., "Walk to X, Y, Z" or "Mine the nearest Oak Log").
3.  **Log every tick**: As the bot perfectly executes the A* pathfinding algorithm or perfectly crafts an item, a logger script writes the exact state and action to a `synthetic_gameplay.jsonl` file.

**Expected Yield**: `50,000+` perfect JSON pairs per hour of simulation.
**Data Type**: Flawless navigation, combat distancing, and inventory management.

---

## 2. Existing Open-Source Datasets
If you don't want to generate everything yourself, you can download massive datasets created by researchers. However, these will need to be **reformatted** into our strict JSON schema.

### A. The MineRL Dataset (Human Gameplay)
*   **What it is**: Developed by OpenAI and Microsoft, MineRL contains over **60 million frames** of human gameplay, mapping exact player inventory/health states to the exact keyboard/mouse actions they took.
*   **Where to get it**: [MineRL GitHub / Documentation](https://minerl.io/dataset/)
*   **How to use it**: We can write a Python script to parse the `minerl` data format and extract the `observation` (HP, inventory, nearby blocks) and `action` (forward, attack, craft) into our JSON prompt structure.

### B. Hugging Face Datasets (Domain Knowledge)
While we want to focus on gameplay, the model still needs to memorize crafting recipes and block properties.
*   **`naklecha/minecraft-question-answer-700k`**: (700,000+ rows) Excellent for broad Minecraft facts.
*   **`minhaozhang/minecraft-question-answer-630k`**: (630,000+ rows) Good for game mechanics.
*   **How to use them**: Instead of conversational Q&A, we rewrite them programmatically. E.g., change "How do I make a stick?" -> `{"goal": "craft_stick", "inv": {"planks": 2}}` -> `{"action": "CRAFT", "target": "stick"}`.

---

## 3. Minecraft Wiki & Recipe Extraction (Knowledge Base)
To ensure the model never hallucinates a recipe or item name, we must feed it the entire Minecraft source code registry.

### A. Minecraft Data Registries
*   **What it is**: Repositories that dump the raw JSON registries of Minecraft (every block ID, every crafting recipe).
*   **Where to get it**: [PrismarineJS/minecraft-data](https://github.com/PrismarineJS/minecraft-data)
*   **How to use it**: We iterate through every crafting recipe in `recipes.json`. For every recipe, we generate 5 synthetic prompts representing a player trying to craft that item. 
    *   *Yield*: ~10,000 perfect crafting state-action pairs.

### B. The Official Minecraft Wiki API
*   **What it is**: The most accurate text source for Minecraft block mechanics.
*   **Where to get it**: `https://minecraft.wiki/api.php`
*   **How to use it**: Scrape properties of items (e.g., "Diamond pickaxe durability = 1561").

---

## The Master Plan: Creating the 100k Dataset

If we proceed with Phase 3, here is how we combine these sources:

| Source | Strategy | Expected Yield |
| :--- | :--- | :--- |
| **Mineflayer Self-Play** | Automated bots walking, attacking, and mining on a local server. | 60,000 pairs |
| **PrismarineJS Recipes** | Converting raw `recipes.json` into state-action JSON pairs. | 15,000 pairs |
| **MineRL Subset** | Extracting survival/combat decisions from human logs. | 15,000 pairs |
| **Wiki / HF Datasets** | Distilled mechanical facts. | 10,000 pairs |
| **TOTAL** | | **100,000+ Pairs** |

### Next Step
To begin, the easiest and highest quality method is **Mineflayer Self-Play**. Since you already have a Node.js bot (`bot/actions.js`), we can write a script named `generate_100k_data.js` that spawns 10 bots on a server, gives them random tasks, and silently logs their perfect JSON decisions until we hit our target.
