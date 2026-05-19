# MineAgent — Complete Roadmap: LLM Training → RL in Minecraft

## Where You Are Now vs Where You're Going

| Phase | Status | What It Does |
|---|---|---|
| Phase 1 (Stage 1) | ✅ Done | Llama 3B learns Minecraft knowledge (12k examples) |
| Phase 2 (Stage 2) | ✅ Done | Behavior cloning on 1.5k session examples |
| **Phase 3** | 🔨 Build now | 100k pure JSON dataset + Qwen-1.5B, loss < 0.05 |
| **Phase 4** | 🔨 After Phase 3 | PPO reinforcement learning in live Minecraft |

---

## Why Phase 3 Before RL?

RL requires a **working base policy** to explore from. Your current Llama-3B model
still outputs conversational text sometimes. You cannot run RL on a model that says
"I will now mine the wood" instead of `{"action":"MINE","target":"oak_log"}`.

Phase 3 fixes this: training on 100k deterministic JSON pairs on a **coder model**
eliminates all English output and gives RL a solid starting point.

---

## Phase 3: Build the Pure JSON LLM

### Step 1 — Generate 100k Dataset (local, ~5 minutes)

```bash
python data_engine/generate_100k_dataset.py \
    --output ./data/stage1_100k.jsonl \
    --count 100000
```

Every example looks like this:

**Input (game state):**
```json
{"hp":20,"food":18,"pos":{"x":10,"y":64,"z":-5},"inv":{"oak_log":5,"wooden_planks":12,"stick":4},"nearby":[{"name":"crafting_table","dist":1.5}],"mobs":[],"time":6000,"raining":false,"goal":"craft_wooden_pickaxe"}
```

**Output (action):**
```json
{"action":"CRAFT","target":"wooden_pickaxe","quantity":1}
```

No English. No explanations. Pure deterministic state → action.

### Step 2 — Upload to Kaggle

1. Go to https://kaggle.com/datasets → New Dataset
2. Name it `mineagent-100k`
3. Upload `stage1_100k.jsonl`

### Step 3 — Run `training/phase3_training.ipynb` on Kaggle T4

Key differences from your current notebook:

| Setting | Phase 1/2 (old) | Phase 3 (new) | Why |
|---|---|---|---|
| Model | Llama-3.2-3B-Instruct | Qwen2.5-Coder-1.5B | Coder model = JSON-native, faster |
| LoRA rank | 16 | **128** | Forces deep behavioral rewiring |
| Target modules | 7 layers | **9 layers** (+ embed + lm_head) | Rewires vocabulary toward JSON |
| Epochs | 5 | **15** | Deterministic data = memorize it |
| LR | 3e-4 | **5e-5** | Precise fine-tuning, not generic |
| Weight decay | 0.0 | **0.1** | Prevents overfitting exact strings |
| NEFTune noise | off | **alpha=5** | Learns rules, not just memorized text |
| Dataset | 12k | **100k** | 8x more data |

### Expected Loss Curve

```
Epoch  1: loss ≈ 1.2  (model still sometimes outputs English)
Epoch  3: loss ≈ 0.4  (mostly JSON, occasional errors)
Epoch  7: loss ≈ 0.15 (valid JSON always, wrong actions sometimes)
Epoch 12: loss ≈ 0.07 (correct actions 80%+ of the time)
Epoch 15: loss ≈ 0.04 (schema adherence near-perfect)
```

The Phase 3 notebook includes a **JSON Schema Adherence Test** (Cell 6) that
tests 4 scenarios and confirms the model outputs valid JSON before you proceed to RL.

---

## Phase 4: Reinforcement Learning in Minecraft

### Architecture

```
Python RL Trainer (train_rl.py)
       │
       │  GET /state  →  JSON game state
       │  POST /action → JSON action result
       │
Node.js Mineflayer Bot (rl_server.js)
       │
       │  mineflayer API
       │
Minecraft Server (localhost:25565)
```

### Step 1 — Setup local Minecraft server

```bash
# Download Paper 1.20.1
wget https://api.papermc.io/v2/projects/paper/versions/1.20.1/builds/196/downloads/paper-1.20.1-196.jar

# Create server folder
mkdir minecraft-rl-server && cd minecraft-rl-server
cp ../paper-1.20.1-196.jar .

# Accept EULA
echo "eula=true" > eula.txt

# server.properties (important settings):
# gamemode=survival
# difficulty=normal
# enable-cheats=true     ← needed for /kill to reset episodes
# max-players=5
# view-distance=6        ← reduce for performance

# Start server
java -Xmx4G -Xms1G -jar paper-1.20.1-196.jar nogui
```

### Step 2 — Start the Mineflayer RL server

```bash
cd inference/
npm install mineflayer mineflayer-pathfinder express vec3
node rl_server.js
# → HTTP API listening on port 3001
# → Bot spawned in Minecraft
```

### Step 3 — Run RL training

```bash
python rl_training/train_rl.py \
    --model Tron101101/mineagent-phase3-lora \
    --steps 50000 \
    --goal craft_iron_pickaxe \
    --output ./checkpoints
```

### How PPO Works With the LLM

```
Episode start:
  Bot spawns in Minecraft world

Each step (up to 200 steps per episode):
  1. Python reads /state → JSON game state
  2. LLM generates JSON action (e.g. {"action":"MINE","target":"oak_log"})
  3. Python POSTs /action to Mineflayer bot
  4. Mineflayer executes action in Minecraft
  5. Result comes back (collected items, damage, etc.)
  6. Reward is computed from result
  7. (state, action, reward, log_prob) stored in buffer

Every 8 steps:
  PPO update:
    - Compute GAE advantages from reward sequence
    - For each (state, action) pair:
        compute new log_prob under current policy
        ratio = exp(new_log_prob - old_log_prob)
        PPO loss = -min(ratio * advantage, clip(ratio, 0.8, 1.2) * advantage)
    - Backprop through LoRA weights only
    - Optimizer step

Every 5000 steps:
  - Save checkpoint
  - If best mean reward → save best_model/
```

### Reward Signal Design

The reward function is dense (fires every step) to make RL sample-efficient:

| Event | Reward | Reasoning |
|---|---|---|
| Collect wood | +2.0 | Fundamental resource |
| Collect iron ore | +3.0 | Mid-game progress |
| Collect diamond | +10.0 | Major milestone |
| Craft any item | +5.0 | Using knowledge correctly |
| Kill hostile mob | +8.0 | Survival skill |
| Take damage | -2.0 per HP | Discourages suicidal behavior |
| Die | -20.0 | Terminal penalty |
| Idle/WAIT | -0.5 | Discourages doing nothing |
| Reach goal | +20.0 | Episode success |

### Progressive Goal Curriculum

Don't train on just one goal. Use a curriculum:

```python
GOAL_CURRICULUM = [
    # Phase A: Basic survival (easy, 0-10k steps)
    "collect_wood",
    "collect_stone",
    # Phase B: Tool crafting (medium, 10k-30k steps)
    "craft_wooden_pickaxe",
    "craft_stone_pickaxe",
    # Phase C: Full progression (hard, 30k-50k steps)
    "craft_iron_pickaxe",
    "craft_iron_sword",
    # Phase D: Advanced (expert, 50k+ steps)
    "find_diamond",
    "craft_diamond_pickaxe",
]
```

---

## Full File Structure

```
minecraft-llm-phase3/
│
├── data_engine/
│   └── generate_100k_dataset.py    ← Run locally to create 100k examples
│
├── training/
│   └── phase3_training.ipynb       ← Upload to Kaggle, run on T4
│
├── rl_training/
│   ├── rl_environment.py           ← Gymnasium env wrapping Mineflayer
│   └── train_rl.py                 ← PPO training loop
│
└── inference/
    └── rl_server.js                ← Node.js bot + HTTP API for RL
```

---

## Complete Timeline

| Week | Task | Time |
|---|---|---|
| Week 1 | Run `generate_100k_dataset.py`, upload to Kaggle | 1 hour |
| Week 1 | Run Phase 3 notebook on Kaggle (15 epochs × 100k) | ~6 hours |
| Week 1 | Verify JSON adherence test passes (≥ 75% correct) | 30 min |
| Week 2 | Setup local Minecraft server + test `rl_server.js` | 2 hours |
| Week 2-4 | Run RL training for 50k steps (curriculum) | ~20 hours |
| Week 4+ | Evaluate, refine reward function, continue training | ongoing |

---

## Quick Start Commands (in order)

```bash
# 1. Generate dataset (local)
python data_engine/generate_100k_dataset.py

# 2. Upload data/stage1_100k.jsonl to Kaggle Dataset "mineagent-100k"

# 3. Run Phase 3 training on Kaggle (upload training/phase3_training.ipynb)

# 4. After Phase 3 completes and uploads to HF, start RL:
java -jar paper-1.20.1-196.jar nogui &          # Minecraft server
node inference/rl_server.js &                    # Mineflayer bot
python rl_training/train_rl.py --steps 50000     # RL training
```
