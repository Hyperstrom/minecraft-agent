# MineAgent — End-to-End Project Plan

> A 0.5B-parameter LLM agent that plays Minecraft on user command via Mineflayer.
> Realistic timeline: **10–12 weeks** of focused work, solo developer.

---

## Table of contents

1. [Why this plan, not another](#1-why-this-plan-not-another)
2. [System design](#2-system-design)
3. [The action vocabulary and state schema (fix these once, forever)](#3-the-action-vocabulary-and-state-schema)
4. [Day-by-day execution plan](#4-day-by-day-execution-plan)
5. [Building the synthetic dataset](#5-building-the-synthetic-dataset)
6. [Training the LLM](#6-training-the-llm)
7. [Polishing the LLM output (constrained decoding)](#7-polishing-the-llm-output)
8. [Integrating with Mineflayer](#8-integrating-with-mineflayer)
9. [Adding memory (short-term + long-term)](#9-adding-memory)
10. [Reinforcement learning fine-tune (DPO)](#10-reinforcement-learning-fine-tune)
11. [Evaluation and metrics](#11-evaluation-and-metrics)
12. [Common failure modes and how to debug them](#12-common-failure-modes)
13. [Hard truths to internalize before starting](#13-hard-truths)

---

## 1. Why this plan, not another

Most "build a Minecraft AI agent" plans on the internet are written by people who haven't shipped one. The traps they fall into:

- **They mix knowledge data into the action model.** Wiki + blogs + transcripts + (state, action) pairs all get dumped into SFT. The model learns to narrate Minecraft instead of play it.
- **They skip the simulator.** They generate fake trajectories with handwritten rules, never validate against a real Minecraft server, and the agent fails on first contact with the real game.
- **They aim for "loss → 0".** Cross-entropy on a stochastic-target task has an irreducible floor. Chasing zero is overfitting.
- **They go straight to RL.** RL on a broken SFT model gives garbage. The order is: SFT → working integration → RL.
- **They pick the wrong size model.** 0.5B is the smallest model that works for this. 7B works much better. If you can afford 1–3B, do that. Your choice of 0.5B is acceptable but tight.

This plan avoids all of those.

---

## 2. System design

You're building **four loosely coupled components**, not one model:

```
┌─────────────────────────────────────────────────────────────────┐
│                          USER                                    │
│                  "go get me some iron"                           │
└─────────────────────────────┬───────────────────────────────────┘
                              │ natural language command
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ COMPONENT 1: GOAL PLANNER                                        │
│ - Maps user command to a sequence of subgoals                    │
│ - v1: Python dict of hardcoded recipes                           │
│ - v2: Small LLM (your same Qwen 0.5B, second LoRA adapter)       │
│ Output: ["find_wood", "craft_wooden_pickaxe", "find_stone",      │
│          "craft_stone_pickaxe", "find_iron_ore", "mine_iron"]    │
└─────────────────────────────┬───────────────────────────────────┘
                              │ current subgoal
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ COMPONENT 2: ACTION POLICY (THE QWEN 0.5B MODEL)                 │
│ Input:  {state JSON, current subgoal, short memory}              │
│ Output: {next single action JSON}                                │
│ This is your existing notebook, retrained on better data.        │
└─────────────────────────────┬───────────────────────────────────┘
                              │ action JSON
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ COMPONENT 3: MINEFLAYER EXECUTOR                                 │
│ - Translates action JSON to Mineflayer API calls                 │
│ - Observes new state, produces fresh state JSON                  │
│ - Handles errors (can't reach target, inventory full, etc.)      │
└─────────────────────────────┬───────────────────────────────────┘
                              │ new state
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ COMPONENT 4: MEMORY                                              │
│ - Short-term: ring buffer of last 8 (state, action) tuples       │
│ - Long-term: SQLite log of (state, action, outcome) for RL       │
│ - Episodic: "what user told me last" + key events                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ (loop back to Component 2)
```

**Critical: keep these components separate.** The LLM does ONE thing — pick the next action. The planner does ONE thing — decompose goals. Memory is ONE component the LLM reads from. If you blur the boundaries you will spend weeks debugging.

### Optional component 5: knowledge retrieval (RAG)

For "how do I make X?" questions. Vector DB of Minecraft Wiki chunks. The planner queries it, NOT the action policy. Build this only if the planner needs it. v1 of the planner doesn't.

---

## 3. The action vocabulary and state schema

**Decide these on Day 1 and never change them.** Every breaking change downstream costs you days.

### Action vocabulary (10 verbs, no more)

```json
{"action": "MINE",   "target": "<block_name>"}
{"action": "MOVE",   "direction": "<north|south|east|west|up|down>", "distance": <int>}
{"action": "MOVE_TO","pos": {"x":<int>,"y":<int>,"z":<int>}}
{"action": "EAT",    "item": "<food_item>"}
{"action": "CRAFT",  "target": "<item>", "quantity": <int>}
{"action": "PLACE",  "block": "<block>", "pos": {"x":<int>,"y":<int>,"z":<int>}}
{"action": "ATTACK", "target": "<entity_name>"}
{"action": "EQUIP",  "item": "<item>", "slot": "<hand|head|chest|legs|feet>"}
{"action": "DROP",   "item": "<item>", "quantity": <int>}
{"action": "WAIT",   "ticks": <int>}
```

Reject every temptation to add an 11th. "LOOK_AT", "JUMP", "USE", "SLEEP", "DRINK" — all of these are compositions of the 10 above or belong in Mineflayer's executor logic. Action vocabulary bloat is the #1 killer of agent projects.

### State schema (v1, fixed forever)

```json
{
  "v": 1,
  "hp": 20,
  "food": 18,
  "pos": {"x": 10, "y": 64, "z": -22},
  "facing": "north",
  "inv": {"wooden_planks": 5, "stick": 4, "wooden_pickaxe": 1},
  "equipped": "wooden_pickaxe",
  "nearby": [
    {"name": "oak_log", "dist": 2.1, "pos": {"x":11,"y":64,"z":-22}},
    {"name": "stone",   "dist": 3.0, "pos": {"x":10,"y":63,"z":-23}}
  ],
  "mobs": [
    {"name": "zombie", "dist": 8.5, "hp": 20, "hostile": true}
  ],
  "time": 13000,
  "raining": false,
  "biome": "plains",
  "goal": "craft_iron_pickaxe",
  "subgoal": "find_stone",
  "memory": ["chopped 4 oak logs", "crafted wooden pickaxe"]
}
```

Rules:
- **`v` field always present.** Lets you add fields in v2 without breaking v1.
- **`nearby` capped at 8 closest blocks.** Top-N truncation keeps context window manageable.
- **`mobs` capped at 4 closest entities.**
- **`memory` is 3–8 short strings.** This is your short-term memory. Section 9 explains.
- **Canonical JSON serialization:** `json.dumps(obj, separators=(',',':'), sort_keys=True)`. One whitespace deviation across the dataset wastes weeks of training.

---

## 4. Day-by-day execution plan

This is 56 working days, structured as 8 weeks plus 4 weeks of polish/RL. Adjust to your pace, but **do not reorder phases**.

### Week 1: Foundation (Days 1–7)

**Goal: prove the environment works end-to-end with a dumb agent. No LLM yet.**

| Day | Task | Deliverable |
|---|---|---|
| 1 | Set up local Minecraft server (PaperMC 1.20). Install Node.js, Mineflayer. Create git repo. | `server/` runs, agent connects |
| 2 | Lock the action vocabulary and state schema (Section 3). Commit `schemas.json` to repo. | `schemas.json` |
| 3 | Write the state extractor: Mineflayer bot → JSON state per tick | `extract_state.js` |
| 4 | Write the action executor: action JSON → Mineflayer API calls. Handle errors gracefully. | `execute_action.js` |
| 5 | Write a "dumb agent": always picks `MINE` on nearest log. Verify state→action→new_state loop. | Bot chops a tree |
| 6 | Write 3 more dumb agents: greedy (closest resource), cautious (flee on low HP), explorer (random walk) | 4 scripted policies |
| 7 | Add JSON logging: every episode writes `episodes/ep_<id>.jsonl` with full (state, action, new_state) trail | Logs in expected schema |

**Exit criteria for Week 1:** Run 10 episodes of 100 steps each with the 4 dumb agents. Every step logged in correct schema. If this isn't working, STOP. Do not proceed to Week 2.

### Week 2: Synthetic data pipeline (Days 8–14)

**Goal: turn the dumb agents into a data-generation pipeline producing 200k (state, action) pairs.**

| Day | Task | Deliverable |
|---|---|---|
| 8 | Add world randomization: random seed, spawn point, time of day, weather per episode | Randomized starts |
| 9 | Add inventory randomization: 50% of episodes start with random inventory (forces edge cases) | Diverse starts |
| 10 | Add mob spawning: configurable mob density per episode (some peaceful, some hostile) | Combat episodes |
| 11 | Build the bucket sampler: episodes tagged with bucket labels (combat, night, full_inv, etc.) | Bucket tagging |
| 12 | Run 2000 episodes overnight. Should produce ~200k (state, action) pairs across buckets. | Raw dataset |
| 13 | Write `convert_to_sft.py`: raw JSONL → `{"messages":[user,assistant]}` format for SFTTrainer | `stage1_synth.jsonl` |
| 14 | Audit by hand: read 100 random samples. Discard any with wrong action for state. Document patterns. | Cleaned dataset |

**Exit criteria for Week 2:** A file `stage1_synth.jsonl` with ~200k entries. Bucket distribution matches your target (~10% combat, ~10% night, etc.). Hand-audit shows >95% reasonable labels.

### Week 3: Train v1 of the action policy (Days 15–21)

**Goal: a working 0.5B LoRA that emits valid action JSON for held-out states.**

| Day | Task | Deliverable |
|---|---|---|
| 15 | Use the improved notebook (already in your project). Update DATA_PATH to point at stage1_synth.jsonl. | Notebook configured |
| 16 | Run training. Expected eval_loss: 0.04–0.06. Time: ~6–8 hours on Kaggle T4. | Trained LoRA r=64 |
| 17 | Build a 500-sample held-out test set covering all buckets. Run JSON validity + action accuracy. | Eval report |
| 18 | If JSON validity <99%, add more JSON-formatting examples to data and retrain. | Validity ≥99% |
| 19 | If action accuracy <80% on some bucket, oversample that bucket and retrain. | Accuracy ≥80% per bucket |
| 20 | Export to GGUF Q8_0. Verify it loads in llama.cpp. | GGUF file |
| 21 | Buffer day for bug fixing, or push results to HF Hub. | v1 model published |

**Exit criteria for Week 3:** Held-out JSON validity ≥99%, mean action accuracy ≥80%, no bucket below 70%.

### Week 4: Close the loop (Days 22–28)

**Goal: the LLM is now driving Mineflayer in real Minecraft.**

| Day | Task | Deliverable |
|---|---|---|
| 22 | Stand up llama.cpp HTTP server hosting your GGUF. Test with curl. | Inference endpoint |
| 23 | Add LLM client to Mineflayer code: extract state → POST to LLM → parse action → execute | End-to-end loop |
| 24 | Run 10 episodes with goal `"chop_wood"`. Log everything. Expect chaos. | Baseline trajectories |
| 25 | Analyze failures: where does it loop? Where does it ignore food? Where does it walk into lava? | Failure taxonomy |
| 26 | Add constrained decoding (Section 7) using outlines or llama.cpp's grammar feature | Valid JSON 100% |
| 27 | Run 10 more episodes. Compare success rate to Day 24. | Improvement measured |
| 28 | Document what works, what doesn't. This drives the next phase. | Failure analysis doc |

**Exit criteria for Week 4:** Agent completes `"chop_wood"` (5 logs) in ≥40% of episodes. Other goals can be worse.

### Week 5: Add memory (Days 29–35)

**Goal: agent stops looping, remembers context, follows user through multi-step commands.**

| Day | Task | Deliverable |
|---|---|---|
| 29 | Implement short-term memory: ring buffer of last 8 actions + outcomes | `memory.js` |
| 30 | Inject memory summary into state JSON as `"memory":[...]` field | Memory in prompt |
| 31 | Add 50k examples to training data where state includes meaningful `"memory"` field | Memory-aware data |
| 32 | Retrain LoRA on augmented data (warm start from v1 weights). 1 epoch. | v2 LoRA |
| 33 | Implement long-term memory: SQLite log of every (state, action, outcome) | Persistent log |
| 34 | Implement episodic memory: extract key events ("died at y=12", "found diamond at...") to a fact store | Fact extraction |
| 35 | Test: give agent a 3-step command ("get wood, then stone, then craft pickaxe"). Verify memory works. | Multi-step demo |

**Exit criteria for Week 5:** Agent follows a 3-subgoal sequence without losing context. Doesn't loop on the same action 5+ times.

### Week 6: Goal planner (Days 36–42)

**Goal: agent accepts natural-language commands from a user.**

| Day | Task | Deliverable |
|---|---|---|
| 36 | Write the v1 planner: Python dict mapping high-level goals to subgoal sequences | `planner.py` |
| 37 | Cover 20 common commands: "chop wood", "make a pickaxe", "find iron", "build shelter", etc. | 20 recipes |
| 38 | Add the chat interface: Mineflayer listens for chat from a player, sends to planner | Chat → goals |
| 39 | Test 20 commands end-to-end. Note which fail and why. | E2E results |
| 40 | For complex commands (build a house), break into stages. v1 planner is fine for simple stuff. | Stage planning |
| 41 | Optional: train a second LoRA adapter for goal decomposition. Skip if v1 dict planner works. | Decision documented |
| 42 | Buffer day for polish. | Stable demo |

**Exit criteria for Week 6:** User can type 10+ different commands and the agent attempts them with reasonable subgoal sequences.

### Week 7: Real human data + scale (Days 43–49)

**Goal: incorporate MineRL or VPT data to improve recovery behavior.**

| Day | Task | Deliverable |
|---|---|---|
| 43 | Download a subset of MineRL (~10GB sample) | Raw MineRL data |
| 44 | Write `convert_minerl.py`: MineRL frames + actions → your state schema + your action verbs | Converter |
| 45 | Run conversion. Expect ~500k–2M (state, action) pairs added | Converted data |
| 46 | Hand-audit 100 converted samples. Fix conversion bugs. | Cleaned data |
| 47 | Combine synthetic + MineRL into `stage2_combined.jsonl` (~2M pairs) | Big dataset |
| 48 | Retrain LoRA from scratch (not warm-start) on combined data. ~12 hours on T4. | v3 LoRA |
| 49 | Re-run all evals. Should see action accuracy ↑, especially on hard buckets. | v3 metrics |

**Exit criteria for Week 7:** Action accuracy ≥85% mean, ≥75% on every bucket. Better recovery behavior in failure scenarios.

### Week 8: DPO setup (Days 50–56)

**Goal: gather preference data and start RL fine-tuning.**

| Day | Task | Deliverable |
|---|---|---|
| 50 | Run v3 agent for 500 episodes across all 20 commands. Log everything. | Rollout dataset |
| 51 | Define success criteria per command: did the subgoal complete? How many steps? | Success labels |
| 52 | Build preference pairs: for the same state, "successful" action > "failed/looping" action | DPO pairs |
| 53 | Aim for 50k preference pairs minimum | Preference dataset |
| 54 | Set up DPO trainer in TRL. Beta=0.1, learning rate 5e-6 (much lower than SFT) | DPO training script |
| 55 | Run DPO for 1 epoch. ~6 hours on T4. | v4 LoRA |
| 56 | Re-run all evals. Compare v3 vs v4. | DPO results |

**Exit criteria for Week 8:** v4 shows ≥10% improvement in episode-completion rate over v3 on the 20-command test suite.

### Weeks 9–12: Polish

- Week 9: Edge cases (lava, drowning, mob swarms). Add 30k targeted examples per edge case.
- Week 10: Better goal planner. If v1 dict is hitting limits, train the second LoRA.
- Week 11: RAG for game knowledge. Wiki dump → embeddings → planner queries.
- Week 12: Demo recording, documentation, public release.

---

## 5. Building the synthetic dataset

This is what your "good enough to train 50% of parameters" question is really about. The key insight: **dataset quality determines what fraction of model capacity you can actually use.** A bad 1M-example dataset wastes a 0.5B model. A good 200k-example dataset uses it fully.

### Quality > quantity (always, but especially below 7B params)

The math: a 0.5B model has ~500M parameters. With LoRA r=64 on attn+MLP (the config in your improved notebook), you're training ~35M params. To train 35M params well, you need roughly **35M × 10 to 35M × 100 = 350M to 3.5B training tokens**.

- 200k examples × ~150 tokens/example × 4 epochs = **120M tokens**. Under-trained.
- 2M examples × ~150 tokens × 3 epochs = **900M tokens**. Solid.
- 5M examples × ~150 tokens × 2 epochs = **1.5B tokens**. Diminishing returns.

So your target is **2M (state, action) pairs**. Not more. Not less.

### The four data sources

| Source | Size | Quality | Time to build | Use for |
|---|---|---|---|---|
| Mineflayer scripted agents | 200k–500k | High (your schema, controllable) | 1 week | Bootstrapping, edge cases |
| MineRL converted | 500k–2M | High (real humans) | 1 week | General behavior, recovery |
| VPT YouTube IDM-labeled | 5M+ | Medium (pseudo-labels) | 2 weeks | Optional, only if low on data |
| Hand-written rule traces | 10k–50k | Very high | 3 days | Specific buckets (combat, lava) |

**My recommendation for v1: 300k Mineflayer synthetic + 1.5M MineRL converted = ~1.8M total.**

### Bucket distribution (this is the lever)

For 1.8M pairs, target this distribution:

| Bucket | Count | Why |
|---|---|---|
| Exploration (no immediate threat, gathering resources) | 540k (30%) | Baseline behavior |
| Crafting chains | 450k (25%) | Multi-step goal completion |
| Combat (mobs present, hostile) | 180k (10%) | Survival skills |
| Night survival | 180k (10%) | Most common death scenario |
| Building (placing blocks structurally) | 180k (10%) | Often missing from datasets |
| Inventory management (full, dropping) | 90k (5%) | Edge case |
| Underground exploration | 90k (5%) | Disorientation |
| Hazard avoidance (lava, drowning, fall) | 90k (5%) | Hardest edge cases |

If a bucket is naturally rare (only 0.5% of random episodes are "lava nearby"), you must **deliberately seed episodes** to hit that bucket. Otherwise the model never learns it.

### Multiple labelers per state (the trick that beats overfitting)

For each generated state, have 2–3 different scripted agents produce their preferred action. Keep all of them as separate training examples:

```
state_X → action_A (from greedy agent)
state_X → action_B (from cautious agent)
state_X → action_C (from goal-driven agent)
```

This teaches the model that **multiple actions are valid** — the right inductive bias for a stochastic-target task. It also raises your floor on dataset usefulness without needing more states.

### Canonical formatting (cannot stress this enough)

Every JSON in your dataset must be produced by:

```python
import json
def canonical(obj):
    return json.dumps(obj, separators=(',', ':'), sort_keys=True, ensure_ascii=False)
```

No spaces after colons. No trailing commas. Keys sorted alphabetically. If one example has `{"hp": 20, "food": 20}` and another has `{"food":20,"hp":20}`, you have just doubled your effective vocabulary cost.

### Schema validation in the pipeline

```python
from jsonschema import validate
STATE_SCHEMA = {...}  # your v1 schema
ACTION_SCHEMA = {...}

def validate_pair(state_json, action_json):
    validate(json.loads(state_json), STATE_SCHEMA)
    validate(json.loads(action_json), ACTION_SCHEMA)
```

Run this on every example before it enters the dataset. Reject malformed ones immediately. Otherwise you'll spend a week debugging a model that's learning to copy your bugs.

---

## 6. Training the LLM

You already have the improved notebook. Recap of what matters:

- **LoRA r=64, alpha=128, dropout=0.05** on attn + MLP only. NOT embeddings, NOT lm_head.
- **bf16 if available, fp16 otherwise.** Never both.
- **No NEFTune** — kills structured-output tasks.
- **Effective batch 64** (per_device=16, grad_accum=4).
- **learning_rate=2e-4**, cosine, warmup 0.03.
- **EarlyStopping** on eval_loss, patience=3.

### Two-stage training schedule

Don't train once on 2M pairs. Train in two stages:

**Stage 1: Synthetic only (200k–300k)**
- 3–4 epochs
- Learning rate 2e-4
- Goal: model learns the JSON schema, basic action selection
- Expected eval_loss: 0.05–0.07

**Stage 2: Combined (2M)**
- 1–2 epochs, warm-start from Stage 1 LoRA
- Learning rate 5e-5 (4× lower)
- Goal: model learns recovery behaviors from MineRL data
- Expected eval_loss: 0.03–0.05

Stage 2 with a fresh model would forget the clean schema discipline. Warm-starting protects that.

### What you cannot do with 0.5B

- It will not learn complex multi-step planning. That's what the planner is for.
- It will not learn to write code. Not its job.
- It will not have strong "common sense" about Minecraft. RAG provides that to the planner.
- It will not generalize to commands far outside the 20 you trained.

Plan around these limits. Don't fight them.

---

## 7. Polishing the LLM output

After SFT, the model emits valid JSON ~99% of the time. The remaining 1% — invalid JSON, hallucinated action verbs, wrong field types — will crash your Mineflayer integration. Three fixes:

### Fix 1: Constrained decoding (most important)

Use a grammar that forces the output to match your action schema. Two options:

**Option A: llama.cpp GBNF grammar.** Write a GBNF file matching your action JSON schema. Pass `--grammar-file action.gbnf` to llama.cpp. The decoder physically cannot emit invalid tokens.

```
# action.gbnf
root ::= "{" "\"action\":" action-verb action-args "}"
action-verb ::= "\"MINE\"" | "\"MOVE\"" | "\"EAT\"" | "\"CRAFT\"" | ...
action-args ::= "," argument (("," argument)*)?
argument ::= "\"target\":" string | "\"quantity\":" number | ...
```

This is bulletproof. JSON validity goes to 100%. Action verbs are always valid. The model can still pick the wrong action, but it cannot emit garbage.

**Option B: `outlines` library** (Python side). Same idea but with a Pydantic schema. Easier to write than GBNF.

### Fix 2: Output validation + fallback

Even with grammars, validate every output server-side:

```python
def parse_action(raw):
    try:
        obj = json.loads(raw)
        validate(obj, ACTION_SCHEMA)
        if obj['action'] not in VALID_VERBS:
            raise ValueError(f"unknown action: {obj['action']}")
        return obj
    except Exception:
        return {"action": "WAIT", "ticks": 20}  # safe fallback
```

Never let a parse error propagate to Mineflayer. Always have a safe default.

### Fix 3: Sampling parameters

For action selection, use **near-greedy decoding**:
- `temperature=0.1` (not 0)
- `top_p=0.9`
- `max_new_tokens=64`
- `repetition_penalty=1.05`

Temperature 0 sometimes loops on bad actions. 0.1 gives just enough exploration to break out.

---

## 8. Integrating with Mineflayer

The integration is a tight loop. The whole loop must complete in <200ms or the agent feels sluggish. Architecture:

```
┌──────────────────────────────────────────────────────────────┐
│              Mineflayer bot (Node.js)                         │
│                                                                │
│  Every game tick (50ms):                                       │
│    1. extractState() → JSON                                    │
│    2. Has it changed meaningfully? If no, skip.                │
│                                                                │
│  When LLM call needed:                                         │
│    3. POST http://localhost:8080/action {state, memory}        │
│    4. Receive {action: ...}                                    │
│    5. validateAction()                                         │
│    6. executeAction() — translates to Mineflayer calls         │
│    7. Update memory                                            │
└──────────────────────────────────────────────────────────────┘
                                ↕
┌──────────────────────────────────────────────────────────────┐
│             LLM server (llama.cpp)                            │
│             Q8_0 GGUF, ~700MB, CPU inference                  │
│             Throughput: ~30–50 tok/s on modern CPU            │
│             Action JSON is ~30 tokens → ~600ms                │
└──────────────────────────────────────────────────────────────┘
```

### Async control loop

Don't block the Mineflayer event loop on LLM calls. Pattern:

```javascript
let pendingAction = null;
let llmInFlight = false;

bot.on('physicsTick', async () => {
    // Execute current action if any
    if (pendingAction) {
        await executeStep(pendingAction);
    }

    // If no action and no in-flight LLM call, request one
    if (!pendingAction && !llmInFlight) {
        llmInFlight = true;
        const state = extractState(bot);
        try {
            const action = await callLLM(state);
            pendingAction = validate(action);
        } finally {
            llmInFlight = false;
        }
    }
});
```

### Don't call the LLM every tick

The LLM picks "what action to start next." Once started, a `MINE` action takes 1–3 seconds to complete (mining a block). Don't re-query the LLM every 50ms during that — only when:
- Current action completed
- Current action failed
- Critical state change (HP dropped, hostile mob within 4 blocks, took damage)

This drops LLM calls from 20/sec to 1/2–3sec, which the 0.5B can handle on CPU.

### Failure handling

Mineflayer actions fail constantly: pathfinding times out, blocks become unreachable, mobs interrupt. Handle each failure as a state update with `"last_action_failed": true` so the LLM can decide what to do next.

---

## 9. Adding memory

You asked for the LLM to "remember past in-game conversation." Three memory layers, each solving a different problem.

### Layer 1: Short-term memory (last N actions)

Just a ring buffer of the last 8 (action_verb, outcome) tuples, serialized as 3–8 short strings injected into the state JSON:

```json
"memory": [
  "chopped 4 oak_log (success)",
  "crafted wooden_pickaxe (success)",
  "moved north 8 blocks (success)",
  "attempted mine stone (failed: no_target)"
]
```

The LLM sees this in every prompt. It's how the model knows "I just crafted a pickaxe so I shouldn't craft another one."

Implementation: append to `bot.memory` after every action, truncate to length 8.

**Training data must include this field.** That's why Day 31 in the plan adds 50k examples with non-empty `"memory"` and retrains.

### Layer 2: Episodic memory (key facts)

Some facts matter across many actions:
- "Player asked me to build a house"
- "Found diamonds at x=120, y=12, z=-45"
- "Died at coords (50, 30, 12) — there's lava there"

These don't fit in a 3–8 string ring buffer. Store them as a list of facts in a Python dict, summarized into 1–2 sentences and injected into the state JSON as a `"facts"` field:

```json
"facts": [
  "User wants a wooden house",
  "Diamond ore at (120,12,-45)",
  "Lava hazard at (50,30,12)"
]
```

Limit to 5 facts at a time. Use a simple relevance scoring (most recent + closest to current position) to pick which 5.

Fact extraction is rule-based, not LLM-based: regex on chat messages, position-based on death events, etc. Don't over-engineer.

### Layer 3: Long-term memory (SQLite log)

Every (state, action, outcome) gets written to `agent.db`:

```sql
CREATE TABLE steps (
    id INTEGER PRIMARY KEY,
    episode_id TEXT,
    step_idx INTEGER,
    state_json TEXT,
    action_json TEXT,
    success INTEGER,
    reward REAL,
    ts INTEGER
);
```

This memory is **not read by the LLM during inference.** It's read by:
- The RL data builder (Week 8)
- Your debugging dashboards
- Future versions when you want to re-train on the agent's own play

### Why not put long history directly in the prompt?

You might think "just dump the last 1000 actions into the prompt." Two problems:
1. Qwen 0.5B has a 32k context but a 256–512 token *training* context. Long prompts at inference produce garbage. The model never saw that distribution during training.
2. Cost: every token costs CPU time at inference. 1000 actions = 30k tokens = 10+ second latency.

Layers 1 and 2 give the LLM exactly enough context. Layer 3 is for offline analysis.

---

## 10. Reinforcement learning fine-tune

After SFT + integration, the agent will look promising in demos and fail in real play. RL closes the gap. Use **DPO**, not PPO. Reasons:

- DPO doesn't need a reward model — preferences are enough
- DPO is stable, PPO is not
- DPO has a reference implementation in TRL
- PPO on a 0.5B model with 35M trainable params will collapse without careful tuning

### Preference data generation

For each (state, action_taken, outcome) in your rollout log, generate alternatives:

1. **Same state, model-sampled alternative actions** at higher temperature
2. **Same state, planner-suggested action** from a different scripted agent
3. **Same state, "what should have happened"** (rule-based)

Label preferences:
- If the taken action succeeded → it's preferred over the alternatives
- If it failed → the alternative is preferred
- If both equivalent → discard the pair

Aim for **50k preference pairs minimum**. More is better but DPO converges fast.

### DPO training config

```python
from trl import DPOTrainer
dpo_args = dict(
    learning_rate=5e-6,        # 40× lower than SFT
    beta=0.1,                  # KL constraint strength
    max_length=512,
    max_prompt_length=384,
    num_train_epochs=1,        # 1 is usually enough
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    bf16=True,
    optim='adamw_8bit',
)
```

Run on top of your SFT LoRA, not from scratch. The SFT model is the reference; DPO nudges it toward preferred actions while staying close to reference.

### What DPO won't fix

DPO improves choices among actions the model already considers. It will NOT teach the model an action verb it never saw. If the model can't EAT at all, DPO won't help — go back to SFT.

---

## 11. Evaluation and metrics

**Loss is not your metric.** Loss is a training signal. Real metrics:

| Metric | Target | How to measure |
|---|---|---|
| JSON validity rate | 100% with grammar, 99%+ without | Parse 1000 held-out outputs |
| Action verb correctness | ≥95% | Compare predicted verb to held-out label |
| Action accuracy (full) | ≥85% | Full action JSON match |
| Per-bucket action accuracy | ≥75% on every bucket | Stratified evaluation |
| Episode completion rate | ≥60% by Week 8, ≥80% by Week 12 | Run 100 episodes per command |
| Mean steps to completion | Decreases over training versions | Episode log analysis |
| Agent death rate | <20% per 1000-step episode | Episode log |
| Looping rate (same action 5+ times in row) | <5% of episodes | Episode log analysis |

Build a `evaluate.py` script that produces this report after every model version. Run it as a CI step.

---

## 12. Common failure modes

These will happen to you. Have a plan for each.

| Symptom | Cause | Fix |
|---|---|---|
| Agent emits invalid JSON | No grammar, too high temp | Constrained decoding (Section 7) |
| Agent loops on same action | No memory, too low temp | Add memory layer, raise temp to 0.15 |
| Agent ignores food, dies of hunger | "EAT" underrepresented in data | Oversample low-food bucket 5× |
| Agent walks into lava | "lava nearby" bucket underrepresented | Add 30k hand-written lava examples |
| Agent does nothing for 100 ticks | "WAIT" is overrepresented | Penalize WAIT in DPO preference data |
| Action accuracy plateau at 70% | Dataset has multiple valid actions per state | This is FINE. Loss can't go lower. Use multiple-labeler approach. |
| Loss decreases but eval action accuracy doesn't | Overfitting to training distribution | Add MineRL data, increase dropout |
| GGUF model behaves differently than LoRA | Q8_0 quantization | Use Q6_K or merge to FP16 |
| Agent is fast but stupid | Model too small | Realistic — 0.5B is the floor. Train longer or move to 1.5B. |

---

## 13. Hard truths

- **You will not match Voyager.** Voyager used GPT-4 in the loop. You're using 0.5B. A reasonable target is "completes 5–10 common goals reliably," not "plays the game open-endedly." This is still impressive.
- **Most of your time goes to data, not training.** Plan accordingly. ~60% data, ~20% integration, ~10% training, ~10% RL.
- **The integration layer is harder than the model.** Mineflayer is finicky. Pathfinding fails. Blocks have weird interactions. Budget time for this.
- **You will retrain at least 5 times.** Each retraining reveals a new bug in the data or a new failure mode. This is normal.
- **There is no "done."** At week 12 you have a working agent that handles a few commands well. Real production agents iterate for years.
- **0.5B is a real constraint.** If you find yourself fighting the model's intelligence ceiling repeatedly, the answer is to switch to Qwen2.5-1.5B, not to make 0.5B smarter. The plan works at 1.5B too — just substitute the model name.

---

## What to build first, concretely

If you're starting Monday:

1. **Day 1 morning:** Install PaperMC, Mineflayer, Node.js. Get a bot to connect to a local server.
2. **Day 1 afternoon:** Commit `schemas.json` with the exact state and action schemas from Section 3.
3. **Day 2:** Write `extract_state.js` (50–100 lines). Test it on a running bot.
4. **Day 3:** Write `execute_action.js` (100–200 lines). Test all 10 verbs.
5. **Day 4:** Wire them together with a "always MINE the nearest log" dumb agent. If it chops a tree, you're on track.

Everything in this document depends on Days 1–4 working. Don't skip them.

Good luck. The plan is realistic, the difficulty is honest, and the result is achievable. Build the foundation carefully and the rest follows.
