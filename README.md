# MineAgent — Autonomous Minecraft AI

> **Phase 2 complete:** LLM-driven autonomous agent using Ollama + Llama 3.2:3b.
> The bot observes the Minecraft world, plans with a local LLM, and acts every 5 seconds.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Node.js](https://img.shields.io/badge/Node.js-18%2B-green)](https://nodejs.org)
[![Ollama](https://img.shields.io/badge/Ollama-llama3.2%3A3b-orange)](https://ollama.ai)
[![Tests](https://img.shields.io/badge/tests-200%2B%20passing-brightgreen)](#testing)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         MineAgent                               │
│                                                                 │
│  Minecraft World                                                │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────┐   Observation JSON   ┌─────────────────────────┐ │
│  │ Mineflayer│ ──────────────────▶ │   FastAPI Backend       │ │
│  │   Bot     │                     │   (Python)              │ │
│  │  (Node)   │ ◀────────────────── │  • Prompt builder       │ │
│  └──────────┘   Action + params    │  • Ollama client        │ │
│       │                            │  • ChromaDB memory      │ │
│       │ Actions:                   │  • Recipe advisor       │ │
│       │  SEEK / MINE / MOVE        │  • Session logger       │ │
│       │  CRAFT / IDLE / CHAT       └──────────┬──────────────┘ │
│       │                                       │                 │
│       │                            ┌──────────▼──────────────┐ │
│       │                            │   Ollama (local)        │ │
│       │                            │   llama3.2:3b           │ │
│       │                            │   ~3-5s per plan        │ │
│       └────────────────────────────└─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

**Flow:** Bot state → `/plan` API → prompt builder → Ollama LLM → JSON action → bot executes

---

## Project Structure

```
minecraft-agent/
├── bot/                          # Node.js Mineflayer bot
│   ├── bot.js                    # Entry point — connects, spawns near player
│   ├── config.js                 # Configuration (reads .env)
│   ├── state_extractor.js        # Observation JSON builder (16-block scan)
│   ├── actions.js                # Mineflayer action wrappers (SEEK, MINE, MOVE…)
│   ├── planner_client.js         # Autonomous plan loop — calls /plan every 5s
│   ├── commands.js               # In-game chat command handler
│   ├── logger.js                 # Coloured console logger
│   ├── package.json
│   └── tests/
│       └── test_state.js         # Unit tests (no server needed)
│
├── backend/                      # Python FastAPI orchestrator
│   ├── main.py                   # API server + Ollama warmup on startup
│   ├── config.py                 # Settings (pydantic-settings)
│   ├── prompt_builder.py         # LLM prompt construction + situation analysis
│   ├── tool_registry.py          # All LLM-callable actions with descriptions
│   ├── ollama_client.py          # Async Ollama HTTP client
│   ├── memory.py                 # ChromaDB vector memory
│   ├── knowledge_seeder.py       # Seeds ChromaDB with Minecraft knowledge
│   ├── recipe_advisor.py         # Craftable item hints from inventory
│   ├── session_logger.py         # Logs every action for future fine-tuning
│   ├── requirements.txt
│   └── tests/
│       ├── test_backend.py       # Backend unit tests
│       └── test_llm_quality.py   # 200+ LLM response quality tests
│
├── data/
│   └── chroma_db/                # ChromaDB persistent storage
├── .env.example                  # Environment variable template
├── .env                          # Your local config (git-ignored)
├── CHANGELOG.md                  # Version history
└── README.md
```

---

## Requirements

| Component | Version | Notes |
|---|---|---|
| Node.js | 18+ | For the Mineflayer bot |
| Python | 3.10+ | For FastAPI backend |
| Ollama | latest | Local LLM server |
| Minecraft Java | 1.21+ | With LAN/server enabled |
| GPU VRAM | ≥ 4 GB | For llama3.2:3b inference |

---

## Quick Start

### 1. Install Ollama & pull model

```powershell
# Set model path (do this once permanently)
[System.Environment]::SetEnvironmentVariable("OLLAMA_MODELS", "E:\ollama\models", "User")

# Pull the model (2 GB download)
ollama pull llama3.2:3b

# Verify
ollama list
# NAME           ID              SIZE    MODIFIED
# llama3.2:3b    ...             2.0 GB  ...
```

### 2. Configure environment

```powershell
cd "e:\Projects\MineCraft Agent"
Copy-Item .env.example .env
# Edit .env:
#   MC_HOST=localhost
#   MC_PORT=25565
#   MC_USERNAME=MineAgent
#   OLLAMA_URL=http://localhost:11434
#   OLLAMA_MODEL=llama3.2:3b
```

### 3. Start backend

```powershell
cd backend
pip install -r requirements.txt
python main.py
# INFO: Warming up Ollama (pre-loading model into GPU)...
# INFO: Ollama warm-up done — model is loaded and ready
# INFO: Uvicorn running on http://0.0.0.0:8000
```

### 4. Start bot

```powershell
cd bot
npm install
node bot.js
# [INFO] Connecting to localhost:25565 as "MineAgent"
# [INFO] Bot spawned!
# [INFO] MineAgent online! Use !goal <text> to set a goal, then !planner on to start.
```

### 5. Play

Open Minecraft, join the LAN world, then type in chat:

```
!goal collect 10 wood blocks
!planner on
```

Bot will SEEK the nearest oak_log → walk to it → mine it. Repeat until goal is complete.

---

## In-Game Commands

| Command | Description |
|---|---|
| `!help` | List all commands |
| `!status` | Health, food, position |
| `!state` | Print full Observation JSON |
| `!inventory` | Show inventory |
| `!follow` | Bot follows you |
| `!stop` | Stop all movement |
| `!goto x y z` | Walk to exact coordinates |
| `!mine <block>` | Mine nearest named block |
| `!say <text>` | Bot sends chat message |
| `!goal <text>` | Set the bot's current goal |
| `!planner on` | Start autonomous AI planning |
| `!planner off` | Stop autonomous AI planning |
| `!planner status` | Show current goal + planner state |
| `!planer off` | Same as above (typo tolerance) |

---

## LLM Action System

The LLM can issue these actions each planning tick:

| Action | Params | When to use |
|---|---|---|
| `SEEK` | `target: str` | Target block NOT in nearby_blocks — bot navigates and mines |
| `MINE` | `block: str` | Target block IS in nearby_blocks |
| `MOVE` | `direction, distance` | Exploration |
| `CRAFT` | `item, count` | Craft when materials available |
| `COLLECT` | `item, quantity` | Pick up dropped items |
| `EAT` | `item` | Eat food from inventory |
| `FOLLOW` | `player` | Follow a player |
| `GOTO` | `x, y, z` | Walk to exact coordinates |
| `CHAT` | `message` | Send in-game message |
| `STOP` | — | Stop all movement |
| `IDLE` | — | No goal set, wait |

### Key LLM Rules (enforced by prompt)

- **NEVER** mine `grass_block`/`dirt` when goal is to gather wood
- **If target block NOT in nearby_blocks** → use `SEEK` to find it
- **If target block IS in nearby_blocks** → use `MINE` directly
- **No goal set** → use `IDLE` and wait for player

---

## API Endpoints

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Version + status |
| `GET` | `/health` | Health check |
| `POST` | `/plan` | Get next action for a given state |
| `GET` | `/history` | Last 20 planned actions |
| `POST` | `/goal` | Update current goal |
| `GET` | `/status` | Ollama + memory status |
| `GET` | `/logs/stats` | LLM vs fallback ratio, actions breakdown |
| `GET` | `/memory/stats` | ChromaDB memory count |
| `POST` | `/memory/search` | Search memories (RAG debug) |
| `GET` | `/debug/ollama` | Test Ollama directly, see raw response |
| `POST` | `/debug/ollama` | Custom prompt test |

---

## Testing

```powershell
cd backend

# Fast unit tests (29 tests, no LLM needed, ~3s)
python -m pytest tests/test_llm_quality.py::TestSituationAnalysis tests/test_llm_quality.py::TestPromptBuilder tests/test_llm_quality.py::TestExtractJson -v

# Full LLM quality tests (200+ tests, requires Ollama running, ~10 min)
python -m pytest tests/test_llm_quality.py -v

# Manual 6-scenario quick test
python tests/test_llm_quality.py
```

**Test coverage:**

| Test class | Tests | What it checks |
|---|---|---|
| `TestSituationAnalysis` | 12 | Correct MINE/SEEK hints generated |
| `TestPromptBuilder` | 10 | Prompt structure, system rules, examples |
| `TestExtractJson` | 7 | JSON parsing from all LLM output formats |
| `TestLLMResponseQuality` | 170+ | Live LLM: SEEK/MINE correctness, never wrong blocks, consistency |

---

## Verify Ollama is Working

```powershell
# 1. Check model is loaded
ollama list

# 2. Direct API test
Invoke-RestMethod http://localhost:11434/api/tags

# 3. Via backend (after python main.py)
# Open in browser:
#   http://localhost:8000/debug/ollama
#   http://localhost:8000/logs/stats
#   http://localhost:8000/docs     ← interactive API docs
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ECONNRESET` on /plan | Backend restarted — planner auto-retries once |
| Bot keeps mining grass | Update to latest — fixed by SEEK action + prompt rules |
| Bot not moving after SEEK | Fixed by async goNear with proper promise resolution |
| `MaxListenersExceededWarning` | Fixed by proper listener cleanup in goNear |
| Ollama timeout on first plan | Fixed by startup warmup — model pre-loaded into GPU |
| Bot spawns underground | Use `!goto <x> 75 <z>` to reach surface |

---

## Roadmap

| Phase | Status | Goal |
|---|---|---|
| **Phase 1** | ✅ Done | Bot connects, state extraction, basic commands, tests |
| **Phase 2** | ✅ Done | Ollama LLM integration, SEEK/MINE/MOVE/CRAFT loop, ChromaDB memory |
| **Phase 3** | 🔜 Next | Dataset collection, QLoRA fine-tuning on Llama 3.2:3b |
| **Phase 4** | 📅 Planned | Docker, CI/CD, monitoring dashboard |

### Phase 3 Plan
- Use `session_logger.py` to record successful action sequences
- Target: 5,000+ (state, action, outcome) triplets
- Fine-tune with QLoRA (r=32, 10 epochs) on RTX 3050 Ti
- Evaluate: action accuracy, goal completion rate

---

## Hardware Tested On

- **GPU:** NVIDIA RTX 3050 Ti (4 GB VRAM)
- **Model:** llama3.2:3b (2 GB VRAM usage)
- **Inference:** ~3–5s per planning tick
- **Planning interval:** 5s (matches inference time)
