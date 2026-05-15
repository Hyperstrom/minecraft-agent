# MineAgent — Autonomous Minecraft AI

> Day 1 build: Mineflayer bot + FastAPI backend with full test suite.

---

## Project Structure

```
minecraft-agent/
├── bot/                        # Node.js Mineflayer bot
│   ├── bot.js                  # Entry point
│   ├── config.js               # Configuration (reads .env)
│   ├── state_extractor.js      # Observation JSON builder
│   ├── actions.js              # Mineflayer action wrappers
│   ├── commands.js             # In-game chat commands
│   ├── logger.js               # Coloured console logger
│   ├── package.json
│   └── tests/
│       └── test_state.js       # Unit tests (no server needed)
│
├── backend/                    # Python FastAPI orchestrator
│   ├── main.py                 # API server
│   ├── config.py               # Settings (pydantic-settings)
│   ├── requirements.txt
│   └── tests/
│       └── test_backend.py     # Unit tests (no server needed)
│
├── .env.example                # Environment variable template
└── README.md
```

---

## Quick Start

### 1. Clone & configure

```powershell
cd "e:\Projects\MineCraft Agent"
Copy-Item .env.example .env
# Edit .env — set MC_HOST, MC_PORT, MC_USERNAME
```

### 2. Bot (Node.js)

```powershell
cd bot
npm install
node bot.js        # starts the bot (needs a running MC server)
```

### 3. Backend (Python)

```powershell
cd backend
pip install -r requirements.txt
python main.py     # starts FastAPI on http://localhost:8000
```

---

## Run Tests (No Minecraft Server Required)

```powershell
# Bot unit tests (21 tests)
cd bot
node tests/test_state.js

# Backend unit tests (12 tests)
cd backend
python tests/test_backend.py
```

---

## In-Game Commands

Once the bot is connected, any player can type in chat:

| Command | Description |
|---|---|
| `!help` | List all commands |
| `!status` | Health, food, position |
| `!state` | Print full Observation JSON |
| `!inventory` | Show inventory |
| `!follow` | Bot follows you |
| `!stop` | Stop movement |
| `!goto x y z` | Walk to coordinates |
| `!mine oak_log` | Mine nearest named block |
| `!say Hello!` | Bot says a message |

---

## API Endpoints (Backend)

| Method | Route | Description |
|---|---|---|
| `GET` | `/` | Version + status |
| `GET` | `/health` | Health check |
| `POST` | `/state` | Receive Observation JSON from bot |
| `GET` | `/state` | Get latest Observation JSON |
| `POST` | `/plan` | Get next action for a given state |
| `GET` | `/history` | Last 20 planned actions |
| `WS` | `/ws/{bot_id}` | Real-time bot ↔ backend channel |

---

## Roadmap

| Phase | Timeline | Goal |
|---|---|---|
| **Day 1** ✅ | Week 1–2 | Bot connects, echoes chat, state extraction, tests |
| **Phase 2** | Week 3–5 | Ollama / Mistral-7B integration, full Observe→Plan→Act loop |
| **Phase 3** | Week 6–10 | Dataset ingestion, QLoRA fine-tuning |
| **Phase 4** | Week 11–12 | Docker, CI/CD, monitoring |
