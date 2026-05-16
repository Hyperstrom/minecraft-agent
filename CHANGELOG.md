# Changelog

All notable changes to MineAgent are documented here.
Format: [Semantic Versioning](https://semver.org/) — `MAJOR.MINOR.PATCH`

---

## [2.3.0] — 2026-05-16 — *Stability & Memory Leak Fixes*

### Fixed
- **`MaxListenersExceededWarning`** — `goNear()` now removes **both** `goal_reached`
  and `path_stop` listeners when either fires. Added `resolved` guard to prevent
  double-resolve. Memory leak is fully plugged.
- **`ECONNRESET` on `/plan`** — Planner now auto-retries once (2s delay) on
  `ECONNRESET`, `ECONNREFUSED`, and `ETIMEDOUT`. Recovers silently from transient
  backend connection drops during long SEEK operations.
- **Bot raised max listeners** — `bot.setMaxListeners(50)` added as safety cap.

### Changed
- `state_extractor.js`: `getNearbyBlocks` scan radius `5 → 16` blocks — bot now
  sees trees up to 16 blocks away, eliminating the root cause of endless SEEK loops.
- `state_extractor.js`: `INTERESTING_BLOCKS` expanded with `acacia_log`,
  `mangrove_log`, `cherry_log`, `cobblestone`, `barrel`, all deepslate ore variants.
- `state_extractor.js`: Grass/dirt now sorted to the **end** of `nearby_blocks` list
  so the LLM sees valuable blocks (logs, ores) first.

---

## [2.2.0] — 2026-05-16 — *SEEK Action & Async Pathfinding*

### Added
- **`SEEK` action** (`actions.js`, `tool_registry.py`) — searches up to 256 blocks
  for a target block, pathfinds to it, then mines it automatically.
  Replaces the broken pattern of MINE being called with wrong blocks.
- **`goNear()` is now truly async** — returns a `Promise` that resolves on
  `goal_reached` / `path_stop` events or after a 30s timeout. Previously it was
  fire-and-forget, causing SEEK to restart every 5s with the bot never moving.
- **`seekBlock()` seek-and-mine** — after navigating to the block, automatically
  digs it. Falls back to `mineNearest()` if chunk update moved the block.
- **`goNearPlayer(username)`** — new action to walk near a specific player.

### Fixed
- Bot kept mining `grass_block` when goal was `gather oak wood` — fixed by:
  1. Expanded scan radius (sees logs at distance)
  2. SEEK navigates AND mines in one operation
  3. Strict prompt rules (never mine wrong blocks)

---

## [2.1.0] — 2026-05-16 — *LLM Prompt Rewrite & Quality Tests*

### Added
- **`_situation_analysis()`** in `prompt_builder.py` — goal-aware hint injected
  into every prompt. Tells LLM exactly: "oak_log found at d=3 → use MINE" or
  "no oak_log in nearby_blocks → use SEEK". Eliminates ambiguity.
- **10 few-shot examples** in system prompt covering all major scenarios.
- **`/debug/ollama` endpoint** (`GET` and `POST`) — test raw Ollama responses
  without running the bot. See `http://localhost:8000/debug/ollama`.
- **200+ automated tests** (`backend/tests/test_llm_quality.py`):
  - 29 fast unit tests: `TestSituationAnalysis`, `TestPromptBuilder`, `TestExtractJson`
  - 170+ live LLM tests: SEEK vs MINE correctness, never-wrong-block enforcement,
    25-run consistency, explore goals, hostile mob handling

### Changed
- **System prompt** completely rewritten with `STRICT RULES` block:
  - NEVER mine a block not in `nearby_blocks`
  - NEVER mine grass/dirt when goal is wood
  - Use SEEK when target not nearby
- **`MINE` description** updated: "ONLY use if block IS in nearby_blocks"
- **`SEEK` description** added: "Search the world for block — use when NOT nearby"

---

## [2.0.0] — 2026-05-16 — *Phase 2: LLM Autonomous Agent*

### Added
- **`planner_client.js`** — autonomous planning loop, calls `/plan` every 5s.
  Dispatches LLM actions: `MOVE`, `MINE`, `CRAFT`, `IDLE`, `CHAT`, `GOTO`, `STOP`.
- **`ollama_client.py`** — async Ollama HTTP client with blocking + streaming modes.
- **`prompt_builder.py`** — builds system + user messages from bot state + memories.
- **`tool_registry.py`** — all LLM-callable actions with descriptions and param schemas.
- **`memory.py`** — ChromaDB vector memory (RAG) with episode storage.
- **`knowledge_seeder.py`** — seeds 150 Minecraft knowledge entries into ChromaDB.
- **`recipe_advisor.py`** — hints at craftable items from current inventory.
- **`session_logger.py`** — logs every (state, action, outcome) for future fine-tuning.
- **`!goal <text>`** command — sets bot's current goal and syncs to backend.
- **`!planner on/off/status`** commands — control the autonomous loop.
- **`!planer off`** — typo tolerance (common mistake).
- **Ollama GPU warmup** on backend startup — model pre-loaded into VRAM to avoid
  cold-start timeouts on the first planning tick.
- **10s startup delay** in planner before first tick — ensures backend is ready.
- **Goal syncing** — `setGoal()` in bot POSTs to `/goal` backend endpoint.
- **`/goal` API endpoint** — update current agent goal via REST.
- **`/logs/stats`** endpoint — breakdown of LLM vs fallback ratio, actions by type.
- **`/memory/search`** endpoint — debug RAG retrieval.
- **`/memory/stats`** endpoint — ChromaDB memory count.
- **`/history`** endpoint — last 20 planned actions with source (llm/fallback).

### Changed
- Switched model: `mistral:7b-instruct` → `llama3.2:3b`
  (fits in 4GB VRAM, ~3s inference vs >25s)
- `PLAN_TIMEOUT` increased: `20s → 60s` to accommodate GPU loading buffers.
- `stop()` now cancels active pathfinding via `actions.stopMoving()`.
- Bot no longer sets default goal on spawn — user must type `!goal <text>`.
- Bot no longer auto-starts planner — user must type `!planner on`.
- Bot spawns near the first real player found in the server.

---

## [1.0.0] — 2026-05-15 — *Phase 1: Foundation*

### Added
- Mineflayer bot connects to Minecraft Java server.
- `state_extractor.js` — builds Observation JSON (position, health, inventory,
  nearby blocks/entities, time, weather).
- `actions.js` — `follow`, `goTo`, `goNear`, `mine`, `mineNearest`, `chat`.
- `commands.js` — `!help`, `!status`, `!state`, `!inventory`, `!follow`,
  `!stop`, `!goto`, `!mine`, `!say`.
- `logger.js` — coloured console output with `[INFO]`, `[WARN]`, `[ERROR]`,
  `[STATE]`, `[CHAT]` levels.
- FastAPI backend with `/health`, `/state`, `/plan` (fallback only), `/history`.
- Unit tests: 21 bot tests (`test_state.js`), 12 backend tests (`test_backend.py`).
- Auto-reconnect on kick/disconnect.
- Heartbeat state log every 30s.
