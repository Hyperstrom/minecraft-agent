"""
main.py  —  MineAgent Backend API  (Phase 2: Complete)
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

import ollama_client
import memory as mem
import session_logger as slog
from config import settings
from prompt_builder import build_messages, build_correction_messages, extract_json
from tool_registry import TOOL_NAMES
from memory import prune as prune_memory, MAX_MEMORIES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("mineagent")

app = FastAPI(
    title="MineAgent Backend",
    description="Autonomous Minecraft AI — Phase 2 Complete",
    version="0.2.1",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Models ────────────────────────────────────────────────────────

class PlayerInfo(BaseModel):
    username:   Optional[str]   = None
    position:   Optional[Dict]  = None
    health:     Optional[float] = None
    food:       Optional[float] = None
    saturation: Optional[float] = None

class Environment(BaseModel):
    time_of_day: Optional[str] = None
    weather:     Optional[str] = None
    dimension:   Optional[str] = None

class ObservationState(BaseModel):
    player:          PlayerInfo
    inventory:       Dict[str, int]       = {}
    nearby_entities: List[Dict[str, Any]] = []
    nearby_blocks:   List[Dict[str, Any]] = []
    environment:     Environment          = Environment()
    goal:            Optional[str]        = None
    timestamp:       str

class ActionResponse(BaseModel):
    action:    str
    params:    Dict[str, Any] = {}
    reasoning: Optional[str]  = None
    source:    str            = "llm"

class GoalRequest(BaseModel):
    goal: str

class MemorySearchRequest(BaseModel):
    query: str
    n:     int = 3

# ── Runtime state ─────────────────────────────────────────────────
latest_state:   Optional[Dict] = None
action_history: List[Dict]     = []
current_goal:   str            = "survive and explore"

# ── Startup ───────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    from knowledge_seeder import seed_all
    logger.info("Seeding knowledge base...")
    seed_all()
    logger.info("Backend ready. Ollama model: %s", settings.ollama_model)

# ── Meta endpoints ────────────────────────────────────────────────

@app.get("/", tags=["meta"])
async def root():
    return {"name": "MineAgent Backend", "version": "0.2.1", "timestamp": _now()}

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

@app.get("/health", tags=["meta"])
async def health():
    return {"status": "healthy", "timestamp": _now()}

@app.get("/status", tags=["meta"])
async def status():
    ollama_ok = await ollama_client.is_available()
    return {
        "ollama":    {"available": ollama_ok, "url": settings.ollama_url, "model": settings.ollama_model},
        "memory":    {"available": mem.is_available(), "count": mem.count()},
        "sessions":  slog.get_stats(),
        "goal":      current_goal,
        "timestamp": _now(),
    }

# ── Goal ──────────────────────────────────────────────────────────

@app.get("/goal", tags=["agent"])
async def get_goal():
    return {"goal": current_goal}

@app.post("/goal", tags=["agent"])
async def set_goal(body: GoalRequest):
    global current_goal
    current_goal = body.goal
    logger.info("Goal updated: %s", current_goal)
    return {"goal": current_goal, "timestamp": _now()}

# ── State ─────────────────────────────────────────────────────────

@app.post("/state", tags=["agent"])
async def receive_state(state: ObservationState):
    global latest_state
    latest_state = state.model_dump()
    logger.info("State | hp=%.0f food=%.0f pos=%s",
                state.player.health or 0, state.player.food or 0,
                state.player.position)
    return {"status": "received", "timestamp": _now()}

@app.get("/state", tags=["agent"])
async def get_state():
    if latest_state is None:
        raise HTTPException(status_code=404, detail="No state received yet")
    return latest_state

# ── Plan ──────────────────────────────────────────────────────────

@app.post("/plan", response_model=ActionResponse, tags=["agent"])
async def plan_action(state: ObservationState):
    goal   = state.goal or current_goal
    action = await llm_planner(state, goal)
    action_history.append({
        "timestamp": _now(),
        "hp":    state.player.health,
        "food":  state.player.food,
        "goal":  goal,
        "action": action.model_dump(),
    })
    logger.info("Action [%s]: %s(%s) — %s", action.source, action.action, action.params, action.reasoning)
    return action

@app.get("/history", tags=["agent"])
async def get_history(limit: int = 20):
    return {"history": action_history[-limit:], "total": len(action_history)}

# ── Memory endpoints ──────────────────────────────────────────────

@app.get("/memory/stats", tags=["memory"])
async def memory_stats():
    return {
        "available": mem.is_available(),
        "count":     mem.count(),
        "timestamp": _now(),
    }

@app.post("/memory/search", tags=["memory"])
async def memory_search(body: MemorySearchRequest):
    """Search ChromaDB for relevant memories (for debugging RAG)."""
    if not mem.is_available():
        raise HTTPException(status_code=503, detail="ChromaDB not available")
    results = mem.retrieve(body.query, n=body.n)
    return {"query": body.query, "results": results, "count": len(results)}

@app.get("/memory/search", tags=["memory"])
async def memory_search_get(q: str = Query(..., description="Search query"), n: int = 3):
    """Search ChromaDB via GET (for easy browser/Postman testing)."""
    if not mem.is_available():
        raise HTTPException(status_code=503, detail="ChromaDB not available")
    results = mem.retrieve(q, n=n)
    return {"query": q, "results": results, "count": len(results)}

@app.post("/memory/prune", tags=["memory"])
async def memory_prune(keep: int = MAX_MEMORIES):
    """Delete oldest episode memories, keeping seeded knowledge intact."""
    deleted = prune_memory(keep=keep)
    return {"deleted": deleted, "remaining": mem.count(), "timestamp": _now()}

# ── Session logs ──────────────────────────────────────────────────

@app.get("/logs/stats", tags=["logs"])
async def log_stats():
    return slog.get_stats()

# ── WebSocket ─────────────────────────────────────────────────────

connected_bots: Dict[str, WebSocket] = {}

@app.websocket("/ws/{bot_id}")
async def websocket_endpoint(websocket: WebSocket, bot_id: str):
    await websocket.accept()
    connected_bots[bot_id] = websocket
    logger.info("Bot WS connected: %s", bot_id)
    try:
        while True:
            raw = await websocket.receive_text()
            msg = json.loads(raw)
            if msg.get("type") == "state":
                global latest_state
                latest_state = msg.get("data", {})
                await websocket.send_json({"type": "ack", "timestamp": _now()})
    except WebSocketDisconnect:
        connected_bots.pop(bot_id, None)
        logger.info("Bot WS disconnected: %s", bot_id)

# ── LLM Planner ───────────────────────────────────────────────────

async def llm_planner(state: ObservationState, goal: str) -> ActionResponse:
    state_dict = state.model_dump()

    # 1. Retrieve relevant memories
    query     = f"Goal:{goal} HP:{state.player.health} Inv:{json.dumps(state.inventory)}"
    memories  = mem.retrieve(query, n=3)

    # 2. Build prompt
    messages  = build_messages(state_dict, goal, memories)

    # 3. Call Ollama
    raw = await ollama_client.chat(messages)

    if raw is None:
        logger.warning("Ollama unavailable — fallback planner")
        fb = simple_planner(state)
        slog.log_interaction(state_dict, goal, memories, None, fb.model_dump(), "fallback")
        return fb

    # 4. Parse JSON
    parsed = extract_json(raw)

    # 5. Retry with correction prompt if invalid
    if parsed is None or parsed.get("action") not in TOOL_NAMES:
        logger.warning("Bad LLM JSON, retrying. Raw: %s", raw[:120])
        corrected_msgs = build_correction_messages(messages, raw)
        raw2   = await ollama_client.chat(corrected_msgs)
        parsed = extract_json(raw2 or "")

    # 6. Final fallback
    if parsed is None or parsed.get("action") not in TOOL_NAMES:
        logger.error("LLM unusable after retry — falling back")
        fb = simple_planner(state)
        slog.log_interaction(state_dict, goal, memories, raw, fb.model_dump(), "fallback")
        return fb

    action = ActionResponse(
        action    = parsed["action"],
        params    = parsed.get("params", {}),
        reasoning = parsed.get("reasoning", ""),
        source    = "llm",
    )

    # 7. Store episode in memory + log
    mem.store(state_dict, action.model_dump())
    slog.log_interaction(state_dict, goal, memories, raw, action.model_dump(), "llm")

    return action


def simple_planner(state: ObservationState) -> ActionResponse:
    """Rule-based fallback when Ollama is unavailable."""
    hp   = state.player.health or 20
    food = state.player.food   or 20
    inv  = state.inventory

    if hp < 8:
        return ActionResponse(action="CHAT", params={"message": "Health critical!"},
                              reasoning=f"HP={hp}/20", source="fallback")
    if food < 6:
        return ActionResponse(action="SEEK", params={"target": "food"},
                              reasoning=f"Food={food}/20", source="fallback")

    wood = ["oak_log","birch_log","spruce_log","jungle_log","dark_oak_log"]
    if sum(inv.get(w, 0) for w in wood) == 0:
        near_log = next((b["name"] for b in state.nearby_blocks if b["name"].endswith("_log")), None)
        if near_log:
            return ActionResponse(action="MINE", params={"block": near_log},
                                  reasoning="No wood in inventory", source="fallback")

    return ActionResponse(action="IDLE", params={}, reasoning="No urgent need", source="fallback")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.backend_host, port=settings.backend_port,
                reload=True, log_level="info")
