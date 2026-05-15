"""
main.py  —  MineAgent Backend API  (Phase 2: LLM-powered planner)
"""

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

import ollama_client
import memory as mem
from config import settings
from prompt_builder import build_messages, build_correction_messages, extract_json
from tool_registry import TOOL_NAMES

# ── Logging ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("mineagent")

# ── App ───────────────────────────────────────────────────────────
app = FastAPI(
    title="MineAgent Backend",
    description="Autonomous Minecraft AI — Phase 2 (Ollama + RAG)",
    version="0.2.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Pydantic models ───────────────────────────────────────────────

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
    source:    str = "llm"   # "llm" | "fallback" | "correction"

class GoalRequest(BaseModel):
    goal: str

# ── Runtime state ─────────────────────────────────────────────────
latest_state:   Optional[Dict] = None
action_history: List[Dict]     = []
current_goal:   str            = "survive and explore"

# ── Routes ────────────────────────────────────────────────────────

@app.get("/", tags=["meta"])
async def root():
    return {"name": "MineAgent Backend", "version": "0.2.0", "timestamp": _now()}

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

@app.get("/health", tags=["meta"])
async def health():
    return {"status": "healthy", "timestamp": _now()}

@app.get("/status", tags=["meta"])
async def status():
    """Show Ollama availability and memory stats."""
    ollama_ok = await ollama_client.is_available()
    return {
        "ollama": {
            "available": ollama_ok,
            "url":       settings.ollama_url,
            "model":     settings.ollama_model,
        },
        "memory": {
            "available": mem.is_available(),
            "count":     mem.count(),
        },
        "goal":      current_goal,
        "timestamp": _now(),
    }

# ── Goal management ───────────────────────────────────────────────

@app.get("/goal", tags=["agent"])
async def get_goal():
    return {"goal": current_goal}

@app.post("/goal", tags=["agent"])
async def set_goal(body: GoalRequest):
    global current_goal
    current_goal = body.goal
    logger.info("Goal updated: %s", current_goal)
    return {"goal": current_goal, "timestamp": _now()}

# ── State management ──────────────────────────────────────────────

@app.post("/state", tags=["agent"])
async def receive_state(state: ObservationState):
    global latest_state
    latest_state = state.model_dump()
    logger.info("State | hp=%.0f food=%.0f goal=%s", state.player.health or 0, state.player.food or 0, state.goal)
    return {"status": "received", "timestamp": _now()}

@app.get("/state", tags=["agent"])
async def get_state():
    if latest_state is None:
        raise HTTPException(status_code=404, detail="No state received yet")
    return latest_state

# ── Planning ──────────────────────────────────────────────────────

@app.post("/plan", response_model=ActionResponse, tags=["agent"])
async def plan_action(state: ObservationState):
    """
    Core endpoint: given an Observation, return the next Action.
    Pipeline: ChromaDB recall → prompt build → Ollama → JSON parse → fallback.
    """
    goal = state.goal or current_goal
    action = await llm_planner(state, goal)

    record = {
        "timestamp": _now(),
        "hp":    state.player.health,
        "goal":  goal,
        "action": action.model_dump(),
    }
    action_history.append(record)
    logger.info("Action [%s]: %s(%s) — %s", action.source, action.action, action.params, action.reasoning)
    return action

@app.get("/history", tags=["agent"])
async def get_history(limit: int = 20):
    return {"history": action_history[-limit:], "total": len(action_history)}

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
    """
    Full pipeline:
      1. Retrieve ChromaDB memories
      2. Build prompt messages
      3. Call Ollama
      4. Extract + validate JSON
      5. Retry once with correction prompt if JSON is invalid
      6. Fall back to rule-based planner if Ollama unavailable
    """
    state_dict = state.model_dump()

    # 1. Memory retrieval
    memories = mem.retrieve(f"Goal: {goal} | Inventory: {json.dumps(state.inventory)}", n=3)

    # 2. Build prompt
    messages = build_messages(state_dict, goal, memories)

    # 3. Call Ollama
    raw = await ollama_client.chat(messages)

    if raw is None:
        logger.warning("Ollama unavailable — using rule-based fallback")
        fb = simple_planner(state)
        fb.source = "fallback"
        return fb

    # 4. Parse JSON
    parsed = extract_json(raw)

    # 5. Retry with correction if parse failed
    if parsed is None or parsed.get("action") not in TOOL_NAMES:
        logger.warning("Invalid LLM response, retrying: %s", raw[:100])
        correction_msgs = build_correction_messages(messages, raw)
        raw2 = await ollama_client.chat(correction_msgs)
        parsed = extract_json(raw2 or "")

    # 6. Final fallback
    if parsed is None or parsed.get("action") not in TOOL_NAMES:
        logger.error("LLM gave unusable response after retry — falling back")
        fb = simple_planner(state)
        fb.source = "fallback"
        return fb

    action = ActionResponse(
        action    = parsed["action"],
        params    = parsed.get("params", {}),
        reasoning = parsed.get("reasoning", ""),
        source    = "llm",
    )

    # Store episode in memory (fire-and-forget)
    mem.store(state_dict, action.model_dump())

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

    wood  = ["oak_log","birch_log","spruce_log","jungle_log","dark_oak_log"]
    logs  = sum(inv.get(w, 0) for w in wood)
    near_log = next((b["name"] for b in state.nearby_blocks if b["name"].endswith("_log")), None)
    if logs == 0 and near_log:
        return ActionResponse(action="MINE", params={"block": near_log},
                              reasoning="No wood in inventory", source="fallback")

    return ActionResponse(action="IDLE", params={}, reasoning="No urgent need", source="fallback")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.backend_host, port=settings.backend_port,
                reload=True, log_level="info")
