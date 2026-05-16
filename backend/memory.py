"""
memory.py — ChromaDB long-term memory: store episodes, retrieve top-k by similarity.
Gracefully disabled if chromadb is not installed.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger("mineagent.memory")

CHROMA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")

_collection = None
_available  = False


def _init():
    """Lazy-initialise ChromaDB. Called on first use."""
    global _collection, _available
    if _collection is not None:
        return _collection

    try:
        import chromadb  # noqa: F401 — optional dependency

        os.makedirs(CHROMA_PATH, exist_ok=True)
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = client.get_or_create_collection(
            name="mineagent_memory",
            metadata={"hnsw:space": "cosine"},
        )
        _available = True
        logger.info("ChromaDB ready | path=%s | memories=%d", CHROMA_PATH, _collection.count())
    except ImportError:
        logger.warning("chromadb not installed — memory disabled. pip install chromadb")
    except Exception as e:
        logger.error("ChromaDB init error: %s", e)

    return _collection


def is_available() -> bool:
    _init()
    return _available


def store(state: Dict[str, Any], action: Dict[str, Any]) -> bool:
    """Embed and store one (state, action) episode."""
    col = _init()
    if col is None:
        return False

    try:
        from datetime import datetime, timezone
        ts  = datetime.now(timezone.utc).isoformat()
        inv = state.get("inventory", {})
        pos = state.get("player", {}).get("position", {})

        # Human-readable text used for embedding
        text = (
            f"Goal: {state.get('goal','none')} | "
            f"Action: {action.get('action','?')}({json.dumps(action.get('params',{}))}) | "
            f"Reason: {action.get('reasoning','')} | "
            f"Inv: {json.dumps(inv)} | Pos: {json.dumps(pos)}"
        )

        col.add(
            documents=[text],
            metadatas=[{"goal": str(state.get("goal")), "action": action.get("action"), "ts": ts}],
            ids=[f"mem_{ts.replace(':','-').replace('+','p')}"],
        )
        return True
    except Exception as e:
        logger.error("Memory store error: %s", e)
        return False


def retrieve(query: str, n: int = 3) -> List[str]:
    """Return top-n memory strings most similar to query."""
    col = _init()
    if col is None or col.count() == 0:
        return []

    try:
        results = col.query(query_texts=[query], n_results=min(n, col.count()))
        return results["documents"][0] if results["documents"] else []
    except Exception as e:
        logger.error("Memory retrieve error: %s", e)
        return []


def count() -> int:
    col = _init()
    if col is None:
        return 0
    try:
        return col.count()
    except Exception:
        return 0


MAX_MEMORIES = 500   # hard cap to prevent unbounded growth

def prune(keep: int = MAX_MEMORIES) -> int:
    """
    Remove oldest memories if collection exceeds `keep` entries.
    Returns number of entries deleted.
    Seeded knowledge (ids starting with 'seed_') is never pruned.
    """
    col = _init()
    if col is None:
        return 0

    total = col.count()
    if total <= keep:
        return 0

    try:
        # Get all episode memories (not seeded knowledge)
        results = col.get(where={"source": {"$ne": "seed"}})
        ids     = results.get("ids", [])
        metas   = results.get("metadatas", [])

        # Sort by timestamp ascending (oldest first)
        paired  = sorted(zip(ids, metas), key=lambda x: x[1].get("ts", ""))
        to_delete = len(ids) - keep
        if to_delete <= 0:
            return 0

        delete_ids = [pid for pid, _ in paired[:to_delete]]
        col.delete(ids=delete_ids)
        logger.info("Pruned %d old memories (total now: %d)", len(delete_ids), col.count())
        return len(delete_ids)

    except Exception as e:
        logger.error("Memory prune error: %s", e)
        return 0
