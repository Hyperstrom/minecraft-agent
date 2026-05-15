"""
ollama_client.py — Async HTTP client for Ollama REST API with retry + fallback.
"""

import logging
from typing import Any, Dict, List, Optional

import httpx

from config import settings

logger = logging.getLogger("mineagent.ollama")

TIMEOUT   = 30.0   # seconds per request
MAX_TOKENS = 256   # keep responses short


async def chat(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.1,
) -> Optional[str]:
    """
    POST to Ollama /api/chat. Returns assistant reply text, or None on failure.
    Low temperature (0.1) for deterministic, structured JSON output.
    """
    model = model or settings.ollama_model
    payload = {
        "model":   model,
        "messages": messages,
        "stream":  False,
        "options": {
            "temperature": temperature,
            "num_predict": MAX_TOKENS,
            "stop": ["\n\n", "```"],   # stop after JSON block
        },
    }

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.post(
                f"{settings.ollama_url}/api/chat",
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]

    except httpx.ConnectError:
        logger.error("Ollama not reachable at %s — is `ollama serve` running?", settings.ollama_url)
    except httpx.TimeoutException:
        logger.error("Ollama timed out after %ss", TIMEOUT)
    except httpx.HTTPStatusError as e:
        logger.error("Ollama HTTP %s: %s", e.response.status_code, e.response.text[:200])
    except (KeyError, Exception) as e:
        logger.error("Unexpected Ollama error: %s", e)

    return None


async def is_available() -> bool:
    """Return True if Ollama is running and the configured model is loaded."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.ollama_url}/api/tags")
            if resp.status_code != 200:
                return False
            models = [m.get("name", "") for m in resp.json().get("models", [])]
            return any(settings.ollama_model in name for name in models)
    except Exception:
        return False
