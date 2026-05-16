"""
ollama_client.py — Async HTTP client for Ollama REST API.
Supports both blocking and streaming modes with fallback.
"""

import json
import logging
from typing import AsyncIterator, Dict, List, Optional

import httpx

from config import settings

logger = logging.getLogger("mineagent.ollama")

TIMEOUT    = 30.0
MAX_TOKENS = 256


async def chat(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.1,
) -> Optional[str]:
    """
    Blocking Ollama /api/chat call.
    Returns full assistant reply text, or None on any failure.
    """
    model = model or settings.ollama_model
    payload = {
        "model":    model,
        "messages": messages,
        "stream":   False,
        "options": {
            "temperature": temperature,
            "num_predict": MAX_TOKENS,
            "stop":        ["\n\n", "```"],
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
    except Exception as e:
        logger.error("Unexpected Ollama error: %s", e)

    return None


async def chat_stream(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.1,
) -> AsyncIterator[str]:
    """
    Streaming Ollama /api/chat call.
    Yields token chunks as they arrive. Use for real-time display.
    Falls back gracefully if streaming fails.
    """
    model = model or settings.ollama_model
    payload = {
        "model":    model,
        "messages": messages,
        "stream":   True,
        "options": {
            "temperature": temperature,
            "num_predict": MAX_TOKENS,
        },
    }

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            async with client.stream(
                "POST",
                f"{settings.ollama_url}/api/chat",
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                        token = chunk.get("message", {}).get("content", "")
                        if token:
                            yield token
                        if chunk.get("done"):
                            break
                    except json.JSONDecodeError:
                        continue

    except httpx.ConnectError:
        logger.error("Ollama streaming: not reachable at %s", settings.ollama_url)
    except Exception as e:
        logger.error("Ollama streaming error: %s", e)


async def chat_stream_full(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.1,
) -> Optional[str]:
    """
    Streaming call that reassembles full text — faster time-to-first-token
    than blocking, but returns the complete string when done.
    """
    parts = []
    async for token in chat_stream(messages, model, temperature):
        parts.append(token)
    return "".join(parts) if parts else None


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
