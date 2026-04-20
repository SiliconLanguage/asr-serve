# SPDX-License-Identifier: BSD-2-Clause-Patent
"""
Agent Hints Manager.

Maintains and propagates routing hints that the Agentic Factory attaches
to every ASR inference request.  Hints are consumed by:
  - The Dynamo Smart Router (priority → worker selection).
  - The vLLM inference worker (kv_ttl_ms → KV-cache eviction policy).
  - The Rust shim (priority → batch ordering).

Hint Schema
-----------
  priority   : int  0–9  – routing priority; higher = more headroom reserved.
  kv_ttl_ms  : int       – KV-cache TTL in milliseconds.
  model_id   : str       – target model (overrides cluster default).
  debug      : bool      – attach verbose trace headers to the request.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ─── Hint Profile ─────────────────────────────────────────────────────────────

@dataclass
class HintProfile:
    """A named set of hints associated with a model evaluation outcome."""
    model_id:   str
    priority:   int   = 5
    kv_ttl_ms:  int   = 500
    debug:      bool  = False
    updated_at: float = field(default_factory=time.monotonic)

    def to_headers(self) -> dict[str, str]:
        """Serialise hints as HTTP headers for injection into requests."""
        headers = {
            "X-Agent-Priority":   str(self.priority),
            "X-Agent-KV-TTL-Ms":  str(self.kv_ttl_ms),
            "X-Agent-Model-ID":   self.model_id,
        }
        if self.debug:
            headers["X-Agent-Debug"] = "true"
        return headers

    def to_dict(self) -> dict[str, Any]:
        return {
            "priority":  self.priority,
            "kv_ttl_ms": self.kv_ttl_ms,
            "model_id":  self.model_id,
            "debug":     self.debug,
        }


# ─── Hints Manager ────────────────────────────────────────────────────────────

class AgentHintsManager:
    """
    Thread-safe manager for Agent Hints.

    The EvaluationLoop and NATAgent write hint profiles; the Rust shim and
    HTTP server read the current active profile to decorate requests.
    """

    def __init__(self, default_model_id: str = "whisper-large-v3") -> None:
        self._lock    = threading.RLock()
        self._active  = HintProfile(model_id=default_model_id)
        self._history: list[HintProfile] = []

    # ── Write API ─────────────────────────────────────────────────────────────

    def set_profile(self, profile: HintProfile) -> None:
        """Replace the active hint profile and record history."""
        with self._lock:
            self._history.append(self._active)
            if len(self._history) > 100:
                self._history.pop(0)
            self._active = profile
            logger.info(
                "agent hints updated: model=%s priority=%d kv_ttl=%d ms",
                profile.model_id, profile.priority, profile.kv_ttl_ms,
            )

    def set_hint(self, key: str, value: Any) -> None:
        """Patch a single hint key in the active profile."""
        with self._lock:
            # Clone the current profile with the updated field.
            current = self._active
            kwargs = current.to_dict()
            kwargs[key] = value
            updated = HintProfile(
                model_id  = kwargs.get("model_id",  current.model_id),
                priority  = int(kwargs.get("priority",  current.priority)),
                kv_ttl_ms = int(kwargs.get("kv_ttl_ms", current.kv_ttl_ms)),
                debug     = bool(kwargs.get("debug",     current.debug)),
            )
            self.set_profile(updated)

    # ── Read API ──────────────────────────────────────────────────────────────

    @property
    def active(self) -> HintProfile:
        with self._lock:
            return self._active

    def get_headers(self) -> dict[str, str]:
        """Return current hints serialised as HTTP request headers."""
        with self._lock:
            return self._active.to_headers()

    def get_dict(self) -> dict[str, Any]:
        """Return current hints as a plain dict."""
        with self._lock:
            return self._active.to_dict()

    # ── History ───────────────────────────────────────────────────────────────

    def recent_history(self, n: int = 10) -> list[HintProfile]:
        with self._lock:
            return list(self._history[-n:])


# ─── KV-Cache TTL Eviction Scheduler ─────────────────────────────────────────

class TTLEvictionScheduler:
    """
    Tracks outstanding KV-cache handles and schedules eviction when TTL expires.

    Used by the Dynamo orchestration layer to call ``NixlTransferManager.evict``
    at the right time.
    """

    def __init__(self) -> None:
        self._lock: threading.RLock = threading.RLock()
        self._entries: dict[str, tuple[float, Any]] = {}  # handle → (deadline, evict_fn)

    def register(self, handle: str, ttl_ms: int, evict_fn: Any) -> None:
        """Register a KV handle for TTL-based eviction."""
        deadline = time.monotonic() + ttl_ms / 1000.0
        with self._lock:
            self._entries[handle] = (deadline, evict_fn)

    def tick(self) -> list[str]:
        """
        Check for expired handles and invoke their evict callbacks.
        Returns the list of evicted handles.
        """
        now = time.monotonic()
        evicted: list[str] = []
        with self._lock:
            expired = [h for h, (dl, _) in self._entries.items() if now >= dl]
            for handle in expired:
                _, evict_fn = self._entries.pop(handle)
                try:
                    evict_fn(handle)
                    evicted.append(handle)
                except Exception as exc:
                    logger.warning("eviction failed for %s: %s", handle, exc)
        return evicted

    async def run_loop(self, interval_s: float = 0.1) -> None:
        """Async eviction loop – run as a background task."""
        import asyncio
        while True:
            evicted = self.tick()
            if evicted:
                logger.debug("TTL evicted %d KV handles", len(evicted))
            await asyncio.sleep(interval_s)


# ─── Convenience Decorator ────────────────────────────────────────────────────

def with_agent_hints(hints_manager: AgentHintsManager):
    """
    Function decorator that injects current agent hints into the first argument
    if it is an ``aiohttp.ClientSession`` request or similar mapping.

    Usage::

        @with_agent_hints(hints_manager)
        async def transcribe(session, url, audio):
            ...
    """
    import functools

    def decorator(fn: Any) -> Any:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            extra_headers = hints_manager.get_headers()
            # Merge into existing headers kwarg if present.
            existing = kwargs.get("headers", {})
            kwargs["headers"] = {**extra_headers, **existing}
            return await fn(*args, **kwargs)
        return wrapper
    return decorator
