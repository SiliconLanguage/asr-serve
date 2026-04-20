# SPDX-License-Identifier: BSD-2-Clause-Patent
"""
Dynamo Smart Router – Prefill-Decode Disaggregation (PDD) for ASR.

Routes incoming ASR requests to specialised prefill workers (GPU compute-
heavy) or decode workers (VRAM memory-bound) based on request state and
current cluster load.  Integrates with the NVIDIA Dynamo role-based
orchestration engine.

References
----------
- NVIDIA Dynamo: https://github.com/ai-dynamo/dynamo
- Prefill-Decode Disaggregation: https://arxiv.org/abs/2401.09670
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)


# ─── Request Model ────────────────────────────────────────────────────────────

class RequestPhase(str, Enum):
    PREFILL = "prefill"
    DECODE  = "decode"


@dataclass
class ASRRequest:
    """A single ASR inference request travelling through the PDD pipeline."""
    request_id:   str
    audio_bytes:  bytes
    phase:        RequestPhase         = RequestPhase.PREFILL
    # KV-cache token handle written by the prefill worker and consumed
    # by the decode worker via NIXL RDMA.
    kv_handle:    str | None           = None
    # Agent hints attached by the Agentic Factory (see agentic-factory/).
    agent_hints:  dict[str, Any]       = field(default_factory=dict)
    enqueue_time: float                = field(default_factory=time.monotonic)


# ─── Worker Registry ──────────────────────────────────────────────────────────

@dataclass
class WorkerEndpoint:
    worker_id:  str
    role:       RequestPhase   # PREFILL or DECODE
    address:    str            # host:port
    load:       float = 0.0    # 0.0 (idle) – 1.0 (saturated)
    healthy:    bool  = True


class WorkerRegistry:
    """In-memory registry of prefill/decode worker endpoints."""

    def __init__(self) -> None:
        self._workers: dict[str, WorkerEndpoint] = {}

    def register(self, endpoint: WorkerEndpoint) -> None:
        self._workers[endpoint.worker_id] = endpoint
        logger.info("registered worker %s role=%s addr=%s",
                    endpoint.worker_id, endpoint.role.value, endpoint.address)

    def deregister(self, worker_id: str) -> None:
        self._workers.pop(worker_id, None)

    def least_loaded(self, role: RequestPhase) -> WorkerEndpoint | None:
        candidates = [
            w for w in self._workers.values()
            if w.role == role and w.healthy
        ]
        if not candidates:
            return None
        return min(candidates, key=lambda w: w.load)

    def update_load(self, worker_id: str, load: float) -> None:
        if worker_id in self._workers:
            self._workers[worker_id].load = load

    def mark_unhealthy(self, worker_id: str) -> None:
        if worker_id in self._workers:
            self._workers[worker_id].healthy = False
            logger.warning("worker %s marked unhealthy", worker_id)


# ─── Smart Router ─────────────────────────────────────────────────────────────

class SmartRouter:
    """
    Dynamo Smart Router for Prefill-Decode Disaggregation.

    The router implements a two-hop scheduling strategy:
      1. A fresh ASR request is dispatched to the *least-loaded prefill worker*.
         The prefill worker processes the audio encoder and writes the resulting
         KV-cache tensor to shared GPU memory, returning a ``kv_handle``.
      2. The request (now carrying ``kv_handle``) is dispatched to the
         *least-loaded decode worker*, which loads the KV cache via NIXL RDMA
         and runs autoregressive token decoding.

    Agent hints (priority, kv_ttl_ms) from the Agentic Factory are honoured
    during worker selection and cache TTL management.
    """

    def __init__(self, registry: WorkerRegistry) -> None:
        self._registry    = registry
        self._in_flight:  dict[str, ASRRequest] = {}
        self._stats       = RouterStats()

    async def route(self, request: ASRRequest) -> str:
        """
        Route *request* through the PDD pipeline and return the transcript.
        """
        self._in_flight[request.request_id] = request
        try:
            transcript = await self._pdd_pipeline(request)
        finally:
            self._in_flight.pop(request.request_id, None)
        return transcript

    async def _pdd_pipeline(self, request: ASRRequest) -> str:
        # ── Phase 1: Prefill ─────────────────────────────────────────────────
        prefill_worker = self._select_worker(RequestPhase.PREFILL, request)
        if prefill_worker is None:
            raise RuntimeError("No healthy prefill workers available")

        logger.debug("routing %s → prefill worker %s",
                     request.request_id, prefill_worker.worker_id)
        t0 = time.monotonic()
        kv_handle = await self._call_prefill(prefill_worker, request)
        self._stats.record_prefill_latency(time.monotonic() - t0)

        # ── Phase 2: Decode ──────────────────────────────────────────────────
        request.phase     = RequestPhase.DECODE
        request.kv_handle = kv_handle

        decode_worker = self._select_worker(RequestPhase.DECODE, request)
        if decode_worker is None:
            raise RuntimeError("No healthy decode workers available")

        logger.debug("routing %s → decode worker %s",
                     request.request_id, decode_worker.worker_id)
        t1 = time.monotonic()
        transcript = await self._call_decode(decode_worker, request)
        self._stats.record_decode_latency(time.monotonic() - t1)

        self._stats.requests_completed += 1
        return transcript

    def _select_worker(self,
                       role: RequestPhase,
                       request: ASRRequest) -> WorkerEndpoint | None:
        """
        Select the best worker for the given role, honouring agent hints.

        Agent hint ``priority`` (int 0–9) biases selection toward the worker
        with the most headroom when priority > 5.
        """
        priority = int(request.agent_hints.get("priority", 5))
        candidate = self._registry.least_loaded(role)

        if candidate is None:
            return None

        # High-priority requests: only assign if worker has headroom.
        if priority > 5 and candidate.load > 0.8:
            logger.warning(
                "high-priority request %s: all %s workers loaded (%.0f%%)",
                request.request_id, role.value, candidate.load * 100,
            )
        return candidate

    # ── Stub RPC calls ────────────────────────────────────────────────────────
    # Replace with actual Dynamo gRPC / NATS transport.

    async def _call_prefill(self,
                             worker: WorkerEndpoint,
                             request: ASRRequest) -> str:
        """Call prefill worker; returns KV-cache handle string."""
        # TODO: serialise request and call worker.address via Dynamo transport
        await asyncio.sleep(0)   # yield to event loop
        return f"kv://{worker.worker_id}/{request.request_id}"

    async def _call_decode(self,
                            worker: WorkerEndpoint,
                            request: ASRRequest) -> str:
        """Call decode worker; returns ASR transcript."""
        # TODO: serialise request (with kv_handle) and call worker via Dynamo
        await asyncio.sleep(0)
        return ""


# ─── Router Statistics ────────────────────────────────────────────────────────

@dataclass
class RouterStats:
    requests_completed:   int   = 0
    prefill_latencies_ms: list  = field(default_factory=list)
    decode_latencies_ms:  list  = field(default_factory=list)

    def record_prefill_latency(self, seconds: float) -> None:
        self.prefill_latencies_ms.append(seconds * 1000)

    def record_decode_latency(self, seconds: float) -> None:
        self.decode_latencies_ms.append(seconds * 1000)

    def p99_prefill_ms(self) -> float:
        return _p99(self.prefill_latencies_ms)

    def p99_decode_ms(self) -> float:
        return _p99(self.decode_latencies_ms)


def _p99(values: list) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = max(0, int(len(sorted_v) * 0.99) - 1)
    return sorted_v[idx]


# ─── CLI entry point ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dynamo Smart Router")
    parser.add_argument("--prefill-workers", type=int, default=2)
    parser.add_argument("--decode-workers",  type=int, default=4)
    parser.add_argument("--nixl-rdma-device", default="mlx5_0")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    registry = WorkerRegistry()

    # Register stub workers (replace with real Dynamo service-discovery).
    for i in range(args.prefill_workers):
        registry.register(WorkerEndpoint(
            worker_id=f"prefill-{i}",
            role=RequestPhase.PREFILL,
            address=f"10.0.1.{10 + i}:50051",
        ))
    for i in range(args.decode_workers):
        registry.register(WorkerEndpoint(
            worker_id=f"decode-{i}",
            role=RequestPhase.DECODE,
            address=f"10.0.2.{10 + i}:50051",
        ))

    router = SmartRouter(registry)
    logger.info("Smart Router ready – prefill=%d decode=%d nixl=%s",
                args.prefill_workers, args.decode_workers,
                args.nixl_rdma_device)

    # In production, replace the loop below with the Dynamo gRPC server.
    async def _demo() -> None:
        req = ASRRequest(
            request_id="demo-001",
            audio_bytes=b"\x00" * 640,
            agent_hints={"priority": 7, "kv_ttl_ms": 500},
        )
        transcript = await router.route(req)
        logger.info("transcript: %r", transcript)
        logger.info("stats: prefill_p99=%.1f ms  decode_p99=%.1f ms",
                    router._stats.p99_prefill_ms(),
                    router._stats.p99_decode_ms())

    asyncio.run(_demo())
