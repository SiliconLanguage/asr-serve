# SPDX-License-Identifier: BSD-2-Clause-Patent
"""
Dynamo Worker Base Class – Prefill and Decode roles.

Each worker registers with the Dynamo cluster, accepts requests from the
Smart Router, and processes its assigned phase (prefill or decode).

Prefill workers
  - Run the ASR encoder (e.g. Whisper encoder / CTC frontend).
  - Write the resulting KV-cache tensors to VRAM.
  - Return a KVCacheDescriptor handle to the router.

Decode workers
  - Receive a KVCacheDescriptor handle via the router.
  - Load the KV-cache tensor via NIXL RDMA.
  - Run autoregressive token decoding.
  - Return the transcript string.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any

from router import ASRRequest, RequestPhase, WorkerEndpoint, WorkerRegistry
from nixl_rdma import KVCacheDescriptor, NixlTransferManager

logger = logging.getLogger(__name__)


# ─── Worker Configuration ─────────────────────────────────────────────────────

@dataclass
class WorkerConfig:
    worker_id:      str   = ""
    role:           RequestPhase = RequestPhase.PREFILL
    listen_address: str   = "0.0.0.0:50051"
    cuda_device:    int   = 0
    rdma_device:    str   = "mlx5_0"
    model_path:     str   = "/models/whisper-large-v3"
    max_batch_size: int   = 8
    kv_ttl_ms:      int   = 500

    def __post_init__(self) -> None:
        if not self.worker_id:
            self.worker_id = f"{self.role.value}-{uuid.uuid4().hex[:8]}"


# ─── Base Worker ──────────────────────────────────────────────────────────────

class DynamoWorker:
    """
    Base class for Dynamo prefill/decode workers.

    Subclass and override ``_process_prefill`` or ``_process_decode``.
    """

    def __init__(self, config: WorkerConfig,
                 registry: WorkerRegistry | None = None) -> None:
        self.config   = config
        self._registry = registry
        self._nixl     = NixlTransferManager(rdma_device=config.rdma_device)
        self._running  = False
        self._request_count = 0

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Initialise NIXL and register with the cluster registry."""
        await self._nixl.start()
        if self._registry is not None:
            self._registry.register(WorkerEndpoint(
                worker_id=self.config.worker_id,
                role=self.config.role,
                address=self.config.listen_address,
            ))
        self._running = True
        logger.info("worker %s (%s) started on device %d",
                    self.config.worker_id,
                    self.config.role.value,
                    self.config.cuda_device)

    async def stop(self) -> None:
        self._running = False
        if self._registry is not None:
            self._registry.deregister(self.config.worker_id)
        await self._nixl.stop()
        logger.info("worker %s stopped", self.config.worker_id)

    # ── Request Dispatch ──────────────────────────────────────────────────────

    async def handle_request(self, request: ASRRequest) -> Any:
        """Dispatch the request to the appropriate processing phase."""
        if request.phase == RequestPhase.PREFILL:
            return await self._process_prefill(request)
        return await self._process_decode(request)

    # ── Phase Implementations (override in subclasses) ────────────────────────

    async def _process_prefill(self, request: ASRRequest) -> KVCacheDescriptor:
        """
        Run the ASR encoder and write the KV cache to VRAM.

        Returns a KVCacheDescriptor whose handle is forwarded to a decode worker.
        """
        raise NotImplementedError

    async def _process_decode(self, request: ASRRequest) -> str:
        """
        Load the KV cache via NIXL RDMA and run autoregressive decoding.

        Returns the ASR transcript string.
        """
        raise NotImplementedError

    # ── Health / Telemetry ────────────────────────────────────────────────────

    @property
    def load(self) -> float:
        """Approximate load in [0, 1] based on active requests."""
        # Placeholder; production code reads GPU utilisation from NVML.
        return min(self._request_count / self.config.max_batch_size, 1.0)


# ─── Stub Prefill Worker ──────────────────────────────────────────────────────

class PrefillWorker(DynamoWorker):
    """
    Prefill worker stub – replace model loading / inference with real code.
    """

    def __init__(self, config: WorkerConfig,
                 registry: WorkerRegistry | None = None) -> None:
        config.role = RequestPhase.PREFILL
        super().__init__(config, registry)
        self._model: Any = None  # load whisper encoder here

    async def start(self) -> None:
        await super().start()
        # TODO: load encoder model onto self.config.cuda_device
        logger.info("prefill worker: encoder model placeholder loaded")

    async def _process_prefill(self, request: ASRRequest) -> KVCacheDescriptor:
        self._request_count += 1
        t0 = time.monotonic()

        # TODO: run real encoder forward pass; write KV tensor to VRAM
        # Stub: allocate a fake device pointer and register with NIXL.
        kv_ptr        = 0x8000_0000 + id(request) % 0x1000_0000
        kv_tensor_sz  = 4 * 1024 * 1024  # 4 MiB placeholder

        kv_ttl = int(request.agent_hints.get("kv_ttl_ms",
                                              self.config.kv_ttl_ms))
        descriptor = await self._nixl.register_tensor(
            device=self.config.cuda_device,
            ptr=kv_ptr,
            bytes_size=kv_tensor_sz,
            shape=(32, 1500, 64),   # (layers, seq_len, head_dim)
            dtype="bfloat16",
            ttl_ms=kv_ttl,
        )

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.debug("prefill %s done in %.1f ms  kv=%s",
                     request.request_id, elapsed_ms, descriptor.handle)
        self._request_count -= 1
        return descriptor


# ─── Stub Decode Worker ───────────────────────────────────────────────────────

class DecodeWorker(DynamoWorker):
    """
    Decode worker stub – replace RDMA load / autoregressive decoding with
    real vLLM / NVIDIA Triton inference code.
    """

    def __init__(self, config: WorkerConfig,
                 registry: WorkerRegistry | None = None) -> None:
        config.role = RequestPhase.DECODE
        super().__init__(config, registry)
        self._model: Any = None  # load decoder model here

    async def start(self) -> None:
        await super().start()
        # TODO: load decoder model onto self.config.cuda_device
        logger.info("decode worker: decoder model placeholder loaded")

    async def _process_decode(self, request: ASRRequest) -> str:
        if request.kv_handle is None:
            raise ValueError("decode request missing kv_handle")

        self._request_count += 1
        t0 = time.monotonic()

        # Reconstruct the KVCacheDescriptor from the handle.
        # In production, the prefill worker serialises the full descriptor
        # and sends it via the Dynamo message bus.
        descriptor = self._nixl._descriptors.get(request.kv_handle)
        if descriptor is None:
            raise KeyError(f"KV handle not found: {request.kv_handle}")

        # Transfer VRAM tensor to this worker's device via NIXL RDMA.
        dst_ptr = await self._nixl.transfer(descriptor,
                                             dst_device=self.config.cuda_device)

        # TODO: run real autoregressive decoding using dst_ptr
        # Stub: return a placeholder transcript.
        transcript = f"[stub transcript for {request.request_id}]"

        elapsed_ms = (time.monotonic() - t0) * 1000
        logger.debug("decode %s done in %.1f ms  ptr=0x%X",
                     request.request_id, elapsed_ms, dst_ptr)
        self._request_count -= 1
        return transcript


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Dynamo ASR Worker")
    parser.add_argument("--role",           choices=["prefill", "decode"],
                        default="prefill")
    parser.add_argument("--listen",         default="0.0.0.0:50051")
    parser.add_argument("--cuda-device",    type=int, default=0)
    parser.add_argument("--rdma-device",    default="mlx5_0")
    parser.add_argument("--model-path",     default="/models/whisper-large-v3")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = WorkerConfig(
        role=RequestPhase(args.role),
        listen_address=args.listen,
        cuda_device=args.cuda_device,
        rdma_device=args.rdma_device,
        model_path=args.model_path,
    )

    WorkerCls = PrefillWorker if cfg.role == RequestPhase.PREFILL else DecodeWorker
    worker    = WorkerCls(cfg)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def _run() -> None:
        await worker.start()
        stop_evt = asyncio.Event()

        def _sig_handler(*_: Any) -> None:
            stop_evt.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _sig_handler)

        logger.info("worker %s running – send SIGINT to stop", worker.config.worker_id)
        await stop_evt.wait()
        await worker.stop()

    loop.run_until_complete(_run())
