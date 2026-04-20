# SPDX-License-Identifier: BSD-2-Clause-Patent
"""
NIXL RDMA Transfer Manager.

Manages zero-copy VRAM-to-VRAM tensor transfers between prefill and decode
workers using the NVIDIA Inference Transfer Library (NIXL).

NIXL provides a RDMA abstraction over NVLink, InfiniBand, and RoCE fabrics.
The ``NixlTransferManager`` class wraps the NIXL Python bindings and exposes
a clean async API for registering KV-cache tensors and initiating transfers.

References
----------
- NIXL: https://github.com/ai-dynamo/nixl
- NVIDIA Dynamo PDD: https://developer.nvidia.com/blog/disaggregated-serving
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ── Optional NIXL import ─────────────────────────────────────────────────────
# nixl is only available on NVIDIA GPU nodes; fall back to a stub so that the
# module can be imported on CPU-only development machines.
try:
    import nixl  # type: ignore[import-untyped]
    _NIXL_AVAILABLE = True
except ImportError:
    nixl = None  # type: ignore[assignment]
    _NIXL_AVAILABLE = False
    logger.warning("nixl package not found – running in stub mode")


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class KVCacheDescriptor:
    """Describes a KV-cache tensor resident in a prefill worker's VRAM."""
    handle:       str             # Globally unique transfer handle
    src_device:   int             # CUDA device ordinal on the source node
    src_ptr:      int             # Device pointer (int representation)
    tensor_bytes: int             # Size in bytes
    shape:        tuple[int, ...]
    dtype:        str             # e.g. "float16", "bfloat16", "nvfp4"
    ttl_ms:       int = 500       # Cache TTL; set by Agent Hints


@dataclass
class TransferStats:
    transfers_initiated:  int   = 0
    transfers_completed:  int   = 0
    bytes_transferred:    int   = 0
    errors:               int   = 0


# ─── Transfer Manager ─────────────────────────────────────────────────────────

class NixlTransferManager:
    """
    Manages NIXL RDMA transfers of KV-cache tensors from prefill → decode.

    Usage
    -----
    ::

        mgr = NixlTransferManager(rdma_device="mlx5_0")
        await mgr.start()

        # On the prefill worker:
        descriptor = await mgr.register_tensor(
            device=0, ptr=cuda_ptr, bytes=size, shape=(32, 2048, 128),
            dtype="bfloat16", ttl_ms=request.agent_hints.get("kv_ttl_ms", 500)
        )

        # Send descriptor.handle to the decode worker (via router).

        # On the decode worker:
        dst_ptr = await mgr.transfer(descriptor, dst_device=1)
    """

    def __init__(self,
                 rdma_device: str = "mlx5_0",
                 max_concurrent: int = 64) -> None:
        self._rdma_device     = rdma_device
        self._max_concurrent  = max_concurrent
        self._semaphore       = asyncio.Semaphore(max_concurrent)
        self._descriptors:    dict[str, KVCacheDescriptor] = {}
        self._nixl_agent:     Any = None
        self._stats           = TransferStats()

    async def start(self) -> None:
        """Initialise the NIXL agent and RDMA fabric."""
        if _NIXL_AVAILABLE:
            self._nixl_agent = nixl.Agent(
                name="asr-serve-pdd",
                rdma_device=self._rdma_device,
            )
            await asyncio.to_thread(self._nixl_agent.init)
            logger.info("NIXL agent initialised on device %s", self._rdma_device)
        else:
            logger.info("NIXL stub mode: no RDMA transfers will occur")

    async def stop(self) -> None:
        """Shut down the NIXL agent."""
        if self._nixl_agent is not None:
            await asyncio.to_thread(self._nixl_agent.close)
            self._nixl_agent = None

    async def register_tensor(self,
                               device:     int,
                               ptr:        int,
                               bytes_size: int,
                               shape:      tuple[int, ...],
                               dtype:      str,
                               ttl_ms:     int = 500) -> KVCacheDescriptor:
        """
        Register a VRAM tensor with NIXL and return a transferable descriptor.

        The returned ``KVCacheDescriptor.handle`` is opaque and should be
        forwarded to the decode worker via the Smart Router.
        """
        handle = str(uuid.uuid4())
        descriptor = KVCacheDescriptor(
            handle=handle,
            src_device=device,
            src_ptr=ptr,
            tensor_bytes=bytes_size,
            shape=shape,
            dtype=dtype,
            ttl_ms=ttl_ms,
        )
        self._descriptors[handle] = descriptor

        if self._nixl_agent is not None:
            await asyncio.to_thread(
                self._nixl_agent.register_memory,
                device, ptr, bytes_size,
            )
        logger.debug("registered KV tensor handle=%s bytes=%d", handle, bytes_size)
        return descriptor

    async def transfer(self,
                       descriptor: KVCacheDescriptor,
                       dst_device: int,
                       dst_ptr:    int | None = None) -> int:
        """
        Initiate a zero-copy VRAM→VRAM RDMA transfer.

        Parameters
        ----------
        descriptor : KVCacheDescriptor
            Descriptor obtained from ``register_tensor`` on the prefill node.
        dst_device : int
            CUDA device ordinal on the decode node.
        dst_ptr : int, optional
            Pre-allocated destination device pointer.  If None, the manager
            will allocate via cuMemAlloc (not shown in stub).

        Returns
        -------
        int
            Device pointer of the destination tensor in decode VRAM.
        """
        async with self._semaphore:
            self._stats.transfers_initiated += 1
            logger.debug("NIXL transfer %s → device %d",
                         descriptor.handle, dst_device)

            if self._nixl_agent is not None:
                result_ptr = await asyncio.to_thread(
                    self._nixl_agent.transfer,
                    descriptor.src_device,
                    descriptor.src_ptr,
                    descriptor.tensor_bytes,
                    dst_device,
                    dst_ptr or 0,
                )
            else:
                # Stub: return a placeholder pointer value.
                result_ptr = dst_ptr or 0xDEADBEEF

            self._stats.transfers_completed += 1
            self._stats.bytes_transferred   += descriptor.tensor_bytes
            return result_ptr

    def evict(self, handle: str) -> None:
        """Remove a KV-cache descriptor (called when TTL expires)."""
        desc = self._descriptors.pop(handle, None)
        if desc is None:
            return
        if self._nixl_agent is not None:
            # Deregister memory region from NIXL fabric.
            try:
                self._nixl_agent.deregister_memory(desc.src_device, desc.src_ptr)
            except Exception as exc:
                logger.warning("NIXL deregister failed for %s: %s", handle, exc)
        logger.debug("evicted KV tensor handle=%s", handle)

    @property
    def stats(self) -> TransferStats:
        return self._stats


# ─── TTL Eviction Loop ────────────────────────────────────────────────────────

async def run_ttl_eviction_loop(mgr: NixlTransferManager,
                                 interval_ms: int = 100) -> None:
    """
    Periodically evict KV-cache descriptors whose TTL has expired.

    Run as an asyncio background task alongside the router.
    """
    import time
    registered_times: dict[str, float] = {}

    while True:
        await asyncio.sleep(interval_ms / 1000.0)
        now = time.monotonic()

        to_evict = [
            handle
            for handle, desc in list(mgr._descriptors.items())
            if handle in registered_times
            and (now - registered_times[handle]) * 1000 > desc.ttl_ms
        ]

        for handle in to_evict:
            mgr.evict(handle)
            registered_times.pop(handle, None)


# ─── CLI smoke-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NIXL RDMA Transfer Manager smoke-test")
    parser.add_argument("--rdma-device", default="mlx5_0")
    parser.add_argument("--ttl-ms",      type=int, default=500)
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    async def _smoke_test() -> None:
        mgr = NixlTransferManager(rdma_device=args.rdma_device)
        await mgr.start()

        desc = await mgr.register_tensor(
            device=0,
            ptr=0x7F000000,
            bytes_size=4 * 1024 * 1024,  # 4 MiB
            shape=(32, 2048, 128),
            dtype="bfloat16",
            ttl_ms=args.ttl_ms,
        )
        logger.info("registered: handle=%s", desc.handle)

        dst_ptr = await mgr.transfer(desc, dst_device=1)
        logger.info("transferred to 0x%X  stats=%s", dst_ptr, mgr.stats)

        mgr.evict(desc.handle)
        await mgr.stop()

    asyncio.run(_smoke_test())
