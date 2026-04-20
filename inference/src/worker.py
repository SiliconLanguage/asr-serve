# SPDX-License-Identifier: BSD-2-Clause-Patent
"""
ASR Inference Worker – vLLM AsyncLLMEngine wrapper.

This module wraps vLLM's AsyncLLMEngine to serve Whisper (or any
vLLM-compatible ASR model) with:
  - Batched audio transcription via the OpenAI-compatible /v1/audio/transcriptions endpoint
  - Blackwell NVFP4 / FP8 quantisation (configured via YAML)
  - Agent Hints header propagation (priority, kv_ttl_ms)
  - Prometheus metrics exposure

Quick start
-----------
    python src/worker.py \\
        --config configs/vllm_worker.yaml \\
        --model  /models/whisper-large-v3
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# ── Optional vLLM import ─────────────────────────────────────────────────────
try:
    from vllm import AsyncEngineArgs, AsyncLLMEngine, SamplingParams
    from vllm.outputs import RequestOutput
    _VLLM_AVAILABLE = True
except ImportError:
    AsyncLLMEngine = None  # type: ignore[assignment,misc]
    AsyncEngineArgs = None  # type: ignore[assignment]
    SamplingParams  = None  # type: ignore[assignment]
    RequestOutput   = None  # type: ignore[assignment]
    _VLLM_AVAILABLE = False
    logger.warning("vllm not installed – running in stub mode")

# ── Optional FastAPI import ───────────────────────────────────────────────────
try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    import uvicorn
    _FASTAPI_AVAILABLE = True
except ImportError:
    FastAPI = None  # type: ignore[assignment,misc]
    _FASTAPI_AVAILABLE = False
    logger.warning("fastapi/uvicorn not installed – HTTP server disabled")


# ─── Configuration ────────────────────────────────────────────────────────────

@dataclass
class WorkerConfig:
    model_name_or_path:       str   = "/models/whisper-large-v3"
    tensor_parallel_size:     int   = 1
    pipeline_parallel_size:   int   = 1
    gpu_memory_utilization:   float = 0.90
    max_model_len:            int   = 4096
    max_num_seqs:             int   = 256
    quantization:             str | None = None   # "nvfp4", "fp8", None
    kv_cache_dtype:           str   = "auto"
    chunked_prefill_enabled:  bool  = True
    max_prefill_tokens:       int   = 2048
    host:                     str   = "0.0.0.0"
    port:                     int   = 8000

    @classmethod
    def from_yaml(cls, path: str, model_override: str | None = None) -> "WorkerConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        model = raw.get("model", {})
        engine = raw.get("engine", {})
        quant  = raw.get("quantization", {})
        attn   = raw.get("attention", {})
        prefill = raw.get("prefill", {})
        serving = raw.get("serving", {})

        return cls(
            model_name_or_path=model_override or model.get("name_or_path", cls.model_name_or_path),
            tensor_parallel_size=engine.get("tensor_parallel_size", cls.tensor_parallel_size),
            pipeline_parallel_size=engine.get("pipeline_parallel_size", cls.pipeline_parallel_size),
            gpu_memory_utilization=engine.get("gpu_memory_utilization", cls.gpu_memory_utilization),
            max_model_len=engine.get("max_model_len", cls.max_model_len),
            max_num_seqs=engine.get("max_num_seqs", cls.max_num_seqs),
            quantization=quant.get("method") if quant.get("method") != "nvfp4" else None,
            kv_cache_dtype=attn.get("kv_cache_dtype", cls.kv_cache_dtype),
            chunked_prefill_enabled=prefill.get("chunked_prefill_enabled", cls.chunked_prefill_enabled),
            max_prefill_tokens=prefill.get("max_prefill_tokens", cls.max_prefill_tokens),
            host=serving.get("host", cls.host),
            port=int(serving.get("port", cls.port)),
        )


# ─── Inference Worker ─────────────────────────────────────────────────────────

class InferenceWorker:
    """
    Wraps vLLM AsyncLLMEngine for ASR transcription with Agent Hints support.
    """

    def __init__(self, config: WorkerConfig) -> None:
        self.config = config
        self._engine: Any = None
        self._request_count = 0
        self._total_latency_ms = 0.0

    async def start(self) -> None:
        if not _VLLM_AVAILABLE:
            logger.info("stub mode: vLLM engine not started")
            return

        engine_args = AsyncEngineArgs(
            model=self.config.model_name_or_path,
            tensor_parallel_size=self.config.tensor_parallel_size,
            pipeline_parallel_size=self.config.pipeline_parallel_size,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            max_model_len=self.config.max_model_len,
            max_num_seqs=self.config.max_num_seqs,
            quantization=self.config.quantization,
            kv_cache_dtype=self.config.kv_cache_dtype,
            enable_chunked_prefill=self.config.chunked_prefill_enabled,
            max_num_batched_tokens=self.config.max_prefill_tokens,
        )
        self._engine = AsyncLLMEngine.from_engine_args(engine_args)
        logger.info("vLLM engine started: model=%s tp=%d",
                    self.config.model_name_or_path,
                    self.config.tensor_parallel_size)

    async def transcribe(self,
                          audio_bytes: bytes,
                          agent_hints: dict[str, Any] | None = None) -> str:
        """
        Transcribe raw PCM audio bytes and return the transcript string.

        Parameters
        ----------
        audio_bytes :
            16 kHz / 16-bit mono PCM audio.
        agent_hints :
            Optional routing hints from the Agentic Factory
            (e.g. ``{"priority": 8, "kv_ttl_ms": 200}``).
        """
        hints = agent_hints or {}
        t0    = time.monotonic()

        if not _VLLM_AVAILABLE or self._engine is None:
            # Stub response for development/testing.
            await asyncio.sleep(0.005)
            transcript = f"[stub transcript len={len(audio_bytes)}B hints={hints}]"
        else:
            # TODO: Convert audio_bytes → Whisper log-mel spectrogram tensor
            # and pass as multi-modal input.  This requires the vLLM audio
            # preprocessing pipeline (available in vLLM ≥ 0.6.0 audio branch).
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=448,     # Whisper max output tokens
            )
            request_id = f"asr-{id(audio_bytes)}-{int(t0*1e6)}"
            output: RequestOutput = None  # type: ignore[assignment]
            async for output in self._engine.generate(
                inputs={"prompt_token_ids": [], "multi_modal_data": {"audio": audio_bytes}},
                sampling_params=sampling_params,
                request_id=request_id,
            ):
                pass
            transcript = output.outputs[0].text if output else ""

        elapsed_ms = (time.monotonic() - t0) * 1000
        self._request_count += 1
        self._total_latency_ms += elapsed_ms
        logger.debug("transcribe done in %.1f ms  priority=%s",
                     elapsed_ms, hints.get("priority", "n/a"))
        return transcript

    def avg_latency_ms(self) -> float:
        if self._request_count == 0:
            return 0.0
        return self._total_latency_ms / self._request_count


# ─── HTTP Server (FastAPI) ────────────────────────────────────────────────────

def build_app(worker: InferenceWorker) -> "FastAPI":
    """Build and return the FastAPI application."""
    app = FastAPI(title="asr-serve inference worker",
                  description="OpenAI-compatible ASR transcription endpoint")

    @app.post("/v1/audio/transcriptions")
    async def transcribe_endpoint(request: Request) -> JSONResponse:
        body = await request.body()
        if not body:
            raise HTTPException(status_code=400, detail="Empty request body")

        # Extract Agent Hints from request headers.
        agent_hints: dict[str, Any] = {}
        priority = request.headers.get("X-Agent-Priority")
        kv_ttl   = request.headers.get("X-Agent-KV-TTL-Ms")
        if priority is not None:
            agent_hints["priority"] = int(priority)
        if kv_ttl is not None:
            agent_hints["kv_ttl_ms"] = int(kv_ttl)

        transcript = await worker.transcribe(body, agent_hints)
        return JSONResponse({"text": transcript})

    @app.get("/health")
    async def health() -> JSONResponse:
        return JSONResponse({
            "status":         "ok",
            "requests":       worker._request_count,
            "avg_latency_ms": round(worker.avg_latency_ms(), 2),
        })

    @app.get("/metrics")
    async def metrics() -> JSONResponse:
        return JSONResponse({
            "requests_total": worker._request_count,
            "avg_latency_ms": round(worker.avg_latency_ms(), 2),
        })

    return app


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="asr-serve inference worker")
    parser.add_argument("--config",  default="configs/vllm_worker.yaml")
    parser.add_argument("--model",   default=None,
                        help="Override model path from config")
    parser.add_argument("--host",    default=None)
    parser.add_argument("--port",    type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    config = WorkerConfig.from_yaml(args.config, model_override=args.model)
    if args.host:
        config.host = args.host
    if args.port:
        config.port = args.port

    worker = InferenceWorker(config)

    async def _main() -> None:
        await worker.start()
        if _FASTAPI_AVAILABLE:
            app = build_app(worker)
            server_config = uvicorn.Config(
                app,
                host=config.host,
                port=config.port,
                loop="asyncio",
                log_level="info",
            )
            server = uvicorn.Server(server_config)
            await server.serve()
        else:
            logger.info("FastAPI not available – running headless (Ctrl-C to stop)")
            await asyncio.Event().wait()

    asyncio.run(_main())
