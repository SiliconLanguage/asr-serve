# SPDX-License-Identifier: BSD-2-Clause-Patent
"""
Continuous ASR model evaluation loop.

Runs on a configurable cadence and:
  1. Fetches active telemetry (live WER samples) from the Dynamo metrics bus.
  2. Benchmarks candidate open-source ASR models against the reference set.
  3. Computes WER / CER / RTF for each model.
  4. Publishes results to the NAT agent which decides whether to promote a new
     model to production.
  5. Updates Agent Hints based on evaluation outcomes.

Metrics
-------
- WER  : Word Error Rate   (lower is better)
- CER  : Character Error Rate
- RTF  : Real-Time Factor  (processing_time / audio_duration; lower is better)
- TTFT : Time-to-First-Token (ms) for streaming mode
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import yaml

from agent_hints import AgentHintsManager, HintProfile

logger = logging.getLogger(__name__)


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class ASRModelCandidate:
    """Metadata for an ASR model under evaluation."""
    model_id:   str
    model_path: str
    version:    str         = "latest"
    is_active:  bool        = False   # True if currently serving production traffic
    tags:       list[str]   = field(default_factory=list)


@dataclass
class EvalSample:
    """A single evaluation sample (audio + reference transcript)."""
    sample_id:      str
    audio_bytes:    bytes
    reference_text: str
    duration_s:     float


@dataclass
class ModelMetrics:
    model_id:   str
    wer:        float         = 1.0
    cer:        float         = 1.0
    rtf:        float         = 1.0
    ttft_ms:    float         = 9999.0
    eval_time:  float         = field(default_factory=time.monotonic)
    n_samples:  int           = 0


# ─── Telemetry Fetcher ────────────────────────────────────────────────────────

class TelemetryFetcher:
    """
    Fetches live telemetry from the Dynamo metrics bus.

    In production, subscribe to a Prometheus remote-write endpoint or a
    NATS JetStream subject that the Dynamo workers publish to.
    """

    def __init__(self, metrics_endpoint: str) -> None:
        self._endpoint = metrics_endpoint

    async def fetch_wer_samples(self, n: int = 100) -> list[EvalSample]:
        """
        Fetch up to *n* recent audio/reference pairs from the telemetry store.
        These are production utterances where the ground-truth transcript is
        available (e.g. from a human-in-the-loop correction pipeline).
        """
        # TODO: implement real metrics fetch (e.g. via NATS or HTTP)
        logger.debug("fetching %d WER samples from %s", n, self._endpoint)
        # Stub: return synthetic samples.
        return [
            EvalSample(
                sample_id=f"sample-{i}",
                audio_bytes=bytes(640),     # 10 ms of silence
                reference_text="hello world",
                duration_s=0.01,
            )
            for i in range(min(n, 5))
        ]


# ─── Model Evaluator ──────────────────────────────────────────────────────────

class ModelEvaluator:
    """
    Evaluates ASR model candidates against telemetry samples.
    """

    def __init__(self, inference_endpoint: str) -> None:
        self._endpoint = inference_endpoint
        # Lazy import to avoid hard dependency on aiohttp in all environments.
        self._session: Any = None

    async def start(self) -> None:
        try:
            import aiohttp
            self._session = aiohttp.ClientSession()
        except ImportError:
            logger.warning("aiohttp not available – using stub HTTP client")

    async def stop(self) -> None:
        if self._session is not None:
            await self._session.close()

    async def transcribe(self,
                          model_id: str,
                          audio_bytes: bytes) -> tuple[str, float]:
        """
        Transcribe audio with *model_id* and return (transcript, latency_ms).
        """
        t0 = time.monotonic()
        if self._session is None:
            # Stub: return placeholder transcript.
            await asyncio.sleep(0.005)
            transcript = "hello world"
        else:
            try:
                async with self._session.post(
                    f"{self._endpoint}/v1/audio/transcriptions",
                    data=audio_bytes,
                    headers={
                        "Content-Type": "application/octet-stream",
                        "X-Model-ID":   model_id,
                    },
                    timeout=10,
                ) as resp:
                    resp.raise_for_status()
                    body = await resp.json()
                    transcript = body.get("text", "")
            except Exception as exc:
                logger.warning("transcription failed for %s: %s", model_id, exc)
                transcript = ""
        latency_ms = (time.monotonic() - t0) * 1000
        return transcript, latency_ms

    async def evaluate_model(self,
                              candidate: ASRModelCandidate,
                              samples: list[EvalSample]) -> ModelMetrics:
        """
        Run the full evaluation suite against *candidate* and return metrics.
        """
        if not samples:
            return ModelMetrics(model_id=candidate.model_id)

        total_wer_errors = 0
        total_words      = 0
        total_cer_errors = 0
        total_chars      = 0
        total_rtf        = 0.0
        total_ttft_ms    = 0.0

        for sample in samples:
            hyp, latency_ms = await self.transcribe(
                candidate.model_id, sample.audio_bytes
            )
            wer_e, n_w = _wer_components(sample.reference_text, hyp)
            cer_e, n_c = _cer_components(sample.reference_text, hyp)

            total_wer_errors += wer_e
            total_words      += n_w
            total_cer_errors += cer_e
            total_chars      += n_c
            total_rtf        += latency_ms / 1000.0 / max(sample.duration_s, 1e-6)
            total_ttft_ms    += latency_ms

        n = len(samples)
        return ModelMetrics(
            model_id  = candidate.model_id,
            wer       = total_wer_errors / max(total_words, 1),
            cer       = total_cer_errors / max(total_chars, 1),
            rtf       = total_rtf / n,
            ttft_ms   = total_ttft_ms / n,
            n_samples = n,
        )


# ─── Evaluation Loop ──────────────────────────────────────────────────────────

class EvaluationLoop:
    """
    Autonomous evaluation and model-promotion control loop.
    """

    def __init__(self,
                 candidates:         list[ASRModelCandidate],
                 telemetry_fetcher:  TelemetryFetcher,
                 evaluator:          ModelEvaluator,
                 hints_manager:      AgentHintsManager,
                 eval_interval_s:    float = 300.0,
                 wer_promote_thresh: float = 0.05) -> None:
        self._candidates         = candidates
        self._telemetry          = telemetry_fetcher
        self._evaluator          = evaluator
        self._hints              = hints_manager
        self._eval_interval_s    = eval_interval_s
        self._wer_promote_thresh = wer_promote_thresh
        self._best_metrics:      dict[str, ModelMetrics] = {}
        self._running            = False

    async def run(self) -> None:
        self._running = True
        logger.info("evaluation loop started  interval=%.0f s  candidates=%d",
                    self._eval_interval_s, len(self._candidates))

        while self._running:
            await self._eval_cycle()
            await asyncio.sleep(self._eval_interval_s)

    async def stop(self) -> None:
        self._running = False

    async def _eval_cycle(self) -> None:
        logger.info("starting evaluation cycle")
        samples = await self._telemetry.fetch_wer_samples(n=100)
        if not samples:
            logger.warning("no telemetry samples available")
            return

        results: list[ModelMetrics] = []
        for candidate in self._candidates:
            metrics = await self._evaluator.evaluate_model(candidate, samples)
            results.append(metrics)
            logger.info("  %s: WER=%.3f CER=%.3f RTF=%.4f TTFT=%.1f ms",
                        metrics.model_id, metrics.wer, metrics.cer,
                        metrics.rtf, metrics.ttft_ms)

        # Sort by WER (primary) then RTF (tiebreak).
        results.sort(key=lambda m: (m.wer, m.rtf))
        best = results[0] if results else None

        if best is not None:
            logger.info("best candidate: %s (WER=%.3f)", best.model_id, best.wer)
            self._best_metrics[best.model_id] = best

            # Attach agent hints based on best model metrics.
            priority = _wer_to_priority(best.wer)
            kv_ttl   = _rtf_to_kv_ttl_ms(best.rtf)
            self._hints.set_profile(HintProfile(
                model_id  = best.model_id,
                priority  = priority,
                kv_ttl_ms = kv_ttl,
            ))
            logger.info("updated agent hints: model=%s priority=%d kv_ttl=%d ms",
                        best.model_id, priority, kv_ttl)

            # Trigger model promotion if WER improvement exceeds threshold.
            active = next(
                (c for c in self._candidates if c.is_active), None
            )
            if active and active.model_id != best.model_id:
                active_metrics = self._best_metrics.get(active.model_id)
                if active_metrics and (
                    active_metrics.wer - best.wer > self._wer_promote_thresh
                ):
                    logger.info(
                        "PROMOTING %s → production (ΔWER=%.3f)",
                        best.model_id,
                        active_metrics.wer - best.wer,
                    )
                    await self._promote(best.model_id)

    async def _promote(self, model_id: str) -> None:
        """Promote *model_id* to production (stub – integrate with Dynamo API)."""
        for c in self._candidates:
            c.is_active = (c.model_id == model_id)
        # TODO: call Dynamo API to hot-swap the active model.
        logger.info("model promoted: %s", model_id)


# ─── WER / CER Helpers ────────────────────────────────────────────────────────

def _edit_distance(a: list, b: list) -> int:
    """Compute Levenshtein edit distance between two sequences."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if a[i - 1] == b[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])
    return dp[n]


def _wer_components(ref: str, hyp: str) -> tuple[int, int]:
    ref_words = ref.lower().split()
    hyp_words = hyp.lower().split()
    return _edit_distance(ref_words, hyp_words), max(len(ref_words), 1)


def _cer_components(ref: str, hyp: str) -> tuple[int, int]:
    ref_chars = list(ref.lower().replace(" ", ""))
    hyp_chars = list(hyp.lower().replace(" ", ""))
    return _edit_distance(ref_chars, hyp_chars), max(len(ref_chars), 1)


def _wer_to_priority(wer: float) -> int:
    """Map WER to a routing priority (0–9); lower WER → higher priority."""
    if wer < 0.05:
        return 9
    if wer < 0.10:
        return 7
    if wer < 0.20:
        return 5
    return 3


def _rtf_to_kv_ttl_ms(rtf: float) -> int:
    """Map RTF to a KV-cache TTL; faster models get shorter TTLs."""
    if rtf < 0.01:
        return 200
    if rtf < 0.05:
        return 350
    return 500


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ASR model evaluation loop")
    parser.add_argument("--config",              default="configs/factory.yaml")
    parser.add_argument("--inference-endpoint",  default="http://10.0.0.5:8000")
    parser.add_argument("--metrics-endpoint",    default="http://10.0.0.5:9090")
    parser.add_argument("--eval-interval",       type=float, default=300.0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    candidates = [
        ASRModelCandidate("whisper-large-v3",     "/models/whisper-large-v3",     is_active=True),
        ASRModelCandidate("whisper-large-v3-turbo","/models/whisper-large-v3-turbo"),
        ASRModelCandidate("canary-1b",             "/models/canary-1b"),
        ASRModelCandidate("parakeet-tdt-1.1b",     "/models/parakeet-tdt-1.1b"),
    ]

    hints_mgr = AgentHintsManager()
    telemetry  = TelemetryFetcher(args.metrics_endpoint)
    evaluator  = ModelEvaluator(args.inference_endpoint)

    loop_runner = EvaluationLoop(
        candidates         = candidates,
        telemetry_fetcher  = telemetry,
        evaluator          = evaluator,
        hints_manager      = hints_mgr,
        eval_interval_s    = args.eval_interval,
    )

    async def _main() -> None:
        await evaluator.start()
        try:
            await loop_runner.run()
        finally:
            await evaluator.stop()

    asyncio.run(_main())
