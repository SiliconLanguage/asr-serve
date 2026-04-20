# SPDX-License-Identifier: BSD-2-Clause-Patent
"""
Unit tests for the Agentic Factory Python modules.

These tests exercise the pure-Python logic only (no GPU, no external services).
Run with:
    python -m pytest tests/ -v
"""

import asyncio
import sys
import os

# Ensure src/ directories are on the import path.
_repo_root = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(_repo_root, "dynamo-orchestration", "src"))

import pytest
from agent_hints import AgentHintsManager, HintProfile, TTLEvictionScheduler


# ─── AgentHintsManager ────────────────────────────────────────────────────────

class TestAgentHintsManager:
    def test_default_profile(self):
        mgr = AgentHintsManager(default_model_id="whisper-large-v3")
        active = mgr.active
        assert active.model_id == "whisper-large-v3"
        assert 0 <= active.priority <= 9
        assert active.kv_ttl_ms > 0

    def test_set_profile(self):
        mgr = AgentHintsManager()
        mgr.set_profile(HintProfile(model_id="canary-1b", priority=8, kv_ttl_ms=250))
        active = mgr.active
        assert active.model_id == "canary-1b"
        assert active.priority == 8
        assert active.kv_ttl_ms == 250

    def test_set_hint_updates_field(self):
        mgr = AgentHintsManager(default_model_id="whisper")
        mgr.set_hint("priority", 9)
        assert mgr.active.priority == 9

    def test_get_headers(self):
        mgr = AgentHintsManager(default_model_id="test-model")
        mgr.set_profile(HintProfile(model_id="test-model", priority=7, kv_ttl_ms=300))
        headers = mgr.get_headers()
        assert headers["X-Agent-Priority"] == "7"
        assert headers["X-Agent-KV-TTL-Ms"] == "300"
        assert headers["X-Agent-Model-ID"] == "test-model"

    def test_history_is_recorded(self):
        mgr = AgentHintsManager(default_model_id="m0")
        mgr.set_profile(HintProfile(model_id="m1", priority=5, kv_ttl_ms=500))
        mgr.set_profile(HintProfile(model_id="m2", priority=5, kv_ttl_ms=500))
        history = mgr.recent_history()
        assert len(history) >= 2
        assert history[-1].model_id == "m1"

    def test_get_dict(self):
        mgr = AgentHintsManager(default_model_id="m")
        d = mgr.get_dict()
        assert "priority" in d
        assert "kv_ttl_ms" in d
        assert "model_id" in d


# ─── HintProfile ──────────────────────────────────────────────────────────────

class TestHintProfile:
    def test_to_headers_no_debug(self):
        p = HintProfile(model_id="x", priority=3, kv_ttl_ms=100, debug=False)
        h = p.to_headers()
        assert "X-Agent-Debug" not in h

    def test_to_headers_with_debug(self):
        p = HintProfile(model_id="x", priority=3, kv_ttl_ms=100, debug=True)
        h = p.to_headers()
        assert h.get("X-Agent-Debug") == "true"

    def test_to_dict_keys(self):
        p = HintProfile(model_id="y", priority=5, kv_ttl_ms=500)
        d = p.to_dict()
        assert set(d.keys()) >= {"priority", "kv_ttl_ms", "model_id", "debug"}


# ─── TTLEvictionScheduler ─────────────────────────────────────────────────────

class TestTTLEvictionScheduler:
    def test_evicts_after_ttl(self):
        scheduler = TTLEvictionScheduler()
        evicted_handles = []

        scheduler.register("handle-1", ttl_ms=1,
                            evict_fn=lambda h: evicted_handles.append(h))
        import time; time.sleep(0.01)
        result = scheduler.tick()
        assert "handle-1" in result
        assert "handle-1" in evicted_handles

    def test_does_not_evict_before_ttl(self):
        scheduler = TTLEvictionScheduler()
        evicted_handles = []

        scheduler.register("handle-live", ttl_ms=60_000,
                            evict_fn=lambda h: evicted_handles.append(h))
        result = scheduler.tick()
        assert "handle-live" not in result
        assert not evicted_handles

    def test_eviction_callback_error_is_swallowed(self):
        scheduler = TTLEvictionScheduler()

        def bad_evict(handle):
            raise RuntimeError("eviction error")

        scheduler.register("bad-handle", ttl_ms=1, evict_fn=bad_evict)
        import time; time.sleep(0.01)
        # Should not raise.
        result = scheduler.tick()
        assert "bad-handle" not in result  # error suppressed, handle cleaned up


# ─── Evaluator helpers ────────────────────────────────────────────────────────

from evaluator import (
    _edit_distance,
    _wer_components,
    _cer_components,
    _wer_to_priority,
    _rtf_to_kv_ttl_ms,
)


class TestEditDistance:
    def test_identical(self):
        assert _edit_distance(["a", "b", "c"], ["a", "b", "c"]) == 0

    def test_one_deletion(self):
        assert _edit_distance(["a", "b", "c"], ["a", "c"]) == 1

    def test_one_insertion(self):
        assert _edit_distance(["a", "c"], ["a", "b", "c"]) == 1

    def test_one_substitution(self):
        assert _edit_distance(["a", "b"], ["a", "x"]) == 1

    def test_empty_ref(self):
        assert _edit_distance([], ["a", "b"]) == 2

    def test_empty_hyp(self):
        assert _edit_distance(["a", "b"], []) == 2


class TestWERComponents:
    def test_perfect_match(self):
        errors, total = _wer_components("hello world", "hello world")
        assert errors == 0

    def test_one_word_wrong(self):
        errors, total = _wer_components("hello world", "hello there")
        assert errors == 1
        assert total == 2

    def test_empty_hypothesis(self):
        errors, total = _wer_components("hello world", "")
        assert errors == 2


class TestCERComponents:
    def test_perfect_match(self):
        errors, total = _cer_components("hello", "hello")
        assert errors == 0

    def test_one_char_wrong(self):
        errors, total = _cer_components("hello", "hxllo")
        assert errors == 1


class TestWERToPriority:
    def test_low_wer_high_priority(self):
        assert _wer_to_priority(0.03) == 9

    def test_medium_wer(self):
        assert _wer_to_priority(0.08) == 7

    def test_high_wer_low_priority(self):
        assert _wer_to_priority(0.25) == 3


class TestRTFToKVTTL:
    def test_fast_rtf(self):
        assert _rtf_to_kv_ttl_ms(0.005) == 200

    def test_medium_rtf(self):
        assert _rtf_to_kv_ttl_ms(0.03) == 350

    def test_slow_rtf(self):
        assert _rtf_to_kv_ttl_ms(0.10) == 500


# ─── Router ──────────────────────────────────────────────────────────────────

from router import (
    ASRRequest,
    RequestPhase,
    WorkerEndpoint,
    WorkerRegistry,
    SmartRouter,
    _p99,
)


class TestWorkerRegistry:
    def test_register_and_retrieve(self):
        reg = WorkerRegistry()
        ep  = WorkerEndpoint("w0", RequestPhase.PREFILL, "10.0.0.1:50051")
        reg.register(ep)
        assert reg.least_loaded(RequestPhase.PREFILL) is ep

    def test_least_loaded_prefers_idle(self):
        reg = WorkerRegistry()
        reg.register(WorkerEndpoint("w0", RequestPhase.DECODE, "10.0.0.1:50051", load=0.9))
        reg.register(WorkerEndpoint("w1", RequestPhase.DECODE, "10.0.0.2:50051", load=0.1))
        best = reg.least_loaded(RequestPhase.DECODE)
        assert best is not None
        assert best.worker_id == "w1"

    def test_unhealthy_excluded(self):
        reg = WorkerRegistry()
        ep  = WorkerEndpoint("w0", RequestPhase.PREFILL, "10.0.0.1:50051")
        reg.register(ep)
        reg.mark_unhealthy("w0")
        assert reg.least_loaded(RequestPhase.PREFILL) is None

    def test_deregister(self):
        reg = WorkerRegistry()
        reg.register(WorkerEndpoint("w0", RequestPhase.PREFILL, "10.0.0.1:50051"))
        reg.deregister("w0")
        assert reg.least_loaded(RequestPhase.PREFILL) is None


class TestP99:
    def test_empty(self):
        assert _p99([]) == 0.0

    def test_single_value(self):
        assert _p99([42.0]) == 42.0

    def test_sorted_percentile(self):
        values = list(range(1, 101))  # 1..100
        result = _p99(values)
        assert result >= 99


@pytest.mark.asyncio
class TestSmartRouter:
    async def test_route_returns_string(self):
        reg = WorkerRegistry()
        reg.register(WorkerEndpoint("p0", RequestPhase.PREFILL, "10.0.0.1:50051"))
        reg.register(WorkerEndpoint("d0", RequestPhase.DECODE,  "10.0.0.2:50051"))
        router = SmartRouter(reg)
        req    = ASRRequest(request_id="r1", audio_bytes=b"\x00" * 640)
        result = await router.route(req)
        assert isinstance(result, str)

    async def test_route_no_prefill_workers_raises(self):
        reg    = WorkerRegistry()
        reg.register(WorkerEndpoint("d0", RequestPhase.DECODE, "10.0.0.2:50051"))
        router = SmartRouter(reg)
        req    = ASRRequest(request_id="r2", audio_bytes=b"\x00" * 640)
        with pytest.raises(RuntimeError, match="prefill"):
            await router.route(req)
