# SPDX-License-Identifier: BSD-2-Clause-Patent
"""
NeMo Agent Toolkit (NAT) Agent for autonomous ASR model governance.

The NATAgent wraps NVIDIA NeMo Agent Toolkit to provide:
  - Autonomous model selection based on evaluation metrics.
  - Dynamic policy generation using an LLM reasoner.
  - Integration with the MCP tool server for structured actions.

The agent runs a ReAct-style (Reason + Act) loop:
  1. Observe: pull current metrics and cluster telemetry.
  2. Reason:  use an LLM to determine whether a model swap is beneficial.
  3. Act:     call MCP tools (evaluate_model, promote_model, set_agent_hint).

References
----------
- NeMo Agent Toolkit: https://github.com/NVIDIA/NeMo-Agent-Toolkit
- ReAct prompting:     https://arxiv.org/abs/2210.03629
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# ── Optional NAT import ───────────────────────────────────────────────────────
try:
    from nemo_agent import Agent, Tool as NATTool   # type: ignore[import-untyped]
    _NAT_AVAILABLE = True
except ImportError:
    Agent    = None   # type: ignore[assignment,misc]
    NATTool  = None   # type: ignore[assignment,misc]
    _NAT_AVAILABLE = False
    logger.warning("nemo-agent-toolkit not installed – running in stub mode")


# ─── Agent Configuration ──────────────────────────────────────────────────────

@dataclass
class NATAgentConfig:
    # LLM used for reasoning (any OpenAI-compatible endpoint).
    llm_endpoint:   str   = "https://integrate.api.nvidia.com/v1"
    llm_model:      str   = "meta/llama-3.1-70b-instruct"
    llm_api_key:    str   = ""     # set via NVIDIA_API_KEY env var
    # How many reasoning steps per cycle before giving up.
    max_iterations: int   = 10
    # Minimum WER improvement required before the agent recommends promotion.
    min_wer_delta:  float = 0.02
    # Cycle interval (seconds).
    cycle_interval: float = 600.0


# ─── Observation ─────────────────────────────────────────────────────────────

@dataclass
class FactoryObservation:
    """Snapshot of the AI Factory state presented to the agent each cycle."""
    timestamp:       float = field(default_factory=time.monotonic)
    active_model:    str   = ""
    active_wer:      float = 1.0
    active_rtf:      float = 1.0
    candidate_models: list[dict[str, Any]] = field(default_factory=list)
    cluster_load:    float = 0.0
    current_hints:   dict[str, Any] = field(default_factory=dict)


# ─── Action ──────────────────────────────────────────────────────────────────

@dataclass
class AgentAction:
    """A single action the agent decides to take."""
    action_type: str    # "evaluate", "promote", "set_hint", "noop"
    parameters:  dict[str, Any] = field(default_factory=dict)
    rationale:   str   = ""


# ─── NATAgent ────────────────────────────────────────────────────────────────

class NATAgent:
    """
    Autonomous ASR governance agent powered by NeMo Agent Toolkit.
    """

    # System prompt that guides the LLM reasoner.
    _SYSTEM_PROMPT = """\
You are an autonomous AI model governance agent for a production speech-to-text
(ASR) inference system.  Your goal is to ensure that the best-performing ASR
model is always serving production traffic, and that routing hints are tuned
for optimal latency and accuracy.

You have access to the following tools:
  - list_models         : List registered ASR models.
  - evaluate_model      : Trigger evaluation (returns WER/CER/RTF).
  - get_metrics         : Retrieve cached metrics.
  - set_agent_hint      : Update routing hints (priority, kv_ttl_ms).
  - promote_model       : Promote a model to production.
  - get_cluster_status  : Check cluster health.

Decision policy:
  1. If any candidate has WER > {min_wer_delta} lower than the active model,
     evaluate it and consider promotion.
  2. If cluster load > 80%, increase kv_ttl_ms to reduce cache pressure.
  3. If active model RTF > 0.02, search for a faster alternative.
  4. Otherwise, do nothing (noop).

Always justify your actions with a brief rationale.
"""

    def __init__(self,
                 config:       NATAgentConfig,
                 mcp_tools:    list[dict[str, Any]] | None = None) -> None:
        self._config    = config
        self._mcp_tools = mcp_tools or []
        self._agent:    Any = None
        self._running   = False
        self._cycle_count = 0
        self._actions_taken: list[AgentAction] = []

    def _build_observation_prompt(self, obs: FactoryObservation) -> str:
        return (
            f"Current factory state:\n"
            f"  active_model : {obs.active_model}\n"
            f"  active_wer   : {obs.active_wer:.4f}\n"
            f"  active_rtf   : {obs.active_rtf:.5f}\n"
            f"  cluster_load : {obs.cluster_load:.1%}\n"
            f"  current_hints: {json.dumps(obs.current_hints)}\n"
            f"  candidates   : {json.dumps(obs.candidate_models, indent=2)}\n\n"
            "Decide what actions to take. Respond with a JSON object:\n"
            '{"actions": [{"action_type": "...", "parameters": {...}, "rationale": "..."}]}\n'
            'Use action_type "noop" if no changes are needed.'
        )

    async def reason(self, obs: FactoryObservation) -> list[AgentAction]:
        """
        Run one reasoning cycle and return the list of actions to take.
        """
        if not _NAT_AVAILABLE or self._agent is None:
            return self._stub_reason(obs)

        prompt = self._build_observation_prompt(obs)
        response = await asyncio.to_thread(self._agent.run, prompt)
        return self._parse_actions(response)

    def _stub_reason(self, obs: FactoryObservation) -> list[AgentAction]:
        """Stub reasoning logic used when NAT is not available."""
        actions: list[AgentAction] = []

        # Example heuristic: if cluster is hot, bump kv_ttl_ms.
        if obs.cluster_load > 0.8:
            actions.append(AgentAction(
                action_type="set_hint",
                parameters={"key": "kv_ttl_ms", "value": 750},
                rationale=f"Cluster load {obs.cluster_load:.1%} > 80%; "
                           "increasing KV cache TTL to reduce eviction pressure.",
            ))

        # If active WER is high, trigger evaluation of alternatives.
        if obs.active_wer > 0.10 and obs.candidate_models:
            for c in obs.candidate_models:
                if not c.get("is_active"):
                    actions.append(AgentAction(
                        action_type="evaluate",
                        parameters={"model_id": c["model_id"]},
                        rationale=(
                            f"Active WER={obs.active_wer:.3f} exceeds 0.10; "
                            f"evaluating candidate {c['model_id']}."
                        ),
                    ))
                    break

        if not actions:
            actions.append(AgentAction(
                action_type="noop",
                rationale="Metrics within acceptable range; no action required.",
            ))

        return actions

    def _parse_actions(self, response: str) -> list[AgentAction]:
        """Parse LLM JSON response into AgentAction list."""
        try:
            data    = json.loads(response)
            actions = data.get("actions", [])
            return [
                AgentAction(
                    action_type=a.get("action_type", "noop"),
                    parameters=a.get("parameters", {}),
                    rationale=a.get("rationale", ""),
                )
                for a in actions
            ]
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("failed to parse LLM response: %s  raw=%r", exc, response)
            return [AgentAction(action_type="noop", rationale="parse error")]

    async def execute_actions(self,
                               actions:  list[AgentAction],
                               mcp_call: Any) -> None:
        """
        Execute a list of agent actions via MCP tool calls.

        Parameters
        ----------
        actions :
            Actions returned by ``reason()``.
        mcp_call :
            Async callable ``(tool_name, arguments) -> str`` that invokes
            the corresponding MCP tool.
        """
        for action in actions:
            logger.info("agent action: type=%s  rationale=%s",
                        action.action_type, action.rationale)
            self._actions_taken.append(action)

            if action.action_type == "noop":
                continue
            if action.action_type == "evaluate":
                await mcp_call("evaluate_model", action.parameters)
            elif action.action_type == "promote":
                await mcp_call("promote_model", action.parameters)
            elif action.action_type == "set_hint":
                await mcp_call("set_agent_hint", action.parameters)
            else:
                logger.warning("unknown action type: %s", action.action_type)

    async def run(self,
                  observation_fn: Any,
                  mcp_call:       Any) -> None:
        """
        Run the agent control loop indefinitely.

        Parameters
        ----------
        observation_fn :
            Async callable ``() -> FactoryObservation``.
        mcp_call :
            Async callable ``(tool_name, arguments) -> str``.
        """
        self._running = True
        logger.info("NAT agent started  cycle_interval=%.0f s",
                    self._config.cycle_interval)

        while self._running:
            self._cycle_count += 1
            logger.info("agent cycle %d", self._cycle_count)
            try:
                obs     = await observation_fn()
                actions = await self.reason(obs)
                await self.execute_actions(actions, mcp_call)
            except Exception as exc:
                logger.error("agent cycle error: %s", exc, exc_info=True)

            await asyncio.sleep(self._config.cycle_interval)

    async def stop(self) -> None:
        self._running = False


# ─── CLI smoke-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="NAT Agent smoke-test")
    parser.add_argument("--llm-endpoint", default="https://integrate.api.nvidia.com/v1")
    parser.add_argument("--llm-model",    default="meta/llama-3.1-70b-instruct")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg   = NATAgentConfig(llm_endpoint=args.llm_endpoint,
                           llm_model=args.llm_model,
                           cycle_interval=5.0)
    agent = NATAgent(cfg)

    async def _obs() -> FactoryObservation:
        return FactoryObservation(
            active_model="whisper-large-v3",
            active_wer=0.15,
            active_rtf=0.008,
            cluster_load=0.85,
            candidate_models=[
                {"model_id": "canary-1b", "is_active": False},
            ],
            current_hints={"priority": 5, "kv_ttl_ms": 500},
        )

    async def _mcp(tool: str, args_: dict) -> str:
        logger.info("MCP call: %s(%s)", tool, args_)
        return "ok"

    async def _run_once() -> None:
        obs     = await _obs()
        actions = await agent.reason(obs)
        await agent.execute_actions(actions, _mcp)
        logger.info("actions taken: %d", len(actions))

    asyncio.run(_run_once())
