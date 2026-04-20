# SPDX-License-Identifier: BSD-2-Clause-Patent
"""
Model Context Protocol (MCP) Client for the AI Factory.

Registers the following MCP tools so that an LLM-based orchestrator (e.g.
Claude, GPT-4) can interact with the asr-serve AI Factory:

  - ``list_models``         : List all registered ASR model candidates.
  - ``evaluate_model``      : Trigger an ad-hoc evaluation of a specific model.
  - ``get_metrics``         : Retrieve current WER/CER/RTF for a model.
  - ``set_agent_hint``      : Override Agent Hints for a model.
  - ``promote_model``       : Promote a model to production.
  - ``get_cluster_status``  : Return current cluster health and load.

References
----------
- MCP specification: https://modelcontextprotocol.io/
- Python MCP SDK:    https://github.com/modelcontextprotocol/python-sdk
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ── Optional MCP import ───────────────────────────────────────────────────────
try:
    from mcp.server import Server as MCPServer                     # type: ignore
    from mcp.server.stdio import stdio_server                      # type: ignore
    from mcp.types import Tool, TextContent, CallToolResult        # type: ignore
    _MCP_AVAILABLE = True
except ImportError:
    MCPServer = None         # type: ignore[assignment,misc]
    stdio_server = None      # type: ignore[assignment]
    Tool = None              # type: ignore[assignment]
    TextContent = None       # type: ignore[assignment]
    CallToolResult = None    # type: ignore[assignment]
    _MCP_AVAILABLE = False
    logger.warning("mcp package not installed – MCP server disabled")


# ─── Factory Context (shared state injected by the evaluator) ─────────────────

class FactoryContext:
    """
    Shared mutable state that MCP tools read/write.

    Injected into the MCP server at startup so tools can call back into the
    EvaluationLoop and AgentHintsManager.
    """

    def __init__(self) -> None:
        self.candidates:    list[dict[str, Any]] = []
        self.metrics_store: dict[str, dict[str, float]] = {}
        self.agent_hints:   dict[str, Any] = {}
        self.cluster_load:  float = 0.0
        # Callbacks set by the evaluator.
        self.evaluate_fn:   Any = None   # async (model_id) -> dict
        self.promote_fn:    Any = None   # async (model_id) -> None

    def update_metrics(self, model_id: str, metrics: dict[str, float]) -> None:
        self.metrics_store[model_id] = metrics

    def update_hints(self, hints: dict[str, Any]) -> None:
        self.agent_hints.update(hints)


# ─── MCP Tool Handlers ────────────────────────────────────────────────────────

async def _handle_list_models(ctx: FactoryContext,
                               arguments: dict[str, Any]) -> str:
    if not ctx.candidates:
        return "No ASR model candidates registered."
    lines = ["Registered ASR model candidates:"]
    for c in ctx.candidates:
        active = " [ACTIVE]" if c.get("is_active") else ""
        lines.append(f"  - {c['model_id']}{active}  version={c.get('version', '?')}")
    return "\n".join(lines)


async def _handle_evaluate_model(ctx: FactoryContext,
                                  arguments: dict[str, Any]) -> str:
    model_id = arguments.get("model_id", "")
    if not model_id:
        return "ERROR: model_id is required"
    if ctx.evaluate_fn is None:
        return f"Evaluation triggered for {model_id} (stub – evaluator not wired)"
    metrics = await ctx.evaluate_fn(model_id)
    return (
        f"Evaluation complete for {model_id}:\n"
        f"  WER={metrics.get('wer', 'n/a'):.3f}  "
        f"CER={metrics.get('cer', 'n/a'):.3f}  "
        f"RTF={metrics.get('rtf', 'n/a'):.4f}  "
        f"TTFT={metrics.get('ttft_ms', 'n/a'):.1f} ms"
    )


async def _handle_get_metrics(ctx: FactoryContext,
                               arguments: dict[str, Any]) -> str:
    model_id = arguments.get("model_id", "")
    if not model_id:
        return "ERROR: model_id is required"
    m = ctx.metrics_store.get(model_id)
    if m is None:
        return f"No metrics available for {model_id}"
    return (
        f"Metrics for {model_id}:\n"
        f"  WER={m.get('wer', 'n/a'):.3f}  "
        f"CER={m.get('cer', 'n/a'):.3f}  "
        f"RTF={m.get('rtf', 'n/a'):.4f}  "
        f"TTFT={m.get('ttft_ms', 'n/a'):.1f} ms"
    )


async def _handle_set_agent_hint(ctx: FactoryContext,
                                  arguments: dict[str, Any]) -> str:
    key   = arguments.get("key", "")
    value = arguments.get("value")
    if not key:
        return "ERROR: key is required"
    ctx.agent_hints[key] = value
    return f"Agent hint set: {key}={value}"


async def _handle_promote_model(ctx: FactoryContext,
                                 arguments: dict[str, Any]) -> str:
    model_id = arguments.get("model_id", "")
    if not model_id:
        return "ERROR: model_id is required"
    if ctx.promote_fn is None:
        return f"Promotion requested for {model_id} (stub – promote_fn not wired)"
    await ctx.promote_fn(model_id)
    return f"Model {model_id} promoted to production."


async def _handle_cluster_status(ctx: FactoryContext,
                                  arguments: dict[str, Any]) -> str:
    return (
        f"Cluster status:\n"
        f"  overall_load={ctx.cluster_load:.1%}\n"
        f"  active_hints={ctx.agent_hints}\n"
        f"  models_evaluated={len(ctx.metrics_store)}"
    )


# ─── Tool Registry ────────────────────────────────────────────────────────────

_TOOLS: list[dict[str, Any]] = [
    {
        "name": "list_models",
        "description": "List all registered ASR model candidates and their active status.",
        "inputSchema": {"type": "object", "properties": {}},
        "handler": _handle_list_models,
    },
    {
        "name": "evaluate_model",
        "description": "Trigger an on-demand evaluation of a specific ASR model against live telemetry.",
        "inputSchema": {
            "type": "object",
            "properties": {"model_id": {"type": "string", "description": "Model identifier"}},
            "required": ["model_id"],
        },
        "handler": _handle_evaluate_model,
    },
    {
        "name": "get_metrics",
        "description": "Retrieve the latest WER/CER/RTF metrics for an ASR model.",
        "inputSchema": {
            "type": "object",
            "properties": {"model_id": {"type": "string"}},
            "required": ["model_id"],
        },
        "handler": _handle_get_metrics,
    },
    {
        "name": "set_agent_hint",
        "description": "Override an Agent Hint key/value that will be attached to inference requests.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "key":   {"type": "string"},
                "value": {},
            },
            "required": ["key", "value"],
        },
        "handler": _handle_set_agent_hint,
    },
    {
        "name": "promote_model",
        "description": "Promote a model candidate to production serving.",
        "inputSchema": {
            "type": "object",
            "properties": {"model_id": {"type": "string"}},
            "required": ["model_id"],
        },
        "handler": _handle_promote_model,
    },
    {
        "name": "get_cluster_status",
        "description": "Return current AI Factory cluster health, load, and active agent hints.",
        "inputSchema": {"type": "object", "properties": {}},
        "handler": _handle_cluster_status,
    },
]


# ─── MCP Server ───────────────────────────────────────────────────────────────

async def run_mcp_server(ctx: FactoryContext) -> None:
    """
    Run the MCP server over stdio.

    The server exposes the AI Factory tools to any MCP-compatible client
    (e.g. Claude Desktop, custom LLM orchestrators).
    """
    if not _MCP_AVAILABLE:
        logger.warning("MCP not available – server not started")
        return

    server = MCPServer("asr-serve-factory")

    @server.list_tools()
    async def list_tools() -> list[Tool]:
        return [
            Tool(
                name        = t["name"],
                description = t["description"],
                inputSchema = t["inputSchema"],
            )
            for t in _TOOLS
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        for t in _TOOLS:
            if t["name"] == name:
                result = await t["handler"](ctx, arguments)
                return [TextContent(type="text", text=result)]
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream,
                         server.create_initialization_options())


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    ctx = FactoryContext()
    ctx.candidates = [
        {"model_id": "whisper-large-v3",      "version": "3.0", "is_active": True},
        {"model_id": "whisper-large-v3-turbo", "version": "3.0"},
        {"model_id": "canary-1b",              "version": "1.0"},
    ]
    ctx.metrics_store = {
        "whisper-large-v3": {"wer": 0.032, "cer": 0.015, "rtf": 0.008, "ttft_ms": 45.0},
    }

    asyncio.run(run_mcp_server(ctx))
