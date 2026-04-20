# Self-Improving Agentic AI Factory

Autonomous evaluation and control loop for the asr-serve AI Factory.
Uses the Model Context Protocol (MCP) and NeMo Agent Toolkit (NAT) to
continuously benchmark open-source ASR models against live telemetry and
attach routing hints to inference requests.

## Architecture

```
┌──────────────────────────────────────────────────┐
│              Agentic AI Factory                   │
│                                                   │
│  ┌─────────────┐   ┌──────────────┐               │
│  │ MCP Client  │   │  NAT Agent   │               │
│  │ (src/mcp_   │   │  (src/nat_   │               │
│  │  client.py) │   │   agent.py)  │               │
│  └──────┬──────┘   └──────┬───────┘               │
│         │                 │                        │
│         ▼                 ▼                        │
│  ┌─────────────────────────────────┐               │
│  │       Evaluator Loop            │               │
│  │     (src/evaluator.py)          │               │
│  └──────────────┬──────────────────┘               │
│                 │ WER / CER / RTF metrics           │
│                 ▼                                  │
│  ┌─────────────────────────────────┐               │
│  │    Agent Hints Manager          │               │
│  │    (src/agent_hints.py)         │               │
│  └─────────────────────────────────┘               │
└──────────────────────────────────────────────────┘
          │                    │
          ▼                    ▼
   Dynamo Router         Inference Workers
  (routing hints)       (KV-cache TTL hints)
```

## Components

| Path | Description |
|---|---|
| `src/evaluator.py` | Continuous model evaluation loop (WER/CER/RTF benchmarks) |
| `src/mcp_client.py` | MCP client – registers tools and exposes factory context |
| `src/nat_agent.py` | NeMo Agent Toolkit agent – autonomous model selection |
| `src/agent_hints.py` | Agent Hints Manager – attaches hints to inference requests |
| `main.go` | Go control plane – low-latency hint propagation to Dynamo |
| `configs/factory.yaml` | AI Factory configuration |

## Quick Start

```bash
# Python agent (requires nemo-agent-toolkit, mcp)
pip install nemo-agent-toolkit mcp openai

python src/evaluator.py \
    --config configs/factory.yaml \
    --inference-endpoint http://10.0.0.5:8000

# Go control plane
go build -o factory-control ./...
./factory-control --config configs/factory.yaml
```
