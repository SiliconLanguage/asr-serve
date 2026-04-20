# NVIDIA Dynamo Orchestration & Prefill-Decode Disaggregation (PDD)

Integrates NVIDIA Dynamo as the AI-factory OS and implements native
Prefill-Decode Disaggregation (PDD) for the ASR inference tier.

## Architecture

```
VAD Edge Tier
     │  (UDP speech frames)
     ▼
┌─────────────────────────────────┐
│       Dynamo Smart Router       │  ← router.py
│  (role-based request routing)   │
└──────────┬──────────────────────┘
           │
     ┌─────┴──────┐
     ▼            ▼
 Prefill        Decode
 Workers        Workers
(GPU-heavy)   (VRAM-bound)
     │            │
     └─────┬──────┘
           │ NIXL RDMA (VRAM→VRAM)
           ▼
     ASR Transcript
```

## Components

| Path | Description |
|---|---|
| `src/router.py` | Dynamo Smart Router – role-based prefill/decode routing |
| `src/nixl_rdma.py` | NIXL RDMA transfer manager – zero-copy VRAM-to-VRAM |
| `src/dynamo_worker.py` | Dynamo worker base class – prefill and decode roles |
| `src/pdd_config.py` | PDD configuration dataclasses |
| `src/main.rs` | Rust orchestration shim (high-throughput request dispatch) |
| `Cargo.toml` | Rust workspace manifest |
| `configs/dynamo_cluster.yaml` | Dynamo cluster configuration |

## Quick Start

```bash
# Python (requires ai-dynamo, nixl packages)
pip install ai-dynamo nixl

python -m dynamo_orchestration.router \
    --prefill-workers 4 \
    --decode-workers  8 \
    --nixl-rdma-device mlx5_0

# Rust shim
cargo build --release
./target/release/dynamo-shim --config configs/dynamo_cluster.yaml
```
