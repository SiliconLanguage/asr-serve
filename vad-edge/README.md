# Zero-Kernel VAD Edge Tier

Implements the eBPF/XDP-based Voice Activity Detection (VAD) edge tier
deployed on AWS Elastic Network Adapters (ENA) with vIOMMU PCI passthrough.
Audio arrives in 10 ms chunks; active-speech frames are routed zero-copy via
UDP to the upstream ASR inference tier.

## Components

| Path | Description |
|---|---|
| `src/xdp_vad.c` | eBPF/XDP kernel program – attaches to ENA, parses RTP/UDP audio frames |
| `src/dpdk_pipeline.c` | DPDK userspace pipeline – polls PMD ring, calls Wasm VAD, forwards speech |
| `src/vad_loader.c` | Userspace BPF loader – pins maps, attaches XDP program via `libbpf` |
| `wasm/vad_module.wat` | WebAssembly VAD module (text format) processed via wasmtime/wasmer |
| `CMakeLists.txt` | CMake build (requires DPDK ≥ 23.11, libbpf ≥ 1.3, wasmtime-c-api) |

## Quick Start

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Load XDP program onto ENA interface (requires CAP_NET_ADMIN)
sudo ./build/vad_loader --iface eth0 --wasm wasm/vad_module.wasm

# Run DPDK pipeline (huge-pages must be configured)
sudo ./build/dpdk_pipeline -- --upstream-ip 10.0.0.10 --upstream-port 5000
```

## Prerequisites

- Linux kernel ≥ 6.1 (BTF, BPF CO-RE)
- DPDK 23.11 (ENA PMD)
- libbpf ≥ 1.3
- wasmtime C API ≥ 20
- clang/llvm ≥ 16 (for eBPF bytecode compilation)
