# Blackwell-Native Inference Tier

vLLM and LLM-d worker configurations targeting NVIDIA Blackwell (GB200/B200)
GPUs, exploiting 5th-generation Tensor Cores and NVFP4 microscaling for
maximum ASR throughput.

## Components

| Path | Description |
|---|---|
| `src/worker.py` | Inference worker – wraps vLLM `AsyncLLMEngine` for ASR |
| `configs/vllm_worker.yaml` | vLLM worker configuration (Blackwell-tuned) |
| `configs/llm_d_worker.yaml` | LLM-d worker configuration |
| `configs/blackwell_optimizations.yaml` | Blackwell-specific kernel / precision settings |

## Quick Start

```bash
# Install dependencies
pip install vllm>=0.5 openai

# Start vLLM ASR worker
python src/worker.py \
    --config configs/vllm_worker.yaml \
    --model  /models/whisper-large-v3

# Or with LLM-d (Kubernetes mode)
kubectl apply -f configs/llm_d_worker.yaml
```

## Blackwell Optimisations

| Feature | Setting |
|---|---|
| Tensor Core generation | 5th-gen (FP8 / NVFP4 microscaling) |
| Weight precision | NVFP4 (4-bit NV microscaling format) |
| Activation precision | FP8-E4M3 |
| Flash Attention | FlashAttention-3 (Blackwell native) |
| KV-cache dtype | FP8-E4M3 |
| Speculative decoding | Eagle-2 draft model |
| Chunked prefill | enabled (max 2048 tokens/chunk) |
