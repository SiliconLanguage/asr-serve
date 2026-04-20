# asr-serve
A zero-kernel AI speech-to-text inference engine. Routes active speech via Graviton4 VAD (eBPF/XDP) to hybrid Inferentia/GPU ASR tier. Orchestrated by NVIDIA Dynamo for Prefill-Decode Disaggregation (PDD), vLLM &amp; Blackwell NVFP4 optimizations. Governed by the Adaptive Agentic AI foundry platform evaluating state-of-the-art open-source ASR models.
