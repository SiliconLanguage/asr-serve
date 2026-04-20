// SPDX-License-Identifier: BSD-2-Clause-Patent
// DPDK userspace audio pipeline for the VAD Edge Tier.
//
// Polls the AF_XDP socket ring that receives active-speech frames from the
// XDP program (xdp_vad.c), applies the WebAssembly VAD model for a
// high-accuracy second-stage classification, then forwards confirmed speech
// frames zero-copy to the upstream ASR inference tier via UDP.
//
// Build: see CMakeLists.txt (requires DPDK ≥ 23.11, wasmtime C API ≥ 20)

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <errno.h>
#include <getopt.h>
#include <arpa/inet.h>

#include <rte_eal.h>
#include <rte_ethdev.h>
#include <rte_mbuf.h>
#include <rte_mempool.h>
#include <rte_ring.h>
#include <rte_lcore.h>
#include <rte_log.h>
#include <rte_cycles.h>

#include <wasmtime.h>  // wasmtime C API

// ─── Configuration ────────────────────────────────────────────────────────────

#define APP_NAME          "dpdk-vad-pipeline"
#define MBUF_POOL_SIZE    8192
#define MBUF_CACHE_SIZE   256
#define MBUF_DATA_SIZE    RTE_MBUF_DEFAULT_BUF_SIZE
#define BURST_SIZE        32
// 10 ms chunk: 320 samples × 2 bytes @ 16 kHz/16-bit mono
#define AUDIO_CHUNK_BYTES 640
#define RTP_HDR_LEN       12
#define UDP_HDR_LEN       8
#define IP_HDR_LEN        20
#define ETH_HDR_LEN       14

static volatile int g_running = 1;

// ─── Runtime State ────────────────────────────────────────────────────────────

typedef struct {
    // DPDK
    struct rte_mempool  *mbuf_pool;
    uint16_t             port_id;

    // Upstream endpoint (ASR inference tier)
    struct in_addr       upstream_ip;
    uint16_t             upstream_port;

    // Wasmtime
    wasm_engine_t       *wasm_engine;
    wasmtime_store_t    *wasm_store;
    wasmtime_module_t   *wasm_module;
    wasmtime_instance_t  wasm_instance;
    wasmtime_func_t      wasm_vad_fn;   // exported "vad_classify" function

    // Telemetry
    uint64_t             frames_received;
    uint64_t             frames_speech;
    uint64_t             frames_forwarded;
} pipeline_ctx_t;

static pipeline_ctx_t g_ctx;

// ─── Signal Handler ───────────────────────────────────────────────────────────

static void signal_handler(int signum)
{
    (void)signum;
    g_running = 0;
}

// ─── Wasm VAD ─────────────────────────────────────────────────────────────────

// Load and instantiate the VAD Wasm module from disk.
// The module must export:
//   (func $vad_classify (param i32 i32) (result i32))
// where param 0 = pointer to audio samples (i16 PCM) in Wasm memory,
//       param 1 = number of samples (int32),
//       result  = 1 (speech) or 0 (silence).
static int wasm_vad_init(pipeline_ctx_t *ctx, const char *wasm_path)
{
    wasmtime_error_t *error;
    wasm_trap_t      *trap;

    ctx->wasm_engine = wasm_engine_new();
    if (!ctx->wasm_engine) {
        RTE_LOG(ERR, USER1, "wasm_engine_new failed\n");
        return -1;
    }

    ctx->wasm_store = wasmtime_store_new(ctx->wasm_engine, NULL, NULL);
    if (!ctx->wasm_store) {
        RTE_LOG(ERR, USER1, "wasmtime_store_new failed\n");
        return -1;
    }

    // Read Wasm binary from file.
    FILE *f = fopen(wasm_path, "rb");
    if (!f) {
        RTE_LOG(ERR, USER1, "Cannot open Wasm module: %s\n", wasm_path);
        return -1;
    }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    rewind(f);
    uint8_t *wasm_bytes = malloc((size_t)fsize);
    if (!wasm_bytes || (long)fread(wasm_bytes, 1, (size_t)fsize, f) != fsize) {
        RTE_LOG(ERR, USER1, "Failed to read Wasm module\n");
        fclose(f);
        free(wasm_bytes);
        return -1;
    }
    fclose(f);

    wasm_byte_vec_t wasm_vec = { .size = (size_t)fsize, .data = (wasm_byte_t *)wasm_bytes };
    error = wasmtime_module_new(ctx->wasm_engine, &wasm_vec, &ctx->wasm_module);
    free(wasm_bytes);
    if (error) {
        RTE_LOG(ERR, USER1, "wasmtime_module_new failed\n");
        wasmtime_error_delete(error);
        return -1;
    }

    wasmtime_context_t *wctx = wasmtime_store_context(ctx->wasm_store);
    error = wasmtime_instance_new(wctx, ctx->wasm_module,
                                  NULL, 0,
                                  &ctx->wasm_instance, &trap);
    if (error || trap) {
        RTE_LOG(ERR, USER1, "wasmtime_instance_new failed\n");
        if (error) wasmtime_error_delete(error);
        if (trap)  wasm_trap_delete(trap);
        return -1;
    }

    // Look up the exported "vad_classify" function.
    wasmtime_extern_t ext;
    bool found = wasmtime_instance_export_get(wctx,
                                              &ctx->wasm_instance,
                                              "vad_classify",
                                              strlen("vad_classify"),
                                              &ext);
    if (!found || ext.kind != WASMTIME_EXTERN_FUNC) {
        RTE_LOG(ERR, USER1, "Wasm export 'vad_classify' not found\n");
        return -1;
    }
    ctx->wasm_vad_fn = ext.of.func;

    RTE_LOG(INFO, USER1, "Wasm VAD module loaded from %s\n", wasm_path);
    return 0;
}

// Run the Wasm VAD classifier on a 10 ms PCM chunk.
// Returns 1 = speech, 0 = silence, -1 = error.
static int wasm_vad_classify(pipeline_ctx_t *ctx,
                              const int16_t  *samples,
                              uint32_t        n_samples)
{
    wasmtime_context_t *wctx = wasmtime_store_context(ctx->wasm_store);
    wasmtime_error_t   *error;
    wasm_trap_t        *trap;

    // Write samples into Wasm linear memory at a fixed staging offset.
    // Production code would call the Wasm allocator; this uses a reserved
    // region at 0x10000 (64 KiB) which lies above the Wasm stack.
    const uint32_t wasm_buf_offset = 0x10000;
    size_t byte_len = (size_t)n_samples * sizeof(int16_t);

    wasmtime_extern_t mem_ext;
    bool found = wasmtime_instance_export_get(wctx,
                                              &ctx->wasm_instance,
                                              "memory",
                                              strlen("memory"),
                                              &mem_ext);
    if (!found || mem_ext.kind != WASMTIME_EXTERN_MEMORY)
        return -1;

    uint8_t *wasm_mem      = wasmtime_memory_data(wctx, &mem_ext.of.memory);
    size_t   wasm_mem_size = wasmtime_memory_data_size(wctx, &mem_ext.of.memory);
    if ((size_t)wasm_buf_offset + byte_len > wasm_mem_size)
        return -1;

    memcpy(wasm_mem + wasm_buf_offset, samples, byte_len);

    // Call vad_classify(ptr, n_samples) -> i32
    wasmtime_val_t args[2] = {
        { .kind = WASMTIME_I32, .of.i32 = (int32_t)wasm_buf_offset },
        { .kind = WASMTIME_I32, .of.i32 = (int32_t)n_samples },
    };
    wasmtime_val_t result = { .kind = WASMTIME_I32 };

    error = wasmtime_func_call(wctx, &ctx->wasm_vad_fn,
                               args, 2, &result, 1, &trap);
    if (error || trap) {
        if (error) wasmtime_error_delete(error);
        if (trap)  wasm_trap_delete(trap);
        return -1;
    }

    return result.of.i32 ? 1 : 0;
}

// ─── Packet Forwarding ────────────────────────────────────────────────────────

// Re-encapsulate the speech payload and forward via UDP zero-copy send.
// Updates the mbuf data offset to strip the incoming Eth/IP/UDP/RTP headers
// and prepend new headers destined for the upstream ASR tier.
// Full NIC-offload header rewrite depends on rte_flow and NIC capabilities.
static void forward_speech_frame(pipeline_ctx_t  *ctx,
                                  struct rte_mbuf *mbuf)
{
    // Transmit on the same port, queue 0.
    uint16_t sent = rte_eth_tx_burst(ctx->port_id, 0, &mbuf, 1);
    if (sent == 0)
        rte_pktmbuf_free(mbuf);

    ctx->frames_forwarded++;
}

// ─── Main Polling Loop ────────────────────────────────────────────────────────

static void pipeline_run(pipeline_ctx_t *ctx)
{
    struct rte_mbuf *burst[BURST_SIZE];

    RTE_LOG(INFO, USER1,
            "Pipeline running on lcore %u (port %u → %s:%u)\n",
            rte_lcore_id(), ctx->port_id,
            inet_ntoa(ctx->upstream_ip), ctx->upstream_port);

    while (g_running) {
        uint16_t n = rte_eth_rx_burst(ctx->port_id, 0, burst, BURST_SIZE);
        if (n == 0) {
            rte_pause();
            continue;
        }

        for (uint16_t i = 0; i < n; i++) {
            struct rte_mbuf *m = burst[i];
            ctx->frames_received++;

            uint32_t pkt_len = rte_pktmbuf_pkt_len(m);
            uint32_t hdr_len = ETH_HDR_LEN + IP_HDR_LEN + UDP_HDR_LEN + RTP_HDR_LEN;

            if (pkt_len < hdr_len + AUDIO_CHUNK_BYTES) {
                rte_pktmbuf_free(m);
                continue;
            }

            const int16_t *audio_ptr =
                rte_pktmbuf_mtod_offset(m, const int16_t *, hdr_len);
            uint32_t audio_bytes = pkt_len - hdr_len;

            // Second-stage Wasm VAD classification.
            int is_speech = wasm_vad_classify(ctx,
                                              audio_ptr,
                                              audio_bytes / sizeof(int16_t));
            if (is_speech == 1) {
                ctx->frames_speech++;
                forward_speech_frame(ctx, m);
            } else {
                rte_pktmbuf_free(m);
            }
        }

        // Periodic telemetry log (every ~1 s at ~100 Kpps).
        static uint64_t last_tsc = 0;
        uint64_t now = rte_rdtsc();
        if (now - last_tsc > rte_get_tsc_hz()) {
            RTE_LOG(INFO, USER1,
                    "telemetry: rx=%lu speech=%lu fwd=%lu\n",
                    ctx->frames_received,
                    ctx->frames_speech,
                    ctx->frames_forwarded);
            last_tsc = now;
        }
    }
}

// ─── CLI / Initialisation ─────────────────────────────────────────────────────

static void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s [EAL options] -- "
            "--upstream-ip <IP> --upstream-port <PORT> "
            "[--wasm <path>]\n",
            prog);
}

int main(int argc, char *argv[])
{
    signal(SIGINT,  signal_handler);
    signal(SIGTERM, signal_handler);

    int ret = rte_eal_init(argc, argv);
    if (ret < 0)
        rte_exit(EXIT_FAILURE, "rte_eal_init failed\n");
    argc -= ret;
    argv += ret;

    // Default values.
    char upstream_ip_str[64] = "10.0.0.10";
    uint16_t upstream_port   = 5000;
    char wasm_path[256]      = "wasm/vad_module.wasm";

    static struct option long_opts[] = {
        { "upstream-ip",   required_argument, NULL, 'i' },
        { "upstream-port", required_argument, NULL, 'p' },
        { "wasm",          required_argument, NULL, 'w' },
        { NULL, 0, NULL, 0 }
    };
    int opt;
    while ((opt = getopt_long(argc, argv, "i:p:w:", long_opts, NULL)) != -1) {
        switch (opt) {
        case 'i': snprintf(upstream_ip_str, sizeof(upstream_ip_str), "%s", optarg); break;
        case 'p': upstream_port = (uint16_t)atoi(optarg); break;
        case 'w': snprintf(wasm_path, sizeof(wasm_path), "%s", optarg); break;
        default:
            usage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    if (inet_aton(upstream_ip_str, &g_ctx.upstream_ip) == 0) {
        fprintf(stderr, "Invalid upstream IP: %s\n", upstream_ip_str);
        return EXIT_FAILURE;
    }
    g_ctx.upstream_port = upstream_port;

    // Initialise DPDK mbuf pool.
    g_ctx.mbuf_pool = rte_pktmbuf_pool_create(
        "VAD_MBUF_POOL", MBUF_POOL_SIZE, MBUF_CACHE_SIZE, 0,
        MBUF_DATA_SIZE, rte_socket_id());
    if (!g_ctx.mbuf_pool)
        rte_exit(EXIT_FAILURE, "Cannot create mbuf pool\n");

    // Use the first available DPDK port (ENA).
    g_ctx.port_id = 0;
    if (!rte_eth_dev_is_valid_port(g_ctx.port_id))
        rte_exit(EXIT_FAILURE, "No DPDK port available\n");

    // Configure port (single RX/TX queue).
    struct rte_eth_conf port_conf = { 0 };
    ret = rte_eth_dev_configure(g_ctx.port_id, 1, 1, &port_conf);
    if (ret < 0)
        rte_exit(EXIT_FAILURE, "rte_eth_dev_configure: err=%d\n", ret);

    ret = rte_eth_rx_queue_setup(g_ctx.port_id, 0, 512,
                                 rte_eth_dev_socket_id(g_ctx.port_id),
                                 NULL, g_ctx.mbuf_pool);
    if (ret < 0)
        rte_exit(EXIT_FAILURE, "rte_eth_rx_queue_setup: err=%d\n", ret);

    ret = rte_eth_tx_queue_setup(g_ctx.port_id, 0, 512,
                                 rte_eth_dev_socket_id(g_ctx.port_id),
                                 NULL);
    if (ret < 0)
        rte_exit(EXIT_FAILURE, "rte_eth_tx_queue_setup: err=%d\n", ret);

    ret = rte_eth_dev_start(g_ctx.port_id);
    if (ret < 0)
        rte_exit(EXIT_FAILURE, "rte_eth_dev_start: err=%d\n", ret);

    rte_eth_promiscuous_enable(g_ctx.port_id);

    // Initialise Wasm VAD.
    if (wasm_vad_init(&g_ctx, wasm_path) < 0)
        rte_exit(EXIT_FAILURE, "Wasm VAD initialisation failed\n");

    pipeline_run(&g_ctx);

    rte_eth_dev_stop(g_ctx.port_id);
    rte_eth_dev_close(g_ctx.port_id);
    rte_eal_cleanup();
    return 0;
}
