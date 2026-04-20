// SPDX-License-Identifier: BSD-2-Clause-Patent
// XDP/eBPF VAD hook – attaches to AWS ENA (Elastic Network Adapter).
//
// Parses incoming UDP/RTP audio frames and performs a fast energy-based
// pre-filter entirely in the kernel datapath.  Frames that survive the
// pre-filter are redirected to the AF_XDP socket bound to the DPDK
// userspace pipeline; all other frames are passed through to the normal
// kernel network stack.
//
// Build with:
//   clang -O2 -g -target bpf -D__TARGET_ARCH_x86 \
//         -I/usr/include/bpf                       \
//         -c xdp_vad.c -o xdp_vad.o

#include <linux/bpf.h>
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/udp.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_endian.h>

// ─── Constants ────────────────────────────────────────────────────────────────

// RTP header is 12 bytes; audio payload starts immediately after.
#define RTP_HDR_LEN      12
// 10 ms @ 16 kHz / 16-bit mono = 320 samples = 640 bytes
#define AUDIO_CHUNK_BYTES 640
// Simple energy threshold (sum of absolute sample values) for pre-filter.
// Tuned for -40 dBFS background noise floor.  The Wasm VAD in userspace
// applies a more accurate model; this just avoids waking the PMD for silence.
#define ENERGY_THRESHOLD  2048

// ─── BPF Maps ─────────────────────────────────────────────────────────────────

// AF_XDP socket map – populated by the userspace loader (vad_loader.c).
struct {
    __uint(type, BPF_MAP_TYPE_XSKMAP);
    __uint(max_entries, 64);
    __type(key, __u32);
    __type(value, __u32);
} xsk_map SEC(".maps");

// Per-CPU statistics counters for observability.
struct vad_stats {
    __u64 pkts_total;
    __u64 pkts_speech;
    __u64 pkts_silence;
    __u64 pkts_malformed;
};

struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, struct vad_stats);
} stats_map SEC(".maps");

// ─── Helpers ──────────────────────────────────────────────────────────────────

static __always_inline struct vad_stats *get_stats(void)
{
    __u32 key = 0;
    return bpf_map_lookup_elem(&stats_map, &key);
}

// Energy-based pre-filter: sum |sample| over the 10 ms chunk.
// Returns non-zero when energy exceeds ENERGY_THRESHOLD.
static __always_inline int energy_above_threshold(void *audio_start,
                                                   void *data_end)
{
    __s16 *samples = audio_start;
    __u32 energy   = 0;
    int   i;

    // Bounded loop (verifier-friendly) – process up to 320 samples.
#pragma unroll
    for (i = 0; i < 320; i++) {
        if ((void *)(samples + i + 1) > data_end)
            break;
        __s16 s = samples[i];
        energy += (s < 0) ? (__u32)(-s) : (__u32)s;
    }
    return energy > ENERGY_THRESHOLD;
}

// ─── XDP Entry Point ──────────────────────────────────────────────────────────

SEC("xdp")
int xdp_vad_prog(struct xdp_md *ctx)
{
    void *data     = (void *)(long)ctx->data;
    void *data_end = (void *)(long)ctx->data_end;

    struct vad_stats *stats = get_stats();
    if (!stats)
        return XDP_PASS;

    stats->pkts_total++;

    // ── Ethernet header ──────────────────────────────────────────────────────
    struct ethhdr *eth = data;
    if ((void *)(eth + 1) > data_end)
        goto malformed;
    if (bpf_ntohs(eth->h_proto) != ETH_P_IP)
        return XDP_PASS;

    // ── IPv4 header ───────────────────────────────────────────────────────────
    struct iphdr *ip = (void *)(eth + 1);
    if ((void *)(ip + 1) > data_end)
        goto malformed;
    if (ip->protocol != IPPROTO_UDP)
        return XDP_PASS;

    __u32 ip_hdr_len = ip->ihl * 4;
    if (ip_hdr_len < sizeof(*ip))
        goto malformed;

    // ── UDP header ────────────────────────────────────────────────────────────
    struct udphdr *udp = (void *)ip + ip_hdr_len;
    if ((void *)(udp + 1) > data_end)
        goto malformed;

    // Only process the well-known RTP audio port range (5000–5999).
    __u16 dst_port = bpf_ntohs(udp->dest);
    if (dst_port < 5000 || dst_port > 5999)
        return XDP_PASS;

    // ── RTP header + audio payload ────────────────────────────────────────────
    void *rtp_hdr   = (void *)(udp + 1);
    void *audio_buf = rtp_hdr + RTP_HDR_LEN;
    if (audio_buf + AUDIO_CHUNK_BYTES > data_end)
        goto malformed;

    // Energy pre-filter – fast path for silence.
    if (!energy_above_threshold(audio_buf, data_end)) {
        stats->pkts_silence++;
        return XDP_PASS; // silence: let kernel stack handle or drop
    }

    // Active speech: redirect to AF_XDP socket (queue 0) for DPDK pipeline.
    stats->pkts_speech++;
    return bpf_redirect_map(&xsk_map, ctx->rx_queue_index, XDP_PASS);

malformed:
    stats->pkts_malformed++;
    return XDP_PASS;
}

char LICENSE[] SEC("license") = "Dual BSD/GPL";
