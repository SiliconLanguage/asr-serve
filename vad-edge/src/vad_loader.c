// SPDX-License-Identifier: BSD-2-Clause-Patent
// Userspace BPF loader for the XDP VAD program.
//
// Uses libbpf to:
//   1. Load xdp_vad.o (compiled by clang -target bpf)
//   2. Attach the XDP program to the requested ENA network interface
//   3. Pin the AF_XDP socket map for use by dpdk_pipeline
//   4. Poll the per-CPU stats map and print telemetry

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <signal.h>
#include <unistd.h>
#include <getopt.h>
#include <net/if.h>
#include <sys/resource.h>

#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include <linux/if_link.h>

// ─── Types ────────────────────────────────────────────────────────────────────

struct vad_stats {
    uint64_t pkts_total;
    uint64_t pkts_speech;
    uint64_t pkts_silence;
    uint64_t pkts_malformed;
};

// ─── Globals ──────────────────────────────────────────────────────────────────

static volatile int g_running = 1;

static void signal_handler(int sig)
{
    (void)sig;
    g_running = 0;
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

static void print_stats(int stats_map_fd, uint32_t nr_cpus)
{
    struct vad_stats aggregate = { 0 };
    uint32_t key = 0;
    struct vad_stats *values = calloc(nr_cpus, sizeof(*values));
    if (!values)
        return;

    if (bpf_map_lookup_elem(stats_map_fd, &key, values) == 0) {
        for (uint32_t i = 0; i < nr_cpus; i++) {
            aggregate.pkts_total     += values[i].pkts_total;
            aggregate.pkts_speech    += values[i].pkts_speech;
            aggregate.pkts_silence   += values[i].pkts_silence;
            aggregate.pkts_malformed += values[i].pkts_malformed;
        }
        printf("[vad_loader] total=%lu speech=%lu silence=%lu malformed=%lu\n",
               aggregate.pkts_total, aggregate.pkts_speech,
               aggregate.pkts_silence, aggregate.pkts_malformed);
    }
    free(values);
}

static void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s --iface <ifname> [--obj <xdp_vad.o>] [--xsk-queue <q>]\n",
            prog);
}

// ─── Main ─────────────────────────────────────────────────────────────────────

int main(int argc, char *argv[])
{
    signal(SIGINT,  signal_handler);
    signal(SIGTERM, signal_handler);

    char iface[IF_NAMESIZE] = { 0 };
    char obj_path[256]      = "xdp_vad.o";
    int  xsk_queue          = 0;

    static struct option long_opts[] = {
        { "iface",     required_argument, NULL, 'i' },
        { "obj",       required_argument, NULL, 'o' },
        { "xsk-queue", required_argument, NULL, 'q' },
        { "wasm",      required_argument, NULL, 'w' },
        { NULL, 0, NULL, 0 }
    };
    int opt;
    while ((opt = getopt_long(argc, argv, "i:o:q:w:", long_opts, NULL)) != -1) {
        switch (opt) {
        case 'i': snprintf(iface, sizeof(iface), "%s", optarg);    break;
        case 'o': snprintf(obj_path, sizeof(obj_path), "%s", optarg); break;
        case 'q': xsk_queue = atoi(optarg);                        break;
        case 'w': /* forwarded to dpdk_pipeline */                 break;
        default:
            usage(argv[0]);
            return EXIT_FAILURE;
        }
    }

    if (iface[0] == '\0') {
        fprintf(stderr, "Error: --iface is required\n");
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    uint32_t ifindex = if_nametoindex(iface);
    if (!ifindex) {
        fprintf(stderr, "Interface '%s' not found: %s\n", iface, strerror(errno));
        return EXIT_FAILURE;
    }

    // Raise RLIMIT_MEMLOCK so that BPF maps can be locked in memory.
    struct rlimit rl = { RLIM_INFINITY, RLIM_INFINITY };
    if (setrlimit(RLIMIT_MEMLOCK, &rl) < 0) {
        perror("setrlimit(RLIMIT_MEMLOCK)");
        return EXIT_FAILURE;
    }

    // Load BPF object.
    struct bpf_object *obj = bpf_object__open_file(obj_path, NULL);
    if (!obj) {
        fprintf(stderr, "bpf_object__open_file(%s) failed: %s\n",
                obj_path, strerror(errno));
        return EXIT_FAILURE;
    }

    if (bpf_object__load(obj)) {
        fprintf(stderr, "bpf_object__load failed: %s\n", strerror(errno));
        bpf_object__close(obj);
        return EXIT_FAILURE;
    }

    // Find and attach the XDP program.
    struct bpf_program *prog = bpf_object__find_program_by_name(obj, "xdp_vad_prog");
    if (!prog) {
        fprintf(stderr, "XDP program 'xdp_vad_prog' not found in %s\n", obj_path);
        bpf_object__close(obj);
        return EXIT_FAILURE;
    }

    int prog_fd = bpf_program__fd(prog);
    if (bpf_xdp_attach(ifindex, prog_fd, XDP_FLAGS_DRV_MODE, NULL) < 0) {
        // Fall back to SKB (generic XDP) mode for environments without DRV support.
        fprintf(stderr, "DRV mode unavailable, retrying with SKB mode\n");
        if (bpf_xdp_attach(ifindex, prog_fd, XDP_FLAGS_SKB_MODE, NULL) < 0) {
            perror("bpf_xdp_attach");
            bpf_object__close(obj);
            return EXIT_FAILURE;
        }
    }
    printf("[vad_loader] XDP program attached to %s (ifindex=%u)\n",
           iface, ifindex);

    // Pin the xsk_map to the BPF filesystem so the DPDK pipeline can open it
    // by path and register its AF_XDP socket file descriptor.
    int xsk_map_fd = bpf_object__find_map_fd_by_name(obj, "xsk_map");
    if (xsk_map_fd < 0) {
        fprintf(stderr, "xsk_map not found in BPF object\n");
    } else {
        const char *pin_path = "/sys/fs/bpf/vad_xsk_map";
        if (bpf_map__pin(bpf_object__find_map_by_name(obj, "xsk_map"), pin_path) < 0)
            fprintf(stderr, "Warning: could not pin xsk_map to %s\n", pin_path);
        else
            printf("[vad_loader] xsk_map pinned at %s (queue %d)\n",
                   pin_path, xsk_queue);
    }

    // Retrieve stats map fd for telemetry.
    int stats_map_fd = bpf_object__find_map_fd_by_name(obj, "stats_map");
    uint32_t nr_cpus = (uint32_t)libbpf_num_possible_cpus();

    printf("[vad_loader] Running. Press Ctrl-C to detach.\n");

    while (g_running) {
        sleep(1);
        if (stats_map_fd >= 0)
            print_stats(stats_map_fd, nr_cpus);
    }

    // Detach XDP program on exit (try both modes; one will no-op).
    bpf_xdp_detach(ifindex, XDP_FLAGS_DRV_MODE, NULL);
    bpf_xdp_detach(ifindex, XDP_FLAGS_SKB_MODE, NULL);
    bpf_object__close(obj);

    printf("[vad_loader] XDP program detached from %s\n", iface);
    return 0;
}
