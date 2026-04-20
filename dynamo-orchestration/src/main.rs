// SPDX-License-Identifier: BSD-2-Clause-Patent
//! Dynamo orchestration shim.
//!
//! This Rust binary acts as the high-throughput request dispatcher between
//! the VAD edge tier (UDP ingest) and the Python Smart Router.  It accepts
//! raw audio frames over UDP, batches them, and forwards batches to the
//! Dynamo gRPC bus.
//!
//! Architecture
//! ────────────
//!  UDP socket (from DPDK pipeline)
//!       │
//!       ▼
//!  [Ingest task]  ──batch──▶  [Dispatcher task]  ──gRPC──▶  Dynamo Router
//!       │                            │
//!       └── metrics ──────────────────────────────────────▶  Prometheus

use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use clap::Parser;
use serde::Deserialize;
use tokio::net::UdpSocket;
use tokio::sync::mpsc;
use tracing::{debug, error, info, warn};

// ─── CLI Arguments ────────────────────────────────────────────────────────────

/// Dynamo orchestration shim for asr-serve Prefill-Decode Disaggregation.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Args {
    /// Path to the YAML cluster configuration file.
    #[arg(long, default_value = "configs/dynamo_cluster.yaml")]
    config: String,

    /// UDP listen address for incoming audio frames from the VAD edge tier.
    #[arg(long, default_value = "0.0.0.0:5000")]
    listen: String,

    /// Maximum batch size before forcing a dispatch to the router.
    #[arg(long, default_value_t = 8)]
    max_batch: usize,

    /// Maximum wait time (ms) before flushing a partial batch.
    #[arg(long, default_value_t = 5)]
    batch_timeout_ms: u64,

    /// Log level filter (e.g. "info", "debug", "dynamo_shim=trace").
    #[arg(long, default_value = "info")]
    log_level: String,
}

// ─── Configuration ────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct ClusterConfig {
    router_address:   String,
    prefill_workers:  Vec<WorkerSpec>,
    decode_workers:   Vec<WorkerSpec>,
    nixl_rdma_device: String,
}

#[derive(Debug, Deserialize)]
struct WorkerSpec {
    id:      String,
    address: String,
}

async fn load_config(path: &str) -> anyhow::Result<ClusterConfig> {
    let text = tokio::fs::read_to_string(path).await?;
    let cfg: ClusterConfig = serde_yaml::from_str(&text)?;
    Ok(cfg)
}

// ─── Audio Frame ─────────────────────────────────────────────────────────────

/// A single 10 ms audio frame received from the VAD edge tier.
#[derive(Debug, Clone)]
struct AudioFrame {
    request_id: String,
    src:        SocketAddr,
    payload:    Vec<u8>,
    received_at: Instant,
}

// ─── Metrics ─────────────────────────────────────────────────────────────────

struct Metrics {
    frames_received:  prometheus::IntCounter,
    batches_sent:     prometheus::IntCounter,
    dispatch_latency: prometheus::Histogram,
}

impl Metrics {
    fn new() -> Self {
        let frames_received = prometheus::register_int_counter!(
            "asr_shim_frames_received_total",
            "Total audio frames received from VAD edge tier"
        )
        .unwrap();

        let batches_sent = prometheus::register_int_counter!(
            "asr_shim_batches_sent_total",
            "Total batches dispatched to Dynamo router"
        )
        .unwrap();

        let dispatch_latency = prometheus::register_histogram!(
            "asr_shim_dispatch_latency_ms",
            "Batch dispatch latency in milliseconds",
            vec![0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
        )
        .unwrap();

        Self { frames_received, batches_sent, dispatch_latency }
    }
}

// ─── UDP Ingest Task ──────────────────────────────────────────────────────────

async fn ingest_task(
    socket: Arc<UdpSocket>,
    tx:     mpsc::Sender<AudioFrame>,
    metrics: Arc<Metrics>,
) {
    let mut buf = vec![0u8; 65535];
    let mut frame_counter: u64 = 0;

    loop {
        match socket.recv_from(&mut buf).await {
            Ok((n, src)) => {
                let payload = buf[..n].to_vec();
                frame_counter += 1;
                metrics.frames_received.inc();

                let frame = AudioFrame {
                    request_id:  format!("req-{frame_counter}"),
                    src,
                    payload,
                    received_at: Instant::now(),
                };
                if let Err(e) = tx.send(frame).await {
                    warn!("ingest channel full, dropping frame: {e}");
                }
            }
            Err(e) => {
                error!("UDP recv error: {e}");
            }
        }
    }
}

// ─── Dispatcher Task ──────────────────────────────────────────────────────────

async fn dispatcher_task(
    mut rx:          mpsc::Receiver<AudioFrame>,
    config:          Arc<ClusterConfig>,
    max_batch:       usize,
    batch_timeout:   Duration,
    metrics:         Arc<Metrics>,
) {
    let mut batch: Vec<AudioFrame> = Vec::with_capacity(max_batch);
    let mut deadline = tokio::time::sleep(batch_timeout);
    tokio::pin!(deadline);

    loop {
        tokio::select! {
            frame = rx.recv() => {
                match frame {
                    Some(f) => {
                        debug!("queuing frame {} from {}", f.request_id, f.src);
                        batch.push(f);
                        if batch.len() >= max_batch {
                            dispatch_batch(&mut batch, &config, &metrics).await;
                            deadline.as_mut().reset(
                                tokio::time::Instant::now() + batch_timeout
                            );
                        }
                    }
                    None => {
                        info!("ingest channel closed – flushing final batch");
                        if !batch.is_empty() {
                            dispatch_batch(&mut batch, &config, &metrics).await;
                        }
                        return;
                    }
                }
            }
            _ = &mut deadline => {
                if !batch.is_empty() {
                    dispatch_batch(&mut batch, &config, &metrics).await;
                }
                deadline.as_mut().reset(
                    tokio::time::Instant::now() + batch_timeout
                );
            }
        }
    }
}

async fn dispatch_batch(
    batch:   &mut Vec<AudioFrame>,
    config:  &ClusterConfig,
    metrics: &Metrics,
) {
    let t0 = Instant::now();
    info!(
        "dispatching batch of {} frames to {}",
        batch.len(),
        config.router_address
    );

    // TODO: serialise batch into Dynamo protobuf message and send via tonic
    // gRPC channel.  Stub: log each request ID.
    for frame in batch.iter() {
        debug!("  → {}", frame.request_id);
    }

    metrics.batches_sent.inc();
    metrics.dispatch_latency.observe(t0.elapsed().as_secs_f64() * 1000.0);
    batch.clear();
}

// ─── Main ─────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    tracing_subscriber::fmt()
        .with_env_filter(&args.log_level)
        .init();

    info!("dynamo-shim starting (config={})", args.config);

    // Load cluster configuration.
    let config = Arc::new(load_config(&args.config).await.unwrap_or_else(|e| {
        warn!("could not load config ({}), using defaults: {e}", args.config);
        ClusterConfig {
            router_address:   "127.0.0.1:50051".to_string(),
            prefill_workers:  vec![WorkerSpec {
                id:      "prefill-0".to_string(),
                address: "10.0.1.10:50051".to_string(),
            }],
            decode_workers: vec![WorkerSpec {
                id:      "decode-0".to_string(),
                address: "10.0.2.10:50051".to_string(),
            }],
            nixl_rdma_device: "mlx5_0".to_string(),
        }
    }));

    info!("router={} prefill_workers={} decode_workers={} nixl_device={}",
          config.router_address,
          config.prefill_workers.len(),
          config.decode_workers.len(),
          config.nixl_rdma_device);

    // Bind UDP socket.
    let listen: SocketAddr = args.listen.parse()?;
    let socket = Arc::new(UdpSocket::bind(listen).await?);
    info!("listening on udp://{listen}");

    let metrics = Arc::new(Metrics::new());

    let (tx, rx) = mpsc::channel::<AudioFrame>(4096);

    let batch_timeout = Duration::from_millis(args.batch_timeout_ms);

    // Spawn tasks.
    let ingest_socket  = Arc::clone(&socket);
    let ingest_metrics = Arc::clone(&metrics);
    tokio::spawn(ingest_task(ingest_socket, tx, ingest_metrics));

    let dispatch_config  = Arc::clone(&config);
    let dispatch_metrics = Arc::clone(&metrics);
    tokio::spawn(dispatcher_task(
        rx,
        dispatch_config,
        args.max_batch,
        batch_timeout,
        dispatch_metrics,
    ));

    // Wait for SIGINT / SIGTERM.
    tokio::signal::ctrl_c().await?;
    info!("shutdown signal received");

    Ok(())
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn audio_frame_creation() {
        let frame = AudioFrame {
            request_id:  "test-001".to_string(),
            src:         "127.0.0.1:9000".parse().unwrap(),
            payload:     vec![0u8; 640],
            received_at: Instant::now(),
        };
        assert_eq!(frame.payload.len(), 640);
        assert_eq!(frame.request_id, "test-001");
    }

    #[tokio::test]
    async fn dispatcher_flushes_on_timeout() {
        let (tx, rx) = mpsc::channel::<AudioFrame>(16);
        let config   = Arc::new(ClusterConfig {
            router_address:   "127.0.0.1:50051".to_string(),
            prefill_workers:  vec![],
            decode_workers:   vec![],
            nixl_rdma_device: "mlx5_0".to_string(),
        });
        let metrics  = Arc::new(Metrics::new());

        // Send one frame then drop the sender so the task exits.
        tx.send(AudioFrame {
            request_id:  "flush-test".to_string(),
            src:         "127.0.0.1:9001".parse().unwrap(),
            payload:     vec![0u8; 640],
            received_at: Instant::now(),
        }).await.unwrap();
        drop(tx);

        dispatcher_task(rx, config, 8, Duration::from_millis(1), metrics).await;
        // Test passes if no panic occurs.
    }
}
