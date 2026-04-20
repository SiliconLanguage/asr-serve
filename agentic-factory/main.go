// SPDX-License-Identifier: BSD-2-Clause-Patent
// AI Factory control plane – Go low-latency hint propagation.
//
// This binary runs on the Graviton4 orchestration node and is responsible
// for the hot path of Agent Hint propagation:
//
//   Agentic Factory (Python) ──gRPC──▶ factory-control ──UDP──▶ Dynamo Shim
//
// Using Go (rather than Python) here achieves predictable sub-millisecond
// hint propagation without GIL or GC pauses.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"log/slog"
	"net"
	"net/http"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"
)

// ─── Configuration ────────────────────────────────────────────────────────────

type Config struct {
	// HTTP API listen address (accepts hint updates from Python evaluator).
	HTTPAddr string `json:"http_addr"`
	// UDP downstream address (Dynamo Rust shim).
	DynamoShimAddr string `json:"dynamo_shim_addr"`
	// Hint broadcast interval.
	BroadcastIntervalMs int `json:"broadcast_interval_ms"`
}

func defaultConfig() Config {
	return Config{
		HTTPAddr:            ":7070",
		DynamoShimAddr:      "127.0.0.1:7071",
		BroadcastIntervalMs: 100,
	}
}

// ─── Agent Hints ─────────────────────────────────────────────────────────────

// HintProfile mirrors agentic-factory/src/agent_hints.py:HintProfile.
type HintProfile struct {
	ModelID   string `json:"model_id"`
	Priority  int    `json:"priority"`
	KVTTLMs   int    `json:"kv_ttl_ms"`
	Debug     bool   `json:"debug"`
	UpdatedAt int64  `json:"updated_at_unix_ms"`
}

// HintsStore is a concurrency-safe store for the active hint profile.
type HintsStore struct {
	mu      sync.RWMutex
	active  HintProfile
	history []HintProfile
}

func NewHintsStore() *HintsStore {
	return &HintsStore{
		active: HintProfile{
			ModelID:  "whisper-large-v3",
			Priority: 5,
			KVTTLMs:  500,
		},
	}
}

func (s *HintsStore) Update(p HintProfile) {
	s.mu.Lock()
	defer s.mu.Unlock()
	p.UpdatedAt = time.Now().UnixMilli()
	s.history = append(s.history, s.active)
	if len(s.history) > 100 {
		s.history = s.history[1:]
	}
	s.active = p
	slog.Info("agent hints updated",
		"model_id", p.ModelID,
		"priority", p.Priority,
		"kv_ttl_ms", p.KVTTLMs,
	)
}

func (s *HintsStore) Active() HintProfile {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.active
}

// ─── HTTP API ─────────────────────────────────────────────────────────────────

// PUT /hints  – update the active hint profile.
// GET /hints  – read the current active hint profile.
// GET /health – liveness probe.
func buildHTTPMux(store *HintsStore) *http.ServeMux {
	mux := http.NewServeMux()

	mux.HandleFunc("/hints", func(w http.ResponseWriter, r *http.Request) {
		switch r.Method {
		case http.MethodGet:
			w.Header().Set("Content-Type", "application/json")
			if err := json.NewEncoder(w).Encode(store.Active()); err != nil {
				slog.Error("encode hints", "err", err)
			}

		case http.MethodPut:
			var p HintProfile
			if err := json.NewDecoder(r.Body).Decode(&p); err != nil {
				http.Error(w, "invalid JSON: "+err.Error(), http.StatusBadRequest)
				return
			}
			store.Update(p)
			w.WriteHeader(http.StatusNoContent)

		default:
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		}
	})

	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if _, err := w.Write([]byte(`{"status":"ok"}`)); err != nil {
			slog.Error("write health", "err", err)
		}
	})

	return mux
}

// ─── UDP Broadcaster ─────────────────────────────────────────────────────────

// broadcastLoop periodically sends the active hint profile to the Dynamo shim
// as a compact JSON datagram.  The Rust shim reads these on its hint listener
// socket and applies them to the in-flight request batch.
func broadcastLoop(ctx context.Context,
	store    *HintsStore,
	shimAddr string,
	interval time.Duration,
) {
	udpAddr, err := net.ResolveUDPAddr("udp4", shimAddr)
	if err != nil {
		slog.Error("resolve shim addr", "addr", shimAddr, "err", err)
		return
	}
	conn, err := net.DialUDP("udp4", nil, udpAddr)
	if err != nil {
		slog.Error("dial shim UDP", "err", err)
		return
	}
	defer conn.Close()

	ticker := time.NewTicker(interval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			payload, err := json.Marshal(store.Active())
			if err != nil {
				slog.Error("marshal hints", "err", err)
				continue
			}
			if _, err = conn.Write(payload); err != nil {
				slog.Warn("send hints to shim", "err", err)
			}
		}
	}
}

// ─── Main ─────────────────────────────────────────────────────────────────────

func main() {
	httpAddr  := flag.String("http-addr",   ":7070",             "HTTP API listen address")
	shimAddr  := flag.String("shim-addr",   "127.0.0.1:7071",   "Dynamo shim UDP address")
	intervalMs := flag.Int("interval-ms",  100,                 "Hint broadcast interval (ms)")
	flag.Parse()

	handler := slog.NewTextHandler(os.Stdout, &slog.HandlerOptions{
		Level: slog.LevelInfo,
	})
	slog.SetDefault(slog.New(handler))

	store := NewHintsStore()
	mux   := buildHTTPMux(store)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// HTTP server.
	srv := &http.Server{
		Addr:         *httpAddr,
		Handler:      mux,
		ReadTimeout:  5 * time.Second,
		WriteTimeout: 5 * time.Second,
	}
	go func() {
		slog.Info("factory-control HTTP API listening", "addr", *httpAddr)
		if err := srv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			slog.Error("HTTP server", "err", err)
		}
	}()

	// UDP broadcaster.
	go broadcastLoop(ctx, store,
		*shimAddr,
		time.Duration(*intervalMs)*time.Millisecond,
	)

	slog.Info("factory-control started",
		"http", *httpAddr,
		"shim", *shimAddr,
		"broadcast_ms", *intervalMs,
	)

	// Wait for SIGINT / SIGTERM.
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	slog.Info("shutting down")
	shutCtx, shutCancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer shutCancel()
	if err := srv.Shutdown(shutCtx); err != nil {
		slog.Error("HTTP shutdown", "err", err)
	}
}
