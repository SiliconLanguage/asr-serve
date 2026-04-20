// SPDX-License-Identifier: BSD-2-Clause-Patent
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

// ─── HintsStore Tests ─────────────────────────────────────────────────────────

func TestHintsStoreDefaultValues(t *testing.T) {
	store := NewHintsStore()
	active := store.Active()

	if active.ModelID == "" {
		t.Error("expected non-empty default model_id")
	}
	if active.Priority < 0 || active.Priority > 9 {
		t.Errorf("priority %d out of range [0,9]", active.Priority)
	}
	if active.KVTTLMs <= 0 {
		t.Errorf("kv_ttl_ms %d must be positive", active.KVTTLMs)
	}
}

func TestHintsStoreUpdate(t *testing.T) {
	store := NewHintsStore()
	store.Update(HintProfile{
		ModelID:  "canary-1b",
		Priority: 8,
		KVTTLMs:  250,
	})

	active := store.Active()
	if active.ModelID != "canary-1b" {
		t.Errorf("model_id: got %q, want %q", active.ModelID, "canary-1b")
	}
	if active.Priority != 8 {
		t.Errorf("priority: got %d, want 8", active.Priority)
	}
	if active.KVTTLMs != 250 {
		t.Errorf("kv_ttl_ms: got %d, want 250", active.KVTTLMs)
	}
	if active.UpdatedAt == 0 {
		t.Error("updated_at should be set on update")
	}
}

func TestHintsStoreHistory(t *testing.T) {
	store := NewHintsStore()
	original := store.Active()

	store.Update(HintProfile{ModelID: "model-a", Priority: 3, KVTTLMs: 300})
	store.Update(HintProfile{ModelID: "model-b", Priority: 7, KVTTLMs: 400})

	if len(store.history) < 2 {
		t.Errorf("expected at least 2 history entries, got %d", len(store.history))
	}
	if store.history[0].ModelID != original.ModelID {
		t.Errorf("history[0] should be original model, got %q", store.history[0].ModelID)
	}
}

func TestHintsStoreConcurrency(t *testing.T) {
	store := NewHintsStore()
	done  := make(chan struct{})

	// Concurrent writers.
	for i := range 10 {
		go func(n int) {
			store.Update(HintProfile{
				ModelID:  "model",
				Priority: n % 10,
				KVTTLMs:  500,
			})
		}(i)
	}
	// Concurrent readers.
	go func() {
		for range 100 {
			_ = store.Active()
		}
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Error("concurrent access timed out")
	}
}

// ─── HTTP API Tests ───────────────────────────────────────────────────────────

func TestHTTPGetHints(t *testing.T) {
	store := NewHintsStore()
	store.Update(HintProfile{ModelID: "whisper", Priority: 6, KVTTLMs: 400})

	mux := buildHTTPMux(store)
	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/hints", nil)
	mux.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Errorf("status: got %d, want 200", rec.Code)
	}

	var profile HintProfile
	if err := json.NewDecoder(rec.Body).Decode(&profile); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if profile.ModelID != "whisper" {
		t.Errorf("model_id: got %q, want %q", profile.ModelID, "whisper")
	}
	if profile.Priority != 6 {
		t.Errorf("priority: got %d, want 6", profile.Priority)
	}
}

func TestHTTPPutHints(t *testing.T) {
	store := NewHintsStore()
	mux   := buildHTTPMux(store)

	body, _ := json.Marshal(HintProfile{
		ModelID:  "canary-1b",
		Priority: 9,
		KVTTLMs:  150,
	})

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPut, "/hints", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	mux.ServeHTTP(rec, req)

	if rec.Code != http.StatusNoContent {
		t.Errorf("status: got %d, want 204", rec.Code)
	}

	active := store.Active()
	if active.ModelID != "canary-1b" {
		t.Errorf("model_id: got %q, want %q", active.ModelID, "canary-1b")
	}
}

func TestHTTPPutHintsInvalidJSON(t *testing.T) {
	store := NewHintsStore()
	mux   := buildHTTPMux(store)

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodPut, "/hints",
		bytes.NewReader([]byte("not-json")))
	mux.ServeHTTP(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Errorf("status: got %d, want 400", rec.Code)
	}
}

func TestHTTPMethodNotAllowed(t *testing.T) {
	store := NewHintsStore()
	mux   := buildHTTPMux(store)

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodDelete, "/hints", nil)
	mux.ServeHTTP(rec, req)

	if rec.Code != http.StatusMethodNotAllowed {
		t.Errorf("status: got %d, want 405", rec.Code)
	}
}

func TestHTTPHealth(t *testing.T) {
	store := NewHintsStore()
	mux   := buildHTTPMux(store)

	rec := httptest.NewRecorder()
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	mux.ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Errorf("status: got %d, want 200", rec.Code)
	}
}

// ─── Broadcast Loop Smoke Test ────────────────────────────────────────────────

func TestBroadcastLoopContextCancel(t *testing.T) {
	store := NewHintsStore()

	// Listen on a local UDP port.
	conn, err := net.ListenPacket("udp4", "127.0.0.1:0")
	if err != nil {
		t.Skipf("could not bind UDP socket: %v", err)
	}
	defer conn.Close()
	addr := conn.LocalAddr().String()

	ctx, cancel := context.WithTimeout(context.Background(), 300*time.Millisecond)
	defer cancel()

	done := make(chan struct{})
	go func() {
		broadcastLoop(ctx, store, addr, 50*time.Millisecond)
		close(done)
	}()

	select {
	case <-done:
		// broadcast loop exited cleanly after context was cancelled.
	case <-time.After(2 * time.Second):
		t.Error("broadcastLoop did not exit after context cancellation")
	}
}
