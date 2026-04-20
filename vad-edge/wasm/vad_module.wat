;; SPDX-License-Identifier: BSD-2-Clause-Patent
;;
;; WebAssembly VAD (Voice Activity Detection) module – text format (.wat).
;;
;; This module implements a lightweight energy + zero-crossing-rate (ZCR)
;; dual-criterion VAD suitable for 16 kHz / 16-bit mono PCM audio.
;;
;; Exported interface
;; ──────────────────
;;   vad_classify(ptr: i32, n_samples: i32) -> i32
;;     ptr       – byte offset into linear memory where i16 PCM samples start
;;     n_samples – number of 16-bit samples in the chunk (320 for 10 ms)
;;     returns   – 1 (speech) or 0 (silence)
;;
;; The caller (dpdk_pipeline.c) writes the PCM chunk at `ptr` before calling.
;;
;; Tunable constants (exported as globals so the host can override them)
;;   energy_threshold – absolute energy per sample above which speech is assumed
;;   zcr_threshold    – zero-crossing rate (0–100) above which speech is assumed
;;
;; NOTE: Replace this module with a compiled TFLite / ONNX Runtime Wasm build
;; for production accuracy.  This stub is intentionally simple so it compiles
;; with any WASM toolchain and runs without external imports.

(module
  ;; ── Memory ──────────────────────────────────────────────────────────────────
  ;; Single 64 KiB page; the host may grow it.
  ;; Layout:  0x0000–0xFFFF  reserved for stack / globals
  ;;          0x10000+        audio sample buffer (written by host before call)
  (memory (export "memory") 2)

  ;; ── Tunable constants ───────────────────────────────────────────────────────
  ;; energy_threshold: average |sample| value above which frame is "speech".
  ;; Default 512 ≈ -50 dBFS for 16-bit audio.
  (global $energy_threshold (export "energy_threshold") (mut i32) (i32.const 512))
  ;; zcr_threshold: zero-crossings per 100 samples above which frame is "speech".
  (global $zcr_threshold    (export "zcr_threshold")    (mut i32) (i32.const 20))

  ;; ── vad_classify ────────────────────────────────────────────────────────────
  ;; (param $ptr i32) (param $n_samples i32) (result i32)
  (func $vad_classify (export "vad_classify")
        (param $ptr      i32)
        (param $n_samples i32)
        (result i32)

    (local $i       i32)   ;; loop index (sample index, not byte index)
    (local $energy  i64)   ;; accumulated |sample| energy
    (local $zcr     i32)   ;; zero-crossing count
    (local $sample  i32)   ;; current sample (sign-extended i16 → i32)
    (local $prev    i32)   ;; previous sample
    (local $avg_e   i32)   ;; average energy per sample
    (local $zcr100  i32)   ;; ZCR per 100 samples (scaled)

    ;; Guard: empty chunk → silence
    (if (i32.le_s (local.get $n_samples) (i32.const 0))
      (then (return (i32.const 0)))
    )

    ;; ── Energy + ZCR accumulation loop ──────────────────────────────────────
    (local.set $i      (i32.const 0))
    (local.set $energy (i64.const 0))
    (local.set $zcr    (i32.const 0))
    (local.set $prev   (i32.const 0))

    (block $break
      (loop $loop
        (br_if $break (i32.ge_s (local.get $i) (local.get $n_samples)))

        ;; Load i16 sample at ptr + i*2, sign-extend to i32.
        (local.set $sample
          (i32.load16_s
            (i32.add
              (local.get $ptr)
              (i32.mul (local.get $i) (i32.const 2))
            )
          )
        )

        ;; Accumulate |sample| into energy (i64 to avoid overflow).
        (local.set $energy
          (i64.add
            (local.get $energy)
            (i64.extend_i32_s
              (select
                (i32.sub (i32.const 0) (local.get $sample))
                (local.get $sample)
                (i32.lt_s (local.get $sample) (i32.const 0))
              )
            )
          )
        )

        ;; Zero-crossing: sign($sample) ≠ sign($prev)
        ;; Approximation: (sample ^ prev) < 0 for opposite signs.
        (if (i32.lt_s
              (i32.xor (local.get $sample) (local.get $prev))
              (i32.const 0))
          (then
            (local.set $zcr (i32.add (local.get $zcr) (i32.const 1)))
          )
        )

        (local.set $prev (local.get $sample))
        (local.set $i    (i32.add (local.get $i) (i32.const 1)))
        (br $loop)
      )
    )

    ;; ── Normalise ────────────────────────────────────────────────────────────
    ;; Average energy per sample (truncated to i32; max i16 energy is 32768).
    (local.set $avg_e
      (i32.wrap_i64
        (i64.div_u (local.get $energy)
                   (i64.extend_i32_u (local.get $n_samples)))
      )
    )

    ;; ZCR per 100 samples.
    (local.set $zcr100
      (i32.div_u
        (i32.mul (local.get $zcr) (i32.const 100))
        (local.get $n_samples)
      )
    )

    ;; ── Decision ─────────────────────────────────────────────────────────────
    ;; Speech if EITHER energy OR ZCR exceeds its threshold.
    (i32.or
      (i32.gt_s (local.get $avg_e)  (global.get $energy_threshold))
      (i32.gt_s (local.get $zcr100) (global.get $zcr_threshold))
    )
  )

  ;; ── set_energy_threshold ────────────────────────────────────────────────────
  (func (export "set_energy_threshold") (param $t i32)
    (global.set $energy_threshold (local.get $t))
  )

  ;; ── set_zcr_threshold ───────────────────────────────────────────────────────
  (func (export "set_zcr_threshold") (param $t i32)
    (global.set $zcr_threshold (local.get $t))
  )
)
