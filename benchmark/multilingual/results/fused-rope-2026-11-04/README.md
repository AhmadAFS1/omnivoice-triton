# Experimental fused RoPE qualification, 4 November 2026

The default-off option preserved every tested output. Its eager B1 latency
result is mixed: median latency increased, while mean and p95 decreased. Keep
the option experimental; these measurements support no general eager speedup.

The product source archive is
`966a9afcf5913c8a4d9a3f29888e95ac2cfff18042356522e5416a805c2e2ccb`,
included under [source](source/fused-rope-product-cpu-v3.tar.gz). Product commit
`deaa1d40d5e716e3059ce7ff04b3d66ff3fb8712` contains the same code and tests.
The checkpoint remains `c5fdb5ccb189668d56333f77ba2629f4cd7535f4`.

## Actual eager API comparison

One RTX 4070 Ti, UUID `GPU-5aada52f-9af2-4506-47f7-75e99aaf51ee`, had 12,282 MiB
physical VRAM and 12,481,134,592 bytes Torch-visible VRAM, driver 610.57.04 and
CUDA 12.8. The host was an AMD Ryzen 7 5700X. Runtime versions were Torch
2.8.0+cu128, Triton 3.4.0, Transformers 5.5.4 and OmniVoice 0.1.4.

Both fresh API workers used eager Triton, B1, collect 0 ms, graphs off, KV cache
off, 16 steps, speed 1, postprocessing off, serial codec decoding, response cache
off, the same 30 full references, case order and startup seed 20261103. Only the
immutable fused option differed. Each arm completed 132 sequential full-WAV HTTP
responses across all 104 language routes and eight voices. No other neural work
or transfer ran on this host during the comparison.

| Complete-WAV latency | OFF | ON | ON relative to OFF |
| --- | ---: | ---: | ---: |
| Median | 328.70 ms | 338.65 ms | +3.0%, slower |
| Mean | 353.92 ms | 349.04 ms | −1.4% |
| p95 | 413.80 ms | 390.44 ms | −5.6% |
| First request | 898.96 ms | 680.33 ms | Separate initialization effects |
| Mean excluding the first request | 349.76 ms | 346.51 ms | −0.9%, diagnostic |

The OFF worker ran first, followed by ON, with one run per case and arm. This
order, cache state and hardware drift limit causal interpretation. The ON
startup's deterministic H16/H8 warmup took 512.44 ms before readiness; it
preserved CPU and CUDA RNG states. Startup is excluded from request latency.
The whole two-arm gate took 115.50 seconds including worker startup and checks.

All 264 HTTP responses and decoded signals passed. All 132 paired WAV bodies
were byte-identical; actual health showed readiness, zero fallback and no CUDA
fault. These are sequential completion measurements. Burst capacity is measured
separately.

Use the [latency analysis](fused-rope-modern264-latency-analysis-v1.json),
[actual completion](modern/MODERN_FUSED_ROPE_EAGER264_COMPLETE.json),
[OFF metadata](modern/off/metadata.json), [ON metadata](modern/on/metadata.json)
and per-arm compressed requests, server logs and telemetry for inspection.
All [264 generated samples and paired manifests](../../samples/rtx4070ti-fused-rope-2026-11-04/README.md)
are included.

## Exactness and quality evidence

The [strict reuse audit](fused-rope-modern264-strict-ASR-reuse-v1.json) rehashed
each actual WAV and joined its exact SHA, ASR language/script ID and expected
Unicode text to the prior 9,532-input canonical set. All 132 OFF and 132 ON
bodies matched: zero new ASR inputs were required. ON inherits the existing
two-judge evidence only through exact waveform identity. No transcript was
relabeled, and output preservation also preserves existing speech errors.

Earlier product gates on one RTX 4070, UUID
`GPU-6eb0a2a5-d251-a91e-8e98-82bc0e371e33`, used 12,282 MiB physical and
12,450,070,528 bytes Torch-visible VRAM, driver 580.95.05, CUDA 12.8 and a Xeon
E5-2686 v4 host:

- [219 CPU contracts](primary/CPU219-tests.log) passed with CUDA devices hidden.
  The initial invocation could not collect tests because pytest-cov was absent;
  disabling those repository addopts was recorded, without dependency changes.
- The [actual product kernel](primary/raw-kernel/outcome.json) matched raw
  16-bit values and strides for 108 tensor pairs, 203 observed sequence lengths
  and four updated-input graph replays.
- The [joined model proof](primary/FUSED_ROPE_PRODUCT_MODEL_JOINED_PARITY.json)
  covers EN/ES/AR at B1/B8: all 16 full-logit/input identities, target-token
  bytes, float audio bytes and PCM WAV bytes matched across six complete pairs.
- The [actual Hybrid API132 gate](primary/FUSED_ROPE_PRODUCT_API132_COMPLETE.json)
  returned 132 valid HTTP 200 WAVs, each byte-identical to its same-host unfused
  anchor. The profile used graph 8, a 16,384-row-token memory policy, KV cache
  off, 16 steps, serial codec and no response cache. Its startup warmup took
  3,750.96 ms. Request timing from that run is not a causal speedup measurement.

The first model harness stopped on an adapter reporting AttributeError after
saving its first pair. The corrected bounded run passed five complete pairs
before its 280-second internal limit. An explicitly authorized, separate
continuation tested only the remaining AR B8 pair. All six complete pairs were
then verified together from retained files. Failed and partial outcomes are
preserved under `primary/model/`; their large NPZ arrays and model-probe WAVs
remain fully local, with a published
[binary hash manifest](primary/model/retained-binary-hash-manifest.json).
The model parity runs overlapped approved CPU/SSH transfer work, so their timing
claims are excluded.

Source-text attribution and licenses are recorded per sample. Model weights,
conditioning WAVs and private keys are excluded from this publication.
