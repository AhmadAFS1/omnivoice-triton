# Experimental rotary fusion

`--fused-rope` is off by default. It combines Qwen3 rotary operations while
preserving separate FP16 product roundings and Torch's output strides. It applies
to the first 24 decoder layers; the last 4 retain their original Torch path.

```bash
omnivoice-api --runner hybrid --dtype fp16 --disable-unused-kv-cache --fused-rope
```

The initial supported configuration uses `triton` or `hybrid`, CUDA, FP16,
disabled LLM KV cache, default `(0,24)` Triton patches and SDPA. SageAttention,
full-layer patching and raw clone-graph prewarm cannot be combined with this
option. Startup rejects other runtime/source/model configurations: Torch
2.8.0+cu128, Triton 3.4.0, Transformers 5.5.4 and OmniVoice 0.1.4 are pinned.

Startup warms only tiny deterministic zero/one tensors, synchronizes the target
device, and verifies unchanged CPU and CUDA RNG states before readiness. Startup
duration is reported separately in `inference_optimizations.fused_rope_metrics`.
The outer model forward binds the actual generation thread, including graph
replays that bypass Python attention. One fused runner is allowed per process.

Future unsupported tensor geometry/layout/device cases use the original Torch
rotary function and increment CPU-only fallback counters. The current fused
geometry covers B1–16, S1–4096, head dimension 128, 16 query/8 key heads and
B×S≤65,535. This does not change serving admission or memory budgets.

During shutdown, an active or still-live foreign generation owner prevents
resource destruction. After its owner stops, device work is fenced, graphs are
released, the original forward/attention functions are restored, and model
weights are released. Failed CUDA fences retain resources for process restart.

This remains an experimental opt-in. Prototype measurements and product/API
qualification are separate evidence; enabling the option does not certify
latency, speech quality, or an untested GPU model. Request-cache identity is
unchanged because the setting is immutable for an app instance.

## Product qualification on 4 November 2026

The frozen product source archive was
`966a9afcf5913c8a4d9a3f29888e95ac2cfff18042356522e5416a805c2e2ccb`,
with checkpoint revision `c5fdb5ccb189668d56333f77ba2629f4cd7535f4` and the
pinned packages above. Actual GPU gates used one RTX 4070: 12,282 MiB physical
VRAM from `nvidia-smi`, 12,450,070,528 bytes Torch-visible VRAM, driver 580.95.05
and CUDA 12.8, on a Xeon E5-2686 v4 host.

- CPU contracts: 219 passed with CUDA devices hidden. An initial collection
  failed because repository addopts requested unavailable pytest-cov; the
  recorded rerun disabled those addopts without changing dependencies.
- Product kernel: raw 16-bit values and output strides matched for 108 tensor
  pairs, 203 observed sequence lengths and 4 graph replays with updated inputs.
- Product model: EN/ES/AR at B1/B8 matched every full-logit/input identity across
  16 steps, final target-token bytes, float audio bytes and PCM WAV bytes. The
  six complete pairs retained 54 WAVs with 27 unique hashes.
- Actual API constructor and startup: all 132 requests, covering all 104 language
  cases plus extra voice cases using 30 full references, returned HTTP 200 and
  valid decoded signals. Every complete WAV body matched the same-host unfused
  anchor byte for byte. Actual health showed ready/active, unchanged CPU/CUDA
  RNG states at startup, zero fallback and no CUDA fault.

The model/API profile used Hybrid, FP16, KV cache disabled, 16 steps, speed 1,
postprocessing off, serial codec decoding, 8 graph entries with a 16,384-row-token
policy, and response caching off. Startup's deterministic H16/H8 warmup took
3,750.96 ms in the API run; startup is separate from request latency.

The first model harness stopped on a reporting-only adapter AttributeError. Its
corrected bounded run passed five pairs before its time limit; an explicitly
authorized continuation tested only the remaining AR B8 pair. A separate CPU
verifier joined and rechecked all six pairs from retained outputs. Failed and
partial runs were preserved. Model parity overlapped approved CPU/SSH transfer
work, so its timings are excluded. The 132-request API run began after explicit
CPU/I/O release and was independently rehashed and decoded after local retention.

The subsequent clean RTX 4070 Ti eager-Triton comparison completed all 264 API
requests with 132/132 paired WAV bodies byte-identical and zero fallback. It used
B1, collect 0 ms, graphs off and the same pinned source, checkpoint, seed, order
and full references. Physical VRAM was 12,282 MiB, Torch-visible VRAM was
12,481,134,592 bytes, driver 610.57.04 and CUDA 12.8, with a Ryzen 7 5700X host.

| Complete-WAV latency | OFF | ON |
| --- | ---: | ---: |
| Median | 328.70 ms | 338.65 ms |
| Mean | 353.92 ms | 349.04 ms |
| p95 | 413.80 ms | 390.44 ms |

This is a mixed result: median worsened 3.0%, while mean improved 1.4% and p95
improved 5.6%. Excluding the first request gives a 0.9% mean reduction. The OFF
worker ran first, with one run per arm; order, cache state and hardware drift
limit causal interpretation. The ON startup warmup took 512.44 ms before
readiness and preserved both RNG states. Keep the option experimental and off
by default; a general eager speedup is unproven.

All 264 actual WAV hashes, ASR language/script IDs and exact expected texts
matched the prior 9,532-input canonical set. No new ASR inputs were required;
the exact bodies inherit existing two-judge evidence. These gates preserve
existing speech errors and do not certify every language's pronunciation.
Earlier prototype speed measurements are separate evidence. Inspect the
[curated proofs and latency](../benchmark/multilingual/results/fused-rope-2026-11-04/README.md)
and [all 264 paired samples](../benchmark/multilingual/samples/rtx4070ti-fused-rope-2026-11-04/README.md).
