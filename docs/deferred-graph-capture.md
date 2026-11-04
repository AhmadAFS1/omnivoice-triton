# Experimental graph-capture deferral

`--cuda-graph-capture-after-calls N` delays capture for each new tensor shape.
The default is0, preserving capture on first use. Positive values require a
`faster` or `hybrid` runner and a positive `--max-cuda-graph-row-tokens` budget.

```sh
omnivoice-api --runner hybrid --disable-unused-kv-cache \
  --max-cuda-graphs 8 --max-cuda-graph-row-tokens 16384 \
  --cuda-graph-capture-after-calls 16
```

A call is one model forward, usually one denoising step. It is **not** one HTTP
request. With16steps, N16 normally lets a previously unseen shape complete its
first utterance eagerly. A subsequent use may capture and then replay it. This
does not change the generation parameters, tensor shape, reference or codec.

The option addresses a measured cold-shape cost: on the RTX4070Ti132-case
sequential fixture, the existing Triton eager path produced132/132 byte-identical
WAVs to hybrid while reducing p95 from about577ms to420ms. That result does not
establish the deferral option's performance, concurrent capacity, or native
speech quality.

A subsequent matched RTX4070 screen exercised N0/N16/N64 and eager profiles on
132 full-reference cases each. Every arm returned132 valid HTTP200 WAVs, and
N16/N64/eager each matched every N0 body SHA. Nearest-rank full-body p95 was
1664/1504/3201/3150ms respectively. Capture counts fell118→40→1 for N0/N16/N64;
N16/N64 deferred1376/2080 of2112 forwards. No capture failures, pressure cleanups
or admission rejections occurred. This ordered single-run screen also recorded
GPU-clock and CPU-frequency drift, so it does not isolate a universal best N.
N64 is not recommended from these results. N16 remains an optional setting,
not a new default. These absolute timings came from the older-CPU4070 host and
must not be substituted for the separate Ti timings above.

Only successful eager returns advance a shape's counter. Memory admission and
the GPU-owner check still run before the eager call. An admission rejection or
model exception does not warm the shape. Captured shapes retain the existing
hit/capture/eviction policy; eviction forgets that shape's observations so its
next appearance defers again. Counters hold at most4096 shape keys, contain no
tensors, and clear with the graph policy. Existing training/gradient/unsupported
calls retain their prior behavior.

`/health` reports the requested `cuda_graph_capture_after_calls`. The inference
optimization snapshot adds the threshold, successful deferred calls and bounded
observation count. Existing capture, fallback, admission and fault counters
remain separate. The feature is an operator option and stays off by default.

Validation:178 tests passed with pinned Torch2.8.0+cu128/Triton3.4.0/
Transformers5.5.4 dependencies and CUDA hidden. The API tests use CPU fake models
and stub hardware discovery; no neural or capacity result is inferred from that
suite. Tests cover threshold boundaries, unchanged default behavior, current
eager memory admission, failed model calls, bounded observations, eviction,
clear, constructor/API validation and option forwarding. Before recommending a
threshold, qualify matched GPU outputs and both cold and repeated-shape loads.
