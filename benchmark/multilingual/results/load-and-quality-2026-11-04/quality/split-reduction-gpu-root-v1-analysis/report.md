# Split CFG and FP16 reduction permission: bounded diagnostic

The experiment completed on the RTX 4070 and retained 108 signal-valid full WAVs,
representing 28 unique bodies. **It does not provide a lossless Split CFG fix.**
All split-versus-concatenated comparisons change final tokens and float/PCM audio.
Disabling reduced-precision GEMM reductions also changes the concatenated B1
baseline, although the concatenated B8 True/False outputs match exactly.

| One warmed full-model observation | B1 | B8 |
|---|---:|---:|
| Concatenated CFG, reduction allowed (current baseline) | 456.9 ms | 3,619.2 ms |
| Concatenated CFG, reduction disabled | 457.3 ms | 3,516.0 ms |
| Split CFG, reduction disabled | 356.1 ms | 2,023.7 ms |
| Split CFG, reduction allowed | 370.8 ms | 2,014.0 ms |

These are single warmed, synchronized generation-plus-codec observations, not
p95, HTTP latency, or concurrent server throughput. Separate traced calls retain
actual conditional/unconditional logits, guided scores/log-probabilities,
argmax/confidence, final target tokens, float arrays and PCM WAVs. All eight
arm/batch cases reproduce their traced tokens/audio in the separately timed call.
The independent CPU audit verifies all 108 body hashes and decoded signal, then
recomputes all twelve arm-pair token/float/PCM comparisons from raw stored arrays.

Scope is one published English sentence with a native Aoede reference at B1/B8.
Reference preparation is held under the original True setting and shared across
arms. Sixteen steps, guidance2, class temperature0, position temperature5, speed1,
postprocessing off, serial codec1, KV off, hybrid graph8/M16384, Split/RoPE/bucket
defaults and the pinned ce48 source/checkpoint are otherwise fixed. Graphs are
fenced and cleared before every reduction-flag transition. The prototype runs in
its own process and restores flags on success; no production setting changes.

Hardware: NVIDIA RTX 4070, physical 12,282 MiB, Torch-visible 12,450,070,528 bytes,
driver580.95.05/CUDA12.8, 200 W, Xeon E5-2686 v4 host. Torch2.8.0+cu128,
Triton3.4.0, Transformers5.5.4 and OmniVoice0.1.4 are pinned. The finite root lease
started at 14:15:23 UTC on November4; both jobs completed within its 300-second
bound. The supervisor and independent process query prove GPU idle afterward.

Blind ASR is queued in the next strict union: 108 logical probe outputs reduce
to 28 unique waveform/script inputs, of which 19 are new relative to the prior
9,532-input ASR corpus. Word quality is not inferred from bit differences alone,
nor from a single English sentence. Existing broader Split word regressions
remain relevant. **No default promotion** follows from these timings.
