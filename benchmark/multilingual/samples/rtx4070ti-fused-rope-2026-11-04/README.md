# Fused RoPE OFF/ON samples, RTX 4070 Ti, 4 November 2026

All 264 complete generated WAVs are included: 132 OFF and 132 ON across all 104
language routes and eight voices. The same case in each arm has an identical
WAV SHA. This preserves the existing audio, including any speech errors.

Use [manifest.json](manifest.json) for target text, exact ASR language/script,
conditioning-reference identity, generation options, response latency, audio
properties and source/license attribution. Use
[paired-manifest.json](paired-manifest.json) to compare each of the 132 cases.
The [qualification and measured results](../../results/fused-rope-2026-11-04/README.md)
include runtime, hardware, exactness gates, latency caveats and strict reuse of
the prior two-judge ASR evidence. The result is mixed: eager B1 median worsened,
while mean and p95 decreased. The option remains experimental and off by default.

The OFF worker ran first; each case has one sample per arm. Both workers used
eager Triton, B1, collect 0 ms, graphs off, KV cache off, 16 steps, speed 1,
postprocessing off, serial codec, response cache off, startup seed 20261103 and
the same 30 full references. Measurements cover full audio returned at once.

These WAVs are generated from the attributed texts in the manifest. Keep the
individual source-text licenses and notices with redistributed material; source
licenses differ by corpus. Repository code uses the
[Apache-2.0 license](../../../../LICENSE). The source metadata does not assert a
new blanket license for all generated audio. Conditioning-reference WAVs and
model weights are excluded.
