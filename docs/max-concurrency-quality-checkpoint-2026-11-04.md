# Concurrency and quality checkpoint — November 4, 2026

This checkpoint describes measured results at 11:30 UTC. The optimization task
continues until 16:00 UTC (10am Chicago). The goal of 1,000 fresh complete-audio
requests in 1–2 seconds on one RTX 4070-class GPU has **not been achieved**.
Full audio remains the API contract. No production deployment or PR13 change is
part of these experiments.

## Fresh generation and cached delivery

| Workload | RTX 4070 | RTX 4070 Ti |
|---|---:|---:|
| 10 distinct requests, full cohort | 8.317s | 4.386s |
| 100 distinct requests, full cohort | 80.850s | 46.326s |
| 1,000 novel requests, full cohort | 760.877s | 493.556s |
| 1,000 novel requests, p95 request latency | 722.015s | 472.370s |
| 500 distinct requests spread over 1s | 386.759s | not this checkpoint |
| 500 distinct requests spread over 2s | 382.478s | not this checkpoint |

Each fresh cohort above returned every requested full, signal-valid WAV. Word
quality is a separate measurement. These columns compare two complete serving
configurations, not an isolated GPU or code A/B: 4070 used source47bd743 with
graph8/M16384/KVoff; Ti used baselinec79ce35 with graph2/KVon. Both used the pinned
OmniVoice0.1.4 checkpoint, hybrid runner, 16steps, Split off and cache off. Their
host CPUs, drivers, power limits and batching/capture behavior differ.

The 4070 is physical12,282MiB/visible12,450,070,528bytes, driver580.95.05,
CUDA12.8,200W, XeonE5-2686v4 with18 effective allocated cores. The Ti is
physical12,282MiB/visible12,481,134,592bytes, driver610.57.04, CUDA12.8,285W,
Ryzen5700X/16threads. Both have Torch2.8.0+cu128, Triton3.4.0 and
Transformers5.5.4. Physical and visible VRAM are separate recorded measurements.

On the 4070, a 500-call simulation assuming one response per20seconds offered
1,500 requests at25RPS. It produced73 valid200s,1,426 HTTP429s and1 connection
ReadError. No CUDA fault was recorded. The one read error accompanied a slow
health probe; available telemetry cannot establish the exact connection cause.
This remains a4.87% workload audio-success result, not a successful500-call
configuration. Ten warmup200s are excluded from that workload denominator.

A separate Ti test delivered1,000 already-cached bodies in0.554s with a
preconnected raw-asyncio client (p95516ms); a dedicated-HTTPX client took1.023s
(p95810ms). These were cache hits. They establish transport/cache delivery for
that fixture, not fresh synthesis capacity or a p95<500ms result.

All2,533 saved primary-core WAVs, including warmups, were retained locally and
independently rehashed/decoded. They contain2,499 unique bodies. The audit
preserves all1,426 rejections and the transport failure separately.

## Quality gates

The complete voice/config grid previously generated2,704 cases covering104
routes, eight voices and three real Lingua presets, plus auto/design cases.
2,700 passed basic signal checks and four were empty WAVs. The sourcece48b25
guard now rejects empty, nonfinite or all-zero generated audio before clipping,
encoding or caching. It does not certify the words or rescue nearly silent
audio. The broader two-judge word sweep is still running at this checkpoint.

Reducing16steps to8 or4 lowered sequential p95 from572ms to386ms or290ms over
104 matched Aoede/video routes. Both Meta judges' CER worsened on22 routes at8
steps and50 at4. The16step default remains. These are single sequential
utterances, not concurrent capacity.

The reference-prefix experiment keeps16steps and sourcece48b25, with
graph8/M16384/KVoff/Splitoff/codec1/cache0. It has132 cases per arm:104 Aoede
routes plus all eight voices for Arabic, English, Spanish and Hindi. English
conditioning changes in110 cases;22 native-reference controls per arm remain
byte-identical.

| English-conditioning cases, n110 | Full reference | Ten-word prefix | Seventeen-word prefix |
|---|---:|---:|---:|
| Sequential p95 full-WAV latency |573.8ms|382.5ms|404.9ms|
| Both ASRs exactly match script |21|43|25|
| New baseline-exact losses |—|3|7|
| Both judges' CER worsen |—|9|6|

Three of the seven seventeen-word exact losses are spacing-only, with zero CER
in both judges. Other flags include judge disagreement and substitutions.
One ten-word LLM transcript repeats a string while CTC nearly matches; this
does not prove repetition in the actual audio. Every outlier remains retained.

The separate CPU SpeechBrain ECAPA screen finds speaker-top1 losses on two
ten-word cases and one seventeen-word case. Those cases differ from the word
flags. Cosines are uncalibrated multilingual/synthetic-voice diagnostics, not
MOS, accent or native-fluency certification. Neither reference prefix is
promoted. The follow-up holds original target frame counts fixed, with a
132/132 original-WAV byte-parity gate before its cropped arms; it is prepared,
not yet run at this checkpoint.

## Scaling work

Implemented and tested changes include bounded admission, exact-response
coalescing/cache, asynchronous reference registration, unused-KV disabling,
pre-capture graph memory admission and generated-audio validation. They address
duplicate work, host blocking, memory faults and invalid responses. Measured
fresh synthesis still has insufficient throughput for the requested targets.

A synchronized warm graph profile on the 4070 attributes98.4% of summed GPU
kernel duration to the repeated LLM graphs. Homogeneous B1/B8 complete-model
samples yield2.27/2.24requests/s; this sample provides no batching speedup.
Earlier unsynchronized decode headers absorbed outstanding LLM work and cannot
be treated as isolated codec cost. A default-off RoPE fusion prototype is now
entering strict tensor-bit/layout gates; shared graph-pool work is CPU-only.

The deployable scaling design remains multiple independent GPU workers behind
deadline-aware admission and routing by prepared-reference locality. Size
replicas from measured **fresh**, quality-qualified throughput and queue delay;
count active calls separately from speaking turns. For example,1,000 calls at
one turn/20s offer50RPS. At2RPS per worker,25replicas would be only an ideal
saturation calculation, with no queue/SLO headroom. This is an illustration,
not a validated fleet capacity. A1,000-request/2second burst needs500RPS; the
current two-host evidence does not support that on one GPU.

## Retained evidence

- [All396 reference-prefix WAVs and paired manifest](../benchmark/multilingual/samples/rtx4070ti-reference-prefix-2026-11-04/README.md).
- [Joint word and speaker review](../benchmark/multilingual/results/quality-2026-11-04/speaker-word-prefix396-joint-v1/joint-report.md), including every flagged case and judge disagreement.
- [Original latency and integrity audit](../benchmark/multilingual/results/quality-2026-11-04/reference-prefix-analysis-v1/report.md).
- [Blind ASR432 results and protocol metadata](../benchmark/multilingual/results/quality-2026-11-04/blind-ASR432/).
- [Primary2533-WAV retention audit](../benchmark/multilingual/results/quality-2026-11-04/candidate47-core-retention-audit-v1/summary.json).
- [Existing208 before/after WAVs across104 routes](../benchmark/multilingual/samples/rtx4070-2026-11-03/paired-manifest.json).

All text-source licenses and language/ASR-proxy caveats remain in the manifests.
Conditioning reference WAVs, ASR/model weights, credentials and rental transfer
keys are not bundled.
