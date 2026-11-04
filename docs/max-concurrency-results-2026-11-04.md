# Fresh synthesis, quality and scaling — November 4, 2026

The requested1,000 fresh complete-audio requests in1–2seconds on one4070-class
GPU is still unmet. The best completed1,000-input test at this checkpoint takes
374.738seconds on an RTX4070Ti, with p95request latency356.228seconds. A separate
sequential132-case fixture reaches p95full-audio420.41ms without changing any
matched waveform. These measure different load conditions.

This report is being updated during the authorized timebox ending16:00UTC,
November4 (10am Chicago). Its older11:30UTC checkpoint remains historical.
Full WAVs remain the response contract throughout; no client streaming or
MuseTalk changes were introduced.

## What improved

| RTX4070Ti, fresh1,000-input burst | Cohort completion | Request p95 | Requests/s |
|---|---:|---:|---:|
| Earlier hybrid, graphs2/KV on, c79 |493.556s|472.370s|2.026|
| Triton eager, B8/KV off, ce48 |429.822s|410.688s|2.327|
| Triton eager, B1/KV off, ce48 |374.738s|356.228s|2.669|

Every arm returned all1,000 full signal-valid WAVs, and every offered error would
remain a failure. B1 drains12.8%faster than B8 on the same source/settings. Its
24.1%shorter drain than the earlier hybrid test is a complete-configuration
comparison: source, runner, graph/KV profile and request execution order differ.
It is not an isolated code-only speedup. Both eager arms use16steps, original
full references, speed1, postprocessing off, serial codec1 and response cacheoff.
Batch grouping can affect sampled words, so signal validity alone is not their
word-quality clearance. The new outputs are undergoing both frozen ASR judges.

The previous HTTPX preconnect phase did not keep all sockets alive until the
burst:986 new TCP connections were observed per eager arm. The run and that
limitation are preserved. The separate5,000-offer assay uses5,000 individually
verified preconnected raw sockets and a600second benchmark-server keepalive;
it also changes collection delay from5ms to0ms. Its workload cycles the original
1,000 inputs five times with cacheoff; it is not5,000 novel inputs.

| Sequential132 full-reference cases on the Ti | Median | p95 | ≤500ms |
|---|---:|---:|---:|
| Hybrid, graphs8 |482.17ms|577.29ms|83/132|
| Base eager |372.41ms|465.69ms|130/132|
| Faster, graphs8, no Triton kernels |520.66ms|621.95ms|24/132|
| Triton eager |333.93ms|420.41ms|131/132|

Triton eager and hybrid match132/132WAVs exactly. Base and Faster also match
132/132, while Base and hybrid differ on every waveform. Existing Triton math
therefore needs its own quality evaluation. Both Meta ASRs exactly match37/132
Triton/hybrid scripts versus35/132Base/Faster; Base loses five baseline-exact
cases and gains three. Mixed language-specific results do not establish one
path as universally better. The hybrid/Base/Faster p95values in this table use
linear interpolation; Triton eager uses nearest rank, as labeled in raw reports.

## Quality and retained samples

The full voice/config sweep covers104 Lingua routes, eight stock voices and
three real presets, plus auto/design cases. Of2,704offers,2,700have valid signal
and four are empty. Both Meta judges have results for every readable clip;
463/2,704match the normalized script exactly under both, and535have zero
character error under both. Judges disagree on2,103cases. This is a diagnostic
screen, not evidence that every disagreement is bad generated speech. Many
routes lack target-language reference calibration, and proxy mappings remain
explicit. The generated-audio guard rejects empty/nonfinite/allzero responses;
it does not repair words or certify near-silent audio.

The16step default remains. Eight/four steps improved sequential speed but
worsened both judges' CER on22/50 of104matched routes. Reference shortening
also remains experimental: when output duration is fixed, English-reference
p95falls570.6→385.0/418.2ms, but three/four baseline-exact word matches are lost
and three/one speaker-top1 ranks are lost at ten/seventeen words. All before and
after WAVs and every flag are retained. Offline ECAPA and selected Whisper
diagnostics supplement Meta LLM/CTC; none substitutes for native listening/MOS.

## Scaling plan and its present limits

The server accepts work through bounded admission and a single GPU owner. Exact
response coalescing/cache can remove duplicate work; prepared-reference caching
avoids re-encoding. Neither makes novel speech free. Memory-aware graph admission
and explicit KV disabling address faults and unnecessary residency. Separate
GPU workers and a central router are needed for aggregate capacity at the
measured fresh rate; a production fleet/router has not been deployed here.

| Active calls, assumed one TTS turn/20s | Offered fresh RPS | Ideal workers at2.669RPS |
|---|---:|---:|
|10|0.5|1|
|100|5|2|
|1,000|50|19|
|5,000|250|94|

These rounded-up numbers are saturation floors, not SLO-qualified capacities.
They leave no queue, variability, failover or networking headroom. With a
sequential service p95already around420ms, the500ms end-to-end TTS budget leaves
little queue delay. Actual fleet sizing must use measured arrival distributions,
per-language/full-reference service costs and all-offer success/latency.
One thousand simultaneous fresh responses within2seconds require500RPS,
about187times the current single-GPU observed rate before any latency headroom.

The next serving design is deadline-aware routing across independent warmed
workers, keyed by model/voice-reference locality, with bounded per-worker queues,
explicit overload responses, cancellation of queued expired work and autoscaling
from fresh service demand. Validate10/100/1,000/5,000active-call schedules separately
from synchronized bursts. Report full-body p50/p95/p99, within500ms/2s over all
offers, quality-qualified successes, rejections, OOMs, cost per generated minute
and per-language regressions. This is a plan; no unmeasured fleet guarantee is
claimed. Cached transport results remain separate from fresh generation.

## Hardware and reproducibility

The Ti host is Ryzen5700X/16threads,32GBRAM,285W, driver610.57.04; physical
VRAM12,282MiB and Torch-visible12,481,134,592bytes. The4070host is XeonE5-2686v4
with18effective cores,128GBRAM,200W, driver580.95.05; physical12,282MiB and
visible12,450,070,528bytes. Both use CUDA12.8, Python3.12.3, Torch/Torchaudio2.8
cu128, Triton3.4, Transformers5.5.4, OmniVoice0.1.4 and the fully hashed
checkpointc5fdb5ccb189668d56333f77ba2629f4cd7535f4. Host differences prevent a
GPU-only causal comparison.

The branch is `max-concurency` on the user's fork. Existing PR13 is untouched.
Raw audio, failed attempts, source snapshots, controller hashes, seeds, exact
options, packages, memory and clocks are retained. ASR runs locally on rentals
or the Mac; no paid transcription API is used. Conditioning WAVs, model weights
and credentials are excluded from published sample packages.

## Inspect the evidence

- [396 fixed-duration before/after WAVs and complete sample manifest](../benchmark/multilingual/samples/rtx4070ti-fixed-reference-2026-11-04/README.md).
- [Independent fixed-duration word/speaker review](../benchmark/multilingual/results/quality-late-2026-11-04/speaker-fixed396-independent-v1/report.md) and [all flagged pairs](../benchmark/multilingual/results/quality-late-2026-11-04/speaker-fixed396-independent-v1/flagged-review.json).
- [104-route voice-grid word readout](../benchmark/multilingual/results/quality-late-2026-11-04/global2704-word-readout-v1/) and [complete7405-input two-judge report](../benchmark/multilingual/results/quality-late-2026-11-04/report-global7405-v1/report.md). The large scored JSONL is losslessly gzipped; its decompressed SHA is verified in publication provenance.
- [Runner-path word comparison](../benchmark/multilingual/results/quality-late-2026-11-04/runner-word-analysis132-v1/summary.json).
- [Selected offline Whisper diagnostics and unsupported-language skips](../benchmark/multilingual/results/quality-late-2026-11-04/whisper-fixed-diagnostics-v1/README.md).
- [Earlier396 variable-duration prefix WAVs](../benchmark/multilingual/samples/rtx4070ti-reference-prefix-2026-11-04/README.md) and [original208 samples across104routes](../benchmark/multilingual/samples/rtx4070-2026-11-03/paired-manifest.json).

These three sample sets contain1,000generated WAV files in total. They are
fixtures and paired diagnostic examples, not1,000certified languages or a
successful1,000-request low-latency burst.
