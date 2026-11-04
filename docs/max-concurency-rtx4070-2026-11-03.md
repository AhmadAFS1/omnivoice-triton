# Multilingual concurrency experiments on RTX 4070 and RTX 4070 Ti

**Measured GPU: NVIDIA GeForce RTX 4070.** `nvidia-smi` reports 12,282 MiB
physical framebuffer memory; PyTorch independently reports 12,450,070,528 visible
bytes. UUID `GPU-6eb0a2a5-d251-a91e-8e98-82bc0e371e33`, driver 580.95.05,
CUDA runtime 12.8, power limit 200 W. Runs began November 3, 2026 in
America/Chicago (November 4 UTC). No other compute processes were present at the
model-probe start. ASR inference runs separately after TTS timing. CPU download
and package preparation may run alongside TTS and are disclosed in run logs.

Branch `max-concurency` starts from `c79ce35`, the head of
[upstream PR 13](https://github.com/newgrit1004/omnivoice-triton/pull/13).
The existing PR branch is unchanged. OmniVoice checkpoint
`k2-fsa/OmniVoice@c5fdb5ccb189668d56333f77ba2629f4cd7535f4` is pinned and its
13 files (3,267,470,260 bytes) were checked against upstream file sizes and
SHA256/Git blob identities. Runtime: Python 3.12.3, Torch/Torchaudio 2.8.0+cu128,
Triton 3.4.0, Transformers 5.5.4, OmniVoice 0.1.4.

## What the language inventory means

Lingua currently offers **104 distinct route codes**, after deduplicating the
Filipino/Pilipino labels. The 34 review-only entries are excluded. The benchmark
corpus contains 299 published utterances: short and medium fixtures for all 104
routes, plus the selected third fixture for 91 routes. The separate load pool
contains 1,105 distinct effective `(language, text, reference)` inputs. Source
revisions, licenses and variety caveats accompany each fixture.

Nine deployed Aoede references cover Arabic, Danish, German, Greek, English,
Spanish, Finnish, French and Urdu. Read-only inspection of Lingua's deployed
manifest and S3 reference objects confirmed that the other 95 routes use the
English reference fallback. A fallback reference is not a native voice-quality
qualification. Twi uses the broader Akan ASR mapping; other script/variety
qualifications remain attached to individual records.

## Complete audio and measurement rules

Every successful response contains a complete WAV. No client audio streaming is
introduced. The main profile uses hybrid FP16 inference, 16 denoising steps,
speed 1, denoise enabled, postprocessing off, and serial codec decoding. Reference
registration is measured separately from warm generation.

The 1,000-response target means **all 1,000 complete, valid HTTP 200 bodies
within two seconds of a common burst barrier**. Fast 429 responses, timeouts,
unissued requests and short-circuit errors cannot satisfy it. Summaries separate
freshly generated outputs, in-flight duplicate followers and completed-cache
hits. A warm cache result is not evidence of fresh transformer throughput.

Burst and independently scheduled Poisson tests retain offered load, actual
dispatch lateness, connection creation, full-body latency, whole-cohort completion
and rejection counts. Client WAV writing, hashing and signal inspection happen
outside the timed network phase. These are loopback worker measurements; the
Lingua gateway, internet, MuseTalk and playback are outside the measured path.

## Changes and admission limits

Complete-WAV reuse is opt-in with `--response-cache-max-bytes`; zero, the default,
preserves independent generation. A positive byte budget enables duplicate
coalescing and an in-memory LRU. `--response-cache-ttl 0` enables coalescing
without retaining completed outputs. TTL defaults to 600 seconds. The byte budget
bounds stored payloads, not total process or in-flight generation memory.

The key covers every bound generation form parameter and the registered prompt
ID in one loaded app/model namespace. Uploads and server-side `save_dir` writes
bypass reuse. An evicted prompt ID must still fail validation. Draining or failed
CUDA workers cannot serve cache hits. A disconnected caller detaches from a
shared job; it cannot cancel other callers' generation. Shutdown drains admitted
jobs within the existing worker budget and retains model resources if live GPU
work has not stopped.

`--response-cache-max-inflight-keys` defaults to 64 distinct producers and
`--response-cache-max-waiters` defaults to 4,096 waiting callers. These limits are
separate from GPU queue admission. Raising them admits longer waits; it does not
reduce generation time. Cache health exposes actual bounds, hit counts, shared
jobs and waiting callers. Entries are also bounded to 4,096 to limit metadata.

Responses expose `X-OmniVoice-Response-Source` (`generated`, `coalesced`, `cache`)
and an origin request ID. Followers/hits retain byte-identical complete audio but
receive fresh HTTP request IDs. They do not inherit the original GPU timing
headers as if another inference happened. Reuse makes stochastic generation
repeatable for the entry lifetime, so it must be chosen deliberately and only
after the selected audio passes the application's quality checks.

## Quality gates

All successful WAVs and request metadata are retained. Paired sequential runs
use the same startup seed and fixture order with response caching disabled.
Before/after hashes distinguish exact preservation from changed audio. Concurrent
sampling has no per-request seed guarantee and is not claimed to be deterministic.

Two local open-source GPU judges, `omniASR_LLM_Unlimited_1B_v2` and
`omniASR_CTC_1B_v2`, transcribe blindly with the known language/script. Expected
words are used only afterward for scoring. Identical WAVs are deduplicated by
audio hash and language, with results propagated to every logical response.
Reports preserve WER, CER, disagreements, deletions and source/reference caveats.
CER is necessary for scripts without space-delimited words. These related ASR
models can share errors; neither agreement nor successful HTTP/signal validation
certifies native pronunciation, speaker identity or perceptual quality.

## Initial 299-case paired result

**RTX 4070, same hardware/runtime above.** Both arms completed all 299 requests
with valid complete WAVs and verified mappings for all 104 routes. Every paired
WAV is byte-identical (598 files independently read and SHA256-checked). This
establishes preservation of the baseline output for these fixtures; it does not
establish that every baseline recording has correct words or native pronunciation.

The candidate here is `1eb820f` with `--disable-unused-kv-cache`, serial codec,
eight graphs and response reuse disabled. It includes the graph eviction
allocator change, so it is **not an isolated KV-cache comparison**.

| Sequential cold-shape QA | Cases | Before p95 full-WAV ms | Combined candidate p95 ms | Exact paired WAVs |
| --- | ---: | ---: | ---: | ---: |
| Short, all routes | 104 | 1,651.08 | 2,181.56 | 104/104 |
| Medium, all routes | 104 | 1,791.39 | 2,229.62 | 104/104 |
| Third selected fixture | 91 | 2,236.56 | 2,453.08 | 91/91 |

These initial latency results regressed. CPU/disk/network ASR asset preparation
ran during both arms, and a backup transfer ran during part of the candidate.
No ASR model inference shared the GPU. The logged capture/eviction intervals
account for much of the extra time, while codec and CPU stages also slowed.
Those log intervals include warmup, capture, eviction and reference-destructor
work; they cannot isolate a causal GPU capture penalty. A six-worker controlled
comparison separates baseline, KV-only and combined settings after downloads and
hashing have stopped, and records GPU clocks/power/temperature plus CPU pressure.

The short-only QA records had only eight of 104 baseline responses below 500 ms.
Longer complete audio and cold graph preparation must therefore be part of the
SLO rather than assuming that every request behaves like a warmed short sample.

The updated worker passed **98 real-dependency contract tests** in 30.71 seconds
on the rental. These use fake models/CPU tensors and actual API/threading code;
they are not real neural inference or GPU-capacity evidence. The selected suite
covers duplicate sharing, cancellation, admission/TTL/LRU bounds, API validation,
actual GPU-owner shutdown confirmation, model-option defaults and split slices.
Two earlier test invocation setup errors were retained separately; the successful
command uses the TTS venv and overrides optional coverage-plugin arguments.

An additional offline Mac/Metal Whisper screen used cached large-v3-turbo weights
on 24 unique WAVs. Seven of nine selected product short fixtures matched normalized
expected text exactly; Hindi and Swahili have small differences requiring review.
Their paired candidate audio is identical. Six of nine prerecorded reference
controls matched exactly, demonstrating ASR calibration limits. These checks
cover selected languages only and do not replace the two Omnilingual judges.

## Experimental compute path

Splitting classifier-free guidance can omit padded unconditional transformer
positions. Initial RTX 4070 model-stage probes showed 1.27–1.78× speed ratios
across English, Spanish and Arabic batches of 1, 2 and 8. All nine profiles changed
final audio-token tensors and WAV bytes, so those results do not qualify it as a
quality-preserving default. Semantic and multilingual quality investigation is
required before recommending it.

Disabling unused LLM KV-cache construction preserved final tokens and complete
WAV bytes in the initial English batch-1 and batch-8 probes. The batch-8 reserved
VRAM peak fell from 4,018,143,232 to 3,313,500,160 bytes; synchronized median model
time moved from 3,621.2 to 3,533.9 ms. This is a two-fixture model-stage probe,
not HTTP capacity or all-language quality evidence.

## Scaling interpretation

One thousand active calls are not necessarily one thousand simultaneous TTS
generations. The illustrative call workload assumes one assistant turn per
20 seconds: 10, 100 and 1,000 calls offer 0.5, 5 and 50 TTS requests per second.
Actual capacity must be based on measured fresh response throughput and a
utilization margin, with queue time included in the latency SLO.

One thousand novel complete responses in two seconds requires at least 500
successful generations per second. A guidance split or larger queue alone does
not establish that capacity. Fresh call traffic needs measured per-GPU admission
limits, short queues, warm language/reference state, and multiple workers once
the offered rate exceeds service capacity. Common approved phrases can use
complete-response reuse, but cache misses retain the model's compute cost.

Fleet admission, atomic cross-worker routing, deadlines and autoscaling remain
separate deployment work; this branch does not implement a production GPU fleet.


## November 4, 08:40 UTC checkpoint

This is an ongoing experiment, with a November 4, 10am Chicago (16:00 UTC)
work deadline. The current requested load is 1,000 simultaneous requests and
500 staggered requests. The complete-WAV target is 1–2 seconds, with p95 below
500 ms as the latency objective. Those objectives are **not achieved for fresh
generation**. No production deployment or change to the original PR was made.

A second rental measures a separate single **NVIDIA GeForce RTX 4070 Ti**,
UUID `GPU-5aada52f-9af2-4506-47f7-75e99aaf51ee`: `nvidia-smi` physical VRAM
12,282 MiB; PyTorch visible memory 12,481,134,592 bytes; driver 610.57.04;
CUDA runtime 12.8; 285 W limit; Ryzen 7 5700X, 16 available logical CPUs.
The primary RTX 4070 uses a Xeon E5-2686 v4 with 18 effective CPU cores.
The model files and 109 installed package versions match across machines.
They are separate experiments, not a controlled GPU-only comparison: GPU, CPU,
driver and host differ. Each timed service run uses exactly one GPU, with no
co-resident ASR inference. Published timing is loopback full-body HTTP latency.

### Fresh requests on RTX 4070 Ti

Baseline `c79ce35`, hybrid FP16, 16 steps, speed 1, denoise on,
postprocessing off, serial codec decode, maximum batch 8, and response caching
disabled. The initial eight-graph profile failed during the 100-request burst:
34 successful WAVs and 66 HTTP 503 responses. Capturing `(16, 8, 521)` required
another 132 MiB with only 20 MiB free. The error occurred inside graph capture;
the worker correctly remained unavailable until process restart. These failures
remain in the denominator.

A separate baseline configuration reduced retained graph count to two, without
changing synthesis parameters or the source. All offers below returned complete
HTTP 200 WAVs and passed the basic signal/hash/protocol checks:

| RTX 4070 Ti, graph count 2 | Successful requests | Last full body from cohort barrier | p95 per-request full body |
| --- | ---: | ---: | ---: |
| Single short request | 1/1 | 0.492 s | 0.492 s |
| Ten-request burst | 10/10 | 4.386 s | See retained per-request log |
| Hundred-request burst | 100/100 | 46.326 s | 45.48 s |
| Thousand distinct requests, 104 routes | 1,000/1,000 | 493.556 s | 472.370 s |

The thousand-request cohort contains 1,000 distinct effective inputs and 1,000
verified distinct WAV hashes. It produced 3,751.32 seconds of audio in total,
mean 3.751 seconds per response, at approximately 2.03 requests/second.
Queue time dominates latency: median queue wait was 246.579 seconds, while
median executing batch time was 3.933 seconds. Of the thousand requests,
108 used a target-language native reference and 892 used the English fallback.
ASR word checks for this cohort are still pending at this checkpoint. HTTP 200,
a decodable WAV and a non-silent signal do not establish correct words.

This measured profile is about 247 times below the 500 fresh responses/second
needed to finish a thousand responses in two seconds. That ratio describes this
workload/profile, not a hardware-impossibility proof. Larger queues preserved
acceptance but caused minutes of waiting. They did not create low-latency capacity.

### Complete-response reuse on RTX 4070 Ti

Candidate `377de39`, with unused KV cache disabled, Split CFG disabled,
two retained graphs, serial codec, 16 steps, speed 1 and a 128 MiB complete-WAV
cache with 600-second TTL. Every request below returned HTTP 200, and all 5,316
successful bodies including warmups were retained and independently hash-checked.

| RTX 4070 Ti workload | Last full body from cohort barrier | p95 per-request full body | Response source |
| --- | ---: | ---: | --- |
| Cold identical burst, 1,000 | 6.789 s | 6.434 s | 1 generated, 551 coalesced, 448 cached |
| Hot identical burst, 1,000 | 2.273 s | 2.024 s | 1,000 cached |
| Hot 104-language set, 1,000 | 2.420 s | 2.155 s | 1,000 cached |
| Hot identical, 500 staggered over 1 s | 1.000 s | 2.52 ms | 500 cached |
| Hot identical, 500 staggered over 2 s | 1.999 s | 2.07 ms | 500 cached |
| Hot 104-language set, 500 over 1 s | 1.000 s | 2.58 ms | 500 cached |
| Hot 104-language set, 500 over 2 s | 1.999 s | 2.13 ms | 500 cached |

Staggered schedules place the last offer at 998 ms or 1,996 ms, respectively;
measured drain after the last client dispatch was 1.91–3.08 ms. These warmed
cached staggered tests meet the latency objective. The cached thousand-request
bursts narrowly miss the two-second cohort objective and miss p95 500 ms.
Cached delivery does not synthesize a thousand new utterances. The median
server-reported hot-cache handler time was 0.03 ms, while client request-body send
times spanned about two seconds. Independent client validation is planned before
attributing all burst delay to server processing.

### Both Omnilingual judges completed the first quality stage

**RTX 4070, hardware/runtime as documented above; ASR ran separately from TTS.**
The two default batch-one judges completed 709 distinct audio/script inputs,
covering all 104 routes, at 07:29:40 UTC. The logical evidence includes 1,125
HTTP response records, 84 model-probe records and nine reference controls.
The quality-preserving candidate's 299 WAVs match baseline bytes exactly, so
its ASR results are identical by construction.

The experimental Split CFG path changed all 299 outputs. Aggregate CER fell
slightly for both judges (LLM 10.184% to 9.927%; CTC 18.087% to 17.988%), but
39 paired cases worsened on both judges and 34 improved on both. One Czech
critical fixture changed from both-judge exact to a mismatch; selected Somali
and Bengali cases also warrant review. Aggregate improvement cannot qualify
unchanged quality across languages. Split CFG remains opt-in and unpromoted.

Several reference recordings themselves produce ASR discrepancies, including a
wrong-script Urdu CTC output. Review-required flags include missing calibration,
source provenance and proxy mappings; they must not be interpreted as counts of
incorrect speech. ASR disagreement and actual word errors remain distinct.

A separate RTX 4070 ASR batching prototype held 41 WAVs fixed. CTC batches of
two/four improved transcription speed 1.28×/2.08× but changed normalized output
for 3 and 5 of the 41 clips, respectively. LLM batch two completed all 41 only 1.037× faster and changed
two transcripts. LLM batch four completed only 31/41 before its time budget.
No batching setting qualifies for automatic promotion; the frozen batch-one
quality results remain unchanged.

### Components and remaining qualification

Worker reference registration now reuses a stable prepared handle for the same
reference audio, exact transcript and preprocessing setting, with concurrent
registrations sharing one preparation. Upload parsing waits outside the shared
control-route thread limit. The Lingua gateway branch
`codex/max-concurency-sdk` separately reuses prepared references across target
languages and presets while keeping every synthesis request independent. Its
CPU-only mocked-HTTP validation passed 27 tests plus ten subtests; this is not
GPU throughput evidence. The prior SDK change forwards explicit instruction and
finite guidance controls instead of silently dropping them.

All eight catalog voices have an English reference. Across the 832 route/voice
pairs, 64 have native-reference assets and 768 require English fallback. The
2,704-case plan covers all 104 routes × eight voices × three actual Lingua
presets, plus auto/design cases; 130 control/boundary cases and 28 expected
rejections are separate. The actual default/video/tutor presets differ in speed,
postprocessing and pauses. That GPU qualification is currently running and is
not represented as completed by this report.

An experimental graph-memory admission option, disabled by default, pre-evicts
by retained graph count and normalized row-token weight before attempting a new
capture. It checks architecture/dtype-specific estimated headroom and can use
eager inference before capture instead. Estimates do not guarantee total VRAM
safety. Insufficient estimated workspace returns 503 while leaving the worker
healthy; an actual failed capture still requires restart. GPU memory-pressure,
output-quality and throughput qualification remains in progress.


### Additional observations through 09:10 UTC

The RTX 4070 Ti voice-control gate and first Aoede sweep returned 404 HTTP 200
responses, but two default-preset outputs were 44-byte, zero-frame WAVs (Somali
and Turkmen). All corresponding video/tutor cases passed the basic signal test.
A controlled Turkmen repeat at speed 1.3 produced a nearly silent 1.60-second
waveform with postprocessing off (RMS 0.0000729); full silence trimming produced
an empty WAV. Speed 1 produced substantive audio in the matched repeats. The
Somali original failure did not recur in its four matched diagnostic repeats.
These are failures, not successful audio. The remaining voice grid proceeds as
a diagnostic coverage audit after the failed gate, with no quality promotion.

A separate 130-case control matrix returned 130 HTTP 200 responses with 129
signal-valid bodies. All 28 expected-rejection checks returned HTTP 400. These
validation-only checks do not enter successful synthesis throughput.

Matched-source sequential Aoede/video step-count experiments on the same
RTX 4070 Ti, physical 12,282 MiB/visible 12,481,134,592 bytes, driver 610.57.04,
Torch CUDA 12.8, completed all 104 routes per arm. Speed, reference, text/order,
postprocessing and startup seed were fixed; generation steps changed:

| RTX 4070 Ti, sequential 104-route sweep | p50 full-body ms | p95 full-body ms | Within 500 ms | Basic valid WAVs |
| --- | ---: | ---: | ---: | ---: |
| 16 steps | 479.38 | 572.03 | 73/104 | 104/104 |
| 8 steps | 331.22 | 386.21 | 103/104 | 104/104 |
| 4 steps | 258.99 | 290.03 | 103/104 | 104/104 |

Each arm's first cold request exceeded 500 ms. Blind ASR quality is pending;
these are experimental sequential settings, not a concurrent-capacity result
or a recommendation to change defaults.

The complete 104-route before/combined-candidate short pairs are now packaged
in `benchmark/multilingual/samples/rtx4070-2026-11-03`, with 208 verified WAVs,
per-pair sources, raw judge evidence, source licenses and exact hardware metadata.
Their combined-candidate audio is byte-identical to baseline for all 104 pairs.
Full 299-case evidence remains retained separately. The memory-admission feature
is committed as `47bd743`, disabled by default; its final real-dependency
CPU/fake-model suite passed 152 tests. Subsequent invalid-audio output validation
is a separate change and was not included in those 152 tests.
