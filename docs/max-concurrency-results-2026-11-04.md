# Fresh synthesis, quality and scaling — November 4, 2026

Updated at 15:56 UTC during the authorized timebox ending at 16:00 UTC today
(10am Chicago). **The target of 1,000 fresh complete-audio responses in 1–2 seconds
on one RTX 4070-class GPU remains unmet.** The best completed test of 1,000 novel
inputs takes 374.738 seconds on the RTX 4070 Ti. A separate sequential fixture
reaches p95 full-audio latency of 420.41 ms; that is a different load condition.

All tests require the full WAV before response completion. No client streaming
or MuseTalk changes were introduced. The branch is `max-concurency` on the user's
fork; the existing PR13 is untouched. No production deployment or fleet rollout
is claimed.

Latency measures loopback TTS API dispatch through the complete response body.
It excludes WAN/gateway, input ASR, conversational LLM, MuseTalk and playback
latency; it is not a full speech-to-speech call measurement.

## Fresh request bursts

| Workload and serving profile | Full cohort | Request p95 | Valid full WAVs |
|---|---:|---:|---:|
| RTX 4070, hybrid graph8/M16384, source47 | 760.877 s | 722.015 s | 1,000/1,000 |
| RTX 4070 Ti, earlier hybrid graph2/KV on, c79 | 493.556 s | 472.370 s | 1,000/1,000 |
| Ti, Triton eager B8/KV off, ce48 | 429.822 s | 410.688 s | 1,000/1,000 |
| Ti, Triton eager B1/KV off, ce48 | **374.738 s** | **356.228 s** | 1,000/1,000 |
| Ti, raw sockets, B1/collect0, ce48, cyclic 1,000 inputs | **1,857.899 s** | **1,765.744 s** | **5,000/5,000** |

Burst and staggered p95 values use the recorded linear interpolation. The
sequential table and capture-deferral section identify their differing methods;
raw values and the independent nearest-rank recomputation remain available.

Every offer remains in the success/failure denominator. Signal validity is
separate from correct words and voice. The eager arms use 16 steps, speed1,
original full references, postprocessing off, serial codec1 and response cache
**off**. B1 achieves 2.669 fresh requests/s and drains 12.8% faster than B8 on the
same source/settings. Its 24.1% shorter drain than the earlier c79 hybrid run is
a complete-configuration comparison: source, runner, graph/KV settings and
request execution order differ. This is not an isolated code-only speedup or a
quality-preservation claim for the concurrent fixture.

The 5,000-offer assay repeats the original 1,000 published inputs five times;
it is **5,000 actual inference executions, not 5,000 novel inputs**. All 5,000
full bodies passed the remote signal checks, with zero transport/protocol or
fatal CUDA errors. Counter deltas and independent Executing/Completed log-member
totals both equal 5,000, and cache remains disabled. The legacy per-response
Inference-Executed header is absent and is retained as *unreported* rather than
invented; global counters/logs establish the work separately. Independent local
SHA/SoundFile verification passed all 5,010 WAVs including ten disjoint warmups.
None of the 5,000 measured responses completed within 500 ms or two seconds.
Both blind word checks are complete; their results are below.

This client proves 5,000 distinct preconnected TCP sockets/file descriptors/local
endpoints, with a measured user-space POST dispatch spread of 77.099 ms. It uses
a 600-second benchmark-server keepalive and no shared HTTPX pool. All complete
bodies stay in RAM during timing; hashing, decoding and saving follow the cohort.
Bodies total 900,536,800 bytes and peak client RSS is 1,049,817,088 bytes. The
profile also changes collection delay from 5 ms to 0 ms. Earlier HTTPX eager
runs observed 986 reconnects each after their preconnect phase; those results and
that limitation remain preserved. Neither connection acceptance nor queue
survival is evidence of the requested latency capacity.

The older 10-request cohorts took 8.317 s on the 4070 and 4.386 s on the Ti;
100 requests took 80.850 s and 46.326 s. They did not all output instantly. These
columns compare complete serving configurations and host CPUs, not GPU-only
causal effects.

## Sequential latency and exact output preservation

| 132 full-reference cases on the Ti | Median | p95 | Full WAV within 500 ms |
|---|---:|---:|---:|
| Hybrid, graphs8 | 482.17 ms | 577.29 ms | 83/132 |
| Base eager | 372.41 ms | 465.69 ms | 130/132 |
| Faster, graphs8, no Triton kernels | 520.66 ms | 621.95 ms | 24/132 |
| Triton eager | **333.93 ms** | **420.41 ms** | **131/132** |

Triton eager and hybrid match all 132 complete WAV bodies exactly. Base and
Faster also match 132/132; Base and hybrid differ on every waveform. Existing
Triton math therefore has its own quality comparison. Both Meta ASRs match
37/132 Triton/hybrid scripts exactly versus 35/132 Base/Faster. Relative to the
hybrid result, Base loses five baseline-exact cases and gains three. Mixed
language-specific results do not establish a universally better path. The first
three p95 values use linear interpolation; Triton eager uses nearest rank.

The new default-off `--fused-rope` implementation is pushed in deaa1d4. On the
pinned RTX 4070, it passed 219 CPU contracts, 108 tensor/layout pairs, 203 observed
sequence lengths, four updated-input graph replays, six loaded-model pairs
(EN/ES/AR at B1/B8, all 16 denoising steps), and an actual API constructor/startup
screen. Every API WAV in the 132-case screen is byte-identical to the unfused
same-host baseline. Startup verifies unchanged CPU/CUDA RNG state before ready;
no runtime fallback or CUDA fault occurred. Initial harness reporting and finite
budget failures are preserved, with the final six-pair proof explicitly joining
five completed pairs and a separately authorized remaining-only continuation.
These gates establish preservation of the tested computation and audio. They
do not certify native pronunciation or concurrent capacity.

The subsequent clean Ti eager OFF/ON pair returned all 264 responses valid and
all 132 paired WAVs byte-identical. Median latency worsened 328.70→338.65 ms,
mean improved 353.92→349.04 ms and p95 improved 413.80→390.44 ms. Excluding each
arm's first request, mean improved by only 0.9%. Fused startup warmup took
512.44 ms separately from request latency. This baseline-first single run is
mixed evidence, not a general eager-throughput speedup; the option stays off by
default. Every OFF and ON body matches prior two-judge evidence by exact SHA,
expected text and ASR language/script, requiring zero new ASR inputs.
These two p95 values use linear interpolation; nearest-rank values are
415.08→391.47 ms and do not change the interpretation.

Capture deferral also remains optional and defaults to zero. A matched 528-request
4070 screen returned all four arms' 132 WAVs valid, with N16/N64/eager each
byte-identical to N0. Nearest-rank p95 was 1,664/1,504/3,201/3,150 ms for
N0/N16/N64/eager. Captures fell 118→40→1 for N0/N16/N64, without capture failures,
pressure cleanups or admission rejections. CPU/GPU clock drift was also observed
in this ordered single-run screen. N64 is not recommended from these results;
there is no universal best-threshold claim.

## Words, voices and the quality limits

Both frozen Omnilingual judges cover all **15,058 unique audio/script inputs**
in the final canonical set, with exact ID/SHA sets and consistent reused
publications. All **44,225 logical outcomes**, including rejected/failed work
and repeated audio, remain preserved. This is completed ASR coverage, not a
claim that all outputs have good words or native pronunciation.
LLM runs use the modern host's fixed protocol and CTC uses the primary host's
legacy-B1 protocol; cross-host namespaces are not silently mixed. ASR runs
locally on the GPUs, without a paid transcription API.

For the 1,000-request B1/B8 comparison, excluding ten warmups, both judges match
131 scripts exactly at B1 versus 120 at B8. B8 loses 54 B1-exact cases and gains
43; both judges' CER worsen on 224 pairs and improve on 184. CTC mean CER changes
0.323→0.328 while LLM changes 0.197→0.159. The judges differ and neither serving
configuration is universally word-quality cleared. A WAV that is valid signal
is not automatically correctly spoken text.

The matched earlier c79 run has 136 both-judge exact scripts, versus 131 at B1
and 120 at B8. All 1,000 semantic text/control/reference signatures, all nine
reference SHA maps and startup seed match. Relative to c79, B1 loses 57 exact
matches and gains 52; B8 loses 59 and gains 43. Both judges' CER worsen on
225/239 pairs and improve on 195/191 at B1/B8. Zero B1 and five B8 bodies match
c79 byte for byte. Runner/KV/source/batching/scheduling and RNG-order differences
limit attribution. The 24.1% shorter B1 drain is therefore **not established as
quality-preserving across this concurrent 1,000-request fixture**. Exact
preservation claims in this report apply only to the explicitly matched runner
and fusion gates above.

A third-family cached Whisper diagnostic selected all 109 c79→B1 Meta exactness
flips. It transcribed 92 supported pairs (184 WAVs, 47 selected routes), with
17 unsupported pairs explicitly skipped. Among 48 supported Meta-loss pairs,
Whisper CER worsens on 20, improves on nine and is unchanged on 19; it loses
zero CER on 11 and gains it on one. Fifteen have identical Whisper transcripts
before and after. Among 44 supported Meta-gain pairs it gains zero CER on eight
and loses it on none. This adds word-risk evidence and judge disagreements,
without estimating a population failure rate or establishing native correctness.
The offline Apple Metal run took 76.839 seconds, with no API call or download.

For the raw 5,000-request repeat study, both judges match 604/5,000 normalized
scripts exactly; 690 have zero CER under both. Among 1,000 inputs synthesized
five times each, 29 are both-judge exact in all five repetitions and 235 are
exact in at least one. The five cycle exact counts are 120/119/114/122/129;
these data do not establish monotonic quality degradation with load. All 5,000
waveforms differ, despite the shared startup seed: it is not a per-request seed.
Per-route and per-input CER variance/range, expected scripts and both hypotheses
remain available. ASR errors and script differences contribute to these scores.

A targeted third-family check of five Mandarin/Cantonese outlier pairs further
illustrates this uncertainty. For one raw Mandarin clip, Whisper matches the
script while the Meta LLM emits 339 characters for a 12-character reference;
that long transcript does not establish literal repeated speech. Another
Mandarin clip and the Cantonese comparisons still require review. No global
quality clearance follows from this selected diagnostic.

The voice/config grid covers all 104 Lingua routes, eight stock voices and three
real presets, plus auto/design cases. Of 2,704 offers, 2,700 have valid signal
and four are empty. Both judges match the normalized script exactly on 463
cases; 535 have zero character error under both. Judges disagree on 2,103 cases.
Only 304 offers jointly have valid full audio within 500 ms and zero CER under
both; 283 satisfy strict normalized exactness as well. Many routes lack native
reference calibration and proxy mappings remain explicit. These are fixture
screens, not proof that every disagreement is bad speech or that every word in
a language is supported. The audio guard rejects empty/nonfinite/all-zero output;
it does not repair words or certify near-silent audio.

The 16-step benchmark/Lingua request profile stays. The worker API fallback
remains 32 steps when the client omits the parameter. Baseline/load experiments
explicitly supply 16; experimental step/control overrides remain labeled.
Eight/four steps improve sequential speed but worsen
both judges' CER on 22/50 of 104 matched routes. Reference shortening also remains
experimental. With original target duration held fixed, English-reference p95
falls 570.6→385.0/418.2 ms at ten/seventeen words, but three/four baseline-exact
word matches are lost and three/one speaker-top1 ranks are lost. All before/after
WAVs and every flag remain available. Offline ECAPA and score-selected Whisper
diagnostics supplement the two Meta judges; none substitutes for native
listening, MOS or speaker-identity certification.

A separate EN B1/B8 four-arm Split CFG/reduction diagnostic retained all 108
signal-valid WAVs. Split is faster in that small warmed screen, but all split
versus concatenated comparisons change tokens and float/PCM audio. Disabling
reduced-precision reductions changes even the B1 concatenated baseline; the B8
concatenated True/False pair is exact. This does not provide a lossless fix for
Split CFG. It stays off; the raw tensors, original failures and later word
checks are kept separately from HTTP availability.

## Scaling plan and present implementation

Implemented foundations include asynchronous request/reference waiting, a single
GPU owner, bounded admission, prepared-reference caching, exact-response
coalescing/cache, explicit unused-KV disabling, graph memory admission, generated
audio validation, optional capture deferral and default-off rotary fusion.
These remove duplicate work, host blocking and some memory faults. They do not
make novel speech free. Deep queues are diagnostic; accepted work can still
wait many minutes.

| Work | Implementation and qualification |
|---|---|
| Async waiting, bounded queues, reference/response reuse, KV and graph-memory controls | Implemented; CPU contracts and the documented GPU/API/load experiments completed |
| Optional rotary fusion and capture deferral | Implemented and output-preservation tested; experimental/default-off, with mixed latency results |
| All 104 routes, eight voices and real presets/sliders | Fixture generation and word/speaker diagnostics completed; universal word/native quality remains unqualified |
| Shared graph pool | Prototype only; shared/alias gates never reached within its budget |
| Fleet router, deadline cancellation, autoscaling and SLO-qualified replica counts | Planned; not deployed or fleet-tested |

The isolated Lingua SDK prepared-reference cache/control-forwarding work is
pushed on its separate branch, with its checks recorded. This experiment does
not claim an application deployment or MuseTalk change.

To reproduce the measured serving configuration, use the pinned runtime and
checkpoint above, register matching full references before timing, and launch:

```sh
python -m omnivoice_triton.cli.api_server \
  --model /path/to/pinned/model --runner triton --device cuda --dtype fp16 \
  --no-asr --ip 127.0.0.1 --port 18080 \
  --batch-collect-ms 0 --max-batch-requests 1 \
  --max-pending-requests 4 --max-pending-gpu-tasks 4 \
  --max-batch-target-tokens 4096 --max-batch-conditioning-tokens 4096 \
  --max-cuda-graphs 0 --audio-decode-batch-size 1 \
  --disable-unused-kv-cache --response-cache-max-bytes 0
```

Supply `mode=clone`, `text=<input>`, its explicit `language` and registered
`prompt_id`, then `num_step=16`, `speed=1`, `guidance_scale=2`, `class_temperature=0`,
`position_temperature=5`, `t_shift=0.1`, `layer_penalty_factor=5`, `denoise=true`,
`postprocess_mode=off` and `insert_pauses=false` with each measured request.
The archived harness also applies startup seed 20261103 before import and
records all transport/timing
conditions; the command alone does not enforce request ordering or reproduce a
published waveform. Keep experimental fusion/Split/short-reference/lower-step
changes separate. The pending4 setting is the measured overload policy, not a
guarantee of the 500 ms SLO.

| Active calls, assuming one TTS turn per 20 s | Fresh offered RPS | Ideal workers at 2.669 RPS |
|---|---:|---:|
| 10 | 0.5 | 1 |
| 100 | 5 | 2 |
| 1,000 | 50 | 19 |
| 5,000 | 250 | 94 |

These rounded-up counts are **saturation floors**, not SLO-qualified capacities.
They leave no queue, variability, networking or failover headroom. A sequential
service p95 around 420 ms leaves little waiting time in a 500 ms full-audio
budget. One thousand simultaneous fresh responses in two seconds need 500 RPS,
about 187 times the current single-GPU observed rate before latency headroom.
A production fleet/router has not been deployed or validated here.

The next qualification should start with two and four matched workers at the
measured low-rate schedules, then increase offered load while keeping all-offer
latency, rejection and joint-word results visible. Use a global bounded router
and prepared-reference locality; multiple queued requests on one GPU are not
additional replicas. Test deadline/cancellation behavior without changing the
full-WAV response contract. Before lowering steps or changing conditioning/math,
compare the 16-step Lingua profile with the worker's 32-step baseline and expand
native reference/listening calibration. The measured roughly187-fold throughput
gap requires a larger computational or fleet change than these kernel gains;
no single-GPU 500RPS result is established.

The next serving design uses independent warmed workers behind deadline-aware
routing by model and prepared-reference locality, bounded per-worker queues,
cancellation of expired queued work, explicit overload responses and scaling
from measured fresh service demand. Validate active calls separately from bursts.
The completed staggered screen assumes 10/100/1,000/5,000 calls and offers
0.5/5/50/250 RPS with fixed Poisson schedules. Results use scheduled-arrival-to-body
latency, including client dispatch lateness:

| Assumed calls / offered RPS | Valid WAVs / all offers | HTTP429 | Accepted-response p95 | Valid within 500 ms / all offers |
|---|---:|---:|---:|---:|
| 10 / 0.5 | 30/30 | 0 | 596.95 ms | 27/30 |
| 100 / 5 | 183/300 | 117 | 1,800.40 ms | 6/300 |
| 1,000 / 50 | 164/3,000 | 2,836 | 1,970.97 ms | 1/3,000 |
| 5,000 / 250 | 146/15,000 | 14,854 | 2,218.99 ms | 0/15,000 |

All 18,330 cohort offers have outcomes: 523 valid HTTP200 WAVs and 17,807
HTTP429 rejections, with zero transport errors. Ten valid warmups are separate.
The profile uses source ce48, eager Triton B1/collect0, pending4, graphs/KV/cache
off, 16 steps, full references and serial codec. Rejections are availability
failures. The p95 column is conditional on accepted responses, not a successful
all-offer SLO. Even the observed 10-call assumption misses p95 below 500 ms.
These are short fixed-seed request schedules, not connected VoIP streams or
long-run fleet qualification.

Both judges match 10/30, 39/183, 23/164 and 15/146 accepted scripts exactly in
the four phases; both zero-CER counts are 12/30, 43/183, 27/164 and 19/146.
All 533 saved WAVs including warmups have complete two-judge evidence and
independent SHA/decode/signal verification. Word checks remain diagnostics.

Requiring valid audio **and** both-ASR normalized exactness within 500 ms gives
9/30, 2/300, 0/3,000 and 0/15,000 over all offers. The corresponding both-zero-CER
joint counts are 11, 2, 0 and 0. These join actual waveform hashes, expected text
and script to scheduled-arrival latency; rejections never qualify. They remain
ASR diagnostics rather than native speech-quality certification.

Cached delivery remains separate. A previous Ti transport screen delivered
1,000 already-cached full bodies in 0.554 s with preconnected raw asyncio
(p95 516 ms), versus 1.023 s with a dedicated HTTPX client (p95 810 ms).
It establishes neither fresh synthesis throughput nor p95 below 500 ms.
The optional new cold1k/hot5k cached assay was prepared and CPU-tested but
**skipped** because the prerequisite staggered retention could not finish before
its 15:12 UTC start gate. No measurement is inferred from that unexecuted plan.

Shared graph-pool research also remains unqualified. Its finite probe exhausted
281 seconds in private reference generation, before the shared or alias gates
ran. All 46 saved WAVs/22 tensor archives were retained; every waveform matches
prior ASR evidence exactly. Partial reference timings are not shared-pool memory,
speed or arbitrary-order safety evidence. No shared-pool code was deployed.

## Hardware and reproducibility

The Ti host is a Ryzen 5700X with 16 visible logical CPUs, about 33.6 GB
OS-visible RAM, 285 W and driver
610.57.04; physical VRAM is 12,282 MiB and Torch-visible memory is 12,481,134,592
bytes. The 4070 host is a Xeon E5-2686 v4 with 18 advertised effective cores,
OS-visible RAM about 135 GB, 200 W and driver 580.95.05; physical VRAM is 12,282 MiB and visible
memory is 12,450,070,528 bytes. Host differences prevent a GPU-only comparison.
Both use CUDA12.8, Python3.12.3, Torch/Torchaudio2.8+cu128, Triton3.4,
Transformers5.5.4, OmniVoice0.1.4 and fully hashed checkpoint
c5fdb5ccb189668d56333f77ba2629f4cd7535f4.

Final allocation snapshots at 15:28 UTC separately record OS-visible RAM and
container limits. The 4070 container has memory.max 64,806,191,104 bytes and a
17.28-core CPU quota across 72 visible logical CPUs; the Ti has memory.max
32,225,886,208 bytes and a 15.36-core quota across 16 logical CPUs. These are
cleanup-time observations, not substitutes for the per-run telemetry. Offer
metadata advertised 18/16 effective CPU cores respectively.

Raw audio, failed attempts, exact source/config snapshots, seeds, packages,
controller hashes, memory and clocks are retained. Conditioning WAVs, model
weights and credentials are excluded from published packages. Final rental
retention and cleanup are complete. Metadata backups include 5,109 hashed files
(1,367,412,367 bytes), in addition to independently retained audio/tensor archives.
Both task transfer authorizations/private keys were removed without reading or
copying private contents, and original SSH authorization bytes were restored
exactly. Only task rentals 59058485 and 59071593 were destroyed, at 15:29:48 and
15:30:16 UTC. Vast inventory verified no remaining task instances at 15:30:54.

## Inspect the evidence

- [Final load, word, methodology and cleanup evidence](../benchmark/multilingual/results/load-and-quality-2026-11-04/README.md), including all run metadata and truthful failures.
- [Complete final 15,058-input two-judge report](../benchmark/multilingual/results/load-and-quality-2026-11-04/full-two-judge-report/report.md), [strict coverage proof](../benchmark/multilingual/results/load-and-quality-2026-11-04/quality/final-evidence-proofs-v1/FINAL15058_TWO_JUDGE_COMPLETE.json), and [raw-five-repeat word analysis](../benchmark/multilingual/results/load-and-quality-2026-11-04/quality/raw5000-repeat-word-analysis-v1/summary.json).
- [Staggered load integrity/latency audit](../benchmark/multilingual/results/load-and-quality-2026-11-04/quality/retained-audit-v1/report.md) and [accepted-word scores](../benchmark/multilingual/results/load-and-quality-2026-11-04/quality/final-staggered-word-analysis-v1/summary.json).
- [Joint all-offer words, signal and latency analysis](../benchmark/multilingual/results/load-and-quality-2026-11-04/quality/final-staggered-joint-slo-v1/README.md).
- [Concurrent before/after Whisper diagnostic](../benchmark/multilingual/results/load-and-quality-2026-11-04/quality/whisper-novel1000-diagnostics-v1/README.md) and [selected Mandarin/Cantonese outlier cross-check](../benchmark/multilingual/results/load-and-quality-2026-11-04/quality/whisper-raw-outlier-diagnostics-v1/README.md).
- [396 fixed-duration before/after WAVs and complete sample manifest](../benchmark/multilingual/samples/rtx4070ti-fixed-reference-2026-11-04/README.md).
- [Independent fixed-duration word/speaker review](../benchmark/multilingual/results/quality-late-2026-11-04/speaker-fixed396-independent-v1/report.md) and [all flagged pairs](../benchmark/multilingual/results/quality-late-2026-11-04/speaker-fixed396-independent-v1/flagged-review.json).
- [104-route voice-grid word readout](../benchmark/multilingual/results/quality-late-2026-11-04/global2704-word-readout-v1/) and [complete earlier 7,405-input two-judge report](../benchmark/multilingual/results/quality-late-2026-11-04/report-global7405-v1/report.md). The large scored JSONL is losslessly gzipped; its decompressed SHA is verified in publication provenance.
- [Runner-path word comparison](../benchmark/multilingual/results/quality-late-2026-11-04/runner-word-analysis132-v1/summary.json).
- [Selected offline Whisper diagnostics and unsupported-language skips](../benchmark/multilingual/results/quality-late-2026-11-04/whisper-fixed-diagnostics-v1/README.md).
- [Earlier 396 variable-duration prefix WAVs](../benchmark/multilingual/samples/rtx4070ti-reference-prefix-2026-11-04/README.md) and [original 208 samples across 104 routes](../benchmark/multilingual/samples/rtx4070-2026-11-03/paired-manifest.json).

The [actual fusion OFF/ON samples](../benchmark/multilingual/samples/rtx4070ti-fused-rope-2026-11-04/README.md)
and [complete product qualification](../benchmark/multilingual/results/fused-rope-2026-11-04/README.md)
add 264 generated WAV files. These four published sets contain 1,264 WAVs. They are
paired diagnostic examples. Concurrent timings and quality qualifications are
reported separately above.
