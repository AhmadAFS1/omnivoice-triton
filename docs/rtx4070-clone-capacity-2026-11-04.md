# RTX 4070: full-audio cloned-voice capacity and hardware bottlenecks

**Actual GPU: NVIDIA GeForce RTX 4070, 12,282 MiB physical VRAM
(`nvidia-smi`), 12,536,250,368 bytes visible to PyTorch.** UUID
`GPU-794dbd73-3f73-b50b-45b3-8dbce083e264`, driver 615.71.09, CUDA runtime
12.8, 200 W limit. Measured November 4, 2026, approximately 20:40–20:58 UTC.
Every performance row below uses this same card. This is actual GPU inference;
the evidence audit is CPU-only and the separate ASR screen uses Apple Metal.

One OmniVoice-Triton server completes about **2.25 fresh cloned-voice requests/s**
for this fixture. Two ordinary server processes reach about **2.18/s**. Two
independent servers using verified CUDA MPS clients reach about **2.55/s**:
**12.94% above the ordinary single-server C8 screen**, not a doubling.
One server with MPS remains about 2.24/s. All 1,260 timed requests return complete,
signal-valid WAVs, with 420 additional warmup WAVs retained. Speech correctness
and speaker identity are separate from those availability counts.

The rental was destroyed at **20:59:34.998 UTC**, with inventory absence verified
at 20:59:36.269. Its approximately 25.6-minute lifetime at the recorded
$0.248519/hour offer implies about **$0.106 compute/storage-rate cost**, before
transfer, billing granularity, tax or other adjustments. This is an estimate,
not an invoice. Only this task's instance, 59158267, was changed.

## What was tested

- User fork, branch `max-concurency`, product source pinned to
  `096946d75e2ddcb2797c375b0ee50a1933793b36`. All 24 product Python source files
  were independently matched to the local fork after the run. This experiment
  changes benchmark wrappers and deployment arrangement, not product/model code.
- OmniVoice 0.1.4, checkpoint
  `c5fdb5ccb189668d56333f77ba2629f4cd7535f4`; all 13 downloaded model files were
  SHA-256 hashed. Torch/Torchaudio 2.8.0+cu128, Triton 3.4.0, Transformers 5.5.4,
  Python 3.12.3. Frozen package and source provenance are retained.
- **Every timed request uses `mode=clone`.** There are 60 distinct short texts
  across 60 Lingua language routes, with nine existing full-length Aoede reference
  clips and their transcripts. These are nine reference clips, not nine independently
  verified speakers. Routes lacking a same-language clip use an explicitly labeled
  English cross-language clone. This is not the earlier complete 104-route/eight-voice grid.
- Original references remain uncut. The English reference is 13.931 seconds;
  mean generated response duration is 2.895 seconds. These relatively long
  conditioning clips materially affect the amount of model computation.
- Prepared clone prompts are registered before timing and reused by `prompt_id`.
  **Generated-response caching and duplicate coalescing are disabled.** Every timed
  request performs inference, independently confirmed by GPU-worker member counters.
- Runner `triton` (eager), FP16, unused KV cache off, 16 steps, speed 1, denoise
  on, postprocessing off, sentence splitting off, serial codec batch 1. Split CFG,
  fused RoPE, SageAttention and CUDA Graphs are off. The unmodified API fallback
  remains 32 steps if a client omits the parameter; this benchmark explicitly sends 16.
- Batch limits 1 and 8; collection delay 0 ms; pending-request limit 128;
  default request buckets 1,2,4,8,16,32 and padding/token budgets. Actual batches
  may be smaller than the configured maximum; distributions are in the raw summaries.
- Each phase sends 60 fresh generations through a bounded client concurrency
  of 1, 8 or 32 for one server, and 2, 8 or 32 for two. Two-server traffic is split
  round-robin across ports 8002/8003. Concurrency is total across both servers.
  The same corpus/order is used in every arm. Each arm first warms its whole corpus.
- Loopback HTTP dispatch through **complete WAV body receipt**, with independent
  preconnected client connections. Decoding, hashing and saving bodies happen after
  phase timing. The client concurrency gate precedes request dispatch: later work
  waits client-side. These are finite closed-loop capacity screens, not all-60-at-once
  arrival tests or a long steady-state production qualification.

Host: AMD EPYC 7F52, 32 visible logical CPUs; cgroup CPU quota 15.36 CPU-time
equivalents and RAM limit 183,352,950,784 bytes. Torch uses 16 intra-op and 32
inter-op threads. No unrelated CUDA process was observed before the diagnostic
model load. Two-server arms intentionally have two co-resident TTS processes;
MPS adds its scoped service. ASR ran on the separate Mac. Setup/backup transfers
occurred around startup or after timing, not as a sustained benchmark load.

## Complete-audio throughput and latency

**All rows: RTX 4070, same physical/visible VRAM above; 60/60 valid full WAVs per row.**
RPS is 60 divided by full cohort wall time. p95 uses linear interpolation.

| Arrangement | Batch limit | Total concurrency | Complete requests/s | Request p95 | Complete within first second of phase |
|---|---:|---:|---:|---:|---:|
| One server | 1 | 1 | 2.249 | 505.7 ms | 2 |
| One server | 1 | 8 | 2.265 | 3,712.3 ms | 2 |
| One server | 1 | 32 | 2.246 | 14,400.8 ms | 2 |
| One server | 8 | 1 | 2.244 | 507.5 ms | 2 |
| One server | 8 | 8 | 2.149 | 4,027.4 ms | 0 |
| One server | 8 | 32 | 2.018 | 17,407.4 ms | 0 |
| Two ordinary servers | 1 | 2 | 2.187 | 1,051.9 ms | 2 |
| Two ordinary servers | 1 | 8 | 2.181 | 3,938.2 ms | 1 |
| Two ordinary servers | 1 | 32 | 2.176 | 15,014.8 ms | 1 |
| Two ordinary servers | 8 | 2 | 2.183 | 1,048.1 ms | 2 |
| Two ordinary servers | 8 | 8 | 2.170 | 4,129.0 ms | 0 |
| Two ordinary servers | 8 | 32 | 1.947 | 18,736.2 ms | 1 |
| Two MPS servers | 1 | 2 | 2.555 | 858.2 ms | 2 |
| Two MPS servers | 1 | 8 | **2.558** | **3,309.9 ms** | 2 |
| Two MPS servers | 1 | 32 | 2.551 | 12,675.3 ms | 2 |
| Two MPS servers | 8 | 2 | 2.538 | 871.2 ms | 2 |
| Two MPS servers | 8 | 8 | 2.388 | 4,628.4 ms | 0 |
| Two MPS servers | 8 | 32 | 2.085 | 21,197.4 ms | 0 |
| One MPS server, control | 1 | 1 | 2.238 | 508.2 ms | 2 |
| One MPS server, control | 1 | 8 | 2.246 | 3,725.5 ms | 2 |
| One MPS server, control | 1 | 32 | 2.244 | 14,460.4 ms | 2 |

For the user's one-second question: this fixture completes **two full cloned
responses in its first second** on the preferred B1 paths. Sustained cohort rates
are approximately 2.25/s or 2.55/s, so individual aligned seconds vary. Some
two-MPS one-second bins contain four completions, but that is not a sustained
four requests/s capacity result. Batching can also deliver several bodies together
after a long wait; a busy completion bin alone does not establish low latency.

With two MPS servers and total concurrency 2, all 60 individual request latencies
are under one second, with p95 858.2 ms. At concurrency 8, only two individual
requests finish within one second of their own dispatch. At concurrency 32,
the p95 is 12.68 seconds. None of these results certifies the requested p95
below 500 ms, thousands of concurrent callers, or 25 fresh cloned requests/s.

These are single ordered screens of 60 inputs each, not randomized repeated
trials or confidence intervals. GPU temperature/clock and stochastic execution
vary. The consistent three B1 points support further qualification of two-MPS
serving; they do not establish a universal production speedup.

## Where time goes

**RTX 4070, ordinary single server, batch limit 1, concurrency 1; means over 60 requests.**

| Stage | Mean |
|---|---:|
| API preparation before queue | 0.55 ms |
| Queue wait | 0.13 ms |
| Whole model generation | **438.25 ms** |
| Iterative generation, included in whole model | **409.78 ms** |
| Audio codec decode, included in whole model | 25.39 ms |
| Audio postprocessing, included in whole model | 0.20 ms |
| WAV response encoding | 1.21 ms |

The model accounts for roughly 99% of server-side elapsed time at low load.
Iterative generation accounts for approximately 93.5% of model time. Its input
preparation is a nested metric, not an extra stage to add. At concurrency 8,
mean model time remains 440.21 ms while mean queue wait reaches 2,877.56 ms;
at concurrency 32, mean queue wait is 10,059.68 ms. HTTP admission can accommodate
more requests, but GPU work still drains at nearly the same rate.

Preparing a new clone is a separate cost. The first reference registration in
the first arm took 613.4 ms, including the initial cold path. The next eight
took 58.0–81.9 ms. Prepared-reference lookup averages about 0.19 ms during
the sequential requests. Model/API readiness was about six seconds per measured
single-server startup. These exclude the initial environment/model download,
external upload time and any reference transcription: known transcripts were provided.

The diagnostic English clone uses a **457-position sequence**, with the two
classifier-free guidance branches combined into a batch dimension of 2 even
for one request. The pinned upstream code creates a full conditioning-plus-target
sequence, pads the unconditional branch to its length, and performs 16 full
model forwards. Bidirectional conditional attention permits the reference states
to depend on target states. Reusing a prepared voice avoids audio re-encoding;
it does not eliminate this reference-conditioned transformer work. A naive
immutable reference KV cache would change the computation.

The scope here is the TTS worker. WAN/mobile transport, Lingua gateway, input
ASR, conversational LLM, MuseTalk and playback were not traced. Their latency or
resource competition cannot be inferred from these loopback measurements.

## What matters in the hardware

**RTX 4070, representative C8 screens; 100 ms NVIDIA-SMI samples aligned to each phase.**

| Arrangement, batch limit 1 | GPU active-time mean | Device VRAM peak | Power mean / 200 W | TTS CPU use | TTS resident RAM |
|---|---:|---:|---:|---:|---:|
| One ordinary server | 96.1% | 2,642 MiB (2.58 GiB) | 191.0 W | not sampled in this arm | not sampled in this arm |
| Two ordinary servers | 99.3% | 5,279 MiB (5.16 GiB) | 191.3 W | not sampled in this arm | not sampled in this arm |
| Two MPS servers | 98.6% | 5,304 MiB (5.18 GiB) | 193.9 W | 2.02 CPU cores total | 4.50 GiB RSS total |
| One MPS server, control | 95.3% | 2,672 MiB (2.61 GiB) | 191.5 W | 1.01 CPU cores | 2.26 GiB RSS |

CPU/RAM figures come from a separate 100 ms process sampler in the MPS/control
arms. CPU excludes the helper and MPS service; a separate MPS-service snapshot
showed 0.2% CPU and about 101 MiB RSS. Summed process RSS double-counts shared
resident pages and is not unique physical RAM consumption. Samples can miss
brief peaks. Torch intra-op thread count is not the count of fully occupied cores.

1. **VRAM capacity is not the main limit in this profile.** One B1 model uses
   only about 22% of physical VRAM, yet the GPU is active around 96% of the time
   and draws about 191 W. Two models fit, but do not create more compute units
   or bandwidth. At B8/C32, two ordinary processes reach 7,675 MiB and two MPS
   processes 8,216 MiB, still below physical capacity. No OOM occurred.
2. **Dense FP16 matrix throughput matters.** Separate observed-shape probes
   and CUDA traces show substantial matrix work. For B1, GEMMs account for
   249.63 ms, **60.8% of profiled CUDA kernel-active time**; attention accounts
   for 68.0 ms/16.6%. The LLM-backbone ranges correlate to 92.3% of kernel time.
   More GPU matrix capability can help this workload, provided the serving
   implementation uses it effectively.
3. **Memory movement and small kernels also matter.** Elementwise/copy kernels
   take 14.0% of B1 kernel time, with additional concatenation, normalization
   and activation work. At B8, GEMMs fall to 52.3% of kernel time while
   elementwise/copy rises to 20.9%; multiplying batch size does not multiply
   throughput. A 128 MiB device-copy probe measures 461.1 GB/s read-plus-write
   traffic. This synthetic result is not measured model DRAM throughput.
4. **TFLOPS must match precision and accumulation.** Observed-dimension dense
   FP16 GEMM probes achieve 44.7–63.3 TFLOPS; a 4096-cubed probe reaches 64.4.
   FP32 shader TFLOPS, dense FP16 Tensor Core throughput and sparse/FP8 AI
   headline figures describe different work. NVIDIA's architecture tables also
   distinguish FP16 versus FP32 accumulation. Those headline figures cannot
   be substituted directly into a TTS capacity estimate.
   [NVIDIA Ada architecture, Appendix A](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf).
5. **CPU scheduling/context overhead leaves some recoverable capacity.**
   Two processes with MPS outperform ordinary time-sliced processes here.
   NVIDIA documents that MPS can allow cross-process kernels to overlap when
   resources permit. Both actual TTS PIDs were verified as MPS clients for
   both batch-limit arms. Merely launching two ordinary HTTP servers did not
   help. MPS was task-scoped and quit afterward.
   [NVIDIA MPS architecture](https://docs.nvidia.com/deploy/mps/architecture.html).

The one-request English shape executes approximately **13.126 trillion FP16
linear-layer FLOPs**, counting multiply-add as two operations, before attention
and other operations. The B8 shape scales that linear work almost exactly eightfold.
At 25 identical requests/s, those linears alone would require about **328 TFLOPS**.
For context, NVIDIA presents an optimistic RTX 4070 FP16 peak of about 116 TFLOPS
in its Vulkan matrix talk; actual PyTorch GEMM rates here are lower and differ
by accumulation/shape. More HTTP concurrency cannot remove that work. Achieving
25/s for this full-reference profile would require substantial algorithmic work
reduction or additional/stronger hardware.
[NVIDIA, Cooperative Matrix 2, slides 5–7](https://www.vulkan.org/user/pages/09.events/vulkanised-2025/T47-Jeff-Bolz-NVIDIA.pdf).

This is a mixed compute/memory workload, not a proven pure bandwidth or pure
Tensor Core saturation result. NVIDIA explains why arithmetic intensity and
actual operation shape determine the limiting resource.
[NVIDIA performance guide](https://docs.nvidia.com/deeplearning/performance/dl-performance-getting-started/index.html).
The rental rejected Nsight Compute counter access with `ERR_NVGPUCTRPERM`.
That failed probe is retained. We did not change host permissions. Therefore
SM occupancy, Tensor Core saturation and actual DRAM bandwidth were not directly
measured. NVIDIA-SMI GPU activity measures busy time, not percentage of theoretical
FLOPS; its memory-utilization counter is memory activity time, not VRAM capacity
or percent of theoretical bandwidth.

Profiler overhead is material: B1 profiled latency was 618.5 ms versus approximately
422.8 ms for the five unprofiled diagnostic repetitions. Kernel shares describe
the instrumented run, not an additive unprofiled latency decomposition. B8
unprofiled mean was 3,421.3 ms for eight outputs. CPU-thread screens at 16/1/4
threads have medians 448.7/457.3/452.3 ms with exact float-audio parity; these
small ordered differences do not demonstrate a CPU-thread optimization.

## Audio and word checks

All **1,680 API WAVs**, including warmups, were independently SHA-256 verified
and decoded locally: 24 kHz mono, finite, nonempty and above the recorded signal
thresholds. Eighteen additional profiler WAVs preserve nine unprofiled/profiled
pairs; float audio is identical in both B1/B8 instrumentation comparisons.

Single-server ordinary versus single-server MPS, sequential C1, returns
**60/60 complete WAV bodies byte-identically** under matched startup seed and
input order. That is useful preservation evidence for that control. Two servers
have independent random-number streams; requests are not seeded individually,
so the two-server comparison is not paired generation by generation.

An additional **offline cached Whisper large-v3-turbo** screen ran on the Apple
M5 Pro integrated GPU (25,769,803,776 bytes shared physical RAM), separately from
RTX 4070 timing, without API calls or model downloads. It screened 1,071 logical
outputs/655 unique waveform-language inputs. The remaining 189 logical outputs
are explicit unsupported-language skips. Arabic dialect routes use a documented
family-language hint; other unsupported languages have no substituted judge.
Expected text was never passed into recognition. Duplicate transcript reuse was
verified consistent by exact waveform SHA and language hint.

This new screen is **Whisper, not a new Omnilingual run**. The earlier two-Meta
15,058-input work remains in its original report and is not transferred to new
waveforms. No native listening/MOS or speaker-identity qualification was performed
in this capacity experiment.

At C8, the ordinary single-server arm has 22/51 normalized-exact Whisper matches
versus 20/51 for two MPS servers. Four baseline-exact cases lose exactness and
two gain it; CER worsens on 14 pairs and improves on 13. ASR uncertainty and
different random streams prevent assigning these changes to MPS. The English
same-language-reference output has a word-check flag in the two-MPS arm.
Before/after samples and all transcripts are retained for review. These rates
are recognition screens, not measured spoken-word failure rates, and do not
establish quality-preserving production capacity.

## The older “25 concurrent” number

The closest synthesis documentation found is
[throughput_analysis.md](throughput_analysis.md): it reports **25 concurrent
requests in 2.69 seconds, approximately 9.29 requests/s**, using **voice-design
mode on an RTX 5070**, dated April 19, 2026. It is not a 25 cloned requests/s
measurement. That historical hardware/source is reported by the older document;
its physical VRAM/UUID are not reverified by this RTX 4070 experiment.

Lingua's separate `docs/chat-load-testing.md` reports **28.16 requests/s** for
the EC2 text-chat backend with real short GPT responses and explicitly excludes
GPU speech workers. Another earlier RTX 4070 experiment *offered* 25 requests/s
but returned only 73/1,500 valid audio responses, with 1,426 rejections and one
connection error. Offered load, backend throughput and complete cloned-voice
synthesis throughput are different measurements. No inspected Markdown evidence
established 25 fresh full cloned-voice completions/s on one RTX 4070.

## Practical next steps

Keep batch limit 1 as the measured low-latency candidate for this workload.
Treat two-process MPS serving as a **modest capacity candidate requiring longer
randomized trials and quality review**, not a default production rollout.
The requested p95 under 500 ms still needs work, including at low load.

The large optimization target is the repeated full-sequence/guidance work, plus
pointwise/copy kernels. Prior split-guidance/reference-shortening/fewer-step
experiments change numerical or speech behavior and remain experimental; their
past quality concerns are not cleared by these hardware measurements. Prepared
reference reuse is useful, but naive KV reuse is not lossless for this bidirectional
generation. Larger VRAM alone cannot supply the missing matrix throughput.

For an unchanged workload at 25 requests/s, 2.55/s implies roughly ten ideal
4070 workers just to match aggregate offered work, before queue/SLO/failover
headroom. That arithmetic is a saturation floor, not a qualified fleet plan.
No production worker, Lingua gateway, MuseTalk flow or PR was deployed/merged.

## Evidence and reproduction

[Evidence bundle and scripts](../benchmark/rtx4070-capacity-2026-11-04/README.md)
includes the raw requests, stage headers, all phase summaries, GPU/CPU telemetry,
both actual MPS-client receipts, source/model hashes, Chrome traces, failed
counter/startup attempts, word transcripts, samples and cleanup proof. Large
metadata streams use lossless gzip with source/saved SHA-256 provenance.

All full audio remains in the local task archive:
`/Users/ahmadsmacair/code/lingua/lingua/work/omnivoice-4070-rps-2026-11-04/retained`.
Before deletion, all **2,025 backup files/376,782,346 bytes** were independently
matched to remote sizes and SHA-256. Source, environments and checkpoint provenance
are explicit; model weights, conditioning clips and private keys are not published.

The original startup harness mistakenly polled the lightweight `/healthz` payload
for the detailed `/health` status. It was stopped before generation and retained
as `startup-healthz-attempt`; the corrected run uses `/health`. Raw linear shape
metadata uses a two-byte estimate even for nine FP32 codec projections; derived
dtype-corrected files repair that estimate. FP16-only FLOP conclusions and all
timed HTTP/GPU outputs are unaffected. The audit also parses NVIDIA-SMI's unit
suffixes explicitly. Original raw evidence was not overwritten by these corrections.
