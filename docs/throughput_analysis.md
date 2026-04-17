# Throughput Analysis

## Goal

Target: support **100 concurrent requests** with **3-5 seconds wall time max** while keeping:

- `num_step=16`
- high-quality audio as a hard constraint
- `hybrid` as the primary serving path unless a faster path clearly preserves quality

This document is a repo-specific optimization plan based on:

- the current FastAPI serving path in `src/omnivoice_triton/cli/api_server.py`
- the batcher in `src/omnivoice_triton/serving/batching.py`
- the runner stack in `src/omnivoice_triton/models/`
- upstream OmniVoice generation behavior
- stored load-test artifacts
- live load tests on an RTX 3090 on 2026-04-17

## Reality Check

Current behavior does **not** meet the target yet.

Observed live results on RTX 3090 with warmed `hybrid` runner:

- `clone`, `num_step=16`, 100 concurrent requests: about **30.9s** wall
- `design`, `num_step=16`, 100 concurrent requests: about **12.1s** wall
- `design`, `num_step=8`, 100 concurrent requests: about **5.7s** wall

Stored repo results are better than the live 3090 run, but still not within target for clone mode:

- `results_100_clone_step16.csv`: about **16.0s** wall for 100 clone requests

Implication:

- **`design` / no-reference mode** may be able to approach the goal with strong optimization work.
- **`clone` mode at `num_step=16`** likely needs more than small tuning. It will require major serving improvements and may still need stronger hardware or multi-replica scheduling to consistently hit 3-5 seconds.

## Main Bottlenecks

### 1. Single in-flight merged batch

The server still executes only **one `model.generate(...)` call at a time**.

Impact:

- 100 concurrent requests drain in sequential waves.
- One large batch blocks every later request.
- Tail latency is dominated by queueing, not just model speed.

### 2. Batch-size instability hurts CUDA Graph reuse

CUDA Graph capture is keyed by exact input shape.

Impact:

- burst traffic produces multiple graph captures for different batch sizes
- cold-path graph capture appears during live traffic
- shape churn adds avoidable overhead

### 3. Clone prompt preparation is still expensive

Clone requests still do meaningful CPU work before they reach the batcher:

- audio decode from upload bytes
- silence trimming
- resampling
- optional ASR
- audio tokenizer encode

The current clone-prompt cache stores tokens on CPU and the generation path copies them back to GPU per request.

### 4. Duplicate work before generation

The API estimates batching cost by tokenizing style/text in the server, then upstream OmniVoice tokenizes again during input preparation.

Impact:

- extra CPU overhead for every request
- extra latency before requests even enter merged execution

### 5. Decode and post-process run after token generation

After token generation, each item still pays:

- audio-token decode
- silence removal
- fade/pad work
- WAV serialization

This is especially costly in large response bursts.

### 6. Default `hybrid` path is not the maximum-speed variant

`triton` and `hybrid` patch only layers `0..23` by default and leave the last 4 layers unpatched for quality safety.

That is a sensible default, but it means there is still some model-side speed headroom to measure.

## Strategy

The fastest path to the goal is not "make batches bigger".

On the RTX 3090, increasing `max_batch_requests` from 32 to 64 made throughput worse because larger merged batches increased `batch_exec_ms` more than they improved utilization.

So the right plan is:

1. stabilize batch shapes
2. remove per-request CPU overhead before batching
3. reduce post-generation overhead
4. benchmark more aggressive model-side optimizations while keeping `num_step=16`
5. only then decide whether a single model replica can meet the target

## Optimization Phases

## Phase 0: Lock Baseline And Instrumentation

Goal: make throughput work measurable and repeatable.

Changes:

- keep `scripts/load_test_api.py` as the standard load harness
- define standard scenarios:
  - `design`, `num_step=16`, 100 requests, concurrency 100
  - `clone`, `num_step=16`, 100 requests, concurrency 100
- capture:
  - wall time
  - p50/p95 latency
  - queue wait
  - batch execution time
  - batch-size distribution
  - graph-capture count by shape
- log separate timings for:
  - request preprocessing
  - prompt-cache hit or miss
  - merged `model.generate(...)`
  - decode/postprocess
  - WAV serialization

Why first:

- we need clean before/after data for each phase
- current headers mostly show batch timing, but not enough sub-stage breakdown

Success criteria:

- every load test run produces a clear stage-level timing breakdown
- graph-capture churn is visible in logs or metrics

## Phase 1: Stabilize Batch Shapes For CUDA Graph Reuse

Goal: stop paying graph-capture and shape-churn penalties during burst traffic.

Changes:

- introduce **batch buckets** for short-lane traffic, for example:
  - `8`
  - `16`
  - `32`
- prefer filling the nearest bucket instead of emitting arbitrary sizes like `18`, `22`, `39`, `40`
- optionally hold requests slightly longer when a batch is near the next bucket boundary
- prewarm graph shapes at startup for the common buckets and common sequence-length bands
- consider separate lanes for:
  - `short_noref`
  - `short_ref`
  - `medium_ref`
  instead of only short/long

Expected impact:

- fewer first-use graph captures during live bursts
- steadier `batch_exec_ms`
- better p95 and p99 latency

Priority:

- **Very high**

Success criteria:

- common burst runs only use a small, fixed set of graph shapes
- graph capture happens at startup or warmup, not mid-run

## Phase 2: Add A Fast Clone Prompt Path

Goal: remove repeated clone prompt prep from the request hot path.

Changes:

- add a **prompt registration** workflow:
  - `POST /clone-prompts`
  - return a `prompt_id`
  - `POST /generate` can use `prompt_id` instead of raw `ref_audio`
- keep registered prompts in:
  - GPU memory if capacity allows, or
  - pinned CPU memory if not
- avoid re-decoding and re-tokenizing the same reference audio for repeated callers
- keep the current upload-based flow as fallback, but make `prompt_id` the throughput path
- require `ref_text` for throughput-sensitive clone serving so ASR never lands on the hot path

Expected impact:

- major reduction in clone-mode CPU overhead
- less variance before requests enter the batcher
- less GPU upload churn for cached prompts

Priority:

- **Very high**

Success criteria:

- repeated clone traffic can avoid raw upload decode and prompt creation entirely
- prompt-cache hits show materially lower pre-batch latency

## Phase 3: Remove Duplicate Server-Side Tokenization

Goal: stop doing token-count estimation work twice.

Changes:

- avoid full tokenizer calls in the server just to estimate conditioning size
- either:
  - cache estimated token counts alongside prompt data, or
  - move the estimate into a lighter-weight heuristic, or
  - have upstream-prepared metadata returned with the prompt object
- keep batching heuristics approximate if needed; they do not need exact token counts to be useful

Expected impact:

- smaller but consistent CPU savings
- lower request admission latency

Priority:

- **High**

Success criteria:

- design and clone requests do not fully tokenize text twice before generation

## Phase 4: Reduce Decode And Postprocess Cost

Goal: keep the GPU focused on generation and reduce serial tail work.

Changes:

- add a throughput-oriented output mode:
  - `postprocess_output=light`
  - lighter silence handling or skip silence removal
- move WAV serialization off the critical path where possible
- benchmark whether decode/postprocess should run in a small worker pool after `model.generate(...)`
- if quality remains acceptable, reduce or disable expensive silence stripping for high-throughput server mode

Notes:

- this phase must preserve audible quality
- the goal is not to change the voice, only to trim avoidable post-work

Expected impact:

- better tail latency
- better burst completion time for large batches

Priority:

- **High**

Success criteria:

- decode/postprocess is no longer a large share of total per-response time

## Phase 5: Benchmark The Fastest Safe Model Variant At `num_step=16`

Goal: squeeze more speed out of the model path without dropping below your quality bar.

Changes to benchmark:

- `patch_range=None` for full-layer Triton patching
- `hybrid + SageAttention` on Ampere
- `bf16` versus `fp16` if numerically stable on the target GPU
- prewarmed graph shapes for the dominant serving buckets

Important:

- this phase is benchmark-driven, not assumption-driven
- full-layer patching may help throughput, but it needs quality verification

Priority:

- **High**

Success criteria:

- pick one serving preset that is fastest while preserving acceptable audio quality at `num_step=16`

## Phase 6: Smarter Scheduler Instead Of Bigger Batches

Goal: improve drain time without creating giant slow batches.

Changes:

- keep batch caps around the best local knee instead of pushing to 64+
- add scheduler logic that prefers:
  - stable buckets
  - same lane
  - same prompt class
  - low padding waste
- optionally let one nearly-full batch launch while a second batch continues collecting if enough queued work exists

Notes:

- this may require more than one worker thread or more than one model replica
- the current one-worker architecture is the structural wall

Priority:

- **Medium to high**

Success criteria:

- wall time improves without increasing p95 batch execution time

## Phase 7: Scale Beyond One In-Flight Generate

Goal: decide whether the 3-5 second target is possible on one GPU, and if not, add the minimum necessary parallelism.

Options:

- two model replicas on one large GPU if memory allows
- process-level sharding with separate queues
- multi-GPU serving when clone throughput is the real product target

Reality:

- if clone mode at `num_step=16` still sits well above target after Phases 1-6, **single in-process single-replica serving is the limit**, not just an implementation bug

Priority:

- **Conditional but likely necessary for clone mode**

Success criteria:

- sustained 100-request burst completion within 3-5 seconds in the actual target mode

## Best Order To Execute

Recommended order:

1. Phase 0
2. Phase 1
3. Phase 2
4. Phase 3
5. Phase 4
6. Phase 5
7. Phase 6
8. Phase 7

That order gives the best chance of gaining real throughput early without rewriting too much too soon.

## Recommended Product Modes

To keep `num_step=16` while maximizing throughput, define explicit serving presets.

### Preset A: High-Quality Design

- `mode=design`
- `num_step=16`
- prewarmed common graph buckets
- full request batching
- lightweight postprocess if quality is preserved

Target:

- strongest candidate for the 3-5 second goal on one strong GPU

### Preset B: High-Quality Clone

- `mode=clone`
- `num_step=16`
- `prompt_id` required for production throughput path
- `ref_text` required
- no ASR in hot path
- cached prompt tokens on GPU or pinned memory

Target:

- likely needs the full Phase 1-7 roadmap

## Acceptance Gates

Do not call the target met unless all are true:

- 100 concurrent requests finish in **<= 5s wall**
- p95 latency stays within the same bound class as wall target
- no on-demand ASR in the hot path
- no graph captures during warmed steady-state runs
- audio quality remains acceptable at `num_step=16`

## Bottom Line

The best optimization opportunities are not the Triton kernels themselves. The repo already got strong wins there.

The next big gains come from:

- **stable batch shapes**
- **pre-registered clone prompts**
- **less request-side CPU work**
- **less serial decode/postprocess work**
- **benchmarking the fastest safe `hybrid` variant at `num_step=16`**

For `design` mode, the 3-5 second goal looks aggressive but plausible with strong serving work.

For `clone` mode, the same goal is a stretch on the current single-replica architecture and may require either:

- a stronger GPU than the current 3090
- multiple replicas
- or both
