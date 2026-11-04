# Multilingual complete-WAV load protocol

Target hardware: **NVIDIA GeForce RTX 4070**. GPU performance remains unverified
until each real run records actual GPU identity, physical VRAM (`nvidia-smi`),
PyTorch-visible VRAM, driver, CUDA/runtime, date and co-resident processes. Plan
creation and the synthetic HTTP smoke are **CPU-only** and execute no TTS/ASR
model inference.

The active product inventory contains 104 route codes. Inputs retain product
language, exact TTS API language, Omnilingual language/script ID, reference
language and native/cross-language voice provenance. The nine local Aoede
references cover nine target languages; 95 other routes use the explicitly
labeled English-reference fallback. Deployed production/S3 reference coverage is
not inferred from this local checkout. Translation/source-variety caveats and
licenses remain in each case. ASR support is not per-language quality proof.

## Run a fresh worker

Use the TTS virtualenv Python. The script launches only its own loopback server,
checks actual runtime configuration and `/languages` membership, and terminates
that process group when the finite run ends. Existing output directories are
rejected to prevent overwriting evidence.

```sh
cd /workspace/omnivoice-max-concurency
HF_HOME="$PWD/hf-cache" baseline/.venv/bin/python load/run.py \
  --source baseline --source-ref c79ce35 \
  --model "$PWD/model" \
  --model-revision c5fdb5ccb189668d56333f77ba2629f4cd7535f4 \
  --label before-all104 --output results/before-all104 \
  --corpus languages/corpus-smoke-104.json \
  --references languages/references/manifest.json \
  --plan load/plans/quality-short-all104.json \
  --startup-seed 20261103
```

No reference registration time is folded into warm generation latency. The
registration log remains separate. A `cold_upload` phase rejects reused reference
cache keys and previously exposed references; it is not a claim of cold physical
GPU startup. The model must already be ready before generation timing starts.

The main inference profile is hybrid FP16, 16 steps, speed 1.0, denoise on,
postprocessing off, a maximum batch of 8, token budgets 4096/4096, 16 pending
GPU generations, 4 pending reference tasks, 5 ms collection and 8 CUDA graphs.
These are declared experiment settings, not optimal settings for every language
or GPU. Every successful body is one complete WAV. No client streaming is used.

`--response-cache-max-bytes` and `--response-cache-ttl` are optional and omitted
for old baselines when unspecified. When supplied they must match
`health.response_cache.max_bytes/ttl_s`. Primary fresh before/after quality
comparisons use cache bytes 0; caching must not change the random-draw sequence
of a claimed matched comparison. `--startup-seed` applies a checked startup seed
before model import; the HTTP API still has no per-request seed and concurrent
scheduling is not deterministic.

Unknown experimental flags go through `--server-extra-flags-json`; their actual
health state can be checked through `--expected-health-json`, using dotted keys
for nested state. For example, known response-cache flags have dedicated options
and should not be smuggled through the generic extra list.

## Distinct workloads

`build_plans.py` writes separate finite JSON plans:

- `quality-all-languages.json`: every supplied short, medium and critical fixture,
  sequentially. The current published corpus has 299 cases: all 104 routes have
  short/medium texts, while 91 have the selected critical fixture. This is not
  native-listening certification.
- `bursts.json`: labeled 1/10/100 input bursts, followed by 1,000 repeated short
  requests. Repeated-content results must report cache/coalescing separately.
- `burst1000-unique-short.json`: 1,000 unique published effective API inputs,
  with disjoint warmup inputs. It uses `unique-load-corpus.json`, not the 299-case
  QA corpus. Spoken durations still vary; the workload is not 1,000 identical
  two-second utterances.
- `cold-identical1000.json`: a disjoint warmup followed by 1,000 requests for one
  previously ungenerated output key; intended to measure in-flight coalescing.
- `hot-identical1000.json`: generate one output first, then offer 1,000 copies of
  that input; intended to measure warm response reuse, not novel inference.
- `hot-multilingual1000.json`: warm all 104 short-language keys, then offer 1,000
  cyclic requests. Actual cache hits and evictions must be inspected.
- `staggered.json`: independently scheduled Poisson arrivals at 0.5, 5 and 50
  requests/second, with 30/300/3,000 offers. Each has a 60-second expected
  count/rate horizon. The finite schedule starts at zero and fixes the offer
  count; the random final arrival is not exactly 60 seconds. Each plan records
  exact seeded schedule statistics and offset hashes, while actual dispatch
  lateness remains in the request log. These represent illustrative
  10/100/1,000 active calls under one assistant response per 20 seconds.
- `staggered5000-optional.json`: 15,000 offers at 250 requests/second, under the
  same assumption. Keep the admission limit at 16 and include every rejection
  in failure accounting. This tests overload behavior, not successful call
  capacity.
- `hot-identical5000-optional.json`: optional warm response reuse, after the
  measured hot1,000 body size estimates retained bodies plus reserve below
  8 GiB. It needs 6,000 HTTP connections and cache waiters, with every successful
  WAV retained. It measures reuse, not fresh synthesis.

A `closed_loop` plan is also supported, but is labeled as self-throttled; it is
not used to infer latency under an independent arrival process. Setting a queue
limit above 1,000 admits long GPU waits and is an explicit diagnostic profile,
not an SLO improvement. Start with 1/10/100 requests; run 1,000 unique requests
with a bounded 1,200-second phase deadline only if the measured floor justifies
it. Root rental control provides the separate billing watchdog.

Generate the plans without model inference:

```sh
python load/build_plans.py --corpus languages/corpus.json \
  --load-pool languages/load-pool.json --output load/generated-plans
```

`orchestrate.py` prints commands by default. It never starts a GPU job unless
`--execute` is explicit. `--jobs` preserves the supplied order; `--skip-labels`
excludes manually completed or currently running jobs. The parent owns GPU
release and the rental's external billing watchdog. Generated plan version 3 is
independent of the running version 3 HTTP harness.

```sh
baseline/.venv/bin/python load/orchestrate.py \
  --root /workspace/omnivoice-max-concurency --candidate-ref FINAL_COMMIT \
  --skip-labels before-all299,candidate-all299 \
  --jobs before-short-curve,candidate-short-curve,before-unique1000,candidate-unique1000 \
  --plan-output core-suite.json
```

The candidate defaults to explicit unused-KV-cache suppression, checked against
actual health, with split CFG disabled. Experimental split CFG must provide its
flags and exact health expectations and stays a separate arm. Both use 16 steps,
8 CUDA graphs and codec decode batch size 1. Unique 1,000 bursts use 1,200 server
pending slots and 1,200 HTTP connections so the client pool does not silently
serialize offers. They preconnect with 1,000 health offers; actual TCP events
still determine connection creation. Warm-cache 1,000 tests use the same HTTP
connection limit. The independent-arrival staggered profiles keep 16 pending
requests, output caching off and 128 HTTP connections.

HTTP harness version 4 adds descriptor preflight after the rental's shell was
found to have a 1,024 soft file limit. It raises only its fresh process and new
server descendants to a sufficient soft limit (normally 16,384), preserves the
hard limit, and records both values plus the actual child process limits.
Insufficient hard limits fail before generation. Version 3 already deferred all
WAV retention until after network completion; version 4 preserves that timing,
inference parameters and request behavior. Existing running version 3 workers
remain unchanged; their small sequential quality cohorts need few sockets.

Planning reserves, rather than performance claims: about 5–12 minutes per
299-case quality arm, 5–20 minutes per unique1,000 arm (1,200-second phase cap),
5–10 minutes for the cache and basic burst profiles, and roughly 5–7 minutes per
three-phase staggered arm including drain and startup. ASR time depends on the
number of distinct retained waveforms; budget 20–45 minutes and calibrate from
its first completed records. The finite suite's soft budget stops between jobs;
it is not a hard billing cap. Prefer the core 1,000 target and quality gate before
optional 5,000 profiles. No estimated duration establishes 500 fresh responses/s.

## Metrics and retained evidence

The user target is **all 1,000 complete responses within 1–2 seconds**, requiring
at least 500 successful responses/second. A 429, transport error or unissued job
cannot satisfy it. Summary fields keep all offered/dispatched/HTTP200 counts,
response-source (`generated/coalesced/cache`) and `Inference-Executed` counts,
origin IDs, unique effective inputs and WAV hashes separate. All-burst success
uses completion time from the common barrier, not a denominator containing only
quickly rejected jobs.

Per-request timing records both dispatch lateness and full-body HTTP duration;
file saving, hashing and signal inspection are reported as later client processing
and excluded from API completion time. TCP creation trace events expose connection
setup. Preconnect request count is not asserted to equal live connection count.
Poisson arrivals are scheduled independently of server completions. Measurements
are loopback and exclude Lingua gateway, public network, MuseTalk and playback.

Every HTTP200 body has a unique path, SHA256, full parameters, language/source
provenance and signal checks. `quality-manifest.json` covers every successful
body. Mapped, readable audio enters `quality-asr-ready.json`; unsupported mapping
and unreadable audio remain explicit in `quality-unmapped.json` and
`quality-invalid-audio.json`. Nothing unscorable is silently marked correct.
A valid mapping can still be a proxy (for example Twi/Akan); case caveats remain
attached to the quality manifest.

Run both GPU Omnilingual variants only after all TTS timing releases the GPU.
Use expected words only for post-recognition scoring, preserve WER/CER, and flag
all disagreements/deletions/number-format issues. CTC and LLM variants share a
family; agreement is not independent ground truth. Whisper cross-checks and
native listening remain necessary for disputed words, names and dialects.

`verify.py` independently checks actual runtime configuration, planned request
counts, every saved WAV hash, ASR eligibility, and whole-burst target accounting.
The CPU synthetic smoke verifies reference reuse, all output retention, rejection
accounting, unmapped language visibility and runtime guards; it is not a GPU
performance or speech-quality result.
