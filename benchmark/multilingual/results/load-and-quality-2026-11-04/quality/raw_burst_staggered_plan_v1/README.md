# Staggered calls / overload — CPU preparation, not a GPU result

**Apple M5 Pro CPU-only preparation, 4 November2026.** No TTS/ASR model or GPU was run for these plans. Planned inference hardware is **NVIDIA GeForce RTX4070Ti**, historical physical12282MiB / Torch-visible12481134592bytes, driver610.57.04 / CUDA12.8. Fresh hardware/idle evidence is mandatory at execution. GPU/VRAM values here are reused host context, not a new test. The active raw5k timing run is separate and unchanged.

Assume each active call offers one complete TTS turn every20seconds. This produces the following fixed-seed Poisson schedules through the unchanged V4 harness:

| Assumed calls | Offered RPS | Offers | Actual last planned offset | Routes offered |
|---:|---:|---:|---:|---:|
|10|0.5|30|64.526s|30|
|100|5|300|63.915s|104|
|1000|50|3000|59.436s|104|
|5000|250|15000|60.539s|104|

The original1,000 published inputs cycle when necessary; there are18,330 cohort offers and ten input-disjoint warmups. These are **offered requests**, not18,330 completed neural inferences or unique texts. The V4 algorithm places the first arrival at offset0, then uses exponential intervals with seed20261104; finite Poisson windows therefore differ slightly from60seconds. The queued-work limit is deliberately4, plus the active batch, so the higher rates measure overload and explicit rejection rather than hiding a large queue.

All stages use sourcece48b253e66b951d87ea42919f945828c62e96a4, Triton eager FP16, batch1, collect0, KVoff, full16steps, speed1, original full references, postoff, codec1, graphs0 and generated-result cache0. The unchanged V4 health validation checks the requested configuration, including pending limit4 and cache-disabled state. Nothing automatically adopts the experimental product RoPE/Split/bucket profiles.

The client is standard HTTPX with maximum256 pooled connections and client keepalive expiry600seconds. This is **not the raw5,000-preconnected-socket assay**. Product/Uvicorn transport defaults remain unchanged; reconnect events, actual dispatch offsets, planned offsets and dispatch lateness are retained from V4 traces. High lateness must be reported as a client/load-generator limitation rather than claimed as the configured offered rate.

`plan.json` contains four distinct measured phases plus ten warmups in one fresh server lifetime. `config.json` has the exact remote V4 command and source/input/helper hashes. `build.py` executes only whitelisted, hash-pinned synchronous V4 helper bodies to verify reference SHAs, effective uniqueness, cyclic expansion, route counts and schedule deadlines on CPU. `run.py` defaults to dry mode. Its future root lease must explicitly attest raw5k retention/idle, subsequent modern LLM-ASR completion/idle, CPU/I/O quiet, source/GPU identity, and exact controller/config SHAs. No such lease is supplied by this plan.

Future execution is bounded to360seconds overall, with a325second worker budget and75second measured-phase limits. The source model files are SHA-verified before launch. The controller observes its owned descendant process groups while the unchanged V4 worker runs, because V4 starts the server in a separate session. Cleanup kills only those owned groups and records fresh SMI idle. The existing task stop helper's exact source snapshot/SHA is retained as additional provenance. No existing V4/product code is edited.

The unchanged V4 policy stops after a phase with a non200/429 response or transport error. Partial runs and later unexecuted phases therefore remain explicit; there is no silent retry. The additional readout uses every configured offer as the denominator and separates HTTP200, signal-valid audio, valid audio within500ms, HTTP429, transport failure and missing outcomes. Rejections never count as completed-audio throughput. HTTP200-only p95 is labeled conditional; words/voice remain pending blind ASR and listening review. This assay estimates capacity under the stated turn-frequency assumption, not native speech quality or a supported-call guarantee.

Two clocks are retained separately: actual POST dispatch→complete body, and **scheduled arrival→complete body = `complete_ms + dispatch_lateness_ms`**. Both have explicitly labeled HTTP200-only p50/p95/p99; valid-audio counts within500ms and2seconds use all configured offers as their denominator. Missing timing, missing outcomes, HTTP rejections/errors, and invalid audio cannot pass those arrival SLOs. A CPU case with400ms dispatch lateness plus200ms body time passes the dispatch500ms check and fails the scheduled-arrival500ms check. The historical `HTTP200_only_p95_ms`/`valid_audio_within500ms_alloffers` keys retain their original actual-dispatch meaning.

CPU commands:

```sh
python3 load/raw_burst_staggered_plan_v1/build.py
python3 load/raw_burst_staggered_plan_v1/run.py
python3 load/raw_burst_staggered_plan_v1/test_cpu.py
```

No neural execution may start before the active raw5k run, the approved intervening RoPE gate, and the following modern ASR work release the GPU, followed by a new root lease.
