# Optional cold1k / hot5k — CPU-ready cached transport assay

**Apple M5 Pro CPU preparation, 4 November2026; no model/GPU execution.** Planned host: **NVIDIA GeForce RTX4070Ti**, historical physical12282MiB / Torch-visible12481134592bytes, driver610.57.04 / CUDA12.8. These are reused host labels; actual execution records fresh GPU/runtime/idle evidence. This optional assay must wait for core ASR and staggered-load completion plus a new root lease, and may be skipped for the deadline.

One native English full-reference key is tested. The disjoint warmup text is `en-load-flores-0307`, “He died in Osaka on Tuesday.” The target is `en-short`, “No one was inside the apartment.” Both use the unchanged full Aoede English reference. Reference registration and the one different-key warmup happen in a fresh server before the target is ever requested. Cache health must show exactly one entry before the cold target cohort.

| Cohort | Requests / distinct target keys | Required neural-member delta | Required response evidence |
|---|---:|---:|---|
|Cold identical |1000 /1|1|Exactly one `generated` / inference-true response; other responses `coalesced` or `cache`, inference-false|
|Hot identical |5000 /1|0|All `cache`, inference-false|

Every complete target WAV must be byte-identical across both cohorts. Actual per-response source/inference headers are retained, never fabricated. Aggregate benchmark counters provide independent neural-work evidence. Each cohort must have the configured number of complete HTTP200 responses, one body SHA, full signal validity, the required counter difference and truthful response-source headers. A failed cold gate stops rather than silently warming/retrying the key and calling a later result cold.

This reuses the exact approved raw parser/preconnection/cohort and owned-group cleanup core **b204405036ae236be8ceefc4f760271fcd13fc18428a86fded9110471ea94665**, wrapper **7a9087c32a7bca6f69afcab96818e83eb9d939250b4a2b80248021f10475bac6**, and frozen V4 input/retention/seed helpers. No existing file is edited. A scoped phase label changes only the imported core module's diagnostic label inside this serial assay; the original1000-cyclic/fresh labels are replaced with explicit one-key cache labels in retained rows.

The API profile is sourcece48b253e66b951d87ea42919f945828c62e96a4, Triton eager FP16, B1/collect0, KVoff, full16steps, speed1, postoff, codec1, graphs0 and an opt-in128MiB complete-response cache. No mathematical optimization changes are introduced. The benchmark-only server wrapper records keepalive600 and CPU member counters. Client/server descriptors are at least16384; every cohort proves1000 or5000 distinct reusable TCP sockets before its barrier, with no reconnect/retry. Standard HTTP Content-Length, duplicate/chunked/compression/truncation rejection,1GiB payload budget and separate peak-RSS evidence come from the reviewed raw core.

Bodies remain in RAM until each cohort ends, then all complete WAVs, float/signal checks, request rows, source headers, source/corpus/reference provenance and standard quality manifests are saved. Full-body dispatch p50/p95/p99, whole-cohort completion, actual dispatch spread and all-offer within2s counts remain separate. Hashing/decoding/file retention is outside network timing. Invalid and partial bodies are retained as failures. The warmup's metadata/error body is persisted before raising on an HTTP error.

The controller has a **120second overall bound** and35second cohort bounds; normal setup/two-cohort execution is expected to be much shorter, but is unmeasured. A fresh exact-controller/config/source/GPU root lease must explicitly confirm completed/retained/idle core ASR, completed/retained/idle staggered work, and CPU/I/O quiet. Model13-asset streaming SHA checks, source/ref/helper hashes and fresh GPU idle precede execution. The server inherits the owned controller group. All post-Popen metadata is inside cleanup guards, reused core cleanup handles a completed leader's surviving descendant, and errors kill the process instead of cleaning up a possibly broken CUDA context. Output directories cannot overwrite prior evidence.

**Cached/coalesced results are not fresh inference capacity.** Six thousand identical target responses provide at most one independent target-audio word observation. Existing blind ASR may be inherited only for that exact waveform SHA and language/script under a compatible declared judge protocol; an unchanged text alone is insufficient. No word/voice/native quality is certified here. The earlier shared-client cold1k6.789s/556new-TCP result used a different transport/profile; this assay records its own exact conditions rather than attributing an unmeasured speedup to one cause.

CPU commands:

```sh
python3 load/raw_burst_cached_plan_v1/run.py
python3 load/raw_burst_cached_plan_v1/test_cpu.py
```

Future execution additionally needs `--execute --lease FRESH_ROOT_LEASE` on the pinned baseline Python. No lease or GPU execution is granted by this document.
