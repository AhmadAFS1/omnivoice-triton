# RTX4070Ti raw5k — complete bodies passed, latency target missed

**NVIDIA GeForce RTX4070Ti, physical12282MiB / Torch-visible12481134592bytes**, driver610.57.04, CUDA12.8; real GPU inference on4 November2026. Fresh preflight and post-run SMI found no other compute processes. No ASR shared the GPU during measurement. Hardware evidence is in the retained run's `metadata.json`; independent hashing/decoding here ran afterward on Apple M5 Pro CPU with no model inference.

All5,000 requests returned complete HTTP200 WAVs with basic signal validity and no protocol/transport failure. **The complete burst took1857.899seconds (30minutes57.899seconds).** This demonstrates completion under queued concurrency, and clearly misses the requested low-latency service target. No cohort response finished within500ms or2seconds. Words, pronunciation and voice fidelity are still unscored.

| Metric | Observed |
|---|---:|
| Offered / complete HTTP200 / signal-valid |5000 /5000 /5000|
| HTTP, transport, protocol and basic-signal failures |0|
| Distinct preconnected sockets / local endpoints |5000 /5000|
| Actual POST-write dispatch spread |77.099ms|
| Full-body latency p50 |932.280s|
| Full-body latency p95 |1765.744s|
| Full-body latency p99 |1840.017s|
| Last complete body from common barrier |1857.899s|
| Signal-valid bodies within500ms /2s, all offers |0 /0|
| Successful full WAVs per whole-burst second |about2.69|
| Retained cohort body bytes |900,536,800|
| Client peak RSS at network completion |1,049,817,088bytes|

This was5,000 fresh neural requests **cycling the same original1,000 text/language/reference inputs five times**, covering104 product routes. It is not5,000 novel inputs. The source was ce48b253e66b951d87ea42919f945828c62e96a4, OmniVoice0.1.4, Torch2.8.0+cu128 / Triton3.4.0 / Transformers5.5.4. The fixed profile was Triton eager FP16, batch1, collection delay0, full16steps, KVoff, full original references, speed1, postoff, codec1 and generated-response cache0. Collection delay0 differs from earlier measured B1 at5ms; source/math were otherwise held to this declared profile. Uvicorn0.44.0 used asyncio selector+h11, with a benchmark-only600second keepalive override.

All9 references were registered and10 input-disjoint warmups finished before the measured cohort. The5,000-member successful batch-counter difference and independent server-log Executing/Completed member totals both equal5,000; cache-disabled health was checked before and after. These are separately labeled benchmark observers. The legacy per-request `Inference-Executed` header remains unreported for all5,000 responses; no such header was invented.

The remote assay/retention completed and released the GPU at14:21:13UTC. The full986,000,960byte result tree was then copied locally. An independent audit verified all5,010 WAV hashes against both request and quality manifests, decoded them as24kHz mono, checked finiteness/RMS/peak, confirmed1,000 case IDs repeated exactly5times in the cohort, and found no signal mismatch. The5,010 WAVs including warmups total902,127,000bytes. Fresh independent SMI was empty at14:25:07 and again at the14:26:08 retained/idle/I/O-quiet handoff to the next approved experiment.

All raw WAVs, complete request rows, source/input/reference provenance, headers, runtime validation, health probes and logs remain under `retained-59071593/results/triton-eager-raw5000-cyclic1000-v1`. This directory contains the standard quality/ASR-ready manifests for a separate blind word-quality screen. `verified-audio5010.jsonl`, `per-route104.json` and `summary.json` contain the CPU audit. Generated-audio cache/coalescing throughput, native word quality and supported active-call capacity are separate questions and are not inferred from this completion result.
