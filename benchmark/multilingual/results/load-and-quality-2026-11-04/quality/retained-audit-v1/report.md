# Staggered call-load assay — RTX 4070 Ti

All four frozen phases completed on 2026-11-04, controller 15:11:32.666–15:16:03.522UTC. The measured profile does not meet p95 under 500 ms: even 0.5 RPS returned successful-request p95 596.9 ms. At higher offered rates, queue admission rejects most offers; those failures remain in every all-offer SLO denominator. These are simulated call populations assuming one TTS turn per 20 seconds, not simultaneous active inference streams.

| Phase / assumed calls | Offers | HTTP200 + valid signal | HTTP429 | Successful arrival→full-body p95(ms) | Valid arrival≤500ms / all offers | Valid arrival≤2s / all offers |
|---|---:|---:|---:|---:|---:|---:|
| calls10-poisson-0_5rps | 30 | 30 | 0 | 596.9 | 27 | 30 |
| calls100-poisson-5rps | 300 | 183 | 117 | 1800.4 | 6 | 183 |
| calls1000-poisson-50rps | 3,000 | 164 | 2,836 | 1971.0 | 1 | 159 |
| calls5000-poisson-250rps | 15,000 | 146 | 14,854 | 2219.0 | 0 | 7 |

Across 18,330 cohort offers:523 valid complete WAVs,17,807 rejections, zero transport errors or missing outcomes. Ten disjoint warmups are separate. All 533 WAVs / 90,994,972 bytes were retained and independently SHA-checked, decoded as 24 kHz mono, and checked for finite samples/RMS/peak. This audit establishes signal validity only; word and voice quality await the two blind Meta ASRs. No human or native-speaker certification is implied.

Hardware/source: NVIDIA GeForce RTX 4070 Ti, 12,282 MiB physical / 12,481,134,592 bytes Torch-visible VRAM, driver 610.57.04, AMD Ryzen 7 5700X host, exclusive GPU, no concurrent model/ASR. Torch 2.8.0+cu128/CUDA 12.8, Triton 3.4.0, Transformers 5.5.4, OmniVoice 0.1.4. Model revision c5fdb5ccb189668d56333f77ba2629f4cd7535f4, worker ce48b253e66b951d87ea42919f945828c62e96a4. This measurement is for a 4070 Ti, not an RTX4070.

Actual validated profile: Triton eager, FP16, full 16 steps, full original registered references, speed 1, postprocessing off, codec decode batch 1, KV off, SplitCFG off, graphs 0, complete-response cache 0, batch size 1, collect 0 ms, max pending 4. Startup seed 20261103 is not a guarantee of deterministic concurrent scheduling. Original 1,000 text inputs are cycled in larger phases; offers are not all novel texts. Original 9 Aoede language references and explicit English fallbacks retain their fixture provenance and ASR/script caveats.

Frozen V4 uses HTTPX 0.28.1 pool 256 rather than 5,000 preconnected sockets; this assay is staggered arrival load. Connection events and dispatch lateness are retained. Arrival→body latency is actual-dispatch→full-body latency plus dispatch lateness; rejected/missing/invalid outcomes never count as audio success. The p95 values above condition on successful HTTP 200; the last two columns keep all offers in the denominator. Sequential warmup timing is not part of the arrival SLO.

Evidence: local full run `retained-59071593/results/triton-eager-staggered-overload-calls-v1`; `summary.json` contains hardware and per-phase actual-dispatch and scheduled-arrival metrics; `verified-audio.jsonl` lists every verified body. The full retention archive has SHA256 22f0801363a898bec9c1b18f0836b5d5fbfb20f7c5077ac1d07b79e8ed0ebe75 (82,565,057 bytes). Fresh independent SMI was empty at 15:18:02.357868 UTC; explicit GPU/CPU/I/O release went to the final ASR owner. Optional cold 1k / hot 5k cache assay was skipped because its authorized latest start 15:12 UTC could not follow verified staggered retention. No cached result is substituted here.
