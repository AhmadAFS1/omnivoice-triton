# RTX 4070 Ti reference-prefix A/B — sequential complete-WAV results

Measured 2026-11-04 on NVIDIA GeForce RTX 4070 Ti: physical 12,282 MiB (`nvidia-smi`), Torch-visible 12,481,134,592 bytes, driver 610.57.04, CUDA 12.8. The linked per-arm metadata records the fresh device evidence, package pins and idle preflight. This report is a local CPU-only audit of retained GPU results; no ASR or speaker model ran here.

All three arms used source `ce48b253e66b951d87ea42919f945828c62e96a4`, 16 steps, speed 1, postprocessing off, hybrid FP16, unused KV off, split CFG off, graph limit 8 / row-token budget 16,384, codec batch 1 and response cache disabled. Each fresh process reset startup seed 20261103. Telemetry and health probes ran; ASR, setup and transfers were absent during timing.

396/396 offers returned HTTP 200 and readable, finite, nonempty audio. The independent protocol verifiers passed and no fatal CUDA/context fault was logged. All 396 generated WAV hashes and decoded duration/RMS/peak match the retained manifests; all 80 reference WAV hashes and durations were verified locally.

132 paired cases per arm cover all 104 Aoede language routes plus all 8 voices on Arabic, English, Spanish and Hindi. 110 cases replace an English reference; 22 keep their native-language reference unchanged. Prefix10/prefix17 mean the first 10/17 English words, with matching shortened transcripts, not 10/17 seconds.

| RTX 4070 Ti cohort | Arm | Requests | Complete-WAV p50 / p95 ms | Within 500 ms | Median output seconds |
|---|---|---:|---:|---:|---:|
| changed_English110 | full | 110 | 481.0 / 573.8 | 73 / 110 | 2.74 |
| changed_English110 | prefix10 | 110 | 322.3 / 382.5 | 110 / 110 | 2.62 |
| changed_English110 | prefix17 | 110 | 378.3 / 404.9 | 110 / 110 | 2.58 |
| unchanged_native22 | full | 22 | 518.5 / 596.1 | 11 / 22 | 2.82 |
| unchanged_native22 | prefix10 | 22 | 515.4 / 590.0 | 11 / 22 | 2.82 |
| unchanged_native22 | prefix17 | 22 | 523.1 / 583.4 | 10 / 22 | 2.82 |
| Aoede_all104 | full | 104 | 479.4 / 571.9 | 75 / 104 | 2.68 |
| Aoede_all104 | prefix10 | 104 | 323.0 / 450.7 | 102 / 104 | 2.60 |
| Aoede_all104 | prefix17 | 104 | 378.8 / 453.4 | 102 / 104 | 2.56 |

Paired effects (each case compared with the same full-reference case):

| RTX 4070 Ti cohort | Variant | Median latency ratio | Median reference-token delta (actual) | Median target-token delta (header estimate) | Identical WAV bodies |
|---|---|---:|---:|---:|---:|
| changed_English110 | prefix10 | 0.656 | -249 | -3 | 0 / 110 |
| unchanged_native22 | prefix10 | 0.999 | 0 | 0 | 22 / 22 |
| changed_English110 | prefix17 | 0.761 | -184 | -4 | 0 / 110 |
| unchanged_native22 | prefix17 | 1.003 | 0 | 0 | 22 / 22 |

The changed-reference latency reduction is accompanied by different generated audio and changed duration/target estimates. It cannot be described as mathematically equivalent inference. All 22 unchanged-reference controls remain byte-identical in both prefix arms; their latency ratios near 1 provide an internal timing control.

Registration returns actual reference audio-token counts. Admission headers are estimates for target/conditioning/sequence length and must not be presented as exact sampler tensor lengths. New graph captures are separately traced from server logs; cache-hit shapes are left unobserved rather than inferred. Per-request latency includes on-demand graph captures and full HTTP body receipt, and excludes worker startup/reference registration. The cohorts were run in one fixed order without repeat/randomized trials.
The pinned worker appends a terminal ASCII period to the unpunctuated 10-word transcript and to the Urdu reference. Forwarded and prepared transcripts are both retained in the paired artifact; no other transcript transformation was observed.

Reference lexical checks passed in the two Meta judges and cached Mac Whisper before this run. Generated speech ASR, speaker similarity and perceptual review remain pending. Nonempty audio and a lower latency do not certify words or voice identity. No default promotion and no concurrent-capacity claim follow from this experiment.

Artifacts: `summary.json` contains source/hardware/pins and every denominator; `paired132.jsonl` exposes absolute local WAV/reference paths, exact texts, SHA hashes and paired deltas for downstream ASR/speaker checks. `verified-audio396.jsonl` and `reference-audit80.json` retain the CPU audit evidence. Original run files remain unchanged.
