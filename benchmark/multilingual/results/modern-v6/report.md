# NVIDIA GeForce RTX 4070 Ti voice/control audit

NVIDIA GeForce RTX 4070 Ti; physical 12,282 MiB via nvidia-smi; visible 12,481,134,592 bytes; driver 610.57.04; CUDA 12.8; per-run full evidence in per-run.json

CPU-only audit on Apple M5 Pro. All 3,154 retained generation WAVs were independently rehashed and decoded; signal/duration/peak/RMS matched the original harness records. No new model or API calls were made.

The grid of 2,704 cases covers 104 routes × eight voices × three presets (2,496 clone cases), plus 104 auto and 104 design cases. All 2,704 returned HTTP 200; 2,700 passed the weak signal check. Four default-preset outputs were zero-frame, 44-byte WAVs. This diagnostic expansion preserved the original failed gate of 404 cases and does not promote any configuration's word/voice quality.

| Configuration | Cases | Valid signal | Sequential HTTP200 p50 / p95 ms |
|---|---:|---:|---:|
| lingua-default | 832 | 828 | 491.6 / 581.9 |
| lingua-tutor | 832 | 832 | 536.5 / 587.0 |
| lingua-video | 832 | 832 | 536.0 / 587.0 |
| worker-auto-not-current-Lingua-UI | 104 | 104 | 257.4 / 301.1 |
| worker-design-not-current-Lingua-UI | 104 | 104 | 259.8 / 303.4 |

Four grid failures: Somali/Aoede, Turkmen/Aoede, Turkmen/Charon and Finnish/Iapetus; all speed 1.3/postfull. The controlled Turkmen repeat returned a nearly silent 1.6-second WAV with postoff and an empty WAV with postfull at matched startup seed. The off result barely passed the weak signal floor and is not word-quality-cleared. Somali's four fresh-start repeats were valid-signal, so the original long-run RNG position was not reproduced.

The 130-case matrix returned 130 HTTP 200 responses and 129 valid signals; the Russian 64-step/CFG 0/speed 0.5/denoise false/preprocess false/postfull case was empty. All 28 validation requests correctly returned 400 and stay outside generation/ASR quality denominators.

| Steps | Requests / valid signal | Sequential p50 / p95 ms | Under 500 ms |
|---|---:|---:|---:|
| 16 | 104 / 104 | 479.4 / 572.0 | 73/104 |
| 8 | 104 / 104 | 331.2 / 386.2 | 103/104 |
| 4 | 104 / 104 | 259.0 / 290.0 | 103/104 |

All step-count results require ASR/native review before promotion; these sequential latency values do not measure 1,000 simultaneous novel requests. Dataset/source/proxy/reference caveats remain unchanged. `verified-audio.jsonl` lists every retained hash/path, `per-route-coverage.json` checks all 26 cells per route, and `invalid-audio.json` preserves all six observed invalid WAV records across grid/matrix/repeats.
