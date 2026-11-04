# Independent relative speaker screen: reference-prefix396

Speaker extraction used Apple M5 Pro CPU, SpeechBrain 1.1.2 / Torch 2.8.0, no NVIDIA/Metal inference or paid API. The model is SpeechBrain ECAPA-TDNN `speechbrain/spkrec-ecapa-voxceleb` at `0f99f2d0ebe89ac095bcc5903c4dd8f72b367286` (Apache-2.0 model card). This independent audit ran scalar CPU arithmetic and hashes only.

The source TTS ran 2026-11-04 on NVIDIA GeForce RTX 4070 Ti, physical 12,282 MiB (`nvidia-smi`), Torch-visible 12,481,134,592 bytes, driver 610.57.04, CUDA 12.8; ce48b253,16 steps/speed1/postoff/KVoff/graph8/M16384/splitoff/codec1/cache0. Each arm has 132 sequential requests and a fresh startup seed. Telemetry and health probes ran; ASR/setup/transfers were absent during TTS timing.

Verified all 376 unique WAV hashes and 192-dimensional unit embeddings, all 7 pinned model assets and the original extraction script hash. Independently recomputed 3,296 own/impostor cosine scores (396 generated clips plus 16 cropped-reference controls, each against 8 full English stock references). Maximum root-score difference was 2.2e-16. All 44 unchanged-reference comparisons have exactly zero own/margin deltas and identical body SHA hashes.

| Changed-English-reference cases (RTX 4070 Ti TTS; CPU ECAPA) | n | Median own-cosine delta | Own top1 before→after | Lost / gained top1 |
|---|---:|---:|---:|---:|
| prefix10 | 110 | -0.0512 | 108→108 | 2 / 2 |
| prefix17 | 110 | -0.0169 | 108→109 | 1 / 2 |

| Speaker-score-selected A/B example | Own cosine before→after | Margin before→after | Nearest other voice after |
|---|---:|---:|---|
| hi/Fenrir prefix10 | 0.627→0.525 | +0.146→-0.015 | Iapetus |
| gu/Aoede prefix10 | 0.509→0.226 | +0.049→-0.072 | Zephyr |
| ha/Aoede prefix17 | 0.507→0.435 | +0.123→-0.006 | Despina |
| hi/Iapetus prefix10 | 0.690→0.351 | +0.369→+0.128 | Fenrir |
| bho/Aoede prefix17 | 0.766→0.484 | +0.251→+0.090 | Despina |

All 16 cropped reference controls retain their intended stock voice as top1. Nevertheless, the same-source crop/full cosine medians are 0.872 (10 words) and 0.934 (17 words). Content and clip length alone move this proxy; an absolute cosine cutoff would therefore be unjustified. Cropped controls are overlapping source audio, not independent regenerated voice-quality observations.

The 17-word version has a smaller median drop and fewer lost-top1 cases in this screen. Aggregate top1 can conceal individual rank flips: its net gain still includes the Hausa/Aoede lost-top1 case. The Hindi/Fenrir and Hausa/Aoede negative margins are small, so these rank flips particularly need listening rather than categorical labels. The Hindi/Iapetus 10-word case has the largest own-cosine drop while retaining top1, showing that rank alone also misses large relative changes.
Own-anchor similarity is only one view. The median paired own-minus-impostor margin changes by +0.0147 for 10 words and -0.0148 for 17 words; these margins do not uniformly favor the longer reference. Generated full-to-candidate cosine medians are 0.633 and 0.729 respectively. Together these are evidence of relative embedding changes, not a calibrated direction or size of perceived voice-quality change.

All generated languages are compared to English full-reference anchors. VoxCeleb training and synthetic, multilingual, short-clip speech differ; the screen has no calibrated probability, MOS, native speech, accent, gender or identity threshold. Use the selected paired WAVs for voice consistency A/B review and inspect matched ASR word results separately. Neither reference shortening is promoted by this audit.

ASR matching rows available at this audit: 396. `AB-review-manifest.json` is explicitly score-selected and retains exact paths, SHA hashes, text, reference identity, durations, speaker scores and any supplied ASR evidence. It cannot estimate population failure rate. `paired-recomputed264.jsonl` retains all comparisons; `summary.json` records source hardware and protocols.
