# Fixed-duration reference shortening: word and speaker review

The shorter-reference variants improve sequential latency but remain experimental.
Holding output duration fixed removes one confound; it does not preserve every
word or the relative speaker scores. Neither variant is a default recommendation.

Each arm has 132 full generated WAVs: all 104 Lingua routes in Aoede, plus all
eight voices for Arabic, English, Spanish and Hindi. English conditioning changes
in 110 cases. The other 22 native-reference cases per arm remain byte-identical.
All 396 WAVs were independently hashed and decoded. Target frame counts and
output durations match the original full-reference arm exactly in every case.

| English-conditioning cases, n=110 | Full reference | Ten words | Seventeen words |
|---|---:|---:|---:|
| Sequential full-WAV p95 | 570.6 ms | 385.0 ms | 418.2 ms |
| Complete in 500 ms | 72 | 110 | 110 |
| Both Meta ASRs match normalized script exactly | 21 | 39 | 32 |
| Lose a baseline both-ASR exact match | — | 3 | 4 |
| Gain a both-ASR exact match | — | 21 | 15 |
| Both judges' CER worsen | — | 5 | 7 |
| Lose ECAPA own-voice top1 rank | — | 3 | 1 |
| Gain ECAPA own-voice top1 rank | — | 2 | 2 |
| Median own-voice cosine change | — | -0.0572 | -0.0269 |

TTS used the dedicated RTX 4070 Ti, physical 12,282 MiB, visible 12,481,134,592
bytes, driver 610.57.04/CUDA 12.8; source ce48b253, hybrid graph8/M16384,
FP16/KV off/Split off, 16 steps, speed1, postprocessing off, serial codec1 and
generated-response cache off. Torch2.8.0+cu128, Triton3.4.0,
Transformers5.5.4 and OmniVoice0.1.4 were pinned. This is one sequential screen,
not concurrent capacity or a randomized repeated latency trial.

The original full arm is byte-identical to the prior full-reference run. The
fixed-length follow-up uses `duration=(T+0.25)/25` to reproduce the original
integer target token count without floating-point flooring errors. At 24 kHz,
the pinned codec produces 960 samples per target frame plus five padding frames.
The raw per-case frames, options and SHAs remain in the paired latency audit.

Speaker extraction used SpeechBrain ECAPA-TDNN on Apple M5 Pro CPU, CUDA/Metal
disabled, no inference API. The independent audit uses scalar dot products and
explicit norms, verifies all 376 source embeddings/audio hashes, recomputes
every eight-voice comparison, and strictly joins all 396 Meta results by request,
audio SHA and expected script. Speaker changes and word flags are separate.
`flagged-review.json` preserves every lost-top1, lost-exact or both-CER-worse pair.

Meta LLM and CTC use their existing frozen protocols, respectively
`modern59071593-batch1-v1-7f8784ef9d826da07f57` and `legacy-batch1`.
They can disagree or mistranscribe. ECAPA was not calibrated for these synthetic,
multilingual voices. These screens do not certify native fluency, accent, MOS or
speaker identity. Human review remains needed before selectively enabling a crop.

An additional cached Whisper large-v3-turbo check on Apple Metal examined a
deliberately selected diagnostic subset. It supports neither a population failure
rate nor coverage of every route. Its fixed-duration Arabic-Sudanese crop results
match the script, whereas the earlier variable-duration crops did not; this is
evidence that output length and judge disagreement matter, not a universal crop
quality clearance. See the separate Whisper records and skip list.
