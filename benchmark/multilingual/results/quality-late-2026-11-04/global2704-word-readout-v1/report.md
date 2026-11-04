# All104 languages × voice/configuration word screen — 2704 exact fixture cases

TTS ran on NVIDIA GeForce RTX 4070 Ti on2026-11-04: physical12,282MiB (nvidia-smi), Torch-visible12,481,134,592bytes, driver610.57.04, CUDA12.8. Source377de39b, hybridFP16, graph2/KVoff/splitoff/codec1/cache0,16steps, sequential requests. No ASR/setup/transfers ran during TTS timing. LLM ASR ran later on this4070Ti; CTC ASR ran onRTX4070 (physical12,282MiB, visible12,450,070,528bytes, driver580.95.05/CUDA12.8). Both judges use BF16 batch1, blind text, no expected-text prompt. Fresh hardware evidence is retained in per-run and per-judge metadata. This is a CPU-only review of those results, not new inference.

Exact fixture join:2704/2704 unique IDs,104routes×26cells, no missing/duplicate cases. The25 original grid arms were selected before scoring; transport/cache/load,28validation checks, step/references and repeat diagnostics are excluded. All2704 returnedHTTP200; 2700 have readable nonempty signals and both ASR judges. Four empty WAVs remain failures. All5400 judge metric rows were independently recomputed and body identities agree with the prior full WAV audit. Repeated ASR evidence is not an independent observation.

| RTX4070Ti tested preset/mode | Offered | Nonempty | Both normalized exact | Both CER0 | LLM exact | CTC exact |
|---|---:|---:|---:|---:|---:|---:|
| lingua-default | 832 | 828 | 99 | 109 | 230 | 114 |
| lingua-tutor | 832 | 832 | 141 | 165 | 316 | 159 |
| lingua-video | 832 | 832 | 143 | 170 | 322 | 162 |
| worker-auto-not-current-Lingua-UI | 104 | 104 | 39 | 43 | 64 | 39 |
| worker-design-not-current-Lingua-UI | 104 | 104 | 41 | 48 | 60 | 43 |

Overall463/2704 offers match both judges after normalization;535 have zero character error in both (spacing differences explain part of the gap). Among2700 judged signals, at least one judge has character errors in2165. These are **word-screen mismatches**, not2165 proven bad utterances: recognizer script/spelling/spacing and pronunciation/word errors are not separated without listening/native review. Conversely, exact ASR does not certify voice identity or natural speech. The grid does not justify global quality approval.

The original Lingua-default preset uses speed1.3/full postprocessing; video uses speed1/off; tutor uses speed1/light plus pause insertion. Their differences change several controls together. Auto/design are worker-only reference-free modes, not current UI cards. No matched earlier2704-case run exists here, so a **new optimization regression count is unavailable**. This reports behavior under V6, not a causal before/after comparison or a claim that the default OmniVoice repository performs the same way.

| Reference profile (RTX4070Ti TTS) | Offered | Nonempty | Both exact | Both CER0 |
|---|---:|---:|---:|---:|
| cross_language_english_reference | 2304 | 2301 | 253 | 314 |
| local_target_language_reference | 192 | 191 | 130 | 130 |
| worker_auto_no_reference | 104 | 104 | 39 | 43 |
| worker_design_no_reference | 104 | 104 | 41 | 48 |

Of64 actual reference controls, 33 match both judges and 31 require transcript/ASR review. All eight full English reference transcripts match both judges, but that calibrates those reference texts only. It does not validate English-reference synthesis in95other languages. Native/fallback rows span different languages and voices; their rates are not a causal reference-language comparison. Exact-reference and target-language calibration remain separate in every case record.

The four empty outputs are Somali/Aoede/default, Turkmen/Aoede/default, Turkmen/Charon/default and Finnish/Iapetus/default. Three additional nonempty weak signals (Arabic/Zephyr/default and Sudanese-Arabic-route/Zephyr/video+tutor) remain separately flagged; they are not repaired, excluded or quality-approved by the word screen.

47 routes have no both-exact cell for their one tested short text. All26 cells match both judges only for de, en, pl. This is neither a complete language certification nor proof a zero-exact route is unsupported. Twi's26cells use the broader Akan ASR proxy; MinNan romanization and uncalibrated target languages require explicit native/recognizer review.

`per-voice-configuration26.json` contains all24 stock-voice/preset combinations plus auto/design; `per-route104.{json,csv}` documents every language. `cases2704.jsonl` retains the exact request/run/body identity, effective controls, both transcripts, scores and calibration/proxy flags. `invalid-audio4.json`, `weak-signal3.json` and `reference-controls64.json` preserve failures and calibration evidence. No MOS, accent, gender, speaker identity, native-quality or default promotion is inferred.
