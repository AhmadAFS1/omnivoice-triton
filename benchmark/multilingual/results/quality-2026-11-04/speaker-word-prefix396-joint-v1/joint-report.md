# Joint word and relative-speaker review: reference-prefix396

This is a CPU-only review of existing results (2026-11-04), with no new inference. TTS and the blind Meta LLM judge ran on NVIDIA GeForce RTX 4070 Ti: physical 12,282 MiB, Torch-visible 12,481,134,592 bytes, driver 610.57.04, CUDA12.8. The blind Meta CTC judge ran on RTX 4070: physical12,282MiB, visible12,450,070,528bytes, driver580.95.05/CUDA12.8. Physical/visible evidence is in each retained TTS/ASR metadata file. ECAPA used Apple M5 Pro CPU. TTS profile: ce48b253,16steps/speed1/postoff/KVoff/graph8/M16384/splitoff/codec1/cache0. ASR/TTS used separate leases; no ASR co-load entered TTS timings.

The join verifies all396 request rows by exact arm, HTTP ID, body SHA and script; 792 judge WER/CER/exact fields were recomputed from the retained blind transcripts. One declared protocol per model is retained: modern59071593-batch1-v1-7f8784ef9d826da07f57 for LLM and legacy-batch1 for CTC. No old primary LLM namespace was mixed in. All44 unchanged-native comparisons remain byte/word/speaker identical.

| Changed English references, n110 (RTX4070Ti TTS) | Both exact full→candidate | New exact losses | Both CER worse | Speaker top1 losses among word flags |
|---|---:|---:|---:|---:|
| prefix10 | 21→43 | 3 | 9 | 0 |
| prefix17 | 21→25 | 7 | 6 | 0 |

The ten-word exact losses are Sudanese-Arabic-route Aoede and Egyptian-Arabic-route Aoede (both judges gain character errors), plus Kinyarwanda Aoede (one LLM character change while CTC stays exact). The seventeen-word seven exact losses include **three spacing-only cases**—Korean, Kinyarwanda and Georgian—whose CER remains zero in both judges. Of the other four, Sudanese-Arabic-route Aoede changes in both judges; Norwegian, Estonian and Slovenian change only in CTC while LLM stays exact. These flags must not all be labeled spoken word loss.

The Sudanese-Arabic-route ten-word LLM output repeats an Internet-related string (CER10.417), while CTC nearly matches the expected short phrase (CER0.083). This is strong judge disagreement, not proof of repeated words in the actual audio. It dominates the macro LLM mean: full0.129→ten-word0.173; excluding only this explicitly retained outlier as a sensitivity check gives0.130→0.079. Both original and sensitivity denominators must remain visible.

Word and speaker flags are different axes. None of the22 word-selected comparisons loses speaker top1. Conversely, Hindi/Fenrir ten-word loses speaker top1 while both ASR judges improve to exact; Gujarati/Aoede ten-word retains the same transcripts/CER despite losing speaker top1; Hausa/Aoede seventeen-word retains the same CER in both judges despite losing speaker top1. A better ECAPA margin cannot clear word errors: Sudanese-Arabic-route seventeen-word has own/margin gains while both judges acquire substitutions.

All22 word-selected comparisons are listed below. E=baseline both-exact loss; C=both CER worsen. Deltas are candidate minus full. Full transcripts, WER/CER, paths/hashes, calibration/proxy caveats and speaker scores are in `word-review22.json`; the union with speaker-selected diagnostics is `joint-review-manifest.json`. Selection is diagnostic and cannot estimate a population failure rate.

| Variant / route | Flags | LLM / CTC CER delta | Own / margin cosine delta | Review reading |
|---|---|---:|---:|---|
| prefix17 / ko | E | +0.000 / +0.000 | +0.011 / -0.039 | spacing only |
| prefix17 / no | E | +0.000 / +0.043 | -0.250 / -0.188 | one judge |
| prefix10 / nan | C | +0.093 / +0.047 | -0.046 / +0.024 | both-judge/native review |
| prefix17 / nan | C | +0.233 / +0.279 | -0.036 / +0.012 | both-judge/native review |
| prefix10 / ar-SD | EC | +10.417 / +0.083 | -0.101 / -0.068 | both-judge/native review |
| prefix17 / ar-SD | EC | +0.250 / +0.250 | +0.102 / +0.161 | both-judge/native review |
| prefix10 / ln | C | +0.030 / +0.030 | +0.061 / +0.062 | both-judge/native review |
| prefix10 / ar-EG | EC | +0.174 / +0.043 | -0.024 / +0.019 | both-judge/native review |
| prefix10 / rw | E | +0.029 / +0.000 | +0.035 / +0.094 | one judge |
| prefix17 / rw | E | +0.000 / +0.000 | +0.050 / +0.070 | spacing only |
| prefix17 / ht | C | +0.042 / +0.250 | -0.262 / -0.083 | both-judge/native review |
| prefix10 / dag | C | +0.295 / +0.045 | -0.052 / +0.067 | both-judge/native review |
| prefix17 / dag | C | +0.023 / +0.045 | -0.056 / -0.043 | both-judge/native review |
| prefix17 / ny | C | +0.040 / +0.040 | -0.014 / -0.040 | both-judge/native review |
| prefix10 / af | C | +0.034 / +0.103 | -0.119 / -0.061 | both-judge/native review |
| prefix17 / et | E | +0.000 / +0.050 | -0.193 / -0.050 | one judge |
| prefix17 / sl | E | +0.000 / +0.042 | -0.039 / -0.075 | one judge |
| prefix10 / sq | C | +0.038 / +0.077 | -0.005 / +0.078 | both-judge/native review |
| prefix17 / ka | E | +0.000 / +0.000 | +0.052 / +0.057 | spacing only |
| prefix10 / tw | C | +0.263 / +0.263 | -0.107 / +0.016 | both-judge/native review |
| prefix10 / haw | C | +0.500 / +0.300 | -0.057 / -0.101 | both-judge/native review |
| prefix17 / haw | C | +0.200 / +0.100 | -0.015 / +0.003 | both-judge/native review |

Min Nan romanization and the Twi→Akan ASR proxy remain explicit limitations. Several other flagged baselines already mismatch both judges; spelling, segmentation and recognition errors remain possible. Dagbani ten-word's LLM drops the opening name while CTC retains it; Chichewa seventeen-word changes an s/z string in both judges; these request native listening, not a categorical failure label. ASR agreement does not certify pronunciation, voice identity or perceived quality, and ECAPA is an uncalibrated synthetic/multilingual proxy. Neither prefix is promoted.

## Duration-fixed follow-up inputs (read-only audit; not executed)

Root's builder `load/build_reference_fixed_duration.py` and fixtures `quality/reference-fixed-duration-v1/{config,corpus396,full-plan,prefix10-plan,prefix17-plan}.json` preserve all132 case IDs/order and original settings except the explicit duration. The pinned tokenizer config SHAeefb2080… and Transformers config-source SHA2fbb4894… establish 25frames/s and960samples/frame at24kHz. The pinned helper appends4800silence samples even with postoff. All132 baseline WAV body hashes and decoded sample counts independently match `(T+5)*960`. `duration=(T+0.25)/25` survives JSON and HTTP-string round trips and floors to the same integer T for every case. All396 new cases preserve reference identity, text, ordering and nonduration options.

`fixed-duration-readonly-audit.json` records everybaseline SHA/T/sample count plus schema/data paths. The future controller must enforce full-fixed132/132 byte equality before cropped arms, assert the enforced header and decoded sample counts, keep all cases on the short iterative path, and preserve source/seed/registration/order. Config gate text alone does not enforce a stop. Using padded WAV duration as the duration argument would add five target frames and invalidate the comparison. Recording actual GenerationTask.target_lens would be optional additional instrumentation; it has not been measured and is not a current gate. No GPU execution or quality promotion follows from this CPU audit.
