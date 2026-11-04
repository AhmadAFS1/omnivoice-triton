# Stage1 completed audio review

Keep split opt-in/unpromoted. Aggregate ASR CER is slightly better, but paired word/signal preservation is not established; review/repeat targeted regressions and baseline Arabic probe failure patterns before broader algorithm rollout. The combined candidate (unused-KV-cache disable plus graph LRU and serving changes) preserves all299 exact WAV bytes under this profile; isolated attribution requires the separate controlled runs.

CPU-only postprocessing on Apple M5 Pro; no new inference. Source TTS/ASR: physical RTX4070,12,282MiB; visible12,450,070,528bytes; CUDA12.8. ASR cards are both1B BF16, blind batch1, with empty co-load checks. Full source metadata is retained in `source-summary.json`.

| Arm | LLM micro CER / WER | CTC micro CER / WER | Both exact | Both CER0 |
|---|---:|---:|---:|---:|
| before-all299 | 10.18% / 29.68% | 18.09% / 43.62% | 30/299 | 35/299 |
| candidate-all299 | 10.18% / 29.68% | 18.09% / 43.62% | 30/299 | 35/299 |
| experimental-all299 | 9.93% / 28.86% | 17.99% / 42.78% | 32/299 | 36/299 |

The combined candidate (unused-KV-cache disable plus graph LRU and serving changes) is299/299 byte-identical; split0/299. The legacy machine field KV_hash_equal refers to this combined candidate, not an isolated KV-only arm. Both judges' CER worsens on39cases and improves on34cases. Only `cs-critical` goes from both judges exactly matching to both mismatching: `úřad` becomes `úřadat` / `úřadad` (CER0→2.25%). The recognized Czech negation and48hours remain. The two cases with at least10percentage-point deterioration in both judges are `so-short` and `bn-medium`; Bengali CTC also switches into Latin script. These require targeted audio/native review, not automatic attribution to TTS word loss.

No case whose expected digit signature was present in both baseline transcripts loses that signature in both split transcripts. This does not validate spoken-number or negation semantics across104languages.

`review_required` counts conflate word differences with uncalibrated targets, source mismatch and critical-content review. Baseline has30both-normalized-exact cases despite only a few eligible screen_matches;287/299 rows carry an uncalibrated/control flag. Punctuation/case are already normalized. Spacing-only differences remain visible in WER while CER is0. CTC's unconditioned script output can inflate CER/WER, e.g. Urdu reference rendered in Devanagari; this is ASR evidence limitation, not proof that speech changed language.

All84old model-probe rows carry nonHTTP/source-corpus flags. EN40/40have bothCER0 (LLM often writes `to day`); ES15/22have both exact; AR12/22have both exact. In each Arabic arm, the same five positions are problematic: B1-000/B2-000/B8-000 near-silent, CTC empty and LLM unrelated; B8-004 both judges only `مرحبا`; B8-007 both only `كيف تشعر اليوم`. The last case has RMS~0.07, so low level alone cannot explain the pattern. These problems already exist in baseline. Previous blind Whisper B1 results agree on EN/ES and also fail on AR; simple gain did not resolve AR.

Five of nine synthetic controls differ: ar,da,el,fi,ur. AR CTC is exact but LLM appends a word; Danish/Greek/Finnish errors are mostly spelling/word boundaries (lowCER); Urdu LLM is close but CTC returns Devanagari and LLM prepends `16`. Four controls de/en/es/fr are exact in both judges. No global row-order, hash or scoring-schema failure was found: all stored metrics recomputed exactly for distinct expected/transcript pairs. This recomputation validates scoring, not native recognition accuracy.

## Per-language micro CER baseline→split

Numbers below are percentages; each language has2–3samples, not a confidence interval. WER/macro metrics, profile and per-case raw deltas are in `per-language.json` and `paired-case-deltas.jsonl`.

| Language / script | n | LLM CER | CTC CER | Both exact before→split |
|---|---:|---:|---:|---:|
| af / afr_Latn | 3 | 8.6%→13.4% | 9.1%→11.5% | 0→0 |
| am / amh_Ethi | 3 | 21.0%→4.2% | 23.5%→5.9% | 1→1 |
| ar / arb_Arab | 3 | 5.3%→6.6% | 6.0%→9.9% | 1→1 |
| ar-AE / afb_Arab | 2 | 9.5%→8.4% | 7.4%→9.5% | 0→0 |
| ar-DZ / arq_Arab | 2 | 2.1%→5.3% | 17.0%→20.2% | 0→0 |
| ar-EG / aec_Arab | 2 | 8.7%→13.5% | 2.9%→8.7% | 1→1 |
| ar-IQ / acm_Arab | 3 | 6.8%→6.8% | 14.2%→15.5% | 0→0 |
| ar-SA / acw_Arab | 2 | 2.1%→1.1% | 2.1%→2.1% | 1→1 |
| ar-SD / apd_Arab | 2 | 7.4%→5.3% | 16.0%→17.0% | 0→0 |
| ar-SY / apc_Arab | 3 | 7.6%→9.7% | 6.9%→6.9% | 0→0 |
| ar-TN / aeb_Arab | 3 | 19.6%→21.0% | 28.7%→27.3% | 0→0 |
| as / asm_Beng | 3 | 4.8%→4.8% | 7.8%→7.2% | 0→0 |
| ayp / ayp_Arab | 2 | 33.7%→22.4% | 19.4%→23.5% | 0→0 |
| be / bel_Cyrl | 3 | 3.7%→3.7% | 7.0%→16.6% | 0→0 |
| bho / bho_Deva | 3 | 13.2%→8.3% | 36.8%→11.1% | 0→0 |
| bjj / bjj_Deva | 2 | 33.3%→32.1% | 101.2%→102.4% | 0→0 |
| bn / ben_Beng | 3 | 4.4%→8.8% | 21.9%→47.5% | 0→0 |
| bs / bos_Latn | 3 | 1.8%→3.7% | 9.8%→9.1% | 0→0 |
| ceb / ceb_Latn | 3 | 6.8%→2.1% | 7.3%→8.4% | 0→0 |
| cs / ces_Latn | 3 | 1.2%→2.4% | 1.2%→2.4% | 2→1 |
| da / dan_Latn | 3 | 2.5%→1.5% | 4.5%→4.0% | 0→0 |
| dag / dag_Latn | 2 | 12.6%→16.5% | 18.9%→21.3% | 0→0 |
| dcc / dcc_Arab | 2 | 11.6%→17.4% | 104.7%→96.5% | 0→0 |
| de / deu_Latn | 3 | 0.0%→0.0% | 1.5%→1.5% | 2→2 |
| dyu / dyu_Latn | 3 | 27.3%→30.2% | 17.4%→20.9% | 0→0 |
| el / ell_Grek | 3 | 0.5%→0.5% | 3.5%→2.5% | 1→1 |
| en / eng_Latn | 3 | 2.2%→3.4% | 7.9%→7.9% | 1→1 |
| es / spa_Latn | 3 | 0.0%→0.0% | 1.6%→1.6% | 1→1 |
| et / ekk_Latn | 3 | 10.5%→9.2% | 10.5%→11.2% | 0→0 |
| fi / fin_Latn | 3 | 2.6%→2.1% | 3.1%→4.7% | 1→1 |
| fil / fil_Latn | 3 | 3.4%→5.6% | 3.4%→5.2% | 0→0 |
| fr / fra_Latn | 3 | 5.6%→5.6% | 1.9%→2.3% | 2→2 |
| gu / guj_Gujr | 3 | 22.6%→20.0% | 23.2%→23.2% | 0→0 |
| ha / hau_Latn | 3 | 5.3%→5.3% | 7.4%→4.2% | 0→0 |
| haw / haw_Latn | 2 | 11.2%→13.3% | 21.4%→19.4% | 0→0 |
| he / heb_Hebr | 3 | 26.4%→31.2% | 113.6%→132.8% | 0→0 |
| hi / hin_Deva | 3 | 5.5%→9.7% | 23.6%→26.7% | 0→0 |
| hr / hrv_Latn | 3 | 3.2%→3.2% | 7.1%→5.8% | 1→1 |
| ht / hat_Latn | 3 | 23.3%→25.3% | 27.3%→28.7% | 0→0 |
| hu / hun_Latn | 3 | 11.8%→14.9% | 9.3%→29.2% | 0→0 |
| hy / hye_Armn | 3 | 7.6%→8.1% | 6.6%→7.1% | 1→1 |
| id / ind_Latn | 3 | 3.2%→2.7% | 5.4%→4.3% | 0→1 |
| ig / ibo_Latn | 3 | 18.9%→20.0% | 17.1%→17.7% | 0→0 |
| it / ita_Latn | 3 | 0.5%→0.5% | 1.0%→1.0% | 1→1 |
| ja / jpn_Jpan | 3 | 4.9%→4.9% | 8.5%→8.5% | 0→0 |
| jv / jav_Latn | 3 | 8.3%→8.3% | 10.6%→11.7% | 0→0 |
| ka / kat_Geor | 3 | 10.1%→12.4% | 19.5%→17.8% | 1→1 |
| kk / kaz_Cyrl | 3 | 4.6%→5.1% | 8.7%→7.1% | 0→1 |
| km / khm_Khmr | 3 | 21.8%→20.8% | 37.0%→38.4% | 0→0 |
| kn / kan_Knda | 3 | 17.5%→19.2% | 21.5%→21.5% | 0→0 |
| ko / kor_Hang | 3 | 4.5%→4.5% | 6.1%→6.1% | 1→1 |
| ln / lin_Latn | 3 | 15.7%→7.1% | 13.7%→10.2% | 0→0 |
| lo / lao_Laoo | 3 | 20.8%→20.3% | 37.6%→38.1% | 0→0 |
| lt / lit_Latn | 3 | 6.1%→9.6% | 7.6%→7.1% | 0→0 |
| mai / mai_Deva | 3 | 14.2%→12.9% | 34.8%→40.0% | 0→0 |
| mi / mri_Latn | 3 | 1.2%→2.3% | 9.8%→13.3% | 0→0 |
| mk / mkd_Cyrl | 3 | 5.5%→3.6% | 39.4%→6.7% | 0→0 |
| ml / mal_Mlym | 3 | 10.7%→8.2% | 9.9%→9.0% | 1→1 |
| mn / mon_Cyrl | 3 | 19.1%→11.7% | 27.2%→10.5% | 0→0 |
| mr / mar_Deva | 3 | 9.7%→9.0% | 11.0%→11.0% | 0→0 |
| ms / zsm_Latn | 3 | 11.2%→7.3% | 16.6%→13.7% | 0→0 |
| my / mya_Mymr | 3 | 24.0%→26.2% | 33.5%→30.8% | 0→0 |
| nan / nan_Latn | 2 | 43.5%→35.2% | 56.5%→55.6% | 0→0 |
| ne / nep_Deva | 3 | 3.0%→3.7% | 15.9%→6.1% | 0→0 |
| nl / nld_Latn | 3 | 10.4%→6.0% | 9.9%→9.9% | 0→0 |
| no / nob_Latn | 3 | 3.7%→3.1% | 6.3%→4.7% | 0→0 |
| nso / nso_Latn | 3 | 3.4%→5.4% | 4.4%→5.4% | 0→0 |
| ny / nya_Latn | 3 | 0.0%→1.2% | 4.3%→3.7% | 0→0 |
| pa / pan_Guru | 3 | 55.7%→53.9% | 63.5%→64.7% | 1→1 |
| pcm / pcm_Latn | 2 | 12.2%→12.2% | 22.0%→19.5% | 0→0 |
| pl / pol_Latn | 3 | 0.5%→0.5% | 0.5%→0.5% | 2→2 |
| pt / por_Latn | 3 | 2.1%→2.1% | 2.6%→2.1% | 1→1 |
| ro / ron_Latn | 3 | 3.9%→3.4% | 4.9%→4.4% | 0→1 |
| ru / rus_Cyrl | 3 | 0.0%→0.0% | 1.1%→1.6% | 2→2 |
| rw / kin_Latn | 3 | 7.1%→7.1% | 1.5%→3.0% | 0→0 |
| sd / snd_Arab | 3 | 29.7%→30.4% | 70.3%→67.6% | 0→0 |
| si / sin_Sinh | 3 | 16.3%→16.9% | 53.5%→51.7% | 0→0 |
| sk / slk_Latn | 3 | 5.0%→5.6% | 5.0%→6.1% | 0→0 |
| skr / skr_Arab | 2 | 44.2%→34.9% | 95.3%→100.0% | 0→0 |
| sl / slv_Latn | 3 | 0.6%→0.6% | 1.2%→1.2% | 2→2 |
| sn / sna_Latn | 3 | 4.1%→1.8% | 5.9%→6.8% | 0→0 |
| so / som_Latn | 3 | 6.5%→8.3% | 7.9%→10.6% | 0→0 |
| sq / als_Latn | 3 | 9.9%→12.6% | 12.0%→12.0% | 0→0 |
| sr / srp_Cyrl | 3 | 3.6%→3.0% | 43.5%→37.5% | 0→0 |
| sv / swe_Latn | 3 | 1.2%→1.2% | 1.2%→0.6% | 1→1 |
| sw / swh_Latn | 3 | 8.7%→6.6% | 9.2%→7.7% | 0→0 |
| ta / tam_Taml | 3 | 12.0%→9.4% | 20.3%→19.3% | 0→0 |
| te / tel_Telu | 3 | 26.1%→24.8% | 23.6%→24.2% | 0→0 |
| tg / tgk_Cyrl | 3 | 3.4%→4.5% | 18.8%→18.8% | 0→0 |
| th / tha_Thai | 3 | 9.9%→11.5% | 8.9%→12.0% | 0→0 |
| ti / tir_Ethi | 3 | 22.1%→21.4% | 42.1%→40.7% | 0→0 |
| tk / tuk_Latn | 3 | 17.1%→27.1% | 28.7%→28.2% | 0→0 |
| tn / tsn_Latn | 3 | 9.3%→6.5% | 5.6%→8.8% | 0→0 |
| tr / tur_Latn | 3 | 18.9%→14.5% | 27.7%→30.2% | 0→0 |
| tw / aka_Latn | 3 | 23.6%→21.3% | 29.2%→21.3% | 0→0 |
| ug / uig_Arab | 3 | 10.9%→10.9% | 25.3%→23.0% | 0→0 |
| uk / ukr_Cyrl | 3 | 8.7%→4.6% | 6.6%→4.6% | 0→0 |
| ur / urd_Arab | 3 | 3.0%→1.2% | 101.2%→103.0% | 0→0 |
| vi / vie_Latn | 3 | 6.8%→8.2% | 8.9%→14.4% | 1→1 |
| xh / xho_Latn | 3 | 4.2%→3.8% | 7.1%→5.7% | 0→0 |
| yo / yor_Latn | 3 | 24.4%→18.1% | 50.6%→47.5% | 0→0 |
| yue / yue_Hant | 3 | 46.9%→46.9% | 85.7%→93.9% | 0→0 |
| zh / cmn_Hans | 3 | 11.7%→11.7% | 55.0%→53.3% | 0→0 |
| zu / zul_Latn | 3 | 3.8%→4.7% | 4.7%→9.0% | 0→0 |

Source files and all calculations are hashed in `analysis-summary.json`. Original reports, ASR outputs and WAVs remain unchanged. ASR-only evidence does not establish MOS, native fluency, speaker identity or absence of word loss.
