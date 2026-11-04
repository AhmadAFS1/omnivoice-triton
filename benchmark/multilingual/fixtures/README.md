# Lingua language coverage and published fixtures

**This is a CPU-only inventory and fixture preparation run. No GPU inference, TTS generation, translation API, or transcription API was used here.** GPU performance and audio quality must be measured by the separate RTX 4070 experiment; recommended or selected hardware is not tested hardware in this inventory.

The authoritative offered product list is `LANGUAGES` in `app/constants/languages.ts`. It is built from `RAW_LANGUAGE_CATALOG`, the shared OmniVoice capability registry, and a deduplication by product language code. Current source counts:

| Registry | Count | Meaning |
| --- | ---: | --- |
| Full raw catalog | 139 rows | Includes review-only entries and an alias |
| Ready labels | 105 | Includes Filipino and Pilipino, both code `fil` |
| Offered product languages | **104** | Ready routes, deduplicated by code |
| Review-only catalog entries | 34 | Kept for tooling, excluded from offered languages |
| Legacy sample-language registry | 24 | A separate legacy voice-sample list |
| Actual local Aoede references | **9** | Existing WAVs verified against the local manifest |
| This test's English-reference fallback targets | **95** | Explicit cross-language cloning profile |

The capability JSON records a September 13, 2026 `/languages` validation. Its `ready` status is configured route support, not new proof of pronunciation, word completeness, low latency, or ASR accuracy. The wizard also preserves previously stored out-of-catalog values; those are not an enumerable supported-language catalog. Some authored UI content has narrower coverage: 34 greeting samples, for example.

## Main files

- `active-languages.json`: all 104 offered routes with exact product code, provider synthesis ID, ASR language/script ID, source notes, and reference profile.
- `inventory-summary.json`: source Git commit, exact file hashes, counts, and scope.
- `review-only-languages.json`: the 34 excluded entries, kept separate from the test denominator.
- `catalog-aliases.json`: the deduplicated Filipino/Pilipino alias.
- `references/manifest.json`: nine portable `file`/`path` reference records, transcripts, hashes, and durations. WAV paths are relative to the manifest.
- `corpus-smoke-104.json`: exactly one published short input per offered route.
- `corpus-medium-104.json`: exactly one additional published input per route.
- `corpus.json`: 299 QA inputs. The 91 FLORES routes have three shared-content cases; the other 13 routes have two published excerpts each.
- `load-pool.json`: 1,105 unique `(language, text, reference)` inputs, containing 1,103 distinct text strings. The first 1,000 include all 104 product routes.
- `load-pool-summary.json`: exact distribution and uniqueness counts. The 91 FLORES routes have 12 inputs each; the 13 other routes have one short input each in this pool. This is a coverage stress fixture, not a measured production traffic distribution.

Each case carries `id`, `language`, `api_language`, `omnilingual_language`, `text`, `length`, `reference_id`, `reference_language`, `voice_profile`, translation/native-review status, and the published source identity/license. Do not remove that provenance when joining requests and ASR scores.

## Reference coverage

The nine actual local Aoede references are Arabic, Danish, German, Greek, English, Spanish, Finnish, French and Urdu. They were previously synthesized with Gemini and are reused here; this work made no new Gemini API call. Their existing seed transcripts still need calibration/native review. Most are longer than the usual 3–10 second reference guidance.

The local manifest/config mentions 24 languages but contains only a partial asset set. The production EC2/S3 manifest and environment overrides were not inspected, so this inventory does not claim production has exactly nine voices or exactly 95 fallbacks. `backend/services/default_voice_samples.py` implements a missing-reference fallback whose configured default is English. This experiment labels its 95 English-reference cases explicitly, rather than pretending they use native-language reference recordings.

## Exact synthesis/ASR mappings and limits

All 104 selected ASR IDs exist in the pinned official Omnilingual list of 1,672 language/script IDs. Membership is not language-quality qualification. Keep synthesis IDs separate from ASR IDs: Nepali synthesis uses `npi`, while the ASR ID is `nep_Deva`.

Important choices are recorded per route:

- Twi has no `twi_*` ID in that list. `aka_Latn` is a **broader Akan proxy**, and `asr_mapping_verified` is false in Twi cases. Run it as a diagnostic if useful; do not count it as Twi-specific native ASR certification.
- Norwegian uses Bokmål, Albanian uses standard/Tosk, Punjabi uses Gurmukhi, Mandarin uses Simplified script, and Cantonese uses Traditional script. Other varieties/scripts require separate qualification.
- Min Nan's selected ASR ID is `nan_Latn`, with a native romanized Wikipedia fixture. This does not cover Han-script Min Nan input.
- Estonian ASR uses `ekk_Latn`; Filipino ASR uses `fil_Latn` with Tagalog FLORES source text explicitly labeled as the dataset variety. Mongolian uses `mon_Cyrl` with Khalkha source text.
- Written Chinese matching does not establish spoken Mandarin/Cantonese identity. FLORES Cantonese text has known variety concerns; no native-language pronunciation claim is made from text matching alone.

Every row sets `native_speech_language_identity_verified` to false and marks overall quality unqualified. The baseline must actually synthesize each route and save its complete WAV. ASR output, signal validity, and HTTP 200 are separate evidence. Native listening is needed for pronunciation, dialect identity, speaker similarity and naturalness. WER is not comparable across writing systems; retain CER and raw words. Spacing and numbers-as-digits can produce false alignment errors. Two Omnilingual variants share a family and are not independent ground truth.

## Published text sources and licenses

No translation was invented to fill a missing language. Sources are:

1. **FLORES-200, 91 routes.** Exact dev lines 213, 352 and 291 (one-based) provide a short negation sentence, a medium statement, and a critical names/number/negation sentence. The corresponding English content is retained for explanation. Published professional translations are not independently native-reviewed by this benchmark. The original archive SHA-256 is `b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6`.
2. **Meta Omnilingual ASR corpus, 9 routes.** Exact public parquet revision `8648ba8946377697b427ae952076e49fc0e5e44d`, metadata/text columns only. Each selected text is a contiguous source transcript span with file, row group, row, speaker/prompt/segment IDs and offsets. Entire original audio must not be scored against a text excerpt. The raw transcript content and its semantic/native correctness were not independently reviewed here.
3. **Native-language Wikipedia articles, 4 routes:** Min Nan, Nigerian Pidgin, Seraiki and Dagbani. Exact revision/page IDs, excerpts and source hashes are retained. These are published community-written native text excerpts, **not gold translations** or independently reviewed speech scripts.

FLORES-200 and Wikipedia text excerpts are **CC BY-SA 4.0**. Meta corpus text is **CC BY 4.0**. These data licenses and attribution remain separate from the benchmark code's license. Preserve per-row source URLs, revision identifiers and attribution. The source download/selection scripts and source snapshots make the text selection inspectable.

Sources:

- Lingua source registry: `app/constants/languages.ts`; exact Git/source hashes are in `inventory-summary.json`.
- [Official Omnilingual language/script list at the pinned commit](https://github.com/facebookresearch/omnilingual-asr/blob/81f51e224ce9e74b02cc2a3eaf21b2d91d743455/src/omnilingual_asr/models/wav2vec2_llama/lang_ids.py)
- [FLORES-200 source and license](https://github.com/facebookresearch/flores)
- [FLORES-200 original dataset documentation](https://github.com/facebookresearch/flores/tree/main/flores200)
- [Meta Omnilingual ASR corpus](https://huggingface.co/datasets/facebook/omnilingual-asr-corpus)
- [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) and [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
