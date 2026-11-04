# 104-language before/after review audio

**Original inference GPU:** NVIDIA GeForce RTX 4070, physical12,282MiB from
`nvidia-smi`, Torch-visible12,450,070,528bytes; driver580.95.05, CUDA12.8,
Torch2.8.0+cu128, OmniVoice0.1.4. Source TTS measurements began2026-11-04UTC;
the directory retains the original local-date benchmark naming convention.
Per-arm dates, runtime settings and co-load evidence are in `paired-manifest.json`
and `export-provenance.json`. Export/scoring were CPU-only on Apple M5 Pro;
they did not generate new speech or run ASR.

`before/` and `after/` contain one short complete WAV for each of104Lingua
routes:208files. Every pair is byte-identical. The before arm is baselinec79ce35;
the after arm is the combined1eb820f candidate with unused KV cache disabled,
graph LRU and serving changes. This comparison does not isolate the contribution
of any single optimization. Both used hybrid inference,16steps, speed1,
postprocessing off, graph limit8, codec decode batch1 and startupseed20261103.
The model revision wasc5fdb5ccb189668d56333f77ba2629f4cd7535f4.

The paired manifest retains expected published text, exact hashes, source and
license provenance, reference identity/language, native-versus-English-fallback
caveats, and both blind Meta Omnilingual1B judge transcripts/metrics/flags. The
reference profile uses the original nine Aoede assets;95target routes use the
explicit English-reference fallback. Twi's broader Akan ASR mapping remains a
proxy. This sample set does not cover the later eight-voice/three-preset grid.

ASR ran separately on the original RTX4070 using BF16, batch1,
`omniASR_LLM_Unlimited_1B_v2` and `omniASR_CTC_1B_v2`; full hardware/software
metadata is copied unchanged into `asr-source-metadata/`; `quality-summary.json`
preserves the complete local source report summary and hashes. Only original
stage1published outputs were consumed, without later ASR batching experiments.
Punctuation/spacing/spelling, uncalibrated-language and word-mismatch flags remain
visible. Byte equality establishes preservation of these generated samples;
it does not certify pronunciation, words, voice identity or all-language quality.

The full299-case runs and load WAVs remain retained outside this review subset.
The inventory,299-case corpus,1105-case load pool and source/license notices are
packaged in `../../fixtures/`. See `verification.json` for independent208-WAV,
manifest and fixture hash checks. No API calls or new GPU inference were used to
create this package.
