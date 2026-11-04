# Complete-WAV multilingual quality postprocessing

**GPU: none.** This helper performs CPU-only file hashing, text scoring, report generation and optional file packaging. Its synthetic validation ran on **Apple M5 Pro, 25,769,803,776 bytes unified RAM**, macOS 27.0.1, Python 3.14.7, UTC 2026-11-04. It did not load a TTS/ASR model, call a paid API, or measure GPU speed. Real TTS and ASR GPU/physical VRAM/visible VRAM/runtime/co-load evidence is carried from each run's metadata; absent evidence stays unverified. See `validation.json` for exact source hashes and validation time.

`report.py` uses only the Python standard library. It retains the exact **104 product routes and 299 published corpus cases** from `../languages`. There are nine prerecorded Aoede target-language references; the other 95 target routes explicitly use an English cross-language clone reference. No report or sample export is evidence of native-language quality certification.

## 1. Prepare unique ASR inputs after timed TTS finishes

```sh
python quality/report.py prepare \
  --runs-root /absolute/benchmark/results \
  --reference-manifest languages/references/manifest.json \
  --output /absolute/benchmark/quality-prepared
```

Paths here assume the benchmark root is the working directory. `--runs-root` recursively discovers `quality-manifest.json`; point it only at real selected runs, excluding CPU/mock smoke directories. Alternatively, repeat `--run-dir` with uniquely named run directories. Arm names are relative directory paths under `--runs-root`, or directory basenames with `--run-dir`. Output directories must be fresh, preserving previous evidence.

The helper joins `requests.jsonl` with the full HTTP200 and ASR-ready manifests. It records failed HTTP requests, missing files, manifest inconsistencies, signal flags, SHA mismatches, source texts/licenses, reference profiles, and cache/coalescing headers. Reference controls are a separate kind and never enter HTTP/TTS availability denominators. Missing request logs explicitly make the failure denominator unknown.

Every retained WAV is hash-verified. Hash reading is memoized by exact file path/stat; separate cached copies are still individually checked. The exported `asr-input-manifest.json` contains **one input per verified original WAV SHA256 plus exact ASR language/script**, before the existing evaluator decodes, chunks or runs models. This avoids decoding/chunking thousands of cached copies. The deduplication key deliberately excludes expected text: one wrong WAV accidentally returned for multiple texts still gets scored against every expected text. An origin request ID is retained as a diagnostic, never trusted in place of actual bytes. Same-origin hash or semantic-input conflicts are reported.

`logical-cases.jsonl` maps every original request/control to its unique ASR input. `coverage.json` contains all 104 routes, including unrun routes. Preparation alone never declares a quality pass.

V3 preparation preserves V5 `voice_id`, `config_type`, actual `generation_options`, reference SHA/transcript/preprocessing identity, transport and preprocessing stage. Validation-only cases are retained as `kind=validation_check`; expected rejections and unexpected HTTP200 boundary responses never enter normal TTS availability or ASR-quality denominators. Unexpected successes keep their WAV bytes/hashes and are reported as guard failures. `validation-checks.jsonl` makes these outcomes independently reviewable.

Non-HTTP split-CFG/KV-cache model-stage probes are also supported. Discovery recognizes a probe only when the per-arm ASR-ready HTTP manifest is absent and the parent metadata contains `prototype_sha256`; otherwise explicitly repeat `--probe-manifest /absolute/probe/arm/quality-manifest.json`. The probe's existing `signal_valid` flag plus verified hash and exact language gates ASR preparation. These records retain `kind=model_probe`, `http_status=null`, separate probe counters/metrics, and an unknown failure denominator. They never become HTTP200 successes. A published-text source joins only when the case identifier, text and route agree exactly; an omitted operational reference is unreported even if the corpus describes an intended fixture reference.

## 2. Run the existing blind GPU ASR evaluator separately

The orchestration owner runs `load/asr/evaluate.py` on `asr-input-manifest.json` after TTS releases its GPU resources. This helper never starts that operation. The existing evaluator reads audio and known language/script only; expected text is used for scoring after transcription and is not fed to ASR. Its default judge cards are:

- `omniASR_LLM_Unlimited_1B_v2`
- `omniASR_CTC_1B_v2`

Keep `<model>.jsonl`, `<model>-metadata.json`, and `prepared.json` together. Existing earlier evaluator outputs are supported when IDs can be joined through their `prepared.json`, or when a hash has only one unambiguous language. Ambiguous hash-only languages, wrong hashes/model cards, and conflicting transcripts for the same hash/language are excluded and flagged.

## 3. Score existing outputs on CPU

```sh
python quality/report.py score \
  --prepared /absolute/benchmark/quality-prepared \
  --asr-dir /absolute/benchmark/asr-outputs \
  --pair baseline=candidate \
  --output /absolute/benchmark/quality-report
```

Repeat `--asr-dir` if the two cards were stored separately. Exactly two distinct cards are required; missing output remains partial/unscored. No transcript is invented for a failed or unavailable input.

V4 treats output without `asr_protocol_config_id` as the frozen legacy batch1
protocol. For each model, one scoring invocation accepts exactly one declared
protocol identity; mixed batch/dtype/config identities fail before comparisons,
even if the recognized text happens to match. Explicit metadata and row IDs must
agree. Score future B1/B2/B4 experiments separately and retain each protocol's
independent inference evidence; a declared ID alone does not prove inference
provenance. The V3 source is preserved in `history/`, and no historical report or
ASR output is rewritten by this change.

Outputs include:

- `per-route.json` / `.csv`: all 104 routes, availability, native-reference/English-fallback counts, mapping caveats, and coverage.
- `scored-logical-cases.jsonl`: both raw transcripts, WER/CER, numeric/spacing/mismatch/disagreement flags, source and audio paths.
- `paired-before-after.jsonl`: matched phase/case/semantic input/occurrence, HTTP outcomes, before/after WAV hashes and judge deltas. Prompt UUIDs are server-local and excluded from semantic matching; reference identity and generation parameters are retained.
- `origin-propagation.json`: every logical request per generated/cache/coalesced origin, with independently verified hashes, unique ASR evidence IDs and conflicts.
- `summary.json` / `report.md`: CPU vs GPU attribution, source hashes, missing evidence and interpretation limits.

Metrics are reported both across logical requests and across **unique audio plus expected text**. Cache hits never become independent quality evidence. Two judge outputs propagate to all matching requests, but each request is compared to its own expected text. An identical paired WAV establishes byte preservation for that input; an unseeded hash difference alone is not a regression.

Normalization matches the previous evaluator: NFKC, case folding, punctuation/symbols to spaces, limited Arabic diacritic/alef/ya normalization, and collapsed whitespace. CER then ignores whitespace. Standard unit-cost Levenshtein provides WER/CER; tied edit-path substitution/deletion/insertion counts may differ from jiwer while total distance remains identical. Digits versus spoken-number representations are review flags, not proven TTS errors. WER across scripts is not directly comparable.

Synthetic reference controls can expose ASR/transcript problems for their target languages. V3 calibration stores every exact reference ID separately, including all64stock-voice assets when supplied; language summaries retain every control instead of overwriting one voice with another. A request with a native reference uses its actual reference's control; fallback reference calibration remains separate from target-language calibration. A target with no matching control stays explicitly uncalibrated even when both judges return its expected text. English controls do not calibrate the95fallback target languages. Twi's `aka_Latn` mapping remains an Akan proxy. Critical-content cases require native review. ASR models can share errors and do not establish MOS, voice similarity, negation/number correctness or native fluency.

`per-configuration.json` additionally groups normal outcomes by route, actual voice ID, config type and mode, preserving applied generation/reference settings. This keeps the2704case control plan inspectable beyond route-level aggregates. The prior V2 helper source is preserved under `history/`; historical analysis outputs and raw ASR/audio files are not rewritten. Every new preparation/report records its helper source SHA256 and format version.

## 4. Export reviewable 104-route short pairs only after real runs

```sh
python quality/report.py export-samples \
  --report /absolute/benchmark/quality-report \
  --pair baseline=candidate \
  --output /absolute/fork/benchmark/multilingual/samples/rtx4070-2026-11-03 \
  --fixtures-output /absolute/fork/benchmark/multilingual/fixtures
```

The export refuses CPU/mock GPU evidence, incomplete route coverage, missing/corrupted bytes, ambiguous duplicate short pairs and any WAV over 10 MB. It requires recorded physical GPU evidence containing `RTX 4070` (configurable explicitly); it does not infer VRAM from that name. The default phase is `qa-short-all-languages`.

It packages `before/<route>.wav`, `after/<route>.wav`, a paired manifest with expected text, hashes, source/licensing, reference caveats, ASR flags and per-arm hardware, plus source-corpus/load-pool/inventory fixtures with their attribution. The original full 299-case and load WAV directories stay retained separately; their original paths remain in the evidence. No export has been run merely to prepare this helper.

## CPU validation

```sh
python quality/test_report.py
```

Eight synthetic tests cover 2,005logical rows reducing to three language-conditioned ASR inputs; shared bytes under three scripts; wrong expected text on a cached copy; failed HTTP and corrupt SHA retention;1,000paired hashes; proxy/missing-control caveats; missing judge outputs; conflicting/ambiguous/wrong-hash transcript exclusion; Unicode/CER/spacing normalization; all104unrun coverage; separation of non-HTTP probe evidence; all64reference IDs; exclusion of expected rejections and retained unexpected200boundary outcomes; V5actual setting/reference propagation; refusal to export CPU/mock samples as GPU evidence; and legacy/new/mixed ASR protocol identities plus metadata agreement. `voice-controls/signal-attribute-check-plan.json` separately defines finite speed/duration, waveform differentiation, optional estimatedF0 and RMS checks. It explicitly distinguishes fixed pitch instructions from nonexistent volume/numeric-pitch API knobs, and ASR words from speaker/attribute identity.
