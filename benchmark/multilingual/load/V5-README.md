# V5 voice/control protocol

`run_v5.py` is a separate future harness. The running V4 source remains
`df51c081d8108bf7dae5778adc2f21398e59d0ca8de394601f4251a55c521652`.
V5 preserves complete-body timing, deferred WAV retention, fresh language/health
validation and startup-seed proof. It adds explicit per-case controls and
separate boundary-check accounting. No V5 GPU inference has been performed by
its development checks.

Cases may carry `worker_options` containing only these exact API Form fields:
`num_step`, `guidance_scale`, `t_shift`, `layer_penalty_factor`,
`position_temperature`, `class_temperature`, `speed`, `duration`, `denoise`,
`preprocess_prompt`, `postprocess_mode`, `insert_pauses`. Unknown fields and
misplaced top-level generation options fail before generation. Instruction text
uses the existing top-level `instruct` field. Voice IDs remain provenance; the
actual clone input is the selected WAV and transcript.

Every default/override is normalized, posted and retained as exact Form values.
Default benchmark steps are16, speed1, denoise/preprocess true, postprocess off,
pauses false, guidance2, shift0.1, layer/position5, class0. Optional duration is
unset. Per-case Lingua presets override these values explicitly. The profile
metadata describes defaults and per-case behavior rather than incorrectly
labeling every request with one fixed speed/postprocess setting.

Reference registration keys include WAV SHA256, transcript and preprocessing
flag. A true/false preprocessing variant gets its own registration; changing
the generate flag on an already prepared prompt is never presented as testing
reference preprocessing. Registered and uploaded transport are logged along
with the actual application point. SHA/transcript-identical reference aliases
share a semantic identity even if their human-readable IDs differ.

The input identity includes text/mode/API language/instruction, every normalized
sent generation field, and reference SHA/transcript/preprocessing. Random prompt
UUIDs are excluded. This distinguishes all three Lingua presets and prevents
alias IDs from inflating uniqueness. Some API fields have documented model
normalization: duration<=0 means unset; positive duration overrides speed.
Configuration-grid variation is therefore not a claim of novel compute for
every numeric spelling or an independent language-quality observation.

Numeric/nonfinite admission tests have `validation_only:true`, explicit expected
HTTP400, and literal form strings such as `"nan"`; JSON NaN/Infinity is rejected.
Their phases are finite sequential checks. Expected rejections are excluded
from TTS throughput and ASR quality. Unexpected HTTP200 bodies are still saved
with hashes, tagged as guard failures, and excluded from ASR-ready output.
Use the canonical V3 quality helper for these fields. Duration0/-1 are accepted
API sentinel cases in the positive matrix, not invalid-number tests. API
acceptance alone does not establish good audio for an extreme model setting.

`voice-plans-v5/jobs.json` contains source-pinned commands, not an execution:

- Gate:104 cases across Arabic/English/Spanish/Hindi, eight voices and three
  presets plus auto/design. This exercises native and English-reference
  fallback profiles; it is four-language coverage, not104-language coverage.
- Full:2,704 cases per arm, with24 clone voice/preset phases covering104 routes
  each, then104 auto and104 design cases. Phase checkpoints retain evidence
  regularly rather than holding every waveform until a single huge phase ends.
- Matrix:130 finite pairwise/vocabulary/boundary cases in five groups. There
  are127 distinct sent-control identities; repeated configurations are labeled
  as configuration observations, not novel-inference capacity.
- Validation:28 expected400 checks, separate from positive quality/throughput.

The64 reference assets contain64 distinct WAV hashes and include all original
nine Aoede assets unchanged. Their native coverage is sparse across nine
languages, not a complete nine-by-eight rectangle. Among832 language/voice
pairs,64 use target-language references and768 use English-reference fallback.
Existing source/proxy/script and native-review caveats remain in every case.

Run the small gate and judge both arms before the full grid. Source-pinned
commands use candidate445f21202de940daf4b26a1ed52710eb1efd63d7 with unused KV
cache disabled, split CFG off, graphs8, codec batch1, output cache off and16
default steps. Per-case controls, such as4/64 steps in the matrix, still apply.
The execution root is an argument to the CPU plan builder, so a different
machine can retain the same plan/profile with independently measured hardware.

CPU checks passed:17 actual mocked HTTP calls,15 retained/hash-verified WAVs,
two expected400 checks, all12 worker fields observed by the handler, registered
preprocessing variants, direct upload, auto/design instructions, and accurate
identity distinctions/alias collapse. Full fixture/schema/hash preflight passed
for2,704/130/28 cases. These are protocol checks, not GPU speed or word-quality
results. `verify_v5.py` independently checks raw options against logged posted
forms, reconstructs identity hashes, validates registration provenance, and
checks complete-body/WAV/denominator accounting.
