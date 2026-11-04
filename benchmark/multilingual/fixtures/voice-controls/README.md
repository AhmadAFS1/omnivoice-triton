# Voice-control inventory and finite QA matrix

**GPU: none for this inventory or matrix.** CPU-only source inspection, file hashing, pure numeric-parser execution and fixture planning ran on Apple M5 Pro,25,769,803,776bytes unified RAM, macOS27.0.1/Python3.14.7, UTC2026-11-04. No TTS, ASR or API inference was performed here. Future inference must retain its own exact GPU, physical/visible VRAM, runtime, date and co-load metadata. These fixtures are **unexecuted**, with no language/voice quality qualification.

The current Lingua checkout is `claude/ios-25fps` at `809cf46262375e5a2b806d970e121e95252a79c7`. Read-only comparison with the catalog worktree at `86b8a2bf6d624bd0155fd0f18c07a1dd9c7283cd` found identical relevant voice UI, SDK, preset and gateway files; learner-facing voice descriptions are also identical. The existing modified catalog gitlink was untouched. Exact source and branch evidence is in `inventory.json` and `branch-audit.json`.

## What users can currently change

Active Lingua voice creation/edit screens offer eight reference-voice cards, preview playback, and custom reference recording/upload. They do **not** offer an acoustic speed, pitch, age, emotion, temperature or guidance slider. Character gender reorders/recommends the voice cards; it does not send a separate acoustic gender parameter. Gender may affect the selected reference because the suggested card changes.

The eight voices are Despina/Aoede/Zephyr/Kore (female) and Charon/Iapetus/Fenrir/Umbriel (male). `app/constants/defaultVoices.ts` maps these IDs to prerecorded reference assets. Its learner-facing descriptions come from `src/ui/copy.ts`, such as Aoede's “Calm and articulate; comfortable for long sessions.” These descriptions/personas label the reference voice and are not generation instructions. Resolver default is Aoede; the wizard suggests Despina for female and Charon for male characters. Legacy `alloy` resolves to Iapetus. Every legacy alias is retained in the source evidence.

Custom voice samples pass through local multipart upload or remote sample URL, with an optional transcript. Character-owned voices can instead be resolved by backend `character_id`; the server owns that sample selection. Stored stock preview playback is reference playback, not fresh model inference.

`app/components/chat/useMessageAudio.ts` also supports a0.7× **playback** rate with pitch preservation. `DynamicLessonPlayer.tsx` has device `Speech.speak` rate/pitch settings. Neither is an OmniVoice generation pitch/speed slider.

## SDK → backend → worker

The current `APIutils.tsx` remote-clone JSON, local-clone multipart, and default-voice JSON builders send the shared clone options from `app/utils/omnivoiceTTSConfig.ts`: `num_step`, `denoise`, `preprocess_prompt`, `postprocess_mode`, optional `insert_pauses`, and `speed`, alongside text/language/reference/voice identity. Three real product presets are defined:

| Product path | Steps | Speed | Postprocess | Insert sentence pauses |
|---|---:|---:|---|---|
| Default chat/voice call/custom preview |16|1.3|full|omitted/false|
| MuseTalk video call |16|1.0|off|omitted/false|
| Learn tutor |16|1.0|light|true|

The SDK options type still declares `instruct`, `guidanceScale`, `exaggeration`, `temperature`, `diffusionSteps`, `minP`, `topP`, `repetitionPenalty`, `seed`, `splitSentences`, and `outputFormat`; current OmniVoice builders do not forward them. Most are legacy Chatterbox controls, not supported OmniVoice knobs. `instruct` and `guidanceScale` are genuine worker fields but their explicitly supplied SDK overrides are currently lost. They should be added conditionally if that SDK capability is desired, preserving all ordinary requests. Do not forward legacy defaults wholesale, particularly the stale `diffusionSteps=4`, or reinterpret `tone` and display descriptions as instructions.

`docs/omnivoice-tts-payload.md` explicitly says to omit `instruct`, `tone` and `guidance_scale` on the canonical clone path and warns that `tone='sweet'` is unsupported. That is a canonical default-payload guideline. The worker and backend separately support an **explicit** fixed-vocabulary clone instruction. Those two meanings should remain distinct in any SDK correction.

`backend/routes/ai_gateway.py` extracts the real generation option keys and forwards them to worker `/generate`. It maps `ar` to`arb` and strips emoji from text/reference text. `diffusion_steps` is only a fallback alias when `num_step` is absent. Legacy `exaggeration`, `temperature`, `min_p`, `top_p`, `repetition_penalty`, `seed`, and `split_sentences` are not forwarded as worker generation fields.

- `mode=clone`: reference upload or worker prompt ID is required; only explicit `instruct`/`voice_instruction` is forwarded as instruction. `tone` is deliberately ignored.
- `mode=design`: instruction can come from explicit `instruct` or the legacy `tone` alias. Invalid fixed-vocabulary content is rejected by the model. Reference/prompt fields are forbidden by the direct worker API.
- `mode=auto`: the backend helper forwards no instruction. The direct worker API rejects a supplied instruction or reference fields.

The backend internal helper recognizes `voice_instruction`, but current public JSON/multipart extras constructors omit it. `instruct` itself is preserved. Do not invent `emotion`, numeric `pitch`, or arbitrary freeform `voice_description` Form fields: they are not worker controls.

The pinned OmniVoice0.1.4 upstream Gradio demo has three actual sliders: speed0.5–1.5/default1.0, steps4–64/default32, and CFG0–4/default2.0. Its fixed instruction vocabulary provides gender2, age5, pitch5, whisper, English accents10 and Chinese dialects12. Chinese aliases exist for gender/age/pitch/whisper. These are underlying worker/demo capabilities, not currently exposed Lingua sliders. At most one value from each category is valid; do not mix an English accent with a Chinese dialect. Accents are scoped to English and dialects to Chinese. Arbitrary “warm,” “angry,” “sad” or “sweet” descriptions are not supported vocabulary.

## Worker numeric/boolean contracts

| Form field | Worker default | Validation / qualification note |
|---|---|---|
|num_step|32|Positive integer, no API upper bound; matrix uses4/16/64|
|guidance_scale|2.0|Finite nonnegative; GUI0–4|
|t_shift|0.1|Finite nonnegative; zero accepted but may be degenerate|
|layer_penalty_factor|5.0|Finite nonnegative, API-only|
|position_temperature|5.0|Finite nonnegative, API-only|
|class_temperature|0.0|Finite nonnegative; distinct from unsupported legacy temperature|
|speed|unset/model1.0|Finite positive when supplied; duration overrides it|
|duration|unset|Finite; nonpositive means unset; positive fixes target duration|
|denoise|true|Denoise conditioning token only applies with reference audio|
|preprocess_prompt|true|Must vary at reference upload/registration, not only generation using an existing prompt ID|
|postprocess_mode|full|full/light/off; legacy aliases accepted|
|insert_pauses|false|Generate sentence pieces then return one complete WAV; positive duration with multiple sentences is rejected|

The original float parsers accepted NaN/+Inf. `numeric-validation-before-finite-fix.json` preserves that CPU-only evidence and source hash. Root subsequently added finite guards; `numeric-validation-observations.json` is refreshed from the updated pure parser functions. This inventory did not edit application or worker product code. No unbounded/extreme numeric value is sent to a GPU by these planning scripts.

## Native/reference asset coverage

All64 successful manifested WAVs already exist locally and match their original SHA256, totaling40,358,400bytes. The separate portable `references/manifest.json` and64WAVs preserve those originals; the earlier nine-Aoede baseline profile remains unchanged.

| Native target language | Available stock voices |
|---|---:|
|ar, da, de, el, en, es, fi|8 each|
|fr|6|
|ur|2|

Across all104routes and8voices,64pairs have a native reference and768pairs explicitly use the **same selected voice's English reference**. No fallback changes voice ID silently. Every voice has an English asset. Local hashes match the previously audited deployed manifest, but earlier independent S3-body verification covered only the nine Aoede references; this task made no new production S3 requests. Original Gemini generation hardware is unverified.

ASR reference calibration must retain **each reference ID**, not overwrite eight controls into one row keyed only by language. ASR matching cannot establish speaker identity or realization of gender/age/pitch/whisper/accent. Those require listening or separately validated measurements.

## Finite, unexecuted QA plan

- `all104-all8voices-and-config-smoke.json`:2496clone cases (104routes×8voices×3real product presets), plus104auto and104fixed-vocabulary design cases, **2704total**. Every route retains exact TTS/ASR script mapping, source/license and native-versus-English reference caveats.
- `pairwise-and-boundary-matrix.json`:**130cases**:36generation-control pairwise cases,60fixed-attribute pairwise cases across12route/script representatives,22English-accent/Chinese-dialect cases, and12API-only numeric boundaries. Every unordered factor/value pair is verified by the builder. This does not exhaust a continuous parameter space or prove language-family quality.
- Numeric/boolean cases concatenate two existing published corpus excerpts, retaining both source records, so `insert_pauses` has multiple sentences to exercise. No translation is invented.
- Prompt-preprocessing variants require direct upload or separately registered prompts whose registration identity includes the preprocessing flag. A changed flag on an old prompt ID is insufficient.
- `validation-only-matrix.json`:**28expected400cases**, with literal Form strings for `nan`/`inf`/`-inf`, invalid numeric domains and unsupported/conflicting instructions. These are separate validation denominators, not throughput successes, audio-quality observations or availability failures. Numeric guards should reject before generation; fixed-vocabulary model validation can reach the GPU-owner thread before rejecting.

V4 currently hardcodes several settings. Coordinate these files with the separate V5 runner before execution: `worker_options` must be forwarded and logged exactly; effective identity must include every applied field, reference WAV hash/transcript/preprocessing and actual voice/language. Fresh generated audio, coalesced duplicates and cache hits remain separate evidence. No existing frozen timing run should be modified to adopt this matrix.

No request in these files has been run merely by building the plan. Quality gates must still check decoded signal, lexical content, numbers/negations where present, before/after audio, reference identity and native review gaps. Accepted controls do not prove good sound, and ASR flags are not automatic TTS defects.

## Separate SDK component and reference-only cache audit

The main809cfsource inventory above remains an as-inspected snapshot. A separate SDK component, `codex/max-concurency-sdk` commit`ed1afae9640a1d2fd54fac39fbd03f1b715ef053`, adds only explicit instruction/CFG forwarding, with clone identity/defaults preserved and nonfinite CFG rejected before serialization. Its33offline payload/config tests, full TypeScript check and repository lint passed. That separate branch is not a claim that production or this main checkout was deployed or modified.

`reference-cache-audit.json` contains a further **CPU-only, read-only identity audit**. On the2496clone fixture offers, the current backend reference-cache schema produces2496keys because it includes target language and generation-only options (steps/CFG/speed/postprocessing). The actual `/clone-prompts` registration inputs are reference audio, reference transcript and preprocessing; its loaded model/tokenizer supplies the preparation namespace. A proposed preparation-only logical key reduces this finite fixture set to64identities **per worker/model/preprocessing namespace**. The39×ratio is key cardinality, not a measured latency or throughput speedup.

The worker's direct-upload/cache-warm prepared-prompt cache already uses audio SHA/transcript/preprocessing. However, `/clone-prompts` currently bypasses that cache and directly creates/registers a new prompt; backend registration misses use this endpoint. The backend warmup dictionary lock protects individual get/set operations, not an entire concurrent miss/registration. A separate change could therefore combine a versioned reference-only identity with bounded registration single-flight and worker/model/process scope. It must keep prompt IDs local to their worker, preserve404re-registration and cancellation/shutdown safety, and never narrow the complete-response key that includes text, target language, instruction and full generation settings. No cache implementation was changed by this audit.
