# API Server

This document explains the FastAPI server implemented in
[src/omnivoice_triton/cli/api_server.py](/workspace/omnivoice-triton/src/omnivoice_triton/cli/api_server.py).

The server wraps this repo's runner API and exposes a simple HTTP interface
for OmniVoice generation through the optimized `omnivoice-triton` backends.

It is intended for:

- local testing
- simple app integration
- Vast.ai deployment
- measuring latency and real-time factor
- comparing `base`, `triton`, `faster`, and `hybrid`

It is not a high-concurrency serving stack. It uses one in-process runner and
serializes inference with a process-local lock.

## Overview

The server is a thin HTTP layer around:

- `omnivoice_triton.create_runner(...)`
- `runner.load_model()`
- `runner.model.generate(...)`
- `runner.model.create_voice_clone_prompt(...)`

It supports three generation modes:

- `auto`: text-only TTS
- `design`: text plus optional `instruct`
- `clone`: text plus uploaded reference audio, with optional `ref_text`

Successful `/generate` responses return inline `audio/wav`.

## Entry Point

The CLI entrypoint is:

```bash
omnivoice-api
```

It is registered in [pyproject.toml](/workspace/omnivoice-triton/pyproject.toml).

You can also run it as a module:

```bash
cd /workspace/omnivoice-triton
.venv/bin/python -m omnivoice_triton.cli.api_server --runner hybrid --device cuda --ip 0.0.0.0 --port 8002
```

## Startup Behavior

At startup, `create_app(...)`:

1. picks the requested device or auto-detects one
2. creates the requested runner via `create_runner(...)`
3. loads the OmniVoice model into that runner
4. optionally loads ASR at startup unless `--no-asr` is used
5. stores the runner and runtime settings in `app.state`

Shutdown unloads the runner and frees GPU memory.

## Runner Modes

Unlike the upstream `omnivoice-demo` Gradio app, this server can run the
optimized backends from this repo.

Available runners:

- `base`: standard PyTorch OmniVoice
- `triton`: Triton kernel fusion
- `faster`: CUDA Graph wrapping
- `hybrid`: Triton + CUDA Graph

Default:

- `hybrid`

Practical recommendation:

- use `hybrid` on CUDA unless you are debugging correctness
- use `base` if you need a non-CUDA fallback

Important device constraint:

- `triton`, `faster`, and `hybrid` require CUDA
- `base` is the only safe fallback for non-CUDA devices

## CLI Options

Main runtime flags:

- `--model`: checkpoint path or Hugging Face repo ID
- `--runner`: `base|triton|faster|hybrid`
- `--device`: explicit device such as `cuda`, `cuda:0`, `cpu`, or `mps`
- `--dtype`: `bf16|fp16|fp32`
- `--ip`: bind address
- `--port`: bind port
- `--root-path`: reverse-proxy root path
- `--no-asr`: skip loading Whisper ASR at startup
- `--save-dir`: persist a copy of each generated WAV
- `--sage-attention`: enable SageAttention on `triton` or `hybrid`

Typical startup:

```bash
cd /workspace/omnivoice-triton
.venv/bin/omnivoice-api --runner hybrid --device cuda --ip 0.0.0.0 --port 8002
```

Faster startup if you always supply `ref_text` for clone mode:

```bash
cd /workspace/omnivoice-triton
.venv/bin/omnivoice-api --runner hybrid --device cuda --no-asr --ip 0.0.0.0 --port 8002
```

Save generated files locally too:

```bash
cd /workspace/omnivoice-triton
.venv/bin/omnivoice-api --runner hybrid --device cuda --port 8002 --save-dir ./generated_wavs
```

## Endpoints

### `GET /health`

Returns runtime status:

- model checkpoint
- runner
- device
- dtype
- sample rate
- whether startup ASR loading was requested
- whether ASR is currently loaded
- whether a save directory is configured
- whether SageAttention is enabled

Example:

```bash
curl http://127.0.0.1:8002/health
```

Typical response:

```json
{
  "status": "ok",
  "model": "k2-fsa/OmniVoice",
  "runner": "hybrid",
  "device": "cuda",
  "dtype": "fp16",
  "sample_rate": 24000,
  "load_asr": false,
  "asr_loaded": false,
  "save_dir": null,
  "sage_attention": false
}
```

### `GET /languages`

Returns the supported language list derived from upstream OmniVoice language
metadata.

Each item includes:

- `id`
- `name`
- `display_name`

Example:

```bash
curl http://127.0.0.1:8002/languages
```

### `POST /generate`

Generates speech and returns a WAV file.

The endpoint expects multipart form data.

Response:

- HTTP `200`
- `Content-Type: audio/wav`
- inline WAV bytes in the response body

## Gradio Parity

This API was checked against the upstream Gradio UI in:

- [.venv/lib/python3.12/site-packages/omnivoice/cli/demo.py](/workspace/omnivoice-triton/.venv/lib/python3.12/site-packages/omnivoice/cli/demo.py)

The API matches the Gradio generation controls for:

- `text`
- `language`
- `instruct`
- `ref_audio`
- `ref_text`
- `num_step`
- `guidance_scale`
- `denoise`
- `speed`
- `duration`
- `preprocess_prompt`
- `postprocess_output`

Behavior parity notes:

- `design` mode allows an empty `instruct`, matching Gradio's fallback behavior
  when no voice-design attributes are selected
- `clone` mode allows optional `instruct`, matching the Gradio clone tab
- `duration <= 0` is treated as unset, matching Gradio's behavior
- `language="Auto"` or blank becomes `None`

Intentional API-only extras:

- `auto` mode is exposed directly
- advanced decoding parameters are accepted even though Gradio does not show
  them in the UI:
  - `t_shift`
  - `layer_penalty_factor`
  - `position_temperature`
  - `class_temperature`

One structural difference:

- Gradio has separate dropdowns for voice-design categories
- the HTTP API accepts the final assembled `instruct` string instead

This is equivalent to what the Gradio app eventually passes into
`model.generate(...)`.

## Request Modes

### `mode=auto`

Text-only generation.

Allowed:

- `text`
- optional `language`
- optional generation settings

Rejected:

- `instruct`
- `ref_audio`
- `ref_text`

Example:

```bash
curl -X POST http://127.0.0.1:8002/generate \
  -F mode=auto \
  -F text='Hello from omnivoice-triton' \
  -o auto.wav
```

### `mode=design`

Voice design from instruction text.

Allowed:

- `text`
- optional `instruct`
- optional `language`
- optional generation settings

Rejected:

- `ref_audio`
- `ref_text`

Important behavior:

- if `instruct` is omitted or blank, generation falls back to text-only style
  conditioning, matching the Gradio design tab when no attributes are selected

Example with explicit design attributes:

```bash
curl -X POST http://127.0.0.1:8002/generate \
  -F mode=design \
  -F text='Hello from omnivoice-triton' \
  -F instruct='female, young adult, high pitch' \
  -o design.wav
```

Example with no selected attributes:

```bash
curl -X POST http://127.0.0.1:8002/generate \
  -F mode=design \
  -F text='Hello from omnivoice-triton' \
  -o design_auto_like.wav
```

### `mode=clone`

Voice cloning from a reference audio clip.

Required:

- `text`
- `ref_audio`

Optional:

- `ref_text`
- `instruct`
- `language`
- generation settings

Behavior:

- the uploaded file is written to a temporary file
- the server calls `model.create_voice_clone_prompt(...)`
- the resulting prompt is passed into `model.generate(...)`
- if `ref_text` is omitted, OmniVoice may use ASR to auto-transcribe
- if `--no-asr` was used, ASR is still allowed to load lazily on first clone
  request without `ref_text`; startup is faster, first such request is slower

Example:

```bash
curl -X POST http://127.0.0.1:8002/generate \
  -F mode=clone \
  -F text='Hello from omnivoice-triton' \
  -F ref_audio=@/full/path/to/ref.wav \
  -F ref_text='Hello from omnivoice-triton' \
  -o clone.wav
```

Clone with extra style control:

```bash
curl -X POST http://127.0.0.1:8002/generate \
  -F mode=clone \
  -F text='Hello from omnivoice-triton' \
  -F ref_audio=@/full/path/to/ref.wav \
  -F ref_text='Hello from omnivoice-triton' \
  -F instruct='whisper' \
  -o clone_whisper.wav
```

## Request Fields

`/generate` accepts these form fields:

- `mode`
- `text`
- `language`
- `instruct`
- `ref_text`
- `num_step`
- `guidance_scale`
- `t_shift`
- `layer_penalty_factor`
- `position_temperature`
- `class_temperature`
- `speed`
- `duration`
- `denoise`
- `preprocess_prompt`
- `postprocess_output`
- `ref_audio`

### Field Semantics

`text`

- required
- blank strings are rejected

`language`

- optional
- blank, `Auto`, or `None` are normalized to `None`
- otherwise passed through to OmniVoice language resolution

`instruct`

- optional in `design`
- optional in `clone`
- rejected in `auto`

`num_step`

- default: `32`
- must be an integer greater than `0`

`guidance_scale`

- default: `2.0`
- must be `>= 0`

`t_shift`

- default: `0.1`
- must be `>= 0`

`layer_penalty_factor`

- default: `5.0`
- must be `>= 0`

`position_temperature`

- default: `5.0`
- must be `>= 0`

`class_temperature`

- default: `0.0`
- must be `>= 0`

`speed`

- optional
- must be `> 0` if provided
- forwarded only when it differs from `1.0`

`duration`

- optional
- if omitted, normal duration estimation is used
- if `<= 0`, it is treated as unset to match Gradio behavior
- if positive, it overrides duration estimation

`denoise`

- default: `true`

`preprocess_prompt`

- default: `true`
- mainly affects clone mode prompt cleanup

`postprocess_output`

- default: `true`

`ref_audio`

- required in clone mode
- ignored or rejected in the other modes

## Example Requests

### Auto Mode

```bash
cd /workspace/omnivoice-triton
curl -X POST http://127.0.0.1:8002/generate \
  -F mode=auto \
  -F text='Hello from the omnivoice triton API server.' \
  -o auto.wav
```

### Design Mode

```bash
cd /workspace/omnivoice-triton
curl -X POST http://127.0.0.1:8002/generate \
  -F mode=design \
  -F text='Hello from the omnivoice triton API server.' \
  -F instruct='female, young adult, high pitch' \
  -o design.wav
```

### Clone Mode

```bash
cd /workspace/omnivoice-triton
curl -X POST http://127.0.0.1:8002/generate \
  -F mode=clone \
  -F text='Hello from the omnivoice triton API server.' \
  -F ref_audio=@/full/path/to/reference.wav \
  -F ref_text='Hello from the omnivoice triton API server.' \
  -o clone.wav
```

### Tuned Generation Example

```bash
cd /workspace/omnivoice-triton
curl -X POST http://127.0.0.1:8002/generate \
  -F mode=design \
  -F text='This is a slower and more deliberate sample.' \
  -F instruct='male, low pitch' \
  -F num_step=24 \
  -F guidance_scale=2.0 \
  -F speed=0.9 \
  -F denoise=true \
  -F preprocess_prompt=true \
  -F postprocess_output=true \
  -o design_tuned.wav
```

### Inspect Response Headers

```bash
curl -i -X POST http://127.0.0.1:8002/generate \
  -F mode=auto \
  -F text='Header check example' \
  -o header_test.wav
```

## Listening To Output

On Linux:

```bash
ffplay -autoexit auto.wav
ffplay -autoexit design.wav
ffplay -autoexit clone.wav
```

Fallback:

```bash
aplay auto.wav
aplay design.wav
aplay clone.wav
```

## Response Headers

Successful `/generate` responses include:

- `Content-Disposition: inline; filename="omnivoice.wav"`
- `X-OmniVoice-Request-Id`
- `X-OmniVoice-Started-At`
- `X-OmniVoice-Finished-At`
- `X-OmniVoice-Latency-Ms`
- `X-OmniVoice-Audio-Duration-S`
- `X-OmniVoice-RTF` if audio duration is non-zero
- `X-OmniVoice-Runner`
- `X-OmniVoice-Device`
- `X-OmniVoice-Peak-Vram-Gb`
- `X-OmniVoice-Saved-Path` when `--save-dir` is enabled

### What `RTF` Means

`RTF` is real-time factor:

```text
RTF = generation_time_seconds / output_audio_seconds
```

Interpretation:

- `RTF < 1.0`: faster than real-time
- `RTF = 1.0`: real-time
- `RTF > 1.0`: slower than real-time

## Save Directory

If `--save-dir` is set:

- the WAV response is still returned inline
- a copy is also persisted locally
- the saved path is added to the response headers

Filename format:

```text
{timestamp}_{mode}_{slug}_{request_id}.wav
```

## Concurrency Model

This server currently uses:

- one in-process runner
- one loaded OmniVoice model
- one `threading.Lock()` around generation

That means requests are serialized within a single server process.

This is good for:

- correctness
- smoke tests
- latency testing
- simple single-GPU app hosting

It is not designed for:

- high throughput
- multi-request batching
- streaming audio

## Error Handling

The server maps common failures to HTTP responses:

- validation issues: `400`
- upstream `ValueError`: `400`
- upstream `RuntimeError`: `500`
- model not yet ready: `503`
- unexpected exceptions: `500`

## Clone Troubleshooting

The most common clone issue is not a server bug. It is a local file path issue
in `curl`.

If you see:

```text
curl: (26) Failed to open/read local data from file/application
```

that means `curl` could not open the file after `@...`.

Examples:

```bash
-F ref_audio=@reference.wav
```

only works if `reference.wav` exists in the current directory.

Safer version:

```bash
-F ref_audio=@/full/path/to/reference.wav
```

Check the file first:

```bash
ls -l /full/path/to/reference.wav
```

Common gotchas:

- the file is not in your current directory
- the filename is different from what you typed
- the path contains spaces and was not quoted correctly
- you used `~` and shell expansion did not behave the way you expected

## Recommended Testing Workflow

For first-pass validation:

1. start the server
2. hit `/health`
3. send one `auto` request
4. send one `design` request
5. send one `clone` request with a known-good local WAV
6. inspect:
   - HTTP status
   - WAV playback
   - latency headers
   - server logs

Example:

```bash
curl http://127.0.0.1:8002/health
```

```bash
curl -X POST http://127.0.0.1:8002/generate \
  -F mode=auto \
  -F text='Testing latency logging' \
  -o smoke.wav
```

Look for:

- `X-OmniVoice-Started-At`
- `X-OmniVoice-Finished-At`
- `X-OmniVoice-Latency-Ms`
- `X-OmniVoice-Audio-Duration-S`
- `X-OmniVoice-RTF`

## Current Limitations

- no request batching
- no audio streaming
- one serialized generation lock
- one runner instance per process
- clone uploads are written to temporary files before prompt construction

These are acceptable for practical testing and lightweight deployment, but not
for high-concurrency production scheduling.
