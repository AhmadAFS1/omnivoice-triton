# API Server

`omnivoice-triton` now includes a FastAPI server backed by this repo's runner API.

It supports:

- `auto`: text-only TTS
- `design`: text + `instruct`
- `clone`: text + uploaded reference audio, with optional `ref_text`
- clone-mode optional `instruct`, matching the upstream Gradio demo

Compared with the upstream `omnivoice-demo` Gradio UI:

- all Gradio generation controls are supported by the API
- `design` mode also allows an empty `instruct`, matching the Gradio fallback
  behavior where no selected voice attributes becomes plain text-only generation
- the API additionally exposes advanced decoding fields not shown in Gradio:
  `t_shift`, `layer_penalty_factor`, `position_temperature`,
  and `class_temperature`

## Launch

Recommended startup on GPU:

```bash
cd /workspace/omnivoice-triton
.venv/bin/omnivoice-api --runner hybrid --device cuda --ip 0.0.0.0 --port 8002
```

Other useful flags:

- `--model k2-fsa/OmniVoice`
- `--runner base|triton|faster|hybrid`
- `--dtype bf16|fp16|fp32`
- `--sage-attention`
- `--no-asr`
- `--save-dir ./generated_wavs`
- `--root-path /your/proxy/path`

## Endpoints

### `GET /health`

Returns runtime metadata:

- model checkpoint
- runner
- device
- dtype
- sample rate
- ASR status
- save-dir status
- SageAttention status

### `GET /languages`

Returns supported language IDs and display names from upstream OmniVoice.

### `POST /generate`

Accepts multipart form data and returns `audio/wav`.

Common fields:

- `mode`
- `text`
- `language`
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

Mode-specific fields:

- `design`: optionally accepts `instruct`
- `clone`: requires `ref_audio`, optionally accepts `ref_text` and `instruct`

## Example Requests

Auto:

```bash
curl -X POST http://127.0.0.1:8002/generate \
  -F mode=auto \
  -F text='Hello from omnivoice-triton'
```

Design:

```bash
curl -X POST http://127.0.0.1:8002/generate \
  -F mode=design \
  -F text='Hello from omnivoice-triton' \
  -F instruct='female, young adult, high pitch'
```

Clone:

```bash
curl -X POST http://127.0.0.1:8002/generate \
  -F mode=clone \
  -F text='Hello from omnivoice-triton' \
  -F ref_audio=@ref.wav \
  -F ref_text='Hello from omnivoice-triton'
```

Clone with extra style control:

```bash
curl -X POST http://127.0.0.1:8002/generate \
  -F mode=clone \
  -F text='Hello from omnivoice-triton' \
  -F ref_audio=@ref.wav \
  -F instruct='whisper'
```

Design with no attributes, matching Gradio's "nothing selected" behavior:

```bash
curl -X POST http://127.0.0.1:8002/generate \
  -F mode=design \
  -F text='Hello from omnivoice-triton'
```

## Response Headers

Successful `/generate` responses include:

- `X-OmniVoice-Request-Id`
- `X-OmniVoice-Started-At`
- `X-OmniVoice-Finished-At`
- `X-OmniVoice-Latency-Ms`
- `X-OmniVoice-Audio-Duration-S`
- `X-OmniVoice-RTF` when audio duration is non-zero
- `X-OmniVoice-Runner`
- `X-OmniVoice-Device`
- `X-OmniVoice-Peak-Vram-Gb`
- `X-OmniVoice-Saved-Path` when `--save-dir` is enabled

## Notes

- One runner instance is loaded per server process.
- Requests are serialized with a process-local lock for safe inference.
- `hybrid` is the default runner because it gives the best speed on this repo's CUDA path.
