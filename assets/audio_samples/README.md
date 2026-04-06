# Audio Samples

Pre-generated audio samples demonstrating each OmniVoice runner mode.

## Directory Structure

```
assets/audio_samples/
├── base/               # BaseRunner (standard PyTorch)
├── triton/             # TritonRunner (Triton kernel fusion)
├── triton_sage/        # TritonRunner + SageAttention
├── faster/             # FasterRunner (CUDA Graph)
├── hybrid/             # TritonFasterRunner (Triton + CUDA Graph)
├── hybrid_sage/        # TritonFasterRunner + SageAttention
└── metadata.json       # Generation metadata and timing info
```

Each directory contains 9 samples: 3 Korean, 3 English, 3 Chinese.

## Sample Types

| Type | Description |
|------|-------------|
| `ko_*.wav` | Korean text samples |
| `en_*.wav` | English text samples |
| `zh_*.wav` | Chinese text samples |

## Regenerating Samples

```bash
# All modes
uv run python scripts/generate_samples.py

# Specific modes only
uv run python scripts/generate_samples.py --modes base triton

# Custom output directory
uv run python scripts/generate_samples.py --output-dir /tmp/my_samples
```

All audio is saved as 32-bit float WAV at 24 kHz.
