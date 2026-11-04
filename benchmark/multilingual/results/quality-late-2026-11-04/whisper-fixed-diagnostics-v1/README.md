# Third-family ASR diagnostic, fixed-duration references

This offline check uses already-cached MLX Whisper large-v3-turbo on the Apple M5
Pro integrated Metal GPU. It makes no paid API call and does not occupy either
NVIDIA rental. Metadata pins model revision, weight SHA, packages, settings,
source SHA and measured device memory. The runner differs from the earlier
variable-duration diagnostic only in two recorded input/output path literals.

Forty diagnostic pairs were selected from the earlier joint word/speaker flags,
then joined to the fixed-duration WAVs by exact original case/arm. Twelve pairs
have no supported Whisper language mapping and are explicitly skipped rather
than assigned a proxy. The remaining 28 pairs contain 52 unique WAVs and 56
logical transcripts, completed in 26.627 seconds. No expected text was supplied
to the transcription model. Scoring uses the expected script afterward.

| Selected pairs | CER worse | CER better | Same |
|---|---:|---:|---:|
| Ten-word crop, n=14 | 1 | 7 | 6 |
| Seventeen-word crop, n=14 | 5 | 4 | 5 |

For the selected Arabic-Sudanese case, both fixed-duration crops and the full
reference match the script under this judge. The earlier variable-duration
crops had CER 0.667/0.417 against full-reference zero. The Arabic-Egyptian case
has identical before/after Whisper output at CER0.0435 in both experiments.
Original Meta disagreements remain evidence; a third judge does not establish
which transcript is correct without listening.

This is a score-selected diagnostic, not a representative language sweep or
native/MOS/speaker certification. `selection.json` retains every inclusion and
unsupported-language skip; `paired-summary.json` retains all transcripts and
paired scoring. Earlier results remain separate and unchanged.
