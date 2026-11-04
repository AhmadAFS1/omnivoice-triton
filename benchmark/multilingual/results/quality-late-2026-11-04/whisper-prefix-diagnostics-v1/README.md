# Third-family ASR diagnostic, variable-duration references

Offline, already-cached MLX Whisper large-v3-turbo ran on Apple M5 Pro integrated
Metal, with no paid API or NVIDIA rental access. All model, source, runtime and
device metadata are pinned in `metadata.json`.

This deliberately selected subset of earlier word/speaker flags has 40 pairs;
12 unsupported-language pairs are skipped explicitly. The remaining 28 pairs
contain 52 unique WAVs/56 logical transcripts and completed in approximately
25.3 seconds. Transcription receives no expected script; known language is
supplied only where supported. The expected script is used for scoring afterward.

The selected Arabic-Sudanese full-reference output matches under Whisper, while
the ten/seventeen-word variable-duration crops have CER0.667/0.417. This provides
an independent word-risk signal; it does not establish that the Meta LLM's
repeated transcript was literally spoken. The selected Arabic-Egyptian
before/after outputs are identical under Whisper at CER0.0435 despite Meta flags.

These are diagnostic disagreements, not a population failure rate, coverage of
all Lingua languages, native listening, MOS or speaker-identity certification.
The fixed-duration follow-up remains a separate immutable result directory.
