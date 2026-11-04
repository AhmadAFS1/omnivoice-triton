# Targeted third-family check of raw-load ASR outliers

The five largest preselected Meta-LLM CER-variance diagnostic cases in the raw
5,000-request study were compared with their matched earlier B1 waveform. These
are five selected pairs, eight unique WAVs and ten logical transcripts, covering
two Mandarin and three Cantonese comparisons. They are not a random sample or
a population failure-rate estimate.

The already-cached MLX Whisper large-v3-turbo model ran offline on Apple M5 Pro
Metal, without an API call, model download or NVIDIA access. It completed in
4.075 seconds; pinned model/runtime/settings/source and all waveform hashes are
retained. Only audio and the supported language hint reach transcription;
expected text and Meta outputs are used afterward for scoring. The runner
differs from the previous diagnostic only in two input/output path literals.

For raw request00358, Whisper matches the Mandarin expected script exactly;
the Meta LLM emitted339characters for a12character reference, while CTC CER
was0.182. For request04358, Whisper CER is0.455 versus Meta LLM27.0/CTC2.636.
Thus long ASR repetitions cannot automatically be treated as literal spoken
repetitions. The three Cantonese comparisons include romanized/mixed-script
Whisper output, making exact Unicode CER difficult to interpret as speech
correctness. Their original hypotheses and scripts remain unchanged.

Both Mandarin baseline Whisper outputs match the script, while one raw output
does and the other does not. The remaining three pairs require review; no new
global language, native-pronunciation or quality clearance follows. This check
does not isolate load as the cause of any change: startup seed is shared but
per-request RNG position/scheduling differs. Read `scored-logical.json` and the
original `top5-LLM-CER-diagnostics.json` for the exact paired evidence.
