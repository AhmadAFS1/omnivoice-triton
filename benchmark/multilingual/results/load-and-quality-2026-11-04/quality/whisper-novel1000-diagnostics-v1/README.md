# Third-family check of concurrent B1 word-match changes

This offline diagnostic selects **every 109 both-Meta exactness flip** between
the earlier c79 hybrid 1,000-request run and the new eager B1 run: 57 losses and
52 gains. It is a purposive follow-up, not a random sample or population failure
rate. Exact original case, waveform SHA, language/script and expected Unicode
text were verified before selection. Neither expected text nor Meta hypotheses
are supplied to the transcription model.

The cached MLX Whisper large-v3-turbo model ran on the Apple M5 Pro integrated
Metal GPU. It completed 184 unique WAVs for 92 supported pairs across 47 selected
routes in 76.839 seconds, with peak allocated memory 2,555,957,138 bytes. The
other 17 pairs have unsupported language hints and remain explicit skips; no
proxy language was substituted. No inference API, model download or NVIDIA
rental access was used. Metadata pins the existing model revision, weight SHA,
packages, settings and unchanged transcription implementation; its runner differs
from the earlier diagnostic only in two recorded input/output path literals.

| Meta-selected group | Supported pairs | Whisper CER worse | Better | Same |
|---|---:|---:|---:|---:|
| Lost a prior both-Meta exact match | 48 | 20 | 9 | 19 |
| Gained a both-Meta exact match | 44 | 7 | 13 | 24 |

For the 48 supported Meta-loss pairs, Whisper loses zero CER on 11 and gains it
on one; 15 pairs have exactly the same Whisper transcript before and after.
For the 44 supported Meta-gain pairs, Whisper gains zero CER on eight and loses
it on none; 22 pairs have identical Whisper transcripts. Thus the third family
supplies additional word-risk evidence for some cases and disagreement for
others. It does not turn every Meta flag into a proved speech defect, nor clear
the faster concurrent configuration as universally quality-preserving.

`paired-summary.json` preserves every expected/transcribed comparison and body
SHA. `selection.json` includes all supported mappings and unsupported-pair skips;
`transcripts.jsonl` retains raw model output. Arabic regional routes use the
family hint `ar`, with no dialect-specific recognition certification. The
historical c79 and B1 runs also differ in runner/KV/source/batching/scheduling
and RNG consumption despite matching semantic inputs, references and startup
seed. These diagnostics do not isolate the causal source of a word change.

ASR can mistranscribe. This is not native listening, MOS or speaker-identity
certification, and it does not qualify every utterance in any selected language.
