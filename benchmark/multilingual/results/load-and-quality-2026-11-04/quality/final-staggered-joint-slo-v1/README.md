# Joint words, valid audio and full-response latency

This CPU-only analysis joins every recorded offer to the completed word evidence
by original request ID, exact WAV SHA, expected Unicode text and ASR language/
script. It requires finite scheduled-arrival-to-full-body latency, including
client dispatch lateness. Rejected/invalid/missing and slow responses cannot pass.
Ten warmups are excluded from the18,330-offer cohort denominator.

| Offered RPS | Offers | Valid + both-ASR exact + ≤500ms | Valid + both-ASR zero CER + ≤500ms |
|---:|---:|---:|---:|
|0.5|30|9|11|
|5|300|2|2|
|50|3,000|0|0|
|250|15,000|0|0|

At the2second threshold, joint strict exact counts are10/30,39/300,23/3,000 and
2/15,000. The underlying run has523valid cohort WAVs and17,807 HTTP429 outcomes;
all remain in the denominator. Each accepted waveform has both frozen judges,
without protocol mixing or new inference by this analysis.

This is a conservative joint diagnostic, not native-listener, MOS, accent or
speaker-identity certification. ASR can be wrong, and language/script proxy and
reference-calibration limitations remain in the original scored records. The
phase labels assume one turn per20seconds; they do not measure connected calls.
`joined18340.jsonl` also retains the ten warmups, with no arrival-SLO claim for
those sequential setup rows. The publication compresses this file losslessly.
