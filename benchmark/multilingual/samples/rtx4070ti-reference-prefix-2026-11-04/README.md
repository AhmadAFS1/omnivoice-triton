# Reference-prefix samples, RTX 4070 Ti, 2026-11-04

These 396 complete generated WAVs preserve all three experimental arms: original
full conditioning, a ten-word English prefix, and a seventeen-word prefix. Each
arm has 132 cases: all 104 routes in Aoede plus all eight stock voices for Arabic,
English, Spanish and Hindi. There are 110 English-conditioning changes and 22
unchanged native-reference controls per candidate arm. Conditioning WAVs and
model weights are not included.

Use `manifest.json` to match the same `comparison_original_case_id` across the
three folders; every body SHA is verified. The target text and its FLORES-200 or
other published source/license attribution accompany every sample. Existing
fixture/source caveats apply, including dialect mappings and ASR proxies.

The shorter references change model conditioning and estimated output length.
They improve latency in this sequential screen but retain word and relative
speaker flags. **Neither arm is promoted to a default.** A separate fixed-target
follow-up was prepared to hold output length constant; it is not these samples.

Read [joint word/speaker review](../../results/quality-2026-11-04/speaker-word-prefix396-joint-v1/joint-report.md)
and [latency/integrity audit](../../results/quality-2026-11-04/reference-prefix-analysis-v1/report.md).
Raw blind Meta transcripts and complete protocol metadata are in the adjacent
`results/quality-2026-11-04/blind-ASR432/` package. No expected text was supplied
to transcription. Known output script was supplied where required by the model.
ASR and ECAPA similarity are diagnostic proxies, not native listening or MOS.
