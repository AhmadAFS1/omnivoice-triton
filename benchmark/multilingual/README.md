# Multilingual complete-WAV benchmarks

The retained before/after evidence uses an **NVIDIA GeForce RTX 4070** with
**12,282 MiB physical VRAM** (`nvidia-smi`) and **12,450,070,528 bytes visible to
Torch**, driver **580.95.05**, CUDA **12.8**, Torch **2.8.0+cu128** and
OmniVoice **0.1.4**. Original inference ran on 2026-11-04 UTC. Per-run timestamps,
co-resident load, source fingerprints and actual settings are in the linked
metadata; CPU postprocessing and packaged tests ran separately on Apple M5 Pro.

- [Scaling measurements and limits](../../docs/max-concurency-rtx4070-2026-11-03.md).
- [104-language before/after WAVs](samples/rtx4070-2026-11-03/README.md): 208
  complete WAVs, all independently SHA-verified; all 104 pairs are byte-identical.
  The [paired manifest](samples/rtx4070-2026-11-03/paired-manifest.json) retains
  expected text, reference origin, source/license attribution and both ASR judges.
- [Audited stage-1 report](results/stage1/report.md),
  [per-language metrics](results/stage1/per-language.json), and
  [299 paired-case deltas](results/stage1/paired-case-deltas.jsonl): baseline
  `c79ce35` versus combined candidate `1eb820f` (unused-KV-cache disable, graph
  LRU and serving changes), and the separate split-CFG experiment. The 299-pair
  preservation result does not isolate one optimization. Split CFG remains
  unpromoted: aggregate CER improved slightly while individual paired regressions
  and baseline failure patterns still require review.
- [Inventory and published-text provenance](fixtures/README.md),
  [corpus attribution](fixtures/corpus-source-coverage.json),
  [FLORES notice](fixtures/FLORES-README.txt), and
  [dataset documentation](fixtures/dataset-readme.md): 104 product routes,
  299 quality cases and a 1,105-case unique-input load pool. FLORES-200 texts are
  CC-BY-SA-4.0; Meta Omnilingual corpus excerpts are CC-BY-4.0; the four
  Wikipedia-language excerpts retain their individual source/license notices.

The first quality profile is hybrid inference, 16 steps, speed 1,
postprocessing off, graph limit 8, codec batch 1 and startup seed 20261103.
The model revision is `c5fdb5ccb189668d56333f77ba2629f4cd7535f4`.
Both blind ASR judges are Meta Omnilingual 1B v2 cards, BF16, batch 1, on the
original RTX 4070. Their unchanged hardware/runtime metadata is bundled in
[asr-source-metadata](samples/rtx4070-2026-11-03/asr-source-metadata/manifest.json).
Later ASR batch-probe configurations are separate experiments and are not mixed
into these scores. ASR disagreement, spelling/spacing differences, uncalibrated
targets and proxy mappings remain explicit; these are screening results, not
native-language, speaker-identity or perceptual-quality certification.

The later voice/control and step-count experiments use a separate **NVIDIA
GeForce RTX 4070 Ti**, also **12,282 MiB physical**, with **12,481,134,592 bytes
Torch-visible**, driver **610.57.04**, CUDA **12.8** and an AMD Ryzen 7 5700X host.
Their frozen candidate is `377de39`, graph limit 2, unused KV cache off, split CFG
off, codec batch 1 and response cache disabled for novel generation. Those
results must keep their own hardware and source labels. Their all-voice audit is
diagnostic: the initial 404-case gate found two zero-frame default-preset outputs,
and broader coverage continues without treating HTTP 200 as word-quality approval.
Sequential latency measurements and cached/coalesced response delivery are
separate from novel simultaneous-inference capacity.

## Reference assets and coverage

**Reference WAVs are not bundled.** The audio in `samples/` is generated review
output, not the conditioning assets required to rerun cloning. Metadata-only
[original-nine](fixtures/reference-profiles/original9-manifest.json) and
[64-stock-reference](fixtures/reference-profiles/native64-manifest.json) profiles
record exact IDs, transcripts and hashes; supply matching assets with explicit
paths when rerunning the harness.

The original 104-route Aoede profile uses nine native-language references and
95 explicitly labeled English-reference fallbacks. The later eight-voice grid
has 64 native assets sparsely distributed across nine languages, not a complete
nine-by-eight rectangle: 64 of its 832 route/voice combinations have target-
language assets and 768 use the selected voice's English reference. Historical
fixture READMEs describe the initial inventory and remain source snapshots.
The [voice/control metadata](fixtures/voice-controls/README.md) and actual
per-case options distinguish the three Lingua presets from experimental worker
instructions and API boundaries. Continuous slider ranges are not exhaustively
qualified by a finite matrix.

## Tools and reproduction

[V4 load protocol](load/README.md) and [V5 voice/control protocol](load/V5-README.md)
cover complete-body timing, source/runtime assertions, retained WAVs, admission
and validation denominators. Builders cover burst/staggered loads, the 500-request
target, voice presets and the 16/8/4-step frontier. Builders generate commands for
an explicit execution task root; provide the model, source checkout and reference
assets separately. Historical absolute paths in result JSON are provenance.

[Quality tools](quality/README.md) perform CPU-only hash verification, preparation,
WER/CER scoring and exact-pair export. The packaged helper defaults to `fixtures/`
when present, with `languages/` retained as a legacy fallback. Its original source
hash and this package-only path adaptation are recorded in
[package-source-provenance.json](package-source-provenance.json). Frozen V4
`load/run.py` remains byte-identical (`df51c081…`); V5 is `1a077901…`.

```sh
python quality/test_report.py
python load/test_arrival_report.py
python load/test_telemetry.py
```

GPU ASR setup/evaluation and the bounded controller live in [load/asr](load/asr/README.md).
Use one declared judge protocol per comparison; unlabeled frozen output is the
legacy BF16/batch-1 protocol. Model weights, reference WAVs, raw full-run audio,
large archives and temporary logs remain outside this portable review package.
The 208 sample WAVs and compact audited stage-1 results retain their original
hashes; [verification.json](samples/rtx4070-2026-11-03/verification.json) and
[package-validation.json](package-validation.json) describe the local checks.
