# Finite local Omnilingual quality judging

The task's isolated ASR environment uses Torch 2.8.0+cu128, torchaudio 2.8.0+cu128,
fairseq2 0.6/fairseq2n 0.6+cu128, Arrow 20, and official Omnilingual source commit
81f51e224ce9e74b02cc2a3eaf21b2d91d743455. `ASR_ASSETS_VERIFIED.json` records the
tokenizer and both model hashes, package pins, and unchanged baseline freeze.
No API transcription service is used. ASR preparation performed no model loading
or GPU inference. Do not run ASR during TTS timing.

After the final TTS worker releases the GPU, prepare canonical logical coverage.
This is CPU file/hash work and includes every retained HTTP outcome, the 84
non-HTTP prototype WAVs under `results`, all completed micro/core/load arms, and
the nine unchanged reference controls. Hash+exact-language/script deduplication
crosses every arm, including cached logical copies; original full WAVs remain.

```sh
cd /workspace/omnivoice-max-concurency
asr-venv/bin/python quality/report.py prepare \
  --runs-root results --reference-manifest languages/references/manifest.json \
  --output quality-prepared/final

# This first command only prints a plan; it cannot load a model.
asr-venv/bin/python asr/run_bounded.py \
  --prepared quality-prepared/final --output asr-results/final \
  --budget-seconds 3300
```

Root alone releases the GPU. Append `--allow-gpu-inference` to execute. Before
each model, the controller requires no existing `nvidia-smi` compute processes.
It uses the unchanged blind `evaluate.py` worker, one model at a time: CTC first,
then LLM. The worker receives audio paths and known language/script, never the
expected transcript. Each worker records physical/visible GPU memory, driver,
runtime, packages, date, and co-load; this is separate from TTS latency evidence.

```sh
asr-venv/bin/python asr/run_bounded.py \
  --prepared quality-prepared/final --output asr-results/final \
  --budget-seconds 3300 --reserve-seconds 120 \
  --pair before-all299=candidate-all299 \
  --report-output quality-reports/final --allow-gpu-inference
```

The finite budget includes audio preparation. The controller stops only its own
ASR process group, retains complete transcript rows and original partial/error
logs, and leaves missing rows explicitly unscored. It reserves time for final
CPU reporting. Inputs are prioritized as reference controls, all299 quality
arms (short/medium/critical), prototype probes, controlled micro runs, then load
WAVs. `progress.json` includes actual completed records and mean ASR time to
calibrate the budget after the first 100; this is an estimate, not a guarantee.

The original evaluator splits ASR input above 30 seconds at quiet boundaries;
this never changes the saved complete TTS WAV. Global audio preparation happens
once; model attempts refer to those same chunks. Output directories are fresh.
To resume after a budget stop, repeat the command with `--resume`, additional
budget, and a fresh `--report-output` name. Validated published transcript rows
are reused without inference; missing inputs get a new attempt directory. The
manifest/evaluator hash, dtype and device must still match. Raw evidence is never
overwritten. Canonical reports read only validated `published` directories, and
`score-command.json` preserves the exact command, including partial coverage.

A staged quality gate can judge completed QA/probes first, release both ASR
models, resume TTS load tests, and then judge only new WAVs. Make a fresh final
canonical preparation and fresh ASR output; pass `--prior-asr-root` for stage1.
Matching prior WAV SHA/script/model evidence is imported with explicit reuse
provenance and zero fresh ASR time, with original stage1 files unchanged.
Different scripts for the same bytes are separate inputs. Stable canonical IDs
are preserved, while the exact SHA/script key controls transcript reuse.

```sh
# After all load testing finishes and releases the GPU again:
asr-venv/bin/python quality/report.py prepare \
  --runs-root results --reference-manifest languages/references/manifest.json \
  --output quality-prepared/final-v2
asr-venv/bin/python asr/run_bounded.py \
  --prepared quality-prepared/final-v2 --output asr-results/final-v2 \
  --prior-asr-root asr-results/stage1 --budget-seconds 3300 \
  --pair before-all299=candidate-all299 \
  --report-output quality-reports/final-v2 --allow-gpu-inference
```

Historical timing is a planning aid. `reference-speed-rtx3090.json` contains the
raw source paths/hashes and hardware for the previous measured RTX3090 run:
938 unique WAV/script inputs, 4,688.67 seconds of audio, 556.19 seconds of LLM
inference plus 39.71 seconds of CTC inference (about 9.93 minutes combined).
That was BF16, batch1, physical 24,576 MiB/visible 25,296,044,032 bytes,
driver 580.65.06, CUDA 12.8, with no other GPU compute process. This is not a
4070 performance measurement. Allowing 45–60 minutes for several thousand
similar unique clips leaves margin, subject to the first measured 4070 records.

Two Omnilingual variants share a family; agreement does not certify quality.
Reference controls, local Whisper disagreement checks and native listening
remain separate evidence. Twi's Akan proxy and all source/script/reference
caveats remain in canonical logical rows. Cached copies are not independent
quality observations, and unavailable mappings/judges are never passes.

CPU-only checks: four unit tests cover truncated/wrong-hash transcript
publication, exact script identity on resume, blind metadata guards, synthetic
subprocess execution/resume/expanded-stage reuse, immutable prior outputs, and
namespace/priority handling. A separate CLI
dry-run verifies it reads no audio and creates no execution directory. These
tests load no TTS/ASR model and measure no GPU performance.
