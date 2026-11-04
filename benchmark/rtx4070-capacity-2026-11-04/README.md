# RTX 4070 cloned-voice capacity evidence — November 4, 2026

**Actual GPU: NVIDIA GeForce RTX 4070, 12,282 MiB physical VRAM
(`nvidia-smi`), 12,536,250,368 bytes PyTorch-visible.** Driver 615.71.09,
CUDA runtime 12.8, 200 W. All TTS experiments in this bundle use this card.
The ASR diagnostics use an Apple M5 Pro integrated GPU with shared RAM;
auditing, hashing and reporting are CPU-only.

Read the [full report](../../docs/rtx4070-clone-capacity-2026-11-04.md).
The product source is pinned to `096946d75e2ddcb2797c375b0ee50a1933793b36`
on `max-concurency`. Response caching is disabled. Every timed input is a
cloned-voice request with a full reference, 16 steps and a complete-WAV response.

The ordinary single-server B1 screen reaches approximately 2.25 complete
requests/s; two ordinary servers about 2.18/s; two verified CUDA MPS clients
about 2.55/s. One MPS server is approximately 2.24/s. There are 21 phases,
1,260/1,260 valid timed WAV responses and 420 retained warmups. These short
finite screens are not fleet, native-quality or p95-under-500-ms certification.

## Files

- [Capacity CSV](derived/capacity.csv) and [derived audit](derived/analysis.json):
  all phases, timing distributions, one-second completion counts, source-level
  execution counters, aligned hardware samples and kernel attribution.
- [Word comparison](derived/word-comparison.json): 1,071 logical Whisper screens,
  655 unique waveform/language inputs and 189 explicit unsupported-language
  skips. This experiment does not rerun Omnilingual. It includes the 60/60
  byte-identical single-server MPS control and mixed two-server word flags.
- [Samples](samples/manifest.json): 18 selected before/after full-reference
  generated samples and 18 profiler samples. The before/after deployment
  comparison uses different process RNG streams; matching words or identity
  are not assumed. Expected scripts, source attribution/license, voice-reference
  metadata and request settings are retained. Conditioning audio is excluded.
- `raw/results`: ordinary one-server B1/B8 arms; `raw/dual`: ordinary two-server
  arms; `raw/dual-mps`: two-server MPS arms and verified actual TTS-client PIDs;
  `raw/single-mps`: the one-server MPS control.
- `raw/profile`: actual CUDA traces, observed linear shapes, unprofiled
  measurements, kernel tables and instrumentation parity. Tracing adds overhead;
  its timing is not HTTP capacity. Dtype-corrected derived shape estimates repair
  the raw helper's two-byte assumption for FP32 codec projections.
- `raw/mps-control`: scoped MPS setup/probe/client/quit receipts and CPU/RSS
  samples. `raw/ncu.csv` retains the denied hardware-counter probe; no host
  permission changes were attempted. `raw/startup-healthz-attempt` preserves
  the stopped startup-only harness error, before any generation.
- [Runtime provenance](raw/provenance.json), [verified backup](derived/BACKUP_VERIFIED.json.gz),
  [rental cleanup](cleanup.json), and [publication manifest](publication-manifest.json).
  All 2,025 remote backup files/376,782,346 bytes were independently matched
  before deletion. All 24 product Python sources match the pinned fork.
- Frozen executed benchmark wrappers are in `raw`, including the initial
  single-server/no-MPS variants. `raw/source.tar.gz` holds the pinned product
  source/lock/fixtures with its original LICENSE; it contains no checkpoint
  weights. Additional CPU audit/ASR/publication code is in `scripts`.

Every full output WAV remains in:
`/Users/ahmadsmacair/code/lingua/lingua/work/omnivoice-4070-rps-2026-11-04/retained`.
Only representative audio is duplicated in Git. Raw request paths identify the
corresponding local phase folders; a WAV path in raw metadata is not a promise
that every workload WAV is published in this repository.

## Lossless compression and integrity

Large JSON/JSONL/CSV/log files are stored with `.gz` appended. Their original
and saved byte lengths and SHA-256 hashes are in `publication-manifest.json`.
Decompression restores the original bytes, including line endings. The manifest
also covers selected WAV files. The small generated CSV, sample manifest,
cleanup receipt and README are auxiliary deliverables outside its source-copy
entry list.

Example inspection:

```bash
gzip -cd raw/profile/B1-trace.json.gz > /tmp/rtx4070-B1-trace.json
gzip -cd raw/dual-mps/batch1/dual-b1-c8/requests.json.gz
```

Use the first file in Chrome/Perfetto trace viewing. The JSON phase summaries
remain small uncompressed files. The report explains profiler overhead and
the distinction between GPU busy time, VRAM capacity and bandwidth utilization.

## Reproduction

Reproduction requires a newly authorized GPU allocation; the recorded rental
has already been destroyed. Copy `raw/source.tar.gz`, the frozen wrappers,
`fixtures/corpus.json` and full conditioning clips from the local archive into
the task layout `/workspace/omnivoice-4070-rps`. The conditioning manifest is
[reference-metadata.json](fixtures/reference-metadata.json); verify every clip SHA.
`raw/bootstrap.sh` installs the locked source environment and downloads the
pinned OmniVoice checkpoint. Each `command.json` records exact API flags.

From the prepared task root:

```bash
export HF_HOME=/workspace/omnivoice-4070-rps/hf-cache
export TOKENIZERS_PARALLELISM=false
source/.venv/bin/python run_capacity_single_executed.py
source/.venv/bin/python profile_pipeline.py
source/.venv/bin/python run_dual_no_mps_executed.py
source/.venv/bin/python run_mps.py
```

The scripts refuse to overwrite existing run directories. The MPS controller
also needs the matching final `run_dual.py` and `run_capacity.py` helpers, checks
both TTS clients, and quits its private daemon in `finally`. Its recorded GPU
UUID guard must be consciously updated for a new owned rental. Do not point it
at an unrelated live GPU. Cache-off health assertions and member-counter audits
are part of the evidence, not optional replacements with cached-response rates.

ASR scripts require the existing local MLX model snapshot recorded in their
metadata and never download models or call an inference API. They are optional
recognition diagnostics, not a substitute for the full Omnilingual/native
quality protocol in the earlier multilingual work.
