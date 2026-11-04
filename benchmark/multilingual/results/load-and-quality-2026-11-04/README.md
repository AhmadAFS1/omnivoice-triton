# Complete-audio load evidence

Read the [results and scaling report](../../../../docs/max-concurrency-results-2026-11-04.md)
for workload, hardware, quality and comparison limits. Each run preserves all
offers, full-body latency, exact source/config/package/seed metadata and failures.
Cached/coalesced traffic and fresh synthesis must remain separate. Rejections,
missing/invalid audio and protocol errors never count as completed speech.

Large JSON/JSONL/CSV/log files use lossless gzip. `publication-provenance.json`
records both saved and decompressed-original SHA256 values; decompression is
independently verified during packaging. Original absolute paths remain source
provenance; generated-audio hashes link the separate sample manifests or the
complete locally retained WAVs. Weights, conditioning audio and keys are excluded.

The raw `server-process-limits.txt` files preserve `/proc/PID/limits` byte for
byte, including column-alignment trailing spaces. CSV line endings and these
raw-output bytes are retained intentionally for their recorded hashes.

The harness copies are archival source snapshots with pinned helper/input hashes,
not self-contained deployments. Reproduction needs the recorded model assets,
matching reference WAVs, runtime and full fixture tree. ASR and signal checks do
not certify native speech quality, MOS or speaker identity.
