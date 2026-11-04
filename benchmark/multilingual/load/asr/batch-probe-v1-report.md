NVIDIA GeForce RTX 4070, physical VRAM12,282MiB and PyTorch-visible VRAM12,450,070,528bytes. These are GPU ASR measurements, not TTS latency. Evidence comes from each configuration's `physical_gpu_nvidia_smi` and `torch_visible_gpu` metadata. The run used driver580.95.05, CUDA12.8, Torch2.8.0+cu128, BF16 and official Omnilingual pipeline defaults on2026-11-04. Only the current ASR worker was reported on this GPU; TTS was stopped.

No batching configuration is promoted. CTC batching was faster but changed transcripts. LLM batch2 offered only a small speedup and also changed transcripts. LLM batch4 did not complete within the finite overall budget. Current quality comparisons should keep the original batch1 configuration for each judge.

The fixed fixture contains41 verified WAVs:32 TTS outputs plus9 original reference controls,29 languages,25 script tags and285.179 seconds of audio. Selection was fixed by provenance, script, language and duration before looking at these ASR outputs. Expected words were never supplied to either model. Each configuration loaded a fresh model; no prior transcript reuse occurred.

| RTX4070 judge | Batch | Complete inputs | Transcribe call wall time | Speed versus own batch1 | Peak allocated/reserved GiB | Changed normalized transcripts versus own batch1 |
|---|---:|---:|---:|---:|---:|---:|
| CTC1Bv2 |1|41/41|9.707s|1.000×|2.320/2.525|Reference configuration|
| CTC1Bv2 |2|41/41|7.598s|1.278×|2.786/2.992|3/41|
| CTC1Bv2 |4|41/41|4.667s|2.080×|3.161/3.428|5/41|
| LLMUnlimited1Bv2 |1|41/41|124.619s|1.000×|4.674/4.906|Reference configuration|
| LLMUnlimited1Bv2 |2|41/41|120.206s|1.037×|4.945/5.174|2/41|
| LLMUnlimited1Bv2 |4|31/41|Incomplete|Unmeasured|No final peak snapshot|1/31 observed;10 missing|

These times sum synchronous pipeline transcribe calls. They exclude model loading, preprocessing, metadata writes and WAV retention. Per-row `asr_wall_s` divides a call's wall time among its chunks; it is not an independently measured clip latency. Batch scheduling used duration bins3/6/12/30seconds, up to32 chunks per pipeline call, controls first and mixed languages. Decoded text length was not used for scheduling.

CTC batch2 differences occurred in Hebrew, Sinhala and Tigrinya; batch4 additionally changed Telugu and Gujarati. LLM batch2 changed Cantonese and Hebrew; the partial batch4 changed one Arabic control. These are judge changes for identical input audio. They do not establish that a TTS output changed or that batch1 is always correct. Existing CTC examples already include Romanization or apparent wrong-language decoding, so transcript parity alone cannot certify accuracy.

The controller released the GPU at08:14:56UTC after603.247seconds including setup and model loads. The frozen comparator labels absent `outcome.json` as `not_executed`; that label is misleading for LLM batch4, which retained31 transcripts before the controller killed it at the overall deadline. The derived JSON report records `partial_overall_deadline` without altering the raw comparator. No completed batch4 LLM speed or all-language accuracy claim is available.

Every configuration has a separate `asr_protocol_config_id` in metadata and rows. Their outputs must not be mixed into one quality comparison or silently substituted for legacy batch1 results. Both original709-case judge outputs and their evaluator/controller hashes were unchanged. The prototype's GPU lease ended before architecture's later TTS/codec work.

Evidence is retained in [the metadata/raw archive](batch-probe-v1-metadata.tar), [the extracted records](batch-probe-v1-retained/ASR_BATCH_PROBE_COMPLETE.json), and [the derived report including all protocol IDs and transcript differences](batch-probe-v1-report.json). Original WAVs remain in their before/after TTS/control locations; the archive intentionally omits duplicate WAV copies.
