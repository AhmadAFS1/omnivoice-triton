# Shared graph pool diagnostic: incomplete, private reference only

The shared-pool candidate and its alias tests never executed. This run establishes
no GPU correctness, memory saving or speedup for shared pools. It changed no
product code or serving default.

Root authorized one finite, last 300-second experiment on the primary RTX 4070.
The supervisor began at 14:55:22 UTC on 4 November 2026. It stopped its child at
15:00:02 during the private-reference AR B8 warmed generation. The child recorded
the controlled KeyboardInterrupt and exited without CUDA cleanup or retry; the
supervisor finished at 15:00:03 after 281.08 seconds with an empty GPU compute
list. An independent check at 15:01:32 also found the GPU idle.

Five private jobs completed: EN/ES/AR B1 and EN/ES B8. The private AR B8 parity
output was saved, but its warmed repeat did not complete. There are 46 saved
WAVs, 27 unique hashes and 22 NPZ files. Every saved WAV was independently
rehashed locally and matched the prior 14,546-input canonical set by exact WAV
SHA, ASR language/script and expected text. Thus this run requires zero new ASR
inputs. This reuses existing audio evidence; it does not qualify the unexecuted
shared arm.

The exact prototype was
`19c498786620bc4dc828520af0feb61eae06c18b18f0b2b9a675aa860718ffc8`,
with probe
`1af5a0687b2579503380afeb219ae6a860e6be405a0b14f6b6dcb2c0b84e63f7`.
Both match the current 26-contract CPU snapshot. Those CPU contracts test fake
ownership and driver behavior; they do not substitute for CUDA execution.
The installed runner source was `ce48b253e66b951d87ea42919f945828c62e96a4`,
source tree `e3e5aa9fb61edc8f24bd62b430651265a2816695eb6567da0ae7fcd20f6f98f7`.
The checkpoint stayed `c5fdb5ccb189668d56333f77ba2629f4cd7535f4`.

The actual GPU was one RTX 4070, UUID
`GPU-6eb0a2a5-d251-a91e-8e98-82bc0e371e33`: 12,282 MiB physical VRAM,
12,450,070,528 bytes Torch-visible VRAM, driver 580.95.05 and CUDA 12.8, on a
Xeon E5-2686 v4 host. Packages were Torch 2.8.0+cu128, Triton 3.4.0,
Transformers 5.5.4 and OmniVoice 0.1.4. The reference profile was Hybrid FP16,
16 steps, KV cache off, no split CFG or fused RoPE, serial codec, speed 1 and
postprocessing off. Private graphs were fenced and cleared between jobs, so
this is not a measurement of eight simultaneously resident private graph pools.

The five separately warmed private times were 526.00, 450.00 and 535.32 ms for
the three B1 cases, and 3,717.35 and 3,452.89 ms for EN/ES B8. These are reference
observations only. Full-logit CPU transfer/hash/serialization and setup consumed
the experiment's finite wall budget. The proposed shared wrapper also performs
full model/config fingerprints around forwards, adding host work whose cost
was not measured here. GPU scratch-memory sharing alone would not establish a
request-throughput gain. Arbitrary replay order also remains beyond PyTorch's
documented same-order pool-sharing recommendation and would require its own
finite empirical qualification.

This flat directory preserves the cap, failure, idle, metadata, reference
measurements, current CPU validation, exact prototype sources and strict ASR
reuse proof. The binary hash manifest points to the complete, SHA-verified local
archive containing every partial WAV and NPZ. No tensor/audio binaries, model
weights, conditioning WAVs or private keys are included here.
