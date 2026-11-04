# Joint complete-WAV latency and word screen — all2704 offers

Reused TTS: NVIDIA GeForce RTX4070Ti, physical12,282MiB/visible12,481,134,592bytes, driver610.57.04/CUDA12.8,2026-11-04; source377de39b,graph2/KVoff/splitoff/codec1/cache0, 16steps,actualpresets,sequential. Physical/visible evidence is in each original metadata. CTC ASR: RTX4070,physical12,282MiB/visible12,450,070,528bytes,driver580.95.05/CUDA12.8. LLM ASR: same4070Ti, later separate lease. This intersection is localCPU-only; no new inference.

Every count uses the complete2704-offer denominator, including allfour empty HTTP200 outputs as failures. Exactfixture/run/request/SHA joins verify all2704 original request latencies. Timing ends when the complete HTTP WAV body has arrived, with a<=500ms threshold. The p95 values below include alloffers and are never conditioned on word-score success.

| RTX4070Ti preset/mode | Alloffers | Signal+<=500ms | BothCER0 separately | JointCER0 | Jointstrict exact | Alloffer p95ms |
|---|---:|---:|---:|---:|---:|---:|
| ALL | 2704 | 1353 | 535 | 304 | 283 | 584.9 |
| lingua-default | 832 | 472 | 109 | 72 | 66 | 581.9 |
| lingua-tutor | 832 | 339 | 165 | 71 | 69 | 587.0 |
| lingua-video | 832 | 336 | 170 | 70 | 68 | 587.0 |
| worker-auto-not-current-Lingua-UI | 104 | 103 | 43 | 43 | 39 | 301.1 |
| worker-design-not-current-Lingua-UI | 104 | 103 | 48 | 48 | 41 | 303.4 |

The joint result is304/2704(11.24%) for bothjudges' zero-character-error screen and283/2704(10.47%) for strict normalized exactness. Separate counts—1353 latency/signal matches and535 bothCER0—must not be presented as the joint result. Four empty results are included as failures even if their HTTP response is fast.

This is a joint **ASR-screen/latency** result, not proven native word fidelity or a production SLO certification. ASR mismatch/spacing/script errors remain possible; exact ASR does not verify voice identity or naturalness. Proxy/calibration strata are retained inside each group, including Twi's broader Akan mapping. No proxy, absent-target-control or mismatched-reference row was silently removed from the denominator.

`joint-slo-per-voice-configuration26.json` covers every stockvoice/preset and auto/design cell across104routes; `joint-slo-per-route104.json` covers everyroute's26cells. The original word report retains reference/native/fallback caveats and four explicit failures. The grid has no matched earlier2704-case arm, so it establishes no new-optimization regression count or causal speed/postprocessing improvement.
