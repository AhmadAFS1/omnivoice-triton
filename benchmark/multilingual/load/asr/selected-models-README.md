This is CPU-only controller implementation and validation. It has not launched a new model. Intended judge hosts are the RTX 4070 Ti modern instance and RTX 4070 primary instance; each actual worker independently records GPU model, physical/visible VRAM, driver/runtime and co-resident load. Historical hardware is not substituted for a new run's evidence.

`run_selected_models.py` preserves the original `run_bounded.py` and `evaluate.py`. Default execution still selects both BF16 batch1 judges, CTC first. `--models` can select one host's assigned judge; the coverage maps always retain both required models. One finished judge produces `selected_models_completed` with `required_models_complete:false`, never a two-judge quality pass. The unchanged evaluator still receives only audio paths and known language/script; expected text is retained only for later scoring.

The modern LLM invocation should use an explicit new configuration namespace:

```sh
/workspace/omnivoice-max-concurency/asr-venv/bin/python \
  /workspace/omnivoice-max-concurency/asr/run_selected_models.py \
  --prepared /workspace/omnivoice-max-concurency/quality-selections/frontier-diagnostics \
  --output /workspace/omnivoice-max-concurency/asr-results/modern-LLM-priority-v1 \
  --models omniASR_LLM_Unlimited_1B_v2 \
  --protocol-namespace modern59071593-batch1-v1
```

That command prints a CPU plan. Root must separately release the GPU and add `--allow-gpu-inference` to execute. Model/package verification and a fresh empty compute-process list remain required. Choose a finite budget and reserve; budgets include preprocessing/imports and model work. Multiple invocations also need one shared absolute lease deadline.

The new protocol ID includes namespace, model, dtype, unchanged evaluator hash, model/tokenizer asset hashes and recorded package versions. Published copies receive this ID; raw worker files remain byte-for-byte unchanged. A different or legacy namespace is visibly skipped during prior imports, so old primary LLM transcripts cannot silently replace fresh modern before/after evaluation. Same-namespace verified WAV/script evidence can be reused across expanded fresh stages. Resume locks the selected model order, namespace and protocol IDs. The new controller refuses to resume original legacy output directories; use explicit prior imports into a fresh output instead.

Primary CTC can use `--models omniASR_CTC_1B_v2 --protocol-namespace legacy-batch1` when root deliberately reuses compatible original CTC evidence. Every reuse records its source and zero new ASR compute time. Keep before/after on the same declared judge configuration and host; record hardware differences rather than treating them as a TTS optimization.

For scoring, pass individual validated `published/<attempt>/` directories from both hosts to canonical `quality/report.py score`, using the full canonical preparation and the default two required judges. Do not pass only one model to make missing coverage disappear. Selection directories contain evaluator manifests, not replacement quality preparations. Protocol guard V4 rejects mixed configurations for the same judge.

Seven CPU tests cover selected-model defaults, duplicate rejection, strict resume, configuration identity, immutable raw publication, cross-namespace rejection, two-directory prior reuse, and a real subprocess using a fake CPU evaluator. The latter proves only the requested judge executes and that the other judge remains explicitly missing. No Torch/model inference occurs in these tests. Ruff and compile checks pass.

`select_future_priority.py` selects a fixed step16 case per all104 routes plus reference controls, and a separate frontier/diagnostic cohort. The latter includes all312 step16/8/4 cases, TK/SO repeats, five zero-frame outcomes and three independently identified weak/short signal cases. Zero-frame outcomes remain unavailable, never ASR inputs. Signal-selected diagnostics are labeled separately from the provenance-fixed104-route sample; no ASR scores drive selection. Two CPU tests prevent missing-route substitution and incorrect ASR inclusion of invalid audio.

`quality/reference-union80-v1/manifest.json` combines the64 original voice references and16 exact-PCM English prefixes. All80 WAV hashes were verified. Both `file` and `path` are portable relative paths; retain the source `quality/` directory layout on each host. Original manifests and WAVs remain unchanged. Prefix text/PCM verification is not yet two-judge or speaker-quality approval.
