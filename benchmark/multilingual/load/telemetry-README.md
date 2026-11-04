# Passive telemetry for future comparison jobs

`telemetry.py` imports no model framework or CUDA package. It starts one
persistent `nvidia-smi` CSV query at 1 Hz and samples Linux proc/sys/cgroup
files. Use the same sampler hash, interval and thread-detail limit around every
comparison arm. Existing V4 harness files and already running workers are
unchanged. No remote sampler has been started during development.

Use a fresh telemetry directory beside the benchmark output. Creating a
directory inside a not-yet-started run output would violate the harness's
fresh-directory guard. The wrapper supports a complete argv JSON file so shell
quoting cannot change nested experimental flags:

```sh
baseline/.venv/bin/python load/telemetry.py \
  --output results/example--telemetry --command-file example-command.json
```

The command file contains the exact argv list for `run.py`, `run_v5.py`, or a
future finite orchestration command. The wrapper records it, tracks that child
and its descendants, and returns its exit status. On interruption it sends its
own child harness SIGINT, allowing the harness to clean up its owned model
server. It never signals unrelated processes. For observation without launching
a benchmark, pass `--track-pid PID --duration SECONDS` instead.

`wrap_plan_telemetry.py` transforms a finite plan into wrapped commands; it only
writes JSON and launches nothing:

```sh
python load/wrap_plan_telemetry.py --plan core-suite.json \
  --output core-suite-with-telemetry.json \
  --sampler /workspace/omnivoice-max-concurency/load/telemetry.py
```

Evidence retained per wrapper:

- `gpu.csv`: timestamps, GPU identity, driver, physical/used VRAM, GPU/memory
  utilization, power/limit, temperature, SM/graphics/memory clocks, performance
  state and clock-event reasons, restricted to fields the installed CLI reports.
  Units follow NVIDIA's `nounits` schema: power in W, clocks in MHz, temperatures
  in C, memory in MiB, utilization in percent. The child uses `TZ=UTC`; raw
  unsupported values/errors remain visible. PyTorch-visible VRAM comes from the
  paired TTS metadata, since this sampler deliberately does not initialize CUDA.
- `samples.jsonl`: host/cgroup CPU/memory/IO pressure, load, per-CPU reported
  frequency and utilization, cgroup CPU quotas/throttle counters/deltas, memory
  pressure/NUMA counters, and tracked-process CPU/IO/thread IDs, top CPU threads,
  CPU affinity and allowed memory nodes. CPU100% means one fully occupied core.
- `metadata.json`: actual sampler hash/settings, command/PIDs, CPU/cgroup/NUMA
  topology and NVIDIA PCI affinity. Allowed nodes and cgroup NUMA counters are
  used instead of an expensive walk of every process's memory mappings.
- `outcome.json`: sampler CPU/time/IO/RSS overhead, overruns, wrapped exit status
  and interruptions. Sampler self CPU plus setup/query child CPU is separate
  from the measured job's CPU. Per-sample collection time excludes later JSON
  writing, while final sampler CPU includes it.

Host load/pressure may include work outside this container. Kernel-reported MHz
is not proof of effective instruction clock. Missing cgroup/sysfs/driver fields
are explicit gaps, not zeros. Every read may race a short-lived process; gone
processes are omitted rather than attributed to a reused PID's old CPU totals.

Three CPU fixture tests check stat parsing, descendant identity, cgroup/pressure
units, frequency fallback and guest-time accounting. A local two-second
CPU-only smoke produced three samples with0.0080s samplerCPU (~0.40% of one
core) on the Mac development host; this is not the rented Xeon's overhead.
A synthetic child-exit check confirms exit status propagation. Actual Linux
and NVIDIA sampling overhead must be taken from each future real wrapper run.
