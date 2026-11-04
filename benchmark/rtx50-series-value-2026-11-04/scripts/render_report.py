"""Render a source-backed report from audited measurements, never estimates as tests."""
import collections
import json
from pathlib import Path
import statistics
import re

ROOT=Path(__file__).resolve().parent
REPO=Path('/Users/ahmadsmacair/omnivoice-triton/omnivoice-triton')
DEST=REPO/'benchmark/rtx50-series-value-2026-11-04'
def read(p):return json.loads(p.read_text())
def table(headers,rows):return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |',*['| '+' | '.join(str(v) for v in row)+' |' for row in rows]])
def phase(g,mode,name):return next(s for s in g['phases'] if s['mode']==mode and s['phase']==name)
def readable(text):
    protected={}
    def keep(match):
        key='\x00'+str(len(protected))+'\x00';protected[key]=match.group();return key
    text=re.sub(r'`[^`\n]+`|(?<=\]\()[^)]+(?=\))|https?://[^\s)]+',keep,text)
    for before,after in [('RTX5060Ti','RTX 5060 Ti'),('RTX5070Ti','RTX 5070 Ti'),('RTX5060','RTX 5060'),('RTX5070','RTX 5070'),('RTX5080','RTX 5080'),('RTX5090','RTX 5090'),('RTX4070','RTX 4070'),('RTX50','RTX 50'),('AppleM5Pro','Apple M5 Pro'),('Python3.12.3','Python 3.12.3'),('Torch2.8','Torch 2.8'),('CUDA12.8','CUDA 12.8'),('Triton3.4','Triton 3.4'),('Transformers5.5.4','Transformers 5.5.4'),('OmniVoice0.1.4','OmniVoice 0.1.4'),('November4,2026','November 4, 2026'),('driver595.84','driver 595.84'),('commit81f','commit 81f'),('onSM120','on sm120'),('labeledSM80','labeled sm80'),('available.','available.')]:text=text.replace(before,after)
    text=re.sub(r'\b(the|The|a|A|all|All|on|On|of|Of|at|At|in|In|or|Or|and|And|for|For|to|To|with|With|about|approximately|across|after|before|only|contains|are|is)(?=\d)',r'\1 ',text)
    text=re.sub(r'\b(?!(?:float|bfloat|cu|sm)\d)([a-z][a-z_-]{2,})(?=\d)',r'\1 ',text)
    text=re.sub(r'(?<=\d)(?=(?:ms|s|MiB|GiB|GB/s|GB|MB|kHz|UTC|TFLOPS|req/s|requests|request|seconds|second|steps|step|graphs|graph|routes|route|languages|language|cases|case|models|model|clips|clip|original|per|product|cloned|visible|warmup|reference|pairs|equivalent|standard|logical|generated|representative|independent|declared|distinct|word|voice|timed|effective)\b)',' ',text)
    text=re.sub(r'(?<=[A-Za-z])(?=\$\d)',' ',text)
    text=re.sub(r'(?<=[A-Za-z]),(?=\d)',', ',text)
    text=re.sub(r';(?=\S)','; ',text)
    text=re.sub(r'(?<=\*\*),(?=\d)',', ',text)
    text=re.sub(r'\b(5060|5070)Ti\b',r'\1 Ti',text)
    text=text.replace('OpenBLAS8','OpenBLAS 8').replace('HTTP429','HTTP 429')
    text=re.sub(r'\bCER(?=\d)','CER ',text)
    for key,value in protected.items():text=text.replace(key,value)
    return text
def main():
    data=read(ROOT/'analysis.json');gs=data['GPUs'];by={g['GPU']:g for g in gs};quality=read(ROOT/'quality-comparison.json');cleanup=read(ROOT/'cleanup-scoped.json');assert cleanup['verified']
    costs=read(ROOT/'rental-cost-estimate.json');qby={q['GPU']:q for q in quality['GPU_comparisons']}
    offers={e['id']:e for e in read(ROOT/'instances.json')['instances']}
    hardware=[];hosts=[];capacity=[];latency=[];resources=[];profiles=[];quality_rows=[];deadlines=[]
    cheap=min(gs,key=lambda g:g['best_finite_throughput']['USD_per1000_valid'])
    low= min(gs,key=lambda g:phase(g,'hybrid-wide','b1-c1')['request_latency_p95_ms'])
    for g in gs:
        h=g['hardware'];ticket=offers[g['id']];smi=h['physical_vram_driver_power'].splitlines()[-1].split(',');physical=smi[2].strip();driver=smi[4].strip();power=smi[5].strip()
        hardware.append([g['GPU'],physical,f"{h['torch_visible_vram_bytes']:,}",h['SMs'],driver,power])
        registration=read(ROOT/'retained'/str(g['id'])/'results/batch1/registrations.json')
        hosts.append([g['GPU'],ticket['offer']['cpu_name'].strip(),ticket['offer']['cpu_cores_effective'],ticket['offer']['geolocation'],f"${g['hourly_USD']:.3f}",f"{statistics.median(r['complete_ms'] for r in registration):.1f}"])
        b=g['best_finite_throughput'];capacity.append([g['GPU'],'`'+b['mode']+'` / `'+b['phase']+'`',f"{b['complete_requests_per_s']:.3f}",f"{b['wall_s']:.2f}",f"{b['request_latency_p95_ms']/1000:.2f}",f"${b['USD_per1000_valid']:.5f}",f"{b['valid_complete_within_first_second']} / {b['max_completions_in_aligned_1s_bin']}"])
        a=phase(g,'results','b1-c1');z=phase(g,'hybrid-wide','b1-c1');passing=[s for s in g['phases'] if s['mode']=='staggered-wide' and s['all_offers_SLO_500ms_fraction']>=.95 and s['rejection_fraction']==0]
        best=max(passing,key=lambda s:s['target_arrival_rate']) if passing else None
        latency.append([g['GPU'],f"{a['request_latency_p95_ms']:.0f}",f"{z['request_latency_p95_ms']:.0f}",f"{z['complete_requests_per_s']:.3f}",str(best['target_arrival_rate']) if best else 'none in tested rates',f"{best['arrival_to_complete_p95_ms']:.0f}" if best else '—'])
        if best:deadlines.append({'GPU':g['GPU'],'phase':best,'nominal_SLO_capacity_per_hourly_USD':best['target_arrival_rate']/g['hourly_USD']})
        n=z['GPU_telemetry'];m=phase(g,'dual-mps','dual-b1-c8')['GPU_telemetry']
        resources.append([g['GPU'],f"{n['utilization.gpu [%]']['mean']:.1f}%",f"{n['memory.used [MiB]']['mean']:.0f}",f"{n['power.draw [W]']['mean']:.0f}",f"{m['utilization.gpu [%]']['mean']:.1f}%",f"{m['memory.used [MiB]']['mean']:.0f}"])
        p=next(s for s in g['profile'] if s['batch']==1);measure=[r['synchronized_full_model_ms'] for r in g['measurements'] if r['batch']==1];copy=next(r for r in g['microbenchmarks'] if r['kind']=='device_copy')
        profiles.append([g['GPU'],f"{statistics.mean(measure):.1f}",str(p['kernels']) if p['kernels'] else 'trace unavailable',f"{p['kernel_sum_ms']:.1f}" if p['kernels'] else '—',f"{100*p['kernel_categories']['GEMM']['sum_ms']/p['kernel_sum_ms']:.1f}%" if p['kernels'] else '—',f"{copy['effective_GB_per_s']:.0f}"])
        q=qby[g['GPU']];quality_rows.append([g['GPU'],f"{q['baseline_graph_byte_identical']}/104",q['arms']['baseline']['both_judges_exact_characters'],q['arms']['throughput']['both_judges_exact_characters'],f"{100*q['arms']['baseline']['models']['omniASR_LLM_Unlimited_1B_v2']['weighted_CER']:.1f}%",f"{100*q['arms']['throughput']['models']['omniASR_LLM_Unlimited_1B_v2']['weighted_CER']:.1f}%"])
    c5070=by['RTX 5070'];c4070=by['RTX 4070'];c5080=by['RTX 5080'];c5090=by['RTX 5090']
    speedratio=phase(c5070,'results','b1-c1')['complete_requests_per_s']/phase(c4070,'results','b1-c1')['complete_requests_per_s']
    costreduction=1-c5070['best_finite_throughput']['USD_per1000_valid']/c4070['best_finite_throughput']['USD_per1000_valid']
    tie=c5080['best_finite_throughput']['USD_per1000_valid']/c5070['best_finite_throughput']['USD_per1000_valid']-1
    break_even=c5070['hourly_USD']*c5080['best_finite_throughput']['complete_requests_per_s']/c5070['best_finite_throughput']['complete_requests_per_s']
    call_best=max(deadlines,key=lambda x:x['nominal_SLO_capacity_per_hourly_USD'])
    equal=sum(q['baseline_graph_byte_identical'] for q in quality['GPU_comparisons'])
    equality_claim=f'The wider-cache optimization preserves **{equal}/624 baseline/after WAV pairs byte-for-byte**,104 per GPU. Its generated audio is unchanged on the tested corpus.' if equal==624 else f'The wider-cache optimization preserves {equal}/624 baseline/after WAV pairs byte-for-byte. Remaining differences require review; a blanket audio-preservation claim is not established.'
    actualcalls={m:sum(q['actual_ASR_inferences_by_model'][m] for q in quality['GPU_comparisons'])+quality['reference_calibration']['models'][m]['actual_asr_inferences'] for m in ['omniASR_CTC_1B_v2','omniASR_LLM_Unlimited_1B_v2']}
    statuses=collections.Counter()
    for g in gs:
        for s in g['phases']:statuses.update(s['status_counts'])
    assert not (set(statuses)-{'200','429'})
    text=f'''# OmniVoice-Triton RTX 50-series rental value — November 4, 2026

Actual TTS inference ran on **GeForce RTX 5060 Ti 16GB, RTX 5070 12GB, RTX 5070 Ti 16GB, RTX 5080 16GB, RTX 5090 32GB and RTX 4070 12GB**. Physical VRAM is from each card's `nvidia-smi`; visible memory is independently from PyTorch. All measurements are from November 4, 2026 UTC. The two Omnilingual judges ran separately on the RTX 5090 (32,607MiB physical;33,670,758,400 visible bytes;driver595.84;CUDA12.8). An additional20-language Whisper diagnostic ran on Apple M5 Pro with24GiB shared unified RAM. Auditing, packaging and charts are CPU-only.

**The measured recommendation is conditional on latency.** {cheap['GPU']} has the lowest cost per completed request among the tested settings. RTX5070 and5080 are effectively tied on throughput value: their cost difference is only {abs(tie)*100:.2f}%, which this single-host screen cannot reliably resolve. The5070 is the cheaper entry point; the5080 serves more traffic per card. {low['GPU']} has the lowest warmed single-request latency. Among tested one-worker staggered profiles, {call_best['GPU']} gives the highest nominal load per rental dollar while meeting the500ms p95 screen with zero rejections. These are rental-offer conclusions, not retail purchase rankings or universal GPU-family rankings.

{table(['Actual GPU','Physical VRAM','Torch-visible bytes','SMs','Driver','Power limit'],hardware)}

The same locked runtime was used for all TTS cards: Python3.12.3, Torch2.8.0+cu128, CUDA12.8, Triton3.4.0, Transformers5.5.4 and OmniVoice0.1.4. CPU thread settings were OMP/MKL/OpenBLAS8. The physical GPU UUID, complete CPU/cgroup description, library freeze and co-resident process evidence are retained per rental. No ASR or MuseTalk inference shared a GPU with timed TTS. The5090's isolated ASR wheel install overlapped an early wide-cache warmup; its final install log timestamp was22:29:00UTC and the first timed C1 request began22:29:14UTC. Later ASR installation/downloads waited until all5090 TTS stages ended. Warmup duration for that arm is not an isolated startup benchmark.

{table(['GPU','Rental host CPU','Offer effective cores','Location','USD/hour incl.100GB disk','Median initial clone registration ms'],hosts)}

Prices are the accepted Vast offers, rather than a general market guarantee. Transport is local HTTP on each rented server. WAN/control-plane routing, upstream speech recognition/LLM work and MuseTalk require their own latency budgets; the Australia and Korea rentals were not measured as Chicago-to-GPU application paths. Provider host CPU performance and driver versions differ, so this is a comparison of usable rental systems, not an isolated causal architecture experiment.

All cards used your `max-concurency` fork at **b7cced4a0274f18aca541f7837c490bb81d8744e**, with identical24 product Python files and the checkpoint `k2-fsa/OmniVoice` revision `c5fdb5ccb189668d56333f77ba2629f4cd7535f4`. The product archive SHA256 is `a1de5675d138983472efd6028e01ec2ee0241730b81675d51161e71b0974fc0f`. No product/kernel source or deployment default was changed by this experiment. The tested speedup is a serving configuration already supported by the branch. Both `triton` and `hybrid` are OmniVoice-Triton runners: hybrid adds CUDA Graphs to the Triton patches.

Every timed generation used a full cloned reference,16 steps, FP16 backbone, speed1, denoise on, postprocessing off, codec batch1 and sentence splitting off. Generated-response caching and response coalescing were disabled. The9 original Aoede reference clips were registered before timing; these are9 clips of the same source voice, not9 independent speakers. The English reference remains13.931 seconds. Many routes use a declared cross-language reference. A prepared `prompt_id` reuses reference preparation while every response still executes cloned TTS inference. New-voice upload/preparation and first-use registration are additional costs. The16-step profile is held constant; it is not evidence that16 steps match a32-step default in quality.

The workload is104 short published-text cases, one per Lingua route, with source/license and language-mapping caveats preserved. It represents short speech turns rather than long narration or thousands of unique voices. Each process receives a startup seed; concurrent/batched workers do not share a per-request RNG stream. Every measured response is a complete24kHz mono WAV. Per-request latency runs from dispatch to receipt of the complete body. Waveform decoding, hashing and file writes occur outside timing.

The closed-loop matrix uses104 requests per phase: one eager worker at batch limits1/8 and concurrency1/8/32; two ordinary workers and two verified MPS workers at batch limit1/concurrency2/8/32; a one-worker MPS control; and hybrid runners with8 graphs/16,384 row tokens or64 graphs/65,536 row tokens at batch limits1/8 and concurrency1/8. Closed-loop concurrency means clients replenish completed requests; it is not a simultaneous104-request burst. Separate staggered tests use common seeded Poisson schedules,40 seconds each at nominal0.5/1/2/4/8 arrivals/sec, with actual offer counts30/41/81/162/340, first arrival at0, batch limit1 and pending limit4. Arrival-to-complete timing includes dispatch lateness. HTTP429 rejections remain in every offered-load denominator.

The following table selects the highest finite throughput observed for each card. Its tail latency is shown because throughput and individual response time are different objectives. One-second counts are from actual aligned completion bins, not throughput rounded into a first-second promise.

{table(['GPU','Best tested mode / phase','Complete req/s','104-request wall s','Request p95 s','USD/1,000 valid','First second / peak aligned1s'],capacity)}

![Measured GPU value](../benchmark/rtx50-series-value-2026-11-04/figures/GPU-value.png)

The matched sequential eager5070 is {100*(speedratio-1):.1f}% faster than the4070. At these offers and each card's best tested serving setting, the5070 costs {100*costreduction:.1f}% less per completed request. A5080 price of approximately${break_even:.3f}/hour would tie the observed5070 throughput value; the actual5080 offer was${c5080['hourly_USD']:.3f}/hour. Small price/host changes can reverse their ordering. The5090's best throughput costs${c5090['best_finite_throughput']['USD_per1000_valid']:.5f}/1,000 requests; its stronger reason for voice calls is latency headroom.

The next table holds batch limit1 fixed for the graph comparison. The staggered columns identify the highest sampled nominal rate with at least95% of **all offers** returning a valid full WAV within500ms and with zero rejections. Each row is a short screen, not statistical or production p95 certification. Staggered two-worker MPS was not tested and could change a call-capacity ranking.

{table(['GPU','Eager C1 p95 ms','64-graph C1 p95 ms','64-graph C1 req/s','Passing staggered nominal req/s','Passing arrival p95 ms'],latency)}

![Staggered deadline success and rejection](../benchmark/rtx50-series-value-2026-11-04/figures/staggered-SLO.png)

On the5090, the8-graph B1 arm made279 captures and271 evictions across warmup plus timed work. There are64 distinct shapes in this short corpus. With64 graph slots and65,536 row tokens, all64 fit; the B1 arm has zero timed recaptures/evictions. Matched C1 throughput increases2.266→5.488 requests/sec and p95 full-WAV latency falls450→197ms. GPU busy time rises39.8%→88.1%, with mean framebuffer use increasing3,248→6,535MiB. This is a concrete way extra VRAM helps: retaining useful graphs removes repeated capture and CPU launch work. Unseen text/reference lengths can introduce new shapes; the measured result assumes the tested warmup and finite shape set. The row-token policy is a memory heuristic with runtime headroom checks, not a strict VRAM partition.

{table(['GPU','One wide-cache worker GPU busy','Mean VRAM MiB','Mean power W','Two MPS workers GPU busy at C8','Two MPS mean VRAM MiB'],resources)}

Two workers can improve throughput when they cover each other's CPU/launch gaps. On the5090, two ordinary eager workers reach about4.53req/s, but one wide-cache worker reaches5.68req/s at C8. On the4070, two ordinary workers reach about2.19req/s and verified MPS raises that to about2.56req/s. More free VRAM therefore does not imply spare compute. GPU-utilization percent is busy time rather than theoretical FLOP utilization; memory-utilization percent is memory activity rather than bandwidth saturation. Per-phase CPU/RSS samples for one/two MPS workers are in the derived audit; RSS sums can double-count shared pages.

Architecture research explains a plausible50-series advantage without treating advertised AI TOPS as this workload. NVIDIA lists5070 versus4070 bandwidth672 versus504GB/s and L2 cache48 versus36MB. Dense FP16/BF16 Tensor throughput with FP32 accumulation is61.7 versus58.3TFLOPS, only about6% higher. FP4 sparse TOPS describe a different arithmetic recipe; this FP16 experiment does not automatically gain their advertised multiplier. [NVIDIA RTX Blackwell architecture, table6](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf).

Consumer RTX50 cards are compute12.0; data-center Blackwell10.0 features and memory systems should not be assigned to them by name. [NVIDIA tuning guide](https://docs.nvidia.com/cuda/archive/12.8.1/blackwell-tuning-guide/index.html). Graph replay reduces repeated CPU submission overhead, which fits the measured5090 improvement. [PyTorch CUDA Graphs](https://docs.pytorch.org/docs/main/notes/cuda.html). MPS permits concurrent scheduling across client processes when resources allow it. [NVIDIA MPS architecture](https://docs.nvidia.com/deploy/mps/architecture.html).

Separate English-clone diagnostics use two warmups and five synchronized full-model repetitions per B1/B8, then separate shape collection and separate profiler runs. Instrumented kernel times are not API timings. The trace is dominated by backbone GEMMs and attention;5090 B1 contains about17,199 kernels, with GEMM accounting for about58% of kernel time. Current traces include CUTLASS/attention implementations labeledSM80; they do not demonstrate use of FP4 or a new Blackwell-only tensor instruction path. Such names identify implementation templates, not a different physical GPU. The model's iterative generation dominates host stage timers; prompt lookup, WAV encoding and codec decode are much smaller for these short turns.

{table(['Actual GPU','Unprofiled model B1 mean ms','Traced kernels','Traced kernel sum ms','GEMM share of kernel time','Synthetic copy GB/s'],profiles)}

Synthetic copy reads and writes128MiB tensors, exceeding L2; it is not a model DRAM-counter measurement. Synthetic FP16 GEMMs at observed dimensions and4096 cubed are also retained. The5060Ti profiler returned no CUDA kernel events, causing the initial diagnostic helper to stop; a separate retry preserved the unavailable-trace status and completed model/shape/thread/microbench diagnostics. No host permission changes were attempted. Its inference measurements are valid GPU runs, while kernel attribution remains unavailable.

Quality was checked separately from performance. The wider-cache optimization preserves **{equal}/624 baseline/after WAV pairs byte-for-byte**,104 per GPU. Thus its generated audio is unchanged on the tested corpus. Both local Meta `omniASR_CTC_1B_v2` and `omniASR_LLM_Unlimited_1B_v2` screened1,872 generated logical cases plus9 reference clips, across all104 declared routes. They ran sequentially in a separate environment, float32, batch1, source commit81f51e224ce9e74b02cc2a3eaf21b2d91d743455, with complete weight/tokenizer hashes. Identical waveform/language inputs reuse transcripts within each shard: actual model inferences are{actualcalls}. Reference text never enters either recognizer. [Omnilingual source and model documentation](https://github.com/facebookresearch/omnilingual-asr).

{table(['TTS GPU','Baseline/graph byte-identical','Baseline both-judge exact chars /104','Throughput both-judge exact chars /104','Baseline LLM weighted CER','Throughput LLM weighted CER'],quality_rows)}

Absolute multilingual quality remains unqualified. These strict ASR screens flag many baseline cases, and the two variants share a model family. Even the9 reference clips produce5 review flags, so ASR mismatch must not be reported as a verified TTS failure rate. CER is character-weighted; word error rates are not comparable across scripts without spaces. The additional cached Whisper model independently screens20 exactly supported standard languages,240 logical baseline/throughput cases, on Apple M5 Pro; it supplies a narrower cross-check rather than104-route coverage. The other84 routes have no Whisper clearance. Neither ASR establishes speaker identity, native pronunciation or MOS. Twi retains its declared Akan proxy, and dialect/script/source caveats remain attached to individual cases. Throughput-arm RNG streams differ from the baseline; their per-language gains and regressions are documented rather than described as a deterministic quality improvement.

Across the six GPUs there were **{data['timed_offers']:,} timed offers**,{data['warmup_offers']:,} warmup offers and **{data['valid_load_WAVs_verified']:,} locally verified load WAVs**, plus profiler samples. Timed HTTP counts are{dict(statuses)}. All accepted200 responses pass decoding, finite-signal, non-silence and sample-rate checks, and source execution counters match actual successful inference members. Rejections are explicit overload outcomes, not successful low-latency replies. The full raw responses, all generated audio, logs, shapes and traces remain in `/Users/ahmadsmacair/code/lingua/lingua/work/omnivoice-50series-value-2026-11-04/retained`. The Git evidence bundle includes losslessly compressed metadata/traces,54 representative generated WAVs, complete scripts, all word checks and SHA manifests; original conditioning clips and checkpoint weights are excluded.

All8 task-created contracts, including2 failed-start rentals, are deleted with API absence verified. Local size and SHA256 checks completed before deleting each productive rental. The estimated compute-plus-allocated-storage lease total is${costs['total_estimated_USD']:.3f}; it includes provisioning and the separate ASR host lifetime, but excludes network charges/provider adjustments and is not an invoice. Failed-start contracts and actual test/cleanup timestamps are retained. The finite cleanup guard remained scoped to this task.

The next optimization target is avoiding shape churn and choosing efficient FP16 GEMM/attention implementations onSM120, with identical-output or paired word checks. A controlled runtime/kernel comparison, longer turns, many unique references, cold voice preparation and staggered two-worker MPS should precede a broad production promise. The tested64-graph configuration is reviewable on `max-concurency`; deployment defaults and production servers were not changed.

For fleet planning, active calls convert to TTS arrivals through `calls × speech turns per call per second`; simultaneous bursts are a separate load. A1,000-request burst completed in2 seconds would require500 complete requests/sec. Dividing by the fastest6.33req/s finite screen already gives a throughput-only lower bound of approximately80 equivalent GPUs before latency headroom. That is a capacity calculation, not a measured fleet result. This experiment provides no evidence for1,000 fresh cloned full-WAV requests in1–2 seconds on a single4070/5070.

The full [evidence index](../benchmark/rtx50-series-value-2026-11-04/README.md), [capacity CSV](../benchmark/rtx50-series-value-2026-11-04/derived/capacity.csv), [audio manifest](../benchmark/rtx50-series-value-2026-11-04/samples/manifest.json) and [publication integrity manifest](../benchmark/rtx50-series-value-2026-11-04/publication-manifest.json) make the figures inspectable. Published text fixtures retain the FLORES CC-BY-SA4.0, Omnilingual corpus CC-BY4.0 and individual Wikipedia-source notices already attached to the original corpus; code retains the repository license.
'''
    text=text.replace(f'The wider-cache optimization preserves **{equal}/624 baseline/after WAV pairs byte-for-byte**,104 per GPU. Thus its generated audio is unchanged on the tested corpus.',equality_claim)
    text=text.replace('throughput-only lower bound of approximately80 equivalent GPUs before latency headroom','planning estimate of approximately80 equivalent GPUs before latency headroom, assuming that observed per-GPU throughput scales linearly')
    text=text.replace('The5060Ti profiler returned no CUDA kernel events','The5060Ti profiler reported `CUPTI_ERROR_INVALID_DEVICE` and returned no CUDA kernel events')
    text=text.replace('These are rental-offer conclusions, not retail purchase rankings or universal GPU-family rankings.','These are rental-offer conclusions, not retail purchase rankings or universal GPU-family rankings. The5080\'s fastest MPS setting remains a quality-review candidate: its observed throughput value does not establish quality preservation. The wider-cache single-worker change has the stronger paired preservation evidence.')
    text=text.replace('Absolute multilingual quality remains unqualified.','Absolute multilingual quality remains unqualified. On the5080 MPS throughput arm, CTC weighted CER rose32.5→35.3% and LLM CER12.5→19.4%. All three recognizers transcribed its English sample with “aside” where the baseline and expected script use “inside”. The different RNG streams prevent attributing this change to MPS alone, but it is sufficient to withhold a quality-preservation claim for that setting. The [baseline sample](../benchmark/rtx50-series-value-2026-11-04/samples/rtx-5080-baseline-en.wav) and [MPS sample](../benchmark/rtx50-series-value-2026-11-04/samples/rtx-5080-throughput-en.wav) are retained for review.')
    text=text.replace(f'actual model inferences are{actualcalls}',f"actual model inferences are{actualcalls['omniASR_CTC_1B_v2']:,} for CTC and{actualcalls['omniASR_LLM_Unlimited_1B_v2']:,} for LLM")
    text=text.replace(f'Timed HTTP counts are{dict(statuses)}',f"Timed HTTP counts are{statuses['200']:,} HTTP200 and{statuses['429']:,} HTTP429")
    text=text.replace('All cards used your `max-concurency` fork','All cards used the AhmadAFS1/omnivoice-triton `max-concurency` branch')
    report=REPO/'docs/rtx50-series-value-2026-11-04.md';report.write_text(readable(text))
    readme=f'''# RTX50-series OmniVoice-Triton evidence — November4,2026

Actual TTS GPUs: RTX5060Ti16GB, RTX507012GB, RTX5070Ti16GB, RTX508016GB, RTX509032GB and RTX407012GB. Exact physical/visible VRAM, driver, CPU, rates and co-resident load are in the [full report](../../docs/rtx50-series-value-2026-11-04.md) and per-rental hardware/provenance. Torch2.8/CUDA12.8/Triton3.4; cloned full-reference,16-step, full-WAV requests; generated-response cache off. Omnilingual ASR uses the RTX5090 separately; the20-language Whisper diagnostic uses AppleM5Pro24GiB unified memory. CPU-only auditing and charts are labeled separately.

- `derived/capacity.csv` covers every measured phase, including rejection/deadline denominators. `derived/analysis.json.gz` contains resource samples and kernel attribution.
- `raw/rtx-*` retains frozen executed harnesses, source hashes, outcomes, logs, health/counter snapshots, request records and CUDA traces. Some inherited helper docstrings say RTX4070; those are historical helper descriptions, not hardware evidence. Runtime GPU checks, tickets and per-card metadata determine actual hardware.
- `quality/omnilingual` contains both judges' raw transcripts, model/runtime metadata, scored cases and reference calibration. `quality/selection` states exactly which three104-route arms were screened. `quality/whisper-core` is an additional20-language diagnostic, not complete coverage.
- `samples/manifest.json` links54 generated examples to expected scripts, clone references, settings and ASR results. All workload audio remains in the local retained directory stated in the full report; metadata paths do not imply every WAV is duplicated in Git.
- `fixtures/corpus.json` preserves per-case source/license/mapping caveats; `reference-manifest.json` describes the9 original clips without publishing conditioning audio. `source.tar.gz` is the exact pinned product archive, including its LICENSE, not model weights.
- `cleanup.json`, `rental-cost-estimate.json`, `QA.json` and `publication-manifest.json` record lifecycle, cost assumptions and integrity. All task-created rental contracts are absent after cleanup.

Large metadata/log/trace files use `.gz` suffixes. The publication manifest records original and saved bytes/SHA256; decompression reproduces the originals. Raw complete backup inventories include all retained output-WAV hashes. Model weights, environments, SSH keys and conditioning WAVs are excluded. Generated audio equality proves the cache change preserved the tested outputs; ASR flags and baseline language quality still require review.
'''
    (DEST/'README.md').write_text(readable(readme))
    print(json.dumps({'report':str(report),'recommendation_lowest_cost':cheap['GPU'],'recommendation_best_tested_call_profile':call_best['GPU'],'graphs_equal_pairs':equal}))
if __name__=='__main__':main()
