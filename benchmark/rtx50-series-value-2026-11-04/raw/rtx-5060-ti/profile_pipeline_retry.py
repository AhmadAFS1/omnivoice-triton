"""RTX4070 clone pipeline + CUDA trace. Instrumented timings are diagnostic.

Run only after API capacity exits. No source/model/precision changes. Warmed
unprofiled measurements, shape collection, and profiler windows are separate.
"""
import collections
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import subprocess
import time

import numpy as np
import soundfile as sf
import torch
from omnivoice import OmniVoiceGenerationConfig
from omnivoice_triton import create_runner

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'profile-retry'


def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')


def command(args):
    try:return subprocess.check_output(args,text=True,stderr=subprocess.STDOUT,timeout=15).strip()
    except Exception as e:return repr(e)


def union_us(events):
    end=-float('inf');total=0
    for a,b in sorted((e['ts'],e['ts']+e['dur']) for e in events):
        total+=max(0,b-max(a,end));end=max(end,b)
    return total


def category(name):
    n=name.lower()
    for f,c in [('gemm','GEMM'),('fmha','attention'),('flash','attention'),('rms_norm','RMSNorm'),('swiglu','SwiGLU'),('catarray','concat-copy'),('elementwise','elementwise/copy'),('copy','elementwise/copy')]:
        if f in n:return c
    return 'other'


def main():
    OUT.mkdir(exist_ok=False)
    prop=torch.cuda.get_device_properties(0)
    assert prop.name in {'NVIDIA GeForce RTX '+x for x in ['4070','5060 Ti','5070','5070 Ti','5080','5090']}
    write(OUT/'hardware.json',{'utc':dt.datetime.now(dt.UTC).isoformat(),'GPU':prop.name,
        'torch_visible_vram_bytes':prop.total_memory,'SMs':prop.multi_processor_count,'compute_capability':[prop.major,prop.minor],
        'physical_vram_driver_power':command(['nvidia-smi','--query-gpu=name,uuid,memory.total,memory.used,driver_version,power.limit','--format=csv']),
        'other_gpu_processes_before_model':command(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv']),
        'nvidia_smi_q':command(['nvidia-smi','-q']), 'cpu':command(['lscpu']),
        'cgroups':{str(p):p.read_text() for p in [Path('/sys/fs/cgroup/cpu.max'),Path('/sys/fs/cgroup/memory.max'),Path('/sys/fs/cgroup/cpuset.cpus.effective')] if p.exists()},
        'python':platform.python_version(),'runtime_cuda':torch.version.cuda,
        'packages':{p:importlib.metadata.version(p) for p in ['torch','torchaudio','transformers','triton','omnivoice','numpy','soundfile']},
        'cpu_threads':torch.get_num_threads(),'cpu_interop_threads':torch.get_num_interop_threads()})
    runner=create_runner('triton',device='cuda',model_id=str(ROOT/'model'),dtype='fp16',disable_unused_kv_cache=True,audio_decode_batch_size=1,decode_postprocess_workers=4)
    runner.load_model();model=runner.model
    refs=json.loads((ROOT/'references/manifest.json').read_text())
    ref=next(r for r in refs if r['id']=='en-aoede')
    p=ROOT/'references'/ref['path'];assert hashlib.sha256(p.read_bytes()).hexdigest()==ref['sha256']
    config=OmniVoiceGenerationConfig(num_step=16,class_temperature=0.0,position_temperature=5.0,denoise=True,postprocess_output=False)
    config.postprocess_mode='off'
    case={'text':'No one was inside the apartment.','language':'en'}
    write(OUT/'workload.json',{'case':case,'reference':ref,'steps':16,'runner':'triton','KV':False,'codec_batch':1,'fused_rope':False,'split':False,'scope':'single cloned English voice; B1/B8; profiler is not HTTP capacity'})
    try:
        with torch.inference_mode():
            t=time.perf_counter();prompt=model.create_voice_clone_prompt(ref_audio=str(p),ref_text=ref['text'],preprocess_prompt=True);torch.cuda.synchronize()
            write(OUT/'clone-prepare.json',{'full_clone_prompt_ms':(time.perf_counter()-t)*1000})

            def generate(b):
                torch.manual_seed(20261104+b);torch.cuda.manual_seed_all(20261104+b)
                torch.cuda.synchronize();t=time.perf_counter()
                audio=model.generate(text=[case['text']]*b,language=['en']*b,voice_clone_prompt=[prompt]*b,speed=[1.0]*b,generation_config=config)
                torch.cuda.synchronize()
                return audio,(time.perf_counter()-t)*1000,model.get_runtime_stage_metrics()

            measurements=[];shapes=[];saved=[];reference_audio={}
            for b in [1,8]:
                generate(b);generate(b)
                baseline=[]
                for rep in range(5):
                    audio,ms,metrics=generate(b);baseline.append(audio)
                    measurements.append({'batch':b,'rep':rep,'synchronized_full_model_ms':ms,'stages_host_wall_ms':metrics})
                reference_audio[b]=baseline[0]
                handles=[];observations=[]
                for name,module in model.named_modules():
                    if isinstance(module,torch.nn.Linear):
                        def hook(layer,args,name=name):
                            x=args[0];m=math.prod(x.shape[:-1]);n,k=layer.weight.shape
                            observations.append({'module':name,'input_shape':list(x.shape),'weight_shape':list(layer.weight.shape),'dtype':str(x.dtype),
                                'M':m,'N':n,'K':k,'FLOPs':2*m*n*k,'minimum_tensor_bytes':x.element_size()*(m*k+n*k+m*n)})
                        handles.append(module.register_forward_pre_hook(hook))
                generate(b)
                for h in handles:h.remove()
                write(OUT/f'B{b}-linear-observations.json',observations);shapes.extend(observations)
                originals={};ranges=['_preprocess_all','_prepare_inference_inputs','_generate_iterative','_generate_chunked']
                for name in ranges:
                    if not hasattr(model,name):continue
                    original=getattr(model,name);originals[name]=original
                    def wrapped(*a,_fn=original,_name=name,**kw):
                        with torch.profiler.record_function('TTS'+_name):return _fn(*a,**kw)
                    setattr(model,name,wrapped)
                original_llm=model.llm.forward
                def llm(*a,**kw):
                    with torch.profiler.record_function('TTS_LLM_forward'):return original_llm(*a,**kw)
                model.llm.forward=llm
                try:
                    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=False,profile_memory=False,with_stack=False) as prof:
                        with torch.profiler.record_function('TTS_FULL'):
                            audio,ms,metrics=generate(b)
                    prof.export_chrome_trace(str(OUT/f'B{b}-trace.json'))
                    (OUT/f'B{b}-operators.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=50))
                finally:
                    model.llm.forward=original_llm
                    for name,original in originals.items():setattr(model,name,original)
                parity=all(np.array_equal(x,y) for x,y in zip(baseline[0],audio,strict=True));assert parity
                for arm,waves in [('unprofiled',baseline[0]),('profiled',audio)]:
                    for i,w in enumerate(waves):
                        p=OUT/f'B{b}-{arm}-{i}.wav';sf.write(p,w,24000,subtype='PCM_16')
                        saved.append({'batch':b,'arm':arm,'path':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'expected':case['text'],'language':'en'})
                events=json.loads((OUT/f'B{b}-trace.json').read_text())['traceEvents']
                kernels=[e for e in events if e.get('cat')=='kernel']
                grouped=collections.defaultdict(lambda:{'calls':0,'sum_ms':0})
                for e in kernels:
                    g=grouped[category(e['name'])];g['calls']+=1;g['sum_ms']+=e['dur']/1000
                span=(max(e['ts']+e['dur'] for e in kernels)-min(e['ts'] for e in kernels))/1000 if kernels else None
                write(OUT/f'B{b}-profile-summary.json',{'batch':b,'profiled_full_ms':ms,'profiled_host_stage_ms':metrics,
                    'profiling_preserved_float_audio_exactly':parity,'CUDA_kernel_trace_available':bool(kernels),'kernels':len(kernels),'kernel_sum_ms':sum(e['dur'] for e in kernels)/1000,
                    'kernel_interval_union_ms':union_us(kernels)/1000,'kernel_span_ms':span,'kernel_categories':dict(grouped),
                    'linear_FLOPs_observed_per_generate':sum(x['FLOPs'] for x in observations),
                    'linears_count':len(observations),'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                    'note':'Kernel active intervals do not prove SM/Tensor Core saturation. Sum may overlap. Host metrics are not separately synchronized GPU stage times.'})
                write(OUT/'measurements.json',measurements);write(OUT/'audio-manifest.json',saved)
            # Host thread sensitivity uses the same prepared clone and per-call
            # seed. This affects CPU scheduling only; verify actual output parity.
            thread_results=[];original_threads=torch.get_num_threads()
            try:
                for threads in dict.fromkeys([original_threads,1,4,original_threads]):
                    torch.set_num_threads(threads);generate(1)
                    for rep in range(5):
                        waves,ms,metrics=generate(1)
                        thread_results.append({'threads':threads,'rep':rep,'synchronized_full_model_ms':ms,
                            'float_audio_equal_to_original_threads':all(np.array_equal(x,y) for x,y in zip(reference_audio[1],waves,strict=True)),
                            'stages_host_wall_ms':metrics})
            finally:torch.set_num_threads(original_threads)
            write(OUT/'cpu-thread-sensitivity.json',thread_results)
            # Synthetic diagnostics: memory copy >L2 and dense FP16 GEMMs, plus real shapes.
            micro=[]
            def measure(fn,reps=15):
                for _ in range(3):fn()
                torch.cuda.synchronize();a=torch.cuda.Event(enable_timing=True);z=torch.cuda.Event(enable_timing=True)
                a.record()
                for _ in range(reps):fn()
                z.record();z.synchronize();return a.elapsed_time(z)/reps
            x=torch.empty(128*1024*1024,device='cuda',dtype=torch.uint8);y=torch.empty_like(x)
            ms=measure(lambda:y.copy_(x),50)
            micro.append({'kind':'device_copy','bytes_read_plus_write':x.numel()*2,'ms':ms,'effective_GB_per_s':x.numel()*2/ms/1e6,'note':'Synthetic read+write bandwidth, not model DRAM-counter measurement'})
            del x,y
            shapes_by_key={(x['M'],x['N'],x['K']) for x in shapes if x['dtype']=='torch.float16'}
            for m,n,k in sorted(shapes_by_key | {(4096,4096,4096)}):
                a=torch.ones((m,k),device='cuda',dtype=torch.float16);w=torch.ones((k,n),device='cuda',dtype=torch.float16);out=torch.empty((m,n),device='cuda',dtype=torch.float16)
                ms=measure(lambda:torch.mm(a,w,out=out))
                micro.append({'kind':'FP16_GEMM','M':m,'N':n,'K':k,'ms':ms,'TFLOPs_per_s':2*m*n*k/ms/1e9,
                    'minimum_tensor_arithmetic_intensity_FLOPs_per_byte':m*n*k/(m*k+n*k+m*n),
                    'note':'Synthetic contiguous matrices at observed dimensions, not measured model GEMM traffic/kernel attribution'})
                del a,w,out
            write(OUT/'microbenchmarks.json',micro)
    finally:
        runner.unload_model()
    write(OUT/'COMPLETE.json',{'utc':dt.datetime.now(dt.UTC).isoformat()})


if __name__=='__main__':main()
