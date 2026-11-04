"""CPU-only independent audit of retained actual GPU inference evidence."""
import bisect
import collections
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import statistics
import argparse
import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parent
MODES=['results','dual','dual-mps','single-mps','hybrid','hybrid-wide','staggered','staggered-hybrid','staggered-wide']
def read(p):return json.loads(p.read_text())
def utc(x):return dt.datetime.fromisoformat(x).timestamp()
def stats(x):return {'mean':float(np.mean(x)),'p50':float(np.percentile(x,50)),'p95':float(np.percentile(x,95)),'max':max(x)} if x else None
def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main(selected_id=None):
    result={'audit_kind':'CPU-only audit; all inference hardware is independently recorded per rental','GPUs':[]}
    for base in sorted((ROOT/'retained').iterdir()):
        if selected_id is not None and base.name!=str(selected_id):continue
        if not (base/'BACKUP_VERIFIED.json').exists():continue
        proof=read(base/'provenance.json');ticket=proof['ticket'];price=ticket['hourly'];phases=[];counters=[];waves=0;offers=0;warm=0
        assert len(proof['product_python_sources'])==24
        for row in proof['product_python_sources']:
            local=Path('/Users/ahmadsmacair/omnivoice-triton/omnivoice-triton')/row['path'].removeprefix('source/')
            assert digest(local)==row['sha256'],local
        for mode in MODES:
            path=base/mode
            if not path.exists():continue
            assert (path/'COMPLETE.json').exists(),f'Incomplete {base.name}/{mode}'
            for before_path in path.rglob('health-before.json'):
                parent=before_path.parent;before=read(before_path);after=read(parent/'health-after.json')
                for health in [before,after]:
                    assert health['response_cache']['enabled'] is False and health['response_cache']['hits']==0 and health['response_cache']['coalesced']==0
                    assert health['device']=='cuda' and health['dtype']=='fp16' and not health['load_asr']
                    assert health['runner']==('hybrid' if mode in ['hybrid','hybrid-wide','staggered-hybrid','staggered-wide'] else 'triton')
                    assert health['llm_use_cache'] is False and not health['fused_rope'] and not health['sage_attention']
                a=before['batcher_queue']['benchmark_raw_burst_v1'];z=after['batcher_queue']['benchmark_raw_burst_v1']
                success=z['members_succeeded']-a['members_succeeded'];failed=z['members_failed']-a['members_failed']
                assert failed==0,(mode,parent,failed)
                group=parent.parent if parent.name.startswith('server') else parent
                expected=0
                for requests in group.glob('*/requests.json'):
                    if requests.parent.name!='warmup':expected+=sum(r['status']==200 and r['signal_valid'] and (not parent.name.startswith('server') or r['server_index']==int(parent.name[-1])) for r in read(requests))
                assert success==expected,(base.name,mode,parent,success,expected)
                counters.append({'path':parent.relative_to(base).as_posix(),'actual_timed_inference_successes':success,'actual_inference_failures':failed,'runner':after['runner'],'cuda_graph_capture_count':after.get('cuda_graph_capture_count'),'cuda_graph_shapes':after.get('cuda_graph_shapes'),'graph_memory_before':before.get('inference_optimizations',{}).get('cuda_graph_memory'),'graph_memory_after':after.get('inference_optimizations',{}).get('cuda_graph_memory')})
            for request_path in sorted(path.rglob('requests.json')):
                rows=read(request_path);valid=[]
                for r in rows:
                    p=request_path.parent/r['path'];assert digest(p)==r['sha256'] and p.stat().st_size==r['bytes']
                    assert r['sent']['mode']=='clone' and r['sent']['num_step']=='16' and r['sent']['denoise']=='true' and r['sent']['postprocess_mode']=='off'
                    assert r['sent']['text']==r['case']['text']
                    if r['status']==200:
                        audio,sr=sf.read(p,dtype='float32',always_2d=True)
                        assert sr==24000 and audio.shape[1]==1 and len(audio)>0 and np.isfinite(audio).all()
                        assert np.max(np.abs(audio))>.001 and np.sqrt(np.mean(audio**2))>1e-5 and r['signal_valid']
                        assert abs(len(audio)/sr-r['audio_duration_s'])<1e-9
                        waves+=1;valid.append(r)
                    else:assert not r['signal_valid']
                if request_path.parent.name=='warmup':warm+=len(rows);continue
                offers+=len(rows);s=read(request_path.parent/'summary.json');wall=s['wall_s']
                assert s['offers']==len(rows) and s['valid_complete_wavs']==len(valid)
                assert abs(s['complete_requests_per_s']-len(valid)/wall)<1e-10
                assert abs(s['request_latency_p95_ms']-np.percentile([r['latency_ms'] for r in valid],95))<1e-6
                assert max(r['finish_offset_s'] for r in rows)<=wall
                assert s['valid_complete_within_first_second']==sum(r['finish_offset_s']<=1 for r in valid)
                group=request_path.parent.parent;telemetry=[]
                p=group/'nvidia-telemetry.csv'
                barrier=statistics.median(utc(r['utc'])-r['start_offset_s'] for r in rows)
                if p.exists():
                    for raw in csv.DictReader(p.open()):
                        raw={k.strip():v.strip() for k,v in raw.items()}
                        try:t=dt.datetime.strptime(raw['timestamp'],'%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=dt.UTC).timestamp()
                        except ValueError:continue
                        if barrier<=t<=barrier+wall:telemetry.append(raw)
                gpu={}
                for key in ['utilization.gpu [%]','utilization.memory [%]','memory.used [MiB]','power.draw [W]','clocks.current.sm [MHz]','clocks.current.memory [MHz]','temperature.gpu']:
                    values=[]
                    for raw in telemetry:
                        try:values.append(float(raw[key].split()[0]))
                        except (ValueError,KeyError):pass
                    gpu[key]=stats(values)
                s.update(GPU=ticket['gpu'],rental_id=ticket['id'],mode=mode,path=request_path.parent.relative_to(base).as_posix(),hourly_USD=price,
                         USD_per1000_valid=price*1000/(3600*s['complete_requests_per_s']),
                         USD_per_generated_audio_hour=price/s['audio_seconds_per_wall_second'],
                         GPU_100ms_samples=len(telemetry),GPU_telemetry=gpu,quality_note='Signal validity only; see separate blind ASR screens')
                if mode.startswith('staggered'):
                    aggregate=next(z for z in read(path/'summary.json') if z['phase']==s['phase'])
                    s['target_arrival_rate']=aggregate['target_arrival_rate']
                    deadlines=[(r['finish_offset_s']-r['scheduled_offset_s'])*1000 for r in valid]
                    for deadline,field in [(500,'valid_full_WAV_within500ms_from_scheduled'),(1000,'valid_full_WAV_within1s_from_scheduled')]:
                        assert s[field]==sum(d<=deadline for d in deadlines)
                        count=s[field];s[f'all_offers_SLO_{deadline}ms_fraction']=count/len(rows)
                        s[f'SLO_{deadline}ms_goodput_per_s']=count/wall
                        s[f'SLO_{deadline}ms_successes_per_USD']=count/wall*3600/price
                    s['rejection_fraction']=sum(r['status']==429 for r in rows)/len(rows)
                phases.append(s)
        profile=[]
        proot=base/'profile-retry' if (base/'profile-retry/COMPLETE.json').exists() else base/'profile'
        for b in [1,8]:
            p=proot/f'B{b}-profile-summary.json'
            if not p.exists():continue
            s=read(p);es=read(proot/f'B{b}-trace.json')['traceEvents'];kernels=[e for e in es if e.get('cat')=='kernel']
            ranges=sorted([e for e in es if e.get('name')=='TTS_LLM_forward' and e.get('cat')=='user_annotation'],key=lambda x:x['ts'])
            assert len(ranges)==16
            ids=set();starts=[e['ts'] for e in ranges]
            for e in es:
                if e.get('cat')!='cuda_runtime':continue
                i=bisect.bisect_right(starts,e['ts'])-1
                if i>=0 and e.get('tid')==ranges[i]['tid'] and e['ts']<ranges[i]['ts']+ranges[i]['dur']:ids.add(e['args'].get('correlation'))
            llm=[e for e in kernels if e['args'].get('correlation') in ids];obs=read(proot/f'B{b}-linear-observations.json')
            s.update(llm_forward_ranges=len(ranges),llm_correlated_kernel_ms=sum(e['dur'] for e in llm)/1000,
                     llm_correlated_kernel_fraction=sum(e['dur'] for e in llm)/sum(e['dur'] for e in kernels) if kernels else None,
                     fp16_linear_FLOPs=sum(x['FLOPs'] for x in obs if x['dtype']=='torch.float16'),fp32_linear_FLOPs=sum(x['FLOPs'] for x in obs if x['dtype']=='torch.float32'))
            profile.append(s)
        controlled=[s for s in phases if not s['mode'].startswith('staggered')]
        best=max(controlled,key=lambda x:x['complete_requests_per_s'])
        low=min([s for s in controlled if s['concurrency']==1],key=lambda x:x['request_latency_p95_ms'])
        g={'id':ticket['id'],'GPU':ticket['gpu'],'hourly_USD':price,'hardware':read(base/'profile/hardware.json'),'provenance':proof,'timed_offers':offers,'warmup_offers':warm,'valid_load_WAVs_verified':waves,'phases':phases,'counter_deltas':counters,'profile':profile,'best_finite_throughput':best,'best_sequential_p95':low}
        for name in ['measurements.json','cpu-thread-sensitivity.json','microbenchmarks.json']:
            if (proot/name).exists():g[name.removesuffix('.json')]=read(proot/name)
        result['GPUs'].append(g)
        print(json.dumps({'GPU':ticket['gpu'],'timed_offers':offers,'valid_WAVs':waves,'best_mode':best['mode'],'best_RPS':best['complete_requests_per_s'],'best_cost_per1000':best['USD_per1000_valid'],'best_C1_p95':low['request_latency_p95_ms']}),flush=True)
    target=ROOT/(f'analysis-{selected_id}.json' if selected_id is not None else 'analysis.json')
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--id',type=int);args=parser.parse_args();main(args.id)
