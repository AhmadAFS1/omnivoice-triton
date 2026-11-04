"""CPU-only audit of real RTX4070 evidence; never performs GPU inference."""
import bisect
import collections
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import statistics

import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parent


def read(p):return json.loads(p.read_text())


def utc(s):return dt.datetime.fromisoformat(s).timestamp()


def stats(xs):
    return {'mean':float(np.mean(xs)),'p50':float(np.percentile(xs,50)),'p95':float(np.percentile(xs,95)),'max':max(xs)} if xs else None


def main():
    output={'audit_kind':'CPU-only raw evidence audit; inference hardware is in retained/profile/hardware.json',
        'hardware':read(ROOT/'retained/profile/hardware.json'),'phases':[],'counter_deltas':[], 'wave_files':0,'timed_requests':0,'profiles':{}}
    refs={r['id']:r for r in read(ROOT/'references/manifest.json')}
    resource_path=ROOT/'retained/mps-control/host-resources.jsonl'
    resources=[json.loads(x) for x in resource_path.read_text().splitlines()] if resource_path.exists() else []
    for mode in ['results','dual','dual-mps','single-mps']:
        base=ROOT/'retained'/mode
        if not base.exists():continue
        assert (base/'COMPLETE.json').exists(),f'{mode} incomplete'
        for bdir in sorted(base.glob('batch*')):
            gpu_rows=[]
            p=bdir/'nvidia-telemetry.csv'
            if p.exists():
                for r in csv.DictReader(p.open()):
                    r={k.strip():v.strip() for k,v in r.items()}
                    timestamp=dt.datetime.strptime(r['timestamp'],'%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=dt.UTC).timestamp()
                    gpu_rows.append((timestamp,r))
            # Each batch-size arm has 180 timed inference executions in total.
            servers=list(bdir.glob('server[0-9]')) or [bdir]
            delta=0
            for server in servers:
                before=read(server/'health-before.json');after=read(server/'health-after.json')
                assert not after['response_cache']['enabled'] and after['runner']=='triton' and after['fused_rope'] is False
                a=before['batcher_queue']['benchmark_raw_burst_v1'];z=after['batcher_queue']['benchmark_raw_burst_v1']
                count=z['members_succeeded']-a['members_succeeded'];delta+=count
                assert z['members_failed']-a['members_failed']==0
            assert delta==180,(mode,bdir.name,delta)
            output['counter_deltas'].append({'mode':mode,'batch':bdir.name,'actual_timed_inference_successes':delta})
            for p in sorted(bdir.glob('*/requests.json')):
                rows=read(p);assert len(rows)==60
                for r in rows:
                    path=p.parent/r['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==r['sha256']
                    wave,sr=sf.read(path,dtype='float32',always_2d=True)
                    assert r['status']==200 and sr==24000 and wave.shape[1]==1 and len(wave)>0 and np.isfinite(wave).all()
                    assert float(np.max(np.abs(wave)))>.001 and float(np.sqrt(np.mean(wave*wave)))>1e-5
                    assert r['sent']['mode']=='clone' and r['sent']['num_step']=='16' and r['sent']['postprocess_mode']=='off'
                    assert r['case']['reference_id'] in refs and r['sent']['text']==r['case']['text']
                    output['wave_files']+=1
                if p.parent.name=='warmup':continue
                summary=read(p.parent/'summary.json');wall=summary['wall_s']
                # Wall is the recorded whole cohort, not mean request latency.
                assert abs(summary['complete_requests_per_s']-60/wall)<1e-12
                assert abs(summary['request_latency_p95_ms']-float(np.percentile([r['latency_ms'] for r in rows],95)))<1e-8
                first=sum(r['finish_offset_s']<=1 for r in rows);assert first==summary['valid_complete_within_first_second']
                assert max(r['finish_offset_s'] for r in rows)<=wall
                barrier=statistics.median(utc(r['utc'])-r['start_offset_s'] for r in rows)
                selected=[r for t,r in gpu_rows if barrier<=t<=barrier+wall]
                gpu={}
                for key in ['utilization.gpu [%]','utilization.memory [%]','memory.used [MiB]','power.draw [W]','clocks.current.sm [MHz]','clocks.current.memory [MHz]','temperature.gpu']:
                    xs=[]
                    for r in selected:
                        try:xs.append(float(r[key].split()[0]))
                        except (KeyError,ValueError):pass
                    gpu[key]=stats(xs)
                host=[r for r in resources if barrier<=utc(r['utc'])<=barrier+wall and len(r['servers'])==len(servers)]
                cpu=[sum(s['cpu_percent_one_core_100'] for s in r['servers']) for r in host if all(s['cpu_percent_one_core_100'] is not None for s in r['servers'])]
                rss=[sum(s['rss_bytes'] for s in r['servers']) for r in host]
                summary.update(mode=mode,server_processes=len(servers),gpu_100ms_samples=len(selected),gpu_telemetry=gpu,
                    host_100ms_samples=len(host),server_cpu_percent_sum_one_core_100=stats(cpu),server_RSS_bytes_sum=stats(rss),
                    reference_ids=sorted({r['case']['reference_id'] for r in rows}),languages=len({r['case']['language'] for r in rows}),
                    resource_caveat='RSS sums count shared resident pages multiple times; CPU percentages cover TTS processes, excluding MPS daemon/driver/helper. Samples may miss brief peaks.')
                output['phases'].append(summary);output['timed_requests']+=60
    for b in [1,8]:
        root=ROOT/'retained/profile';es=read(root/f'B{b}-trace.json')['traceEvents']
        ranges=sorted([e for e in es if e.get('name')=='TTS_LLM_forward' and e.get('cat')=='user_annotation'],key=lambda x:x['ts'])
        assert len(ranges)==16
        starts=[e['ts'] for e in ranges];ids=set()
        for e in es:
            if e.get('cat')!='cuda_runtime':continue
            i=bisect.bisect_right(starts,e['ts'])-1
            if i>=0 and e.get('tid')==ranges[i]['tid'] and e['ts']<ranges[i]['ts']+ranges[i]['dur']:ids.add(e['args'].get('correlation'))
        kernels=[e for e in es if e.get('cat')=='kernel'];llm=[e for e in kernels if e['args'].get('correlation') in ids]
        obs=read(root/f'B{b}-linear-observations.json')
        fp16=[x for x in obs if x['dtype']=='torch.float16'];fp32=[x for x in obs if x['dtype']=='torch.float32']
        # Raw shape metadata used a 2-byte estimate for every dtype. The FP32
        # codec's derived tensor-byte estimates below correct that raw field.
        corrected=[{**x,'dtype_bytes':2 if x['dtype']=='torch.float16' else 4,'corrected_minimum_tensor_bytes':(2 if x['dtype']=='torch.float16' else 4)*(x['M']*x['K']+x['N']*x['K']+x['M']*x['N'])} for x in obs]
        (ROOT/f'B{b}-linear-observations-dtype-corrected.json').write_text(json.dumps(corrected,indent=2)+'\n')
        s=read(root/f'B{b}-profile-summary.json')
        s.update(llm_forward_ranges=16,llm_correlated_kernel_ms=sum(e['dur'] for e in llm)/1000,
            llm_correlated_kernel_fraction=sum(e['dur'] for e in llm)/sum(e['dur'] for e in kernels),
            fp16_linear_FLOPs=sum(x['FLOPs'] for x in fp16),fp32_codec_linear_FLOPs=sum(x['FLOPs'] for x in fp32),
            fp16_FLOPs_per_completed_request=sum(x['FLOPs'] for x in fp16)/b,
            per_kernel_category_percent={k:v['sum_ms']/s['kernel_sum_ms']*100 for k,v in s['kernel_categories'].items()})
        output['profiles'][str(b)]=s
    (ROOT/'analysis.json').write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'timed_requests':output['timed_requests'],'saved_load_WAVs_verified':output['wave_files'],'profiles':len(output['phases'])}))


if __name__=='__main__':main()
