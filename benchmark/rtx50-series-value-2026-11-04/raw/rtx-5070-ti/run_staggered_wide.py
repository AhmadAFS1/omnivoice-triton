"""Runtime-verified GPU: fresh cloned full-WAV Poisson arrival/SLO assay.

Five40-second fixed-seed schedules, same across cards. All offers including429
remain in the denominator. Deadline starts at scheduled arrival, not dispatch.
"""
import asyncio
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import httpx
import numpy as np
from scheduled_helpers import ROOT,SOURCE,phase,write


async def main():
    out=ROOT/'staggered-wide';out.mkdir(exist_ok=False)
    cases=json.loads((ROOT/'corpus.json').read_text());refs=json.loads((ROOT/'references/manifest.json').read_text());assert len(cases)==104
    cmd=[sys.executable,str(ROOT/'server_observer.py'),'--seed','20261104','--evidence',str(out/'server-seed.json'),
        '--model',str(ROOT/'model'),'--runner','hybrid','--device','cuda','--dtype','fp16','--no-asr','--disable-unused-kv-cache',
        '--batch-collect-ms','0','--max-batch-requests','1','--max-pending-requests','4','--max-cuda-graphs','64','--max-cuda-graph-row-tokens','65536',
        '--audio-decode-batch-size','1','--response-cache-max-bytes','0','--ip','127.0.0.1','--port','8002']
    write(out/'command.json',cmd);log=(out/'server.log').open('w')
    process=subprocess.Popen(cmd,cwd=SOURCE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);telemetry=None
    try:
        async with httpx.AsyncClient(base_url='http://127.0.0.1:8002',timeout=180,trust_env=False) as client:
            for _ in range(240):
                if process.poll() is not None:raise RuntimeError('server exited')
                try:
                    h=await client.get('/health')
                    if h.status_code==200 and h.json().get('status')=='ok':break
                except httpx.TransportError:pass
                await asyncio.sleep(1)
            else:raise TimeoutError('ready')
            h=h.json();assert h['max_batch_requests']==1 and not h['response_cache']['enabled'] and h['batcher_queue']['max_pending_requests']==4
            write(out/'health-start.json',h);ids={};registrations=[]
            for ref in refs:
                p=ROOT/'references'/ref['path'];data=p.read_bytes();assert hashlib.sha256(data).hexdigest()==ref['sha256']
                t=time.perf_counter();r=await client.post('/clone-prompts',data={'ref_text':ref['text'],'preprocess_prompt':'true'},files={'ref_audio':(p.name,data,'audio/wav')});ms=(time.perf_counter()-t)*1000
                r.raise_for_status();ids[ref['id']]=r.json()['prompt_id'];registrations.append({'reference':ref,'complete_ms':ms,'response':r.json()})
            write(out/'registrations.json',registrations)
            await phase(out/'warmup','warmup',cases,ids,1)
            telemetry_log=(out/'nvidia-telemetry.csv').open('w');telemetry=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,name,uuid,utilization.gpu,utilization.memory,memory.used,power.draw,power.limit,clocks.sm,clocks.mem,temperature.gpu,pstate','--format=csv','-lms','100'],stdout=telemetry_log,stderr=subprocess.STDOUT)
            write(out/'health-before.json',(await client.get('/health')).json())
            rng=np.random.default_rng(20261104);indices=rng.permutation(len(cases));corpus=[cases[int(i)] for i in indices];summaries=[];plans=[]
            for rate in [.5,1,2,4,8]:
                rng=np.random.default_rng(20261104+int(rate*100));offsets=[0.0]
                while True:
                    next_time=offsets[-1]+float(rng.exponential(1/rate))
                    if next_time>=40:break
                    offsets.append(next_time)
                name='poisson-rps'+str(rate).replace('.','_');selected=[corpus[i%104] for i in range(len(offsets))]
                plan={'phase':name,'target_arrival_rate':rate,'window_s':40,'offsets_s':offsets,'case_ids':[c['id'] for c in selected],'pending_limit':4,'seed':20261104+int(rate*100)}
                plans.append(plan);write(out/'plans.json',plans)
                result=await phase(out/name,name,selected,ids,len(selected),offsets=offsets,observation_s=40)
                result['target_arrival_rate']=rate;summaries.append(result);write(out/'summary.json',summaries)
            write(out/'health-after.json',(await client.get('/health')).json())
    finally:
        if telemetry is not None:telemetry.terminate();telemetry.wait(timeout=10);telemetry_log.close()
        if process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
        log.close();write(out/'exit.json',{'returncode':process.returncode,'utc':dt.datetime.now(dt.UTC).isoformat()})
    write(out/'COMPLETE.json',{'utc':dt.datetime.now(dt.UTC).isoformat()})


if __name__=='__main__':asyncio.run(asyncio.wait_for(main(),timeout=800))
