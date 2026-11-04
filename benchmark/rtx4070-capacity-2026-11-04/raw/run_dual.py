"""Two independent OmniVoice-Triton processes on one physical RTX4070.

Compare with run_capacity's identical full-reference cloned-voice corpus. Each
process has its own weights, queue, prepared voices and CUDA context. No MPS.
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
from run_capacity import ROOT,SOURCE,phase,write


async def run():
    out=ROOT/os.environ.get('OMNIVOICE_DUAL_OUTPUT','dual');out.mkdir(exist_ok=False)
    cases=json.loads((ROOT/'corpus.json').read_text())
    refs=json.loads((ROOT/'references/manifest.json').read_text())
    used={x['reference_id'] for x in cases};summaries=[]
    urls=['http://127.0.0.1:8002','http://127.0.0.1:8003']
    for batch in [1,8]:
        dest=out/f'batch{batch}';dest.mkdir();processes=[];logs=[];telemetry=None
        try:
            for index in range(2):
                d=dest/f'server{index}';d.mkdir()
                cmd=[sys.executable,str(ROOT/'server_observer.py'),'--seed','20261104','--evidence',str(d/'server-seed.json'),
                    '--model',str(ROOT/'model'),'--runner','triton','--device','cuda','--dtype','fp16','--no-asr',
                    '--disable-unused-kv-cache','--batch-collect-ms','0','--max-batch-requests',str(batch),
                    '--max-pending-requests','128','--max-cuda-graphs','0','--audio-decode-batch-size','1',
                    '--response-cache-max-bytes','0','--ip','127.0.0.1','--port',str(8002+index)]
                write(d/'command.json',cmd);log=(d/'server.log').open('w');logs.append(log)
                processes.append(subprocess.Popen(cmd,cwd=SOURCE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
            prompt_ids=[]
            for index,url in enumerate(urls):
                d=dest/f'server{index}'
                async with httpx.AsyncClient(base_url=url,timeout=180,trust_env=False) as c:
                    for _ in range(240):
                        if processes[index].poll() is not None:raise RuntimeError('server exited')
                        try:
                            h=await c.get('/health')
                            if h.status_code==200 and h.json().get('status')=='ok':break
                        except httpx.TransportError:pass
                        await asyncio.sleep(1)
                    else:raise TimeoutError('ready')
                    health=h.json();write(d/'health-start.json',health)
                    assert health['runner']=='triton' and health['response_cache']['enabled'] is False and health['max_batch_requests']==batch
                    ids={};registrations=[]
                    for ref in refs:
                        if ref['id'] not in used:continue
                        p=ROOT/'references'/ref['path'];body=p.read_bytes();assert hashlib.sha256(body).hexdigest()==ref['sha256']
                        t=time.perf_counter();r=await c.post('/clone-prompts',data={'ref_text':ref['text'],'preprocess_prompt':'true'},files={'ref_audio':(p.name,body,'audio/wav')})
                        ms=(time.perf_counter()-t)*1000;r.raise_for_status();ids[ref['id']]=r.json()['prompt_id']
                        registrations.append({'reference':ref,'complete_ms':ms,'response':r.json()})
                    write(d/'registrations.json',registrations);prompt_ids.append(ids)
            await phase(dest/'warmup','warmup',cases,prompt_ids,2,urls)
            if os.environ.get('CUDA_MPS_PIPE_DIRECTORY'):
                server_list=subprocess.run(['nvidia-cuda-mps-control'],input='get_server_list\n',text=True,capture_output=True,timeout=10)
                evidence={'server_list':server_list.stdout,'stderr':server_list.stderr,'model_processes':[p.pid for p in processes],'clients':{}}
                for pid in server_list.stdout.split():
                    if pid.isdigit():
                        r=subprocess.run(['nvidia-cuda-mps-control'],input=f'get_client_list {pid}\n',text=True,capture_output=True,timeout=10)
                        evidence['clients'][pid]={'stdout':r.stdout,'stderr':r.stderr}
                write(dest/'actual-mps-clients.json',evidence)
                clients={int(x) for v in evidence['clients'].values() for x in v['stdout'].split() if x.isdigit()}
                assert {p.pid for p in processes}<=clients,'Both TTS processes must be verified MPS clients'
            telemetry_log=(dest/'nvidia-telemetry.csv').open('w')
            telemetry=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,name,uuid,utilization.gpu,utilization.memory,memory.used,power.draw,power.limit,clocks.sm,clocks.mem,temperature.gpu,pstate','--format=csv','-lms','100'],stdout=telemetry_log,stderr=subprocess.STDOUT)
            async with httpx.AsyncClient(timeout=30,trust_env=False) as c:
                for index,url in enumerate(urls):write(dest/f'server{index}'/'health-before.json',(await c.get(url+'/health')).json())
                for concurrency in [2,8,32]:
                    name=f'dual-b{batch}-c{concurrency}'
                    s=await phase(dest/name,name,cases,prompt_ids,concurrency,urls);s['server_processes']=2;s['server_batch_limit']=batch
                    summaries.append(s);write(out/'summary.json',summaries)
                for index,url in enumerate(urls):write(dest/f'server{index}'/'health-after.json',(await c.get(url+'/health')).json())
        finally:
            if telemetry is not None:telemetry.terminate();telemetry.wait(timeout=10);telemetry_log.close()
            for p in processes:
                if p.poll() is None:
                    os.killpg(p.pid,signal.SIGTERM)
                    try:p.wait(timeout=30)
                    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            for l in logs:l.close()
            write(dest/'exits.json',{'codes':[p.returncode for p in processes],'utc':dt.datetime.now(dt.UTC).isoformat()})
    write(out/'COMPLETE.json',{'utc':dt.datetime.now(dt.UTC).isoformat(),'profiles':len(summaries)})


if __name__=='__main__':asyncio.run(asyncio.wait_for(run(),timeout=1800))
