"""RTX 4070 full-WAV clone capacity; cache disabled; finite 30-minute run.

Independent HTTP connections keep client pool waits out of per-request timings.
Bodies are decoded/hashed/saved after each phase, outside its timing interval.
Product code and generation parameters remain unchanged.
"""
import asyncio
import collections
import datetime as dt
import hashlib
import io
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import httpx
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent
OUT = ROOT / os.environ.get('OMNIVOICE_CAPACITY_OUTPUT','results')
SOURCE = ROOT / 'source'


def write(p, x):
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2) + '\n')


def percentile(x, p):
    return float(np.percentile(x, p)) if x else None


def summarize(rows, wall):
    valid = [r for r in rows if r.get('status') == 200 and r.get('signal_valid')]
    done = [r['finish_offset_s'] for r in valid]
    bins = collections.Counter(int(t) for t in done)
    durations = [r['audio_duration_s'] for r in valid]
    headers = {}
    for key in sorted({k for r in valid for k in r['headers'] if k.endswith('-ms')}):
        values = [float(r['headers'][key]) for r in valid if key in r['headers']]
        headers[key] = {'mean': float(np.mean(values)), 'p50': percentile(values, 50), 'p95': percentile(values, 95)}
    return {
        'offers': len(rows), 'valid_complete_wavs': len(valid),
        'status_counts': dict(collections.Counter(str(r.get('status')) for r in rows)),
        'wall_s': wall, 'complete_requests_per_s': len(valid) / wall,
        'valid_complete_within_first_second': sum(t <= 1 for t in done),
        'request_latency_p50_ms': percentile([r['latency_ms'] for r in valid], 50),
        'request_latency_p95_ms': percentile([r['latency_ms'] for r in valid], 95),
        'valid_request_latency_within_1s': sum(r['latency_ms'] <= 1000 for r in valid),
        'completion_counts_per_1s_from_phase_start': {str(i): bins.get(i, 0) for i in range(math.ceil(wall))},
        'max_completions_in_aligned_1s_bin': max(bins.values(), default=0),
        'audio_duration_mean_s': float(np.mean(durations)) if durations else None,
        'audio_seconds_per_wall_second': sum(durations) / wall,
        'actual_batch_size_distribution_per_response': dict(collections.Counter(r['headers'].get('x-omnivoice-batch-requests') for r in valid)),
        'timing_headers_per_response_ms': headers,
        'note': 'Batch timing is shared by members; response-weighted means are not a sum of GPU work. First-second count starts at the phase barrier. All offers are kept.',
    }


async def phase(out, name, cases, prompt_ids, concurrency, base_urls=None):
    out.mkdir()
    sem = asyncio.Semaphore(concurrency)
    base_urls=base_urls or ['http://127.0.0.1:8002']
    clients = [httpx.AsyncClient(base_url=base_urls[i%len(base_urls)], timeout=180, trust_env=False,
        limits=httpx.Limits(max_connections=1, max_keepalive_connections=1, keepalive_expiry=600)) for i,_ in enumerate(cases)]
    responses = []
    try:
        await asyncio.gather(*(c.get('/healthz') for c in clients))
        barrier = time.perf_counter()

        async def one(i, case):
            async with sem:
                start = time.perf_counter()
                ids=prompt_ids[i%len(base_urls)] if isinstance(prompt_ids,list) else prompt_ids
                data = dict(mode='clone', text=case['text'], language=case['api_language'],
                    prompt_id=ids[case['reference_id']], num_step='16', speed='1.0',
                    denoise='true', postprocess_mode='off', insert_pauses='false')
                row = {'id': f'{name}-{i:03}', 'case': case, 'sent': data,
                    'start_offset_s': start-barrier, 'server_index':i%len(base_urls), 'utc': dt.datetime.now(dt.UTC).isoformat()}
                body = b''
                try:
                    r = await clients[i].post('/generate', data=data)
                    body = r.content
                    end = time.perf_counter()
                    row.update(status=r.status_code, headers={k:v for k,v in r.headers.items() if k.startswith('x-omnivoice')},
                        latency_ms=(end-start)*1000, finish_offset_s=end-barrier, bytes=len(body))
                except Exception as e:
                    end = time.perf_counter()
                    row.update(status=None, error=repr(e), latency_ms=(end-start)*1000, finish_offset_s=end-barrier)
                responses.append((row, body))

        await asyncio.gather(*(one(i,c) for i,c in enumerate(cases)))
        wall = time.perf_counter()-barrier
        # No waveform decoding/disk writes in the measurement window.
        rows=[]
        for row, body in responses:
            row['sha256'] = hashlib.sha256(body).hexdigest()
            suffix='.wav' if row.get('status') == 200 else '.error'
            p=out/(row['id']+suffix);p.write_bytes(body);row['path']=p.name
            row['signal_valid']=False
            if row.get('status') == 200:
                try:
                    audio,sr=sf.read(io.BytesIO(body),dtype='float32',always_2d=True)
                    peak=float(np.abs(audio).max()) if audio.size else 0
                    rms=float(np.sqrt(np.mean(audio*audio))) if audio.size else 0
                    row.update(audio_duration_s=len(audio)/sr, sample_rate=sr,channels=audio.shape[1],peak=peak,rms=rms,
                        signal_valid=bool(audio.size and np.isfinite(audio).all() and peak>0.001 and rms>1e-5 and sr==24000))
                except Exception as e:row['decode_error']=repr(e)
            rows.append(row)
        rows.sort(key=lambda x:x['id'])
        write(out/'requests.json',rows)
        summary=summarize(rows,wall)
        summary.update(phase=name,concurrency=concurrency,clone_mode_only=True,response_cache=False)
        write(out/'summary.json',summary)
        print(json.dumps({k:summary[k] for k in ['phase','offers','valid_complete_wavs','wall_s','complete_requests_per_s','valid_complete_within_first_second','request_latency_p95_ms']}),flush=True)
        return summary
    finally:
        await asyncio.gather(*(c.aclose() for c in clients))


async def run():
    OUT.mkdir(exist_ok=False)
    cases=json.loads((ROOT/'corpus.json').read_text())
    refs=json.loads((ROOT/'references/manifest.json').read_text())
    assert len(cases)==104 and all(c['mode']=='clone' for c in cases)
    used={c['reference_id'] for c in cases}
    all_summary=[]
    for b in ([1] if os.environ.get('OMNIVOICE_SINGLE_BATCH1_ONLY')=='1' else [1,8]):
        dest=OUT/f'batch{b}';dest.mkdir()
        command=[sys.executable,str(ROOT/'server_observer.py'),'--seed','20261104','--evidence',str(dest/'server-seed.json'),
            '--model',str(ROOT/'model'),'--runner','triton','--device','cuda','--dtype','fp16','--no-asr',
            '--disable-unused-kv-cache','--batch-collect-ms','0','--max-batch-requests',str(b),
            '--max-pending-requests','128','--max-cuda-graphs','0','--audio-decode-batch-size','1',
            '--response-cache-max-bytes','0','--ip','127.0.0.1','--port','8002']
        write(dest/'command.json',command)
        log=(dest/'server.log').open('w')
        proc=subprocess.Popen(command,cwd=SOURCE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        telemetry=None
        try:
            async with httpx.AsyncClient(base_url='http://127.0.0.1:8002',timeout=180,trust_env=False) as c:
                start=time.perf_counter()
                for _ in range(240):
                    if proc.poll() is not None:raise RuntimeError('server exited')
                    try:
                        h=await c.get('/health')
                        if h.status_code==200 and h.json().get('status')=='ok':break
                    except httpx.TransportError:pass
                    await asyncio.sleep(1)
                else:raise TimeoutError('ready timeout')
                write(dest/'health-start.json',h.json())
                health=h.json()
                assert health['runner']=='triton' and health['dtype']=='fp16'
                assert health['response_cache']['enabled'] is False
                assert health['load_asr'] is False and health['audio_decode_batch_size']==1
                assert health['max_batch_requests']==b and health['batch_collect_ms']==0
                write(dest/'startup.json',{'model_server_ready_s':time.perf_counter()-start})
                prompt_ids={};registration=[]
                for ref in refs:
                    if ref['id'] not in used:continue
                    p=ROOT/'references'/ref.get('path',ref['file'])
                    assert hashlib.sha256(p.read_bytes()).hexdigest()==ref['sha256']
                    t=time.perf_counter()
                    r=await c.post('/clone-prompts',data={'ref_text':ref['text'],'preprocess_prompt':'true'},
                        files={'ref_audio':(p.name,p.read_bytes(),'audio/wav')})
                    elapsed=(time.perf_counter()-t)*1000;r.raise_for_status()
                    prompt_ids[ref['id']]=r.json()['prompt_id']
                    registration.append({'reference':ref,'complete_ms':elapsed,'response':r.json()})
                write(dest/'registrations.json',registration)
                # Full corpus warmup. Every timed request still runs inference.
                await phase(dest/'warmup','warmup',cases,prompt_ids,1)
                telemetry_log=(dest/'nvidia-telemetry.csv').open('w')
                telemetry=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,name,uuid,utilization.gpu,utilization.memory,memory.used,power.draw,power.limit,clocks.sm,clocks.mem,temperature.gpu,pstate','--format=csv','-lms','100'],stdout=telemetry_log,stderr=subprocess.STDOUT)
                before=(await c.get('/health')).json();write(dest/'health-before.json',before)
                for concurrency in [1,8,32]:
                    name=f'b{b}-c{concurrency}'
                    all_summary.append(await phase(dest/name,name,cases,prompt_ids,concurrency))
                    write(OUT/'summary.json',all_summary)
                write(dest/'health-after.json',(await c.get('/health')).json())
        finally:
            if telemetry is not None:
                telemetry.terminate();telemetry.wait(timeout=10);telemetry_log.close()
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            log.close()
            write(dest/'exit.json',{'server_returncode':proc.returncode,'utc':dt.datetime.now(dt.UTC).isoformat()})
    write(OUT/'COMPLETE.json',{'utc':dt.datetime.now(dt.UTC).isoformat(),'profiles':len(all_summary)})


if __name__=='__main__':
    asyncio.run(asyncio.wait_for(run(),timeout=1800))
