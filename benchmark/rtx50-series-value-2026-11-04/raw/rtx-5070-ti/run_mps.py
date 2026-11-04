"""Scoped optional MPS qualification and dual-server screen on owned RTX4070.

No GPU compute mode, clock, power or host permission changes. The daemon/pipe is
private to this task and quit in finally. Inaccessible MPS is retained as a skip.
"""
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import psutil

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'mps-control'


def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')


def main():
    OUT.mkdir(exist_ok=False);pipe=OUT/'pipe';logs=OUT/'logs';pipe.mkdir();logs.mkdir()
    env={**os.environ,'CUDA_VISIBLE_DEVICES':os.environ['TASK_GPU_UUID'],
        'CUDA_MPS_PIPE_DIRECTORY':str(pipe),'CUDA_MPS_LOG_DIRECTORY':str(logs),'OMNIVOICE_DUAL_OUTPUT':'dual-mps','OMNIVOICE_DUAL_BATCH1_ONLY':'1'}
    def control(text):
        p=subprocess.run(['nvidia-cuda-mps-control'],input=text+'\n',text=True,capture_output=True,timeout=15,env=env)
        return {'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
    probe=None;status='failed';outcome={};stop=threading.Event();sampler=None
    try:
        p=subprocess.run(['nvidia-cuda-mps-control','-d'],text=True,capture_output=True,timeout=15,env=env)
        write(OUT/'daemon-start.json',{'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'scoped_environment':{k:env[k] for k in ['CUDA_VISIBLE_DEVICES','CUDA_MPS_PIPE_DIRECTORY','CUDA_MPS_LOG_DIRECTORY']}})
        if p.returncode:
            status='unsupported_or_inaccessible';return
        code="import torch,json,pathlib,sys; a=torch.ones(1,device='cuda'); torch.cuda.synchronize(); pathlib.Path(sys.argv[1]).write_text(json.dumps({'gpu':torch.cuda.get_device_name(0),'pid':__import__('os').getpid()})); sys.stdin.readline()"
        log=(OUT/'probe.log').open('w')
        probe=subprocess.Popen([sys.executable,'-c',code,str(OUT/'probe-client.json')],env=env,stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,text=True)
        for _ in range(150):
            if (OUT/'probe-client.json').exists() or probe.poll() is not None:break
            time.sleep(.2)
        info=control('get_server_list');outcome['probe_server_list']=info
        pids=[p for p in info['stdout'].split() if p.isdigit()];clients={}
        for p in pids:clients[p]=control('get_client_list '+p)
        outcome['probe_clients']=clients
        connected={int(x) for r in clients.values() for x in r['stdout'].split() if x.isdigit()}
        if probe.pid not in connected:
            status='unsupported_or_inaccessible';return
        outcome['verified_probe_client']=True
        probe.stdin.close();probe.wait(timeout=15);log.close();probe=None
        def sample_resources():
            previous={}
            with (OUT/'host-resources.jsonl').open('w') as stream:
                while not stop.is_set():
                    now=time.perf_counter();items=[]
                    for process in psutil.process_iter(['pid','cmdline']):
                        try:
                            if str(ROOT/'server_observer.py') not in (process.info['cmdline'] or []):continue
                            cpu=process.cpu_times();seconds=cpu.user+cpu.system;pid=process.pid
                            old=previous.get(pid);previous[pid]=(now,seconds)
                            items.append({'pid':pid,'cpu_percent_one_core_100':(seconds-old[1])/(now-old[0])*100 if old else None,
                                'cpu_seconds':seconds,'rss_bytes':process.memory_info().rss,'threads':process.num_threads(),'command':process.info['cmdline']})
                        except (psutil.NoSuchProcess,psutil.AccessDenied):continue
                    stream.write(json.dumps({'utc':dt.datetime.now(dt.UTC).isoformat(),'perf_counter':now,'servers':items})+'\n');stream.flush();stop.wait(.1)
        sampler=threading.Thread(target=sample_resources,daemon=True);sampler.start()
        with (ROOT/'dual-mps.log').open('w') as stream:
            p=subprocess.run([sys.executable,str(ROOT/'run_dual.py')],env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=1900)
        outcome['dual_returncode']=p.returncode
        status='completed' if p.returncode==0 else 'dual_failed'
        if p.returncode==0:
            single_env={**env,'OMNIVOICE_CAPACITY_OUTPUT':'single-mps','OMNIVOICE_SINGLE_BATCH1_ONLY':'1'}
            with (ROOT/'single-mps.log').open('w') as stream:
                p=subprocess.run([sys.executable,str(ROOT/'run_capacity.py')],env=single_env,stdout=stream,stderr=subprocess.STDOUT,timeout=1900)
            outcome['single_mps_returncode']=p.returncode
            if p.returncode:status='single_mps_failed'
    finally:
        stop.set()
        if sampler is not None:sampler.join(timeout=5)
        if probe is not None:
            if probe.stdin and not probe.stdin.closed:probe.stdin.close()
            try:probe.wait(timeout=10)
            except subprocess.TimeoutExpired:probe.kill();probe.wait()
        try:outcome['daemon_quit']=control('quit')
        except Exception as e:outcome['daemon_quit_error']=repr(e)
        write(OUT/'outcome.json',{'utc':dt.datetime.now(dt.UTC).isoformat(),'status':status,**outcome})


if __name__=='__main__':main()
