"""Finite sequential GPU ASR shards on the5090; independent of other TTS cards.

Inputs/outputs are hashed; reference text is never passed to the ASR pipeline.
Each shard starts both pinned models afresh. No external transcription API.
"""
import datetime as dt
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import time
from transport import root,remote,opts,entries,connect

ASR_ID=59167099
MODELS=['omniASR_CTC_1B_v2','omniASR_LLM_Unlimited_1B_v2']
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ssh(host,port,command,timeout=35,check=True):
    return subprocess.run(['ssh',*opts,'-p',port,'root@'+host,command],capture_output=True,text=True,timeout=timeout,check=check)
def stage_case(dest,row,src):
    target=dest/'audio'/(row['sha256']+'.wav');target.parent.mkdir(parents=True,exist_ok=True)
    if not target.exists():shutil.copyfile(src,target)
    assert sha(target)==row['sha256']
    return {**row,'path':'audio/'+target.name}
def make_gpu_manifest(gpu):
    base=root/'retained'/str(gpu['id']);dest=root/'quality-input'/str(gpu['id']);dest.mkdir(parents=True,exist_ok=True)
    selected=[('baseline','results/batch1/b1-c1'),('graphs','hybrid-wide/batch1/b1-c1'),('throughput',gpu['best_finite_throughput']['path'])]
    rows=[]
    for arm,phase in selected:
        requests=read(base/phase/'requests.json');assert len(requests)==104
        for r in requests:
            assert r['status']==200 and r['signal_valid']
            item={'id':f"{gpu['id']}-{arm}-{r['case']['id']}",'sha256':r['sha256'],'text':r['case']['text'],'language':r['case']['language'],
                  'lang_id':r['case']['omnilingual_language'],'case':r['case'],'GPU':gpu['GPU'],'rental_id':gpu['id'],'run':arm,'phase':phase,
                  'original_path':str(base/phase/r['path']),'request':r['sent']}
            rows.append(stage_case(dest,item,base/phase/r['path']))
    write(dest/'manifest.json',rows)
    write(dest/'selection.json',{'GPU':gpu['GPU'],'rental_id':gpu['id'],'cases':len(rows),'unique_audio_language_inputs':len({(r['sha256'],r['lang_id']) for r in rows}),'selection':selected,'note':'104 routes each; ASR screen of these three arms only; no MOS/speaker/native-language certification'})
    return dest
def make_refs():
    dest=root/'quality-input'/'references';dest.mkdir(parents=True,exist_ok=True);corpus=read(root/'corpus.json');rows=[]
    for r in read(root/'references/manifest.json'):
        lang=r.get('language',r['id'].split('-')[0]);case=next(c for c in corpus if c['language']==lang)
        item={'id':'reference-'+r['id'],'sha256':r['sha256'],'text':r['text'],'language':lang,'lang_id':case['omnilingual_language'],'run':'reference-calibration','reference':r,'source_generation_hardware':'unverified external Aoede reference; current5090 is ASR hardware only'}
        rows.append(stage_case(dest,item,root/'references'/r['path']))
    write(dest/'manifest.json',rows);return dest
def run_shard(dest,host,port):
    name=dest.name;remote_in=f'{remote}/quality-input/{name}';remote_out=f'{remote}/quality-results/{name}'
    local=root/'quality-results'/name
    if (local/'BACKUP_VERIFIED.json').exists():return
    archive=dest.with_suffix('.tar.gz')
    subprocess.run(['tar','czf',str(archive),'-C',str(dest.parent),name],check=True,timeout=60)
    ssh(host,port,f'mkdir -p {remote}/quality-input {remote}/quality-results')
    subprocess.run(['scp',*opts,'-P',port,str(archive),f'root@{host}:{remote}/quality-input/'],capture_output=True,check=True,timeout=90)
    ssh(host,port,f'cd {remote}/quality-input && tar xzf {shlex.quote(archive.name)}')
    program=f'''import datetime as dt,json,pathlib,subprocess,os,signal
r=pathlib.Path({remote!r});out=pathlib.Path({remote_out!r});out.mkdir(exist_ok=False)
assert (r/'FINAL_TTS_COMPLETE').exists() and (r/'provenance.json').exists() and (r/'ASR_READY.json').exists()
identity=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.total','--format=csv,noheader'],text=True).strip()
assert identity.startswith('NVIDIA GeForce RTX 5090, GPU-694a2f09-ee4a-12f1-07c6-5fa2e0dab6f2,')
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
env={{**os.environ,'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4','TOKENIZERS_PARALLELISM':'false','PYTHONNOUSERSITE':'1'}}
command=[str(r/'asr-venv/bin/python'),str(r/'asr/evaluate.py'),'--manifest',{(remote_in+'/manifest.json')!r},'--output',str(out),'--models',*{MODELS!r},'--dtype','float32','--device','cuda:0']
start=dt.datetime.now(dt.UTC).isoformat()
error=None
with (out/'controller.log').open('w') as f:
 p=subprocess.Popen(command,env=env,cwd=r,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 try:p.wait(timeout=2400)
 except subprocess.TimeoutExpired:
  error='finite ASR shard time budget exceeded';os.killpg(p.pid,signal.SIGKILL);p.wait()
files=[]
import hashlib
for path in sorted(out.rglob('*')):
 if path.is_file():
  with path.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
  files.append({{'path':str(path.relative_to(out)),'bytes':path.stat().st_size,'sha256':digest}})
(out/'backup-manifest.json').write_text(json.dumps({{'files':files}},indent=2)+'\\n')
(out/'CONTROLLER.json').write_text(json.dumps({{'start_utc':start,'end_utc':dt.datetime.now(dt.UTC).isoformat(),'command':command,'returncode':p.returncode,'error':error,'ASR_GPU':identity,'TTS_source_ref':'b7cced4a0274f18aca541f7837c490bb81d8744e','source_sha256':hashlib.sha256((r/'asr/evaluate.py').read_bytes()).hexdigest(),'pinned_assets':json.loads((r/'ASR_READY.json').read_text())}},indent=2)+'\\n')
'''
    compile(program,'quality-worker-'+name+'.py','exec')
    wrapper=root/f'quality-worker-{name}.py';wrapper.write_text(program)
    subprocess.run(['scp',*opts,'-P',port,str(wrapper),f'root@{host}:{remote}/'],capture_output=True,check=True,timeout=30)
    code=f'cd {remote}; if test ! -d {shlex.quote(remote_out)}; then nohup source/.venv/bin/python {shlex.quote(wrapper.name)} > quality-worker-{name}.log 2>&1 < /dev/null & echo $!; fi'
    print(json.dumps({'stage':'ASR_started','shard':name,'utc':dt.datetime.now(dt.UTC).isoformat(),'result':ssh(host,port,code).stdout.strip()}),flush=True)
    deadline=time.monotonic()+2500
    while True:
        p=ssh(host,port,f'cat {shlex.quote(remote_out)}/CONTROLLER.json',check=False)
        if p.returncode==0:receipt=json.loads(p.stdout);break
        assert time.monotonic()<deadline,'ASR shard wait exceeded'
        time.sleep(20)
    local.mkdir(parents=True,exist_ok=True)
    tar_path=local/'evidence.tar.gz'
    with tar_path.open('wb') as f:
        subprocess.run(['ssh',*opts,'-p',port,'root@'+host,f'cd {shlex.quote(remote_out)} && tar czf - .'],stdout=f,check=True,timeout=180)
    subprocess.run(['tar','xzf',str(tar_path),'-C',str(local)],check=True,timeout=90)
    manifest=read(local/'backup-manifest.json')
    for r in manifest['files']:
        p=local/r['path'];assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'],p
    write(local/'BACKUP_VERIFIED.json',{'utc':dt.datetime.now(dt.UTC).isoformat(),'files':len(manifest['files']),'bytes':sum(r['bytes'] for r in manifest['files']),'all_size_and_SHA256_match':True,'controller_returncode':receipt['returncode']})
    print(json.dumps({'stage':'ASR_retained','shard':name,'returncode':receipt['returncode'],'cases':read(local/'summary.json')['cases'] if (local/'summary.json').exists() else None}),flush=True)
    assert receipt['returncode']==0,receipt
def main():
    host,port,_=connect(next(e for e in entries() if e['id']==ASR_ID))
    deadline=time.monotonic()+6000
    while not (root/'retained'/str(ASR_ID)/'BACKUP_VERIFIED.json').exists() or ssh(host,port,f'test -f {remote}/ASR_READY.json && test -f {remote}/provenance.json',check=False).returncode:
        assert time.monotonic()<deadline
        time.sleep(20)
    run_shard(make_refs(),host,port)
    expected={59166781,59167078,59167099,59167113,59168131,59168148};done=set()
    while done!=expected:
        for eid in [59167099,59167113,59168148,59168131,59166781,59167078]:
            if eid in done or not (root/'retained'/str(eid)/'BACKUP_VERIFIED.json').exists():continue
            interpreter=str(root/'analysis-venv/bin/python')
            subprocess.run([interpreter,str(root/'analyze.py'),'--id',str(eid)],check=True,timeout=240)
            gpu=read(root/f'analysis-{eid}.json')['GPUs'][0]
            run_shard(make_gpu_manifest(gpu),host,port);done.add(eid)
        assert time.monotonic()<deadline,'Quality queue wait exceeded'
        if done!=expected:time.sleep(20)
    write(root/'QUALITY_COMPLETE.json',{'utc':dt.datetime.now(dt.UTC).isoformat(),'GPU_shards':sorted(done),'reference_shard':True,'models':MODELS,'dtype':'float32','ASR_host':ASR_ID,'paid_API_calls':0})
if __name__=='__main__':main()
