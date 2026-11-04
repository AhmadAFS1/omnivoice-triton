"""Finite GPU-test stage controller; physical card is verified before inference."""
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).resolve().parent
ticket=json.loads((root/'ticket.json').read_text())
identity=subprocess.check_output(['nvidia-smi','--query-gpu=name,uuid,memory.total,driver_version,power.limit','--format=csv,noheader'],text=True).strip()
assert len(identity.splitlines())==1 and identity.startswith('NVIDIA GeForce '+ticket['gpu']+',')
env={**os.environ,'HF_HOME':str(root/'hf-cache'),'OMP_NUM_THREADS':'8','MKL_NUM_THREADS':'8','OPENBLAS_NUM_THREADS':'8','TOKENIZERS_PARALLELISM':'false',
    'TASK_GPU_UUID':identity.split(',')[1].strip(),'OMNIVOICE_DUAL_BATCH1_ONLY':'1'}
plan=[('single','run_capacity.py',1900),('profile','profile_pipeline.py',600),('ordinary-dual','run_dual.py',1900),('MPS','run_mps.py',2100),('staggered','run_staggered.py',900)]
outcomes=[]
(root/'matrix-identity.json').write_text(json.dumps({'utc':dt.datetime.now(dt.UTC).isoformat(),'actual_GPU':identity,'ticket':ticket,'environment':{k:env[k] for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','TASK_GPU_UUID']},'stages':plan},indent=2)+'\n')
for name,script,budget in plan:
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
    assert not active,f'GPU not released before {name}: {active}'
    assert (root/script).is_file(),f'Missing finite stage {script}'
    start=time.time()
    with (root/f'{name}-controller.log').open('w') as stream:
        p=subprocess.run([sys.executable,str(root/script)],cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=budget)
    outcomes.append({'stage':name,'script':script,'returncode':p.returncode,'start_utc':dt.datetime.fromtimestamp(start,dt.UTC).isoformat(),'wall_s':time.time()-start})
    (root/'matrix-outcomes.json').write_text(json.dumps(outcomes,indent=2)+'\n')
    print(json.dumps(outcomes[-1]),flush=True)
    time.sleep(2)
(root/'MATRIX_COMPLETE.json').write_text(json.dumps({'utc':dt.datetime.now(dt.UTC).isoformat(),'stage_outcomes':outcomes},indent=2)+'\n')
