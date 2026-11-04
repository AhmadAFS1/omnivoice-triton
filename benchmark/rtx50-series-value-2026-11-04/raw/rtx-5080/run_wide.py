"""Follow the frozen matrix with a matched larger graph-cache configuration."""
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).resolve().parent
deadline=time.monotonic()+3300
while not (root/'ALL_TTS_COMPLETE').exists():
    assert time.monotonic()<deadline
    time.sleep(2)
assert not (root/'FINAL_TTS_COMPLETE').exists()
outcomes=[]
env={**os.environ,'OMP_NUM_THREADS':'8','MKL_NUM_THREADS':'8','OPENBLAS_NUM_THREADS':'8','TOKENIZERS_PARALLELISM':'false'}
for script in ['run_hybrid_wide.py','run_staggered_wide.py']:
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip();assert not active,active
    t=time.time()
    with (root/(script+'.log')).open('w') as f:
        p=subprocess.run([sys.executable,str(root/script)],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=1200)
    outcomes.append({'script':script,'returncode':p.returncode,'start_utc':dt.datetime.fromtimestamp(t,dt.UTC).isoformat(),'wall_s':time.time()-t})
    (root/'wide-outcomes.json').write_text(json.dumps(outcomes,indent=2)+'\n');print(json.dumps(outcomes[-1]),flush=True)
(root/'FINAL_TTS_COMPLETE').write_text(json.dumps({'utc':dt.datetime.now(dt.UTC).isoformat(),'outcomes':outcomes},indent=2)+'\n')
