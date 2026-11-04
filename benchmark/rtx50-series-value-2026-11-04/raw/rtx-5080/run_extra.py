"""Wait for the immutable eager/MPS matrix, then test existing hybrid runner."""
import datetime as dt
import json
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).resolve().parent
deadline=time.monotonic()+3000
while not (root/'MATRIX_COMPLETE.json').exists():
    assert time.monotonic()<deadline,'Main matrix did not finish in finite budget'
    time.sleep(2)
active=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
assert not active
with (root/'hybrid-controller.log').open('w') as stream:
    p=subprocess.run([sys.executable,str(root/'run_hybrid.py')],cwd=root,stdout=stream,stderr=subprocess.STDOUT,timeout=2500)
(root/'EXTRA_COMPLETE.json').write_text(json.dumps({'utc':dt.datetime.now(dt.UTC).isoformat(),'hybrid_returncode':p.returncode},indent=2)+'\n')
