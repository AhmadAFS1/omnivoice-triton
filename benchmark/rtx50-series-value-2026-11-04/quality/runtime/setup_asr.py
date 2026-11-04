"""Stage pinned local Omnilingual CTC/LLM judges only after all TTS timing."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parent
deadline=time.monotonic()+4800
while not (ROOT/'ALL_TTS_COMPLETE').exists():
    assert time.monotonic()<deadline,'Timed TTS did not release the GPU'
    time.sleep(3)
assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip()
assert not (ROOT/'asr-venv').exists()
uv='/root/.local/bin/uv';python=str(ROOT/'asr-venv/bin/python')
env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONNOUSERSITE':'1','PYTHONPATH':'','UV_CACHE_DIR':str(ROOT/'uv-cache'),'UV_LINK_MODE':'hardlink','UV_HTTP_TIMEOUT':'600','UV_CONCURRENT_DOWNLOADS':'3','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
def run(args,timeout=1000):
    print(json.dumps({'utc':dt.datetime.now(dt.UTC).isoformat(),'command':args}),flush=True)
    subprocess.run(args,env=env,cwd=ROOT,timeout=timeout,check=True)
before=subprocess.check_output([uv,'pip','freeze','--python',str(ROOT/'source/.venv/bin/python')],env=env)
run([uv,'venv','--python','/usr/bin/python3.12',str(ROOT/'asr-venv')])
base=[uv,'pip','install','--python',python,'--index-strategy','unsafe-best-match']
run([*base,'torch==2.8.0+cu128','torchaudio==2.8.0+cu128','--index-url','https://download.pytorch.org/whl/cu128'])
run([*base,'fairseq2n==0.6+cu128','--extra-index-url','https://fair.pkg.atmeta.com/fairseq2/whl/pt2.8.0/cu128'])
run([*base,'fairseq2[arrow]==0.6','omnilingual-asr @ git+https://github.com/facebookresearch/omnilingual-asr.git@81f51e224ce9e74b02cc2a3eaf21b2d91d743455','numpy==1.26.4','pyarrow==20.0.0','soundfile==0.13.1','jiwer==4.0.0'])
run([uv,'pip','check','--python',python])
freeze=subprocess.check_output([uv,'pip','freeze','--python',python],env=env);(ROOT/'asr-freeze.txt').write_bytes(freeze)
run([python,str(ROOT/'asr/prefetch.py'),'--output',str(ROOT/'asr/assets.json')],1800)
actual=json.loads((ROOT/'asr/assets.json').read_text())['assets'];expected=json.loads((ROOT/'asr/expected-assets.json').read_text())['assets']
for e in expected:
    row=next(x for x in actual if x['name']==e['name']);assert (row['bytes'],row['sha256'])==(e['bytes'],e['sha256'])
after=subprocess.check_output([uv,'pip','freeze','--python',str(ROOT/'source/.venv/bin/python')],env=env);assert before==after
(ROOT/'ASR_READY.json').write_text(json.dumps({'utc':dt.datetime.now(dt.UTC).isoformat(),'assets':actual,'baseline_freeze_unchanged':True,'freeze_sha256':hashlib.sha256(freeze).hexdigest(),'model_inference_performed':False,'ASR_source_commit':'81f51e224ce9e74b02cc2a3eaf21b2d91d743455'},indent=2)+'\n')
