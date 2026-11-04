"""CPU-only immutable TTS archive; never includes credentials or model weights."""
import datetime as dt
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent
DIRS=['source/src','results','dual','dual-mps','single-mps','staggered','staggered-hybrid','hybrid','hybrid-wide','staggered-wide','profile','profile-retry','mps-control']
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def info(p):return {'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)}
def command(args):return subprocess.check_output(args,text=True,stderr=subprocess.STDOUT,timeout=20).strip()
def main():
    assert (ROOT/'ALL_TTS_COMPLETE').exists() and (ROOT/'FINAL_TTS_COMPLETE').exists()
    assert (ROOT/'MATRIX_COMPLETE.json').exists() and (ROOT/'EXTRA_COMPLETE.json').exists()
    # ASR may install/download after timing, but inference is deliberately not
    # launched until this TTS proof has captured the idle card.
    active=command(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'])
    assert not active,active
    source=[info(p) for p in sorted((ROOT/'source/src').rglob('*.py'))]
    model=[info(p) for p in sorted((ROOT/'model').rglob('*')) if p.is_file() and '.cache' not in p.parts]
    proof={'utc':dt.datetime.now(dt.UTC).isoformat(),'source_ref':'b7cced4a0274f18aca541f7837c490bb81d8744e',
           'source_archive':info(ROOT/'source.tar.gz'),'product_python_sources':source,
           'model_pin':json.loads((ROOT/'model-pin.json').read_text()),'model_files_hashes_only_not_backed_up':model,
           'packages':{x:importlib.metadata.version(x) for x in ['torch','triton','omnivoice','transformers','httpx','uvicorn','numpy','soundfile']},
           'final_GPU_idle_before_ASR':True,'final_GPU':command(['nvidia-smi','--query-gpu=name,uuid,memory.total,memory.used,driver_version,power.limit','--format=csv']),
           'ticket':json.loads((ROOT/'ticket.json').read_text()),'MPS_outcome':json.loads((ROOT/'mps-control/outcome.json').read_text())}
    (ROOT/'provenance.json').write_text(json.dumps(proof,indent=2)+'\n')
    # Explicit allow list, rather than walking model/environment/key directories.
    paths=[p for p in ROOT.iterdir() if p.is_file() and (p.suffix in {'.py','.json','.csv','.txt','.sh','.log'} or p.name in {'ALL_TTS_COMPLETE','READY'}) and not p.name.startswith(('asr-','setup_asr')) and p.name!='backup-manifest.json']
    for name in DIRS:
        paths += [p for p in (ROOT/name).rglob('*') if p.is_file() and not p.is_symlink() and 'pipe' not in p.relative_to(ROOT).parts]
    rows=[info(p) for p in sorted(set(paths))]
    (ROOT/'backup-manifest.json').write_text(json.dumps({'utc':dt.datetime.now(dt.UTC).isoformat(),'scope':'TTS outputs, traces, source and logs; model weights, environments, reference clips, keys and ASR excluded','files':rows},indent=2)+'\n')
    print(json.dumps({'files':len(rows),'bytes':sum(x['bytes'] for x in rows),'GPU_idle':True}))
if __name__=='__main__':main()
