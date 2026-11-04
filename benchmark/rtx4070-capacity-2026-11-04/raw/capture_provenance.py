"""CPU-only final backup manifest; excludes keys, model weights and environments.

Model files are hashed as provenance only. They are not part of the backup.
"""
import datetime as dt
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parent


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as s:
        for b in iter(lambda:s.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()


def info(p):return {'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':sha(p)}


def command(args):return subprocess.check_output(args,text=True,stderr=subprocess.STDOUT,timeout=15)


def main():
    for name in ['results','dual','dual-mps','single-mps','profile']:assert (ROOT/name/'COMPLETE.json').exists()
    gpu_processes=command(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader']).strip()
    assert not gpu_processes,'Do not finalize while GPU work remains'
    source=[info(p) for p in sorted((ROOT/'source/src').rglob('*.py'))]
    model=[info(p) for p in sorted((ROOT/'model').rglob('*')) if p.is_file() and '.cache' not in p.parts]
    document={'utc':dt.datetime.now(dt.UTC).isoformat(),'source_ref':'096946d75e2ddcb2797c375b0ee50a1933793b36',
        'source_archive':info(ROOT/'source.tar.gz'),'product_python_sources':source,'model_pin':json.loads((ROOT/'model-pin.json').read_text()),
        'model_files_hashes_only_not_backed_up':model,'model_bytes':sum(x['bytes'] for x in model),
        'packages':{x:importlib.metadata.version(x) for x in ['torch','triton','omnivoice','transformers','httpx','uvicorn','numpy','soundfile']},
        'final_GPU_idle':gpu_processes=='','final_GPU':command(['nvidia-smi','--query-gpu=name,uuid,memory.total,memory.used,driver_version,power.limit','--format=csv']),
        'MPS_outcome':json.loads((ROOT/'mps-control/outcome.json').read_text()),
        'scripts':[info(p) for p in sorted(ROOT.glob('*.py'))]}
    (ROOT/'provenance.json').write_text(json.dumps(document,indent=2)+'\n')
    excluded=['model/','hf-cache/','uv-cache/','source/.venv/','references/','mps-control/pipe/']
    files=[]
    for p in sorted(ROOT.rglob('*')):
        rel=str(p.relative_to(ROOT))
        if p.is_file() and not any(rel.startswith(x) for x in excluded) and rel!='backup-manifest.json':files.append(info(p))
    (ROOT/'backup-manifest.json').write_text(json.dumps({'utc':dt.datetime.now(dt.UTC).isoformat(),'excluded_prefixes':excluded,'files':files},indent=2)+'\n')
    print(json.dumps({'backup_files':len(files),'backup_bytes':sum(x['bytes'] for x in files),'model_files_hashed':len(model),'GPU_idle':True}))


if __name__=='__main__':main()
