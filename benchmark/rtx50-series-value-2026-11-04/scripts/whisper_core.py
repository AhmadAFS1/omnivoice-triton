"""Additional blind offline diagnostic for20 exactly supported core languages.

Runs on the operator's Apple GPU, never on a TTS test GPU. It supplements the
104-route Omnilingual screens; it does not qualify unsupported languages.
"""
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import time
import unicodedata
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
import mlx.core as mx
import mlx_whisper
from mlx_whisper.tokenizer import LANGUAGES
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT=Path(__file__).resolve().parent
MODEL=Path.home()/'.cache/huggingface/hub/models--mlx-community--whisper-large-v3-turbo/snapshots/a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb'
CORE=['en','es','ar','zh','fr','de','it','pt','ru','hi','ja','ko','nl','pl','tr','vi','id','ur','el','fi']
IDS=[59167099,59167113,59168148,59168131,59166781,59167078]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def norm(v):
    v=unicodedata.normalize('NFKC',v).casefold()
    v=''.join(' ' if unicodedata.category(c).startswith(('P','S')) else c for c in v)
    v=''.join(c for c in v if c!='\u0640' and not '\u064b'<=c<='\u065f' and c!='\u0670')
    v=v.translate(str.maketrans({'أ':'ا','إ':'ا','آ':'ا','ٱ':'ا','ى':'ي'}));return ' '.join(v.split())
def distance(a,b):
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        cur=[i]
        for j,y in enumerate(b,1):cur.append(min(prev[j]+1,cur[-1]+1,prev[j-1]+(x!=y)))
        prev=cur
    return prev[-1]
def main():
    assert all(x in LANGUAGES for x in CORE)
    dest=ROOT/'whisper-core';dest.mkdir(exist_ok=False)
    digest=sha(MODEL/'weights.safetensors');assert digest=='951ed3fc1203e6a62467abb2144a96ce7eafca8fa77e3704fdb8635ff3e7f8a6'
    meta={'started_utc':dt.datetime.now(dt.UTC).isoformat(),'GPU':mx.device_info(),'physical_unified_system_memory_bytes':int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True)),
          'memory_note':'Apple integrated GPU shares unified system RAM; no separate dedicated VRAM','measurement_kind':'Offline GPU ASR diagnostic on Apple M5 Pro; independent of timed rented RTX TTS',
          'model':str(MODEL),'weight_sha256':digest,'script_sha256':sha(Path(__file__)),'language_mapping':'Exact20 supported standard language IDs; no dialect substitution',
          'packages':{p:importlib.metadata.version(p) for p in ['mlx','mlx-whisper','numpy','scipy','soundfile']},'expected_supplied_to_recognizer':False,'model_download_bytes':0,'paid_API_calls':0}
    write(dest/'metadata.json',meta);seen={};rows=[];done=set();deadline=time.monotonic()+4800
    while done!=set(IDS):
        for eid in IDS:
            manifest=ROOT/'quality-input'/str(eid)/'manifest.json'
            if eid in done or not manifest.exists():continue
            selected=[x for x in json.loads(manifest.read_text()) if x['language'] in CORE and x['run'] in ['baseline','throughput']]
            assert len(selected)==40
            for item in selected:
                path=manifest.parent/item['path'];assert sha(path)==item['sha256'];key=(item['sha256'],item['language'])
                reused=key in seen
                if not reused:
                    audio,sr=sf.read(path,dtype='float32',always_2d=True);assert np.isfinite(audio).all()
                    g=math.gcd(sr,16000);audio=resample_poly(audio.mean(axis=1),16000//g,sr//g).astype(np.float32)
                    start=time.perf_counter();answer=mlx_whisper.transcribe(audio,path_or_hf_repo=str(MODEL),language=item['language'],temperature=0.0,condition_on_previous_text=False,initial_prompt=None,prefix=None,word_timestamps=False,verbose=None)
                    seen[key]={'text':answer['text'].strip(),'asr_wall_s':time.perf_counter()-start,'original_id':item['id']}
                hyp=norm(seen[key]['text']);ref=norm(item['text']);rc,hc=ref.replace(' ',''),hyp.replace(' ','')
                row={**item,'transcript':seen[key]['text'],'normalized_exact':ref==hyp,'exact_characters_ignoring_whitespace':rc==hc,'reference_characters':len(rc),'character_edits':distance(rc,hc),'cer':distance(rc,hc)/len(rc),
                     'asr_wall_s':0 if reused else seen[key]['asr_wall_s'],'reused_from_id':seen[key]['original_id'] if reused else None,'quality_certified':False}
                rows.append(row)
                with (dest/'transcripts.jsonl').open('a') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
            done.add(eid);print(json.dumps({'GPU_shard':eid,'done_cases':len(rows),'unique_ASR_inputs':len(seen)}),flush=True)
        assert time.monotonic()<deadline,'Core diagnostic wait exceeded'
        if done!=set(IDS):time.sleep(20)
    write(dest/'summary.json',{'cases':len(rows),'languages':CORE,'unique_recognizer_inputs':len(seen),'exact_character_cases':sum(x['exact_characters_ignoring_whitespace'] for x in rows),'aggregate_CER':sum(x['character_edits'] for x in rows)/sum(x['reference_characters'] for x in rows),'quality_certified':False,'note':'20 standard languages only. Remaining84 routes not screened by this model; full104-route coverage is the two Omnilingual models. ASR mismatch is not a verified TTS failure.'})
    meta['finished_utc']=dt.datetime.now(dt.UTC).isoformat();write(dest/'metadata.json',meta)
if __name__=='__main__':main()
