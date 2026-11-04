"""Offline cached Whisper diagnostic, separate Apple M5 Pro GPU.

All supported timed outputs are screened. Unsupported languages are explicit
skips. ASR is blind to the expected text. This is not Omnilingual ASR or native
quality, speaker identity, MOS, or a verified spoken-word failure rate.
"""
import collections
import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
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


def sha(p):
    d=hashlib.sha256()
    with p.open('rb') as s:
        for b in iter(lambda:s.read(8*1024**2),b''):d.update(b)
    return d.hexdigest()


def normalize(text):
    v=unicodedata.normalize('NFKC',text).casefold()
    v=''.join(' ' if unicodedata.category(c).startswith(('P','S')) else c for c in v)
    v=''.join(c for c in v if c!='\u0640' and not '\u064b'<=c<='\u065f' and c!='\u0670')
    v=v.translate(str.maketrans({'أ':'ا','إ':'ا','آ':'ا','ٱ':'ا','ى':'ي'}))
    return ' '.join(v.split())


def distance(a,b):
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        cur=[i]
        for j,y in enumerate(b,1):cur.append(min(prev[j]+1,cur[-1]+1,prev[j-1]+(x!=y)))
        prev=cur
    return prev[-1]


def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='word-check');parser.add_argument('--bases',nargs='+',default=['retained/results','retained/dual','retained/dual-mps','retained/single-mps']);args=parser.parse_args()
    dest=ROOT/args.output;dest.mkdir(exist_ok=False)
    assert sha(MODEL/'weights.safetensors')=='951ed3fc1203e6a62467abb2144a96ce7eafca8fa77e3704fdb8635ff3e7f8a6'
    selected=[];skips=[]
    for base in args.bases:
        for p in sorted((ROOT/base).glob('batch*/*/requests.json')):
            if p.parent.name=='warmup':continue
            for r in json.loads(p.read_text()):
                assert r['status']==200 and r['signal_valid']
                language=r['case']['language'];hint='ar' if language.startswith('ar-') else language
                item={'phase':Path(base).name+'/'+p.parent.name,'id':r['id'],'path':str(p.parent/r['path']),'sha256':r['sha256'],
                    'language':language,'hint':hint,'expected':r['case']['text'],'reference_id':r['case']['reference_id']}
                if hint in LANGUAGES:selected.append(item)
                else:skips.append({**item,'reason':'Whisper language unsupported; no substitute language'})
    write(dest/'selection.json',{'selected':selected,'skips':skips})
    metadata={'started_at':dt.datetime.now(dt.UTC).isoformat(),'hardware':mx.device_info(),
        'measurement_kind':'Offline ASR on Apple M5 Pro integrated GPU; no co-resident ASR on RTX4070 during TTS',
        'model':str(MODEL),'weight_sha256':sha(MODEL/'weights.safetensors'),'script_sha256':sha(Path(__file__)),
        'packages':{p:importlib.metadata.version(p) for p in ['mlx','mlx-whisper','numpy','scipy','soundfile']},
        'model_download_bytes':0,'inference_API_calls':0,'expected_supplied_to_recognizer':False,
        'mapping':'Arabic dialect routes use family hint ar, not dialect-specific validation. All other route codes require exact Whisper support.'}
    write(dest/'metadata.json',metadata)
    seen={};rows=[];start=time.perf_counter()
    for i,item in enumerate(selected):
        p=Path(item['path']);assert sha(p)==item['sha256'];key=(item['sha256'],item['hint'])
        if key not in seen:
            audio,sr=sf.read(p,dtype='float32',always_2d=True);assert np.isfinite(audio).all();g=math.gcd(sr,16000)
            a=resample_poly(audio.mean(axis=1),16000//g,sr//g).astype(np.float32)
            t=time.perf_counter();r=mlx_whisper.transcribe(a,path_or_hf_repo=str(MODEL),language=item['hint'],temperature=0.0,
                condition_on_previous_text=False,initial_prompt=None,prefix=None,word_timestamps=False,verbose=None)
            seen[key]={'text':r['text'].strip(),'asr_wall_s':time.perf_counter()-t}
        transcript=seen[key]['text'];ref,hyp=normalize(item['expected']),normalize(transcript)
        rc,hc=ref.replace(' ',''),hyp.replace(' ','')
        row={**item,'transcript':transcript,'normalized_exact':ref==hyp,'cer':distance(rc,hc)/len(rc) if rc else None,
            'quality_certified':False,'asr':seen[key]}
        rows.append(row)
        with (dest/'transcripts.jsonl').open('a') as s:s.write(json.dumps(row,ensure_ascii=False)+'\n')
        if (i+1)%50==0:print(json.dumps({'done':i+1,'total':len(selected),'seconds':time.perf_counter()-start}),flush=True)
    grouped={}
    for phase in sorted({r['phase'] for r in rows}):
        x=[r for r in rows if r['phase']==phase]
        grouped[phase]={'screened':len(x),'normalized_exact':sum(r['normalized_exact'] for r in x),'mean_CER':float(np.mean([r['cer'] for r in x]))}
    write(dest/'summary.json',{'total_timed_outputs':len(rows)+len(skips),'screened':len(rows),'unsupported_skips':len(skips),
        'unique_recognizer_inputs':len(seen),'phases':grouped,'ASR_wall_s':time.perf_counter()-start,
        'warning':'Recognition agreement is a diagnostic, not native quality or proof of the spoken words. Unsupported languages not cleared.'})
    metadata['finished_at']=dt.datetime.now(dt.UTC).isoformat();write(dest/'metadata.json',metadata)


if __name__=='__main__':main()
