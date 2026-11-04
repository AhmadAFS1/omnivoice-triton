"""Copy nonsecret benchmark evidence to the fork; all raw WAVs remain local.

Large text streams are losslessly gzipped and independently decompressed/SHA
verified. Never copy checkpoint binaries or conditioning audio into the repo.
"""
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent
DEST=Path('/Users/ahmadsmacair/omnivoice-triton/omnivoice-triton/benchmark/rtx4070-capacity-2026-11-04')


def sha(data):return hashlib.sha256(data).hexdigest()


def main():
    DEST.mkdir(exist_ok=False);records=[]
    def copy(src,target,compress=False):
        data=src.read_bytes();target=DEST/target
        if compress:target=target.with_name(target.name+'.gz');saved=gzip.compress(data,mtime=0)
        else:saved=data
        target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(saved)
        actual=target.read_bytes();decoded=gzip.decompress(actual) if compress else actual
        assert decoded==data
        records.append({'source':str(src.relative_to(ROOT)),'saved':str(target.relative_to(DEST)),
            'source_bytes':len(data),'source_sha256':sha(data),'saved_bytes':len(saved),'saved_sha256':sha(saved),'lossless_gzip':compress})
    for p in sorted((ROOT/'retained').rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(ROOT/'retained')
        if rel.parts[0]=='source' or any(x in rel.parts for x in ['__pycache__','pipe']):continue
        if p.suffix in ['.json','.jsonl','.csv','.log','.txt','.py','.sh'] or p.name=='source.tar.gz':
            copy(p,Path('raw')/rel,p.stat().st_size>150000 and p.suffix not in ['.py','.sh'])
    for name in ['analysis.json','word-comparison.json','BACKUP_VERIFIED.json','plan.json',
        'B1-linear-observations-dtype-corrected.json','B8-linear-observations-dtype-corrected.json']:
        p=ROOT/name;copy(p,Path('derived')/name,name.startswith('B'))
    for name in ['analyze.py','check_words_offline.py','publish_evidence.py']:
        copy(ROOT/name,Path('scripts')/name)
    for folder in ['word-check-baseline','word-check-mps']:
        for p in sorted((ROOT/folder).glob('*')):
            if p.is_file():copy(p,Path('quality')/folder/p.name,p.stat().st_size>150000)
    copy(ROOT/'corpus.json',Path('fixtures/corpus.json'))
    copy(ROOT/'references/manifest.json',Path('fixtures/reference-metadata.json'))
    # Selected same-language reference cases before/after deployment arrangement.
    samples=[]
    for mode,phase,label in [('results','b1-c8','before-one-server'),('dual-mps','dual-b1-c8','after-two-MPS-servers')]:
        folder=ROOT/'retained'/mode/'batch1'/phase
        for r in json.loads((folder/'requests.json').read_text()):
            c=r['case']
            if c['language']!=c['reference_language']:continue
            name=f'{label}-{c["language"]}.wav';copy(folder/r['path'],Path('samples')/name)
            samples.append({'label':label,'file':name,'sha256':r['sha256'],'case':c,'request':r['sent'],
                'caveat':'Two independent process RNG streams; same text/reference/settings but not paired random draws or proven quality preservation.'})
    for r in json.loads((ROOT/'retained/profile/audio-manifest.json').read_text()):
        name='profiler-'+r['path'];copy(ROOT/'retained/profile'/r['path'],Path('samples')/name)
        samples.append({**r,'file':name,'label':'profiler-instrumentation-parity','caveat':'Same-seed diagnostic, not a capacity observation.'})
    (DEST/'samples/manifest.json').write_text(json.dumps(samples,ensure_ascii=False,indent=2)+'\n')
    a=json.loads((ROOT/'analysis.json').read_text())
    with (DEST/'derived/capacity.csv').open('w') as stream:
        fields=['mode','phase','concurrency','server_processes','offers','valid_complete_wavs','wall_s','complete_requests_per_s','request_latency_p95_ms','valid_complete_within_first_second','valid_request_latency_within_1s']
        w=csv.DictWriter(stream,fieldnames=fields);w.writeheader()
        for r in a['phases']:w.writerow({k:r[k] for k in fields})
    instance=json.loads((ROOT/'instances.json').read_text())['instances'][0]
    cleanup=json.loads((ROOT/'resource-cleanup.json').read_text())
    scoped={'instance_id':instance['id'],'GPU':instance['gpu'],'label':instance['label'],'created_at':instance['created_at'],'destroyed_at':instance['destroyed_at'],
        'absence_verified_at':instance['absence_verified_at'],'recorded_hourly_rate':instance['offer']['dph_total'],
        'verified_no_task_instances':cleanup['verified'],'remaining_task_ids':cleanup['remaining_task_ids']}
    (DEST/'cleanup.json').write_text(json.dumps(scoped,indent=2)+'\n')
    (DEST/'publication-manifest.json').write_text(json.dumps({'files':records,'saved_entries':len(records),'selected_audio_files':len(samples),
        'full_WAV_archive':'Local task retained directory; all1680 API and18 profiler WAVs preserved. No model weights, conditioning clips or keys published.'},indent=2)+'\n')
    print(json.dumps({'published_entries':len(records),'selected_audio':len(samples),'source_bytes':sum(x['source_bytes'] for x in records),'saved_bytes':sum(x['saved_bytes'] for x in records)}))


if __name__=='__main__':main()
