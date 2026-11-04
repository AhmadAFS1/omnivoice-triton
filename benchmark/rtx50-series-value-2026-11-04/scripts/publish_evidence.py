"""CPU-only evidence packaging; weights, environments, keys and prompts excluded."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parent
REPO=Path('/Users/ahmadsmacair/omnivoice-triton/omnivoice-triton')
DEST=REPO/'benchmark/rtx50-series-value-2026-11-04'
def read(p):return json.loads(p.read_text())
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def main():
    assert (ROOT/'QUALITY_COMPLETE.json').exists() and (ROOT/'whisper-core/summary.json').exists()
    data=read(ROOT/'analysis.json');assert len(data['GPUs'])==6
    manifest=[]
    def save(src,rel,compress=None):
        if compress is None:compress=src.suffix in {'.json','.jsonl','.csv','.log','.txt'} and src.stat().st_size>65536
        dst=DEST/(rel+('.gz' if compress else ''));dst.parent.mkdir(parents=True,exist_ok=True)
        if compress:
            with src.open('rb') as f,dst.open('wb') as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0,compresslevel=9) as out:shutil.copyfileobj(f,out)
        else:shutil.copyfile(src,dst)
        manifest.append({'source':src.relative_to(ROOT).as_posix(),'saved':dst.relative_to(DEST).as_posix(),'source_bytes':src.stat().st_size,'source_sha256':sha(src),'saved_bytes':dst.stat().st_size,'saved_sha256':sha(dst),'gzip':compress})
    for gpu in data['GPUs']:
        eid=gpu['id'];base=ROOT/'retained'/str(eid);slug=gpu['GPU'].lower().replace(' ','-')
        for src in sorted(base.rglob('*')):
            rel=src.relative_to(base)
            if not src.is_file() or src.is_symlink() or rel.parts[0]=='source' or src.suffix in {'.wav','.error','.gz','.pyc'}:continue
            if src.suffix not in {'.py','.json','.jsonl','.csv','.log','.txt','.sh'}:continue
            save(src,'raw/'+slug+'/'+rel.as_posix())
        qbase=ROOT/'quality-results'/str(eid)
        for src in sorted(qbase.iterdir()):
            if src.is_file() and src.suffix in {'.json','.jsonl','.log'}:save(src,'quality/omnilingual/'+slug+'/'+src.name)
        save(ROOT/'quality-input'/str(eid)/'selection.json','quality/selection/'+slug+'.json')
        save(ROOT/'quality-input'/str(eid)/'manifest.json','quality/selection/'+slug+'-manifest.json')
    for name in ['references']:
        for src in sorted((ROOT/'quality-results'/name).iterdir()):
            if src.is_file() and src.suffix in {'.json','.jsonl','.log'}:save(src,'quality/omnilingual/'+name+'/'+src.name)
    for src in sorted((ROOT/'whisper-core').iterdir()):
        if src.is_file() and src.suffix in {'.json','.jsonl'}:save(src,'quality/whisper-core/'+src.name)
    for src in sorted((ROOT/'asr-runtime').rglob('*')):
        if src.is_file():save(src,'quality/runtime/'+src.relative_to(ROOT/'asr-runtime').as_posix())
    for name in ['analysis.json','quality-comparison.json','capacity.csv','word-checks.csv','QUALITY_COMPLETE.json','analysis-freeze.txt']:
        save(ROOT/name,'derived/'+name,compress=(True if name=='analysis.json' else False if name=='capacity.csv' else None))
    for name in ['GPU-value.png','GPU-value.svg','staggered-SLO.png','staggered-SLO.svg']:
        save(ROOT/name,'figures/'+name)
    for name in ['corpus.json','model-pin.json','plan.json','research.json','source.tar.gz']:
        save(ROOT/name,'fixtures/'+name,compress=False)
    save(ROOT/'cleanup-scoped.json','cleanup.json',compress=False)
    save(ROOT/'rental-cost-estimate.json','rental-cost-estimate.json',compress=False)
    refs=read(ROOT/'references/manifest.json');save(ROOT/'references/manifest.json','fixtures/reference-manifest.json',compress=False)
    scripts=['analyze.py','assemble.py','plot_results.py','render_report.py','publish_evidence.py','quality_queue.py','whisper_core.py','run_all.py','run_capacity.py','run_dual.py','run_mps.py','run_extra.py','run_hybrid.py','run_staggered.py','run_staggered_hybrid.py','run_wide.py','run_hybrid_wide.py','run_staggered_wide.py','scheduled_helpers.py','profile_pipeline.py','profile_pipeline_retry.py','seeded_server.py','server_observer.py','capture_provenance.py']
    for name in scripts:save(ROOT/name,'scripts/'+name,compress=False)
    for name in ['evaluate.py','prefetch.py','expected-assets.json']:save(ROOT/'asr'/name,'scripts/asr/'+name,compress=False)
    samples=[]
    for gpu in data['GPUs']:
        eid=gpu['id'];scored=read(ROOT/'quality-results'/str(eid)/'scored.json')
        for arm in ['baseline','graphs','throughput']:
            for lang in ['en','ar','zh']:
                item=next(x for x in scored if x['source']['run']==arm and x['language']==lang)
                src=item['source'];path=Path(src['original_path']);assert sha(path)==item['sha256']
                name=gpu['GPU'].lower().replace(' ','-')+'-'+arm+'-'+lang+'.wav';save(path,'samples/'+name,compress=False)
                samples.append({'GPU':gpu['GPU'],'physical_and_visible_VRAM_evidence':gpu['hardware']['physical_vram_driver_power'],'torch_visible_vram_bytes':gpu['hardware']['torch_visible_vram_bytes'],
                                'arm':arm,'path':name,'sha256':item['sha256'],'phase':src['phase'],'case':src['case'],'parameters':src['request'],'reference':next(r for r in refs if r['id']==src['case']['reference_id']),
                                'ASR_hardware':'RTX5090 /32607MiB physical /CUDA12.8 /float32 batch1','judges':item['judges'],'flags':item['flags'],'quality_certified':False})
    write(DEST/'samples/manifest.json',samples)
    write(DEST/'publication-manifest.json',{'scope':'All retained raw metadata/logs/traces and selected generated WAVs. Entire workload WAVs remain local in work/omnivoice-50series-value-2026-11-04/retained. No model weights, environments, SSH keys or conditioning audio copied.','files':manifest})
    for row in manifest:
        p=DEST/row['saved'];assert p.stat().st_size==row['saved_bytes'] and sha(p)==row['saved_sha256']
        if row['gzip']:
            with gzip.open(p,'rb') as f:body=f.read()
            assert len(body)==row['source_bytes'] and hashlib.sha256(body).hexdigest()==row['source_sha256']
    print(json.dumps({'published_files':len(manifest),'published_bytes':sum(r['saved_bytes'] for r in manifest),'selected_audio':len(samples),'all_saved_and_decompressed_bytes_verified':True}))
if __name__=='__main__':main()
