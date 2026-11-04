"""CPU-only cross-run and ASR-protocol audit, then reproducible comparison tables."""
import collections
import csv
import datetime as dt
import hashlib
import json
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parent
IDS=[59166781,59168131,59167078,59168148,59167099,59167113]
MODELS=['omniASR_CTC_1B_v2','omniASR_LLM_Unlimited_1B_v2']
def read(p):return json.loads(p.read_text())
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def utc(x):return dt.datetime.fromisoformat(x).timestamp()
def main():
    corpus={c['id']:c for c in read(ROOT/'corpus.json')};gpus=[];quality=[];all_scored=[];protocols=[]
    for eid in IDS:
        gpu=read(ROOT/f'analysis-{eid}.json')['GPUs'][0];base=ROOT/'retained'/str(eid)
        assert gpu['id']==eid and len(gpu['counter_deltas'])==14
        for p in base.rglob('requests.json'):
            for r in read(p):
                assert r['case']==corpus[r['case']['id']],(p,r['id'])
                assert r['sent']['language']==r['case']['api_language'] and r['sent']['text']==r['case']['text']
        mps=read(base/'dual-mps/batch1/actual-mps-clients.json');clients={int(p) for v in mps['clients'].values() for p in v['stdout'].split()}
        assert set(mps['model_processes'])<=clients and len(mps['model_processes'])==2
        assert gpu['provenance']['MPS_outcome']['daemon_quit']['returncode']==0
        resources=[json.loads(s) for s in (base/'mps-control/host-resources.jsonl').read_text().splitlines()]
        for phase in gpu['phases']:
            if phase['mode'].startswith('staggered'):
                plan=next(z for z in read(base/phase['mode']/'plans.json') if z['phase']==phase['phase'])
                phase['target_arrival_rate']=plan['target_arrival_rate']
                rows=read(base/phase['path']/'requests.json')
                assert len(rows)==len(plan['offsets_s'])==len(plan['case_ids'])
                assert all(r['scheduled_offset_s']==offset and r['case']['id']==case for r,offset,case in zip(rows,plan['offsets_s'],plan['case_ids'],strict=True))
            if phase['mode'] not in ['dual-mps','single-mps']:continue
            rows=read(base/phase['path']/'requests.json');barrier=statistics.median(utc(r['utc'])-r['start_offset_s'] for r in rows);n=2 if phase['mode']=='dual-mps' else 1
            selected=[r for r in resources if barrier<=utc(r['utc'])<=barrier+phase['wall_s'] and len(r['servers'])==n]
            cpu=[sum(s['cpu_percent_one_core_100'] for s in r['servers']) for r in selected if all(s['cpu_percent_one_core_100'] is not None for s in r['servers'])]
            rss=[sum(s['rss_bytes'] for s in r['servers']) for r in selected]
            phase['TTS_process_CPU_one_core100_mean']=statistics.mean(cpu) if cpu else None
            phase['TTS_process_RSS_sum_mean_bytes']=statistics.mean(rss) if rss else None
            phase['host_resource_samples']=len(selected)
        quality_base=ROOT/'quality-results'/str(eid);scored=read(quality_base/'scored.json');summary=read(quality_base/'summary.json');receipt=read(quality_base/'CONTROLLER.json')
        assert receipt['returncode']==0 and read(quality_base/'BACKUP_VERIFIED.json')['all_size_and_SHA256_match']
        assert receipt['source_sha256']==sha(ROOT/'asr/evaluate.py')
        for model in MODELS:
            meta=read(quality_base/f'{model}-metadata.json');assert meta['dtype']=='float32' and meta['batch_size']==1 and meta['blind'] and not meta['expected_text_prompt']
            assert meta['supported_language_script_ids']==1672 and meta['physical_gpu_nvidia_smi'].startswith('NVIDIA GeForce RTX 5090, GPU-694a2f09-ee4a-12f1-07c6-5fa2e0dab6f2,')
            assert not meta['co_resident_gpu_processes']
            protocols.append({'model':model,'source_sha':receipt['source_sha256'],'packages':meta['packages'],'dtype':meta['dtype'],'batch_size':meta['batch_size'],'ASR_source_commit':receipt['pinned_assets']['ASR_source_commit'],'assets':[(x['name'],x['bytes'],x['sha256']) for x in receipt['pinned_assets']['assets']]})
        assert len(scored)==312 and summary['cases']==312
        by={(x['source']['run'],x['language']):x for x in scored};assert len(by)==312
        item={'id':eid,'GPU':gpu['GPU'],'ASR_hardware':'RTX5090 /32607MiB physical /CUDA12.8 /float32 batch1; separate from TTS','baseline_graph_byte_identical':0,'arms':{},'paired_deltas':[]}
        for arm in ['baseline','graphs','throughput']:
            subset=[x for x in scored if x['source']['run']==arm];assert len(subset)==104
            item['arms'][arm]={'cases':104,'both_judges_exact_characters':sum(all(j['exact_characters_ignoring_whitespace'] for j in x['judges']) for x in subset),'flagged_by_at_least_one_judge_or_signal_rule':sum(bool(x['flags']) for x in subset),'models':{}}
            for model in MODELS:
                js=[next(j for j in x['judges'] if j['model_card']==model) for x in subset]
                item['arms'][arm]['models'][model]={'exact_character_cases':sum(j['exact_characters_ignoring_whitespace'] for j in js),'character_edits':sum(j['character_edits'] for j in js),'reference_characters':sum(j['reference_characters'] for j in js),'weighted_CER':sum(j['character_edits'] for j in js)/sum(j['reference_characters'] for j in js)}
        for lang in sorted(c['language'] for c in corpus.values()):
            b=by['baseline',lang];g=by['graphs',lang];t=by['throughput',lang]
            equal=b['sha256']==g['sha256'];item['baseline_graph_byte_identical']+=equal
            delta={'language':lang,'case':b['source']['case'],'byte_identical_baseline_graphs':equal,'baseline_sha256':b['sha256'],'graph_sha256':g['sha256'],'throughput_sha256':t['sha256'],'models':{}}
            for model in MODELS:
                js={arm:next(j for j in x['judges'] if j['model_card']==model) for arm,x in [('baseline',b),('graphs',g),('throughput',t)]}
                if equal:assert js['baseline']['transcript']==js['graphs']['transcript']
                delta['models'][model]={arm:{k:j[k] for k in ['transcript','cer','wer','exact_characters_ignoring_whitespace']} for arm,j in js.items()}
            item['paired_deltas'].append(delta)
        item['actual_ASR_inferences_by_model']={model:summary['models'][model]['actual_asr_inferences'] for model in MODELS}
        quality.append(item);all_scored+=scored;gpus.append(gpu)
    for model in MODELS:
        matching=[p for p in protocols if p['model']==model];assert all(p==matching[0] for p in matching),'ASR protocol drift; do not merge'
    source=[{r['path']:r['sha256'] for r in g['provenance']['product_python_sources']} for g in gpus];assert all(x==source[0] for x in source)
    weights=[{r['path']:r['sha256'] for r in g['provenance']['model_files_hashes_only_not_backed_up']} for g in gpus];assert all(x==weights[0] for x in weights)
    assert all(g['provenance']['source_archive']['sha256']=='a1de5675d138983472efd6028e01ec2ee0241730b81675d51161e71b0974fc0f' for g in gpus)
    assert all(g['hardware']['cpu_threads']==8 for g in gpus)
    reference_summary=read(ROOT/'quality-results/references/summary.json')
    analysis={'audit_kind':'CPU-only aggregation of independently audited real GPU runs','source_and_model_identical_all_six_GPUs':True,'GPUs':gpus,
              'timed_offers':sum(g['timed_offers'] for g in gpus),'warmup_offers':sum(g['warmup_offers'] for g in gpus),'valid_load_WAVs_verified':sum(g['valid_load_WAVs_verified'] for g in gpus)}
    write(ROOT/'analysis.json',analysis)
    proof={'audit_kind':'CPU-only aggregation; actual ASR hardware RTX5090, not the TTS cards','GPU_comparisons':quality,'logical_generated_ASR_cases':len(all_scored),'reference_calibration':reference_summary,
           'global_unique_generated_audio_language_inputs':len({(x['sha256'],x['lang_id']) for x in all_scored}),'source_script_sha256':sha(ROOT/'asr/evaluate.py'),'ASR_protocols_identical_within_model':True,'protocols':protocols[:2],
           'limitations':['ASR flags are not verified TTS failures.','CTC/LLM variants share a model family; disagreement and baseline errors require review.','No MOS, native listening or speaker identity qualification.','Only selected three arms per GPU are transcribed; all other outputs are retained.','One Twi route uses the declared Akan-family proxy; Arabic/dialect IDs and writing scripts retain case caveats.']}
    write(ROOT/'quality-comparison.json',proof)
    with (ROOT/'capacity.csv').open('w',newline='') as f:
        fields=['GPU','mode','phase','concurrency','offers','valid_complete_wavs','wall_s','complete_requests_per_s','request_latency_p50_ms','request_latency_p95_ms','valid_complete_within_first_second','max_completions_in_aligned_1s_bin','hourly_USD','USD_per1000_valid','audio_duration_mean_s','audio_seconds_per_wall_second','all_offers_SLO_500ms_fraction','rejection_fraction']
        writer=csv.DictWriter(f,fields,extrasaction='ignore');writer.writeheader()
        for g in gpus:writer.writerows(g['phases'])
    with (ROOT/'word-checks.csv').open('w',newline='') as f:
        fields=['GPU','arm','language','model','transcript','CER','WER','exact_characters','wave_SHA256'];writer=csv.DictWriter(f,fields);writer.writeheader()
        for x in all_scored:
            for j in x['judges']:writer.writerow({'GPU':x['source']['GPU'],'arm':x['source']['run'],'language':x['language'],'model':j['model_card'],'transcript':j['transcript'],'CER':j['cer'],'WER':j['wer'],'exact_characters':j['exact_characters_ignoring_whitespace'],'wave_SHA256':x['sha256']})
    print(json.dumps({'GPUs':len(gpus),'timed_offers':analysis['timed_offers'],'warmups':analysis['warmup_offers'],'retained_load_WAVs':analysis['valid_load_WAVs_verified'],'logical_generated_ASR_cases':len(all_scored),'baseline_graph_equal':sum(q['baseline_graph_byte_identical'] for q in quality)}))
if __name__=='__main__':main()
