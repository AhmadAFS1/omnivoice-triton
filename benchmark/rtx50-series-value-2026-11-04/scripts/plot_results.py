"""CPU-only figures from verified actual GPU test data; no inference."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parent
def main():
    data=json.loads((ROOT/'analysis.json').read_text());order=['RTX 5060 Ti','RTX 5070','RTX 5070 Ti','RTX 5080','RTX 5090','RTX 4070']
    rows=sorted(data['GPUs'],key=lambda x:order.index(x['GPU']));assert len(rows)==6
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    labels=[r['GPU'].replace('RTX ','RTX\n')+'\n'+('32' if r['GPU']=='RTX 5090' else '12' if r['GPU'] in ['RTX 5070','RTX 4070'] else '16')+' GB' for r in rows]
    fig,axes=plt.subplots(1,3,figsize=(15,5),layout='constrained')
    metrics=[([r['best_finite_throughput']['complete_requests_per_s'] for r in rows],'Completed cloned requests / sec','Best finite throughput profile'),
             ([r['best_finite_throughput']['USD_per1000_valid'] for r in rows],'Rental USD / 1,000 complete requests','Cost at measured throughput'),
             ([r['best_sequential_p95']['request_latency_p95_ms'] for r in rows],'p95 complete-WAV latency (ms)','Best sequential profile')]
    for ax,(values,ylabel,title) in zip(axes,metrics,strict=True):
        bars=ax.bar(np.arange(6),values,color=['#3567a8']*5+['#8393a7'],width=.65)
        ax.set_xticks(np.arange(6),labels,fontsize=9);ax.set_ylabel(ylabel);ax.set_title(title,fontweight='bold',pad=18);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
        for rect,v in zip(bars,values,strict=True):ax.annotate(f'{v:.3f}' if ylabel.startswith('Rental') else f'{v:.2f}' if 'sec' in ylabel else f'{v:.0f}',(rect.get_x()+rect.get_width()/2,rect.get_height()),xytext=(0,5),textcoords='offset points',ha='center',fontsize=10)
        ax.set_ylim(0,max(values)*1.2)
    axes[2].axhline(500,color='#b53737',linestyle='--',linewidth=1.2,label='500 ms target');axes[2].legend(frameon=False,fontsize=9)
    fig.suptitle('OmniVoice-Triton · actual RTX GPU tests · November 4, 2026\n104 short multilingual cloned turns · warmed · 16 steps · full WAV · response cache off',fontsize=13,fontweight='bold')
    fig.savefig(ROOT/'GPU-value.png',dpi=180);fig.savefig(ROOT/'GPU-value.svg');plt.close(fig)
    phases=[[next(s for s in r['phases'] if s['mode']=='staggered-wide' and s['target_arrival_rate']==rate) for rate in [.5,1,2,4,8]] for r in rows]
    good=np.array([[s['all_offers_SLO_500ms_fraction'] for s in rs] for rs in phases]);reject=np.array([[s['rejection_fraction'] for s in rs] for rs in phases])
    fig,axes=plt.subplots(1,2,figsize=(12,6),layout='constrained')
    for ax,values,title,cmap in [(axes[0],good,'All offers: valid full WAV within 500 ms','YlGnBu'),(axes[1],reject,'Offers rejected with HTTP 429','OrRd')]:
        im=ax.imshow(values,vmin=0,vmax=1,cmap=cmap,aspect='auto');ax.set_xticks(range(5),['0.5','1','2','4','8']);ax.set_yticks(range(6),[r['GPU'] for r in rows]);ax.set_xlabel('Nominal Poisson arrivals / sec');ax.set_title(title,pad=14,fontweight='bold')
        for y in range(6):
            for x in range(5):ax.text(x,y,f'{values[y,x]*100:.1f}%',ha='center',va='center',color='white' if values[y,x]>.65 else '#17202a',fontsize=11)
        fig.colorbar(im,ax=ax,fraction=.03,pad=.02)
    fig.suptitle('Staggered traffic · hybrid runner, 64 graphs / 65,536 row-token budget\n40 sec per schedule · actual offers: 30 / 41 / 81 / 162 / 340 · pending limit 4',fontweight='bold')
    fig.savefig(ROOT/'staggered-SLO.png',dpi=180);fig.savefig(ROOT/'staggered-SLO.svg');plt.close(fig)
if __name__=='__main__':main()
