from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;BASE=Path('/root/autodl-tmp')
fixed=json.loads((BASE/'PRISM_PAPER_SPLINE_REPAIR_20261026/results/repaired/SRU_H2S/RESULTS.json').read_text())
effects=[e for e in fixed['effects'] if e['delay']==8 and e['skip']==0]
cpu=json.loads((ROOT/'CPU_COMPLEXITY.json').read_text())['records'];new=json.loads((ROOT/'REPAIRED_CPU_COMPLEXITY.json').read_text())['records']
fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
ax=axes[0];vals=[e['increase_percent'] for e in effects];lo=[v-e['CI95'][0] for v,e in zip(vals,effects)];hi=[e['CI95'][1]-v for v,e in zip(vals,effects)]
ax.bar(['remove KC','remove W','remove A'],vals,color=['#21618c','#3498db','#76d7c4']);ax.errorbar(np.arange(3),vals,yerr=[lo,hi],fmt='none',color='#263238',capsize=4)
ax.set_ylabel('RMSE increase (%)');ax.set_title('SRU H2S: delay8 / history16')
for i,v in enumerate(vals):ax.text(i,v+4,f'{v:.1f}%',ha='center',fontsize=9)
names=['PRISM','LSTM','Transformer','iTransformer','TimeXer']
p=next(r for r in new if r['delay']==8);net=[next(r for r in cpu if r['delay']==8 and r['model']==n and r['ensemble']==1) for n in names[1:]]
vals=[p['median_us']]+[r['median_us'] for r in net]
ax=axes[1];ax.barh(names,vals,color=['#117864']+['#5dade2']*4);ax.invert_yaxis();ax.set_xlabel('Median CPU latency (microseconds)');ax.set_title('One CPU thread; includes A updates')
ax.set_xlim(0,max(vals)*1.22)
for i,v in enumerate(vals):ax.text(v+8,i,f'{v:.0f}',va='center',fontsize=9)
vals=[p['compact_deployment_bytes']/1024]+[r['compact_deployment_bytes']/1024 for r in net]
ax=axes[2];ax.barh(names,vals,color=['#117864']+['#5dade2']*4);ax.invert_yaxis();ax.set_xscale('log');ax.set_xlabel('Compact deployment size (KiB; log scale)');ax.set_title('Preprocessing included; training logs excluded')
ax.set_xlim(1,max(vals)*2)
for i,v in enumerate(vals):ax.text(v*1.08,i,f'{v:.1f}',va='center',fontsize=9)
fig.suptitle('Exploratory development results; intervals are descriptive, not search-adjusted',fontsize=11)
fig.savefig(ROOT/'SRU_ABLATION_COMPLEXITY.png',dpi=180);fig.savefig(ROOT/'SRU_ABLATION_COMPLEXITY.pdf');plt.close(fig)
print('FIGURES_SAVED',flush=True)
