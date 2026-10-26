from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent;j=json.loads((ROOT/'CZ128_HISTORY_REVIEW.json').read_text())
labels=[('CZ_Rod1',4),('CZ_Rod1',8),('CZ_Rod1',16),('CZ_Rod2',4),('CZ_Rod2',8),('CZ_Rod2',16)]
fig,ax=plt.subplots(figsize=(10,4.2),layout='constrained');x=np.arange(len(labels));width=.25
for i,(mode,color) in enumerate([('fixed128','#aab7b8'),('E128','#117864'),('E16','#5dade2')]):
    rs=[next(r for r in j['rows'] if r['entity']==e and r['delay']==d and r['encoding']==mode) for e,d in labels]
    vals=[r['full_RMSE']/r['baseline_RMSE'] for r in rs]
    ax.bar(x+(i-1)*width,vals,width=width,label=mode+' (control)' if mode=='E16' else mode,color=color)
ax.axhline(1,color='#922b21',linestyle='--',label='Strongest matched baseline')
ax.set_yscale('log');ax.set_xticks(x,[f'Rod1 d{d}' if e=='CZ_Rod1' else f'Rod2 d{d}' for e,d in labels])
ax.set_ylabel('RMSE / best baseline RMSE (log scale; lower is better)')
ax.set_title('CZ:128-step main history, multiresolution E128 and16-step refit control\nExploratory reused-development comparison; all failures retained')
ax.legend(ncol=4,loc='upper left',fontsize=9)
fig.savefig(ROOT/'CZ_E128_HISTORY_COMPARISON.png',dpi=180);fig.savefig(ROOT/'CZ_E128_HISTORY_COMPARISON.pdf');plt.close(fig)
print('CZ_HISTORY_FIGURE_SAVED')
