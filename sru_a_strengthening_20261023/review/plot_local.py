from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
r=json.loads((ROOT/'COMBINED_REVIEW.json').read_text(encoding='utf-8'))
good=[c for c in r['cases'] if c['retained_baseline_goal_achieved']]
plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(12,7.4),gridspec_kw={'width_ratios':[1.25,1]})
for row,c in enumerate(good):
    ax=axes[row,0]; vals={q['variant']:q['RMSE'] for q in c['rows']}; keys=['full','no_KC','no_W','no_A']; names=['完整','去 KC','去 W','去 A','最强保留基线']
    yy=[vals[k] for k in keys]+[c['strongest_retained_baseline_RMSE']]
    bars=ax.bar(np.arange(5),yy,color=['#168681','#a3b9c9','#a3b9c9','#dd9363','#726e9e'],width=.64)
    ax.set_xticks(np.arange(5),names);ax.set_ylabel('RMSE（原记录单位）');ax.set_ylim(0,max(yy)*1.2)
    for b,y in zip(bars,yy):ax.text(b.get_x()+b.get_width()/2,y+max(yy)*.025,f'{y:.5f}',ha='center',fontsize=9)
    ax.set_title(f"标签延迟 {c['label_delay']} 步 · 完整比最强保留基线低 {c['full_RMSE_reduction_vs_strongest_percent']:.2f}%",loc='left',fontsize=11)
    ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    ax=axes[row,1]; ef={e['control']:e for e in c['effects'] if not e['skip']};g=np.array([ef[k]['gain_percent'] for k in keys[1:]]);ci=np.array([ef[k]['CI95'] for k in keys[1:]])
    ax.errorbar(g,np.arange(3),xerr=np.array([g-ci[:,0],ci[:,1]-g]),fmt='o',color='#168681',ecolor='#5d8083',capsize=4,ms=7)
    ax.axvline(0,color='#888',lw=1,ls='--');ax.set_yticks(np.arange(3),['去 KC','去 W','去 A']);ax.invert_yaxis();ax.set_xlabel('删除后 RMSE 增加（%）');ax.set_xlim(-5,max(ci[:,1])*1.22)
    for i,v in enumerate(g):ax.text(ci[i,1]+max(ci[:,1])*.04,i,f'+{v:.1f}%',va='center',fontsize=10)
    ax.set_title('三项删除效应及 95% 描述性区间',loc='left',fontsize=11);ax.grid(axis='x',alpha=.15)
fig.suptitle('SRU H2S：条件残差反馈 A 的两个开发候选',fontsize=16,x=.06,ha='left',y=.985)
fig.text(.06,.925,'同期估计 y(t)；当前过程量可见、当前目标不可见。KC/W/A 删除后分别独立重训。',fontsize=10,color='#555')
fig.text(.06,.035,'开发段已反复用于研究，区间未校正搜索；对最强基线的配对区间仍跨零。保留尾段未开启。\n基线保留此前更强的冻结 ADAPTIVE_ARX；延迟 4 用有限 Urysohn KC，延迟 16 用线性过程 KC。',fontsize=9,color='#555',linespacing=1.7)
fig.subplots_adjust(left=.075,right=.975,top=.855,bottom=.155,wspace=.29,hspace=.47)
fig.savefig(ROOT/'SRU_A_strengthening.png',dpi=170,facecolor='white')
fig.savefig(ROOT/'SRU_A_strengthening.pdf',facecolor='white')
print('PLOT_SAVED')
