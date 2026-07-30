import os, json, time
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt
from artifact_tool import Blob, SpreadsheetFile, Workbook

IN='/mnt/data/实验数据1-张(2).xlsx'
OUT_JSON='/mnt/data/CL_OPS_UOI_CPU验证结果.json'
OUT_XLSX='/mnt/data/CL_OPS_UOI_CPU验证结果.xlsx'
P1='/mnt/data/CL_OPS_UOI_跨棒改善率.png'
P2='/mnt/data/CL_OPS_UOI_晶升核对比.png'

win=SpreadsheetFile.import_xlsx(Blob.load(IN))
spec={'Sheet1':'A1:J20104','Sheet2':'A1:G20628'}
rec={}; breaks={}; starts={}
for s,addr in spec.items():
    vals=win.worksheets.get_item(s).get_range(addr).values
    hdr=vals[0]; A=np.asarray(vals[1:],dtype=np.float64); ix={str(v):i for i,v in enumerate(hdr) if v is not None}
    y=A[:,ix['晶体直径']]
    br=(np.where(np.abs(np.diff(y))>0.20)[0]+1).astype(int).tolist()
    st=max(br)+1 if br else 0
    breaks[s]=br; starts[s]=st
    rec[s]={
        'y':y,
        'power':A[:,ix['主加热功率']],
        'cl':A[:,ix['晶升速度']],
        'kl':A[:,ix['埚升速度']],
        'cr':A[:,ix['晶转速度']],
        'kr':A[:,ix['埚转速度']],
        'start':st,
    }

def Bmat(L=128,R=32):
    t=np.arange(L)[:,None]; r=np.arange(R)[None,:]
    B=np.cos(np.pi*(t+0.5)*r/L)
    B[:,0]/=np.sqrt(L)
    if R>1: B[:,1:]*=np.sqrt(2/L)
    return B

def make_design(u,y,h=15,L=128,R=32,order=1,mu=None,sd=None):
    if mu is None: mu=float(u.mean())
    if sd is None: sd=float(u.std())
    z=np.clip((u-mu)/sd,-3.5,3.5)
    W=sliding_window_view(z,L)[:len(z)-L-h+1][:,::-1]
    B=Bmat(L,R)
    H=[W]
    if order>=2: H.append((W*W-1)/np.sqrt(2))
    if order>=3: H.append((W*W*W-3*W)/np.sqrt(6))
    X=np.concatenate([q@B for q in H],axis=1)
    d=y[L-1+h:]-y[L-1:-h]
    return X,d,B,mu,sd,z

def fit_stats(X,y):
    xm=X.mean(0); ym=float(y.mean()); Xc=X-xm; yc=y-ym
    return xm,ym,Xc.T@Xc,Xc.T@yc

def solve(st,lams,R):
    xm,ym,G,b=st
    diag=np.concatenate([np.full(R,float(v)) for v in lams])
    A=G+np.diag(diag)
    try: c=cho_solve(cho_factor(A,lower=True,check_finite=False),b,check_finite=False)
    except Exception: c=np.linalg.lstsq(A,b,rcond=None)[0]
    return float(ym-xm@c),c

def choose_lambda(X,d,R,order):
    n=len(d); cut=int(0.72*n); gap=256
    tr=slice(0,cut-gap); va=slice(cut+gap,None)
    st=fit_stats(X[tr],d[tr]); base=np.mean((d[va]-d[tr].mean())**2)
    if order==1:
        grid=[(x,) for x in np.logspace(-4,5,10)]
    else:
        grid=[(a,b,b) for a in [1e-3,1e-1,10,1e3] for b in [1e-1,10,1e3,1e5,1e8,1e12]]
    best=None
    for la in grid:
        i,c=solve(st,la,R); mse=np.mean((d[va]-(i+X[va]@c))**2); ratio=float(mse/base)
        if best is None or ratio<best[0]: best=(ratio,la)
    return best

def candidate_signal(sheet,candidate,params=None):
    r=rec[sheet]; st=r['start']
    if candidate=='crystal_lift': return r['cl'][st:],None
    X=np.c_[r['cl'][st:],r['kl'][st:]]
    if params is None:
        mu=X.mean(0); sd=X.std(0); Z=(X-mu)/sd
        vals,vecs=np.linalg.eigh(Z.T@Z/len(Z)); v=vecs[:,-1]
        if v.sum()<0: v=-v
        params={'mu':mu,'sd':sd,'v':v,'explained':float(vals[-1]/vals.sum())}
    return ((X-params['mu'])/params['sd'])@params['v'],params

def bootstrap(base_loss,model_loss,block=256,nboot=1000,seed=7):
    rng=np.random.default_rng(seed); n=len(base_loss); nb=int(np.ceil(n/block)); vals=[]
    for _ in range(nboot):
        ss=rng.integers(0,n,size=nb)
        idx=np.concatenate([(np.arange(block)+s)%n for s in ss])[:n]
        vals.append(1-model_loss[idx].mean()/base_loss[idx].mean())
    a=np.asarray(vals)
    return {'lo95':float(np.quantile(a,.025)),'median':float(np.median(a)),'hi95':float(np.quantile(a,.975)),'p_positive':float(np.mean(a>0))}

def static_cubic(u,y,h,mu,sd):
    z=np.clip((u-mu)/sd,-3.5,3.5)[:-h]
    X=np.c_[z,(z*z-1)/np.sqrt(2),(z*z*z-3*z)/np.sqrt(6)]
    d=y[h:]-y[:-h]
    return X,d

def select_simple_ridge(X,d):
    n=len(d); cut=int(.72*n); gap=256; tr=slice(0,cut-gap); va=slice(cut+gap,None)
    best=None
    for lam in np.logspace(-4,5,10):
        i,c=solve(fit_stats(X[tr],d[tr]),(lam,),X.shape[1])
        m=np.mean((d[va]-(i+X[va]@c))**2)
        if best is None or m<best[0]:best=(m,lam)
    return best[1]

def fit_direction(train,test,candidate,h=15,L=128,R=32):
    u,p=candidate_signal(train,candidate)
    v,_=candidate_signal(test,candidate,p)
    y=rec[train]['y'][rec[train]['start']:]; yt=rec[test]['y'][rec[test]['start']:]
    X,d,B,mu,sd,z=make_design(u,y,h,L,R,1)
    Xt,dt,_,_,_,zt=make_design(v,yt,h,L,R,1,mu,sd)
    sel1=choose_lambda(X,d,R,1); i1,c1=solve(fit_stats(X,d),sel1[1],R); pr1=i1+Xt@c1
    X3,d3,B,_,_,_=make_design(u,y,h,L,R,3,mu,sd)
    X3t,dt3,_,_,_,_=make_design(v,yt,h,L,R,3,mu,sd)
    sel3=choose_lambda(X3,d3,R,3); i3,c3=solve(fit_stats(X3,d3),sel3[1],R); pr3=i3+X3t@c3
    base=np.full_like(dt,float(d.mean())); bl=(dt-base)**2; l1=(dt-pr1)**2; l3=(dt-pr3)**2
    Xs,ds=static_cubic(u,y,h,mu,sd); Xst,dst=static_cubic(v,yt,h,mu,sd)
    ls=select_simple_ridge(Xs,ds); is_,cs=solve(fit_stats(Xs,ds),(ls,),Xs.shape[1])
    ps=is_+Xst@cs; bs=np.full_like(dst,float(ds.mean())); static_imp=float(1-np.mean((dst-ps)**2)/np.mean((dst-bs)**2))
    m=len(dt)//2
    halves={'first':float(1-l1[:m].mean()/bl[:m].mean()),'second':float(1-l1[m:].mean()/bl[m:].mean())}
    placebo=[]
    for sh in [256,512,1024,2048,4096]:
        Xsh,dsh,_,_,_,_=make_design(np.roll(v,sh),yt,h,L,R,1,mu,sd)
        psh=i1+Xsh@c1; bsh=np.full_like(dsh,float(d.mean()))
        placebo.append({'shift':sh,'improvement':float(1-np.mean((dsh-psh)**2)/np.mean((dsh-bsh)**2))})
    klin=B@c1
    kblocks=[B@c3[j*R:(j+1)*R] for j in range(3)]
    return {
        'candidate':candidate,'train':train,'test':test,'h':h,'L':L,'R':R,
        'linear_cv_ratio':float(sel1[0]),'linear_lambda':list(map(float,sel1[1])),
        'K_cv_ratio':float(sel3[0]),'K_lambda':list(map(float,sel3[1])),
        'base_rmse':float(np.sqrt(bl.mean())),'FIR_rmse':float(np.sqrt(l1.mean())),'K_rmse':float(np.sqrt(l3.mean())),
        'FIR_improvement':float(1-l1.mean()/bl.mean()),'K_improvement':float(1-l3.mean()/bl.mean()),
        'K_vs_FIR':float(1-l3.mean()/l1.mean()),'static_improvement':static_imp,
        'FIR_bootstrap':bootstrap(bl,l1,seed=17),'K_bootstrap':bootstrap(bl,l3,seed=19),
        'halves':halves,'placebo':placebo,'test_ood_fraction':float(np.mean((zt<z.min())|(zt>z.max()))),
        'linear_kernel':klin.tolist(),'K_blocks':[k.tolist() for k in kblocks],
        'pca':None if p is None else {'v':p['v'].tolist(),'explained':p['explained']},'n_test':int(len(dt))
    }

def horizon_check(train,test,candidate,h):
    u,p=candidate_signal(train,candidate);v,_=candidate_signal(test,candidate,p)
    y=rec[train]['y'][rec[train]['start']:];yt=rec[test]['y'][rec[test]['start']:]
    X,d,B,mu,sd,z=make_design(u,y,h,128,32,1);Xt,dt,_,_,_,_=make_design(v,yt,h,128,32,1,mu,sd)
    sel=choose_lambda(X,d,32,1);i,c=solve(fit_stats(X,d),sel[1],32);pr=i+Xt@c;b=np.full_like(dt,float(d.mean()))
    return {'candidate':candidate,'train':train,'test':test,'h':h,'improvement':float(1-np.mean((dt-pr)**2)/np.mean((dt-b)**2))}

T=time.time(); results=[]
for cand in ['crystal_lift','joint_lift']:
    for tr,te in [('Sheet1','Sheet2'),('Sheet2','Sheet1')]:
        results.append(fit_direction(tr,te,cand))

horizon=[]
for h in [1,60]:
    for tr,te in [('Sheet1','Sheet2'),('Sheet2','Sheet1')]:
        horizon.append(horizon_check(tr,te,'crystal_lift',h))

agreement={}
for cand in ['crystal_lift','joint_lift']:
    pair=[r for r in results if r['candidate']==cand]
    k1=np.asarray(pair[0]['linear_kernel']);k2=np.asarray(pair[1]['linear_kernel'])
    agreement[cand]=float(np.corrcoef(k1,k2)[0,1])
for r in results:
    k=np.asarray(r['linear_kernel']);r['peak_lag']=int(np.argmax(np.abs(k)));r['peak_value']=float(k[r['peak_lag']])

correlation={}
for s,r in rec.items():
    st=r['start']; correlation[s]=float(np.corrcoef(r['cl'][st:],r['kl'][st:])[0,1])

payload={'input':os.path.basename(IN),'breaks':breaks,'starts':starts,'lift_correlations':correlation,
         'results':results,'horizon_checks':horizon,'kernel_agreement':agreement,'elapsed_seconds':float(time.time()-T)}
with open(OUT_JSON,'w',encoding='utf8') as f: json.dump(payload,f,ensure_ascii=False,indent=2)

labels=[];fir=[];kk=[]
for r in results:
    labels.append(('晶升' if r['candidate']=='crystal_lift' else '联合升速')+'\n'+r['train']+'→'+r['test'])
    fir.append(100*r['FIR_improvement']);kk.append(100*r['K_improvement'])
x=np.arange(len(labels));w=.36
plt.figure(figsize=(10,5));plt.bar(x-w/2,fir,w,label='线性时滞核');plt.bar(x+w/2,kk,w,label='非线性K层');plt.axhline(0,linewidth=1);plt.xticks(x,labels);plt.ylabel('MSE改善 / %');plt.legend();plt.tight_layout();plt.savefig(P1,dpi=180);plt.close()
plt.figure(figsize=(9,5))
for r in [q for q in results if q['candidate']=='crystal_lift']:
    plt.plot(np.arange(r['L']),r['linear_kernel'],label=r['train']+'→'+r['test'])
plt.axhline(0,linewidth=1);plt.xlabel('滞后样本（0=当前）');plt.ylabel('核系数');plt.legend();plt.tight_layout();plt.savefig(P2,dpi=180);plt.close()

wb=Workbook.create()
s=wb.worksheets.add('结论摘要')
summary=[['CL-OPS-UOI CPU验证：压制AR后的锚定增量模型'],
         ['结论','晶升/联合升速方向存在跨晶棒稳定的线性时滞核；完整非线性幅值K尚未超过线性核'],
         ['主目标','用 y(t) 作锚点，预测 Δ15y；不输入更早历史直径'],
         ['Sheet1突变',','.join(map(str,breaks['Sheet1']))],['Sheet2突变',','.join(map(str,breaks['Sheet2']))],
         ['晶升-埚升相关',f"Sheet1={correlation['Sheet1']:.5f}; Sheet2={correlation['Sheet2']:.5f}"],
         ['晶升核跨方向相关',f"{agreement['crystal_lift']:.5f}"],['运行耗时/秒',payload['elapsed_seconds']]]
s.get_range('A1:B8').values=summary;s.merge_cells('A1:B1');s.get_range('A1').format={'fill':'#0F766E','font':{'bold':True,'color':'#FFFFFF','size':14},'horizontal_alignment':'center'}
s.get_range('A2:A8').format={'fill':'#DDF4EF','font':{'bold':True}};s.get_range('A:B').format.wrap_text=True;s.get_range('A:A').format.column_width=24;s.get_range('B:B').format.column_width=80

rws=wb.worksheets.add('跨棒结果')
head=['候选','训练→测试','h','L','R','静态改善%','线性核改善%','非线性K改善%','K相对线性核%','线性核95%下界%','K95%下界%','P(K>0)','前半线性核%','后半线性核%','OOD比例%','峰值滞后','峰值系数','非线性λ']
rows=[head]
for r in results:
    rows.append([r['candidate'],r['train']+'→'+r['test'],r['h'],r['L'],r['R'],100*r['static_improvement'],100*r['FIR_improvement'],100*r['K_improvement'],100*r['K_vs_FIR'],100*r['FIR_bootstrap']['lo95'],100*r['K_bootstrap']['lo95'],r['K_bootstrap']['p_positive'],100*r['halves']['first'],100*r['halves']['second'],100*r['test_ood_fraction'],r['peak_lag'],r['peak_value'],str(r['K_lambda'])])
rws.get_range_by_indexes(0,0,len(rows),len(head)).values=rows;rws.get_range('A1:R1').format={'fill':'#0F766E','font':{'bold':True,'color':'#FFFFFF'},'wrap_text':True};rws.freeze_panes.freeze_rows(1);rws.get_range('A:R').format.column_width=14;rws.get_range('B:B').format.column_width=18;rws.get_range('R:R').format.column_width=24
rws.get_range('F2:Q10').format.number_format='0.000';rws.get_range('G2:H10').conditional_formats.add_color_scale({'minColor':'#FECACA','midColor':'#FEF3C7','maxColor':'#DCFCE7'})

hws=wb.worksheets.add('预测步与安慰剂')
hrows=[['候选','训练→测试','h','线性核改善%']]+[[d['candidate'],d['train']+'→'+d['test'],d['h'],100*d['improvement']] for d in horizon]
hrows.append([]);hrows.append(['候选','训练→测试','循环平移样本','改善%'])
for r in results:
    for p in r['placebo']:hrows.append([r['candidate'],r['train']+'→'+r['test'],p['shift'],100*p['improvement']])
hws.get_range_by_indexes(0,0,len(hrows),4).values=[x+[None]*(4-len(x)) for x in hrows];hws.get_range('A1:D1').format={'fill':'#0F766E','font':{'bold':True,'color':'#FFFFFF'}};hws.get_range('A:D').format.column_width=20

kws=wb.worksheets.add('晶升核')
pair=[r for r in results if r['candidate']=='crystal_lift']
krows=[['lag',pair[0]['train']+'→'+pair[0]['test'],pair[1]['train']+'→'+pair[1]['test']]]
for i in range(128):krows.append([i,pair[0]['linear_kernel'][i],pair[1]['linear_kernel'][i]])
kws.get_range_by_indexes(0,0,len(krows),3).values=krows;kws.get_range('A1:C1').format={'fill':'#0F766E','font':{'bold':True,'color':'#FFFFFF'}};kws.get_range('A:C').format.column_width=20;kws.freeze_panes.freeze_rows(1)

pws=wb.worksheets.add('数据处理协议')
prows=[['项目','处理'],['控制输入','主加热功率、晶升、晶转、埚升、埚转'],['输出','晶体直径'],['删除/不进入主支路','晶体长度、加热元件温度、氩气流量、炉压'],['历史输出','仅 y(t) 作锚点，不使用 y(t-1), y(t-2), ...'],['突变','|Δ直径|>0.20 mm 切为硬边界；不插值，不平滑连接，窗口不跨界'],['主验证','Sheet1→Sheet2 与 Sheet2→Sheet1'],['解释边界','晶升与埚升高度共线，只能认证升速控制族，不能独立因果归因'],['K层判定','线性时滞核属于K层线性子模型；非线性幅值块必须额外优于线性核才算完整K成功']]
pws.get_range_by_indexes(0,0,len(prows),2).values=prows;pws.get_range('A1:B1').format={'fill':'#0F766E','font':{'bold':True,'color':'#FFFFFF'}};pws.get_range('A:A').format.column_width=22;pws.get_range('B:B').format.column_width=84;pws.get_range('A:B').format.wrap_text=True

SpreadsheetFile.export_xlsx(wb).save(OUT_XLSX)
print(json.dumps({'json':OUT_JSON,'xlsx':OUT_XLSX,'plots':[P1,P2],'elapsed':payload['elapsed_seconds'],'results':[{k:r[k] for k in ['candidate','train','test','FIR_improvement','K_improvement','K_vs_FIR','static_improvement','peak_lag']} for r in results],'agreement':agreement},ensure_ascii=False,indent=2))
