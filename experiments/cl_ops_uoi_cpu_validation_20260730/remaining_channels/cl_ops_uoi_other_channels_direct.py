import os, json, csv, time, zipfile, re, hashlib
import xml.etree.ElementTree as ET
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.linalg import cho_factor, cho_solve
import matplotlib.pyplot as plt

IN='/mnt/data/实验数据1-张(2).xlsx'
OUTDIR='/mnt/data/CL_OPS_UOI_其余通道验证_20260730'
os.makedirs(OUTDIR,exist_ok=True)
OUT_JSON=os.path.join(OUTDIR,'CL_OPS_UOI_其余通道_QK验证.json')
OUT_CSV=os.path.join(OUTDIR,'RESULTS_SUMMARY.csv')
OUT_MD=os.path.join(OUTDIR,'RESULTS_REPORT.md')
P1=os.path.join(OUTDIR,'其余通道_独立QK改善率.png')
P2=os.path.join(OUTDIR,'其余通道_联合升速后条件改善率.png')
P3=os.path.join(OUTDIR,'其余通道_线性核跨棒对比.png')
ZIP_OUT='/mnt/data/CL_OPS_UOI_其余通道_QK验证_20260730.zip'
L=128; R=32; H=15

# ---------- direct XLSX reader (OOXML, no external spreadsheet library) ----------
def col_index(cell_ref):
    letters=re.match(r'([A-Z]+)',cell_ref).group(1); n=0
    for ch in letters: n=n*26+ord(ch)-64
    return n-1

def read_xlsx(path):
    with zipfile.ZipFile(path) as z:
        ns={'a':'http://schemas.openxmlformats.org/spreadsheetml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships','p':'http://schemas.openxmlformats.org/package/2006/relationships'}
        shared=[]
        if 'xl/sharedStrings.xml' in z.namelist():
            root=ET.fromstring(z.read('xl/sharedStrings.xml'))
            for si in root.findall('a:si',ns): shared.append(''.join(t.text or '' for t in si.findall('.//a:t',ns)))
        wb=ET.fromstring(z.read('xl/workbook.xml'))
        relroot=ET.fromstring(z.read('xl/_rels/workbook.xml.rels'))
        rels={r.attrib['Id']:r.attrib['Target'] for r in relroot}
        out={}
        for sh in wb.find('a:sheets',ns):
            name=sh.attrib['name']; rid=sh.attrib['{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id']
            target=rels[rid]
            if not target.startswith('xl/'): target='xl/'+target.lstrip('/')
            root=ET.fromstring(z.read(target)); rows=[]; maxc=0
            for row in root.findall('.//a:sheetData/a:row',ns):
                vals={}
                for c in row.findall('a:c',ns):
                    idx=col_index(c.attrib['r']); typ=c.attrib.get('t'); v=c.find('a:v',ns)
                    if v is None: val=None
                    elif typ=='s': val=shared[int(v.text)]
                    elif typ=='b': val=bool(int(v.text))
                    else:
                        try: val=float(v.text)
                        except: val=v.text
                    vals[idx]=val; maxc=max(maxc,idx+1)
                rows.append(vals)
            matrix=[]
            for d in rows: matrix.append([d.get(i,None) for i in range(maxc)])
            out[name]=matrix
        return out

raw=read_xlsx(IN)
rec={}; breaks={}; starts={}
for s,rows in raw.items():
    hdr=rows[0]; A=np.asarray(rows[1:],dtype=np.float64); ix={str(v):i for i,v in enumerate(hdr)}
    y=A[:,ix['晶体直径']]; br=(np.where(np.abs(np.diff(y))>0.20)[0]+1).astype(int).tolist(); st=max(br)+1 if br else 0
    breaks[s]=br; starts[s]=st
    rec[s]={'y':y,'power':A[:,ix['主加热功率']],'cl':A[:,ix['晶升速度']],'cr':A[:,ix['晶转速度']],'kl':A[:,ix['埚升速度']],'kr':A[:,ix['埚转速度']],'start':st}

# ---------- signal transforms ----------
def causal_ma(x,w):
    x=np.asarray(x,float); cs=np.cumsum(np.r_[0.0,x]); i=np.arange(len(x)); j=np.maximum(0,i-w+1)
    return (cs[i+1]-cs[j])/(i-j+1)
def hp(x,w): return np.asarray(x,float)-causal_ma(x,w)
def signal(sheet,c):
    r=rec[sheet]; st=r['start']
    return {'power':r['power'][st:],'crystal_rotation':r['cr'][st:],'crucible_rotation':r['kr'][st:],'crystal_rotation_hp512':hp(r['cr'][st:],512),'crucible_rotation_hp32':hp(r['kr'][st:],32)}[c]
def joint_signal(train,test):
    a,b=rec[train],rec[test]; sa,sb=a['start'],b['start']
    X=np.c_[a['cl'][sa:],a['kl'][sa:]]; Xt=np.c_[b['cl'][sb:],b['kl'][sb:]]
    mu=X.mean(0); sd=X.std(0); Z=(X-mu)/sd; vals,vecs=np.linalg.eigh(Z.T@Z/len(Z)); v=vecs[:,-1]
    if v.sum()<0:v=-v
    return Z@v,((Xt-mu)/sd)@v,{'mu':mu.tolist(),'sd':sd.tolist(),'v':v.tolist(),'explained':float(vals[-1]/vals.sum())}

# ---------- K model ----------
def Bmat():
    t=np.arange(L)[:,None]; r=np.arange(R)[None,:]; B=np.cos(np.pi*(t+.5)*r/L); B[:,0]/=np.sqrt(L); B[:,1:]*=np.sqrt(2/L); return B
B=Bmat()
def design(u,y,h,order,mu=None,sd=None):
    if mu is None:mu=float(np.mean(u))
    if sd is None:sd=float(np.std(u))
    if sd<1e-12:sd=1.0
    z=np.clip((u-mu)/sd,-3.5,3.5); W=sliding_window_view(z,L)[:len(z)-L-h+1][:,::-1]
    Hs=[W]
    if order>=2:Hs.append((W*W-1)/np.sqrt(2))
    if order>=3:Hs.append((W**3-3*W)/np.sqrt(6))
    X=np.concatenate([q@B for q in Hs],axis=1); d=y[L-1+h:]-y[L-1:-h]
    return X,d,mu,sd,z

def stats(X,y):
    xm=X.mean(0); ym=float(y.mean()); Xc=X-xm; yc=y-ym; return xm,ym,Xc.T@Xc,Xc.T@yc
def solve(st,lams):
    xm,ym,G,b=st; diag=np.concatenate([np.full(R,v) for v in lams]); A=G+np.diag(diag)
    try:c=cho_solve(cho_factor(A,lower=True,check_finite=False),b,check_finite=False)
    except:c=np.linalg.lstsq(A,b,rcond=None)[0]
    return ym-xm@c,c

def select(X,d,order):
    n=len(d); cut=int(.72*n); gap=256; tr=slice(0,cut-gap); va=slice(cut+gap,None); st=stats(X[tr],d[tr]); base=np.mean((d[va]-d[tr].mean())**2)
    grids=[(x,) for x in np.logspace(-4,5,10)] if order==1 else [(a,b,b) for a in [1e-3,1e-1,10,1e3] for b in [1e-1,10,1e3,1e5,1e8,1e12]]
    best=None
    for lam in grids:
        inter,c=solve(st,lam); ratio=np.mean((d[va]-(inter+X[va]@c))**2)/base
        if best is None or ratio<best[0]:best=(float(ratio),lam)
    return best

def fit(u,y,v,yt,h,order,target_train=None,target_test=None):
    X,d,mu,sd,z=design(u,y,h,order); Xt,dt,_,_,zt=design(v,yt,h,order,mu,sd)
    if target_train is not None:d=np.asarray(target_train)
    if target_test is not None:dt=np.asarray(target_test)
    sel=select(X,d,order); inter,c=solve(stats(X,d),sel[1]); return {'X':X,'d':d,'Xt':Xt,'dt':dt,'z':z,'zt':zt,'sel':sel,'inter':inter,'c':c,'pred':inter+Xt@c}

def bootstrap(base_loss,model_loss,seed):
    rng=np.random.default_rng(seed); block=256; n=min(len(base_loss),len(model_loss)); nb=n//block
    b=base_loss[:nb*block].reshape(nb,block).sum(1); m=model_loss[:nb*block].reshape(nb,block).sum(1)
    idx=rng.integers(0,nb,size=(1000,nb)); vals=1-m[idx].sum(1)/b[idx].sum(1)
    return {'lo95':float(np.quantile(vals,.025)),'median':float(np.median(vals)),'hi95':float(np.quantile(vals,.975)),'p_positive':float(np.mean(vals>0))}

def summarize(lin,non,target,base_pred,seed):
    bl=(target-base_pred)**2; ll=(target-lin['pred'])**2; nl=(target-non['pred'])**2; klin=B@lin['c']; blocks=[B@non['c'][j*R:(j+1)*R] for j in range(3)]; m=len(target)//2
    return {'base_rmse':float(np.sqrt(bl.mean())),'linear_rmse':float(np.sqrt(ll.mean())),'nonlinear_rmse':float(np.sqrt(nl.mean())),'linear_improvement':float(1-ll.mean()/bl.mean()),'nonlinear_improvement':float(1-nl.mean()/bl.mean()),'nonlinear_vs_linear':float(1-nl.mean()/ll.mean()),'linear_bootstrap':bootstrap(bl,ll,seed),'nonlinear_bootstrap':bootstrap(bl,nl,seed+1),'nonlinear_vs_linear_bootstrap':bootstrap(ll,nl,seed+2),'halves':{'first':float(1-ll[:m].mean()/bl[:m].mean()),'second':float(1-ll[m:].mean()/bl[m:].mean())},'ood_fraction':float(np.mean((lin['zt']<lin['z'].min())|(lin['zt']>lin['z'].max()))),'linear_lambda':list(map(float,lin['sel'][1])),'nonlinear_lambda':list(map(float,non['sel'][1])),'linear_kernel':klin.tolist(),'nonlinear_blocks':[x.tolist() for x in blocks],'peak_lag':int(np.argmax(np.abs(klin))),'peak_value':float(klin[np.argmax(np.abs(klin))])}

lift_cache={}
def lift_base(train,test,h):
    key=(train,test,h)
    if key in lift_cache:return lift_cache[key]
    u,v,p=joint_signal(train,test); y=rec[train]['y'][rec[train]['start']:];yt=rec[test]['y'][rec[test]['start']:]; f=fit(u,y,v,yt,h,1); tr=f['d']-(f['inter']+f['X']@f['c']); te=f['dt']-f['pred']; lift_cache[key]=(tr,te,p,f); return lift_cache[key]

def run_direction(train,test,cand,h=H,conditional=False):
    u,v=signal(train,cand),signal(test,cand); y=rec[train]['y'][rec[train]['start']:];yt=rec[test]['y'][rec[test]['start']:]
    if conditional:
        tr,te,p,_=lift_base(train,test,h); lin=fit(u,y,v,yt,h,1,tr,te); non=fit(u,y,v,yt,h,3,tr,te); out=summarize(lin,non,te,np.zeros_like(te),30 if train=='Sheet1' else 40); out['pca']=p
    else:
        lin=fit(u,y,v,yt,h,1); non=fit(u,y,v,yt,h,3); base=np.full_like(lin['dt'],float(lin['d'].mean())); out=summarize(lin,non,lin['dt'],base,10 if train=='Sheet1' else 20)
    out.update({'mode':'conditional_after_joint_lift' if conditional else 'standalone','candidate':cand,'train':train,'test':test,'h':h,'L':L,'R':R,'n_test':len(lin['dt'])}); return out

def run_linear_h(train,test,cand,h,conditional):
    u,v=signal(train,cand),signal(test,cand);y=rec[train]['y'][rec[train]['start']:];yt=rec[test]['y'][rec[test]['start']:]
    if conditional:
        tr,te,_,_=lift_base(train,test,h); f=fit(u,y,v,yt,h,1,tr,te); target=te;base=np.zeros_like(te)
    else:
        f=fit(u,y,v,yt,h,1);target=f['dt'];base=np.full_like(target,float(f['d'].mean()))
    return {'candidate':cand,'train':train,'test':test,'h':h,'conditional':conditional,'linear_improvement':float(1-np.mean((target-f['pred'])**2)/np.mean((target-base)**2))}

cands=['power','crystal_rotation','crucible_rotation','crystal_rotation_hp512','crucible_rotation_hp32']
t0=time.time();stand=[];cond=[]
for c in cands:
    for tr,te in [('Sheet1','Sheet2'),('Sheet2','Sheet1')]:
        stand.append(run_direction(tr,te,c,H,False)); print('stand',c,tr,flush=True)
        cond.append(run_direction(tr,te,c,H,True)); print('cond',c,tr,flush=True)
horiz=[]
for c in cands[:3]:
    for h in [1,60]:
        for tr,te in [('Sheet1','Sheet2'),('Sheet2','Sheet1')]:
            horiz.append(run_linear_h(tr,te,c,h,False));horiz.append(run_linear_h(tr,te,c,h,True))

def status(res,c):
    p=[x for x in res if x['candidate']==c];ag=float(np.corrcoef(p[0]['linear_kernel'],p[1]['linear_kernel'])[0,1]);q=all(x['linear_improvement']>0 and x['linear_bootstrap']['lo95']>0 for x in p);lk=q and ag>=.70;nk=all(x['nonlinear_vs_linear']>0 and x['nonlinear_vs_linear_bootstrap']['lo95']>0 for x in p);return {'Q_pass':q,'linear_K_pass':lk,'nonlinear_K_pass':nk,'kernel_agreement':ag}
statuses={'standalone':{c:status(stand,c) for c in cands},'conditional_after_joint_lift':{c:status(cond,c) for c in cands}}
payload={'input':os.path.basename(IN),'protocol':{'anchor':'y(t)','history_output_excluded':True,'h_primary':H,'L':L,'R':R,'joint_lift_frozen_baseline':True,'highpass_diagnostics':{'crystal_rotation':512,'crucible_rotation':32}},'breaks':breaks,'starts':starts,'standalone':stand,'conditional_after_joint_lift':cond,'horizon_checks':horiz,'statuses':statuses,'elapsed_seconds':time.time()-t0}
with open(OUT_JSON,'w',encoding='utf8') as f:json.dump(payload,f,ensure_ascii=False,indent=2)

cn={'power':'主加热功率','crystal_rotation':'晶转速度','crucible_rotation':'埚转速度','crystal_rotation_hp512':'晶转高通','crucible_rotation_hp32':'埚转高通'}
with open(OUT_CSV,'w',newline='',encoding='utf-8-sig') as f:
    w=csv.writer(f);w.writerow(['mode','candidate','train','test','linear_improvement_pct','linear_ci_lo_pct','nonlinear_improvement_pct','nonlinear_vs_linear_pct','kernel_agreement','Q_pass','linear_K_pass','nonlinear_K_pass','ood_pct','peak_lag'])
    for mode,res in [('standalone',stand),('conditional_after_joint_lift',cond)]:
        for x in res:
            s=statuses[mode][x['candidate']];w.writerow([mode,cn[x['candidate']],x['train'],x['test'],100*x['linear_improvement'],100*x['linear_bootstrap']['lo95'],100*x['nonlinear_improvement'],100*x['nonlinear_vs_linear'],s['kernel_agreement'],s['Q_pass'],s['linear_K_pass'],s['nonlinear_K_pass'],100*x['ood_fraction'],x['peak_lag']])

# figures
plt.rcParams['font.sans-serif']=['DejaVu Sans']
def bars(res,path,title):
    labs=[];a=[];b=[]
    for c in cands:
        for tr in ['Sheet1','Sheet2']:
            x=next(q for q in res if q['candidate']==c and q['train']==tr);labs.append(c+'\n'+x['train']+'→'+x['test']);a.append(100*x['linear_improvement']);b.append(100*x['nonlinear_improvement'])
    xx=np.arange(len(labs));ww=.38;plt.figure(figsize=(15,6));plt.bar(xx-ww/2,a,ww,label='linear lag K');plt.bar(xx+ww/2,b,ww,label='nonlinear K');plt.axhline(0,color='black',lw=.8);plt.xticks(xx,labs,rotation=25,ha='right');plt.ylabel('MSE improvement (%)');plt.title(title);plt.legend();plt.tight_layout();plt.savefig(path,dpi=170);plt.close()
bars(stand,P1,'Standalone Q/K validation');bars(cond,P2,'Conditional Q/K after frozen joint-lift channel')
fig,axs=plt.subplots(3,2,figsize=(13,12));axs=axs.ravel()
for ax,c in zip(axs,cands):
    for x in [q for q in stand if q['candidate']==c]:ax.plot(np.arange(L),x['linear_kernel'],label=x['train']+'→'+x['test'])
    ax.axhline(0,color='black',lw=.6);ax.set_title(c);ax.legend(fontsize=8)
axs[-1].axis('off');plt.tight_layout();plt.savefig(P3,dpi=170);plt.close()

# Markdown report
lines=['# CL-OPS-UOI 其余输入通道 Q/K 验证','',f'- 运行时间：{payload["elapsed_seconds"]:.2f} s','- 主协议：仅当前直径 `y(t)` 作锚点，不使用更早历史直径。','- 联合升速作为已通过且冻结的主通道；另检验其他通道的条件增量。','', '## 门禁状态','', '|模式|通道|Q|线性K|非线性K|跨方向核相关|','|---|---|---:|---:|---:|---:|']
for mode in statuses:
    for c,s in statuses[mode].items():lines.append(f'|{mode}|{cn[c]}|{s["Q_pass"]}|{s["linear_K_pass"]}|{s["nonlinear_K_pass"]}|{s["kernel_agreement"]:.4f}|')
lines += ['', '## 主结果（h=15）','', '|模式|通道|方向|线性K改善|95%下界|非线性K改善|非线性相对线性|OOD|','|---|---|---|---:|---:|---:|---:|---:|']
for mode,res in [('独立',stand),('联合升速后',cond)]:
    for x in res:lines.append(f'|{mode}|{cn[x["candidate"]]}|{x["train"]}→{x["test"]}|{100*x["linear_improvement"]:.3f}%|{100*x["linear_bootstrap"]["lo95"]:.3f}%|{100*x["nonlinear_improvement"]:.3f}%|{100*x["nonlinear_vs_linear"]:.3f}%|{100*x["ood_fraction"]:.3f}%|')
with open(OUT_MD,'w',encoding='utf8') as f:f.write('\n'.join(lines)+'\n')

# Package + hashes
with zipfile.ZipFile(ZIP_OUT,'w',zipfile.ZIP_DEFLATED) as z:
    z.write(__file__,arcname='src/cl_ops_uoi_other_channels_direct.py')
    for p in [OUT_JSON,OUT_CSV,OUT_MD,P1,P2,P3]:z.write(p,arcname='results/'+os.path.basename(p))
for p in [OUT_JSON,OUT_CSV,OUT_MD,P1,P2,P3,ZIP_OUT]:
    with open(p,'rb') as f: print(hashlib.sha256(f.read()).hexdigest(),p,flush=True)
print(json.dumps({'statuses':statuses,'elapsed_seconds':payload['elapsed_seconds'],'zip':ZIP_OUT},ensure_ascii=False,indent=2))
