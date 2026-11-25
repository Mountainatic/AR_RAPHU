"""Frozen ablations, horizon alignment, drift stress, and mechanism audits."""

import argparse
from functools import lru_cache
from pathlib import Path
from unittest.mock import patch

import core as c


np=c.np
p=c.p
VARIANTS=("full","A_only","no_difference","no_A","no_rho","force_all")
THRESHOLDS=((.15,.30),(.20,.35),(.25,.40),(.30,.45),(.35,.50))


def register(module,**details):
    out=c.ROOT/module
    c.x.bind(out/"REGISTRATION.json",dict(code_lock=c.verify_code(),runner=c.sha(Path(__file__)),**details))
    return out


class Forced(p.c.Engine):
    @lru_cache(maxsize=128)
    def c_route(self,end):
        model=self.c_select(end)
        return model,dict(admitted=model is not None,forced=True)

    @lru_cache(maxsize=128)
    def w_route(self,end):
        cc,w=self.w_select(end)
        return cc,w,dict(admitted=w is not None,forced=True)

    @lru_cache(maxsize=128)
    def a_route(self,end):
        _,_,_,audit=super().a_route(end)
        for family in sorted(audit["families"],key=lambda f:audit["families"][f]["mean_child_mse"]):
            cc,w,a=self.a_select(end,family)
            if a is not None:
                return cc,w,a,dict(admitted=True,selected_family=family,forced=True)
        cc,w,_=self.w_route(end)
        return cc,w,None,dict(admitted=False,forced=True)


class OnlyA(p.c.Engine):
    def c_route(self,end):
        return None,dict(admitted=False,removed=True)

    def w_route(self,end):
        return None,None,dict(admitted=False,removed=True)


def build_arm(q,a,k,fit,stop,variant):
    ref=k.copy()
    if variant=="force_all":
        ref["routing_status"]="ADMITTED"
    if variant=="A_only":
        ref["routing_status"]="BYPASSED"
    ctx=p.context(q,a,ref,fit,stop)
    if variant=="no_difference":
        ctx["x"][:,1::2]=0.
        ctx["g"][:,1::2]=0.
    if variant=="no_rho":
        ctx["g"][:]=1.
    cls={"force_all":Forced,"A_only":OnlyA}.get(variant,p.c.Engine)
    return cls(p.PIPE,ctx,q.calibration[0]),ref


def fit_ablation():
    root=register("E2",entities=c.MAIN,variants=VARIANTS,fit="C/W/A refit using the restricted parent stream; no_A deletes only terminal readout",
        difference="Zero all difference-view feature and support columns",A_only="Estimate A on the anchor residual with its usual family vote",
        no_rho="Unity trust and native drift routing disabled",force_all="Use certified candidates without structural admission gates; retain support and drift safeguards")
    all_models={}
    for name in c.MAIN:
        q,k,fit,frozen,_=c.load(name)
        c.configure(q)
        hi=q.calibration[1]
        a=c.data(q)
        models={}
        for arm in VARIANTS:
            if arm in ("full","no_A"):
                ref,model=k,frozen.copy()
                if arm=="no_A":
                    model["A"]=None
            else:
                model_engine,ref=build_arm(q,a,k,fit,hi,arm)
                model=model_engine.train(hi).copy()
            model["issue_from"]=q.train
            models[arm]=dict(reference=ref,chain=model)
            print("E2 frozen",name,arm,flush=True)
        all_models[name]=models
    c.x.bind(root/"FROZEN.json",dict(code_lock=c.verify_code(),registration=c.sha(root/"REGISTRATION.json"),models=all_models))


def evaluate_ablation():
    root=c.ROOT/"E2"
    lock=c.read(root/"FROZEN.json")
    assert lock["code_lock"]==c.verify_code() and lock["registration"]==c.sha(root/"REGISTRATION.json")
    result=[]
    for name in c.MAIN:
        q,k,fit,_,_=c.load(name)
        c.configure(q)
        a=c.data(q,full=True)
        t,masks=c.rows(q)
        for arm,m in lock["models"][name].items():
            eng,_=build_arm(q,a,k,fit,int(t[-1])+1,arm)
            model=m["chain"]
            py=p.c.predict(eng,t,model,failsafe=arm!="no_rho")
            for label,mask in masks.items():
                scores=c.score(a[t[mask]+q.H,0],py["pred"][mask],a[t[mask]-q.D,0])
                result.append(dict(data=name,split=label,variant=arm,N=int(mask.sum()),**scores,
                    K=m["reference"]["routing_status"]=="ADMITTED",C=model["C"] is not None,W=model["W"] is not None,
                    A=model["A"]["spec"][0][0] if model["A"] else "",
                    fallback_rows=int(py["active"][mask].sum())))
    c.csv_table(root/"ABLATION.csv",result)
    c.write(root/"EVIDENCE.json",result)


def sequence_case(q,k,fit,chain):
    a=c.data(q)
    start=q.train
    width=512
    site=q.calibration[0]
    stable=a[site:site+width].copy()
    chip=np.tile(stable,(4,1))
    delta=20*np.maximum(np.ptp(a[:q.parent,1:],axis=0),1e-8)
    chip[width:2*width,1:]+=delta
    full=np.ascontiguousarray(np.r_[a,chip])
    t=np.arange(start,start+len(chip))
    yes,err=c.cert_pair(q,full,k,fit,chain,t)
    no,_=c.replay(q,full,k,fit,chain,t,monitor=False)
    transitions=[dict(offset=int(i),event=int(yes["event"][i])) for i in np.flatnonzero(yes["event"])]
    assert any(e["event"]==1 for e in transitions) and any(e["event"]==-1 for e in transitions)
    table=[dict(offset=i,shift=int(width<=i<2*width),trust=yes["trust"][i],mean=yes["mean"][i],
                fallback=int(yes["active"][i]),event=int(yes["event"][i]),
                correction_on=yes["pred"][i]-full[t[i],0],correction_off=no[i]-full[t[i],0])
           for i in range(len(chip))]
    return table,dict(shift_start=width,shift_end=2*width,source_start=site,transitions=transitions,parity=err,
                      intervention="20 parent-range process offsets simultaneously, labels copied unmodified")


def horizon():
    root=register("E3",H=c.HORIZONS,D=0,L=128,target_rule="Use identical physical target indices >= split+max(H)",
        variants=("PRISM","Persistence","ARX"),drift="Rod2 H16 with guard on/off, plus a repeated train-only synthetic offset pulse")
    rows=[]
    for name in c.RODS:
        for h in c.HORIZONS:
            q,k,fit,chain,_=c.load(name,h)
            a=c.data(q,full=True)
            t=q.test_origins()
            yy,diag=c.replay(q,a,k,fit,chain,t)
            dst=c.parent(h)/name
            with np.load(dst/"prism/PREDICTIONS.npz") as prior:
                np.testing.assert_array_equal(prior["origins"],t)
                np.testing.assert_allclose(yy,prior["PRISM"],rtol=0,atol=5e-10)
            out={"PRISM":yy,"Persistence":a[t,0]}
            linear=dst/"linear/PREDICTIONS.npz"
            if linear.exists():
                with np.load(linear) as prior:
                    np.testing.assert_array_equal(prior["origins"],t)
                    out["ARX"]=prior["ARX"]
            for segment,lo,hi in (("development",q.train+max(c.HORIZONS),q.dev),("tail",q.dev+max(c.HORIZONS),q.total)):
                use=(t+h>=lo)&(t+h<hi)
                np.testing.assert_array_equal(t[use]+h,np.arange(lo,hi))
                for method,y in out.items():
                    rows.append(dict(data=name,split=segment,H=h,N=int(use.sum()),target_start=lo,target_stop=hi,method=method,
                        **c.score(a[t[use]+h,0],y[use],a[t[use],0]),
                        fallback_rows=int(diag[use,2].sum()) if method=="PRISM" else ""))
            print("E3 horizon",name,h,flush=True)
    c.csv_table(root/"HORIZON.csv",rows)
    q,k,fit,chain,_=c.load("CZ_Rod2")
    a=c.data(q,full=True)
    t,masks=c.rows(q)
    on,d=c.replay(q,a,k,fit,chain,t)
    off,_=c.replay(q,a,k,fit,chain,t,monitor=False)
    result=[]
    for label,mask in masks.items():
        for mode,y in (("guard_on",on),("guard_off",off)):
            result.append(dict(data=q.name,split=label,method=mode,N=int(mask.sum()),
                guard_rows=int(d[mask,2].sum()) if mode=="guard_on" else 0,
                on_off_max_abs=float(np.max(abs(on[mask]-off[mask]))),
                **c.score(a[t[mask]+q.H,0],y[mask],a[t[mask],0])))
    c.csv_table(root/"ROD2_FALLBACK.csv",result)
    table,receipt=sequence_case(q,k,fit,chain)
    c.csv_table(root/"ROD2_SYNTHETIC.csv",table)
    c.write(root/"EVIDENCE.json",dict(common=rows,fallback=result,synthetic=receipt))


def alignment(q,k,fit,chain,train):
    hi=q.calibration[1]
    eng=c.engine(q,train,k,fit,hi)
    t=np.arange(eng.lo,hi)
    cc=chain["C"] or eng.c_select(hi)
    result=[]
    if cc:
        coef=np.asarray(cc["coef"])
        trust=eng.ctx["g"][t]
        X=(eng.ctx["x"][t]-cc["mean"])/cc["scale"]*trust
        u=trust.max(1)
        y=eng.y[t]-eng.k[t]
        r=y-X@coef-u*cc["intercept"]
        alpha,ratio=cc["spec"]
        alpha*=max(np.std(y),1e-10)
        grad=-X.T@r/len(t)+alpha*(1-ratio)*coef
        nz=coef!=0.
        v=np.where(nz,np.abs(grad+alpha*ratio*np.sign(coef)),np.maximum(np.abs(grad)-alpha*ratio,0.))
        intercept=float(abs(u@r/len(t)))
        np.testing.assert_allclose(eng.c_eval(t,cc)[0], eng.k[t]+X@coef+u*cc["intercept"],rtol=0,atol=1e-12)
        assert intercept<1e-10
        result.append(dict(data=q.name,module="C",deployed=chain["C"] is not None,N=len(t),
            intercept_gradient=intercept,max_KKT=float(v.max()),objective=float(.5*np.mean(r*r)+alpha*(ratio*np.abs(coef).sum()+
                .5*(1-ratio)*(coef@coef))),status="MEASURED"))
    c_w,w=eng.w_select(hi)
    adopted=chain["W"] is not None
    if adopted:
        w,c_w=chain["W"],chain["C"]
    if w:
        first,stream=eng.c_stream(hi,tuple(c_w["spec"]) if c_w else None)
        t=np.arange(first,hi)
        kc,z,rho=stream[:,t-eng.lo]
        family,n,alpha,direction=w["spec"]
        z=np.clip(z,*w["limits"])
        mono=family==p.PIPE.w.MONOTONE
        B=p.PIPE.w.build_w_design(z,z,family,n).train*(direction if mono else 1.)
        B=B*rho[:,None]
        coef=np.asarray(w["contract"]["coefficient"])
        P=p.c.difference_penalty(len(coef))
        r=B@coef-(eng.y[t]-kc)
        grad=2*(B.T@r/len(t)+alpha*(P@coef))
        violation=np.where(coef>1e-9,abs(grad),np.maximum(-grad,0.)) if mono else abs(grad)
        np.testing.assert_allclose(eng.w_eval(kc,z,rho,w),kc+B@coef,rtol=0,atol=1e-12)
        result.append(dict(data=q.name,module="W",deployed=adopted,N=len(t),
            intercept_gradient=None,max_KKT=float(violation.max()),objective=float(np.mean(r*r)+alpha*(coef@P@coef)),
            status=w["contract"]["numerical_certificate"]["status"]))
    return result


def mechanisms():
    root=register("E5",thresholds=THRESHOLDS,gate_window=128,
        bias="Shift all physical process channels by five parent ranges at training boundary and wait L+H+D",
        audit="Separate level and difference view diagnostics. Compare additive and telescoping identities on common tail.")
    align=[]; bias=[]; audit=[]; sensitivity=[]; attribution=[]
    for name in c.MAIN:
        q,k,fit,chain,_=c.load(name)
        train=c.data(q)
        align.extend(alignment(q,k,fit,chain,train))
        a=c.data(q,full=True)
        t,masks=c.rows(q)
        py,err=c.cert_pair(q,a,k,fit,chain,t)
        cut=int(t[masks["tail"]][len(t[masks["tail"]])//2])
        sub=t[t<=cut]
        trial=q.poison(a,cut)
        changed,_=c.cert_pair(q,trial,k,fit,chain,sub)
        clean,diag=c.replay(q,a,k,fit,chain,sub)
        corrupt,other=c.replay(q,trial,k,fit,chain,sub)
        np.testing.assert_array_equal(clean,corrupt)
        np.testing.assert_array_equal(diag,other)
        err_poison=float(np.max(abs(py["pred"][:len(sub)]-changed["pred"])))
        assert err_poison<=5e-10
        sel=masks["tail"]
        anchor=a[t[sel]-q.D,0]
        y=a[t[sel]+q.H,0]
        comp=py["components"][sel]
        cumulative=np.column_stack((anchor,anchor[:,None]+np.cumsum(comp,axis=1)))
        squared=(y[:,None]-cumulative)**2
        gain=-np.diff(squared,axis=1)
        telescoping=float(np.max(abs(gain.sum(1)-(squared[:,0]-squared[:,-1]))))
        reconstruct=float(np.max(abs(cumulative[:,-1]-py["pred"][sel])))
        audit.append(dict(data=name,cut=cut,prefix_N=len(sub),python_C_max_abs=err,native_poison_max_abs=0.,
                         python_poison_max_abs=err_poison,additive_max_abs=reconstruct,SSE_identity_max_abs=telescoping))
        attribution.extend(dict(data=name,stage=m,active_rows=int(np.count_nonzero(comp[:,i])),
            mean_SSE_gain=float(np.mean(gain[:,i])),sum_SSE_gain=float(np.sum(gain[:,i])))
            for i,m in enumerate(("K","C","W","A")))
        alt=a.copy()
        alt[q.train:,1:]+=5*np.maximum(np.ptp(train[:q.parent,1:],axis=0),1e-8)
        clean_engine=c.engine(q,a,k,fit,int(t[-1])+1)
        dirty_engine=c.engine(q,alt,k,fit,int(t[-1])+1)
        span=t[t>=q.train+q.L+q.mature]
        for i,view in enumerate(clean_engine.ctx["names"]):
            clean_g=clean_engine.ctx["g"][span,i]
            dirty_g=dirty_engine.ctx["g"][span,i]
            dv=clean_engine.ctx["x"][span,i]-dirty_engine.ctx["x"][span,i]
            bias.append(dict(data=name,view=view,N=len(span),clean_mean_rho=float(clean_g.mean()),
                  shifted_mean_rho=float(dirty_g.mean()),max_rho_change=float(np.max(abs(dirty_g-clean_g))),
                  max_abs_raw_response_change=float(np.max(abs(dv)))))
        for enter,exit_ in THRESHOLDS:
            with patch.object(p.c,"DRIFT_ENTER",enter),patch.object(p.c,"DRIFT_EXIT",exit_):
                y_pred=p.c.predict(clean_engine,t,chain)
            for label,use in masks.items():
                sensitivity.append(dict(data=name,split=label,enter=enter,exit=exit_,N=int(use.sum()),
                    fallback_rows=int(y_pred["active"][use].sum()),
                    **c.score(a[t[use]+q.H,0],y_pred["pred"][use],a[t[use]-q.D,0])))
        print("E5",name,flush=True)
    tables=dict(ALIGNMENT=align,BIAS_PRESSURE=bias,CAUSAL=audit,ATTRIBUTION=attribution,SENSITIVITY=sensitivity)
    for key,t in tables.items():
        c.csv_table(root/(key+".csv"),t)
    c.write(root/"EVIDENCE.json",tables)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module",choices=("e2","e3","e5"))
    m=parser.parse_args().module
    if m=="e2":
        fit_ablation()
        evaluate_ablation()
    elif m=="e3":
        horizon()
    else:
        mechanisms()
