"""Restricted refits and read-only probes against the sealed PRISM snapshot."""

import argparse
from dataclasses import replace
from pathlib import Path
import pickle
import sys
from time import perf_counter
from unittest.mock import patch

import numpy as np


ARMS = ("Full", "No_difference", "FIR_only", "Trust_one", "HAC_off", "Level_state")
SRU = ("MEA_GAS", "AIR_MEA1", "AIR_MEA2", "SWS_GAS", "AIR_SWS_TOT")


def rmse(y, prediction):
    return float(np.sqrt(np.mean((prediction-y)**2)))


def metrics(q, full, rows, predictor):
    from benchmark_addendum import fault
    trials = fault(q,full)
    stress = rows[:512]
    clean = predictor(full)[rows]
    oscillation = predictor(trials["oscillation"])[stress]
    return dict(clean_rmse=rmse(full.truth[rows],clean),oscillation_rmse=rmse(full.truth[stress],oscillation),
                oscillation_shift_mae=float(abs(oscillation-clean[:512]).mean()))


def refit(q, model, variant):
    from laguerre_anchored import Profiled,Standardized
    import laguerre_dual_views as dv
    from laguerre_dual_views import Fold,independent_taps,readout
    from laguerre_ecm import Dynamics
    from laguerre_joint import Endo
    from laguerre_process import fixed_projection
    safe=model.safe
    fmap=model.mapping
    near=fmap.dynamics.near
    triplet=q.model_fit["parameters"]
    curve=q.model_fit["curvature"]
    train=q.train
    rows=q.rows
    delay=q.gap
    x=lambda s:s.x if q.cz else s.views()
    ystates=safe.dynamics.batch
    if variant=="No_difference":
        support=fmap.support
        fmap=replace(fmap,basis=fmap.basis[::2],
                     support=replace(support, lower=support.lower[::2],
                                     upper=support.upper[::2],sigma=support.sigma[::2]))
        x=lambda s:s.x
    elif variant=="FIR_only":
        fmap=replace(fmap,dynamics=Dynamics(near,()))
    else:
        dy=safe.dynamics
        yd=Dynamics(dy.near+1,())
        def ystates(anchor):
            states=yd.batch(anchor)
            ready=np.arange(len(anchor))[:,None] >= dy.anchor_delay+np.arange(dy.near+1)
            states=np.where(ready,states-dy.center,0.)
            return np.column_stack((states[:,1:],states[:,0]))
        spec=q.model_fit["safe"]
        a=train.anchor()
        e=ystates(a)
        if spec["admitted"]:
            intercept,beta,_=Standardized(e[rows],(train.truth-a)[rows]).whole(spec["alpha"],spec["mix"])
        else:
            intercept,beta=0.,np.zeros(e.shape[1])
        safe=Endo(safe.dynamics,intercept,beta)
    a=train.anchor()
    state=ystates(a)
    safe_hat=safe.predict(a,state)
    design=fmap.design(x(train))
    taps=design.raw.shape[2]
    use_dual=not q.cz and variant!="No_difference"
    if use_dual:
        mask=np.tile(independent_taps(near,delay)[:,:taps],(train.x.shape[1],1))
    else:
        mask=np.ones(design.raw.shape[1:3],dtype=bool)
    feature=design.raw[rows,:,:,0][:,mask]
    proof=Profiled(Standardized(np.column_stack((state[rows],feature)),(train.truth-safe_hat)[rows]),state.shape[1])
    f=Fold(fmap,design,rows,state,safe,safe_hat,mask,design.raw[rows].mean(axis=0),train.truth,proof,
           near,delay if use_dual else near)
    def bins(r,t):
        if r==t:
            return fixed_projection(r,t+1)[:t,:-1]
        return fixed_projection(r,t)
    with patch.object(dv,"fixed_projection",bins):
        head,certificate=readout(f,triplet["alpha"],triplet["mix"],curve)
    def predict(s):
        aa=s.anchor()
        ss=ystates(aa)
        y0=safe.predict(aa,ss)
        if not model.exogenous_on:
            return y0
        d=fmap.design(x(s))
        return head.mix(d.raw,d.trust,ss,y0)[0]
    return predict,certificate


def switched(q,model,mode):
    def predict(s):
        a=s.anchor()
        t=model.safe.dynamics.batch(a)
        y0=model.safe.predict(a,t)
        if not model.exogenous_on and mode!="HAC_off":
            return y0
        b=model.mapping.design(s.x if q.cz else s.views())
        trust=np.ones_like(b.trust) if mode=="Trust_one" else b.trust
        return model.joint.mix(b.raw,trust,t,y0)[0]
    return predict


def ablation(q, model, full, rows, truth_archive):
    base=lambda s:model.predict(s)["prediction"]
    result={"Full":dict(**metrics(q,full,rows,base),type="frozen")}
    for key,score in (("clean",result["Full"]["clean_rmse"]),
                      ("oscillation",result["Full"]["oscillation_rmse"])):
        np.testing.assert_allclose(score,truth_archive["panels"][key]["metrics"]["PRISM"]["rmse"],
                                   atol=2e-12,rtol=0.)
    for arm in ARMS[1:]:
        if arm=="No_difference" and q.cz:
            result[arm]=dict(**metrics(q,full,rows,base),type="already_level_only")
            continue
        if arm in ("Trust_one","HAC_off"):
            pred,meta=switched(q,model,arm),{"type":"frozen_switch"}
        else:
            start=perf_counter()
            pred,cert=refit(q,model,arm)
            meta=dict(type="restricted_training_refit",refit_seconds=perf_counter()-start,fit=cert)
        result[arm]=dict(**metrics(q,full,rows,pred),**meta)
        print(q.a.task,arm,result[arm],flush=True)
    return result


def impulse(q,model):
    n=64
    h=model.mapping.dynamics.batch(np.r_[1.,np.zeros(n-1)])
    theta=model.joint.theta
    per_view=[]
    train_views=q.train.x if q.cz else q.train.views()
    for t,b in enumerate(model.mapping.basis):
        raw=train_views[:,t]
        z=float(b.standardize(float(np.median(raw))))
        # Local derivative of the frozen spline/saturation at the view median.
        step=1e-4
        slope=(b.transform(np.array(z+step))-b.transform(np.array(z-step)))/(2.*step*b.scale)
        per_view.append(np.stack((h@theta[t,:,0]/b.scale,h@(theta[t]@slope)),axis=0))
    per_view=np.stack(per_view,axis=0)
    channels=q.train.x.shape[1]
    merged=per_view.copy() if q.cz else per_view[::2]+per_view[1::2]
    if not q.cz:
        merged[:,:,q.gap:]-=per_view[1::2,:,:-q.gap]
    physical=(merged if model.exogenous_on else np.zeros_like(merged))
    scale=q.train.x.std(axis=0)
    names=SRU if channels==5 and not q.cz else tuple(f"XMEAS{i}" if i<=22 else f"XMV{i-22}"
                                                       for i in range(1,channels+1))
    if q.cz:
        names=("heater","crystal_lift","crucible_lift","crystal_rotation","crucible_rotation")
    rows=[]
    for i,n_ in enumerate(names):
        sl=slice(i,i+1) if q.cz else slice(2*i,2*i+2)
        local=physical[i,1]
        rows.append(dict(channel=n_,index=i,active_views=int(np.count_nonzero(model.joint.weights[sl])),
                         deployed_active=bool(model.exogenous_on and np.any(model.joint.weights[sl])),
                         linear_nnz=int(np.count_nonzero(theta[sl,:,0])),
                         curve_nnz=int(np.count_nonzero(theta[sl,:,1:])),
                         peak_lag=int(np.argmax(abs(local))) if local.any() else None,
                         l2_one_training_sd=float(scale[i]*np.linalg.norm(local)),
                         raw_linear=physical[i,0].tolist(),raw_local=local.tolist(),
                         per_sd_local=(scale[i]*local).tolist()))
    return rows


def isolation(q,model,full,rows,attribution):
    clean=model.predict(full)["prediction"]
    x=full.x.copy()
    k=q.first
    check=[]
    inactive=[r["index"] for r in attribution if not r["deployed_active"]]
    if inactive:
        varying=[j for j in inactive if np.ptp(q.train.x[:,j])>0.]
        j=(varying or inactive)[0]
        scale=float(q.train.x[:,j].std()) if varying else 0.
        amplitude=100.*scale if varying else 100.
        for label,change in (("bias",np.ones(512)),("oscillation",(-1.)**np.arange(512))):
            x[:]=full.x
            x[k:k+512,j]+=amplitude*change
            input_shift=float(np.max(abs(x[:,j]-full.x[:,j])))
            assert input_shift>0.
            touched=type(full).from_values(np.c_[full.lab,x],16) if q.cz else full.with_x(x)
            actual=model.predict(touched)["prediction"]
            delta=float(np.max(abs(actual-clean)))
            assert delta==0.
            check.append(dict(ingress="raw_sensor",sensor=attribution[j]["channel"],scenario=label,
                              std_multiplier=100. if varying else None,training_sd=scale,
                              max_input_shift=input_shift,max_shift=delta,window_points=512,
                              gate_admitted=bool(model.exogenous_on)))
    if not q.cz:
        # The raw-input experiment requires both views to be zero. If only a view is
        # inactive, perturb at the post-view input and identify that intervention.
        source=q.train.views()
        inactive_views=np.flatnonzero(~np.any(model.joint.theta,axis=(1,2)) & (np.ptp(source,axis=0)>0.))
        if len(inactive_views):
            j=int(inactive_views[0])
            views=full.views()
            a0=views[k:k+512,j].copy()
            views[k:k+512,j]+=100.*source[:,j].std()*((-1.)**np.arange(512))
            shift=float(np.max(abs(views[k:k+512,j]-a0)))
            assert shift>0.
            d=model.mapping.design(views)
            a=full.anchor()
            state=model.safe.dynamics.batch(a)
            pred,_=model.joint.mix(d.raw,d.trust,state,model.safe.predict(a,state),model.exogenous_on)
            error=float(np.max(abs(pred-clean)))
            assert error==0.
            check.append(dict(ingress="post_view",sensor=attribution[j//2]["channel"],
                              view="difference" if j%2 else "level",scenario="100sigma_oscillation",
                              max_input_shift=shift,max_shift=error,window_points=512,
                              gate_admitted=bool(model.exogenous_on)))
    return dict(raw_inactive_count=len(inactive),probes=check)


def trace(q,model,full,attribution):
    from benchmark_addendum import fault,feature_clip,load_pickle,prior_linear_path
    from laguerre_screen import digest
    weight=prior_linear_path(q)
    if q.tep:
        from tep_baselines import linear_predict
        expected=__import__("benchmark_addendum").read(q.a.existing/"cpu/FIT.json")
        assert digest(weight)==expected["weights"]["linear.pkl"]
        linear,_=load_pickle(weight)
        def predict(name,s,t):
            return s.anchor()[t]+linear_predict(linear[name],q.linear(s,t))
    else:
        expected=__import__("benchmark_addendum").read(q.a.paired/"LOCK.json")
        assert digest(weight)==expected["baselines"]
        linear,_=load_pickle(weight)
        def predict(name,s,t):
            x=q.linear(s,t)
            y=[]
            for arm in linear[name]:
                r,loc,sd=arm["model"]
                y.append(r.predict(arm["scale"].transform(x))*sd+loc+
                         (s.anchor()[t] if arm["recipe"]=="delta" else 0.))
            return np.mean(y,axis=0)
    channel=max(attribution,key=lambda r:r["l2_one_training_sd"])["index"]
    chosen=2*channel+int(model.joint.weights[2*channel+1]>model.joint.weights[2*channel])
    start=q.first
    ids=np.arange(start-64,start+512+256)
    clean_x=full.x[ids,channel]
    scale=q.train.x[:,channel].std()
    lo,hi=np.quantile(q.train.x,[.01,.99],axis=0)
    prepared=[]
    clean_rho=None
    for label,s in fault(q,full).items():
        d=model.mapping.design(s.views())
        a=s.anchor()
        state=model.safe.dynamics.batch(a)
        pred,trust=model.joint.mix(d.raw,d.trust,state,model.safe.predict(a,state))
        if label=="clean":
            clean_rho=trust[ids]
            continue
        e=trust[ids]-clean_rho
        post=e[ids>=start+512]
        countdown=np.flatnonzero(np.convolve(np.r_[abs(post)<=.01],np.ones(16,dtype=int),mode="valid")==16)
        prepared.append(dict(label=label,channel=attribution[channel]["channel"],
            view="difference" if chosen%2 else "level",recovery_to_clean_001_for_16=int(countdown[0]) if len(countdown) else None,
            max_recovery_above_clean=float(max(post.max(),0.)),
            series=dict(offset=(ids-start).tolist(),sensor_fault_sigma=((s.x[ids,channel]-clean_x)/scale).tolist(),
                        rho_view=d.trust[ids,chosen].tolist(),rho_active=trust[ids].tolist(),clean_rho=clean_rho.tolist(),
                        prism_error=(pred[ids]-full.truth[ids]).tolist(),
                        arx_error=(predict("ARX",s,ids)-full.truth[ids]).tolist(),
                        clipped_en_error=(predict("ElasticNet",feature_clip(q,s,lo,hi),ids)-full.truth[ids]).tolist())))
    return prepared


def main(a):
    sys.path[:0]=[str(a.source),str(a.baseline_code)]
    if a.tep_code:
        sys.path.insert(2,str(a.tep_code))
    from benchmark_addendum import Protocol,read,issue_digest
    from laguerre_screen import digest,save_json
    q=Protocol(a)
    q.verify_source()
    p=q.mode_dir/"MODEL.pkl"
    assert digest(p)==q.model_fit["model"]
    m=pickle.loads(p.read_bytes())
    full,rows=q.all_data()
    archive=read(a.reference)
    assert archive["commit"]==q.lock["commit"]
    assert archive["panels"]["clean"]["issue_sha256"]==issue_digest(rows)
    fits=ablation(q,m,full,rows,archive)
    h=impulse(q,m)
    isolated=isolation(q,m,full,rows,h)
    traces=None if q.cz else trace(q,m,full,h)
    a.out.mkdir(parents=True,exist_ok=True)
    save_json(a.out/"STUDY.json",dict(task=a.task,commit=q.lock["commit"],frozen_model=digest(p),
        source_lock=digest(a.lock),study=digest(Path(__file__)),training_issues=q.model_fit["training_issues"],
        score_issues=issue_digest(rows),score_archive=digest(a.reference),n=len(rows),g_exo=m.exogenous_on,
        alpha=q.model_fit["parameters"]["alpha"],curvature=q.model_fit["curvature"],ablation=fits,
        horizon=q.horizon,exogenous_delay=0 if q.cz else q.gap,endo_level_coef=q.model_fit["endo_level_coef"],
        attribution=h,isolation=isolated,traces=traces))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("task",choices=("TEP_D1","H2S","Rod1","Rod2"))
    for n in ("source","lock","model","data","out","reference","baseline_code"):
        p.add_argument(n,type=Path)
    for n in ("workbook","tep-code","existing","paired"):
        p.add_argument("--"+n,type=Path)
    p.add_argument("--tep-helper-digest")
    main(p.parse_args())
