"""Generate FP64 parameters, verify native parity, and time the matched readouts."""

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import platform
import shutil
import subprocess
import sys
import tempfile

import numpy as np


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def c_repr(value):
    if isinstance(value, (float, np.floating)):
        return float(value).hex()
    if isinstance(value, (int, np.integer, bool)):
        return str(int(value))
    return "{"+",".join(c_repr(x) for x in value)+"}"


def literal(name, dimension, value, ctype="double"):
    return f"static const {ctype} {name}{dimension}={c_repr(value)};\n"


def sizes(**values):
    return "".join(f"#define {k} {int(v)}\n" for k,v in values.items())


def prism_header(model, channels):
    from laguerre_dual_views import DualModel
    safe, head = model.safe, model.joint
    endo = safe.dynamics
    mapping = model.mapping
    dual = isinstance(model,DualModel)
    active = np.flatnonzero(head.weights) if model.exogenous_on else np.empty(0,dtype=int)
    dynamics = mapping.dynamics
    near, banks = dynamics.near, dynamics.banks
    n = near+sum(order for _,order in banks)
    conf=sizes(PRISM_MODEL=1,CHANNELS=channels,SPAN=endo.span,YTAPS=endo.near,
               ANCHOR_DELAY=endo.anchor_delay,DUAL=dual,ACTIVE=len(active),XNEAR=near,XTAPS=n,
               BANKS=len(banks),POLES=len(mapping.support.poles),RAW_HISTORY=model.delay if dual else 1)
    header=literal("anchor_center","",endo.center)+literal("safe_bias","",safe.location)
    header+=literal("safe_beta","[YTAPS+1]",safe.beta)
    header+=literal("joint_bias","",head.location if len(active) else 0.)
    header+=literal("joint_beta","[YTAPS+1]",head.beta_y if len(active) else np.zeros_like(safe.beta))
    if not len(active):
        return conf,header
    support=mapping.support
    header+=literal("moment_poles","[POLES]",support.poles)
    header+=literal("bank_order","[BANKS]",[order for _,order in banks],"int")
    order_max=max(order for _,order in banks)
    bank_a=np.zeros((len(banks),order_max,order_max))
    bank_b=np.zeros((len(banks),order_max))
    for t,(pole,order) in enumerate(banks):
        i,j=np.indices((order,order))
        lower=i>j
        a=np.eye(order)*pole
        a[lower]=(1-pole*pole)*(-pole)**(i[lower]-j[lower]-1)
        bank_a[t,:order,:order]=a
        bank_b[t,:order]=np.sqrt(1-pole*pole)*(-pole)**np.arange(order)
    header+=literal("bank_a",f"[BANKS][{order_max}][{order_max}]",bank_a)
    header+=literal("bank_b",f"[BANKS][{order_max}]",bank_b)
    view_parameters=[]
    for g in active:
        b=mapping.basis[g]
        knots=np.zeros(5)
        poly=np.zeros((4,4,2))
        pieces=0
        if b.rotation.size and np.any(head.theta[g,:,1:] != 0.):
            pieces=len(b.knots)-1
            knots[:len(b.knots)]=b.knots
            c=np.einsum("epk,kq->peq",b.spline.c,b.rotation)
            w0=b.project[0]@b.rotation
            w1=b.project[1]@b.rotation
            c[:,2]-=w1
            c[:,3]-=w0+b.knots[:-1,None]*w1
            poly[:pieces,:,:w0.size]=c
        mean_pad,sigma_pad=support.padding
        theta=np.pad(head.theta[g],((0,0),(0,1)))
        view_parameters.append([g//2 if dual else g, g%2 if dual else 0, pieces,
            b.center,b.scale,b.limit,head.weights[g],head.center[g],knots,poly,
            support.lower[g]-mean_pad,support.upper[g]+mean_pad,support.sigma[g]+sigma_pad,
            support.width*np.maximum(support.sigma[g],.25),theta])
    header+=literal("views","[ACTIVE]",view_parameters,"View")
    return conf,header


def dense_header(arm, channels, tep):
    if tep:
        from tep_baselines import LENGTH
        scale, reg, loc, sd=arm
        delta=True
    else:
        reg,loc,sd=arm["model"]
        scale=arm["scale"]
        delta=arm["recipe"]=="delta"
    coef=np.asarray(reg.coef_,dtype=float)*sd/scale.scale_
    bias=loc+float(reg.intercept_)*sd-scale.mean_@coef
    if tep:
        y=coef[:4].copy()
        y[0]-=y[1:].sum()
        lag=(coef[4+channels:].reshape(-1,channels)/4.).repeat(4,axis=0)
        assert len(lag)==LENGTH
        lag[0]+=coef[4:4+channels]
    else:
        length=len(coef)//(1+channels)
        assert len(coef)==length*(1+channels)
        y=coef[:length]
        lag=coef[length:].reshape(length,channels)
    conf=sizes(CHANNELS=channels,YHIST=len(y),XHIST=len(lag),BASE_DELTA=delta)
    return conf+literal("baseline_bias","",bias)+literal("baseline_y","[YHIST]",y)+literal("baseline_x","[XHIST][CHANNELS]",lag)


def baseline_expected(data, rows, arm, tep):
    if tep:
        from tep_baselines import design,linear_predict
        return data.anchor()[rows]+linear_predict(arm,design(data,rows)[1])
    reg, location, sd=arm["model"]
    l=len(reg.coef_)//(1+data.x.shape[1])
    j=rows[:,None]-np.arange(l)
    x=np.column_stack((data.anchor()[j],data.x[j].reshape((len(rows),-1))))
    return (reg.predict(arm["scale"].transform(x))*sd+location
            +(data.anchor()[rows] if arm["recipe"]=="delta" else 0.))


def ns_stats(times):
    return dict(n=len(times), median_ns=float(np.median(times)),
                p999_ns=int(np.quantile(times,.999,method="higher")), max_observed_ns=int(times.max()))


class Runner:
    def __init__(self, wsl, core):
        self.wsl=wsl
        self.core=core

    def path(self, path):
        p=path.resolve()
        return (f"/mnt/{p.drive[0].lower()}/"+p.as_posix()[3:]) if self.wsl else str(p)

    def call(self, *args, capture=False):
        prefix=["wsl","-d",self.wsl,"--"] if self.wsl else []
        cmd=prefix+[self.path(v) if isinstance(v,Path) else str(v) for v in args]
        return subprocess.run(cmd,check=True,capture_output=capture,text=capture).stdout

    def executable(self, folder, parameters):
        write_sources(folder,parameters)
        exe=folder/"stream"
        self.call("gcc","-O3","-std=c11","-mavx2","-mfma","-ffp-contract=off",
                  folder/"stream.c","-lm","-o",exe)
        return exe

    def replay(self, exe, data, first, repeats, timer=False):
        in_bin=exe.parent/"input.bin"
        out_bin=exe.parent/"output.bin"
        matrix=np.column_stack((data.anchor(),data.x)).astype("<f8")
        matrix.tofile(in_bin)
        self.call("taskset","-c",self.core,exe,in_bin,out_bin,len(data),first,repeats,int(timer))
        with out_bin.open("rb") as stream:
            state_bytes=int(np.fromfile(stream,dtype="<u8",count=1)[0])
            y=np.fromfile(stream,dtype="<f8",count=3*len(data)).reshape(-1,3)
            times=np.fromfile(stream,dtype="<u8")
        assert len(times)==(len(data)-first)*repeats
        return y,dict(state_bytes=state_bytes,**ns_stats(times))

    def platform(self):
        return dict(core=self.core,wsl=self.wsl,cpu=self.call("lscpu",capture=True).split("Model name:")[1].splitlines()[0].strip(),
                    kernel=self.call("uname","-r",capture=True).strip(),
                    compiler=self.call("gcc","--version",capture=True).splitlines()[0],
                    flags="-O3 -std=c11 -mavx2 -mfma -ffp-contract=off",
                    clock="CLOCK_MONOTONIC_RAW",sample="full step including state update, output and timer overhead")


def kkt(problem,beta,alpha,mix,endo=0):
    from laguerre_anchored import ENDOGENOUS_RIDGE
    from laguerre_process import kkt_error
    b=beta*problem.scale/problem.yscale
    g=problem.g.copy()
    diag=np.diag_indices_from(g)
    g[diag]+=np.r_[np.full(endo,ENDOGENOUS_RIDGE),np.full(len(g)-endo,alpha*(1.-mix))]
    grad=g@b-problem.h
    return max(float(np.max(abs(grad[:endo]),initial=0.)),
               kkt_error(g[endo:,endo:],problem.h[endo:]-g[endo:,:endo]@b[:endo],alpha*mix,b[endo:]))


def curve_kkt(f, model, alpha):
    from laguerre_anchored import Standardized
    from laguerre_process import fixed_projection
    raw=f.design.raw[f.rows]
    active=np.flatnonzero(np.any(model.joint.theta[...,0]!=0.,axis=1))
    groups,coefs,blocks=[],[],[]
    for g in active:
        p=fixed_projection(f.near,raw.shape[2])
        if g%2==0:
            p[min(f.near,f.delay):f.near]=0.
            norms=np.linalg.norm(p,axis=0)
            p=p[:,norms>0]/norms[norms>0]
        x=np.einsum("ntp,tk->nkp",raw[:,g,:,1:],p,optimize=True).reshape(len(f.rows),-1)
        groups.append(x)
        b=np.linalg.lstsq(p,model.joint.theta[g,:,1:],rcond=None)[0]
        coefs.extend(b.ravel())
        px=f.design.curvature[g,1:,1:]
        blocks.append(np.kron(np.eye(p.shape[1]),px+.01*np.eye(len(px))))
    base=np.einsum("ngt,gt->n",raw[...,0],model.joint.theta[...,0],optimize=True)
    y=f.target[f.rows]-f.safe_hat[f.rows]-base-f.state[f.rows]@model.joint.beta_y
    proof=Standardized(np.column_stack(groups),y)
    beta=np.array(coefs)*proof.scale/proof.yscale
    gradient=proof.g@beta-proof.h
    offset=0
    for penalty in blocks:
        sl=slice(offset,offset+len(penalty))
        scale=proof.scale[sl]
        gradient[sl]+=alpha*(penalty/np.outer(scale,scale))@beta[sl]
        offset+=len(penalty)
    return float(max(abs(gradient),default=0.))


def audit_fit(train,model,frozen,task,prepare,config):
    from laguerre_dual_views import ENDO_MIX,mature_problem
    s=frozen.get("safe",frozen.get("endo"))
    prm=frozen["parameters"]
    f=prepare(train,len(train),prm["near"],s,config)
    assert f.rows[0]==frozen["training_issues"][0] and f.rows[-1]+1==frozen["training_issues"][1]
    _,a,state,p=mature_problem(train,f.rows,s["near"])
    np.testing.assert_allclose(f.safe.beta,model.safe.beta,rtol=0.,atol=1e-10)
    safe=kkt(p,model.safe.beta,s["alpha"],ENDO_MIX) if s["admitted"] else None
    safe_bias=(float(abs(np.mean(model.safe.predict(a[f.rows],state[f.rows])-train.truth[f.rows])))
               if s["admitted"] else None)
    theta=model.joint.theta
    joint=kkt(f.proof.standardized,np.r_[model.joint.beta_y,theta[f.mask,0]],prm["alpha"],prm["mix"],f.state.shape[1])
    curve=frozen.get("curvature",frozen.get("curve"))
    residual=curve_kkt(f,model,curve) if curve is not None else None
    if max(safe or 0.,joint,residual or 0.)>2e-7:
        raise RuntimeError(f"{task}: KKT exceeds the registered optimization tolerance")
    stored=frozen.get("fit",frozen.get("coefficients"))["kkt"]
    return dict(safe_admitted=s["admitted"],safe_slope_inf=safe,safe_intercept_abs=safe_bias,joint_linear_inf=joint,
                joint_linear_archived=stored,curvature_slope_inf=residual)


def write_sources(folder,parameters):
    d=Path(__file__).resolve().parent
    for name in ("stream.c","stream.h"):
        shutil.copy2(d/name,folder/name)
    conf,decl=parameters if isinstance(parameters,tuple) else (parameters,"")
    (folder/"model.h").write_text(conf,encoding="ascii",newline="\n")
    (folder/"parameters.h").write_text(decl,encoding="ascii",newline="\n")


def evaluate(out,runner,models,data,fit_prepare,config,baseline,first,truth_rows,task,repeats,export):
    from laguerre_blind_cz import Direct
    from laguerre_screen import pulse
    from laguerre_unified import apply_bias
    results={}
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        scratch=Path(tmp)
        for name,(model,fit,full,prefix) in models.items():
            folder=scratch/name
            folder.mkdir()
            parameters=prism_header(model,full.x.shape[1])
            executable=runner.executable(folder,parameters)
            clean=model.predict(full)
            peak=first
            audit={}
            cases={"clean":full}
            if isinstance(full,Direct):
                if task=="Rod2" and name=="H16":
                    additions=np.random.default_rng(20261125).choice((-1,1),256)
                    scale=4.*prefix.x.std(axis=0)
                    for label,amount in (("oscillation",np.c_[additions,-additions].ravel()),("bias",np.ones(512))):
                        change=np.column_stack((full.lab,full.x))
                        change[peak:peak+512,1:]+=scale*amount[:,None]
                        cases[label]=Direct.from_values(change,full.delay)
            else:
                scale=prefix.x.std(axis=0)
                cases.update(oscillation=pulse(full,peak,scale),bias=apply_bias(full,peak,scale))
            for label,trial in cases.items():
                observed,timing=runner.replay(executable,trial,first,repeats if label=="clean" else 1)
                python=clean if label=="clean" else model.predict(trial)
                expected=np.column_stack([python[k] for k in ("prediction","endo","trust")])
                err=np.max(abs(expected-observed),axis=0)
                if max(err)>=1e-12:
                    raise RuntimeError(f"{task}/{name}/{label}: native parity failed {err}")
                zero=observed[:,2]==0.
                fallback=float(np.max(abs(observed[zero,0]-observed[zero,1]),initial=0.))
                assert fallback==0.
                audit[label]=dict(prediction=float(err[0]),endogenous=float(err[1]),trust=float(err[2]),
                                  max_zero_trust_fallback=fallback)
                if label=="clean":
                    clean_timing=timing
            if not results:
                _,timer=runner.replay(executable,full,first,repeats,True)
            result=dict(parity=audit,latency=clean_timing,kkt=audit_fit(prefix,model,fit,task,fit_prepare,config),
                        native_parameters=digest(folder/"model.h"),native_readout=digest(folder/"parameters.h"),
                        active_groups=int(np.count_nonzero(model.joint.weights)) if model.exogenous_on else 0)
            results[name]=result
            if export:
                destination=export/name
                destination.mkdir(parents=True,exist_ok=True)
                write_sources(destination,parameters)
        timings={}
        if baseline:
            is_tep=task.startswith("TEP")
            for name in ("ARX","ElasticNet"):
                raw=baseline[name]
                if not is_tep:
                    raw=raw[0]
                folder=scratch/name
                folder.mkdir()
                exe=runner.executable(folder,dense_header(raw,data.x.shape[1],is_tep))
                y,latency=runner.replay(exe,data,first,repeats)
                error=float(np.max(abs(y[truth_rows,0]-baseline_expected(data,truth_rows,raw,is_tep))))
                if error>=2e-10:
                    raise RuntimeError(f"{task}/{name} native baseline mismatch: {error}")
                timings[name]=dict(latency=latency,native_max_abs=error)
    return dict(cases=results,timer_only=timer,baselines=timings)


def run(a):
    sys.path.insert(0,str(a.source))
    from laguerre_blind_cz import HORIZONS,PROTOCOL,Direct,full_rod,prepare as prepare_cz,train_values
    from laguerre_blind_so2 import rows_h2s_workbook,verify_phase_one
    from laguerre_control import load_prefix
    from laguerre_dual_views import prepare
    from laguerre_screen import CONFIGS,Series,append,read_tep,receipt,save_json
    from laguerre_unified import full_h2s
    f=verify_phase_one(a.lock)
    commit="7bf849fa0c2453ea58412cc631c3cddf9b6d4e3c"
    assert f["commit"]==commit and all(digest(a.source/k)==v for k,v in f["code"].items())
    models={}
    copies={}
    def load(root):
        state=read(root/"FROZEN.json")
        path=root/"MODEL.pkl"
        assert digest(path)==state["model"]
        copies[root.name]=digest(root/"FROZEN.json")
        return pickle.loads(path.read_bytes()),state
    if a.task in PROTOCOL:
        a.registered=read(a.model/"REGISTRATION.json")
        sealed=read(a.model/"TEST_LOCK.json")
        assert a.registered["phase_one"]==digest(a.lock)
        assert all(digest(a.model/f"H{h}"/"FROZEN.json")==v for h,v in sealed["horizons"].items())
        assert digest(a.model/"REGISTRATION.json")==sealed["registration"]
        base_manifest=a.model/"Baselines/FROZEN.json"
        assert digest(base_manifest)==sealed["baseline"]
        if a.baselines:
            assert digest(a.baselines)==read(base_manifest)["linear"]
        train=train_values(a.model)
        data=full_rod(a.workbook,a.task,train,a.data)
        _,first,end,(lo,hi)=PROTOCOL[a.task]
        idx=np.arange(first,end-16)
        idx=idx[(idx+16<lo)|(idx+16>=hi)]
        config=CONFIGS["sru"][a.registered["envelope_index"]]
        for h in HORIZONS:
            name=f"H{h}"
            models[name]=(*load(a.model/name),Direct.from_values(data,h),Direct.from_values(train,h))
        main=Direct.from_values(data,16)
        fit_prepare=prepare_cz
    else:
        model,fit=load(a.model)
        official=read(a.model/"RESULTS.json")
        if a.task=="SO2":
            expected=read(a.model/"TEST_LOCK.json")
            assert digest(a.model/"FROZEN.json")==expected["files"]["FROZEN.json"]
            assert official["test_lock"]==digest(a.model/"TEST_LOCK.json")
            if a.baselines:
                assert digest(a.baselines)==expected["files"]["classical/SRU_H2S/linear/MODELS.pkl"]
            reg=read(a.model/"REGISTRATION.json")
            assert reg["phase_one_sha"]==digest(a.lock)
            assert digest(a.model/"TRAIN.npz")==reg["train_sha"]
            a0=rows_h2s_workbook(a.workbook,10081)
            with np.load(a.model/"TRAIN.npz",allow_pickle=False) as p:
                np.testing.assert_array_equal(a0[:6048],p["values"])
            main=Series(a0[:,1:],a0[:,0],a0[:,0],8)
            prefix=main.through(6048)
            first=8064
            config=CONFIGS["sru"][reg["support_index"]]
        else:
            assert official["frozen"]==digest(a.model/"FROZEN.json")
            assert all(digest(a.source/k)==v for k,v in fit["sources"].items())
            parent=a.authority/"FROZEN.json"
            assert fit["authority"]==digest(parent)
            task,d=("tep",int(a.task[-1])) if a.task.startswith("TEP") else ("sru",8)
            assert fit["data"]==receipt(a.data,task,("train","development"))
            if task=="tep":
                assert official["source"]==digest(a.data/"tail.npz")
                if a.baselines:
                    assert digest(a.baselines)==read(a.baselines.parent/"FIT.json")["weights"]["linear.pkl"]
            elif a.baselines:
                assert digest(a.baselines)==official["pressure"]["baseline_receipts"]["linear_sha256"]
            prefix,dev=load_prefix(task,a.data,d)
            main=(append(append(prefix,dev),read_tep(a.data,"tail",d)) if task=="tep"
                  else full_h2s(prefix,dev,a.workbook))
            first=len(prefix)+len(dev)
            config=CONFIGS[task][read(parent)["selected"]]
        idx=np.arange(first,len(main))
        models[a.task]=(model,fit,main,prefix)
        fit_prepare=prepare
    runner=Runner(a.wsl,a.core)
    weights=None
    if a.baselines:
        weights=pickle.loads(a.baselines.read_bytes())
    out=evaluate(a.output,runner,models,main,fit_prepare,config,weights,first,idx,a.task,a.repeats,a.export)
    save_json(a.output,dict(commit=commit,task=a.task,source_lock=digest(a.lock),model_receipts=copies,
              platform=runner.platform(),kernel=digest(Path(__file__).with_name("stream.c")),
              native_header=digest(Path(__file__).with_name("stream.h")),audit_script=digest(Path(__file__)),
              baseline_weights=digest(a.baselines) if a.baselines else None,**out))
    print(a.task,json.dumps(out),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for arg in ("source","lock","model","data","output"):
        p.add_argument(arg,type=Path)
    p.add_argument("task",choices=("TEP_D1","TEP_D2","H2S","SO2","Rod1","Rod2"))
    for arg in ("authority","workbook","baselines","export"):
        p.add_argument("--"+arg,type=Path)
    p.add_argument("--core",type=int,required=True)
    p.add_argument("--repeats",type=int,default=60)
    p.add_argument("--wsl")
    run(p.parse_args())
