"""Count archived parameters and time one fixed-recipe fit per public baseline."""

import argparse
import json
from pathlib import Path
import pickle
import sys
from time import perf_counter

import numpy as np
from sklearn.base import clone
from sklearn.preprocessing import StandardScaler


def count(*arrays):
    return sum(int(np.count_nonzero(x)) for x in arrays)


def torch_count(torch, file, key):
    arm=torch.load(file,map_location="cpu",weights_only=False)[key]
    return sum(int(torch.count_nonzero(v)) for name,v in arm.items()
               if name.rsplit(".",1)[-1] not in ("pe","position"))


def tree_count(booster):
    def descend(z):
        if "leaf_value" in z:
            return int(z["leaf_value"]!=0.)
        return int(z["threshold"]!=0.)+descend(z["left_child"])+descend(z["right_child"])
    return sum(descend(item["tree_structure"]) for item in booster.dump_model()["tree_info"])


def run(a):
    sys.path[:0]=[str(a.source),str(a.baseline_code),str(a.tep_code)]
    from benchmark_addendum import Protocol,load_pickle,pls_fit,prior_linear_path,read,fit_gru
    from laguerre_dual_views import prepare,readout
    from laguerre_pressure import archived
    from laguerre_screen import digest,save_json
    from lightgbm import Booster
    import torch
    torch.set_num_threads(1)
    q=Protocol(a)
    q.verify_source()
    report_q=read(a.final/"TEST_LOCK.json")
    native=read(a.tep_code/f"{a.task}_NATIVE.json")
    assert native["source_lock"]==digest(a.lock)
    c=q.train.x.shape[1]
    native_state=native["cases"][a.task]["latency"]["state_bytes"]
    ring=native["baselines"]["ARX"]["latency"]["state_bytes"]
    model_file=q.mode_dir/"MODEL.pkl"
    assert digest(model_file)==q.model_fit["model"]==report_q["model"]
    m=pickle.loads(model_file.read_bytes())
    assert m.exogenous_on
    rows=q.rows
    lagonly=4 if q.tep else q.history
    cpu="Xeon CPU, one affinity core"
    gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else cpu
    device="cuda" if torch.cuda.is_available() else "cpu"
    if device=="cuda":
        torch.cuda.init()
    output={}

    def timed(method,action,state,parameters,clock=cpu,extra=0.,aux_static=0):
        if clock!=cpu:
            torch.cuda.synchronize()
        start=perf_counter()
        if action is not None:
            action()
        if clock!=cpu:
            torch.cuda.synchronize()
        seconds=perf_counter()-start if action is not None else 0.
        output[method]=dict(online_state_bytes=state,auxiliary_static_bytes=aux_static,
                            per_seed_nonzero=parameters,nonzero_total=sum(parameters),
                            refit_seconds=float(seconds+extra),incremental_seconds=float(seconds),
                            timing_device=clock)
        print(a.task,method,output[method],flush=True)

    def prism_fit():
        f=prepare(q.train,len(q.train),m.mapping.dynamics.near,q.model_fit["safe"],m.mapping.support)
        p=q.model_fit["parameters"]
        frozen_result=readout(f,p["alpha"],p["mix"],q.model_fit["curvature"])[0]
        np.testing.assert_allclose(frozen_result.theta,m.joint.theta,atol=1e-7,rtol=0.)

    nonzero=count(m.safe.beta,m.safe.location,m.joint.theta,m.joint.beta_y,m.joint.location)
    timed("PRISM",prism_fit,native_state,[nonzero])
    timed("Persistence",None,8,[0])

    path=prior_linear_path(q)
    if q.tep:
        from tep_baselines import design,fit_tf,lgbm,tokens
        assert digest(path)==read(path.parent/"FIT.json")["weights"]["linear.pkl"]
        reg=read(a.existing/"cpu/FIT.json")["fit"]
        old_rows=rows
        linear=lambda method,r: design(q.train,r)[0 if method=="AR" else 1]
        transformer_cfg=read(a.existing/"transformer/FIT.json")
        tree_files=[a.existing/"cpu"/f"LGB_{i}.txt" for i in q.seed]
        neural_files=[a.existing/"transformer"/f"TF_{i}.pt" for i in q.seed]
        for f in (*tree_files,*neural_files):
            assert digest(f)==read(f.parent/"FIT.json")["weights"][f.name]
        def gb():
            d=linear("ARX",old_rows)
            y=(q.train.truth-q.train.anchor())[old_rows]
            lgbm(reg["LightGBM"],0).fit(d,y)
        def tf_fit():
            x=tokens(q.train,rows)
            y=(q.train.truth-q.train.anchor())[rows]
            fit_tf(x,y,np.arange(len(y)),transformer_cfg["fit"]["spec"],0,
                   transformer_cfg["fit"]["epochs"],device)
        tf_key="state"
        tf_device=gpu
    else:
        assert digest(path)==read(a.paired/"LOCK.json")["baselines"]
        tf,e1,lin=archived(a.archive)
        values=np.c_[q.train.lab,q.train.x]
        old_rows=tf.Q.train_origins()
        linear=lambda method,r:lin.design(tf.Q,values,r,method)
        tree_files=[a.existing/"LightGBM"/f"seed_{i}.txt" for i in q.seed]
        neural_files=[a.existing/"Transformer"/f"seed_{i}.pt" for i in q.seed]
        for f in (*tree_files,*neural_files):
            assert digest(f)==read(f.with_suffix(".json"))["sha256"]
        def gb():
            name,structure,rounds,penalty=read(a.existing/"LightGBM/seed_0.json")["config"]
            d=linear("ARX",old_rows)
            y=tf.Q.target(values,old_rows,name)[0]
            e1.head((name,tuple(structure),rounds,penalty),0).fit(d,y)
        tf.ARCH=a.sru_arch
        tf.c.data=lambda _:values
        tf.init()
        torch.set_num_threads(1)
        tf_state=torch.load(neural_files[0],map_location="cpu",weights_only=False)
        def tf_fit():
            tf.fit(tuple(tf_state["config"]),0,None,tf_state["epochs"])
        tf_key="weight"
        tf_device=cpu

    models,_=load_pickle(path)
    for method in ("AR","ARX","ElasticNet"):
        if q.tep:
            scale,estimator,mu,sd=models[method]
            saved=[(estimator,mu,sd)]
            recipe="delta"
        else:
            arm=models[method][0]
            estimator,mu,sd=arm["model"]
            saved=[arm["model"] for arm in models[method]]
            recipe=arm["recipe"]
        def fit_lin():
            x=linear(method,old_rows)
            y=q.train.truth[old_rows]-(q.train.anchor()[old_rows] if recipe=="delta" else 0.)
            clone(estimator).fit(StandardScaler().fit_transform(x),(y-mu)/sd)
        params=[count(r.coef_,r.intercept_) for r,_,_ in saved]
        timed(method,fit_lin,lagonly*8+8 if method=="AR" else ring,params)
    timed("ElasticNet_clipped",lambda:np.quantile(q.train.x,[.01,.99],axis=0),ring,
          output["ElasticNet"]["per_seed_nonzero"],extra=output["ElasticNet"]["refit_seconds"],aux_static=16*c)
    timed("LightGBM",gb,ring,[tree_count(Booster(model_file=str(f))) for f in tree_files])
    timed("Transformer",tf_fit,ring,[torch_count(torch,f,tf_key) for f in neural_files],tf_device)
    dpls=a.final/"dpls"
    d=read(dpls/"FIT.json")
    assert digest(dpls/"FIT.json")==report_q["fits"]["dpls"]
    assert digest(dpls/"DPLS.pkl")==d["weights"]["DPLS.pkl"]
    pls=pickle.loads((dpls/"DPLS.pkl").read_bytes())[1]
    def fit_pls():
        y=(q.train.truth-q.train.anchor())[rows]
        pls_fit(q.linear(q.train,rows),y,d["fit"]["components"])
    timed("DPLS",fit_pls,ring,[count(pls.coef_,pls.intercept_)])
    gpath=a.final/"gru"
    g=read(gpath/"FIT.json")
    assert digest(gpath/"FIT.json")==report_q["fits"]["gru"]
    assert all(digest(gpath/f)==h for f,h in g["weights"].items())
    def gru_fit():
        t=(q.train.truth-q.train.anchor())[rows]
        x=q.sequence(q.train,rows)
        fit_gru(x,t,np.arange(len(t)),g["fit"]["width"],g["fit"]["epochs"],0,device)
    timed("GRU",gru_fit,ring,[torch_count(torch,gpath/f,"state") for f in g["weights"]],gpu)

    a.out.mkdir(parents=True,exist_ok=True)
    save_json(a.out/f"{a.task}_INVENTORY.json",dict(task=a.task,commit=q.lock["commit"],
             source_lock=digest(a.lock),archive_model=digest(model_file),pressure_native=digest(a.tep_code/f"{a.task}_NATIVE.json"),
             timing="One seed-zero selected-recipe fit including feature preparation; excludes selection, historical HAC nesting and artifact I/O",
             state="PRISM and dense ARX/Elastic Net use measured C struct bytes; other methods show causal history storage, excluding temporary activations and immutable coefficients",
             count="Readout coefficients including bias for PRISM and linear methods; effective DPLS slopes; trainable tensors for neural models and thresholds/leaf values for LightGBM",
             rows=dict(prism=[int(rows[0]),int(rows[-1])+1],archival=[int(old_rows[0]),int(old_rows[-1])+1]),
             code=digest(Path(__file__)),methods=output))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("task",choices=("TEP_D1","H2S"))
    for n in ("source","lock","model","data","out","baseline_code","existing","final","tep_code"):
        p.add_argument(n,type=Path)
    for n in ("archive","paired","sru-arch"):
        p.add_argument("--"+n,type=Path)
    p.add_argument("--tep-helper-digest",required=True)
    run(p.parse_args())
