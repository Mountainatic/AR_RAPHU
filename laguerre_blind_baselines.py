"""Train and replay the registered SRU baseline recipes on an unseen target."""

import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import importlib
import json
from multiprocessing import get_context
from pathlib import Path
import pickle
from statistics import median
import sys

import numpy as np

from laguerre_screen import digest, save_json


def legacy(archive):
    sys.path.insert(0,str(archive))
    return tuple(importlib.import_module(n) for n in ("tf_sweep","e1","linear"))


def read_train(out):
    record=json.loads((out/"REGISTRATION.json").read_text())
    if digest(out/"TRAIN.npz")!=record["train_sha"]:
        raise ValueError("Unbound baseline training data")
    with np.load(out/"TRAIN.npz",allow_pickle=False) as p:
        return p["values"]


def cpu(archive,out):
    tf,e,linear=legacy(archive)
    q=tf.Q
    a=read_train(out)
    layer=out/"classical"
    layer.mkdir(exist_ok=False)
    linear.ROOT=layer
    linear.data=lambda _:a
    save_json(layer/"REGISTRATION.json",dict(source=digest(Path(linear.__file__)),
      protocol_parent=digest(out/"REGISTRATION.json"),ridge=linear.RIDGE,elastic_net=linear.ENET))
    linear.train(q)
    d=out/"LightGBM"
    d.mkdir(exist_ok=False)
    folds=[(tr,ev,linear.design(q,a,tr,"ARX"),linear.design(q,a,ev,"ARX"))
           for tr,ev in tf.c.x.folds(q)]
    def loss(h):
        terms=[]
        for tr,ev,x,v in folds:
            y,_=q.target(a,tr,h[0]); truth,_=q.target(a,ev,h[0])
            fitted=e.head(h,20261125).fit(x,y)
            err=truth-fitted.booster_.predict(v,num_threads=1)
            terms.append((float(err@err),len(err)))
        return dict(config=h,mse=sum(s for s,_ in terms)/sum(n for _,n in terms))
    with ThreadPoolExecutor(max_workers=3) as pool:
        grid=list(pool.map(loss,e.GRID))
    pick=min(grid,key=lambda g:g["mse"])["config"]
    t=q.train_origins()
    x=linear.design(q,a,t,"ARX")
    y,_=q.target(a,t,pick[0])
    files={}
    for seed in e.SEEDS:
        m=e.head(pick,seed).fit(x,y)
        dest=d/f"seed_{seed}.txt"
        m.booster_.save_model(str(dest))
        files[dest.name]=digest(dest)
    save_json(d/"FROZEN.json",dict(config=pick,grid=grid,files=files,source=digest(Path(e.__file__))))
    print("CPU",pick,flush=True)


def transformer_settings(archive,out,initialize):
    tf,_,_=legacy(archive)
    tf.ARCH=archive.parent.parent/"PRISM_SRU_MATCHED_WINDOWS_REPEAT_20261025/baseline_models.py"
    tf.ROOT=out/"Transformer"
    if initialize:
        train=read_train(out)
        tf.c.data=lambda _:train
    tf.init()
    return tf


def tf_worker(archive,out):
    transformer_settings(archive,out,True)


def transformer(archive,out):
    tf,_,_=legacy(archive)
    old=tf.ROOT/"REGISTRATION.json"
    base=out/"Transformer"
    base.mkdir(exist_ok=False)
    (base/"selection").mkdir()
    (base/"REGISTRATION.json").write_bytes(old.read_bytes())
    tf.ROOT=base
    with ProcessPoolExecutor(max_workers=5,mp_context=get_context("spawn"),
                             initializer=tf_worker,initargs=(archive,out)) as pool:
        screen=tf.select(pool,tf.GRID,"screen",tf.SCREEN)
        best=tf.select(pool,[tuple(j["config"]) for j in screen[:3]],"refine",tf.REFINE)[0]
        n=round(median(a["epoch"] for a in best["folds"]))
        files=dict(pool.map(tf.refit,[(tuple(best["config"]),n,seed) for seed in tf.SEEDS]))
    save_json(base/"FROZEN.json",dict(files=files,selected=best,epochs=n,source=digest(Path(tf.__file__))))
    print("TRANSFORMER",best["config"],n,flush=True)


def holdup(out):
    paths=["REGISTRATION.json","TRAIN.npz","FROZEN.json","MODEL.pkl",
           "classical/REGISTRATION.json","classical/SRU_H2S/linear/FIT.json",
           "classical/SRU_H2S/linear/MODELS.pkl","LightGBM/FROZEN.json","Transformer/FROZEN.json",
           "Transformer/REGISTRATION.json"]
    files={p:digest(out/p) for p in paths}
    for kind in ("LightGBM","Transformer"):
        p=out/kind
        freeze=json.loads((p/"FROZEN.json").read_text())
        for name,sha in freeze["files"].items():
            if digest(p/name)!=sha:
                raise ValueError("Baseline model changed")
            files[f"{kind}/{name}"]=sha
    save_json(out/"TEST_LOCK.json",dict(files=files,baseline_adapter=digest(Path(__file__))))


def predictors(archive,out):
    tf=transformer_settings(archive,out,False)
    _,e,linear=legacy(archive)
    q=tf.Q
    stored=pickle.loads((out/"classical/SRU_H2S/linear/MODELS.pkl").read_bytes())
    lgb=json.loads((out/"LightGBM/FROZEN.json").read_text())
    boosters=[e.lgb.Booster(model_file=str(out/"LightGBM"/name)) for name in lgb["files"]]
    neural=[]
    for name in json.loads((tf.ROOT/"FROZEN.json").read_text())["files"]:
        state=tf.torch.load(tf.ROOT/name,map_location="cpu",weights_only=False)
        net=tf.model_class()("Transformer",6,state["config"][0],q.L).eval()
        net.load_state_dict(state["weight"])
        neural.append((net,state))
    def run(a,t):
        anchor=a[t-q.D,0]
        out={"Persistence":anchor[None,:]}
        for kind,arms in stored.items():
            x=linear.design(q,a,t,kind)
            out[kind]=np.array([m.predict(j["scale"].transform(x))*scale+center+
                       (anchor if j["recipe"]=="delta" else 0.)
                       for j in arms for m,center,scale in [j["model"]]])
        x=linear.design(q,a,t,"ARX")
        out["LightGBM"]=np.array([m.predict(x,num_threads=1)+(anchor if lgb["config"][0]=="delta" else 0.)
                                for m in boosters])
        x=tf.features(a,t)
        prediction=[]
        for net,state in neural:
            mean,std,center,scale=state["stats"]
            v=tf.torch.from_numpy((x-mean)/std)
            with tf.torch.inference_mode():
                z=tf.torch.cat([net(f) for f in v.split(64)]).numpy().astype(float)
            prediction.append(z*scale+center+(anchor if state["config"][2]=="delta" else 0.))
        out["Transformer"]=np.array(prediction)
        return out
    return run


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=("cpu","transformer","lock"))
    p.add_argument("archive",type=Path)
    p.add_argument("out",type=Path)
    a=p.parse_args()
    if a.action=="lock":
        holdup(a.out)
    else:
        {"cpu":cpu,"transformer":transformer}[a.action](a.archive,a.out)

