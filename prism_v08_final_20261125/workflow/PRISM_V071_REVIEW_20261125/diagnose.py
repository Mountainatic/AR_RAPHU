"""One-factor SRU rollbacks and training-fold route evidence for the frozen v0.7."""

import argparse
from contextlib import contextmanager
import itertools
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
V07 = ROOT.parent/"PRISM_V07_20261125"
if not V07.exists():
    V07 = ROOT.parent/"prism_v07_20261125"
sys.path.insert(0,str(V07))
import v07 as v
from v07 import CASES, D, EMBARGO, Engine, g, h, inputs, np, poisoned, read, rows, s, score, sha, write


ENTITY = "SRU_H2S"
FIXED_WINDOWS = ("all4","all8")
ARMS = (*FIXED_WINDOWS,"shared_window","wide_upper_only","prior_K_only")
ALPHA_EXTRA = (.1,.3,1.)
CONFIG = ("registration","fit","freeze","evaluate")


def registration():
    path = ROOT/"REGISTRATION.json"
    config = dict(source_sha256=sha(Path(__file__)), reference_sha256=sha(V07/"ALL_FROZEN.json"),
        SRU_arms=ARMS, floor_controls=[1e-5,1e-4],
        floor_method="remove candidates only; if every archived minimizer survives, the rerun is an exact identity",
        shared_window_choice="minimize sum of view-level pooled SSE from identical early K fold rows",
        C_method="compare implementation and candidate list from prior sprint and v0.7",
        extra_small_K_control="widen the high end because 9/10 SRU winners reached the registered upper bound",
        upper_grid_append=ALPHA_EXTRA, reserved_tail_accessed=False)
    config = h.json.loads(h.json.dumps(config))
    if path.exists():
        assert read(path)==config
    else:
        write(path,config)
    return config


@contextmanager
def k_search(windows=None, extra=()):
    original_grid,original_windows = v.K_GRID,v.K_WINDOWS
    try:
        if windows is not None:
            v.K_WINDOWS = {n:original_windows[n] for n in windows}
        more = tuple((family,width,tuple(alpha*value for value in ratio))
                     for family,width in v.K_FAMILIES for alpha in extra
                     for ratio in (((1.,1.,0.),) if width==1 else ((1.,1.,1.),(1.,.1,10.))))
        v.K_GRID = original_grid+more
        yield
    finally:
        v.K_GRID,v.K_WINDOWS = original_grid,original_windows


def fit():
    registration()
    original = v.frozen(ENTITY)
    k_ridge = [x["spec"][2][0] for x in original["K_audit"]]
    floor = {str(x):dict(identity=all(a>=x for a in k_ridge),minimum_selected=min(k_ridge)) for x in (1e-5,1e-4)}
    assert all(item["identity"] for item in floor.values())
    # v0.7 inherits both the candidate selector and the estimator fit from the preceding sprint.
    assert Engine.c_fit is s.SharedFit.c_fit and Engine.c_select is s.SharedFit.c_select
    c_control = dict(identity=True,grid=s.C_GRID,fit_source=sha(v.SPRINT/"sprint.py"),
                     alpha_note="sklearn ElasticNet and identical target-standardized penalty in both versions")
    geo,pipe,data,_,stages = inputs(ENTITY)
    k = stages["K"]
    lo,hi = geo["calibration"]
    models,k_audit = {},{}
    for label,window in (("all4",4),("all8",8),("wide_upper_only",None)):
        with k_search(None if window is None else (window,), ALPHA_EXTRA if window is None else ()):
            models[label],k_audit[label] = v.fit_k(geo,pipe,data,k)
        print("K",label,flush=True)
    losses = {key:sum(x["fold_pooled_SSE"] for x in k_audit[key]) for key in FIXED_WINDOWS}
    selected_window = min(losses,key=losses.get)
    models["prior_K_only"] = read(v.SPRINT/ENTITY/"MODEL.json")["K"]["near8_tuned"]
    assert sha(v.SPRINT/ENTITY/"MODEL.json")==read(v.SPRINT/"ALL_FROZEN.json")["models"][ENTITY]
    training = {}
    for label,kfit in models.items():
        ctx = g.context(ENTITY,pipe,data,k,kfit,hi)
        engine = Engine(pipe,ctx,lo)
        audits = []
        for a,b in s.blocks(lo,hi,.70,4):
            model = engine.train(a-EMBARGO)
            t = rows(a,b)
            pred = engine.predict(t,model)
            audits.append(dict(rows=[a,b],gates=model["gates"],
                               metrics={step:score(data[t,0],value) for step,value in pred.items()}))
        training[label] = dict(K=kfit, model=engine.train(hi), outer=audits)
        print("FIT",label,flush=True)
    write(ROOT/"MODEL.json",dict(arms=training, shared=selected_window, shared_loss=losses,
         K_audit=k_audit,floor_controls=floor,C_control=c_control))
    write(ROOT/"FIT.json",dict(model=sha(ROOT/"MODEL.json"),development_accessed=False))


def freeze():
    registration()
    path = ROOT/"MODEL.json"
    assert sha(path)==read(ROOT/"FIT.json")["model"]
    write(ROOT/"ALL_FROZEN.json",dict(model=sha(path), registration=sha(ROOT/"REGISTRATION.json")))


def evaluate():
    registration()
    path = ROOT/"MODEL.json"
    seal = read(ROOT/"ALL_FROZEN.json")
    assert sha(path)==seal["model"] and sha(ROOT/"REGISTRATION.json")==seal["registration"]
    m = read(path)
    geo,pipe,data,rebind,stages = inputs(ENTITY,development=True)
    lo,hi = geo["development"]
    t = rows(lo,hi)
    train_lo = geo["calibration"][0]
    cut = int(t[len(t)//2])
    alt = poisoned(data,cut)
    clean = pipe.accessor
    actual = read(V07/ENTITY/"RESULT.json")
    result = dict(baseline=dict(metrics=actual["metrics"],gates=actual["gates"]))
    for label,arm in m["arms"].items():
        ctx = g.context(ENTITY,pipe,data,stages["K"],arm["K"],hi)
        p = Engine(pipe,ctx,train_lo).predict(t,arm["model"])
        pipe.accessor = rebind(alt)
        cc = g.context(ENTITY,pipe,alt,stages["K"],arm["K"],cut+1)
        q = Engine(pipe,cc,train_lo).predict(t[t<=cut],arm["model"])
        pipe.accessor = clean
        for step,z in q.items():
            np.testing.assert_allclose(p[step][:len(z)],z,rtol=0,atol=5e-11)
        result[label] = dict(metrics={step:score(data[t,0],z) for step,z in p.items()},
                             gates=arm["model"]["gates"],outer=arm["outer"])
        print(label,result[label]["metrics"]["KCWA"]["RMSE"],flush=True)
    result["shared_window"] = result[m["shared"]]
    for key,l in (("floor_1e-5","1e-05"),("floor_1e-4","0.0001")):
        assert m["floor_controls"][l]["identity"]
        result[key] = result["baseline"]
    result["previous_C_grid"] = result["baseline"]
    w = {}
    for e in CASES:
        r = read(V07/e/"RESULT.json")
        w[e] = r["gates"]["W"]
    write(ROOT/"RESULT.json",dict(N=len(t),arms=result, shared_choice=m["shared"],shared_loss=m["shared_loss"],
         K_audit=m["K_audit"],floor_controls=m["floor_controls"],C_control=m["C_control"],
         W=w,poison_cut=cut))


if __name__=="__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action",choices=CONFIG)
    {"registration":registration,"fit":fit,"freeze":freeze,"evaluate":evaluate}[cli.parse_args().action]()
