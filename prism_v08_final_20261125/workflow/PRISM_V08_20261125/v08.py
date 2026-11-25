"""Frozen PRISM v0.8 train, lock and development-replay entry point."""

import argparse
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
OLDER = ROOT.parent/"PRISM_V07_20261125"
if not OLDER.exists():
    OLDER = ROOT.parent/"prism_v07_20261125"
sys.path.insert(0,str(OLDER))
import v07 as v
from v07 import CASES, EMBARGO, Engine, g, h, np, poisoned, read, rows, s, score, sha, write


MODES = ("standalone_robust","fusion_fidelity")
GRID_SPLIT = 1e-5
OriginalBasis = g.NaturalBasis


class EndpointBasis(OriginalBasis):
    def __init__(self,values,width):
        x = values.reshape(-1)
        self.mean,self.scale = float(x.mean()),max(float(x.std()),1e-9)
        z = (x-self.mean)/self.scale
        p = np.arange(1,width+2)/(width+2)
        if width>1:
            p[[0,-1]] = .05,.95
        self.knots = np.unique(np.quantile(z,p))
        self.center = self.columns(z).mean(axis=0)
        self.dimension = len(self.center)


def inputs(entity,development=False):
    # We use frozen K reference contracts only. Never resume an archived fit on
    # a host with a different environment binding.
    from canonical.cz_crossrod_l128_20261008 import canonical_engine
    original = canonical_engine.Checkpoints
    with TemporaryDirectory(prefix="prism08-checkpoint-") as temp:
        def fresh(root,binding,*,resume=False):
            return original(Path(temp)/"isolated",binding,resume=False)

        with patch.object(canonical_engine,"Checkpoints",fresh), patch.object(
                v.legacy,"ReadThroughCheckpoints",lambda output,source:source):
            pipe_data=v.inputs(entity,development=development)
    return pipe_data


def policy():
    sources = (Path(__file__),OLDER/"v07.py",v.SPRINT/"sprint.py",
               s.OLDER/"guarded.py",s.OLDER/"common.py",s.OLDER/"resolution.py")
    origin = v.legacy.registration()
    return dict(version="PRISM v0.8",protocol=origin["protocol"],geometry=origin["geometry"],
        sources={str(p):sha(p) for p in sources}, earlier_protocol=sha(OLDER/"REGISTRATION.json"),
        natural_outer_quantiles=[.05,.95], shrinkage_modes={
            "standalone_robust":list(v.K_ALPHAS),
            "fusion_fidelity":[x for x in v.K_ALPHAS if x<=GRID_SPLIT]},
        treatment_of_modes="parallel K feature generation, shared downstream recipes, same training folds",
        near_bins=[4,8],lambda_per_row=True,
        downstream="K-C-A core with conditional W, active-coefficient trust and family-separated A vote",
        mode_gate=dict(blocks=4, minimum_wins=3, mean_gain_above_se=.5, training_outer_embargo=EMBARGO),
        score_data_not_used_for_selection=True,
        history="warm K and use actual mature residuals replayed from the first post-fit issue",
        holdout="development review first, then bind data authority, models and inference to a one-time run")


def register():
    path = ROOT/"REGISTRATION.json"
    obj = h.json.loads(h.json.dumps(policy()))
    if path.exists():
        assert read(path)==obj
    else:
        write(path,obj)


def fit_rng(e):
    register()
    geo,pipe,data,_,stages=inputs(e)
    lo,hi=geo["calibration"]
    k=stages["K"]
    saved={}
    original=v.K_GRID
    try:
        g.NaturalBasis=EndpointBasis
        for mode in MODES:
            v.K_GRID=original if mode==MODES[0] else tuple(x for x in original if x[2][0]<=GRID_SPLIT)
            kfit,audit=v.fit_k(geo,pipe,data,k)
            ctx=g.context(e,pipe,data,k,kfit,hi)
            engine=Engine(pipe,ctx,lo)
            evidence=[]
            for a,b in s.blocks(lo,hi,.70,4):
                prior=engine.train(a-EMBARGO)
                t=rows(a,b)
                pred=engine.predict(t,prior)
                evidence.append(dict(rows=[a,b],fit_stop=a-EMBARGO,metrics={
                    name:score(data[t,0],p) for name,p in pred.items()}))
            saved[mode]=dict(K=kfit,K_audit=audit,chain=engine.train(hi),outer=evidence)
    finally:
        g.NaturalBasis=OriginalBasis
        v.K_GRID=original
    evidence=[]
    for x,y in zip(saved[MODES[0]]["outer"],saved[MODES[1]]["outer"],strict=True):
        assert x["rows"]==y["rows"]
        a,b=(x["metrics"]["KCWA"]["MSE"],y["metrics"]["KCWA"]["MSE"])
        evidence.append(dict(rows=x["rows"],parent_MSE=a,child_MSE=b,gain=a-b))
    vote=s.vote(evidence)
    chosen=MODES[int(vote["admitted"])]
    path=ROOT/e/"MODEL.json"
    write(path,dict(entity=e,chosen=chosen,mode_vote=vote,models=saved))
    write(path.with_name("FIT.json"),dict(sha256=sha(path),development_read=False))
    print(e,chosen,vote["wins"],flush=True)


def freeze():
    register()
    models={}
    for e in CASES:
        p=ROOT/e/"MODEL.json"
        models[e]=sha(p)
        assert models[e]==read(p.with_name("FIT.json"))["sha256"]
    path=ROOT/"MODEL_LOCK.json"
    manifest=dict(registration=sha(ROOT/"REGISTRATION.json"),models=models)
    if path.exists():
        assert read(path)==manifest
    else:
        write(path,manifest)


def sealed(e):
    register()
    lock=read(ROOT/"MODEL_LOCK.json")
    p=ROOT/e/"MODEL.json"
    assert lock["registration"]==sha(ROOT/"REGISTRATION.json") and lock["models"][e]==sha(p)
    return read(p)


def predict(e,pipe,k,data,model,t):
    stop=int(t[-1])+1
    context=g.context(e,pipe,data,k,model["K"],stop)
    return Engine(pipe,context,model["chain"]["issue_from"]).predict(t,model["chain"])


def development(e):
    state=sealed(e)
    model=state["models"][state["chosen"]]
    geo,pipe,data,rebind,stages=inputs(e,development=True)
    lo,hi=geo["development"]; t=rows(lo,hi)
    k=stages["K"]
    p=predict(e,pipe,k,data,model,t)
    clean=pipe.accessor
    cuts=[int(t[19]),int(t[len(t)//2])]
    for cut in cuts:
        alt=poisoned(data,cut)
        pipe.accessor=rebind(alt)
        q=predict(e,pipe,k,alt,model,t[t<=cut])
        for stage,z in q.items():
            np.testing.assert_allclose(p[stage][:len(z)],z,rtol=0,atol=5e-11)
    pipe.accessor=clean
    path=ROOT/e/"DEVELOPMENT.json"
    r=dict(entity=e,N=len(t),selected=state["chosen"],mode_vote=state["mode_vote"],
           metrics={name:score(data[t,0],p[name]) for name in p},
           A=model["chain"]["gates"]["A"]["selected_family"],W=model["chain"]["W"] is not None,
           K_audit=model["K_audit"],causal_prefix_cuts=cuts)
    write(path,r)
    print(e,r["metrics"]["KCWA"]["RMSE"],flush=True)


if __name__=="__main__":
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action",choices=("register","fit","freeze","development"))
    cli.add_argument("entity",nargs="?",choices=CASES)
    a=cli.parse_args()
    if a.action in ("fit","development"):
        if a.entity is None:
            cli.error("entity required")
        {"fit":fit_rng,"development":development}[a.action](a.entity)
    else:
        {"register":register,"freeze":freeze}[a.action]()
