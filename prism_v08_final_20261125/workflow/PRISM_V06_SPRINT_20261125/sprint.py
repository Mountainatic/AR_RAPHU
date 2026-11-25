"""Frozen near-lag, shared-calibration and joint-readout PRISM experiment."""

import argparse
from functools import lru_cache
import itertools
from pathlib import Path
import sys
import warnings

from scipy.signal import lfilter
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet, Ridge

ROOT = Path(__file__).resolve().parent
OLDER = ROOT.parent/"PRISM_V06_GUARD_NESTED_RESOLUTION_20261124"
if not OLDER.exists():
    OLDER = ROOT.parent/"prism_v06_guard_nested_resolution_20261124"
sys.path.insert(0, str(OLDER))
import guarded as g
from common import CASES, h, inputs, legacy, np, poisoned, read, score, sha, write
from resolution import process


D, L, GAP = h.D, h.L, h.D-1
EMBARGO = L+D-1
FAMILIES = ("LINEAR_DISTRIBUTED_LAG", "RANK_1_URYSOHN",
            "RANK_2_URYSOHN", "FULL_FINITE_URYSOHN")
K_PENALTIES = (1e-5, .001, .03)
C_GRID = tuple(itertools.product((.001, .01, .1), (.5, .9)))
A_SHAPES = (("ema", 4), ("ema", 7), ("mix", 4), ("mix", 8),
            ("ar", (1, 8)), ("ar", (1, 16)), ("ar", (2, 32)), ("ar", (4, 64)))
A_GRID = tuple(itertools.product(A_SHAPES, (h.LEGACY_ALPHA, .05)))
J_GRID = (.001, .01, .1)
ARMS = {
    "main": ("near8_tuned", "auto", "active", "mixed"),
    "ema_only": ("near8_tuned", "auto", "active", "ema"),
    "ar_only": ("near8_tuned", "auto", "active", "ar"),
    "near4_tuned": ("near4_tuned", "auto", "active", "mixed"),
    "near8_fixed": ("near8_fixed", "auto", "active", "mixed"),
    "w_full_old_k4": ("near4_old", "auto", "min", "mixed"),
    "w_full_old_k4_ref": ("near4_old", "bypass", "min", "mixed"),
    "w_full_old_k4_active": ("near4_old", "auto", "active", "mixed"),
    "w_full_legacy_k_ref": ("legacy", "bypass", "min", "mixed"),
}
POLICY = dict(
    arms=ARMS, K=dict(families=FAMILIES, width=4, penalties_normed_to_n=K_PENALTIES,
                      natural_column_scale="centered train RMS, mapped back before inference",
                      windows_4=g.RAW_WINDOWS, windows_8=[*range(9),24,60,128],
                      reference_and_route="frozen October reference and switching rule",
                      fold_fractions=[.48,.67,.84,1.]),
    A=dict(grid=A_GRID, delayed_ema_half_lives=[4,8,16,32,64], legacy_ema_half_lives=g.EMA),
    C=C_GRID, W="prior four spline recipes; fixed percentiles; active or minimum trust",
    joint=dict(grid=J_GRID, input="active K/C/W/A additive increments", ridge_center=1.,
               weights_bounds=[0.,2.], intercept=False),
    gate=dict(required_wins=3, blocks=4, mean_gain_in_SE=.5, relative_MSE_floor=0.),
    shared=dict(gap=GAP, issue_at="fit_stop+D-1", conditional_parameter_cross_fitting=True,
                stream_blocks=8, initial_fraction=.21, gate_initial=.55, outer_embargo=EMBARGO),
    development="unchanged and opened only after sealing all fits",
)


def registration():
    paths = [Path(__file__), OLDER/"guarded.py", OLDER/"common.py", OLDER/"resolution.py"]
    previous = legacy.registration()
    return dict(protocol=previous["protocol"], geometry=previous["geometry"], policy=POLICY,
                source={p.name:sha(p) for p in paths}, prior=sha(OLDER/"guarded"/"ALL_FROZEN.json"),
                reserved_tail_accessed=False)


def register():
    path = ROOT/"REGISTRATION.json"
    cfg = h.json.loads(h.json.dumps(registration()))
    if path.exists():
        assert read(path)==cfg, "Registration changed"
    else:
        write(path,cfg)


def verify():
    assert h.json.loads(h.json.dumps(registration()))==read(ROOT/"REGISTRATION.json")


def rows(lo, hi):
    return np.arange(lo, hi, dtype=np.int64)


def blocks(lo, hi, fraction, count):
    edges = lo+np.floor(np.linspace(fraction, 1., count+1)*(hi-lo)).astype(int)
    edges[-1] = hi
    return [(int(a), int(b)) for a,b in zip(edges[:-1],edges[1:])]


def k_contract(pipe, x, t, y, edges, spec, normalize):
    family, width, penalties = spec
    if edges is None:
        v = x
        basis = g.NaturalBasis(v, width)
        phi = basis.transform(v)
    else:
        basis = g.NaturalBasis(x[t[0]-edges[-1]+1:t[-1]+1], width)
        v = g.bucket(x,t,edges)[:,:,0]
        phi = g.bucket(basis.transform(x[:t[-1]+1]),t,edges)
    scale = np.maximum(np.sqrt(np.mean(phi*phi,axis=(0,1))),1e-7) if normalize else np.ones(phi.shape[2])
    z = phi/scale
    model = g.k_surface(pipe,v,y,z,basis,dict(family=family,m_x=width,lambdas=penalties))
    beta = np.asarray(model["theta"])
    model["theta"] = (beta/scale[:beta.shape[1]]).tolist()
    return model,g.limits(v)


def k_predict(pipe, x, t, edges, model):
    basis = g.NaturalBasis.load(model["basis"])
    phi = basis.transform(x) if edges is None else g.bucket(basis.transform(x[:t[-1]+1]),t,edges)
    return pipe.u.predict_contract_from_design(phi,model)


def k_recipe(pipe, data, lo, hi, edges, x, legacy_spec, mode):
    fit_t = rows(lo,hi)
    y = data[:,0]
    if mode == "fixed":
        model,limits = k_contract(pipe,x,fit_t,y[fit_t]-y[fit_t-D],edges,legacy_spec,False)
        return model,limits,dict(spec=legacy_spec)
    # Keep distinct scale, lag-smoothing and amplitude-smoothing paths available.
    grid = [(family, 1 if family==FAMILIES[0] else 4,
             tuple(alpha*rel for rel in ratio))
            for family in FAMILIES for alpha in K_PENALTIES
            for ratio in (((1.,1.,0.),) if family==FAMILIES[0] else ((1.,1.,1.),(1.,.1,10.)))]
    starts = lo+np.floor(np.array([.48,.67,.84,1.])*(hi-lo)).astype(int)
    starts[-1] = hi
    fold_losses = [[] for _ in grid]
    for start,end in zip(starts[:-1],starts[1:]):
        t = rows(lo,int(start)-EMBARGO)
        ev = rows(start,end)
        xt = x[:len(t)] if edges is None else x
        xe = x[start-lo:end-lo] if edges is None else x
        target = y[ev]-y[ev-D]
        for i,spec in enumerate(grid):
            model,_ = k_contract(pipe,xt,t,y[t]-y[t-D],edges,spec,True)
            pred = k_predict(pipe,xe,ev,edges,model)
            fold_losses[i].append(float(np.mean((target-pred)**2)))
    risks = [float(np.mean(loss)) for loss in fold_losses]
    i = int(np.argmin(risks))
    model,limits = k_contract(pipe,x,fit_t,y[fit_t]-y[fit_t-D],edges,grid[i],True)
    if edges is None:
        # Same-bin control can isolate rank/penalty selection from lag expansion.
        old,_ = k_contract(pipe,x,fit_t,y[fit_t]-y[fit_t-D],edges,legacy_spec,False)
        old_rms = float(np.sqrt(np.mean((y[fit_t]-y[fit_t-D]-k_predict(pipe,x,fit_t,edges,old))**2)))
    else:
        old_rms = None
    return model,limits,dict(spec=grid[i], train_validation_MSE=risks[i], grid_risks=risks,
                             grid=grid, old_full_fit_RMSE=old_rms)


def fit_k(entity, geo, pipe, data, k, variant, old):
    if variant in {"near4_old","legacy"}:
        return old["K"]["hybrid" if variant=="near4_old" else "legacy"], []
    lo,hi = geo["parent"]
    audit = []
    models = []
    binned = variant=="natural_binned_tuned"
    mode = "fixed" if variant=="near8_fixed" else "tuned"
    for rep in k["representations"]:
        family = rep["family"]
        base = (family,int(rep["m_x"]),tuple(rep["lambdas"]))
        if binned:
            if family==FAMILIES[0]:
                index = len(models)
                models.append(old["K"]["natural"][index])
                continue
            f = h.frame(entity,lo,hi,"K_fit")
            v,_ = pipe._values(f,rep)
            ct,lim,rec = k_recipe(pipe,data,lo,hi,None,v,base,mode)
            models.append([dict(view="natural",contract=ct,support=lim)])
            audit.append(dict(channel=rep["channel"],view="natural",**rec))
            continue
        signal = process(data[:hi],rep,"reference")[:,0]
        delta = np.r_[np.zeros(D),signal[D:]-signal[:-D]]
        views = []
        for name, x in (("level",signal),("delta",delta)):
            edges = g.RAW_WINDOWS if variant=="near4_tuned" else (*range(9),24,60,128)
            if name=="delta":
                edges = (*edges[:-1],L-D)
            ct,lim,rec = k_recipe(pipe,data,lo,hi,edges,x,base,mode)
            views.append(dict(view=name,edges=edges,contract=ct,support=lim))
            audit.append(dict(channel=rep["channel"],view=name,**rec))
        models.append(views)
    return models,audit


def a_design(resid, start, t, shape):
    family, size = shape
    source = resid[start:int(t[-1])-D+1]
    relative = t-start-D
    if family=="ar":
        stride, span = size
        offsets = np.arange(0,span,stride)
        positions = relative[:,None]-offsets
        return np.where(positions>=0,resid[np.maximum(start+positions,start)],0.)
    half = np.asarray(g.EMA[:size] if family=="ema" else (4,8,16,32,64))
    alpha = 1.-np.exp2(-1./half)
    state = np.column_stack([lfilter([a],[1.,a-1.],source) for a in alpha]) if len(source) else np.zeros((1,len(alpha)))
    p = relative if family=="ema" else relative-size
    tail = np.where(p[:,None]>=0,state[np.maximum(p,0)],0.)
    if family=="ema":
        return tail
    positions = relative[:,None]-np.arange(size)
    exact = np.where(positions>=0,resid[np.maximum(start+positions,start)],0.)
    return np.c_[exact,tail]


def vote(evidence):
    z = np.asarray([item["gain"] for item in evidence])
    n = len(z)
    se = float(z.std(ddof=1)/np.sqrt(n))
    wins = int((z>0).sum())
    gain = float(z.mean())
    return dict(wins=wins,mean_gain=gain,SE=se,threshold=.5*se,
                admitted=bool(wins>=3 and gain>.5*se),blocks=evidence)


def compare(t, truth, parent, child, fit_end, spec=None):
    p = (truth[t]-parent)**2
    c = (truth[t]-child)**2
    return dict(rows=[int(t[0]),int(t[-1])+1], fit_end=fit_end, spec=spec,
                parent_mse=float(p.mean()), child_mse=float(c.mean()), gain=float((p-c).mean()))


class SharedFit:
    def __init__(self, pipe, ctx, lo, c_policy, trust_policy, a_policy):
        self.pipe = pipe
        self.ctx = ctx
        self.lo = lo
        self.c_policy = c_policy
        self.trust_policy = trust_policy
        self.a_grid = [s for s in A_GRID if a_policy=="mixed" or s[0][0]==a_policy]
        self.k = g.c_base(ctx,True)
        self.y = ctx["y"]-ctx["anchor"]
        w = pipe.w
        self.w_grid = [(w.MONOTONE,4,10.,s) for s in (-1,1)]
        self.w_grid += [(w.NATURAL_CUBIC,6,a,1) for a in (1.,100.)]

    def c_fit(self,t,spec):
        if spec is None:
            return None
        alpha,ratio = spec
        raw = self.ctx["x"][t]
        mean,scale = raw.mean(0),raw.std(0)
        scale[scale<1e-10] = 1.
        x = g.c_features(self.ctx,t,mean,scale,True)
        y = self.y[t]-self.k[t]
        model = ElasticNet(alpha=alpha*max(float(y.std()),1e-10),l1_ratio=ratio,
                           max_iter=20000,tol=1e-5)
        with warnings.catch_warnings():
            warnings.simplefilter("error",ConvergenceWarning)
            try:
                model.fit(x,y)
            except ConvergenceWarning:
                return None
        return dict(mean=mean.tolist(),scale=scale.tolist(),coef=model.coef_.tolist(),
                    intercept=float(model.intercept_), spec=spec)

    def c_eval(self,t,model):
        kc = g.c_predict(self.ctx,t,self.k,model,True)
        trust = self.ctx["g"][t]
        if model is None:
            ref = self.ctx["ref"]
            return kc,self.ctx["x"][t,ref],trust[:,ref]
        coef = np.abs(model["coef"])
        if self.trust_policy=="min":
            weight = trust.min(1)
        elif coef.sum()>0.:
            weight = trust@coef/coef.sum()
        else:
            weight = trust.max(1)
        return kc,kc,weight

    @lru_cache(maxsize=128)
    def c_select(self,end):
        if self.c_policy=="bypass":
            return None
        split = self.lo+int(.70*(end-self.lo))
        train,t = rows(self.lo,split-GAP),rows(split,end)
        candidates = []
        for spec in C_GRID:
            model = self.c_fit(train,spec)
            if model is not None:
                pred = self.c_eval(t,model)[0]
                candidates.append((float(np.mean((self.y[t]-pred)**2)),spec))
        return self.c_fit(rows(self.lo,end),min(candidates)[1]) if candidates else None

    @lru_cache(maxsize=128)
    def c_route(self,end):
        evidence = []
        for a,b in blocks(self.lo,end,.55,4):
            t = rows(a,b)
            m = self.c_select(a-GAP)
            evidence.append(compare(t,self.y,self.k[t],self.c_eval(t,m)[0],a-GAP,
                                    m["spec"] if m else None))
        audit = vote(evidence)
        m = self.c_select(end) if audit["admitted"] else None
        return m,audit

    @lru_cache(maxsize=128)
    def c_stream(self,end,spec):
        section = blocks(self.lo,end,.21,8)
        first = section[0][0]
        output = np.empty((3,end-self.lo))
        for a,b in section:
            m = self.c_fit(rows(self.lo,a-GAP),spec)
            output[:,a-self.lo:b-self.lo] = self.c_eval(rows(a,b),m)
        return first,output

    def w_fit(self,t,kc,z,trust,spec):
        if spec is None:
            return None
        select = trust>=.95 if self.trust_policy=="min" else trust>=.25
        if select.sum()<26 or np.std(z[select])<1e-10:
            return None
        z = z[select]
        limits = np.quantile(z,[.01,.99]).tolist()
        z = np.clip(z,*limits)
        if np.ptp(z)<1e-10:
            return None
        family,knots,smoothness,direction = spec
        _,model = self.pipe.w.fit_w_correction(z,self.y[t[select]]-kc[select],z,family=family,
                          knot_count=knots,smoothness=smoothness,direction=direction)
        return dict(contract=model,limits=limits,spec=spec,eligible=int(select.sum())) if self.pipe.cert(model) else None

    def w_eval(self,kc,z,trust,model):
        if model is None:
            return kc
        return kc + trust*self.pipe.w.predict_w_correction(np.clip(z,*model["limits"]),model["contract"])

    @lru_cache(maxsize=128)
    def w_select(self,end):
        c,_ = self.c_route(end)
        first,stream = self.c_stream(end,c["spec"] if c else None)
        split = first+int(.68*(end-first))
        train = rows(first,split-GAP)
        t = rows(split,end)
        fit_x,score_x = stream[:,train-self.lo],stream[:,t-self.lo]
        usable = []
        for spec in self.w_grid:
            m = self.w_fit(train,*fit_x,spec)
            if m is not None:
                p = self.w_eval(*score_x,m)
                usable.append((float(np.mean((self.y[t]-p)**2)),spec))
        if not usable:
            return c,None
        ix = rows(first,end)
        return c,self.w_fit(ix,*stream[:,ix-self.lo],min(usable)[1])

    @lru_cache(maxsize=128)
    def w_route(self,end):
        evidence = []
        for a,b in blocks(self.lo,end,.55,4):
            t = rows(a,b)
            c,w = self.w_select(a-GAP)
            v = self.c_eval(t,c)
            evidence.append(compare(t,self.y,v[0],self.w_eval(*v,w),a-GAP,
                            dict(W=w["spec"] if w else None, C=c["spec"] if c else None,
                                 eligible=w["eligible"] if w else 0)))
        audit = vote(evidence)
        c,w = self.w_select(end)
        audit["final_candidate"] = (dict(spec=w["spec"], eligible=w["eligible"]) if w else None)
        return c,w if audit["admitted"] else None,audit

    @lru_cache(maxsize=128)
    def cw_stream(self,end,c_spec,w_spec):
        first,stream = self.c_stream(end,c_spec)
        sections = blocks(self.lo,end,.21,8)
        components = np.empty((end-first,3))
        for a,b in sections:
            train = rows(first,a-GAP)
            w = self.w_fit(train,*stream[:,train-self.lo],w_spec) if len(train)>=26 else None
            t = rows(a,b)
            v = stream[:,t-self.lo]
            q = self.w_eval(*v,w)
            components[t-first] = np.c_[self.k[t],v[0]-self.k[t],q-v[0]]
        return first,components

    def design(self, residual, start, t, spec):
        return a_design(residual,start,t,spec[0])

    def a_fit(self, residual, start, t, spec):
        if spec is None:
            return None
        x = self.design(residual,start,t,spec)
        _,m = self.pipe.a.fit_mature_residual_ar(x,residual[t],x,alpha=spec[1],mu=0.)
        return dict(spec=spec,contract=m) if self.pipe.cert(m) else None

    def a_eval(self, residual, start, t, model):
        if model is None:
            return np.zeros(len(t))
        x = self.design(residual,start,t,model["spec"])
        z = self.pipe.a.predict_mature_residual_ar(x,model["contract"])
        return np.where(t>=start+D,z,0.)

    def from_routes(self, end):
        c,w,_ = self.w_route(end)
        start,p = self.cw_stream(end,c["spec"] if c else None,w["spec"] if w else None)
        residual = np.zeros(end)
        residual[start:] = self.y[start:end]-p.sum(1)
        return c,w,start,p,residual

    @lru_cache(maxsize=128)
    def a_select(self,end):
        c,w,first,_,r = self.from_routes(end)
        split = first+int(.69*(end-first))
        train = rows(first+D,split-GAP)
        t = rows(split,end)
        grid = []
        for spec in self.a_grid:
            m = self.a_fit(r,first,train,spec)
            if m is not None:
                pred = self.a_eval(r,first,t,m)
                grid.append((float(np.mean((r[t]-pred)**2)),spec))
        return c,w,self.a_fit(r,first,rows(first+D,end),min(grid)[1]) if grid else None

    @lru_cache(maxsize=128)
    def a_route(self,end):
        evidence = []
        for a,b in blocks(self.lo,end,.55,4):
            t = rows(a,b)
            c,w,m = self.a_select(a-GAP)
            model = dict(C=c,W=w,A=m,issue_from=a)
            comp = self.components(t,model)
            evidence.append(compare(t,self.y,comp[:,:3].sum(1),comp.sum(1),a-GAP,m["spec"] if m else None))
        audit = vote(evidence)
        c,w,m = self.a_select(end)
        return c,w,m if audit["admitted"] else None,audit

    def components(self,t,model):
        start = model["issue_from"]
        assert t[0]>=start
        whole = rows(start,int(t[-1])+1)
        k = self.k[whole]
        v = self.c_eval(whole,model["C"])
        q = self.w_eval(*v,model["W"])
        residual = np.zeros(int(t[-1])+1)
        residual[start:] = self.y[whole]-q
        a = self.a_eval(residual,start,t,model["A"])
        relative = t-start
        return np.c_[k[relative],(v[0]-k)[relative],(q-v[0])[relative],a]

    @lru_cache(maxsize=128)
    def base_model(self,end):
        c,w,a,aa = self.a_route(end)
        _,ac = self.c_route(end)
        _,_,aw = self.w_route(end)
        return dict(C=c,W=w,A=a,issue_from=end+GAP,gates=dict(C=ac,W=aw,A=aa))

    def joint_stream(self,end,base):
        _,_,first,cw,r = self.from_routes(end)
        t = rows(first,end)
        a = np.zeros(len(t))
        selected = base["A"]
        if selected is not None:
            for lo,hi in blocks(first,end,.27,5):
                fit_t = rows(first+D,lo-GAP)
                model = self.a_fit(r,first,fit_t,selected["spec"])
                a[lo-first:hi-first] = self.a_eval(r,first,rows(lo,hi),model)
        start = first+int(.27*(end-first))
        return rows(start,end),np.c_[cw,a][start-first:]

    @staticmethod
    def joint_weights(x,y,alpha):
        scale = np.maximum(np.sqrt(np.mean(x*x,axis=0)),1e-9)
        z = x/scale
        adjustment = Ridge(alpha=alpha*len(y),fit_intercept=False).fit(z,y-x.sum(1)).coef_/scale
        return np.clip(1.+adjustment,0.,2.)

    @lru_cache(maxsize=128)
    def joint_select(self,end):
        base = self.base_model(end)
        t,x = self.joint_stream(end,base)
        split = int(.7*len(t))
        subset = slice(split-GAP)
        weights = [self.joint_weights(x[subset],self.y[t[subset]],p) for p in J_GRID]
        risks = [np.mean((self.y[t[split:]]-x[split:]@w)**2) for w in weights]
        alpha = J_GRID[int(np.argmin(risks))]
        return base,dict(alpha=alpha, weights=self.joint_weights(x,self.y[t],alpha).tolist())

    def train(self,end,joint):
        if not joint:
            return self.base_model(end)
        evidence = []
        for a,b in blocks(self.lo,end,.55,4):
            t = rows(a,b)
            base,j = self.joint_select(a-GAP)
            x = self.components(t,base)
            evidence.append(compare(t,self.y,x.sum(1),x@j["weights"],a-GAP,j["alpha"]))
        audit = vote(evidence)
        base,j = self.joint_select(end)
        return dict(base,J=j if audit["admitted"] else None,J_candidate=j,J_audit=audit)

    def predict(self,t,model):
        x = self.components(t,model)
        anchor = self.ctx["anchor"][t]
        stages = np.cumsum(x,axis=1)+anchor[:,None]
        p = {name:stages[:,i] for i,name in enumerate(("K","KC","KCW","KCWA"))}
        if "J_candidate" in model:
            p["J_candidate"] = anchor+x@model["J_candidate"]["weights"]
            p["selected"] = p["J_candidate"] if model["J"] else p["KCWA"]
        return p


def source_model(entity):
    path = OLDER/"guarded"/entity/"MODEL.json"
    assert sha(path)==read(OLDER/"guarded"/"ALL_FROZEN.json")["models"][entity]
    return read(path)


def fit(entity):
    verify()
    geo,pipe,data,_,stages = inputs(entity)
    k = stages["K"]
    lo,hi = geo["calibration"]
    old = source_model(entity)
    arms,k_models,k_audit,outer = {},{},{},{}
    keys = sorted({spec[0] for spec in ARMS.values()}|{"natural_binned_tuned"})
    contexts = {}
    for kind in keys:
        k_models[kind],k_audit[kind] = fit_k(entity,geo,pipe,data,k,kind,old)
        contexts[kind] = g.context(entity,pipe,data,k,k_models[kind],hi)
        print("K",entity,kind,flush=True)
    for arm,(kind,c_policy,trust,a_policy) in ARMS.items():
        engine = SharedFit(pipe,contexts[kind],lo,c_policy,trust,a_policy)
        joint = arm=="main"
        audits = []
        for a,b in blocks(lo,hi,.70,4):
            end = a-EMBARGO
            model = engine.train(end,joint)
            t = rows(a,b)
            p = engine.predict(t,model)
            audits.append(dict(fit_end=end,rows=[a,b],gates=model["gates"],J=model.get("J_audit"),
                               metrics={stage:score(data[t,0],z) for stage,z in p.items()}))
        arms[arm] = engine.train(hi,joint)
        outer[arm] = audits
        print("CHAIN",entity,arm,[v["admitted"] for v in arms[arm]["gates"].values()],
              arms[arm].get("J") is not None,flush=True)
    natural = g.combo(pipe,contexts["natural_binned_tuned"],lo,hi,"natural_guard_robust")
    path = ROOT/entity/"MODEL.json"
    path.parent.mkdir(exist_ok=True)
    write(path,dict(K=k_models,K_audit=k_audit,arms=arms,outer=outer,natural_staged=natural))
    write(path.with_name("FIT.json"),dict(entity=entity,sha256=sha(path),development_access=False))


def freeze():
    verify()
    items = {e:ROOT/e/"MODEL.json" for e in CASES}
    models = {e:sha(p) for e,p in items.items()}
    assert all(models[e]==read(items[e].with_name("FIT.json"))["sha256"] for e in CASES)
    write(ROOT/"ALL_FROZEN.json",dict(models=models, registration=sha(ROOT/"REGISTRATION.json")))


def evaluate(entity):
    verify()
    path = ROOT/entity/"MODEL.json"
    seal = read(ROOT/"ALL_FROZEN.json")
    assert sha(path)==seal["models"][entity] and sha(ROOT/"REGISTRATION.json")==seal["registration"]
    frozen = read(path)
    geo,pipe,data,rebind,stages = inputs(entity,development=True)
    k = stages["K"]
    lo,hi = geo["development"]
    t = rows(lo,hi)
    training_lo = geo["calibration"][0]
    cuts = [int(t[37]),int(t[len(t)//2]),int(t[-1])]
    ctx,dirty = {},{}
    for kind in {v[0] for v in ARMS.values()}|{"natural_binned_tuned"}:
        ks = frozen["K"][kind]
        ctx[kind] = g.context(entity,pipe,data,k,ks,hi)
        clean = pipe.accessor
        dirty[kind] = []
        for cut in cuts:
            alt = poisoned(data,cut)
            pipe.accessor = rebind(alt)
            dirty[kind].append(g.context(entity,pipe,alt,k,ks,cut+1))
        pipe.accessor = clean
    result = {}
    for arm,(kind,c_policy,trust,a_policy) in ARMS.items():
        model = frozen["arms"][arm]
        engine = SharedFit(pipe,ctx[kind],training_lo,c_policy,trust,a_policy)
        p = engine.predict(t,model)
        for cut,alt in zip(cuts,dirty[kind]):
            alt_engine = SharedFit(pipe,alt,training_lo,c_policy,trust,a_policy)
            sub = t[t<=cut]
            for stage,q in alt_engine.predict(sub,model).items():
                np.testing.assert_allclose(p[stage][:len(q)],q,atol=5e-11,rtol=0.)
        results = {key:score(data[t,0],pred) for key,pred in p.items()}
        route = {s:(model[s]["spec"] if model[s] else None) for s in ("C","W","A")}
        result[arm] = dict(metrics=results,routes=route,gates=model["gates"],J=model.get("J_audit"),
                           J_weights=model.get("J_candidate"),J_admitted=model.get("J") is not None)
        print("SCORE",entity,arm,{s:round(v["RMSE"],8) for s,v in results.items()},flush=True)
    m = frozen["natural_staged"]
    natural = g.predict(pipe,ctx["natural_binned_tuned"],"natural_guard_robust",m,t,m["upstream_available_from"])
    for cut,alt in zip(cuts,dirty["natural_binned_tuned"]):
        sub = t[t<=cut]
        q = g.predict(pipe,alt,"natural_guard_robust",m,sub,m["upstream_available_from"])
        for stage in q:
            np.testing.assert_allclose(natural[stage][:len(sub)],q[stage],atol=5e-11,rtol=0.)
    write(path.with_name("RESULT.json"),dict(entity=entity,N=len(t),arms=result,
          outer=frozen["outer"],natural_staged=dict(metrics={s:score(data[t,0],p) for s,p in natural.items()},
          gates=m["gates"]), K_audit=frozen["K_audit"],causal_prefix_checks=cuts))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("register","fit","freeze","evaluate"))
    parser.add_argument("entity",nargs="?",choices=CASES)
    args = parser.parse_args()
    if args.action in ("fit","evaluate"):
        if args.entity is None:
            parser.error("fit and evaluate require an entity")
        {"fit":fit,"evaluate":evaluate}[args.action](args.entity)
    else:
        {"register":register,"freeze":freeze}[args.action]()


if __name__=="__main__":
    main()
