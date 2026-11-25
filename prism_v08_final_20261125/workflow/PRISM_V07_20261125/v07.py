"""PRISM v0.7: fixed training algorithm and sealed development replay."""

import argparse
from functools import lru_cache
import itertools
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SPRINT = ROOT.parent/"PRISM_V06_SPRINT_20261125"
if not SPRINT.exists():
    SPRINT = ROOT.parent/"prism_v06_sprint_20261125"
sys.path.insert(0,str(SPRINT))
import sprint as s
from sprint import CASES, D, EMBARGO, GAP, L, g, h, inputs, legacy, np, poisoned, read, rows, score, sha, write
from resolution import process


FAMILIES = ("ema","ar","mix")
K_ALPHAS = (1e-9,1e-7,1e-5,.001,.03)
K_FAMILIES = (("LINEAR_DISTRIBUTED_LAG",1),("RANK_1_URYSOHN",4),
              ("RANK_2_URYSOHN",4),("FULL_FINITE_URYSOHN",4))
K_GRID = tuple((family,width,tuple(alpha*value for value in ratio))
    for family,width in K_FAMILIES for alpha in K_ALPHAS
    for ratio in (((1.,1.,0.),) if width==1 else ((1.,1.,1.),(1.,.1,10.))))
K_WINDOWS = {4:g.RAW_WINDOWS,8:(*range(9),24,60,128)}
A_GRID = tuple(itertools.product(s.A_SHAPES,(1e-8,1e-5,.001)))
W_SMOOTH = (.001,.015,.08)
SCRIPT = Path(__file__).resolve()


def config():
    reg = legacy.registration()
    paths = (SCRIPT,SPRINT/"sprint.py",s.OLDER/"common.py",s.OLDER/"guarded.py",s.OLDER/"resolution.py")
    return dict(version="PRISM v0.7",protocol=reg["protocol"],geometry=reg["geometry"],
        policy=dict(K=dict(windows=K_WINDOWS, grid=K_GRID, risks="training forward paired-row MSE",
                           penalty="SSE penalties are multiplied by fold fit count",
                           width_scale="fit-only centered column RMS",
                           amplitude="exact integral of squared second derivative in standardized amplitude",
                           rank_solver="deterministic SVD initialization and penalized ALS",
                           reference="archived October K reference and K route"),
                    C=dict(grid=s.C_GRID,trust="view-specific 1-99% support with cubic taper"),
                    W=dict(smooth=W_SMOOTH,trust="C absolute coefficient weighted trust; reference trust if C bypass"),
                    A=dict(grid=A_GRID,families=FAMILIES,gate="each family competes after its own four votes"),
                    shared_calibration=dict(label_gap=GAP,outer_embargo=EMBARGO,parameter_cross_fitting=True),
                    admission=dict(wins=3,blocks=4,mean_gain_margin_in_SE=.5),
                    diagnostics="paired Rod1 fixed-bin amplitude penalty; separate from model choice"),
        source={path.name:sha(path) for path in paths},
        prior_geometry=sha(SPRINT/"REGISTRATION.json"),
        reserved_tail_accessed=False)


def bound():
    assert h.json.loads(h.json.dumps(config()))==read(ROOT/"REGISTRATION.json")


def register():
    path = ROOT/"REGISTRATION.json"
    if path.exists():
        bound()
    else:
        write(path,config())


def curvature(basis,scale):
    knots = basis.knots
    out = np.zeros((len(scale),len(scale)))
    if len(knots)<3:
        return out
    right = knots[-1]
    alternate = knots[-2]
    for lo,hi in zip(knots[:-1],knots[1:]):
        midpoint = (lo+hi)/2.
        z = midpoint+(hi-lo)/(2.*np.sqrt(3.))*np.array([-1.,1.])
        d = np.maximum(z[:,None]-knots[:-2],0.)/(right-knots[:-2])
        d -= (np.maximum(z-alternate,0.)/(right-alternate))[:,None]
        d = 6.*d/scale[None,1:]
        out[1:,1:] += (hi-lo)/2.*(d.T@d)
    assert np.all(out[0]==0.) and np.linalg.eigvalsh(out).min()>=-1e-10
    return out


class Surface:
    def __init__(self,pipe,x,t,y,edges,width,mode):
        self.u = pipe.u
        self.v = x if edges is None else g.bucket(x,t,edges)[:,:,0]
        sample = self.v if edges is None else x[t[0]-edges[-1]+1:t[-1]+1]
        self.basis = g.NaturalBasis(sample,width)
        phi = self.basis.transform(self.v) if edges is None else g.bucket(self.basis.transform(x[:t[-1]+1]),t,edges)
        self.scale = np.maximum(np.sqrt(np.mean(phi*phi,axis=(0,1))),1e-7)
        p = phi/self.scale
        self.nt,self.nx = p.shape[1:]
        self.stat = self.u.DesignStatistics.from_design(p.reshape(len(p),-1),y)
        self.linear = self.u.DesignStatistics.from_design(p[:,:,0],y)
        self.curve = curvature(self.basis,self.scale) if mode=="curvature" else (
            self.u._difference_penalty_cached(self.nx) if mode=="difference" else np.zeros((self.nx,self.nx)))
        self.time = self.u._difference_penalty_cached(self.nt)
        self.limits = g.limits(self.v)
        self.n = len(t)

    def solve(self,spec,scaled=True):
        family,_,alpha = spec
        ridge,lag,amp = (np.array(alpha)*self.n if scaled else alpha)
        nt,nx = self.nt,self.nx
        penalty_t = ridge*np.eye(nt)+lag*self.time
        if family==K_FAMILIES[0][0]:
            beta,bias,cert = self.u._statistics_solve(self.linear,penalty_t)
            t = beta[:,None]
        else:
            penalty = np.kron(penalty_t,np.eye(nx))+amp*np.kron(np.eye(nt),self.curve)
            matrix,bias,cert = self.u._statistics_solve(self.stat,penalty)
            t = matrix.reshape(nt,nx)
            if family.startswith("RANK_"):
                rank = min(int(family.split("_")[1]),nt,nx)
                t,bias,cert = self.low_rank(t,rank,penalty_t,ridge*np.eye(nx)+amp*self.curve,penalty)
        assert np.isfinite(t).all() and np.isfinite(bias)
        return dict(family=family,basis=self.basis.metadata(),theta=(t/self.scale[:t.shape[1]]).tolist(),
                    intercept=float(bias),certificate=cert)

    def low_rank(self,initial,rank,t_penalty,x_penalty,surface_penalty):
        u = self.u
        _,d,right = np.linalg.svd(initial,full_matrices=False)
        v = right[:rank].T*np.sqrt(np.maximum(d[:rank],1e-15))[None,:]
        p_t = np.kron(np.eye(rank),t_penalty)
        p_x = np.kron(np.eye(rank),x_penalty)
        last = np.inf
        matrix = initial
        for _ in range(40):
            a,_,_ = u._statistics_solve(self.stat.project(u._u_transform(v,self.nt)),p_t)
            l = a.reshape(rank,self.nt).T
            b,bias,_ = u._statistics_solve(self.stat.project(u._v_transform(l,self.nx)),p_x)
            v = b.reshape(rank,self.nx).T
            matrix = l@v.T
            mse = u._statistics_mse(self.stat,matrix.ravel(),bias)
            objective = mse+(a@p_t@a+b@p_x@b)/self.n
            if abs(last-objective)<1e-10+1e-6*abs(objective):
                break
            last = objective
        left,_,right = np.linalg.svd(matrix,full_matrices=False)
        transform = u._component_transform(left,right,rank)
        w,bias,cert = u._statistics_solve(self.stat.project(transform),transform.T@surface_penalty@transform)
        return (transform@w).reshape(self.nt,self.nx),bias,cert


def choose_k(pipe,signal,y,lo,hi,view,binned=False,mode="curvature"):
    windows = {0:None} if binned else {
        n:edges if view=="level" else (*edges[:-1],L-D) for n,edges in K_WINDOWS.items()}
    candidates = list(itertools.product(windows,K_GRID))
    errors = np.zeros(len(candidates))
    folds = s.blocks(lo,hi,.48,3)
    # Reuse each fold's transformed tensors and sufficient statistics across penalties.
    for a,b in folds:
        fit_t = rows(lo,a-EMBARGO)
        ev = rows(a,b)
        z = signal[:len(fit_t)] if binned else signal
        w = signal[a-lo:b-lo] if binned else signal
        cache = {}
        for i,(near,spec) in enumerate(candidates):
            key = near,spec[1]
            edges = windows[near]
            if key not in cache:
                surface = Surface(pipe,z,fit_t,y[fit_t],edges,spec[1],mode)
                basis = surface.basis
                phi = (basis.transform(w) if binned else
                       g.bucket(basis.transform(w[:ev[-1]+1]),ev,edges))
                cache[key] = surface,phi
            surface,phi = cache[key]
            ct = surface.solve(spec)
            err = y[ev]-pipe.u.predict_contract_from_design(phi,ct)
            errors[i] += err@err
    i = int(np.argmin(errors))
    near,spec = candidates[i]
    t = rows(lo,hi)
    edges = windows[near]
    surface = Surface(pipe,signal,t,y[t],edges,spec[1],mode)
    ct = surface.solve(spec)
    by_near = {n:float(min(errors[i] for i,v in enumerate(candidates) if v[0]==n)) for n in windows}
    audit = dict(near=near,spec=spec,fold_pooled_SSE=float(errors[i]),lag_alternatives_SSE=by_near,
                 knots=ct["basis"]["knots"],fit_count=len(t),penalty=mode)
    return dict(view="natural" if binned else view,edges=edges,contract=ct,support=surface.limits),audit


def fit_k(geo, pipe, data, k):
    lo,hi = geo["parent"]
    y = np.zeros(hi); y[D:] = data[D:hi,0]-data[:hi-D,0]
    models, audits = [], []
    for rep in k["representations"]:
        base = process(data[:hi],rep,"reference")[:,0]
        delta = np.r_[np.zeros(D),base[D:]-base[:-D]]
        sub = []
        for view,signal in (("level",base),("delta",delta)):
            model,audit = choose_k(pipe,signal,y,lo,hi,view)
            sub.append(model)
            audits.append(dict(channel=rep["channel"],view=view,**audit))
        models.append(sub)
    return models,audits


class Engine(s.SharedFit):
    def __init__(self,pipe,ctx,lo):
        super().__init__(pipe,ctx,lo,"auto","active","mixed")
        w = pipe.w
        self.w_grid = [(w.MONOTONE,4,a,d) for a,d in itertools.product(W_SMOOTH,(-1,1))]
        self.w_grid += [(w.NATURAL_CUBIC,6,a,1) for a in W_SMOOTH]

    def w_fit(self,t,kc,z,trust,spec):
        if spec is None:
            return None
        n = int((trust>=.25).sum())
        family,knots,alpha,direction = spec
        out = super().w_fit(t,kc,z,trust,(family,knots,alpha*n,direction))
        if out is not None:
            out["spec"] = spec
        return out

    def a_fit(self,residual,start,t,spec):
        if spec is None:
            return None
        model = super().a_fit(residual,start,t,(spec[0],spec[1]*len(t)))
        if model is not None:
            model["spec"] = spec
        return model

    @lru_cache(maxsize=128)
    def a_select(self,end,family):
        c,w,start,_,r = self.from_routes(end)
        split = start+int(.69*(end-start))
        train,t = rows(start+D,split-GAP),rows(split,end)
        risk = []
        for spec in A_GRID:
            if spec[0][0]!=family:
                continue
            m = self.a_fit(r,start,train,spec)
            if m is not None:
                child = self.a_eval(r,start,t,m)
                risk.append((float(np.mean((r[t]-child)**2)),spec))
        if not risk:
            return c,w,None
        return c,w,self.a_fit(r,start,rows(start+D,end),min(risk)[1])

    @lru_cache(maxsize=128)
    def a_route(self,end):
        entries = {}
        for family in FAMILIES:
            evidence = []
            for a,b in s.blocks(self.lo,end,.55,4):
                t = rows(a,b)
                c,w,m = self.a_select(a-GAP,family)
                x = self.components(t,dict(C=c,W=w,A=m,issue_from=a))
                evidence.append(s.compare(t,self.y,x[:,:3].sum(1),x.sum(1),a-GAP,m["spec"] if m else None))
            v = s.vote(evidence)
            v["mean_child_mse"] = float(np.mean([b["child_mse"] for b in evidence]))
            entries[family] = v
        passing = [f for f in FAMILIES if entries[f]["admitted"]]
        chosen = min(passing,key=lambda f:entries[f]["mean_child_mse"]) if passing else None
        if chosen is None:
            c,w,_ = self.w_route(end)
            return c,w,None,dict(admitted=False,selected_family=None,families=entries)
        c,w,m = self.a_select(end,chosen)
        return c,w,m,dict(admitted=m is not None,selected_family=chosen,families=entries)

    def train(self,end):
        return self.base_model(end)


def rod1_k_controls(geo,pipe,data,k,lo,hi):
    entity = "CZ_Rod1"
    p,q = geo["parent"]
    train = rows(p,q)
    frame = h.frame(entity,p,q,"K_Dx")
    data_y = np.zeros(q); data_y[D:] = data[D:q,0]-data[:q-D,0]
    original = s.source_model(entity)["K"]["natural"]
    i = [rep["channel"] for rep in k["representations"]].index("joint_lift")
    rep = k["representations"][i]
    x,_ = pipe._values(frame,rep)
    spec = rep["family"],int(rep["m_x"]),tuple(rep["lambdas"])
    results = {}
    for mode,tuned in itertools.product(("difference","curvature","zero"),(False,True)):
        if tuned:
            result, audit = choose_k(pipe,x,data_y,p,q,"natural",binned=True,mode=mode)
        else:
            surf = Surface(pipe,x,train,data_y[train],None,spec[1],mode)
            c = surf.solve(spec,scaled=False)
            result = dict(view="natural",contract=c,support=surf.limits)
            audit = dict(spec=spec,knots=c["basis"]["knots"],fit_count=len(train))
            audit["linear_direction_penalty"] = float(surf.curve[0,0])
            audit["penalty_eigenvalues"] = np.linalg.eigvalsh(surf.curve).tolist()
        models = [value if j!=i else [result] for j,value in enumerate(original)]
        ctx = g.context(entity,pipe,data,k,models,hi)
        chain = g.combo(pipe,ctx,lo,hi,"natural_guard_robust")
        results[f"{mode}_{'tuned' if tuned else 'fixed'}"] = dict(models=models,chain=chain,
                 audit=audit,reference_view=i)
    return results


def fit(entity):
    bound()
    geo,pipe,data,_,stages = inputs(entity)
    lo,hi = geo["calibration"]
    k = stages["K"]
    models,k_audit = fit_k(geo,pipe,data,k)
    print("K",entity,[(v["channel"],v["view"],v["near"]) for v in k_audit],flush=True)
    ctx = g.context(entity,pipe,data,k,models,hi)
    engine = Engine(pipe,ctx,lo)
    outer = []
    for a,b in s.blocks(lo,hi,.70,4):
        end = a-EMBARGO
        model = engine.train(end)
        t = rows(a,b)
        p = engine.predict(t,model)
        outer.append(dict(rows=[a,b],fit_end=end,gates=model["gates"],
                          metrics={k:score(data[t,0],z) for k,z in p.items()}))
    model = engine.train(hi)
    control = rod1_k_controls(geo,pipe,data,k,lo,hi) if entity=="CZ_Rod1" else {}
    path = ROOT/entity/"MODEL.json"
    path.parent.mkdir(exist_ok=True)
    write(path,dict(entity=entity,K=models,K_audit=k_audit,chain=model,outer=outer,rod1_control=control))
    write(path.with_name("FIT.json"),dict(sha256=sha(path),development_accessed=False))
    print("FIT",entity,{name:item.get("selected_family",item["admitted"]) for name,item in model["gates"].items()},flush=True)


def freeze():
    bound()
    hashes = {}
    for e in CASES:
        path = ROOT/e/"MODEL.json"
        hashes[e] = sha(path)
        assert hashes[e]==read(path.with_name("FIT.json"))["sha256"]
    write(ROOT/"ALL_FROZEN.json",dict(models=hashes,registration=sha(ROOT/"REGISTRATION.json")))


def frozen(entity):
    bound()
    path = ROOT/entity/"MODEL.json"
    lock = read(ROOT/"ALL_FROZEN.json")
    assert lock["models"][entity]==sha(path) and lock["registration"]==sha(ROOT/"REGISTRATION.json")
    return read(path)


def evaluate(entity):
    m = frozen(entity)
    geo,pipe,data,rebind,stages = inputs(entity,development=True)
    k = stages["K"]
    lo,hi = geo["development"]
    train_lo = geo["calibration"][0]
    t = rows(lo,hi)
    ctx = g.context(entity,pipe,data,k,m["K"],hi)
    chain = m["chain"]
    engine = Engine(pipe,ctx,train_lo)
    p = engine.predict(t,chain)
    accessor = pipe.accessor
    cuts = [int(t[13]),int(t[len(t)//2]),int(t[-1])]
    for cut in cuts:
        alt = poisoned(data,cut)
        pipe.accessor = rebind(alt)
        cc = g.context(entity,pipe,alt,k,m["K"],cut+1)
        replay = Engine(pipe,cc,train_lo).predict(t[t<=cut],chain)
        for name,z in replay.items():
            np.testing.assert_allclose(p[name][:len(z)],z,atol=5e-11,rtol=0.)
    pipe.accessor = accessor
    controls = {}
    for name,control in m["rod1_control"].items():
        cc = g.context(entity,pipe,data,k,control["models"],hi)
        model = control["chain"]
        q = g.predict(pipe,cc,"natural_guard_robust",model,t,model["upstream_available_from"])
        i = control["reference_view"]
        c = model["C"]
        rms = None
        if c is not None:
            contribution = g.c_features(cc,t,c["mean"],c["scale"],True)[:,i]*c["coef"][i]
            rms = float(np.sqrt(np.mean(contribution**2)))
        controls[name] = dict(audit=control["audit"],metrics={s:score(data[t,0],v) for s,v in q.items()},
                              joint_lift_C_contribution_RMS=rms)
    metrics = {name:score(data[t,0],z) for name,z in p.items()}
    trust = {name:dict(mean=float(v.mean()),zero_rows=int((v==0.).sum()))
             for name,v in zip(ctx["names"],ctx["g"][t].T)}
    write(ROOT/entity/"RESULT.json",dict(entity=entity,N=len(t),metrics=metrics,K_audit=m["K_audit"],
          gates=chain["gates"],outer=m["outer"],causal_prefix_cuts=cuts,rod1_control=controls,trust=trust,
          routes={s:chain[s]["spec"] if chain[s] else None for s in ("C","W","A")}))
    print("SCORED",entity,metrics["KCWA"]["RMSE"],flush=True)


if __name__=="__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action",choices=("register","fit","freeze","evaluate"))
    cli.add_argument("entity",nargs="?",choices=CASES)
    args = cli.parse_args()
    if args.action in ("fit","evaluate"):
        if args.entity is None:
            cli.error("an entity is required")
        {"fit":fit,"evaluate":evaluate}[args.action](args.entity)
    else:
        {"register":register,"freeze":freeze}[args.action]()
