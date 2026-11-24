"""Common five-run scoring over the frozen matched development protocol."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent / "PRISM_V06_KCW_QC32_H0_D4_L128_SUITE_20261124"
OLD = ROOT.parent / "PRISM_V06_KCWA_H0_D4_L128_BASELINES_20261124"
sys.path.insert(0,str(PRIOR / "baselines"))
import protocol as matched

np = matched.np
read, write, sha = matched.read, matched.write, matched.sha
ENTITIES = matched.RODS
SEEDS = (0,1,2,3,4)
METRICS = ("RMSE","MAE","MSE","R2")
CLASSICAL = ("PCR","PLS","ElasticNet","RBF_SVR","RandomForest","HistGradientBoosting")
CORE = ("Persistence","AR","ARX")
PRISM = ("v0.6_KCW","v0.6_KCWA","KCW_fixed_A","Persistence_QC32","KCW_QC32")
NEURAL = matched.NAMES


def registration():
    origin = matched.ensure_bound("suite.py")
    return dict(seeds=SEEDS, metrics=METRICS, protocol=origin["protocol"], rows=origin["rows"],
                prior_registration=sha(matched.ROOT/"REGISTRATION.json"),
                prior_baseline_freeze=sha(matched.ROOT/"ALL_FROZEN.json"),
                prior_prism_freeze=sha(OLD/"ALL_FROZEN.json"))


def train_rows(entity):
    a = matched.values(entity)
    t = matched.issues(entity)
    assert t[-1] == len(a)-matched.D
    return a,t


def development(entity):
    a = matched.values(entity,evaluate=True)
    t = matched.issues(entity,evaluate=True)
    assert (t[0],t[-1]+1)==matched.GEOMETRY[entity][1:]
    return a,t


def design(a,t,cutoff=matched.D):
    offsets = np.arange(matched.L)
    history = t[:,None]-offsets
    assert history.min() >= cutoff
    own = a[history-cutoff,0]
    process = a[history,1:].reshape(len(t),-1)
    return np.c_[own,process]


def target(a,t,recipe):
    if recipe not in ("level","delta"):
        raise ValueError(recipe)
    anchor = a[t-matched.D,0] if recipe=="delta" else np.zeros(len(t))
    return a[t,0]-anchor, anchor


def score(y,p):
    assert y.shape == p.shape and np.isfinite(p).all()
    errors = y-p
    mse = float(np.mean(errors**2))
    return dict(RMSE=float(np.sqrt(mse)), MAE=float(np.mean(np.abs(errors))), MSE=mse,
                R2=float(1-mse/np.var(y)))


def poison(a,t,i=73):
    cut = t[i]
    changed = a.copy()
    changed[cut-matched.D+1:,0] += 1234.
    changed[cut+1:,1:] -= 1200.
    return changed,t[:i+1]


def run_result(entity,model,t,predictions,metadata):
    a, expected = development(entity)
    np.testing.assert_array_equal(t,expected)
    assert len(predictions)==len(SEEDS)
    metrics = [dict(seed=seed, **score(a[t,0],prediction)) for seed,prediction in zip(SEEDS,predictions,strict=True)]
    return dict(entity=entity,model=model,N=len(t),development=[int(t[0]),int(t[-1])+1],
                runs=metrics, **metadata)
