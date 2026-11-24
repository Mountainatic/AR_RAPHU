"""Repeat the matched deterministic AR/ARX fits and replay the frozen PRISM points."""

import pickle
import runpy
import sys

from common import (CORE, OLD, PRIOR, PRISM, ROOT, SEEDS, development, matched, np, poison,
                    read, registration, run_result, sha, train_rows, write)

DEST = ROOT / "core"


def run(entity):
    origin=registration()
    sys.path.insert(0,str(OLD))
    import suite as old
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    old.verify_freeze(entity)
    train, fit=train_rows(entity)
    config_path=matched.ROOT/entity/"linear.json"
    selection=read(config_path)["selected"]
    linear_design=runpy.run_path(str(matched.REFERENCE/"cpu.py"))["linear_design"]
    dest=DEST/entity
    dest.mkdir(parents=True,exist_ok=True)
    params={}
    for name in ("AR","ARX"):
        alpha,recipe,_=selection[name]
        x=linear_design(train,fit,name)
        y=train[fit,0]-(train[fit-matched.D,0] if recipe=="delta" else 0.)
        models=[]
        for seed in SEEDS:
            sc=StandardScaler().fit(x)
            head=Ridge(alpha=alpha,solver="cholesky").fit(sc.transform(x),y)
            models.append(dict(seed=seed,recipe=recipe,scaler=sc,model=head))
        params[name]=models
    model_file=dest/"LINEAR_MODELS.pkl"
    tmp=model_file.with_suffix(".tmp")
    tmp.write_bytes(pickle.dumps(params,protocol=5))
    tmp.replace(model_file)
    del params
    write(dest/"FROZEN.json",dict(models_sha256=sha(model_file),reference_sha256=sha(config_path),
                                protocol=origin,source=sha(ROOT/"core_runs.py")))

    seal=read(dest/"FROZEN.json")
    assert seal["models_sha256"]==sha(model_file)
    params=pickle.loads(model_file.read_bytes())
    a,t=development(entity)
    contaminated,prefix=poison(a,t)
    by_model={k:[] for k in (*CORE,*PRISM)}
    with np.load(matched.ROOT/entity/"linear_evaluation.npz") as ref:
        np.testing.assert_array_equal(ref["target_index"],t)
        for seed in SEEDS:
            by_model["Persistence"].append(a[t-matched.D,0])
            np.testing.assert_array_equal(by_model["Persistence"][-1],ref["Persistence"])
            for name in ("AR","ARX"):
                model=params[name][seed]
                assert model["seed"]==seed
                z=linear_design(a,t,name)
                np.testing.assert_array_equal(linear_design(contaminated,prefix,name),z[:len(prefix)])
                p=model["model"].predict(model["scaler"].transform(z))
                if model["recipe"]=="delta":
                    p+=a[t-matched.D,0]
                np.testing.assert_allclose(p,ref[name],rtol=0,atol=1e-10)
                if seed:
                    np.testing.assert_array_equal(p,by_model[name][0])
                by_model[name].append(p)

    rod=entity!=old.sru.ENTITY
    record=old.registration()
    pipe, data, _=old.pipeline(entity,record,development=True)
    np.testing.assert_array_equal(a,data)
    old_models=read(old.path(entity,"FROZEN_MODEL.json"))
    hybrid_models=read(PRIOR/entity/"FROZEN_MODEL.json")
    start,stop=int(t[0]),int(t[-1])+1
    frame=old.hybrid.frame(entity,start,stop,"development",y=a[:,0])
    anchor=a[t-matched.D,0]
    refs=dict(zip(PRISM,("kcw","v06_kcwa","kcw_fixed_a","persistence_qc32","kcw_qc32"),strict=True))
    with np.load(old.path(entity,"predictions_private.npz")) as ref:
        np.testing.assert_array_equal(ref["target_index"],t)
        for _ in SEEDS:
            p,q,_=old.prediction(pipe,old_models["original_kcwa"],frame,anchor)
            f=old.fixed_prediction(entity,pipe,a,old_models,start,stop)
            hybrid,_=(old.hybrid.predictions(entity,a,pipe,hybrid_models,start,stop) if rod
                      else old.sru.predict(a,pipe,hybrid_models,start,stop))
            for name, prediction in zip(PRISM,(q,p,f,hybrid["persistence_qc32"],hybrid["kcw_qc32"]),strict=True):
                np.testing.assert_array_equal(prediction,ref[refs[name]])
                by_model[name].append(prediction)

    for name,predictions in by_model.items():
        kind="matched_linear" if name in CORE else "prism_reference"
        method="five deterministic full refits" if name in ("AR","ARX") else "five deterministic forward replays"
        row=run_result(entity,name,t,predictions,dict(kind=kind,repeat_kind=method,deterministic=True))
        write(dest/f"{name}.json",row)
    write(dest/"COMPLETE.json",dict(records=len(by_model),freeze=sha(dest/"FROZEN.json")))
    print("CORE_COMPLETE",entity,flush=True)


if __name__=="__main__":
    run(sys.argv[1])
