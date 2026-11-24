"""Extend the matched neural baselines to five individually scored seeds."""

from concurrent.futures import ThreadPoolExecutor
import runpy
import subprocess
import sys

from common import (ENTITIES, NEURAL, ROOT, SEEDS, development, matched, np, poison, read,
                    registration, run_result, sha, train_rows, write)

DEST = ROOT / "neural"


def folder(entity,name):
    return DEST/entity/name


def register():
    value = dict(common=registration(), source=sha(matched.REFERENCE/"neural.py"),
                 wrapper=sha(ROOT/"neural.py"), common_source=sha(ROOT/"common.py"),
                 policy="Use train-selected configuration and epoch count. Verify and replay seeds 0–2, refit seeds 3–4, then score five separate predictions.")
    dst = DEST/"REGISTRATION.json"
    if dst.exists():
        assert read(dst) == matched.json.loads(matched.json.dumps(value))
    else:
        write(dst,value)


def bound():
    reg=read(DEST/"REGISTRATION.json")
    assert reg["common"]==matched.json.loads(matched.json.dumps(registration()))
    assert reg["source"]==sha(matched.REFERENCE/"neural.py") and reg["wrapper"]==sha(ROOT/"neural.py")
    assert reg["common_source"]==sha(ROOT/"common.py")


def source(entity):
    sys.path.insert(0,str(matched.model_source(entity)))
    import baseline_models
    assert baseline_models.ROOT.resolve()==matched.model_source(entity).resolve()
    return runpy.run_path(str(matched.REFERENCE/"neural.py"))


def fit(entity,name):
    bound()
    import torch
    code=source(entity)
    a,t=train_rows(entity)
    prior=matched.ROOT/entity/"neural"/name
    original=read(prior/"FROZEN.json")
    assert original["last_training_target"]==int(t[-1])
    dest=folder(entity,name)
    dest.mkdir(parents=True,exist_ok=True)
    seal=dest/"FROZEN.json"
    if seal.exists():
        verify(entity,name)
        return
    files={}
    for seed in SEEDS:
        filename=f"seed_{seed}.pt"
        if seed<3:
            p=prior/filename
            assert sha(p)==original["files"][filename]
        else:
            p=dest/filename
            job=code["fit"](entity,name,a,t,None,original["selected"]["config"],seed,original["epochs"])
            tmp=p.with_suffix(".tmp")
            torch.save(job,tmp)
            tmp.replace(p)
        files[str(p)]=sha(p)
    write(seal,dict(original=sha(prior/"FROZEN.json"),config=original["selected"]["config"],
                    epochs=original["epochs"],files=files))
    print("FROZEN_NEURAL",entity,name,flush=True)


def verify(entity,name):
    seal=read(folder(entity,name)/"FROZEN.json")
    assert len(seal["files"])==len(SEEDS)
    assert seal["original"]==sha(matched.ROOT/entity/"neural"/name/"FROZEN.json")
    assert all(sha(ROOT/p)==digest for p,digest in seal["files"].items())
    return seal


def evaluate(entity):
    bound()
    import torch
    code=source(entity)
    a,t=development(entity)
    poison_a, prefix=poison(a,t)
    seal=read(DEST/"ALL_FROZEN.json")
    assert seal["registration"]==sha(DEST/"REGISTRATION.json")
    with np.load(matched.ROOT/entity/"neural_evaluation.npz") as earlier:
        np.testing.assert_array_equal(earlier["target_index"],t)
        for name in NEURAL:
            dest=folder(entity,name)
            assert sha(dest/"FROZEN.json")==seal["receipts"][str(dest.relative_to(DEST))]
            record=verify(entity,name)
            predictions=[]
            for seed,filename in zip(SEEDS,record["files"],strict=True):
                obj=torch.load(filename,map_location="cpu",weights_only=False)
                assert list(obj["config"])==record["config"] and obj["seed"]==seed and obj["epochs"]==record["epochs"]
                width,_,recipe=obj["config"]
                model=code["predictor"](entity,name,width)
                model.load_state_dict(obj["state"])
                x,_,anchor,_=code["batch"](a,t,recipe,obj["stats"])
                p=code["predict"](model,x,anchor,obj["stats"])
                z,_,anch,_=code["batch"](poison_a,prefix,recipe,obj["stats"])
                np.testing.assert_allclose(code["predict"](model,z,anch,obj["stats"]),p[:len(prefix)],rtol=0,atol=1e-6)
                if seed<3:
                    np.testing.assert_array_equal(p,earlier[f"{name}_{seed}"])
                predictions.append(p)
            result=run_result(entity,name,t,predictions,dict(kind="neural", config=record["config"],
                    original_seed_replay_exact=True, added_refit_seeds=[3,4], causal_checks=len(SEEDS)))
            write(dest/"RESULT.json",result)
            print("SCORED_NEURAL",entity,name,flush=True)


def launch(action,entity,name=None):
    command=[sys.executable,"-B","-u",__file__,action,entity]
    if name:
        command.append(name)
    dest=folder(entity,name) if name else DEST/entity
    dest.mkdir(parents=True,exist_ok=True)
    with (dest/f"{action}.log").open("a",encoding="utf-8") as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)


def main():
    register()
    cases=[(e,n) for n in ("Transformer_expanded","Transformer_original","TimeXer","iTransformer","LSTM")
           for e in ("CZ_Rod2","CZ_Rod1","SRU_H2S")]
    with ThreadPoolExecutor(2) as pool:
        futures=[pool.submit(launch,"fit",e,n) for e,n in cases]
        for future in futures:
            future.result()
    receipts={}
    for e,n in cases:
        verify(e,n)
        dest=folder(e,n)
        receipts[str(dest.relative_to(DEST))]=sha(dest/"FROZEN.json")
    write(DEST/"ALL_FROZEN.json",dict(registration=sha(DEST/"REGISTRATION.json"),receipts=receipts))
    for e in ENTITIES:
        launch("evaluate",e)
        print("NEURAL_COMPLETE",e,flush=True)
    write(DEST/"COMPLETE.json",dict(result_count=len(cases),freeze=sha(DEST/"ALL_FROZEN.json")))


if __name__=="__main__":
    if len(sys.argv)==1:
        main()
    else:
        {"fit":fit,"evaluate":evaluate}[sys.argv[1]](*sys.argv[2:])
