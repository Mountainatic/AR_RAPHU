"""Replay frozen-policy SRU H2S baselines under matched sensor stress."""

import argparse
import importlib
import json
from pathlib import Path
import pickle
import sys
import warnings

import numpy as np


SEEDS = (0,1,2,3,4)


def archived(archive):
    sys.path.insert(0,str(archive))
    return importlib.import_module("tf_sweep"),importlib.import_module("e1"),importlib.import_module("linear")


def fit_recipe(archive, train, output, model, seed):
    from laguerre_screen import digest,save_json
    tf,e1,linear = archived(archive)
    with np.load(train/"train.npz",allow_pickle=False) as p:
        values = p["values"]
    q = tf.Q
    origins = q.train_origins()
    out = output/model
    out.mkdir(parents=True,exist_ok=True)
    if model == "Transformer":
        tf.ARCH = archive.parent.parent/"PRISM_SRU_MATCHED_WINDOWS_REPEAT_20261025/baseline_models.py"
        tf.c.data = lambda _:values
        tf.init()
        f = json.loads((tf.ROOT/"FROZEN.json").read_text())
        config = tuple(f["selected"]["config"])
        path = out/f"seed_{seed}.pt"
        temp = path.with_suffix(".tmp")
        tf.torch.save(tf.fit(config,seed,None,f["epochs"]),temp)
        temp.replace(path)
    else:
        f = json.loads((e1.ROOT/q.name/"LightGBM/FROZEN.json").read_text())
        recipe,structure,rounds,penalty = f["selected"]["config"]
        config = (recipe,tuple(structure),rounds,penalty)
        x = linear.design(q,values,origins,"ARX")
        y,_ = q.target(values,origins,recipe)
        estimator = e1.head(config,seed).fit(x,y)
        path = out/f"seed_{seed}.txt"
        temp = path.with_suffix(".tmp")
        estimator.booster_.save_model(str(temp))
        temp.replace(path)
    hash_ = digest(path)
    reference = f["files"][path.name]
    save_json(out/f"seed_{seed}.json",dict(model=model,seed=seed,config=config,
        original_sha256=reference,sha256=hash_,byte_exact=hash_==reference))
    print(model,seed,"original checksum matched",hash_==reference,flush=True)


def predictors(archive, paired, refit):
    from laguerre_screen import digest
    tf,e1,linear = archived(archive)
    q = tf.Q
    tf.ARCH = archive.parent.parent/"PRISM_SRU_MATCHED_WINDOWS_REPEAT_20261025/baseline_models.py"
    tf.init()
    path = paired/"SRU_H2S/linear/MODELS.pkl"
    frozen = json.loads((paired/"LOCK.json").read_text())
    if digest(path) != frozen["baselines"]:
        raise ValueError("Linear baseline receipt changed")
    from sklearn.exceptions import InconsistentVersionWarning
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always",InconsistentVersionWarning)
        models = pickle.loads(path.read_bytes())
    version_warnings = sorted({str(x.message) for x in seen if isinstance(x.message,InconsistentVersionWarning)})
    bank = {}
    for model in ("LightGBM","Transformer"):
        folder = refit/model
        models_list = []
        for i in SEEDS:
            note = json.loads((folder/f"seed_{i}.json").read_text())
            p = folder/f"seed_{i}.{('pt' if model=='Transformer' else 'txt')}"
            if note["sha256"]!=digest(p):
                raise ValueError("Baseline weight changed")
            if model=="LightGBM":
                models_list.append((e1.lgb.Booster(model_file=str(p)),note))
            else:
                record = tf.torch.load(p,map_location="cpu",weights_only=False)
                net = tf.model_class()("Transformer",6,record["config"][0],q.L).eval()
                net.load_state_dict(record["weight"])
                models_list.append(((net,record),note))
        bank[model] = models_list

    def run(values, rows):
        offset = values[rows-q.D,0]
        out = {}
        for kind in ("ARX","ElasticNet"):
            features = linear.design(q,values,rows,kind)
            shots = []
            for item in models[kind]:
                m,center,scale = item["model"]
                y = m.predict(item["scale"].transform(features))*scale+center
                shots.append(y+(offset if item["recipe"]=="delta" else 0.))
            out[kind] = np.array(shots)
        x = linear.design(q,values,rows,"ARX")
        out["LightGBM"] = np.array([m.predict(x,num_threads=1)+(offset if n["config"][0]=="delta" else 0.)
                                   for m,n in bank["LightGBM"]])
        x = tf.features(values,rows)
        shots = []
        for (net,r),_ in bank["Transformer"]:
            center,scale,mean,sd = r["stats"]
            z = tf.torch.from_numpy((x-center)/scale)
            with tf.torch.inference_mode():
                projected = tf.torch.cat([net(v) for v in z.split(64)]).numpy()
            shots.append(projected.astype(float)*sd+mean+(offset if r["config"][2]=="delta" else 0.))
        out["Transformer"] = np.array(shots)
        return out

    receipts = {arm:[n for _,n in entries] for arm,entries in bank.items()}
    receipts["linear_sha256"] = digest(path)
    receipts["linear_version_warnings"] = version_warnings
    return run,receipts


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive",type=Path)
    p.add_argument("train",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("arm",choices=("Transformer","LightGBM"))
    p.add_argument("seed",type=int,choices=SEEDS)
    a=p.parse_args()
    fit_recipe(a.archive,a.train,a.output,a.arm,a.seed)


if __name__=="__main__":
    main()

