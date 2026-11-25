"""CPU, frozen FP32 replay of the five registered expanded Transformer seeds."""

import argparse
import importlib.util
import json
import os
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import torch


L,D=128,4


def architecture(path):
    spec=importlib.util.spec_from_file_location("frozen_baseline_models",path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def features(data,issue,stats):
    t=np.atleast_1d(issue)
    j=t[:,None]-np.arange(L-1,-1,-1)
    x=np.concatenate((data[j,1:],data[j-D,0,None]),axis=2).astype(np.float32)
    return torch.from_numpy((x-stats[0])/stats[1])


def forecast(net,stats,recipe,data,t):
    b=features(data,t,stats)
    q=net(b).numpy().astype(np.float64)*stats[3]+stats[2]
    if recipe=="delta":
        q+=data[np.asarray(t)-D,0]
    return q


def run(args):
    if args.timing:
        os.sched_setaffinity(0,{args.core})
    torch.set_num_threads(1 if args.timing else 2)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    with np.load(args.data) as v:
        data=v["data"]
        t=v["issues"]
        cut=int(v["cut"]) if "cut" in v else None
    manifest=json.loads(Path(args.manifest).read_text())
    arch=architecture(next(f for f in manifest if f.endswith("baseline_models.py")))
    seed_files={int(Path(f).stem.rsplit("_",1)[-1]):f for f in manifest if f.endswith(".pt")}
    assert tuple(sorted(seed_files))==tuple(range(5))
    predictions={}; timing={}; causal={}
    if cut is not None and not args.timing:
        evil=data.copy()
        evil[cut-D+1:,0]+=1234.
        evil[cut+1:,1:]-=1200.
        prefix=t[t<=cut]
    with torch.inference_mode():
        for seed,path in sorted(seed_files.items()):
            weights=torch.load(path,map_location="cpu",weights_only=False)
            assert weights["seed"]==seed
            width,_,recipe=weights["config"]
            net=arch.build("Transformer",6,width,L).eval()
            net.load_state_dict(weights["state"])
            stats=weights["stats"]
            if args.timing:
                # Batch size one, each call builds the legal input and writes a scalar.
                sample=t
                scratch=np.empty(len(sample))
                for j in sample[:16]:
                    forecast(net,stats,recipe,data,[j])
                reps=[]
                for _ in range(7):
                    start=perf_counter_ns()
                    for i,j in enumerate(sample):
                        scratch[i]=forecast(net,stats,recipe,data,[j])[0]
                    reps.append((perf_counter_ns()-start)/len(sample))
                timing[str(seed)]=dict(median_ns=float(np.median(reps)),repeats_ns=reps,rows=len(sample),width=width)
            else:
                p=np.concatenate([forecast(net,stats,recipe,data,j) for j in np.array_split(t,(len(t)+63)//64)])
                predictions[f"seed_{seed}"]=p
                if cut is not None:
                    q=np.concatenate([forecast(net,stats,recipe,evil,j)
                                      for j in np.array_split(prefix,(len(prefix)+63)//64)])
                    np.testing.assert_allclose(q,p[:len(q)],atol=5e-7,rtol=0)
                    causal[str(seed)]=float(np.max(np.abs(q-p[:len(q)])))
            print("TRANSFORMER",seed,flush=True)
    if args.timing:
        Path(args.output).write_text(json.dumps(dict(core=args.core,backend=torch.__version__,
            threads=torch.get_num_threads(),batch=1,seeds=timing),indent=2)+"\n")
    else:
        np.savez(args.output,**predictions)
        Path(args.output).with_suffix(".json").write_text(json.dumps(dict(causal=causal,torch=torch.__version__),indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data")
    parser.add_argument("manifest")
    parser.add_argument("output")
    parser.add_argument("--timing",action="store_true")
    parser.add_argument("--core",type=int,default=64)
    run(parser.parse_args())
