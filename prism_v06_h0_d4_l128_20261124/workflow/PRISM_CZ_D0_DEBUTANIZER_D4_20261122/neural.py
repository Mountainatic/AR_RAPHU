"""Train-only model selection and frozen evaluation of direct CZ neural baselines."""
import itertools
import statistics
import sys
import time

from protocol import (CACHE, CHANNELS, D, H, L, NAMES, ROOT, ensure_bound, folds, issues,
                      np, read, sequence, sha, values, write)
import torch

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.set_float32_matmul_precision("highest")
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

ORIGINAL = [(32,.001,t) for t in ("level","delta")] + [(64,.0003,t) for t in ("level","delta")]
EXPANDED = list(itertools.product((32,64,128),(.0001,.0003,.001),("level","delta")))
FINAL_SEEDS = (0,1,2)
BATCH = 256


def predictor(rod, name, width):
    sys.path.insert(0, str(CACHE / ("CZ_Rod1" if rod == "Debutanizer" else rod)))
    from baseline_models import build
    kind = "Transformer" if name.startswith("Transformer") else name
    return build(kind, CHANNELS, width, L).cuda()


def batch(a, o, recipe, stats=None):
    x = sequence(a, o).astype(np.float32)
    anchor = a[o-D,0] if recipe == "delta" else np.zeros(len(o))
    target = a[o+H,0] - anchor
    if stats is None:
        stats = (x.mean((0,1), dtype=np.float64).astype(np.float32),
                 np.maximum(x.std((0,1), dtype=np.float64), 1e-6).astype(np.float32),
                 float(target.mean()), max(float(target.std()),1e-6))
    x = torch.as_tensor((x-stats[0])/stats[1], device="cuda")
    target = torch.as_tensor(((target-stats[2])/stats[3]).astype(np.float32), device="cuda")
    return x, target, anchor, stats


def predict(model, x, anchor, stats):
    model.eval()
    with torch.no_grad():
        pred = torch.cat([model(chunk) for chunk in x.split(512)]).cpu().numpy().astype(np.float64)
    return pred*stats[3] + stats[2] + anchor


def fit(rod, name, a, train, test, config, seed, epochs):
    width, lr, recipe = config
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = predictor(rod, name, width)
    x, y, _, stats = batch(a, train, recipe)
    if test is not None:
        vx, _, anchor, _ = batch(a,test,recipe,stats)
    opt = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-4)
    best, best_epoch = float("inf"), 0
    scores = []
    for epoch in range(1, epochs+1):
        model.train()
        for ix in torch.randperm(len(x),device="cuda").split(BATCH):
            opt.zero_grad(set_to_none=True)
            loss = ((model(x[ix])-y[ix])**2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step()
        if test is not None:
            risk = float(np.mean((a[test+H,0]-predict(model,vx,anchor,stats))**2))
            assert np.isfinite(risk)
            scores.append(risk)
            if risk < best:
                best, best_epoch = risk, epoch
            if epoch-best_epoch >= 6:
                break
    if test is not None:
        return dict(MSE=best,N=len(test),epoch=best_epoch, trace=scores,
                    fit_latest_target=int(train[-1]+H), score_first_issue=int(test[0]))
    return dict(state={k:v.detach().cpu() for k,v in model.state_dict().items()},
                stats=stats, config=config, seed=seed, epochs=epochs,
                parameters=sum(p.numel() for p in model.parameters()))


def train(rod, name):
    ensure_bound("neural.py")
    o = issues(rod)
    a = values(rod)
    dest = ROOT / rod / "neural" / name
    receipt = dest / "FROZEN.json"
    if receipt.exists():
        assert all(sha(dest/k) == v for k,v in read(receipt)["files"].items())
        return
    clock = time.monotonic()

    def select(grid, stage, cap):
        ledger = []
        for i, spec in enumerate(grid):
            score = dest / f"{stage}_{i}.json"
            if score.exists():
                obj = read(score)
                assert obj["config"] == list(spec)
            else:
                parts = [fit(rod,name,a,x,v,spec,20261023,cap) for x,v in folds(o)]
                risk = np.average([p["MSE"] for p in parts],weights=[p["N"] for p in parts])
                obj = dict(config=list(spec),MSE=float(risk),folds=parts,order=i)
                write(score,obj)
            ledger.append(obj)
            print("CV",rod,name,stage,i,obj["MSE"],flush=True)
        return sorted(ledger,key=lambda x:(x["MSE"],x["order"]))

    if name == "Transformer_expanded":
        screen = select(EXPANDED,"screen",8)
        grid = [row["config"] for row in screen[:3]]
        grid.extend(spec for spec in ORIGINAL if list(spec) not in grid and spec not in grid)
    else:
        grid = ORIGINAL
    selected = select(grid,"refine",24)[0]
    epochs = round(statistics.median(row["epoch"] for row in selected["folds"]))
    files = {}
    for seed in FINAL_SEEDS:
        path = dest / f"seed_{seed}.pt"
        tmp = path.with_suffix(".tmp")
        torch.save(fit(rod,name,a,o,None,selected["config"],seed,epochs),tmp)
        tmp.replace(path)
        files[path.name] = sha(path)
        print("REFIT",rod,name,seed,epochs,flush=True)
    write(receipt,dict(selected=selected,epochs=epochs,files=files,
                        last_training_target=int(o[-1]+H),runtime_seconds=time.monotonic()-clock))
    print("FROZEN",rod,name,flush=True)


def eval_models(rod):
    ensure_bound("neural.py")
    o = issues(rod,True)
    a = values(rod,True)
    result = {}
    for name in NAMES:
        dest = ROOT / rod / "neural" / name
        for filename, expected in read(dest/"FROZEN.json")["files"].items():
            assert sha(dest/filename) == expected
            obj = torch.load(dest/filename,map_location="cpu",weights_only=False)
            width, _, recipe = obj["config"]
            model = predictor(rod,name,width)
            model.load_state_dict(obj["state"])
            x,_,anchor,_ = batch(a,o,recipe,obj["stats"])
            p = predict(model,x,anchor,obj["stats"])
            i = 73
            poison = a.copy()
            poison[o[i]-D+1:,0] += 1000
            poison[o[i]+1:,1:] -= 1000
            xx,_,aa,_ = batch(poison,o[:i+1],recipe,obj["stats"])
            np.testing.assert_allclose(p[:i+1],predict(model,xx,aa,obj["stats"]),atol=1e-6,rtol=0)
            result[f"{name}_{obj['seed']}"] = p
            del model
        result[name] = np.stack([result[f"{name}_{seed}"] for seed in FINAL_SEEDS]).mean(0)
    np.savez(ROOT/rod/"neural_evaluation.npz", origin=o,target_index=o+H,**result)


if __name__ == "__main__":
    mode,rod,*rest=sys.argv[1:]
    if mode == "train":
        train(rod,rest[0])
    else:
        eval_models(rod)
