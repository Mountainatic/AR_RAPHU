"""Bind final development, baseline receipts and tail authority without opening observations."""

from pathlib import Path

from v08 import CASES, ROOT, h, read, register, sealed, sha, v, write


BASE=ROOT.parent
REPEAT=BASE/"PRISM_V06_CLASSIC_BASELINES_REPEAT_20261124"
PRIOR=BASE/"PRISM_V071_REVIEW_20261125"/"candidate"
AUTHORITY=BASE/"PRISM_V06_SRU_CZ_SCALE_SEARCH_20261020"/"DATASETS.json"
PROVENANCE=BASE/"PRISM_COMPLETION_20261119"/"SPLIT_PROVENANCE_AUDIT.json"


def bound_files(paths):
    return {str(p):sha(p) for p in paths}


def baselines(e):
    matched=v.legacy.baseline
    fixed=read(matched.ROOT/"ALL_FROZEN.json")
    linear=matched.ROOT/e/"linear.json"
    path=linear.with_suffix(".pkl")
    assert sha(linear)==fixed["receipts"][str(linear.relative_to(matched.ROOT))]
    assert sha(path)==read(linear)["sha256"]
    root=REPEAT/("classical" if e=="SRU_H2S" else "elastic_gram")/e/"ElasticNet"
    elastic=root/"FROZEN.json"
    el={str(root/key):expected for key,expected in read(elastic)["files"].items()}
    ne=REPEAT/"neural"
    transformer=ne/e/"Transformer_expanded"/"FROZEN.json"
    ne_lock=read(ne/"ALL_FROZEN.json")
    assert sha(transformer)==ne_lock["receipts"][str(transformer.parent.relative_to(ne))]
    tf=read(transformer)["files"]
    assert sorted(Path(p).stem for p in el)==[f"seed_{i}" for i in range(5)]
    assert sorted(Path(p).stem for p in tf)==[f"seed_{i}" for i in range(5)]
    assert all(sha(Path(p))==digest for p,digest in {**el,**tf}.items())
    architecture=matched.model_source(e)/"baseline_models.py"
    assert sha(architecture)==read(matched.ROOT/"REGISTRATION.json")["source_sha256"][str(architecture)]
    return dict(linear=bound_files([linear,path]),elastic_net={**bound_files([elastic]),**el},
                transformer={**bound_files([transformer,architecture]),**tf},
                persistence="frozen rule y[t-4]")


def main():
    register()
    d=read(AUTHORITY)
    registry={r["id"]:r for r in d["datasets"]}
    origin=read(PROVENANCE)
    simd=read(ROOT/"SIMD.json")
    assert simd["model_lock"]==sha(ROOT/"MODEL_LOCK.json")
    previous=read(PRIOR/"ALL_FROZEN.json")
    assert all(sha(PRIOR/e/"MODEL.json")==previous["models"][e] for e in CASES)
    guard={}; baseline={}; tail={}
    for e in CASES:
        sealed(e)
        new=read(ROOT/e/"DEVELOPMENT.json")["metrics"]["KCWA"]["RMSE"]
        record=read(PRIOR/e/"RESULT.json")
        old=record["results"][record["winner"]]["metrics"]["KCWA"]["RMSE"]
        guard[e]=dict(v071=old,v08=new,relative_change=(new-old)/old,non_degrading=new<=old)
        z=registry[e]
        lo,hi=z["validation_end"],z["total_rows"]
        assert [lo,hi]==origin[e]["candidate_tail"]
        if e=="SRU_H2S":
            source=AUTHORITY.parent/"raw"/"SRU_data.txt"
            expected=d["SRU"]["sha256"]
            history=dict(prior_scored_range=[lo,hi],
                         authority=sha(PROVENANCE.with_name("SRU_TAIL_RESULTS.json")),
                         prior_protocol="2026-11-19 D=8; same physical y rows")
        else:
            source=Path(d["CZ"]["workbook"])
            expected=d["CZ"]["sha256"]
            x=origin[e]
            history=dict(prior_scored_range=[x["prior_scored_origin_min"],x["prior_scored_origin_max"]+1],
                         prior_artifact_sha256=x["prior_sha256"],prior_scored_rows=x["prior_scored_rows_in_tail"])
        assert source.is_file()
        tail[e]=dict(rows=[lo,hi],target=z["target"],inputs=z["columns"],path=str(source),
                     archived_source_sha256=expected,historical_target_scoring=history)
        baseline[e]=baselines(e)
    dependencies=[p for root in (h.CANONICAL,h.CODE/"src"/"prism_benchmark")
                  for p in sorted(root.rglob("*")) if p.is_file() and p.suffix in (".py",".json")]
    release=dict(model_lock=sha(ROOT/"MODEL_LOCK.json"),formal_registration=sha(ROOT/"REGISTRATION.json"),
                 development=bound_files([ROOT/e/"DEVELOPMENT.json" for e in CASES]),
                 simd=sha(ROOT/"SIMD.json"),source=bound_files([ROOT/"v08.py",Path(__file__),*dependencies]),
                 frozen_baselines=baseline,tail_metadata=tail,tail_source_provenance=sha(PROVENANCE),
                 data_authority_receipt=sha(AUTHORITY),
                 development_review=guard,
                 candidate_nondegradation=all(x["non_degrading"] for x in guard.values()),
                 current_run_tail_values_read=False,
                 interpretation="historically scored tail rows; prospective independent lockbox is unavailable here")
    path=ROOT/"RELEASE_LOCK.json"
    if path.exists():
        assert read(path)==release
    else:
        write(path,release)
    print("RELEASE_LOCK",sha(path),flush=True)


if __name__=="__main__":
    main()
