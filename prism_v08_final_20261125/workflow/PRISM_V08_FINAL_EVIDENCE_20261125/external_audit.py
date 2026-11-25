"""Assess whether the requested Debutanizer archive qualifies as zero-contact evidence."""

from tail_replay import CASES, EXPECTED, ROOT, lock, read, sha, write


def main():
    lock()
    base=ROOT.parent/"PRISM_CZ_D0_DEBUTANIZER_D4_20261122"
    early=base/"DEBUT"/"REGISTRATION.json"
    archive=ROOT.parent/"PRISM_DEBUTANIZER_FULL_BASELINES_20261122"
    later=archive/"REGISTRATION.json"
    prior=archive/"RESULTS.json"
    receipt=read(early)
    benchmark=read(prior)
    assert receipt["data_sha256"]==read(later)["data_sha256"]
    assert "Debutanizer" in receipt["rods"] and benchmark["N"]>0
    assert "Debutanizer" not in CASES
    result=dict(release=EXPECTED,candidate="Debutanizer",status="EXCLUDED_FROM_ZERO_CONTACT",
         evidence={p.name if p!=early else "DEBUT_original_registration":dict(path=str(p),sha256=sha(p))
                   for p in (early,later,prior)},
         earlier_protocol={k:benchmark[k] for k in ("H","D","L")},
         earlier_scored_origins=benchmark["targets"], earlier_N=benchmark["N"],
         prior_use="historical train, architecture comparison and baseline scoring; same registered source hashes",
         eligible_frozen_entry_entities=list(CASES),v08_external_fit_or_score_executed=False,
         raw_external_observations_read_by_this_audit=False)
    write(ROOT/"EXTERNAL_PROVENANCE.json",result)


if __name__=="__main__":
    main()
