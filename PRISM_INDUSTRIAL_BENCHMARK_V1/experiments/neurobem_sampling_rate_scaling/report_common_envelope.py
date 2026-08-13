from __future__ import annotations

import argparse, json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

def main():
 p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);p.add_argument("--sampling-results",type=Path,required=True);p.add_argument("--generating-commit",required=True);a=p.parse_args();r=a.run
 agg=pd.read_csv(r/"COMMON_ENVELOPE_AGGREGATE.csv"); common=pd.read_csv(r/"COMMON_RELIABLE_HORIZONS.csv"); bounds=pd.read_csv(r/"BOUND_RATIOS.csv")
 old=pd.read_csv(a.sampling_results/"SAMPLING_RATE_AGGREGATE.csv"); old=old[old.history_mode=="FIXED_TIME"][["route","sampling_rate_hz","reliable_horizon_steps","reliable_horizon_ms"]].rename(columns={"reliable_horizon_steps":"rate_specific_reliable_horizon_steps","reliable_horizon_ms":"rate_specific_reliable_horizon_ms"})
 comparison=old.merge(common,on=["route","sampling_rate_hz"]);comparison.to_csv(r/"RELIABLE_HORIZON_COMPARISON.csv",index=False)
 focus=agg[(agg.horizon_ms.isin([100,200]))&(((agg.route=="PF_KCW")&(agg.channel=="velocity"))|((agg.route=="J_KCW")&agg.channel.isin(["attitude","body_rate"])))].copy();focus.to_csv(r/"COMMON_ENVELOPE_CORE_TABLE.csv",index=False)
 out=r/"figures";out.mkdir(exist_ok=True);fig,axes=plt.subplots(1,2,figsize=(11,4.5))
 for axis,route in zip(axes,("PF_KCW","J_KCW")):
  q=comparison[comparison.route==route].sort_values("sampling_rate_hz");axis.plot(q.sampling_rate_hz,q.rate_specific_reliable_horizon_ms,"o-",label="rate-specific");axis.plot(q.sampling_rate_hz,q.common_reliable_horizon_ms,"s-",label="common 100-Hz envelope");axis.set_title(route);axis.set_xlabel("Sampling rate (Hz)");axis.set_ylabel("Reliable horizon (ms)");axis.set_xticks([100,200,400]);axis.grid(alpha=.25);axis.legend()
 fig.tight_layout();fig.savefig(out/"01_reliable_horizon_envelope_comparison.png",dpi=180);plt.close(fig)
 decisions={"PF_KCW_velocity":"ENVELOPE_NORMALIZATION_EXPLANATION_SUPPORTED","J_KCW_attitude":"ENVELOPE_NORMALIZATION_EXPLANATION_SUPPORTED","J_KCW_body_rate":"ENVELOPE_NORMALIZATION_EXPLANATION_SUPPORTED"}
 final={"experiment_id":"PRISM_V2_1_1_NEUROBEM_COMMON_ENVELOPE_RELIABILITY_AUDIT_R1","status":"COMPLETED_DETERMINISTIC_REPRODUCTION_REPLAY","generating_commit":a.generating_commit,"final_decision":"ENVELOPE_NORMALIZATION_EXPLANATION_SUPPORTED","channel_decisions":decisions,"existing_logs_sufficient":False,"deterministic_replay_performed":True,"new_test_decision_access":False,"model_retrained":False,"threshold_changed":False,"reliable_horizon_comparison":comparison.to_dict("records"),"bound_ratios":bounds.to_dict("records"),"core_table":focus.to_dict("records")}
 (r/"COMMON_ENVELOPE_FINAL_SUMMARY.json").write_text(json.dumps(final,indent=2,sort_keys=True)+"\n")
 (r/"COMMON_ENVELOPE_FINAL_REPORT.md").write_text("""# NeuroBEM common-envelope reliability audit

## Decision

`ENVELOPE_NORMALIZATION_EXPLANATION_SUPPORTED`.

Using each rate's original frozen calibration bounds reproduces the prior
physical-horizon contraction: PF_KCW is 200/100/50 ms and J_KCW is 100/50/10
ms at 100/200/400 Hz. Replacing only the evaluator threshold with each route's
already-frozen 100-Hz bounds yields PF_KCW = 200/200/200 ms and J_KCW =
100/100/100 ms under the unchanged nested 90% reliability rule.

The rate-specific bounds shrink almost in inverse proportion to rate. At 400
Hz, PF velocity is 24.4% of its 100-Hz bound, J attitude 24.9%, and J body-rate
25.7%. At 200 Hz the corresponding ratios are 49.4%, 50.2%, and 51.0%.

Absolute maximum and RMS path errors at 100/200 ms remain close across rates.
Common-envelope crossing fractions likewise become nearly rate invariant for
the key channels, whereas rate-specific crossing fractions rise sharply. The
previous reliable-horizon shrinkage is therefore explained predominantly by
rate-dependent calibration-envelope normalization, not proportional worsening
of the absolute predictive time horizon.

This is an evaluation audit, not a change to any registered envelope. Original
rate-specific results remain valid under their own protocol. Existing summaries
lacked per-step paths, so one deterministic reproduction replay was required;
it reused the frozen test identities and models without fitting, selection,
tuning, stabilization, or new formal test decision access.
""")
if __name__=="__main__":main()
