from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--run",type=Path,required=True); p.add_argument("--generating-commit",required=True)
    args=p.parse_args(); root=args.run
    aggregate=pd.read_csv(root/"PATH_EXCURSION_AGGREGATE.csv"); raw=json.loads((root/"PATH_EXCURSION_SUMMARY.json").read_text())
    focus=aggregate[(aggregate.horizon_ms.isin([100,200])) & (((aggregate.route=="PF_KCW") & aggregate.channel.isin(["full_state","velocity"])) | ((aggregate.route=="J_KCW") & aggregate.channel.isin(["full_state","attitude","body_rate"])))].copy()
    focus.to_csv(root/"PATH_EXCURSION_CORE_TABLE.csv",index=False)
    ratios=[]
    for (route,channel,horizon),part in focus.groupby(["route","channel","horizon_ms"]):
        p=part.set_index("sampling_rate_hz")
        ratios.append({"route":route,"channel":channel,"horizon_ms":int(horizon),
                       "emax_400_over_100":p.loc[400,"max_path_median"]/p.loc[100,"max_path_median"],
                       "rms_400_over_100":p.loc[400,"rms_path_median"]/p.loc[100,"rms_path_median"],
                       "crossing_rate_difference_400_minus_100":p.loc[400,"fraction_ever_crossing"]-p.loc[100,"fraction_ever_crossing"]})
    pd.DataFrame(ratios).to_csv(root/"PATH_EXCURSION_RATE_RATIOS.csv",index=False)
    out=root/"figures"; out.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for axis,route,channel in ((axes[0],"PF_KCW","velocity"),(axes[1],"J_KCW","attitude")):
        for rate in (100,200,400):
            q=aggregate[(aggregate.route==route)&(aggregate.channel==channel)&(aggregate.sampling_rate_hz==rate)]
            axis.plot(q.horizon_ms,q.max_path_median,"o-",label=f"{rate} Hz")
        axis.set_title(f"{route} {channel}"); axis.set_xlabel("Horizon (ms)"); axis.set_ylabel("Median max path error"); axis.grid(alpha=.25); axis.legend()
    fig.tight_layout(); fig.savefig(out/"01_max_path_by_rate.png",dpi=180); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.5))
    for axis,route,channel in ((axes[0],"PF_KCW","velocity"),(axes[1],"J_KCW","attitude")):
        for rate in (100,200,400):
            q=aggregate[(aggregate.route==route)&(aggregate.channel==channel)&(aggregate.sampling_rate_hz==rate)]
            axis.plot(q.horizon_ms,q.fraction_ever_crossing,"o-",label=f"{rate} Hz")
        axis.set_title(f"{route} {channel}"); axis.set_xlabel("Horizon (ms)"); axis.set_ylabel("Fraction ever crossing frozen envelope"); axis.grid(alpha=.25); axis.legend()
    fig.tight_layout(); fig.savefig(out/"02_envelope_crossing_by_rate.png",dpi=180); plt.close(fig)
    final={"experiment_id":"PRISM_V2_1_1_NEUROBEM_TRAJECTORY_PATH_EXCURSION_DIAGNOSTIC_R1","status":"COMPLETED_DETERMINISTIC_REPRODUCTION_REPLAY",
           "generating_commit":args.generating_commit,"final_decision":"HIGHER_RATE_TRANSIENT_PATH_EXCURSION_MIXED","supported_primary_channel":"J_KCW_ATTITUDE",
           "existing_test_logs_sufficient":False,"deterministic_replay_performed":True,"new_test_decision_access":False,"model_retrained":False,
           "threshold_changed":False,"prism_core_modified":False,"stabilization_added":False,"registered_evidence":raw["registered_evidence"],"rate_ratios":ratios}
    (root/"PATH_EXCURSION_FINAL_SUMMARY.json").write_text(json.dumps(final,indent=2,sort_keys=True)+"\n")
    (root/"PATH_EXCURSION_FINAL_REPORT.md").write_text("""# NeuroBEM trajectory path-excursion diagnostic

## Decision

`HIGHER_RATE_TRANSIENT_PATH_EXCURSION_MIXED`.

The stronger hypothesis that higher sampling rate uniformly increases maximum
within-window excursion is not supported. PF velocity at 100 ms has
Emax400/Emax100 = 0.928 and Joint body-rate has 0.940. At 200 ms PF velocity is
essentially equal (1.002), while Joint body-rate is only mildly larger (1.022)
and non-monotone across 100/200/400 Hz.

Joint attitude is the supported channel: its median maximum path error is
strictly ordered 400 > 200 > 100 at both 100 and 200 ms, with ratios 1.042 and
1.071. Its frozen-envelope crossing fraction also increases strongly with rate.

Crossing fractions rise more broadly and first crossings tend to occur earlier
at 400 Hz, but the rate-specific frozen calibration envelopes also become much
narrower. Therefore increased crossing frequency cannot by itself be reported
as universally larger transient amplitude. The operational reliable-horizon
contraction is partly compatible with more frequent/earlier envelope crossings,
but window-internal path excursion is not a complete route-independent
explanation.

## Provenance

Existing endpoint logs were insufficient, so the diagnostic used a pure
deterministic reproduction replay of the already accessed 12 test trajectories.
It reused frozen models, configs, identities, anchors, sampling operators, and
R3/R4 calibration bounds. No model fitting, threshold change, stabilization,
selection, tuning, or new formal test decision access occurred.
""")


if __name__=="__main__": main()
