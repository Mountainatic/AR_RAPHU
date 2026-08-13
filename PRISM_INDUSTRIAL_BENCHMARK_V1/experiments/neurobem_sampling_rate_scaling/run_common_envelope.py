from __future__ import annotations

import argparse, json, multiprocessing as mp, subprocess, time
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.neurobem_manifold_switch.data import NeuroBEMSource
from experiments.neurobem_recursive_stability.core import frame_arrays, rollout
from prism_benchmark.neurobem_literature import read_neurobem_csv
from .composition_consistency import RATE_KEYS, RATES, ROUTES, exact_time_index, registered_anchor_times
from .path_excursion import CHANNELS, excursion_metrics, replay_path
from .resampling import resample_track_b
from .run_composition_audit import _load_frozen_adapters, audit_frozen_r3_r4_reproduction_gate
from .run_experiment import _safe, atomic_json

_C={}
def _sha(p:Path)->str:return sha256(p.read_bytes()).hexdigest()

def _task(task):
    name,route=task; source=_C["source"]; adapters=_C["adapters"]; cfg=_C["cfg"]; bounds=_C["bounds"]
    native=read_neurobem_csv(source.path(name)); native_state,_=frame_arrays(native); frames={r:resample_track_b(native,r) for r in RATES}; rows=[]; reliability=[]
    anchors=registered_anchor_times(native,history_ms=cfg["history_duration_ms"],maximum_horizon_ms=max(cfg["physical_horizons_ms"]),spacing_ms=cfg["anchor_spacing_ms"],maximum_anchors=cfg["maximum_anchors_per_trajectory"])
    common= bounds["hz100_h20"][route]
    for anchor in anchors:
        initial=native_state[exact_time_index(native,anchor,400)]
        for rate in RATES:
            path=replay_path(adapters[rate],route,frames[rate],native,anchor,max(cfg["physical_horizons_ms"]),initial,rate)
            specific=bounds[RATE_KEYS[rate]][route]
            for horizon in cfg["physical_horizons_ms"]:
                steps=round(horizon*rate/1000)
                for channel in CHANNELS:
                    absolute=excursion_metrics(path,specific,channel,steps,rate)
                    common_metrics=excursion_metrics(path,common,channel,steps,rate)
                    rows.append({"trajectory_id":name,"route":route,"sampling_rate_hz":rate,"anchor_time_seconds":anchor,"horizon_ms":horizon,"channel":channel,
                                 "terminal_error":absolute["terminal_error"],"max_path_error":absolute["max_path_error"],"rms_path_error":absolute["rms_path_error"],
                                 **{f"rate_specific_{k}":v for k,v in absolute.items() if k not in ("terminal_error","max_path_error","rms_path_error")},
                                 **{f"common_{k}":v for k,v in common_metrics.items() if k not in ("terminal_error","max_path_error","rms_path_error")}})
    state,control=frame_arrays(frames[100])
    for rate in RATES:
        state,control=frame_arrays(frames[rate]); steps=min(round(cfg["rollout_duration_seconds"]*rate),len(state)-adapters[rate].history)
        for interval_ms in cfg["reliability_candidate_ms"]:
            interval=round(interval_ms*rate/1000)
            result=rollout(adapters[rate],route,state,control,steps,{"velocity":1e300,"attitude":1e300,"body_rate":1e300},5,resync_interval=interval)
            inside=result.finite & (result.errors["velocity"]<=common["velocity"]) & (result.errors["attitude"]<=common["attitude"]) & (result.errors["body_rate"]<=common["body_rate"])
            reliability.append({"trajectory_id":name,"route":route,"sampling_rate_hz":rate,"interval_ms":interval_ms,"fraction_inside_common":float(inside.mean()),"trajectory_reliable_common":bool(inside.mean()>=cfg["reliability_probability_minimum"])})
    return rows,reliability

def main():
    p=argparse.ArgumentParser();
    for n in ("config","sampling_freeze","composition_summary","data_root","release_root","split_manifest","output_root"):p.add_argument("--"+n.replace("_","-"),type=Path,required=True)
    a=p.parse_args(); cfg=json.loads(a.config.read_text()); freeze=json.loads(a.sampling_freeze.read_text()); comp=json.loads(a.composition_summary.read_text()); assert comp["test_accessed"]
    gate=audit_frozen_r3_r4_reproduction_gate(a.sampling_freeze,freeze); source=NeuroBEMSource(a.data_root,a.split_manifest,a.release_root); adapters=_load_frozen_adapters(freeze)
    global _C; _C={"source":source,"adapters":adapters,"cfg":cfg,"bounds":freeze["reliability_bounds"]}; tasks=[(n,r) for n in source.names("test") for r in ROUTES]
    with mp.get_context("fork").Pool(min(cfg["workers"],len(tasks))) as pool: out=list(pool.imap(_task,tasks,chunksize=1))
    _C={}; detailed=pd.DataFrame(sum((x[0] for x in out),[])); rel=pd.DataFrame(sum((x[1] for x in out),[]))
    g=detailed.groupby(["route","channel","horizon_ms","sampling_rate_hz"])
    agg=g.agg(terminal_median=("terminal_error","median"),max_path_median=("max_path_error","median"),rms_path_median=("rms_path_error","median"),
              rate_specific_crossing_fraction=("rate_specific_ever_crossing","mean"),common_crossing_fraction=("common_ever_crossing","mean"),
              common_fraction_inside_median=("common_fraction_inside_envelope","median"),common_first_cross_median_ms=("common_first_cross_ms","median"),
              common_crossing_count_median=("common_envelope_crossing_count","median"),common_reentry_count_median=("common_reentry_count","median")).reset_index()
    passed=rel.groupby(["route","sampling_rate_hz","interval_ms"]).trajectory_reliable_common.mean().reset_index(name="reliable_trajectory_fraction")
    horizons=[]
    for (route,rate),q in passed.groupby(["route","sampling_rate_hz"]):
        ok=q[q.reliable_trajectory_fraction>=cfg["reliability_probability_minimum"]].interval_ms
        horizons.append({"route":route,"sampling_rate_hz":rate,"common_reliable_horizon_ms":None if ok.empty else int(ok.max()),"common_reliable_horizon_steps":None if ok.empty else int(ok.max()*rate/1000)})
    boundrows=[]
    for route,channel in (("PF_KCW","velocity"),("J_KCW","attitude"),("J_KCW","body_rate")):
        b100=freeze["reliability_bounds"]["hz100_h20"][route][channel]
        for rate in RATES:
            b=freeze["reliability_bounds"][RATE_KEYS[rate]][route][channel]; boundrows.append({"route":route,"channel":channel,"sampling_rate_hz":rate,"rate_specific_bound":b,"common_bound":b100,"bound_ratio_vs_100":b/b100})
    run=a.output_root/f"{cfg['protocol_id']}_{time.strftime('%Y%m%dT%H%M%S')}";run.mkdir(parents=True)
    detailed.to_csv(run/"COMMON_ENVELOPE_DETAILED.csv",index=False);agg.to_csv(run/"COMMON_ENVELOPE_AGGREGATE.csv",index=False);rel.to_csv(run/"COMMON_RELIABILITY_TRAJECTORIES.csv",index=False);passed.to_csv(run/"COMMON_RELIABILITY_GRID.csv",index=False);pd.DataFrame(horizons).to_csv(run/"COMMON_RELIABLE_HORIZONS.csv",index=False);pd.DataFrame(boundrows).to_csv(run/"BOUND_RATIOS.csv",index=False)
    meta={"status":"COMPLETED_DETERMINISTIC_REPRODUCTION_REPLAY","git_commit":subprocess.check_output(("git","rev-parse","HEAD"),text=True).strip(),"existing_logs_sufficient":False,"deterministic_replay_performed":True,"new_test_decision_access":False,"test_used_for_tuning":False,"model_retrained":False,"threshold_changed":False,"r3_r4_reproduction_gate":gate,"common_horizons":horizons}
    atomic_json(run/"COMMON_ENVELOPE_SUMMARY.json",meta);print(json.dumps(_safe(meta),indent=2))
if __name__=="__main__":main()
