#!/usr/bin/env python
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd
p=argparse.ArgumentParser(); p.add_argument('--csv',required=True); a=p.parse_args()
f=pd.read_csv(a.csv)
for col in ['utilization_gpu_percent','memory_used_mib','memory_total_mib','temperature_c','power_w']:
    f[col]=pd.to_numeric(f[col],errors='coerce')
summary={
 'samples':int(len(f)),
 'mean_gpu_utilization_percent':float(f.utilization_gpu_percent.mean()),
 'p95_gpu_utilization_percent':float(f.utilization_gpu_percent.quantile(.95)),
 'peak_memory_used_mib':float(f.memory_used_mib.max()),
 'memory_total_mib':float(f.memory_total_mib.max()),
 'peak_memory_fraction':float(f.memory_used_mib.max()/f.memory_total_mib.max()),
 'peak_temperature_c':float(f.temperature_c.max()),
 'mean_power_w':float(f.power_w.mean()),
}
if summary['mean_gpu_utilization_percent']<60 and summary['peak_memory_fraction']<.6:
 summary['recommendation']='Increase workers-per-device by 1 or 2.'
elif summary['peak_memory_fraction']>.85:
 summary['recommendation']='Reduce workers-per-device by 1.'
else:
 summary['recommendation']='Current concurrency is reasonable.'
print(json.dumps(summary,indent=2))
