#!/usr/bin/env python
"""Focused CUDA profiler for the V20 sequence-first KAN path."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, torch
from stage1.model import Stage1TargetDelayKAN
from stage1.sequence_ops import ContiguousWindowBatch
from stage1.experiment_utils import input_ranges_from_train, read_json, save_json

ROOT=Path(__file__).resolve().parents[1]

def main():
 p=argparse.ArgumentParser(); p.add_argument('--output',default='results_stage1/STAGE1_DUAL_SOLVER_V20/gpu_profile')
 p.add_argument('--steps',type=int,default=50); a=p.parse_args()
 if not torch.cuda.is_available(): raise RuntimeError('CUDA is required')
 z=np.load(ROOT/'results_stage1/KAN_O1_v12/data_snapshot.npz'); split=read_json(ROOT/'results_stage1/KAN_O1_v12/split_manifest.json')
 train=np.asarray(split['train']); x=z['X'].astype('float32'); y=z['y_clean'].astype('float32')
 ranges=input_ranges_from_train(x,train); device=torch.device('cuda')
 windows=torch.as_tensor(x[train],device=device); target=torch.as_tensor(y[train],device=device)
 batch=ContiguousWindowBatch.from_windows(windows,target)
 model=Stage1TargetDelayKAN(10,32,hidden_kan=8,kan_grid_size=7,epsilon=0.,active_mask=torch.ones(10,dtype=torch.bool,device=device),input_grid_ranges=ranges,response_execution_mode='auto').to(device)
 opt=torch.optim.Adam(model.parameters(),lr=.003)
 out=ROOT/a.output; out.mkdir(parents=True,exist_ok=True)
 torch.cuda.reset_peak_memory_stats()
 activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]
 with torch.profiler.profile(activities=activities,record_shapes=True,profile_memory=True,with_stack=False) as prof:
  for step in range(a.steps):
   opt.zero_grad(set_to_none=True); pred=model.forward_contiguous(batch,return_aux=False).squeeze(-1)
   loss=(pred-target).square().mean(); loss.backward(); opt.step()
   prof.step()
 prof.export_chrome_trace(str(out/'trace.json'))
 table=prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=40)
 (out/'top_cuda_ops.txt').write_text(table,encoding='utf-8')
 save_json(out/'summary.json',{'steps':a.steps,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved(),'gpu':torch.cuda.get_device_name(0)})
 print(table)
if __name__=='__main__': main()
