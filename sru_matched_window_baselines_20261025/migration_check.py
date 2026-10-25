"""Check frozen weights in strict FP32 against CPU FP64 before any new training."""
from pathlib import Path
import sys,json,copy
import experiment as e
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parent
if __name__=='__main__':
    old=Path(sys.argv[1]);e.reg();a=np.vstack([np.load(ROOT/'data/train.npz')['values'],np.load(ROOT/'data/development.npz')['values']]);o=np.arange(6048,8064);proof=[]
    for name in ['LSTM','Transformer','iTransformer','TimeXer']:
        obj=torch.load(old/'runs/D4'/name/'seed_0.pt',map_location='cpu',weights_only=False);c=obj['config'];model=e.build(name,6,c['d'],c['L']).cuda();model.load_state_dict(obj['state'])
        x,_,anchor,sc=e.make_batch(a,o,c,4,obj['scaler']);p=e.infer(model,x,sc,anchor,c)
        previous=pd.read_parquet(old/'results'/f'D4_{name}_seed0.parquet').y_pred.to_numpy()
        cpu=copy.deepcopy(model).cpu().double();oracle=e.infer(cpu,x.cpu().double(),sc,anchor,c)
        np.testing.assert_allclose(p,oracle,rtol=3e-5,atol=2e-6)
        proof.append(dict(model=name,old_window=128,seed=0,max_absolute_difference_to_FP64=float(np.max(np.abs(p-oracle))),
            descriptive_difference_to_previous_default_kernel=float(np.max(np.abs(p-previous))),rtol=3e-5,atol=2e-6,passed=True))
        del model,cpu,x;torch.cuda.empty_cache()
    e.dump(ROOT/'MIGRATION_CHECK.json',dict(status='COMPLETED',new_GPU=torch.cuda.get_device_name(0),previous_GPU='RTX4090D',checks=proof,
        strict_FP32_to_FP64_passed=True,previous_default_kernel_equivalence_not_claimed=True,
        existing_development_predictions_read_only=True,no_new_fits=True,used_for_hyperparameter_selection=False,reserved_tail_read=False))
    print('MIGRATION_CHECK_PASSED',json.dumps(proof),flush=True)
