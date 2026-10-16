"""Read-only v0.6 stage residual diagnostics after the global target barrier."""
import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
import sys,json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parent;sys.path[:0]=[str(ROOT/'code'),str(ROOT/'code/src')]
from canonical.cz_crossrod_l128_20261008.invariants import file_sha,dump
from canonical.cz_crossrod_l128_20261008.discrete_observation import DiscreteObservationSpec
from canonical.cz_crossrod_l128_20261008.discrete_diagnostics import segmented_diagnostics
def main():
    status=json.loads((ROOT/'FINALIZATION_STATUS.json').read_text())
    if status['status']!='E1_AND_BASELINE_TARGET_INFERENCE_COMPLETED':raise RuntimeError('Global frozen inference must finish before diagnostics')
    output=ROOT/'frozen_v06_diagnostics'
    if output.exists():raise RuntimeError('Diagnostics output is immutable')
    obs=DiscreteObservationSpec(sample_period_seconds=2.,time_origin_policy='EXPLICIT_RAW_SEGMENT_ORIGIN',timestamp_field=None,
        output_unit='recorded_diameter_unit_unverified',segment_policy='EXPLICIT_ENTITY_SEGMENTS_AND_CONTIGUOUS_ORIGINS',
        observation_availability_policy='EXCLUSIVE_ORIGIN_INPUTS_THROUGH_T_MINUS_1',head_contract_id='CZ_L128_H1_FROZEN_STAGEWISE')
    output.mkdir();summary=[];bindings={}
    for case in ('Rod_1_to_Rod_2','Rod_2_to_Rod_1'):
        for split in ('validation','test'):
            receipt=ROOT/'inference'/case/split/'INFERENCE_RECEIPT.json';sealed=json.loads(receipt.read_text());bindings[str(receipt)]=file_sha(receipt)
            assert sealed['fit_calls']==0 and sealed['selection_authority'] is False
            for stage in ('K','C','W','A'):
                path=ROOT/'inference'/case/split/(stage+'_pointwise.parquet')
                assert file_sha(path)==sealed['artifact_sha256'][str(path)]
                f=pd.read_parquet(path);quantity='post-'+stage+(' innovation' if stage=='A' else ' residual')
                data=pd.DataFrame(dict(entity_id=f.entity_id,segment_id=f.entity_id,split_id=split,raw_index=f.origin))
                level_path=ROOT/'pointwise_enriched'/case/split/('PRISM_'+stage+'.parquet')
                level=pd.read_parquet(level_path)
                assert level[['entity_id','origin']].equals(f[['entity_id','origin']])
                assert (level.y_true.to_numpy()==f.y_true.to_numpy()).all()
                bindings[str(level_path)]=file_sha(level_path)
                data['recorded_level']=level.future_level_true.to_numpy()
                data['registered_target']=f.y_true.to_numpy();data[quantity]=f.residual.to_numpy()
                result=segmented_diagnostics(data,obs,value_columns=('recorded_level','registered_target',quantity))
                dest=output/case/split/stage;dest.mkdir(parents=True)
                local=[]
                for j,(s,psd,acf) in enumerate(result):
                    record=dict(s,direction=case,split=split,stage=stage,prediction_role='FROZEN_OUT_OF_SAMPLE',
                        white_noise_reference_band_role='REPORTING_ONLY_NOT_SELECTION',training_called=False)
                    local.append(record)
                    if psd is not None:psd.to_csv(dest/(str(j)+'_PSD.csv'),index=False);acf.to_csv(dest/(str(j)+'_ACF_PACF.csv'),index=False)
                dump(dest/'SUMMARY.json',local);summary.extend(local)
    dump(output/'SUMMARY.json',dict(status='FROZEN_OUT_OF_SAMPLE_DIAGNOSTICS_COMPLETED',segments=summary,receipt_sha256=bindings,
        formal_whiteness_gate_used=False,grid_candidate_fed_to_model=False,fit_calls=0,registered_target='D[t]-D[t-1]'))
    dump(ROOT/'FROZEN_DIAGNOSTICS_STATUS.json',dict(status='COMPLETED',summary=str(output/'SUMMARY.json'),fit_calls=0))
if __name__=='__main__':main()
