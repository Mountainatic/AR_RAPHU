"""Scientific boundary and estimator regression tests for read-only whitening."""
import copy
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
sys.path.insert(0,str(PROJECT/'src'))
from experiments.whitening.common import (AUTHORITY_COMMIT,AUTHORITY_BRANCH,DT,sha256,contiguous_frames,evaluation_rod)
from experiments.whitening.residual_contract import residuals
from experiments.whitening.acf_diagnostics import acf,diagnostics,h4_overlap_reference,h4_nonoverlap
from experiments.whitening.gaussianity_diagnostics import gaussianity
from experiments.whitening.quantization_audit import quantization
from experiments.whitening.input_residual_ccf import lagged_association
from experiments.whitening.k_representation_audit import registered_profiles,audit_case,geometry_resolution
from experiments.whitening.load_frozen_predictions import load_frozen
from experiments.whitening.run_whitening_audit import run,verify

def fixture_frame(n=512,split='source-validation'):
    rng=np.random.default_rng(25)
    x=rng.normal(size=n)
    return pd.DataFrame(dict(direction='Rod_1_to_Rod_2',horizon='H1',split=split,entity_id='Rod_1_segment_3',origin=np.arange(n),
        base_origin_id=[str(i) for i in range(n)],y_true=x,prediction_K=x*.1,prediction_C=x*.1,prediction_W=x*.1,prediction_A=x*.5))

def test_bypass_residual_identity_exact():
    f,a=residuals(fixture_frame(),{'C':'BYPASSED','W':'W_NOT_APPLICABLE'})
    assert np.array_equal(f.r_K,f.r_C)
    assert np.array_equal(f.r_C,f.r_W)

def test_bypass_violation_is_rejected_not_repaired():
    f=fixture_frame(); f.prediction_C+=1e-5
    with pytest.raises(AssertionError,match='identity'):residuals(f,{'C':'ZERO_IDENTITY'})

def test_missing_stage_never_inferred():
    f=fixture_frame().drop(columns='prediction_A'); f,a=residuals(f)
    assert f.r_A.isna().all() and a['A']=='NOT_AVAILABLE_FROM_FROZEN_ARTIFACT'

def test_no_cross_segment_acf():
    f=fixture_frame(16); f.loc[8:,'origin']+=100
    pieces=list(contiguous_frames(f)); assert len(pieces)==2
    for _,g in pieces:
        r,pairs=acf(g.y_true,12)
        assert pairs[8]==0 and np.isnan(r[8])
    f['segment_id']=['a']*8+['b']*8; f.origin=np.arange(16)
    assert len(list(contiguous_frames(f)))==2

def test_no_cross_oof_model_boundary():
    f=fixture_frame(16);f['outer_fold']=[0]*8+[1]*8
    assert [len(g) for _,g in contiguous_frames(f)]==[8,8]

def test_lag_to_seconds():
    assert 7*DT==14
    assert lagged_association(np.arange(16.),np.arange(16.))[7]['seconds']==14

def test_h4_overlap_theory_and_simulated_innovations():
    assert np.array_equal(h4_overlap_reference(3)[1:],[.75,.5,.25])
    innovations=np.random.default_rng(20).normal(size=200000)
    sums=np.convolve(innovations,np.ones(4),mode='valid')
    r,_=acf(sums,3)
    np.testing.assert_allclose(r[1:],[.75,.5,.25],atol=.012)
    for phase in h4_nonoverlap(sums):assert abs(acf(phase,1)[0][1])<.025

def test_gaussianity_is_not_whiteness():
    rng=np.random.default_rng(6);x=np.empty(20000); x[0]=0
    noise=rng.normal(size=len(x))
    for i in range(1,len(x)):x[i]=.92*x[i-1]+noise[i]
    shuffled=rng.permutation(x)
    original=gaussianity(x); other=gaussianity(shuffled)
    for metric in ('mean','std','skewness','excess_kurtosis','qq_R2','jarque_bera_statistic'):
        assert np.isclose(original[metric],other[metric],rtol=1e-10,atol=1e-10)
    assert acf(x,1)[0][1]>.9 and abs(acf(shuffled,1)[0][1])<.03

def test_k_registry_is_called_and_can_expand(monkeypatch):
    from prism_benchmark import v2_k
    from prism_benchmark.cz_l256_nowcast import view
    calls=[]
    def registry(v,ch,c):calls.append(ch);return [(2,8),(2,128)]
    monkeypatch.setattr(v2_k,'channel_profiles',registry)
    profiles,rows=registered_profiles(view('input_only',h_steps=1),'joint_lift',{},[4,8])
    assert calls==['joint_lift'] and profiles[-1][1]==128
    assert max(r['maximum_history_seconds'] for r in rows)==256

def test_authority_actual_h1_registry_and_geometry():
    from prism_benchmark.v211_config import load_v211_configs
    from prism_benchmark.cz_l256_nowcast import view
    _,v21,v2=load_v211_configs(PROJECT,protocol='representative_stage1')
    profiles,rows=registered_profiles(view('input_only',h_steps=1),'joint_lift',v2,v21['K_C']['m_tau'])
    assert len(profiles)>1 and (2,8) in profiles
    assert {p[0] for p in profiles}=={2,4,8}
    assert min(r['minimum_near_term_bin_width_seconds'] for r in rows)==4
    assert max(r['maximum_history_seconds'] for r in rows)==16

def test_rod_mapping():
    assert evaluation_rod('Rod_1_to_Rod_2','target-rod frozen test')=='Rod2'
    assert evaluation_rod('Rod_2_to_Rod_1','target-rod frozen test')=='Rod1'
    assert evaluation_rod('Rod_1_to_Rod_2','source-validation')=='Rod1'
    assert evaluation_rod('Rod_2_to_Rod_1','source-validation')=='Rod2'

def test_split_and_entity_isolation():
    f=fixture_frame(16);g=f.copy();g['split']='target-rod frozen test';g['entity_id']='Rod_2_segment_1'
    pieces=list(contiguous_frames(pd.concat([f,g],ignore_index=True)))
    assert len(pieces)==2
    for _,piece in pieces:assert piece.split.nunique()==piece.entity_id.nunique()==1

def test_acf_matches_direct_reference():
    x=np.random.default_rng(12).normal(size=513);r,p=acf(x,900);z=x-x.mean()
    expected=np.correlate(z,z,'full')[len(x)-1:]/np.dot(z,z)
    np.testing.assert_allclose(r[:len(x)],expected,rtol=1e-11,atol=1e-12)
    assert np.isnan(r[len(x):]).all()

def test_constant_residual_has_undefined_acf_not_fake_white():
    s,r,*_=diagnostics(np.zeros(1000))
    assert np.isnan(r).all() and np.isnan(s['acf_energy_1_900'])

def test_quantization_synthetic_fixture():
    x=np.array([0,.01,0,-.01,0,.01,0,-.01])
    s,t,intervals,r=quantization(x)
    assert s['fraction_exactly_zero']==.5 and s['fraction_on_001_grid']==1
    assert np.isclose(s['estimated_nearest_grid_spacing'],.01)
    assert s['distance_to_001_grid_max']<1e-12
    assert sum(v['count'] for v in t)==len(x)-1
    assert intervals==[dict(interval_samples=2,interval_seconds=4.,count=3)]
    assert set(r)=={'y','event','sign'}

def test_ccf_direction_and_first_differences():
    rng=np.random.default_rng(98);x=rng.normal(size=3000);r=np.r_[rng.normal(size=7),x[:-7]]
    rows=lagged_association(x,r);peak=max(rows,key=lambda r:abs(r['correlation']))
    assert peak['lag_samples']==7 and peak['correlation']>.999
    rows=lagged_association(x,r,differenced=True)
    assert rows[7]['correlation']>.999 and rows[7]['seconds']==14

def test_event_indicator_and_sign_are_exact_not_grid_tolerance():
    x=np.array([0.,1e-10,0.,-1e-10,0.,0.])
    summary,transitions,intervals,vectors=quantization(x)
    assert intervals==[dict(interval_samples=2,interval_seconds=4.,count=1)]
    np.testing.assert_allclose(vectors['event'],acf((x!=0).astype(float),160)[0],equal_nan=True)
    np.testing.assert_allclose(vectors['sign'],acf(np.sign(x),160)[0],equal_nan=True)

def test_geometry_needs_coverage():
    intervals=[(0,2),(2,4),(4,6),(6,8)]
    assert geometry_resolution(intervals,8,16)=='YES'
    assert geometry_resolution(intervals,20,40)=='NO'
    assert geometry_resolution(intervals,4,8)=='COARSE'

def test_source_stage_join_mismatch_rejected(tmp_path):
    f=fixture_frame(32);p=tmp_path/'f.csv';f.to_csv(p,index=False)
    g=f.copy();g['y_pred']=g.prediction_C;g.y_true+=1;p2=tmp_path/'g.csv';g.to_csv(p2,index=False)
    source=dict(path=str(p),sha256=sha256(p),direction='Rod_1_to_Rod_2',horizon='H1',lineage='test',split='source-validation',
        stage_paths={'C':dict(path=str(p2),sha256=sha256(p2),same_chain=True)})
    with pytest.raises(ValueError,match='target/support'):load_frozen(source)

def test_k_support_rejects_test_dataframe(tmp_path):
    f=fixture_frame();f['split']='test';p=tmp_path/'test.parquet';f.to_parquet(p,index=False)
    case=dict(protocol='representative_stage1',h_steps=1,task_id='CZ_DIAM_RAW2S_CURRENT_L256_H1',train_path=str(p),train_sha256=sha256(p))
    with pytest.raises(ValueError,match='training rows ONLY'):audit_case(case,PROJECT)

def test_per_fold_selected_geometry_does_not_use_final_profile(tmp_path,monkeypatch):
    from prism_benchmark import cpu_data,cz_k_support
    f=fixture_frame(100);f['split']='train';p=tmp_path/'train.parquet';f.to_parquet(p,index=False)
    monkeypatch.setattr(cpu_data,'inner_folds',lambda *a:[(np.arange(10),np.arange(10,20))])
    monkeypatch.setattr(cz_k_support,'_profile_support_availability',lambda frame,folds,profiles,**kw:
        [dict(profile=list(profile),status='AVAILABLE') for profile in profiles])
    record=dict(channel='joint_lift',profile=[2,8],m_tau=4,intervals=[[0,2],[2,4],[4,6],[6,8]],active=False,route='ZERO_IDENTITY')
    folded={**record,'profile':[2,4],'intervals':[[0,2],[2,4]],'geometry_source':'frozen fold fixture'}
    case=dict(protocol='representative_stage1',h_steps=1,task_id='CZ_DIAM_RAW2S_CURRENT_L256_H1',train_path=str(p),train_sha256=sha256(p),
        direction='Rod_1_to_Rod_2',horizon='H1',lineage='synthetic',selected_records=[record],selected_fold_records={'0':[folded]})
    registered,available,selected,summaries=audit_case(case,PROJECT)
    rows=[r for r in summaries if r['channel']=='joint_lift']
    assert [(r['outer_fold'],r['maximum_selected_history_seconds']) for r in rows]==[(None,16),(0,8)]
    assert len([r for r in selected if r['channel']=='joint_lift' and r['outer_fold']==0])==2

def config_for_frame(path):
    return dict(authority_commit=AUTHORITY_COMMIT,authority_branch=AUTHORITY_BRANCH,authority_project=str(PROJECT),whitening_code_commit='test',
        sampling_interval=2.,test_selection_use=False,model_fitting=False,route_changes=False,
        diagnostic_parameters=dict(acf_max_lag=900,pacf_max_lag=80,ccf_max_lag=160,windows=4,ccf_stability_min_abs=.1,ccf_stability_min_windows=3),
        frozen_files={str(path):sha256(path)},dataset_workbook_hashes={'fixture':sha256(path)},k_cases=[],wrapper_findings=[],missing=[],
        sources=[dict(path=str(path),sha256=sha256(path),direction='Rod_1_to_Rod_2',horizon='H1',lineage='synthetic',split='target-rod frozen test',
                      stage_columns=dict(K='prediction_K',C='prediction_C',W='prediction_W',A='prediction_A'))])

def test_no_test_selection_end_to_end(tmp_path,monkeypatch):
    # Tripwires on actual authority fitting/routing functions: test data can
    # reach descriptive analysis, but no decision/fit path may be called.
    from prism_benchmark import v2_k,strict_oof_selection,v211_k
    def forbidden(*a,**kw):raise AssertionError('test reached model fit/selection/routing')
    monkeypatch.setattr(v2_k,'run_channel',forbidden)
    monkeypatch.setattr(v211_k,'run_k_channel',forbidden)
    monkeypatch.setattr(strict_oof_selection,'strict_nested_oof_select',forbidden)
    f=fixture_frame(1024,'target-rod frozen test');f['entity_id']='Rod_2_segment_1'
    p=tmp_path/'test.csv';f.to_csv(p,index=False)
    c=config_for_frame(p);before=copy.deepcopy(c)
    result=run(c,tmp_path/'audit')
    assert result['test_selection_use'] is False and result['authority_model_parameter_or_route_modified'] is False
    assert c==before
    assert (tmp_path/'audit/REPORT.md').is_file()
    stats=pd.read_csv(tmp_path/'audit/GLOBAL_WHITENESS/RESIDUAL_STATS.csv')
    assert set(stats.window)=={0,1,2,3,4}

def test_post_freeze_mutation_rejected(tmp_path):
    p=tmp_path/'x.csv';fixture_frame().to_csv(p,index=False);c=config_for_frame(p)
    p.write_text('changed')
    with pytest.raises(ValueError,match='Frozen file changed'):verify(c)

def test_test_driven_diagnostic_config_rejected(tmp_path):
    p=tmp_path/'x.csv';fixture_frame().to_csv(p,index=False);c=config_for_frame(p)
    c['diagnostic_parameters']['acf_max_lag']=200
    with pytest.raises(ValueError,match='parameters changed'):verify(c)
