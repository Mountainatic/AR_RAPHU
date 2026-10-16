import json,sys,copy
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from canonical.cz_h1_posthoc_v22_accel12h.invariants import digest,file_sha,ProtocolError,frame_hash
from canonical.cz_h1_posthoc_v22_accel12h.registries import load_registry
from canonical.cz_h1_posthoc_v22_accel12h.hierarchical_search import registered_screen
from canonical.cz_h1_posthoc_v22_accel12h.replay_policy import ReplaySession,cadence,recipe_binding
from canonical.cz_h1_posthoc_v22_accel12h.checkpoint import Checkpoints
from canonical.cz_h1_posthoc_v22_accel12h.canonical_engine import CanonicalPipeline
from canonical.cz_h1_posthoc_v22_accel12h.final_development_cf import replay
from canonical.cz_h1_posthoc_v22_accel12h.runtime_budget import RuntimeBudget,BudgetStop,launch_gate

ROOT=Path(__file__).parents[1];MODULE=ROOT/'canonical/cz_h1_posthoc_v22_accel12h'
sys.path.insert(0,str(ROOT/'src'))
PROTOCOL=json.loads((MODULE/'CANONICAL_PROTOCOL.json').read_text(encoding='utf-8'))

def frame(n,entity='segment',start=1):
    o=np.arange(start,start+n)
    return pd.DataFrame(dict(entity_id=entity,origin=o,split='train',target_stop_exclusive=o+1,
        latest_available_target_index=o-1,dependency_start=o-1,dependency_stop_exclusive=o+1,
        sample_id=[entity+str(i) for i in o],view_sample_id=[entity+str(i) for i in o],y_true=np.sin(o)))

def test_screen_all_69_exact_first_rank_one_before_targets():
    _,specs=load_registry(MODULE/'K_MODEL_REGISTRY.json');p=SimpleNamespace(K_specs=specs)
    screen=registered_screen(p)
    assert len(screen)==69 and len(set(s[:2] for s in screen))==69
    assert all(s[2:]==screen[0][2:] for s in screen)
    first=next(s for s in specs if s[2]=='RANK_1_URYSOHN')
    assert screen[0]==first
    for s in screen:
        structural=[r for r in specs if r[:2]==s[:2]]
        assert len(structural)==196 and set(r[2] for r in structural)==set(r[2] for r in specs)
    assert max(s[0][1] for s in screen)==256

def test_protocol_new_policy_and_all_execution_barriers_false():
    assert PROTOCOL['revision']=='CZ_H1_POSTHOC_ACCEL12H_V1'
    assert not any(PROTOCOL['execution_authorization'][k] for k in ('formal_development_training','source_validation_reporting','target_test_inference'))
    assert PROTOCOL['final_development_cf']['refit_block_rows']==320
    assert PROTOCOL['final_development_cf']['reselect_block_rows']==1280
    assert PROTOCOL['geometry']['inner']['block_rows']==PROTOCOL['geometry']['outer']['block_rows']==320

@pytest.mark.parametrize('n',[1,319,320,321,1280,1281,3000,15000])
def test_cadence_full_coverage_tail(n):
    T=frame(n);blocks=list(cadence(T))
    assert np.array_equal(np.concatenate([b[2] for b in blocks]),np.arange(n))
    assert [b[4] for b in blocks if b[5]]==list(range(0,len(blocks),4))
    assert all(len(b[2])==320 for b in blocks[:-1])
    assert len(blocks[-1][2])==n-320*(len(blocks)-1)
    assert all(not len(f) or f[-1]<e[0] for _,f,e,*_ in blocks)

def test_entity_local_cadence_resets_and_no_cross_entity():
    T=pd.concat([frame(1700,'one'),frame(900,'two')],ignore_index=True)
    blocks=list(cadence(T));assert [b[4] for b in blocks if b[3]=='two']==[0,1,2]
    assert all(T.iloc[b[2]].entity_id.nunique()==1 for b in blocks)

@pytest.mark.parametrize('alpha',[1e-8,1e-6,1e-4,.01,.1,1.,10.,100.,10000.])
def test_multi_alpha_authority_prediction_contract_and_risk_bit_equivalence(alpha):
    from prism_benchmark.v2_c import _ridge_fit
    from canonical.cz_h1_posthoc_v22_accel12h.ridge_path import RidgePath
    from canonical.cz_h1_posthoc_v22_accel12h.risk import mse_evidence
    rng=np.random.default_rng(903);x=rng.normal(size=(801,17));x[:,-1]=1.
    y=rng.normal(size=801);z=rng.normal(size=(320,17))
    old,ct=_ridge_fit(x,y,z,alpha);new,nct=RidgePath(x,y).solve(z,alpha)
    assert np.array_equal(old,new) and ct==nct
    assert mse_evidence(y[:320],old)==mse_evidence(y[:320],new)

def mock_pipe(tmp_path):
    p=object.__new__(CanonicalPipeline)
    p.mode='accelerated_benchmark';p.root=tmp_path/'run';p.policy_id='policy';p.protocol=copy.deepcopy(PROTOCOL)
    p.context={};p.replay_session=None;p.artifacts=[];p.fit_calls=0;p.tune_calls=[]
    p.registration=dict(canonical_protocol_sha256={},authority_bindings_sha256_lf={})
    p.checkpoints=Checkpoints(p.root,dict(policy='frozen'))
    p.partition_id=lambda T:digest(T.sample_id.tolist())
    def tune(stage,T):
        p.tune_calls.append((stage,T.copy()))
        parent=None if stage=='K' else p.learn({'C':'K','W':'C'}[stage],T)
        # Keep no numerical recipe in this fixture; refit-only must keep identity.
        return dict(stage=stage,parent=parent,domain=False,recipe=None,representations=[] if stage=='K' else None,
            routing_status='MODEL_UNAVAILABLE')
    p.tune=tune
    p.emit=lambda state,T,source=None:np.zeros(len(T))
    p.response_coordinate=lambda state,T:(np.zeros(len(T)),np.zeros(len(T),bool),{})
    p.overlap_components=lambda state,T:(pd.DataFrame({'up_x':np.zeros(len(T))},index=T.index),{})
    return p

def test_replay_no_admission_no_refit_only_search_once_dag(tmp_path):
    p=mock_pipe(tmp_path);T=frame(3000)
    routes={'K':'BYPASSED','C':'BYPASSED'}
    with ReplaySession(p,routes) as s:
        for _,fit,ev,entity,local,reselect in cadence(T):
            before=len(p.tune_calls)
            state=s.at_block('C',T.iloc[fit],entity=entity,local_index=local,reselect=reselect)
            if not reselect:assert len(p.tune_calls)==before
            assert state['routing_status']=='BYPASSED' and not state['admission_vote_produced']
        # K still selects a past-only representation at supported refreshes.
        assert p.tune_calls and all(stage=='K' for stage,_ in p.tune_calls)
        count=len(p.tune_calls)
        first=s.select('C',T.iloc[:1600]);second=s.select('C',T.iloc[:1600])
        assert first==second and len(p.tune_calls)==count+1

def test_current_block_target_mutation_cannot_change_recipe_or_predictor(tmp_path):
    T=frame(3000);changed=T.copy();changed.loc[2560:,'y_true']=1e99
    out=[]
    for k,universe in enumerate((T,changed)):
        p=mock_pipe(tmp_path/str(k));models=[]
        with ReplaySession(p,{'K':'BYPASSED','C':'BYPASSED'}) as s:
            for _,fit,ev,entity,local,reselect in cadence(universe):
                state=s.at_block('C',universe.iloc[fit],entity=entity,local_index=local,reselect=reselect)
                if local==8:models.append(state)
        out.append(models)
    assert out[0]==out[1]

def test_replay_full_retained_rows_and_cached_predictions(tmp_path):
    p=mock_pipe(tmp_path);T=frame(3000)
    parent=dict(stage='C',routing_status='BYPASSED',parent=dict(stage='K',routing_status='BYPASSED',parent=None))
    freeze=dict(policy_id=p.policy_id,partition_id=p.partition_id(T),upstream_stage='C',selected_spec=('frozen',))
    a,proof,_=replay(p,'C',T,freeze,frozen_parent=parent)
    calls=len(p.tune_calls)
    b,second,_=replay(p,'C',T,freeze,frozen_parent=parent)
    pd.testing.assert_frame_equal(a,b);assert proof==second and len(p.tune_calls)==calls
    assert len(a)==3000 and proof['coverage']['excluded_rows']==0
    assert [r['entity_local_block'] for r in proof['provenance'] if r['reselection_checkpoint']]==[0,4,8]
    assert all(not r['global_route_re_admitted'] for r in proof['provenance'])

def test_no_recipe_not_invented_on_refit_only(tmp_path,monkeypatch):
    p=mock_pipe(tmp_path)
    with ReplaySession(p,{'K':'ADMITTED','C':'ADMITTED','W':'ADMITTED'}) as s:
        s.at_block('W',frame(0),entity='segment',local_index=0,reselect=True)
        monkeypatch.setattr(p,'tune',lambda *args:pytest.fail('REFIT_ONLY called MODEL selection'))
        state=s.at_block('W',frame(1500),entity='segment',local_index=1,reselect=False)
        assert not state['domain']

def test_runtime_budget_resume_and_stop(tmp_path):
    now=[0.];contract=PROTOCOL['runtime_budget']
    a=RuntimeBudget(tmp_path,contract,clock=lambda:now[0]);now[0]=39000.;a.close()
    b=RuntimeBudget(tmp_path,contract,clock=lambda:now[0]);now[0]+=601
    with pytest.raises(BudgetStop,match='SOFT'):b.before_batch()
    b.close();assert json.loads(b.path.read_text())['cumulative_seconds']==39601
    c=RuntimeBudget(tmp_path,contract,clock=lambda:now[0]);now[0]+=1800
    with pytest.raises(BudgetStop,match='HARD'):c.before_batch()
    c.close()

@pytest.mark.parametrize('projection',[36000,36001])
def test_runtime_launch_gate(projection):
    if projection<=36000:launch_gate(dict(safety_factor=1.25,projected_with_safety_factor=projection),PROTOCOL)
    else:
        with pytest.raises(ProtocolError,match='TIME_BUDGET_GATE_FAILED'):launch_gate(dict(safety_factor=1.25,projected_with_safety_factor=projection),PROTOCOL)

@pytest.mark.parametrize('projection',[
    {}, {'safety_factor':1.25}, {'projected_with_safety_factor':0},
    {'safety_factor':float('nan'),'projected_with_safety_factor':0},
    {'safety_factor':float('inf'),'projected_with_safety_factor':0},
    {'safety_factor':1.25,'projected_with_safety_factor':float('nan')},
    {'safety_factor':1.25,'projected_with_safety_factor':float('inf')},
    {'safety_factor':1.25,'projected_with_safety_factor':-1},
    {'safety_factor':True,'projected_with_safety_factor':0},
    {'safety_factor':1.25,'projected_with_safety_factor':False},
    {'safety_factor':'1.25','projected_with_safety_factor':0},
    {'safety_factor':1.25,'projected_with_safety_factor':'0'},
])
def test_runtime_gate_rejects_missing_nonfinite_negative_or_non_numeric_projection(projection):
    with pytest.raises(ProtocolError,match='TIME_BUDGET_GATE_FAILED'):
        launch_gate(projection,PROTOCOL)

def test_target_free_cache_hash_excludes_targets_and_includes_input_metadata():
    from canonical.cz_h1_posthoc_v22_accel12h.feature_cache import input_frame_hash
    T=frame(32);changed=T.copy();changed.y_true=1e90
    assert input_frame_hash(T)==input_frame_hash(changed)
    changed.origin+=1;assert input_frame_hash(T)!=input_frame_hash(changed)

def test_solver_and_h4_authority_source_unchanged():
    baseline=json.loads((MODULE/'ACCEL_BASELINE.json').read_text())
    original=ROOT/'canonical/cz_h1_posthoc_v22_final_conformance'
    for name,sha in baseline['baseline_python_sha256'].items():assert file_sha(original/name)==sha
    for name,sha in baseline['baseline_registry_sha256'].items():assert file_sha(MODULE/name)==sha

def test_support_limited_precedes_target_statistics_and_solver(tmp_path):
    from prism_benchmark import v2_urysohn as u
    from canonical.cz_h1_posthoc_v22_accel12h.feature_cache import prepare
    from canonical.cz_h1_posthoc_v22_accel12h.candidate_status import SupportLimited
    class ForbiddenStatistics:
        @staticmethod
        def from_design(*args):pytest.fail('Support-limited branch constructed target statistics')
    fake=SimpleNamespace(AmplitudeBasis=u.AmplitudeBasis,tensor_design=u.tensor_design,DesignStatistics=ForbiddenStatistics)
    p=SimpleNamespace(root=tmp_path,policy_id='same',u=fake,
        registration=dict(canonical_protocol_sha256={'K_MODEL_REGISTRY.json':'same'},authority_bindings_sha256_lf={}),
        v2=dict(row_caps=dict(minimum_rows_for_any_fitted_candidate=2,minimum_rows_per_free_parameter=2)))
    values=np.arange(36,dtype=float).reshape(3,12)
    with pytest.raises(SupportLimited):prepare(p,channel_name='x',profile=(2,256),mt=12,fit=frame(3),values=values,width=8,family='RANK_3_URYSOHN')

@pytest.mark.parametrize('name',['validation.parquet','test.parquet'])
def test_read_guard_blocks_bytes_before_open(tmp_path,name):
    from canonical.cz_h1_posthoc_v22_accel12h.read_guard import DevelopmentReadGuard
    guard=DevelopmentReadGuard()
    with pytest.raises(ProtocolError,match='Held-out'):guard.check(tmp_path/name)
    guard.check(tmp_path/'train.parquet');assert len(guard.reads)==1

def test_actual_hierarchical_dispatch_69_then_all_196_on_local_common_support(tmp_path,monkeypatch):
    from canonical.cz_h1_posthoc_v22_accel12h import k_channel as kc
    from canonical.cz_h1_posthoc_v22_accel12h.risk import mse_evidence
    from canonical.cz_h1_posthoc_v22_accel12h.partition import ordered_ids
    p=mock_pipe(tmp_path);p.K_registry,p.K_specs=load_registry(MODULE/'K_MODEL_REGISTRY.json')
    p.k_grid=lambda ch:(p.K_registry['profiles'],[4,8,12],256)
    p.accessor=SimpleNamespace(has_columns=lambda ch:True)
    p.native=lambda T,h:(T.origin.to_numpy()>=h)
    p.cap=lambda T,n:T
    p.v2=dict(row_caps=dict(minimum_rows_for_any_fitted_candidate=2,minimum_rows_per_free_parameter=2,single_channel_k_fit=99999),
        K_module=dict(penalties=dict(lambda_0=[.01])))
    def values(accessor,T,ch,profile,mt):return np.tile(T.origin.to_numpy(float)[:,None],(1,mt)),[[0,1]]
    p.v2_k=SimpleNamespace(profile_values=values,_als_kwargs=lambda config:{})
    p._profile_values=lambda T,ch,profile,mt:values(p.accessor,T,ch,profile,mt)
    p.u=SimpleNamespace(fit_contract=lambda x,y,f,m,l,**kw:dict(parameter_count=1,family=f),
        fit_prepared_contract=lambda prep,f,m,l,**kw:dict(parameter_count=1,family=f))
    from canonical.cz_h1_posthoc_v22_accel12h import feature_cache
    monkeypatch.setattr(feature_cache,'prepare',lambda pipe,**kw:SimpleNamespace(phi=np.empty((len(kw['fit']),kw['mt'],kw['width']))))
    def stable(grid,full,folded,**kw):return grid[0],full(grid[0]),folded(grid[0]),{}
    p.k=SimpleNamespace(select_smallest_stable_full_and_folds=stable,_candidate_valid=lambda *a:True)
    calls=[]
    def folded(pipe,prepared,profile,mt,j,fit,ev,mask,x,z,family,mx,lt,lx):
        calls.append(dict(profile=profile,mt=mt,family=family,score=ordered_ids(ev.loc[mask]),fit=ordered_ids(fit)))
        # A fully deterministic reporting fixture chooses a short coordinate.
        penalty=(profile[1]/1000.+mt/10000.) if family=='RANK_1_URYSOHN' and mx==4 and lt==lx==0 else 1.
        return mse_evidence(ev.y_true.to_numpy()[mask],ev.y_true.to_numpy()[mask]+penalty,ordered_ids(ev.loc[mask]))
    monkeypatch.setattr(kc,'_compute_fold',folded)
    out=kc.channel(p,frame(3000),tmp_path/'selection','x')
    receipt=json.loads((tmp_path/'selection/x_selection.json').read_text())
    assert receipt['temporal_screen_count']==69 and len(receipt['temporal_screen_status'])==69
    assert len(receipt['candidate_status'])==196
    assert set(r['spec'][2] for r in receipt['candidate_status'])==set(s[2] for s in p.K_specs)
    first=calls[:69*4];second=calls[69*4:]
    assert len(first)==276 and len(second)==784
    assert len(set((c['profile'],c['mt']) for c in first))==69
    star=out['reps'][0]['profile']
    assert all(list(c['profile'])==star and c['mt']==receipt['temporal_coordinate_star'][1] for c in second)
    assert len({tuple(c['score']) for c in first[::4]})==1

@pytest.mark.parametrize('comparison',[False,True])
@pytest.mark.parametrize('profile',[(1,8),(1,16),(1,32),(1,64),(1,96),(1,128)])
def test_cached_a_preparation_prediction_contract_audit_exact_authority(profile,comparison):
    from canonical.cz_h1_posthoc_v22_final_conformance.canonical_engine import CanonicalPipeline as Original
    from prism_benchmark import v21_a,v211_selection
    source=frame(600);source['residual']=np.cos(source.origin/13.)
    source['residual_source_role']='CROSS_FITTED_PREQUENTIAL';source['up_x']=np.sin(source.origin/23.)
    fit=source.iloc[:350].copy();ev=source.iloc[350:].copy()
    pipes=[]
    for cls in (Original,CanonicalPipeline):
        p=object.__new__(cls);p.mode='full_registry_smoke';p.policy_id='policy';p.fit_calls=0
        p.A_profiles=[(1,n) for n in (8,16,32,64,96,128)];p.A_max=128;p.a=v21_a;p.cert=v211_selection.numerical_contract_passes
        pipes.append(p)
    spec=(profile,.01,.1)
    old=Original.fit_residual_recipe(pipes[0],'A',spec,fit,ev,source,comparison=comparison)
    new=pipes[1].fit_residual_recipe('A',spec,fit,ev,source,comparison=comparison)
    assert old[0]==new[0] and np.array_equal(old[1],new[1]) and np.array_equal(old[2],new[2]) and old[3]==new[3]
    again=pipes[1].fit_residual_recipe('A',spec,fit,ev,source,comparison=comparison)
    assert new[0]==again[0] and np.array_equal(new[1],again[1]) and new[3]==again[3]

def test_k_model_dag_reuse_ignores_only_irrelevant_downstream_routes(tmp_path):
    p=mock_pipe(tmp_path);T=frame(1500)
    original=p._key('K',T,'MODEL')
    with ReplaySession(p,{'K':'BYPASSED','C':'BYPASSED','W':'ADMITTED'}):
        a=p._key('K',T,'MODEL');ca=p._key('C',T,'MODEL');wa=p._key('W',T,'MODEL')
    with ReplaySession(p,{'K':'BYPASSED','C':'ADMITTED','W':'BYPASSED'}):
        b=p._key('K',T,'MODEL');cb=p._key('C',T,'MODEL');wb=p._key('W',T,'MODEL')
    assert original==a==b and ca==cb and wa!=wb

@pytest.mark.parametrize('alpha',[1e-8,1e-4,.01,1.,100.,10000.])
@pytest.mark.parametrize('mu',[0.,.001,.1,1.,100.])
def test_mature_ar_reused_gram_is_bit_equivalent(alpha,mu):
    from prism_benchmark.v21_a import fit_mature_residual_ar
    from canonical.cz_h1_posthoc_v22_accel12h.ridge_path import MatureARPath
    rng=np.random.default_rng(411);x=rng.normal(size=(713,32));x[:,-1]=1.
    z=rng.normal(size=(320,32));target=rng.normal(size=713);up=rng.normal(size=(713,4));up[:,2]=0
    expected,contract=fit_mature_residual_ar(x,target,z,alpha=alpha,mu=mu,upstream_predictions=up)
    actual,new=MatureARPath(x,target,up).solve(z,alpha,mu)
    assert contract==new and np.array_equal(expected,actual)

@pytest.mark.parametrize('width',[1,4,8])
def test_k_target_free_materialization_identical_prepared_statistics(tmp_path,width):
    from prism_benchmark import v2_urysohn as u
    from canonical.cz_h1_posthoc_v22_accel12h.feature_cache import prepare
    rng=np.random.default_rng(217);T=frame(700);values=rng.normal(size=(700,8))
    p=SimpleNamespace(root=tmp_path,policy_id='same',u=u,
        registration=dict(canonical_protocol_sha256={'K_MODEL_REGISTRY.json':'same'},authority_bindings_sha256_lf={}),
        v2=dict(row_caps=dict(minimum_rows_for_any_fitted_candidate=2,minimum_rows_per_free_parameter=2)))
    old=u.prepare_contract_fit(values,T.y_true.to_numpy(),width)
    new=prepare(p,channel_name='x',profile=(2,16),mt=8,fit=T,values=values,width=width,family='RANK_1_URYSOHN')
    assert old.phi.shape==new.phi.shape and old.basis.metadata()==new.basis.metadata()
    for role in ('full_statistics','linear_statistics'):
        a=getattr(old,role);b=getattr(new,role)
        assert np.array_equal(a.gram,b.gram) and np.array_equal(a.rhs,b.rhs)

@pytest.mark.parametrize('width',[1,4,8])
@pytest.mark.parametrize('family',['LINEAR_DISTRIBUTED_LAG','RANK_1_URYSOHN','RANK_2_URYSOHN','RANK_3_URYSOHN','FULL_FINITE_URYSOHN'])
@pytest.mark.parametrize('lambdas',[(.01,0.,0.),(.1,.01,.1)])
def test_compact_k_statistics_all_authority_families_bit_equivalent(tmp_path,width,family,lambdas,monkeypatch):
    from prism_benchmark import v2_urysohn as u
    from canonical.cz_h1_posthoc_v22_accel12h.feature_cache import prepare
    if family=='LINEAR_DISTRIBUTED_LAG' and width!=1:return
    rng=np.random.default_rng(227);T=frame(700);values=rng.normal(size=(700,8))
    p=SimpleNamespace(root=tmp_path,policy_id='same',u=u,
        registration=dict(canonical_protocol_sha256={'K_MODEL_REGISTRY.json':'same'},authority_bindings_sha256_lf={}),
        v2=dict(row_caps=dict(minimum_rows_for_any_fitted_candidate=2,minimum_rows_per_free_parameter=2)))
    expected=u.fit_contract(values,T.y_true.to_numpy(),family,width,lambdas,als_max_iterations=10)
    compact=prepare(p,channel_name='x',profile=(2,16),mt=8,fit=T,values=values,width=width,family=family)
    actual=u.fit_prepared_contract(compact,family,width,lambdas,als_max_iterations=10)
    assert expected==actual
    z=rng.normal(size=(320,8))
    assert np.array_equal(u.predict_contract(z,expected),u.predict_contract(z,actual))
    monkeypatch.setattr(u,'tensor_design',lambda *a:pytest.fail('A cache hit reconstructed dense tensor design'))
    again=prepare(p,channel_name='x',profile=(2,16),mt=8,fit=T,values=values,width=width,family=family)
    assert actual==u.fit_prepared_contract(again,family,width,lambdas,als_max_iterations=10)

def test_lag_rows_constructed_once_across_overlapping_partitions(tmp_path):
    from canonical.cz_h1_posthoc_v22_accel12h.feature_cache import profile_values
    calls=[]
    def original(accessor,T,channel,profile,mt):
        calls.extend(zip(T.entity_id,T.origin));return np.column_stack((T.origin,T.origin*2.)).astype(float),[(0,2),(2,4)]
    T=frame(500);p=SimpleNamespace(root=tmp_path,policy_id='same',context={'C1_manifest':'same'},accessor=None,
        v2_k=SimpleNamespace(profile_values=original),
        registration=dict(canonical_protocol_sha256={'K_MODEL_REGISTRY.json':'same'},authority_bindings_sha256_lf={}))
    for subset in (T.iloc[:300],T.iloc[150:],T.iloc[::-1]):
        actual,intervals=profile_values(p,subset,'x',(2,8),4)
        assert np.array_equal(actual,np.column_stack((subset.origin,subset.origin*2.)).astype(float))
    assert len(calls)==500 and len(set(calls))==500
    T['y_true']+=100
    profile_values(p,T,'x',(2,8),4);assert len(calls)==500

def test_lag_cache_refreshes_after_another_worker_extends_it(tmp_path):
    from canonical.cz_h1_posthoc_v22_accel12h.feature_cache import profile_values
    calls=[]
    def original(accessor,T,channel,profile,mt):
        calls.extend(zip(T.entity_id,T.origin));return T.origin.to_numpy(float)[:,None],[(0,4)]
    def worker():return SimpleNamespace(root=tmp_path,policy_id='same',context={'C1_manifest':'same'},accessor=None,
        v2_k=SimpleNamespace(profile_values=original),registration=dict(canonical_protocol_sha256={'K_MODEL_REGISTRY.json':'same'},authority_bindings_sha256_lf={}))
    T=frame(500);a=worker();b=worker()
    profile_values(a,T.iloc[:300],'x',(2,8),4)
    profile_values(b,T.iloc[150:],'x',(2,8),4)
    actual,_=profile_values(a,T,'x',(2,8),4)
    assert np.array_equal(actual[:,0],T.origin.to_numpy(float)) and len(calls)==500

def test_sealed_fold_evidence_is_never_evicted_by_hot_budget(tmp_path):
    from canonical.cz_h1_posthoc_v22_accel12h.fold_evidence_cache import FoldEvidenceCache
    from canonical.cz_h1_posthoc_v22_accel12h.risk import mse_evidence
    path=tmp_path/'fold.sqlite3';cache=FoldEvidenceCache(path,1);calls=[]
    def compute():calls.append(1);return mse_evidence(np.arange(320.),np.arange(320.)+1)
    for i in range(8):cache.evaluate({'partition':i},('recipe',),compute);cache.flush()
    cache.close();cache=FoldEvidenceCache(path,1)
    for i in range(8):cache.evaluate({'partition':i},('recipe',),compute)
    assert len(calls)==8 and cache.snapshot()['evictions']==0
    cache.close()

def test_compact_checkpoint_binding_hash_rejects_tampering(tmp_path):
    cp=Checkpoints(tmp_path/'run',dict(source='frozen',data='frozen'))
    cp.save('MODEL','key',dict(answer=5));assert cp.load('MODEL','key')==dict(answer=5)
    path=cp.path('MODEL','key');body=json.loads(path.read_text());body['binding_sha256']='changed';path.write_text(json.dumps(body))
    with pytest.raises(ProtocolError,match='identity'):cp.load('MODEL','key')

def test_compact_input_cache_excludes_targets_and_target_mutation_rebinds_statistics(tmp_path):
    import pickle,zlib
    from prism_benchmark import v2_urysohn as u
    from canonical.cz_h1_posthoc_v22_accel12h.feature_cache import prepare
    rng=np.random.default_rng(234);T=frame(700);values=rng.normal(size=(700,8))
    p=SimpleNamespace(root=tmp_path,policy_id='same',u=u,
        registration=dict(canonical_protocol_sha256={'K_MODEL_REGISTRY.json':'same'},authority_bindings_sha256_lf={}),
        v2=dict(row_caps=dict(minimum_rows_for_any_fitted_candidate=2,minimum_rows_per_free_parameter=2)))
    kwargs=dict(channel_name='x',profile=(2,16),mt=8,values=values,width=4,family='RANK_1_URYSOHN')
    first=prepare(p,fit=T,**kwargs)
    path=next((tmp_path/'feature_cache/x').glob('*.pickle'));before=file_sha(path)
    payload=pickle.loads(zlib.decompress(path.read_bytes()))
    assert set(payload)=={'basis','shape','full_input_statistics','linear_input_statistics'}
    assert set(payload['full_input_statistics'])=={'gram','feature_sum','rows'}
    changed=T.copy();changed['y_true']+=np.cos(changed.origin/5.)
    second=prepare(p,fit=changed,**kwargs)
    expected=u.prepare_contract_fit(values,changed.y_true.to_numpy(),4)
    assert np.array_equal(second.full_statistics.rhs,expected.full_statistics.rhs)
    assert not np.array_equal(first.full_statistics.rhs,second.full_statistics.rhs)
    assert file_sha(path)==before and len(list((tmp_path/'target_statistics_cache/x').glob('*.pickle')))==2

@pytest.mark.parametrize('family',['LINEAR_DISTRIBUTED_LAG','RANK_1_URYSOHN','RANK_2_URYSOHN','RANK_3_URYSOHN','FULL_FINITE_URYSOHN'])
def test_fixed_recipe_fit_cache_ignores_score_support_but_not_fit_targets(tmp_path,family):
    from prism_benchmark import v2_urysohn as u
    from canonical.cz_h1_posthoc_v22_accel12h.feature_cache import prepare
    from canonical.cz_h1_posthoc_v22_accel12h.fit_contract_cache import fit_fixed_recipe
    rng=np.random.default_rng(271);T=frame(700);values=rng.normal(size=(700,8));width=1 if family=='LINEAR_DISTRIBUTED_LAG' else 4
    p=SimpleNamespace(root=tmp_path,policy_id='same',u=u,fit_calls=0,
        registration=dict(canonical_protocol_sha256={'K_MODEL_REGISTRY.json':'same'},authority_bindings_sha256_lf={}),
        v2=dict(row_caps=dict(minimum_rows_for_any_fitted_candidate=2,minimum_rows_per_free_parameter=2)))
    kwargs=dict(channel_name='x',profile=(2,16),mt=8,values=values,width=width,family=family)
    prepared=prepare(p,fit=T,**kwargs)
    expected=u.fit_contract(values,T.y_true.to_numpy(),family,width,(.01,.01,.01),als_max_iterations=10)
    first=fit_fixed_recipe(p,prepared,family,width,(.01,.01,.01),als_max_iterations=10)
    assert first==expected and p.fit_calls==1
    # Two distinct scoring supports request the same fit. Neither scoring
    # inputs nor labels are in the fixed-recipe numerical node's identity.
    for n in (320,61):
        actual=fit_fixed_recipe(p,prepare(p,fit=T,**kwargs),family,width,(.01,.01,.01),als_max_iterations=10)
        z=rng.normal(size=(n,8));assert np.array_equal(u.predict_contract(z,actual),u.predict_contract(z,expected))
    assert p.fit_calls==1
    first['intercept']=999 # Caller mutation must not poison the sealed fit.
    again=fit_fixed_recipe(p,prepared,family,width,(.01,.01,.01),als_max_iterations=10)
    assert again==expected and p.fit_calls==1
    changed=T.copy();changed['y_true']+=np.cos(changed.origin)
    fit_fixed_recipe(p,prepare(p,fit=changed,**kwargs),family,width,(.01,.01,.01),als_max_iterations=10)
    assert p.fit_calls==2

@pytest.mark.parametrize('kind',['continuous','binary','low_cardinality'])
@pytest.mark.parametrize('family',['LINEAR_DISTRIBUTED_LAG','RANK_1_URYSOHN','FULL_FINITE_URYSOHN'])
def test_score_design_reuse_preserves_batch_broadcast_prediction_and_risk(tmp_path,kind,family,monkeypatch):
    from prism_benchmark import v2_urysohn as u
    from canonical.cz_h1_posthoc_v22_accel12h.feature_cache import prepare,predict_cached_score,predict_materialized
    from canonical.cz_h1_posthoc_v22_accel12h.risk import mse_evidence
    rng=np.random.default_rng(319);T=frame(719);ev=frame(320,start=720)
    if kind=='continuous':values=rng.normal(size=(719,8));z=rng.normal(size=(320,8))
    else:
        high=2 if kind=='binary' else 5
        values=rng.integers(0,high,size=(719,8)).astype(float);z=rng.integers(0,high,size=(320,8)).astype(float)
    width=1 if family=='LINEAR_DISTRIBUTED_LAG' else 4
    p=SimpleNamespace(root=tmp_path,policy_id='same',u=u,
        registration=dict(canonical_protocol_sha256={'K_MODEL_REGISTRY.json':'same'},authority_bindings_sha256_lf={}),
        v2=dict(row_caps=dict(minimum_rows_for_any_fitted_candidate=2,minimum_rows_per_free_parameter=2)))
    ct=u.fit_contract(values,T.y_true.to_numpy(),family,width,(.01,.01,.01),als_max_iterations=10)
    expected=u.predict_contract(z,ct)
    prepared=prepare(p,channel_name='x',profile=(2,16),mt=8,fit=T,values=values,width=width,family=family)
    original=u.tensor_design;calls=[]
    def counted(*a,**kw):calls.append(1);return original(*a,**kw)
    monkeypatch.setattr(u,'tensor_design',counted)
    first=predict_cached_score(p,prepared,z,ct,ev)
    changed=ev.copy();changed['y_true']+=1e4
    second=predict_cached_score(p,prepared,z,ct,changed)
    assert len(calls)==1 and np.array_equal(first,expected) and np.array_equal(second,expected)
    assert mse_evidence(ev.y_true,first)==mse_evidence(ev.y_true,expected)
    long=np.tile(z,(27,1));phi=original(long,prepared.basis)
    assert np.array_equal(predict_materialized(p,phi,ct),u.predict_contract(long,ct))
