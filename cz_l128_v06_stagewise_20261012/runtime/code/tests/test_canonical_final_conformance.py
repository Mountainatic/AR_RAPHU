import itertools,json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from canonical.cz_h1_posthoc_v22_final_conformance import model_tuning as mt
from canonical.cz_h1_posthoc_v22_final_conformance.canonical_engine import CanonicalPipeline
from canonical.cz_h1_posthoc_v22_final_conformance.fold_geometry import folds,placement,geometry_audit,_optimize,require_top_level_geometry
from canonical.cz_h1_posthoc_v22_final_conformance.partition import validate_partition,ordered_ids,support_hash
from canonical.cz_h1_posthoc_v22_final_conformance.invariants import ProtocolError,NumericallyInvalid,digest
from canonical.cz_h1_posthoc_v22_final_conformance.risk import registered_best,mse_evidence,paired_route
from canonical.cz_h1_posthoc_v22_final_conformance.candidate_status import SupportLimited,DomainNotApplicable,no_available_model,require_k_support
from canonical.cz_h1_posthoc_v22_final_conformance.checkpoint import Checkpoints
from canonical.cz_h1_posthoc_v22_final_conformance.maximal_inference import build_support
from canonical.cz_h1_posthoc_v22_final_conformance.pointwise import paired_pointwise
from canonical.cz_h1_posthoc_v22_final_conformance.a_state import feature_rows,common_scoring_mask,build_a_training_state_from_cf_residual

G={'block_rows':320,'count':4,'end_fractions':[.4,.6,.8,1.]}
ROOT=Path(__file__).parents[1];MODULE=ROOT/'canonical/cz_h1_posthoc_v22_final_conformance'

def frame(n=3000,entity='segment',start=1):
    o=np.arange(start,start+n)
    return pd.DataFrame(dict(entity_id=entity,origin=o,split='train',target_stop_exclusive=o+1,
        latest_available_target_index=o-1,dependency_start=o-1,dependency_stop_exclusive=o+1,
        sample_id=[entity+str(i) for i in o],view_sample_id=[entity+str(i) for i in o],y_true=np.sin(o)))

def test_entity_placement_exact_nonoverlap_metadata_only():
    T=pd.concat([frame(1100,'early'),frame(1900,'late')],ignore_index=True)
    fs=folds(T,G);assert len(fs)==4
    assert all(len(e)==320 and T.iloc[e].entity_id.nunique()==1 for _,e in fs)
    assert all(a[-1]<b[0] for (_,a),(_,b) in zip(fs,fs[1:]))
    audit=geometry_audit(T,G);assert audit['score_blocks'][0]['displacement_rows']!=0
    changed=T.copy();changed.y_true=1e100
    assert placement(T,G)==placement(changed,G)
    assert all('support_hash' in r and 'displacement_fraction' in r for r in audit['score_blocks'])

@pytest.mark.parametrize('seed',range(12))
def test_joint_optimizer_matches_exhaustive_sum_max_lex(seed):
    rng=np.random.default_rng(seed);legal=tuple(sorted(rng.choice(np.arange(320,2300,40),size=10,replace=False).tolist()))
    nominal=tuple(sorted(rng.integers(320,2300,size=4).tolist()))
    combinations=[x for x in itertools.combinations(legal,4) if all(a<=b-320 for a,b in zip(x,x[1:]))]
    objective=lambda x:(sum(abs(a-b) for a,b in zip(x,nominal)),max(abs(a-b) for a,b in zip(x,nominal)),x)
    assert _optimize(legal,nominal)==(min(combinations,key=objective) if combinations else None)

def test_geometry_tie_lexical_and_small_base_case():
    legal=(320,640,960,1280,1600)
    nominal=(480,800,1120,1440)
    assert _optimize(legal,nominal)==(320,640,960,1280)
    assert folds(frame(1279),G)==[]
    with pytest.raises(ProtocolError,match='HARD_FAIL_PREFLIGHT'):require_top_level_geometry(frame(1279),G)
    p=object.__new__(CanonicalPipeline);state=p._base_case('K',None)
    assert state['routing_status']=='POLICY_NOT_ESTIMABLE_IN_PARTITION'
    assert not state['admission_vote_produced'] and state['geometry_status']=='SCORING_GEOMETRY_UNAVAILABLE'

@pytest.mark.parametrize('fault',['sample_id','origin','order','entity','maturity','available'])
def test_partition_invariants_fail_fast(fault):
    T=frame(20)
    if fault=='sample_id':T.loc[1,'sample_id']=T.loc[0,'sample_id']
    elif fault=='origin':T.loc[1,'origin']=T.loc[0,'origin']
    elif fault=='order':T.loc[3,'origin']=0
    elif fault=='entity':T.loc[2:3,'entity_id']='second'
    elif fault=='maturity':T.loc[1,'target_stop_exclusive']=T.loc[1,'origin']
    else:T.loc[1,'latest_available_target_index']=T.loc[1,'origin']
    with pytest.raises(ProtocolError):validate_partition(T)

def test_equal_count_different_ordered_ids_rejected():
    a=mse_evidence([1,2],[0,0],['a','b']);b=mse_evidence([1,2],[0,0],['b','a'])
    with pytest.raises(ProtocolError,match='supports'):registered_best({'a':a,'b':b})

def test_identity_only_target_mutation_excluded_from_model_but_retained_outer():
    T=frame(320);mask=np.arange(320)>=128;parent=np.zeros(320);child=parent.copy();child[mask]=T.y_true.to_numpy()[mask]
    a=mt.risk(T,T.y_true,child,mask)
    changed=T.y_true.to_numpy().copy();changed[~mask]+=100
    b=mt.risk(T,changed,child,mask);assert a==b
    before=paired_route(T.y_true,parent,child);after=paired_route(changed,parent,child)
    assert after['parent_oof_risk']>before['parent_oof_risk'] and after['child_oof_risk']>before['child_oof_risk']
    assert after['n']==320

@pytest.mark.parametrize('stage',['K','C','W','A'])
@pytest.mark.parametrize('status',['SUPPORT_LIMITED','DOMAIN_NOT_APPLICABLE','NUMERICALLY_INVALID'])
def test_recipe_status_never_produces_zero_risk_or_bypass(tmp_path,stage,status):
    p=SimpleNamespace(checkpoints=Checkpoints(tmp_path/'run',dict(frozen='same')))
    cls={'SUPPORT_LIMITED':SupportLimited,'DOMAIN_NOT_APPLICABLE':DomainNotApplicable,'NUMERICALLY_INVALID':NumericallyInvalid}[status]
    def bad():raise cls('retained')
    spec=('channel',(2,8),4,'LINEAR_DISTRIBUTED_LAG',1,0.,0.) if stage=='K' else ('family',1)
    result=mt.candidate(p,stage,'partition',spec,bad)
    assert result['status']==status and 'risk' not in result and result['status']!='BYPASSED'
    assert mt.candidate(p,stage,'partition',spec,lambda:pytest.fail('sealed result recomputed'))==result

def test_all_support_limited_and_all_numeric_are_distinct():
    assert no_available_model(['SUPPORT_LIMITED']*4)=='SUPPORT_LIMITED'
    with pytest.raises(NumericallyInvalid):no_available_model(['NUMERICALLY_INVALID']*4)
    with pytest.raises(SupportLimited):require_k_support(dict(family='LINEAR_DISTRIBUTED_LAG',parameter_count=100),3,dict(row_caps=dict(minimum_rows_for_any_fitted_candidate=2,minimum_rows_per_free_parameter=2)))

def test_solver_runtime_failure_tagged_but_protocol_failure_hard_stops(tmp_path):
    p=SimpleNamespace(checkpoints=Checkpoints(tmp_path/'run',dict(frozen='same')))
    def solver():raise RuntimeError('solver diverged')
    def protocol():raise ProtocolError('sealed artifact changed')
    r=mt.candidate(p,'W','partition',('family',1),solver)
    assert r['status']=='NUMERICALLY_INVALID' and 'risk' not in r
    with pytest.raises(ProtocolError):mt.candidate(p,'W','other',('family',1),protocol)

@pytest.mark.parametrize('empty_first_fold',[False,True])
def test_c_all_alpha_recipes_inner_minimum_not_first_certified(tmp_path,empty_first_fold):
    p=object.__new__(CanonicalPipeline);p.mode='solver_smoke';p.fit_calls=0;p.policy_id='fixed';p.artifacts=[]
    p.protocol={'geometry':{'inner':G}};p.C_specs=[('ADDITIVE_COMPRESSED',1.),('ADDITIVE_COMPRESSED',2.),('ADDITIVE_JOINT_BASIS',1.),('ADDITIVE_JOINT_BASIS',2.)]
    p.checkpoints=Checkpoints(tmp_path/'run',dict(protocol='same'));p.learn=lambda stage,T:dict(representations=[] if empty_first_fold and len(T)<500 else [{},{}]);p.emit=lambda state,T:np.zeros(len(T))
    p._c_design=lambda parent,fit,ev:({f:(fit.y_true.to_numpy()[:,None],ev.y_true.to_numpy()[:,None]) for f in ('ADDITIVE_COMPRESSED','ADDITIVE_JOINT_BASIS')},[],np.ones(len(fit),bool),np.ones(len(ev),bool))
    called=[]
    def solve(x,y,z,alpha):called.append(alpha);return z[:,0]+(alpha-2),dict(alpha=alpha)
    p.c=SimpleNamespace(_ridge_fit=solve);p.cert=lambda ct:True
    state=mt.tune_c(p,frame(2000),dict(representations=[{},{}]),tmp_path/'evidence')
    assert state['selected_alpha']==2. and set(called)=={1.,2.} and 0. not in called
    audit=json.loads((tmp_path/'evidence/C_ALPHA_SELECTION_AUDIT.json').read_text());assert len(audit['candidate_status'])==4
    hashes={r['risk']['ordered_score_support_hash'] for r in audit['candidate_status']};assert len(hashes)==1
    assert audit['common_support'][0]['rows']==(0 if empty_first_fold else 320)

def test_c_registry_exact_scope_and_no_implicit_zero():
    r=json.loads((MODULE/'C_MODEL_REGISTRY.json').read_text());assert r['families']==['ADDITIVE_COMPRESSED','ADDITIVE_JOINT_BASIS']
    assert 0 not in r['alpha_grid'] and len(r['candidates'])==2*len(r['alpha_grid'])

def test_outer_model_no_admitted_and_id_survives_global_freeze():
    T=frame(320);parent=np.zeros(320);child=T.y_true.to_numpy();mask=np.ones(320,bool)
    kwargs=dict(stage='K',kind='outer_oof',parent=parent,model_child=child,deployed=child,row_domain=mask,policy_id='policy',partition_id='fit',parameters={'MODEL_hash':'fold-only'},fold=0)
    with pytest.raises(ProtocolError):paired_pointwise(T,routing_status='ADMITTED',**kwargs)
    rows,proof=paired_pointwise(T,routing_status=None,**kwargs)
    assert rows.global_routing_status.isna().all() and 'deployment_model_id' not in rows
    original=rows.outer_realization_id.copy();frozen=rows.copy();frozen['global_routing_status']='BYPASSED';frozen['global_deployed_prediction']=parent
    assert frozen.outer_realization_id.equals(original) and proof['parameters']=={'MODEL_hash':'fold-only'}

@pytest.mark.parametrize('split',['validation','test'])
def test_maximal_metadata_subset_and_fixed_within_entity_boundary(split):
    task='CZ_DIAM_RAW2S_CURRENT_L256_H1';direction='Rod_1_to_Rod_2'
    def ids(T):
        T=T.drop(columns='y_true').copy()
        import hashlib
        T['base_origin_id']=[hashlib.sha256(f'{task}|{direction}|segment|{o}'.encode()).hexdigest() for o in T.origin]
        T['view_sample_id']=[hashlib.sha256(('dynamic|'+s).encode()).hexdigest() for s in T.base_origin_id]
        T['sample_id']=T.view_sample_id
        return T
    old=ids(frame(20,start=256));old['split']=split
    raw=pd.DataFrame(dict(entity_id='segment',row_in_entity=np.arange(276)))
    other=ids(frame(30)) if split=='validation' else old.iloc[:0]
    new,audit=build_support(old,other,raw,split=split,task=task,direction=direction)
    assert set(ordered_ids(old))<=set(ordered_ids(new)) and audit['split_boundaries_unchanged']
    assert audit['restored_rows']==(0 if split=='validation' else 255)
    assert not audit['target_values_read']
    with pytest.raises(ProtocolError):build_support(old.assign(y_true=1.),other,raw,split=split,task=task,direction=direction)

def test_w_and_a_common_score_intersection_excludes_identity_rows():
    from canonical.cz_h1_posthoc_v22_final_conformance.a_registry import load_a_registry
    from canonical.cz_h1_posthoc_v22_final_conformance.invariants import file_sha
    _,profiles=load_a_registry(MODULE/'A_TEMPORAL_REGISTRY.json',file_sha(MODULE/'A_TEMPORAL_REGISTRY.json'))
    source=frame(500);source['residual']=source.y_true;source['residual_source_role']='CROSS_FITTED_PREQUENTIAL'
    typed=build_a_training_state_from_cf_residual(source,policy_id='fixed',h_steps=1)
    mask=common_scoring_mask(source,typed,profiles,128)
    assert not mask[:128].any() and mask[128:].all()
    for profile in profiles:
        _,native,_=feature_rows(source,typed,profile,128,0.)
        assert np.all(native[mask])
    r=mt.risk(source,source.residual,np.zeros(500),mask)
    assert r.rows==372 and r.ordered_score_support_hash==support_hash(source.loc[mask])

def test_full_registry_prefix_reader_and_explicit_train_resume():
    from canonical.cz_h1_posthoc_v22_final_conformance import data
    import inspect
    assert "context['protocol'][mode]['prefix_rows']" in inspect.getsource(data.load_development)
    from canonical.cz_h1_posthoc_v22_final_conformance import run_canonical
    assert 'resume=args.resume' in inspect.getsource(run_canonical.main)
    assert 'run_development(args.output_root,resume=True)' not in inspect.getsource(run_canonical.main)

def test_frozen_a_h1_w_and_final_replay_regressions():
    p=json.loads((MODULE/'FINAL_PROTOCOL.json').read_text())
    assert p['A_initialization']['target_test']=='EMPTY_PER_ENTITY_SEGMENT' and not p['A_initialization']['pre_score_warmup']
    assert p['final_development_cf']['block_rows']==320
    assert not any(p['execution_authorization'][k] for k in ('formal_development_training','source_validation_reporting','target_test_inference'))
    assert (MODULE/'A_TEMPORAL_REGISTRY.json').read_bytes()==(ROOT/'canonical/cz_h1_posthoc_v22/A_TEMPORAL_REGISTRY.json').read_bytes()
    from canonical.cz_h1_posthoc_v22_final_conformance.temporal_support import final_replay_blocks
    T=pd.concat([frame(333,'A'),frame(450,'B')],ignore_index=True)
    blocks=final_replay_blocks(T,320);assert sum(len(e) for _,e in blocks)==len(T)
    assert any(len(e)<320 for _,e in blocks) and all(not len(f) or f[-1]<e[0] for f,e in blocks)

def test_k_exact_common_and_selected_reference_assembly(tmp_path):
    p=object.__new__(CanonicalPipeline);p.mode='full_registry_smoke';p.fit_calls=0;p.policy_id='fixed';p.artifacts=[]
    p.protocol={'geometry':{'inner':G}};p.K_specs=[((2,8),4,'LINEAR_DISTRIBUTED_LAG',1,0.,0.),((2,256),4,'LINEAR_DISTRIBUTED_LAG',1,0.,0.)]
    p._support_id=support_hash
    p.checkpoints=Checkpoints(tmp_path/'run',dict(protocol='same'))
    p.k_grid=lambda ch:([(2,8),(2,256)],[4],256)
    p.accessor=SimpleNamespace(has_columns=lambda ch:True);p.native=lambda T,L:T.origin.to_numpy()>=L;p.cap=lambda T,n:T
    p.v2={'row_caps':{'single_channel_k_fit':99999,'minimum_rows_for_any_fitted_candidate':2,'minimum_rows_per_free_parameter':0},
          'K_module':{'penalties':{'lambda_tau':[0.],'pilot':{'lambda_0':1.},'lambda_0':[1.]}}}
    def values(accessor,T,ch,profile,mt):return np.column_stack([T.origin,np.full(len(T),profile[1]),T.origin*0,T.origin*0]),[]
    def contract(x):return dict(family='LINEAR_DISTRIBUTED_LAG',parameter_count=5,history=int(x[0,1]),certificate={'status':'PASS'})
    p.v2_k=SimpleNamespace(profile_values=values,_als_kwargs=lambda cfg:{})
    p.u=SimpleNamespace(prepare_contract_fit=lambda x,y,w:SimpleNamespace(phi=np.zeros((len(x),4,1)),values=x),
        fit_prepared_contract=lambda prep,*a,**k:contract(prep.values),fit_contract=lambda x,*a,**k:contract(x),
        predict_contract=lambda x,ct:np.sin(x[:,0])+(0 if ct['history']==8 else 10))
    p.k=SimpleNamespace(_candidate_valid=lambda c,n,cfg:True,
        select_smallest_stable_full_and_folds=lambda l,full,folded,**kw:(l[0],full(l[0]),folded(l[0]),[]))
    T=frame(1500);state=mt.tune_k(p,T,tmp_path/'tuning')
    audit=state['reference_assembly_audit'];assert all(v==8 for v in audit['selected_representation_history_by_channel'].values())
    risks=audit['reference_risk_by_channel'];assert len({r['ordered_score_support_hash'] for r in risks.values()})==1
    assert not audit['outer_held_out_used'] and state['reference_channel']=='main_heater_power'
    selected=json.loads((tmp_path/'tuning/main_heater_power_selection.json').read_text())
    assert any(a['rows']>b['rows'] for a,b in zip(audit['assembly_support_by_fold'],selected['common_support']))
    assert selected['candidate_status'][1]['status']=='SUPPORT_LIMITED'
    assert selected['selected_risk']['ordered_score_support_hash']!=risks['main_heater_power']['ordered_score_support_hash']
    for row in selected['common_support']:
        assert row['ordered_score_support_hash']==digest(row['ordered_score_ids'])
    state.update(stage='K',routing_status='BYPASSED')
    assert len(state['representations'])==4 and not p.emit(state,T).any()
    q,native,_=p.response_coordinate(dict(stage='C',parent=state,routing_status='BYPASSED'),T)
    assert np.ptp(q[native])>0
    count=p.fit_calls
    limited=mt.tune_k(p,frame(1280),tmp_path/'small')
    assert limited['realization_status']=='SUPPORT_LIMITED' and not limited['domain'] and not limited['representations']
    assert set(limited['channel_status'].values())=={'SUPPORT_LIMITED'} and p.fit_calls==count

@pytest.mark.parametrize('empty_first_fold',[False,True])
def test_w_recipe_selection_executes_only_coordinate_common_support(tmp_path,monkeypatch,empty_first_fold):
    p=object.__new__(CanonicalPipeline);p.mode='full_registry_smoke';p.fit_calls=0;p.policy_id='fixed';p.artifacts=[]
    p.W_specs=[('GOOD',1,0.,0.,1),('BAD',1,0.,0.,1)];p.protocol={'geometry':{'inner':G}}
    p._support_id=support_hash
    p.checkpoints=Checkpoints(tmp_path/'run',dict(frozen='same'));p._w_domain=lambda q:{'pass_':len(q)>1}
    source=frame(1920);source['_position']=np.arange(1920);source['CF_fold']=np.r_[-np.ones(640,int),np.repeat(np.arange(4),320)]
    source['coordinate_available']=True
    for j in range(4):source.loc[640+j*320:649+j*320,'coordinate_available']=False
    if empty_first_fold:source.loc[640:959,'coordinate_available']=False
    source['response_coordinate']=source.origin.astype(float);source['residual']=1.
    source.loc[~source.coordinate_available,'residual']=1000.
    path=tmp_path/'selection';path.write_bytes(b'frozen')
    p.response_coordinate=lambda parent,T:(T.origin.to_numpy(float),np.ones(len(T),bool),{})
    p.stream=lambda upstream,T:(source,[],str(path));p.cert=lambda ct:True
    called=[]
    def fit(q,r,z,**kw):
        called.append((len(q),np.max(r),len(z)))
        return np.full(len(z),1. if kw['family']=='GOOD' else 0.),{'family':kw['family']}
    p.w=SimpleNamespace(fit_w_correction=fit)
    monkeypatch.setattr(mt,'replay',lambda *args:(source,{'coverage':{'replay_rows':1920}},str(path)))
    state=mt.tune_residual(p,'W',frame(3000),{},tmp_path/'tuning')
    assert state['recipe'][0]=='GOOD' and all(maximum==1 for _,maximum,_ in called)
    assert [r['rows'] for r in state['model_common_support']]==([0,310,310,310] if empty_first_fold else [310]*4)
    assert len({r['ordered_score_support_hash'] for r in state['model_risks'].values()})==1

@pytest.mark.parametrize('mode,expected',[('solver_smoke',1700),('full_registry_smoke',2100)])
def test_actual_prefix_loading_uses_own_protocol_parameter(monkeypatch,mode,expected):
    from canonical.cz_h1_posthoc_v22_final_conformance import data
    monkeypatch.setattr(data,'file_sha',lambda p:'fixed')
    monkeypatch.setattr(pd,'read_parquet',lambda p:frame(3000))
    monkeypatch.setattr(data,'_labels',lambda samples,paths:samples)
    context=dict(mode=mode,shared='irrelevant',C1_manifest={'source_base_sha256':'fixed'},protocol={
        'execution_authorization':{'development_only_solver_smoke':True},'solver_smoke':{'prefix_rows':1700},'full_registry_smoke':{'prefix_rows':2100}})
    assert len(data.load_development(context))==expected

@pytest.mark.parametrize('resume',[False,True])
def test_actual_train_cli_requires_explicit_resume(monkeypatch,resume):
    import sys
    from canonical.cz_h1_posthoc_v22_final_conformance import run_canonical
    calls=[];runner=SimpleNamespace(run_development=lambda root,*,resume:calls.append(resume))
    monkeypatch.setattr(run_canonical,'open_case',lambda **kw:(runner,None,{}))
    args=['runner','train','--authority-project','unused','--c1-root','unused','--release-dir','unused','--release-sha','pinned','--case-id','Rod_1_to_Rod_2','--output-root','unused']
    if resume:args+=['--resume']
    monkeypatch.setattr(sys,'argv',args);run_canonical.main();assert calls==[resume]

def test_new_namespace_w_fallback_and_a_cold_start():
    p=object.__new__(CanonicalPipeline)
    K={'stage':'K','routing_status':'BYPASSED','reference_channel':'joint_lift'};C={'stage':'C','routing_status':'BYPASSED','parent':K}
    p.predict=lambda state,T:(T.origin.to_numpy(float),np.ones(len(T),bool),{})
    q,native,proof=p.response_coordinate(C,frame(20));assert np.ptp(q)>0 and native.all() and proof['interface']=='PSI_K_RETAINED_REFERENCE_MODEL_SCALAR'
    p.emit=lambda state,T:np.full(len(T),3.)
    def predict(state,T,*,a_state):
        x,mask,audit=feature_rows(T,a_state,(1,8),128,0.);out=np.full(len(T),3.);out[mask]+=x.sum(axis=1)
        return out,mask,audit
    p.predict=predict
    T=frame(30);a,mask,proof=p.online_a({'parent':{'partition_model_id':'W'}},T)
    assert not mask[:8].any() and mask[8:].all() and not proof['training_backcast_used']
    T.loc[20:,'y_true']+=1e6;b,_,_=p.online_a({'parent':{'partition_model_id':'W'}},T)
    np.testing.assert_array_equal(a[:21],b[:21])

def test_frozen_registry_payload_hash_and_no_identity():
    from canonical.cz_h1_posthoc_v22_final_conformance.registries import load_registry
    for name in ('K_MODEL_REGISTRY.json','C_MODEL_REGISTRY.json','W_MODEL_REGISTRY.json'):
        body,specs=load_registry(MODULE/name)
        assert not body['identity_is_candidate'] and len(specs)==len(body['candidates'])
        assert 'BYPASS' not in json.dumps(body['candidates']) and 'EXACT_ZERO' not in json.dumps(body['candidates'])
    k,_=load_registry(MODULE/'K_MODEL_REGISTRY.json')
    assert len(k['profiles'])==23 and max(h for _,h in k['profiles'])==256 and k['m_tau']==[4,8,12]

def test_maximal_support_reader_never_projects_values(monkeypatch):
    from canonical.cz_h1_posthoc_v22_final_conformance import maximal_inference as mi
    import sys
    class PQ:
        @staticmethod
        def ParquetFile(path):return SimpleNamespace(schema=SimpleNamespace(names=['entity_id','origin','view_sample_id','y_true','current_level','crystal_diameter']))
        @staticmethod
        def read_table(path,columns):
            assert columns==['entity_id','origin','view_sample_id']
            return SimpleNamespace(to_pandas=lambda:pd.DataFrame(columns=columns))
    monkeypatch.setitem(sys.modules,'pyarrow.parquet',PQ)
    monkeypatch.setitem(sys.modules,'pyarrow',SimpleNamespace(parquet=PQ,__path__=[]))
    assert list(mi.metadata('metadata-only').columns)==['entity_id','origin','view_sample_id']

def test_exact_authority_k_w_a_registries_and_hashes_unchanged():
    pytest.importorskip('pyreadr')
    from prism_benchmark.v211_config import load_v211_configs
    from prism_benchmark import v211_k,v211_w,v2_k
    from prism_benchmark.cz_l256_nowcast import view
    from canonical.cz_h1_posthoc_v22_final_conformance.registries import load_registry
    from canonical.cz_h1_posthoc_v22_final_conformance.invariants import source_sha
    v211,v21,v2=load_v211_configs(ROOT,protocol='cz_h1_k_temporal_registry_v2')
    task=view('dynamic',h_steps=1,task_id='CZ_DIAM_RAW2S_CURRENT_L256_H1')
    _,ks=load_registry(MODULE/'K_MODEL_REGISTRY.json');_,ws=load_registry(MODULE/'W_MODEL_REGISTRY.json');_,aas=load_registry(MODULE/'A_MODEL_REGISTRY.json')
    for ch in ('main_heater_power','joint_lift','crystal_rotation','crucible_rotation'):
        profiles=v2_k.channel_profiles(task,ch,v2)
        assert ks==v211_k._structural_specs(profiles,list(v21['K_C']['m_tau']),v2['K_module']['penalties']['lambda_tau'],v21['K_C']['m_x'],v2['K_module']['penalties']['lambda_x'])
    assert ws==v211_w._w_candidates(v211,v21,v2,1,False)
    assert aas==[(tuple(profile),float(a),float(mu)) for profile in json.loads((MODULE/'A_TEMPORAL_REGISTRY.json').read_text())['profiles_in_registered_order'] for a in v2['A_module']['ridge_alpha_grid'] for mu in v21['A']['soft_overlap_mu']]
    frozen=json.loads((ROOT/'configs/cz_h1_k_temporal_registry_v2/AUTHORITY_UNCHANGED_BINDINGS.json').read_text())['files']
    assert all(source_sha(ROOT/relative)==h for relative,h in frozen.items())

def test_actual_authority_w_nonconstant_q_zero_baseline_new_namespace(tmp_path):
    pytest.importorskip('pyreadr')
    from canonical.cz_h1_posthoc_v22_final_conformance.invariants import file_sha
    protocol=json.loads((MODULE/'FINAL_PROTOCOL.json').read_text())
    registration=dict(canonical_protocol_sha256={'A_TEMPORAL_REGISTRY.json':file_sha(MODULE/'A_TEMPORAL_REGISTRY.json')})
    p=CanonicalPipeline(dict(authority=ROOT,shared=tmp_path,registration=registration,protocol=protocol,policy_id='synthetic'),tmp_path/'unused',mode='metadata_only')
    p.mode='solver_smoke';T=frame(100);T['response_coordinate']=np.linspace(-2,2,100);T['coordinate_available']=True
    T['parent_prediction']=0.;T['residual']=T.response_coordinate**2+T.response_coordinate
    ct,correction,mask,_=p.fit_residual_recipe('W',p.W_specs[0],T,T,T)
    assert ct is not None and mask.all() and np.ptp(correction)>0 and p.fit_calls==1

def test_maximal_inference_metadata_byte_drift_before_label_projection(tmp_path,monkeypatch):
    pytest.importorskip('pyarrow')
    from canonical.cz_h1_posthoc_v22_final_conformance import data,runtime_seal
    source=tmp_path/'source';source.mkdir();p=tmp_path/'test.parquet';p.write_bytes(b'changed')
    monkeypatch.setattr(runtime_seal,'verify_c1',lambda *a:None)
    monkeypatch.setattr(data,'_labels',lambda *a:pytest.fail('target projection before byte verification'))
    context=dict(mode='inference_only',deployment_checkpoint_verified=True,inference_operation='target_test',shared=source,case_id='case',release_dir=tmp_path,
        C1_manifest={'source_shared':str(source)},protocol={'execution_authorization':{'target_test_inference':True}},
        registration={'C1_case_bindings':{'case':{'maximal_inference_metadata':{'test':{'relative':'test.parquet','sha256':'sealed-old'}}}}})
    with pytest.raises(ProtocolError,match='metadata drift'):data.load_inference(context,'target_test')

def test_authority_c_predictive_alpha_nonfirst_real_synthetic(tmp_path):
    pytest.importorskip('pyreadr')
    from prism_benchmark import v2_c,v211_selection
    from canonical.cz_h1_posthoc_v22_final_conformance.registries import load_registry
    p=object.__new__(CanonicalPipeline);p.mode='full_registry_smoke';p.policy_id='synthetic';p.artifacts=[];p.fit_calls=0
    p.protocol={'geometry':{'inner':G}};_,p.C_specs=load_registry(MODULE/'C_MODEL_REGISTRY.json')
    p.checkpoints=Checkpoints(tmp_path/'run',dict(protocol='same'))
    p.learn=lambda stage,T:dict(representations=[{},{}]);p.emit=lambda state,T:np.zeros(len(T))
    def design(parent,fit,ev):
        x=((fit.origin.to_numpy()%2)*2-1).astype(float)[:,None];z=((ev.origin.to_numpy()%2)*2-1).astype(float)[:,None]
        return {f:(x,z) for f in ('ADDITIVE_COMPRESSED','ADDITIVE_JOINT_BASIS')},[],np.ones(len(fit),bool),np.ones(len(ev),bool)
    p._c_design=design;p.c=v2_c;p.cert=v211_selection.numerical_contract_passes
    T=frame(2000);T.y_true=0.;T.loc[:99,'y_true']=100*((T.origin.iloc[:100]%2)*2-1)
    result=mt.tune_c(p,T,dict(representations=[{},{}]),tmp_path/'evidence')
    assert result['selected_alpha']!=p.C_specs[0][1]
    audit=json.loads((tmp_path/'evidence/C_ALPHA_SELECTION_AUDIT.json').read_text())
    assert len(audit['candidate_status'])==len(p.C_specs) and all(r['status']=='AVAILABLE' for r in audit['candidate_status'])

def test_paired_outer_risk_binds_actual_ids_and_zero_model_is_not_bypass():
    T=frame(320);ids=ordered_ids(T)
    evidence=paired_route(T.y_true,np.zeros(320),np.zeros(320),score_ids=ids)
    assert evidence['parent_risk_evidence']['ordered_score_support_hash']==digest(ids)
    assert evidence['child_risk_evidence']['ordered_score_support_hash']==digest(ids)
    r=mse_evidence(np.zeros(320),np.zeros(320),ids)
    assert registered_best({'REGISTERED_LINEAR_MODEL_WITH_ZERO_COEFFICIENT':r})=='REGISTERED_LINEAR_MODEL_WITH_ZERO_COEFFICIENT'
    assert 'BYPASS' not in registered_best({'REGISTERED_LINEAR_MODEL_WITH_ZERO_COEFFICIENT':r})

def test_nonfinite_risk_bound_is_numerically_invalid():
    with np.errstate(over='ignore',invalid='ignore'):
        with pytest.raises(NumericallyInvalid):mse_evidence([1e308],[1e308],['one'])
