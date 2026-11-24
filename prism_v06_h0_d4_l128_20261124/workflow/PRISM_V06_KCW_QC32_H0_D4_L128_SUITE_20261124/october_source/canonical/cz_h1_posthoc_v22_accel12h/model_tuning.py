"""Strict MODEL recipes: native fit, exact common score, independently tagged failures."""
import numpy as np
import json
from .invariants import NumericallyInvalid,ProtocolError,digest
from .candidate_status import (SupportLimited,DomainNotApplicable,classify_exception,require_k_support,no_available_model)
from .partition import ordered_ids,support_hash
from .risk import RiskEvidence,mse_evidence,aggregate,registered_best,equivalence_bound
from .fold_geometry import folds
from .final_development_cf import replay
from .a_state import build_a_training_state_from_cf_residual,common_scoring_mask

ERRORS=(RuntimeError,ValueError,np.linalg.LinAlgError)

def risk(frame,y,p,mask):
    mask=np.asarray(mask,bool)
    if not mask.any():raise SupportLimited('No exact common MODEL scoring rows')
    return mse_evidence(np.asarray(y)[mask],np.asarray(p)[mask],ordered_ids(frame.loc[mask]))

def candidate(self,stage,partition,spec,compute):
    key=digest(dict(partition=partition,spec=spec));role=stage+'_CANDIDATE'
    saved=self.checkpoints.load(role,key)
    if saved is not None:return saved
    try:
        evidence=compute()
        body=dict(status='AVAILABLE',risk=dict(value=evidence.value,bound=evidence.bound,rows=evidence.rows,
            ordered_score_support_hash=evidence.ordered_score_support_hash))
    except ProtocolError:raise
    except ERRORS as exc:body=dict(status=classify_exception(exc),error=str(exc),error_type=type(exc).__name__)
    body.update(stage=stage,recipe=spec,family=spec[3] if stage=='K' else spec[0] if stage in {'C','W'} else 'MATURE_RESIDUAL_AR')
    body=json.loads(json.dumps(body))
    self.checkpoints.save(role,key,body)
    return body

def absent_stage(self,stage,T,parent,destination,status,reason):
    specs={'C':self.C_specs,'W':self.W_specs,'A':self.A_specs}[stage]
    partition=self._key(stage,T,'MODEL');ledger=[]
    def missing():
        if status in {'SUPPORT_LIMITED','POLICY_NOT_ESTIMABLE_IN_PARTITION'}:raise SupportLimited(reason)
        raise DomainNotApplicable(reason)
    for spec in specs:ledger.append(candidate(self,stage,partition,spec,missing))
    self._save(destination/'MODEL_COMMON_SUPPORT_AUDIT.json',dict(stage=stage,candidate_status=ledger,common_support=[],identity_reason=reason,identity_rows_in_recipe_risk=False))
    return self._unavailable_model(stage,parent,status,reason=reason)

def tune_k(self,T,destination):
    from .k_acceleration import tune_k_parallel
    return tune_k_parallel(self,T,destination)

def tune_c(self,T,parent,destination):
    if len(parent.get('representations') or [])<2:
        status='POLICY_NOT_ESTIMABLE_IN_PARTITION' if parent.get('routing_status')=='POLICY_NOT_ESTIMABLE_IN_PARTITION' else 'DOMAIN_NOT_APPLICABLE'
        return absent_stage(self,'C',T,parent,destination,status,'FEWER_THAN_TWO_LEGAL_K_REPRESENTATIONS')
    records=[]
    for j,(fi,ei) in enumerate(folds(T,self.protocol['geometry']['inner'])):
        fit=T.iloc[fi];ev=T.iloc[ei];ks=self.learn('K',fit)
        if len(ks.get('representations') or [])<2:
            records.append((fit,ev,None,None,None,np.zeros(len(fit),bool),np.zeros(len(ev),bool)))
            continue
        dd,_,ff,ee=self._c_design(ks,fit,ev)
        records.append((fit,ev,dd,self.emit(ks,fit),self.emit(ks,ev),ff,ee))
    ridge_paths={}
    risks={};ledger=[];partition=self._key('C',T,'MODEL')
    for spec in self.C_specs:
        family,alpha=spec
        def compute():
            evidence=[]
            for j,(fit,ev,dd,pf,pe,ff,ee) in enumerate(records):
                if not ee.any():continue
                if ff.sum()<2:raise SupportLimited('C nonempty common score fold lacks native fit support')
                design=dd[family][0][ff]
                if not design.shape[1] or not np.any(np.ptp(design,axis=0)>0):raise DomainNotApplicable('C recipe has no lawful fit input variation')
                self.fit_calls+=1
                from .ridge_path import RidgePath
                if (j,family) not in ridge_paths:ridge_paths[j,family]=RidgePath(design,fit.y_true.to_numpy(float)[ff]-pf[ff])
                correction,ct=ridge_paths[j,family].solve(dd[family][1][ee],alpha)
                if not self.cert(ct):raise NumericallyInvalid('C ridge certificate failed')
                evidence.append(mse_evidence(ev.y_true.to_numpy(float)[ee],pe[ee]+correction,ordered_ids(ev.loc[ee])))
            if not evidence:raise SupportLimited('C exact aggregate common scoring support is empty')
            return aggregate(evidence)
        result=candidate(self,'C',partition,spec,compute);ledger.append(result)
        if result['status']=='AVAILABLE':risks[spec]=RiskEvidence(**result['risk'])
    self._save(destination/'C_ALPHA_SELECTION_AUDIT.json',dict(candidate_status=ledger,risks={str(s):r.to_json() for s,r in risks.items()},
        common_support=[dict(fold=j,rows=int(r[-1].sum()),ordered_score_support_hash=support_hash(r[1].loc[r[-1]]),
            empty_common_support_excluded_for_all_recipes=not r[-1].any()) for j,r in enumerate(records)],all_registered_alphas_evaluated=True))
    if not risks:return self._unavailable_model('C',parent,no_available_model(r['status'] for r in ledger))
    family,alpha=registered_best(risks);dd,columns,mf,_=self._c_design(parent,T,T);kp=self.emit(parent,T)
    if mf.sum()<2:return self._unavailable_model('C',parent,'SUPPORT_LIMITED')
    self.fit_calls+=1;_,ct=self.c._ridge_fit(dd[family][0][mf],T.y_true.to_numpy(float)[mf]-kp[mf],dd[family][0][mf],alpha)
    if not self.cert(ct):raise NumericallyInvalid('Selected C final refit certificate failed')
    return dict(parent=parent,domain=True,recipe=family,selected_alpha=alpha,contract=ct,joint_columns=columns,fit_target='y_true-F_K')

def tune_residual(self,stage,T,parent,destination):
    upstream={'W':'C','A':'W'}[stage]
    if stage=='W':
        q,native,_=self.response_coordinate(parent,T)
        if not self._w_domain(q[native])['pass_']:return absent_stage(self,stage,T,parent,destination,'DOMAIN_NOT_APPLICABLE','W_FROZEN_RESPONSE_COORDINATE_DOMAIN_ABSENT')
    source,provenance,path=self.stream(upstream,T)
    specs=list(self.W_specs) if stage=='W' else list(self.A_specs)
    if self.mode=='solver_smoke':
        seen=set();debug=[]
        for spec in specs:
            group=(spec[0],spec[-1]) if stage=='W' else spec[0]
            if group not in seen:seen.add(group);debug.append(spec)
        specs=debug
    typed=build_a_training_state_from_cf_residual(source,policy_id=self.policy_id,h_steps=1) if stage=='A' else None
    records=[];supports=[]
    for j in range(4):
        ev=source.loc[source.CF_fold==j].copy();fit=source.loc[source._position<int(ev._position.min())].copy()
        for entity,g in ev.groupby('entity_id',sort=False):fit=fit.loc[(fit.entity_id!=entity)|(fit.target_stop_exclusive-1<=int(g.latest_available_target_index.min()))]
        common=common_scoring_mask(ev,typed,self.A_profiles,self.A_max) if stage=='A' else ev.coordinate_available.to_numpy(bool)
        records.append((fit,ev,common));supports.append(dict(fold=j,rows=int(common.sum()),ordered_score_ids=ordered_ids(ev.loc[common]),ordered_score_support_hash=support_hash(ev.loc[common])))
    risks={};ledger=[];partition=self._key(stage,T,'MODEL')
    for spec in specs:
        def compute():
            evidence=[]
            for fit,ev,common in records:
                if not common.any():continue
                ct,p,mask,audit=self.fit_residual_recipe(stage,spec,fit,ev,source,comparison=True)
                if ct is None:
                    cls=SupportLimited if audit.get('candidate_status')=='SUPPORT_LIMITED' else DomainNotApplicable
                    raise cls(stage+' recipe has no lawful native fit domain on nonempty common score fold')
                if not np.array_equal(mask,common):raise DomainNotApplicable(stage+' recipe differs from registered exact common score domain')
                evidence.append(risk(ev,ev.residual,p,common))
            if not evidence:raise SupportLimited(stage+' exact aggregate common scoring support is empty')
            return aggregate(evidence)
        result=candidate(self,stage,partition,spec,compute);ledger.append(result)
        if result['status']=='AVAILABLE':risks[spec]=RiskEvidence(**result['risk'])
    self._save(destination/'MODEL_COMMON_SUPPORT_AUDIT.json',dict(stage=stage,common_support=supports,identity_rows_in_recipe_risk=False,candidate_status=ledger))
    if not risks:return self._unavailable_model(stage,parent,no_available_model(r['status'] for r in ledger))
    spec=registered_best(risks)
    from .invariants import file_sha
    freeze=dict(role='FROZEN_RECIPE_BEFORE_FINAL_REPLAY',policy_id=self.policy_id,partition_id=self.partition_id(T),upstream_stage=upstream,selected_spec=spec,inner_source_sha256=file_sha(path))
    self._save(destination/'SELECTED_RECIPE_BEFORE_FINAL_REPLAY.json',freeze)
    final,proof,final_path=replay(self,upstream,T,freeze,frozen_parent=parent)
    ct,_,mask,audit=self.fit_residual_recipe(stage,spec,final,final,final)
    if ct is None:return self._unavailable_model(stage,parent,audit.get('candidate_status','DOMAIN_NOT_APPLICABLE'),reason='FINAL_SELECTED_NATIVE_RESIDUAL_DOMAIN_ABSENT')
    model=dict(parent=parent,domain=True,recipe=spec,contract=ct,source_path=final_path,source_sha256=file_sha(final_path),final_replay_coverage=proof['coverage'],
        inner_selection_source_path=path,selection_CF_provenance=provenance,final_fit_rows=int(mask.sum()),uses_in_sample_parent_residual=False,
        model_common_support=supports,model_risks={str(s):r.to_json() for s,r in risks.items()})
    if stage=='A':model.update(profile=list(spec[0]),residual_mean=audit['residual_mean'],maximum_lags=self.A_max)
    return model
