"""Versioned H1 adapter around unchanged authority numerical families.

All routing/tuning uses registered paired rows. Metadata absence completes the
candidate by its identity parent; numerical failure is never an identity rule.
"""
import copy
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from .invariants import ProtocolError,NumericallyInvalid,digest,dump,frame_hash,file_sha,prediction_hash
from .risk import RiskEvidence,mse_evidence,aggregate,registered_best,paired_route
from .checkpoint import Checkpoints
from .fold_geometry import folds,selection_blocks
from .a_state import (TRAINING_CF,INFERENCE_DEPLOYED,feature_rows,common_scoring_mask,
    build_a_training_state_from_cf_residual,build_a_inference_state_from_deployment_residual)
from .a_registry import load_a_registry
from .pointwise import paired_pointwise,export_pointwise,a_training_prediction_audit
from .final_development_cf import replay
from .deployment import physics_first_status,require_execution

CHANNELS=['main_heater_power','joint_lift','crystal_rotation','crucible_rotation']
UPSTREAM={'C':'K','W':'C','A':'W'}


class CanonicalPipeline:
    def __init__(self,context,root,*,mode):
        if mode not in {'metadata_only','solver_smoke','full_registry_smoke','formal_development','inference_only'}:
            raise ProtocolError('Unknown engine execution mode')
        self.mode=mode;self.root=Path(root);self.context=dict(context,mode=mode)
        self.registration=context['registration'];self.protocol=context['protocol']
        self.policy_id=context['policy_id'];self.shared=Path(context['shared'])
        if mode=='formal_development':require_execution(self.protocol,'formal_development')
        if mode in {'solver_smoke','full_registry_smoke'} and not self.protocol['execution_authorization']['development_only_solver_smoke']:
            raise ProtocolError('Development solver smoke is not authorized')
        sys.path.insert(0,str(Path(context['authority'])/'src'))
        from prism_benchmark import v211_k,v211_w,v2_k,v2_c,v21_a,v2_urysohn,v211_selection,v211_support
        from prism_benchmark.v211_config import load_v211_configs
        from prism_benchmark.cz_l256_nowcast import view
        from prism_benchmark.cpu_data import BaseAccessor
        self.k=v211_k;self.w=v211_w;self.v2_k=v2_k;self.c=v2_c;self.a=v21_a
        self.u=v2_urysohn;self.cert=v211_selection.numerical_contract_passes
        self.native=v211_support.native_support_mask;self.cap=v2_k._cap
        self._support_id=v211_support.support_id_hash
        self.view=view('dynamic',h_steps=1,task_id='CZ_DIAM_RAW2S_CURRENT_L256_H1')
        self.v211,self.v21,self.v2=load_v211_configs(Path(context['authority']),protocol='cz_h1_k_temporal_registry_v2')
        module=Path(__file__).parent
        registry,self.A_profiles=load_a_registry(module/'A_TEMPORAL_REGISTRY.json',
            self.registration['canonical_protocol_sha256']['A_TEMPORAL_REGISTRY.json'])
        self.A_max=registry['maximum_lags'];self.A_registry=registry
        self.BaseAccessor=BaseAccessor
        self.accessor=None if mode in {'metadata_only','inference_only'} else self._make_accessor(self.shared,'train')
        self.cache={};self.tune_cache={};self.stream_cache={};self.fit_calls=0
        self.deployed=None;self.artifacts=[]
        self.checkpoints=None
        if mode in {'solver_smoke','full_registry_smoke','formal_development'}:
            from .environment import fingerprint
            environment=fingerprint()
            binding=dict(release_sha=context.get('release_sha'),policy_id=self.policy_id,
                C1_manifest=context.get('C1_manifest'),environment=environment,mode=mode,
                case_id=context.get('case_id'),candidate_registry=self.registration['canonical_protocol_sha256'])
            self.checkpoints=Checkpoints(self.root,binding,resume=context.get('resume',False))

    def _make_accessor(self,shared,split):
        import pyarrow.parquet as pq
        partitions=['train'] if split=='train' else ['train','validation'] if split=='validation' else ['train','validation','test']
        available=set(CHANNELS)
        for part in partitions:
            path=Path(shared)/'base_data/cz_czochralski'/(part+'.parquet')
            if path.exists():available &= set(pq.ParquetFile(path).schema.names)
        columns=[ch for ch in CHANNELS if ch in available]
        accessor=self.BaseAccessor(Path(shared),'cz_czochralski',split,columns)
        accessor.warm_prefixes(columns)
        return accessor

    def partition_id(self,frame):return self._support_id(frame)

    def k_grid(self,channel):
        profiles=self.v2_k.channel_profiles(self.view,channel,self.v2)
        return profiles,list(self.v21['K_C']['m_tau']),max(h for _,h in profiles)

    def structural_grid_request(self,T,channel):
        profiles,mts,history=self.k_grid(channel)
        return dict(profiles=profiles,m_tau_values=mts,local_comparison_history=history,
            folds=folds(T,self.protocol['geometry']['inner']),channel=channel)

    def _fit_guard(self):
        if self.mode not in {'solver_smoke','full_registry_smoke','formal_development'}:
            raise ProtocolError('learn/refit is forbidden in this engine mode')

    def _key(self,stage,T,role):
        return digest(dict(policy=self.policy_id,mode=self.mode,stage=stage,role=role,frame=frame_hash(T)))

    def _save(self,path,value):
        dump(path,value);self.artifacts.append(str(Path(path).resolve()))

    def _references(self,value):
        paths=set()
        def visit(item):
            if isinstance(item,dict):
                for key,v in item.items():
                    if key in {'source_path','inner_selection_source_path','pointwise_path'} and isinstance(v,str):
                        p=Path(v);paths.add(str(p.resolve()))
                        side=p.with_suffix('.json') if key!='pointwise_path' else p.with_suffix(p.suffix+'.manifest.json')
                        if side.exists():paths.add(str(side.resolve()))
                    else:visit(v)
            elif isinstance(item,(list,tuple)):
                for v in item:visit(v)
        visit(value);return sorted(paths)

    def _na(self,stage,parent,reason):
        return dict(stage=stage,parent=parent,domain=False,domain_audit=dict(reason=reason),
            representations=[] if stage=='K' else None,routing_status='NOT_APPLICABLE',
            identity_is_model_candidate=False)

    def _values(self,T,r):
        return self.v2_k.profile_values(self.accessor,T,r['channel'],tuple(r['profile']),int(r['m_tau']))

    def _features(self,state,T):
        reps=state.get('representations') or []
        if not len(T):return np.empty((0,len(reps))),[],np.zeros(0,dtype=bool)
        mask=np.ones(len(T),dtype=bool)
        for r in reps:mask &= self.native(T,int(r['profile'][1]))
        if not reps:return np.empty((len(T),0)),[],np.zeros(len(T),dtype=bool)
        scalars=np.zeros((len(T),len(reps)));raw=[]
        for i,r in enumerate(reps):
            basis=self.u.basis_from_metadata(r['contract']['basis'])
            if mask.any():
                values,_=self._values(T.loc[mask],r)
                scalars[mask,i]=self.u.predict_contract(values,r['contract'])
                transformed=basis.transform(values).reshape(int(mask.sum()),-1)
                design=np.zeros((len(T),transformed.shape[1]));design[mask]=transformed
            else:
                # Do not ask the accessor for a nonexistent history. This
                # dimension is not used for fitting on an empty domain.
                dimension=basis.transform(np.zeros((1,int(r['m_tau'])))).reshape(1,-1).shape[1]
                design=np.zeros((len(T),dimension))
            raw.append(design)
        return scalars,raw,mask

    def _k_root(self,state):
        while state['stage']!='K':state=state['parent']
        return state

    def response_coordinate(self,parent,T):
        """Frozen MODEL latent q, distinct from the routed F_C baseline."""
        if parent['stage']!='C':raise ProtocolError('W coordinate requires a C parent')
        if parent['routing_status']=='ADMITTED' and parent.get('domain'):
            q,mask,_=self.predict(parent,T)
            return q,mask,dict(interface='PSI_C_FROZEN_MODEL_SCALAR')
        root=self._k_root(parent)
        q,mask,_=self.predict(root,T)
        return q,mask,dict(interface='PSI_K_RETAINED_REFERENCE_MODEL_SCALAR',
            reference_channel=root.get('reference_channel'),K_route=root['routing_status'])

    def overlap_components(self,parent,T):
        root=self._k_root(parent);reps=root.get('representations') or []
        output=pd.DataFrame(index=T.index);domains={}
        for channel in CHANNELS:
            r=next((r for r in reps if r['channel']==channel),None)
            values=np.zeros(len(T));mask=np.zeros(len(T),dtype=bool)
            if r is not None:
                mask=self.native(T,int(r['profile'][1]))
                if mask.any():
                    x,_=self._values(T.loc[mask],r);values[mask]=self.u.predict_contract(x,r['contract'])
            output['up_'+channel]=values
            domains[channel]=dict(available_rows=int(mask.sum()),identity_rows=int((~mask).sum()),
                role='RETAINED_REPRESENTATION_OR_EXPLICIT_NA_IDENTITY')
        return output,domains

    def predict(self,state,T,*,a_state=None):
        stage=state['stage'];parent=np.zeros(len(T)) if stage=='K' else self.emit(state['parent'],T)
        child=parent.copy();mask=np.zeros(len(T),dtype=bool);audit={}
        if not len(T):return child,mask,audit
        if (state.get('domain') and 'contract' in state) or (stage=='K' and state.get('domain')):
            if stage=='K':
                r=state['representations'][state['reference_index']]
                mask=self.native(T,int(r['profile'][1]))
                if mask.any():
                    x,_=self._values(T.loc[mask],r);child[mask]=self.u.predict_contract(x,r['contract'])
            elif stage=='C':
                p,raw,mask=self._features(state['parent'],T)
                if mask.any():
                    x=p if state['recipe']=='ADDITIVE_COMPRESSED' else np.concatenate([r[:,ix] for r,ix in zip(raw,state['joint_columns'])],axis=1)
                    child[mask]+=self.c._ridge_predict(x[mask],state['contract'])
            elif stage=='W':
                q,mask,audit=self.response_coordinate(state['parent'],T)
                if mask.any():child[mask]+=self.w.predict_w_correction(q[mask],state['contract'])
            elif stage=='A':
                if a_state is None:raise ProtocolError('A prediction requires explicit typed residual state')
                x,mask,audit=feature_rows(T,a_state,state['profile'],self.A_max,state['residual_mean'])
                if mask.any():child[mask]+=self.a.predict_mature_residual_ar(x,state['contract'])
        if not np.isfinite(child).all():raise NumericallyInvalid('Nonfinite MODEL prediction')
        return child,mask,audit

    def emit(self,state,T,source=None):
        if state['routing_status'] not in {'MODEL','ADMITTED'}:
            return np.zeros(len(T)) if state['stage']=='K' else self.emit(state['parent'],T)
        return self.predict(state,T,a_state=source)[0]

    def support_for(self,state,T,*,representations=False):
        # Every deployment row has a well-defined identity completion. Raw
        # basis availability is returned only by the explicit feature view.
        if representations:return T.loc[self._features(self._k_root(state),T)[2]].copy()
        return T.copy()

    def learn(self,stage,T,gid=None):
        self._fit_guard()
        if gid is not None:raise ProtocolError('Heater growth arms are not registered')
        T=T.reset_index(drop=True);key=self._key(stage,T,'ADMISSION')
        if key in self.cache:return self.cache[key]
        saved=self.checkpoints.load('ADMISSION_'+stage,key)
        if saved is not None:
            state=saved['state'];self.artifacts.extend(saved['artifacts']);self.cache[key]=state;return state
        parent=None if stage=='K' else self.learn(UPSTREAM[stage],T)
        geometry=folds(T,self.protocol['geometry']['outer'])
        destination=self.root/'models'/stage/key
        if not geometry:
            state=self._na(stage,parent,'UNAVAILABLE_COMPLETE_REGISTERED_GEOMETRY')
        else:
            targets=[];parents=[];children=[];evidence=[]
            for j,(fi,ei) in enumerate(geometry):
                fit=T.iloc[fi].copy();ev=T.iloc[ei].copy();model=self.tune(stage,fit)
                pp=np.zeros(len(ev)) if stage=='K' else self.emit(model['parent'],ev)
                if stage=='A':cp,mask,provenance=self.online_a(model,ev)
                else:cp,mask,_=self.predict(model,ev);provenance={}
                local_route='ADMITTED' if model['domain'] else 'NOT_APPLICABLE'
                record,proof=paired_pointwise(ev,stage=stage,kind='outer_oof',parent=pp,
                    model_child=cp,deployed=cp,row_domain=mask,routing_status=local_route,
                    policy_id=self.policy_id,partition_id=self.partition_id(fit),parameters=model,
                    state_provenance=provenance,fold=j)
                path=destination/f'admission_fold_{j}_private.parquet'
                export_pointwise(path,record,proof,resume=self.context.get('resume',False));self.artifacts.extend([str(path.resolve()),str(path.with_suffix('.parquet.manifest.json').resolve())])
                proof['pointwise_path']=str(path.resolve())
                evidence.append(proof);targets.extend(ev.y_true);parents.extend(pp);children.extend(cp)
            selection=paired_route(targets,parents,children)
            # Global paired admission is recorded before final-T MODEL refit.
            self._save(destination/'ROUTE_BEFORE_FINAL_REFIT.json',selection)
            state=copy.deepcopy(self.tune(stage,T))
            if not state['domain']:selection=dict(selection,routing_status='NOT_APPLICABLE')
            state.update(routing_status=selection['routing_status'],selection=selection,outer_realizations=evidence)
        state.update(stage=stage,admission_policy_id=self.policy_id,partition_id=self.partition_id(T),
            support_hash=self.partition_id(T),fit_rows=len(T),identity_is_model_candidate=False,
            execution_mode=self.mode,formal_eligible=self.mode=='formal_development')
        typed=None
        if stage=='A' and state.get('domain'):
            source=pd.read_parquet(state['source_path'])
            typed=build_a_training_state_from_cf_residual(source,policy_id=self.policy_id,h_steps=1)
            _,_,audit=feature_rows(T,typed,state['profile'],self.A_max,state['residual_mean'],expected_role=TRAINING_CF)
            state['A_training_prediction_audit']=a_training_prediction_audit(T,self.emit(state,T,typed),
                cf_state=typed,feature_audit=audit,policy_id=self.policy_id,
                partition_id=self.partition_id(T),parameters=state.get('recipe'))
        state['training_prediction_hash']=prediction_hash(self.emit(state,T,typed))
        state['prediction_support_hash']=self.partition_id(T)
        if stage=='A':state['PF_status']=physics_first_status(state)
        state['partition_model_id']=digest(state)
        self._save(destination/'state.json',state);self.cache[key]=state
        artifacts=self._references(state)+[str((destination/'state.json').resolve())]
        self.checkpoints.save('ADMISSION_'+stage,key,dict(state=state,artifacts=artifacts),artifacts)
        return state

    def tune(self,stage,T):
        self._fit_guard();key=self._key(stage,T,'MODEL');T=T.reset_index(drop=True)
        if key in self.tune_cache:return self.tune_cache[key]
        saved=self.checkpoints.load('MODEL_'+stage,key)
        if saved is not None:
            self.artifacts.extend(saved['artifacts']);self.tune_cache[key]=saved['model'];return saved['model']
        parent=None if stage=='K' else self.learn(UPSTREAM[stage],T)
        destination=self.root/'tuning'/stage/key
        if not folds(T,self.protocol['geometry']['inner']):model=self._na(stage,parent,'UNAVAILABLE_REGISTERED_MODEL_GEOMETRY')
        elif stage=='K':model=self.tune_k(T,destination)
        elif stage=='C':model=self.tune_c(T,parent,destination)
        else:model=self.tune_residual(stage,T,parent,destination)
        model.update(stage=stage,parent=parent if stage!='K' else None,
            routing_status='MODEL' if model['domain'] else 'NOT_APPLICABLE',
            policy_id=self.policy_id,tuning_partition_id=self.partition_id(T),identity_is_model_candidate=False)
        model['MODEL_hash']=digest(model);self._save(destination/'model.json',model)
        self.tune_cache[key]=model
        artifacts=self._references(model)+[str((destination/'model.json').resolve())]
        self.checkpoints.save('MODEL_'+stage,key,dict(model=model,artifacts=artifacts),artifacts)
        return model

    def tune_k(self,T,destination):
        self._fit_guard()
        reps=[];assembly=[];statuses={};geometry=folds(T,self.protocol['geometry']['inner'])
        for channel in CHANNELS:
            profiles,mts,common=self.k_grid(channel)
            if not self.accessor.has_columns([channel]):
                statuses[channel]='NOT_APPLICABLE';continue
            specs=self.k._structural_specs(profiles,mts,self.v2['K_module']['penalties']['lambda_tau'],
                self.v21['K_C']['m_x'],self.v2['K_module']['penalties']['lambda_x'])
            if self.mode=='solver_smoke':
                # Fixed debug coverage, never a formal policy/candidate change:
                # One fixed registered linear recipe per history. Debug
                # objects carry formal_eligible=False and cannot be deployed.
                specs=[s for s in specs if s[0][0]==2 and s[1]==4 and s[2]=='LINEAR_DISTRIBUTED_LAG'
                    and s[4]==float(self.v2['K_module']['penalties']['lambda_tau'][0])]
            prepared={};fold_data={};risks={};invalid=[];candidate_domains={}
            for spec in specs:
                candidate_key=digest(dict(partition=self._key('K',T,'MODEL'),channel=channel,spec=spec))
                cached=self.checkpoints.load('K_CANDIDATE',candidate_key)
                profile,mt,family,mx,lt,lx=spec;group=(profile,mt)
                if group not in fold_data:
                    records=[]
                    for fi,ei in geometry:
                        fit=self.cap(T.iloc[fi].loc[self.native(T.iloc[fi],profile[1])],int(self.v2['row_caps']['single_channel_k_fit']))
                        ev=T.iloc[ei].copy();mask=self.native(ev,common)
                        values=None
                        if len(fit)>=2:
                            values,_=self.v2_k.profile_values(self.accessor,fit,channel,profile,mt)
                            if not np.any(np.ptp(values,axis=0)>0):values=None
                        z=None
                        if mask.any():z,_=self.v2_k.profile_values(self.accessor,ev.loc[mask],channel,profile,mt)
                        records.append((fit,ev,mask,values,z))
                    fold_data[group]=records
                if cached is not None:
                    if cached['status']=='COMPLETED':risks[spec]=RiskEvidence(**cached['risk'])
                    elif cached['status']=='NUMERICALLY_INVALID':invalid.append(dict(spec=spec,error=cached['error']))
                    else:candidate_domains[str(spec)]='NOT_APPLICABLE'
                    continue
                native=T.loc[self.native(T,profile[1])]
                if len(native)<2:
                    candidate_domains[str(spec)]='NOT_APPLICABLE'
                    self.checkpoints.save('K_CANDIDATE',candidate_key,dict(status='NOT_APPLICABLE'));continue
                try:
                    evidence=[]
                    for j,(fit,ev,mask,x,z) in enumerate(fold_data[group]):
                        p=np.zeros(len(ev))
                        if x is not None and mask.any():
                            width=1 if family=='LINEAR_DISTRIBUTED_LAG' else mx
                            prep_key=(group,j,width)
                            if prep_key not in prepared:prepared[prep_key]=self.u.prepare_contract_fit(x,fit.y_true.to_numpy(float),width)
                            ct=self.u.fit_prepared_contract(prepared[prep_key],family,mx,
                                (float(self.v2['K_module']['penalties']['pilot']['lambda_0']),lt,lx),**self.v2_k._als_kwargs(self.v2))
                            self.fit_calls+=1
                            if not self.k._candidate_valid(ct,len(fit),self.v2):raise NumericallyInvalid('K structural certificate failed')
                            p[mask]=self.u.predict_contract(z,ct)
                        evidence.append(mse_evidence(ev.y_true,p))
                    risks[spec]=aggregate(evidence)
                    r=risks[spec]
                    self.checkpoints.save('K_CANDIDATE',candidate_key,dict(status='COMPLETED',risk=dict(value=r.value,bound=r.bound,rows=r.rows)))
                except (ValueError,np.linalg.LinAlgError,NumericallyInvalid) as exc:
                    invalid.append(dict(spec=spec,error=str(exc)));candidate_domains[str(spec)]='NUMERICALLY_INVALID'
                    self.checkpoints.save('K_CANDIDATE',candidate_key,dict(status='NUMERICALLY_INVALID',error=str(exc),error_type=type(exc).__name__))
            if not risks:
                statuses[channel]='NUMERICALLY_INVALID' if invalid else 'NOT_APPLICABLE';continue
            spec=registered_best(risks);profile,mt,family,mx,lt,lx=spec
            native=T.loc[self.native(T,profile[1])];x,intervals=self.v2_k.profile_values(self.accessor,native,channel,profile,mt)
            if not np.any(np.ptp(x,axis=0)>0):statuses[channel]='NOT_APPLICABLE';continue
            def full(l0):
                self.fit_calls+=1
                return self.u.fit_contract(x,native.y_true.to_numpy(float),family,mx,(l0,lt,lx),**self.v2_k._als_kwargs(self.v2))
            def folded(l0):
                result=[]
                for j,(fit,ev,mask,xx,z) in enumerate(fold_data[(profile,mt)]):
                    p=np.zeros(len(ev));ct={'numerical_certificate':{'status':'NOT_APPLICABLE_IDENTITY'}}
                    domain=xx is not None and mask.any()
                    if xx is not None and mask.any():
                        self.fit_calls+=1;ct=self.u.fit_contract(xx,fit.y_true.to_numpy(float),family,mx,(l0,lt,lx),**self.v2_k._als_kwargs(self.v2))
                        p[mask]=self.u.predict_contract(z,ct)
                    result.append(dict(fold=j,contract=ct,fit_rows=len(fit),prediction=p,
                        domain=bool(domain),risk=mse_evidence(ev.y_true,p)))
                return result
            try:
                l0,ct,replayed,stability=self.k.select_smallest_stable_full_and_folds(
                    self.v2['K_module']['penalties']['lambda_0'],full,folded,
                    valid_full=lambda c:self.k._candidate_valid(c,len(native),self.v2),
                    valid_fold=lambda r:not r['domain'] or self.k._candidate_valid(r['contract'],r['fit_rows'],self.v2),parallel_workers=1)
            except (ValueError,np.linalg.LinAlgError,RuntimeError) as exc:
                statuses[channel]='NUMERICALLY_INVALID';self._save(destination/(channel+'_failure.json'),dict(error=str(exc)));continue
            reps.append(dict(channel=channel,profile=list(profile),m_tau=mt,family=family,m_x=mx,
                lambdas=[l0,lt,lx],contract=ct,intervals=intervals,
                native_fit_support_hash=self.partition_id(native),numerical_ridge_audit=stability,
                representation_preserved=True))
            assembly.append(aggregate(r['risk'] for r in replayed));statuses[channel]='AVAILABLE'
            self._save(destination/(channel+'_selection.json'),dict(selected_spec=spec,
                selected_risk=risks[spec].to_json(),candidate_count=len(specs),
                registered_profiles=len(profiles),maximum_history_steps=common,
                numerical_invalid_candidates=invalid,candidate_domains=candidate_domains,
                risks={str(s):r.to_json() for s,r in risks.items()},
                smoke_restricted_candidates=self.mode=='solver_smoke'))
        if not reps:
            if 'NUMERICALLY_INVALID' in statuses.values():raise NumericallyInvalid('All K representations unavailable with numerical failures')
            return dict(domain=False,representations=[],channel_status=statuses)
        reference=registered_best({i:r for i,r in enumerate(assembly)})
        return dict(domain=True,representations=reps,reference_index=reference,
            reference_channel=reps[reference]['channel'],channel_status=statuses,
            assembly_risks=[r.to_json() for r in assembly],recipe='REGISTERED_K_REPRESENTATIONS')

    def _c_design(self,parent,fit,ev):
        x,raw,mf=self._features(parent,fit);z,re,me=self._features(parent,ev)
        columns=[];xf=[];xe=[];cfg=self.v2['C_module']['joint_basis']
        for f,e in zip(raw,re):
            ix=self.c._qr_columns(f[mf],int(cfg['per_channel_max_columns']),float(cfg['per_channel_qr_relative_diagonal_floor']))
            columns.append(ix.tolist());xf.append(f[:,ix]);xe.append(e[:,ix])
        return {'ADDITIVE_COMPRESSED':(x,z),'ADDITIVE_JOINT_BASIS':(np.concatenate(xf,axis=1),np.concatenate(xe,axis=1))},columns,mf,me

    def tune_c(self,T,parent,destination):
        self._fit_guard()
        if len(parent.get('representations') or [])<2:return self._na('C',parent,'FEWER_THAN_TWO_LEGAL_REPRESENTATIONS')
        _,_,full_mask=self._features(parent,T)
        if full_mask.sum()<2:return self._na('C',parent,'NO_NATIVE_C_ASSEMBLY')
        designs,columns,mf,_=self._c_design(parent,T,T);kp=self.emit(parent,T)
        records=[]
        for fi,ei in folds(T,self.protocol['geometry']['inner']):
            fit=T.iloc[fi];ev=T.iloc[ei];ks=self.learn('K',fit)
            pf=self.emit(ks,fit);pe=self.emit(ks,ev)
            if len(ks.get('representations') or [])<2:records.append((fit,ev,None,pf,pe,None,None));continue
            _,_,fit_mask=self._features(ks,fit)
            if fit_mask.sum()<2:records.append((fit,ev,None,pf,pe,None,None));continue
            dd,_,ff,ee=self._c_design(ks,fit,ev);records.append((fit,ev,dd,pf,pe,ff,ee))
        risks={};contracts={};certificates={}
        for family in ['ADDITIVE_COMPRESSED','ADDITIVE_JOINT_BASIS']:
            trials=[]
            for alpha in sorted(set([0.]+list(map(float,self.v2['C_module']['joint_basis']['ridge_alpha_grid'])))):
                try:
                    self.fit_calls+=1
                    _,ct=self.c._ridge_fit(designs[family][0][mf],T.y_true.to_numpy(float)[mf]-kp[mf],designs[family][0][mf],alpha)
                    valid=self.cert(ct);observations=[]
                    for fit,ev,dd,pf,pe,ff,ee in records:
                        p=pe.copy()
                        if dd is not None and ee.any():
                            self.fit_calls+=1
                            correction,fc=self.c._ridge_fit(dd[family][0][ff],fit.y_true.to_numpy(float)[ff]-pf[ff],dd[family][1][ee],alpha)
                            valid &= self.cert(fc);p[ee]+=correction
                        observations.append(mse_evidence(ev.y_true,p))
                    trials.append(dict(alpha=alpha,certified=bool(valid)))
                    if valid:risks[family]=aggregate(observations);contracts[family]=(alpha,ct);break
                except (ValueError,np.linalg.LinAlgError) as exc:trials.append(dict(alpha=alpha,error=str(exc)))
            certificates[family]=trials
        family=registered_best(risks);alpha,ct=contracts[family]
        self._save(destination/'C_INNER_EVIDENCE.json',dict(risks={f:r.to_json() for f,r in risks.items()},certificates=certificates))
        return dict(parent=parent,domain=True,recipe=family,selected_alpha=alpha,contract=ct,
            joint_columns=columns,fit_target='y_true-F_K')

    def stream(self,upstream,T):
        self._fit_guard()
        key=self._key(upstream,T,'SELECTION_CF')
        if key in self.stream_cache:return self.stream_cache[key]
        records=[];proof=[]
        for fi,ei,context,score,j in selection_blocks(T,self.protocol['geometry']['inner'],
                self.protocol['geometry']['selection_cf_context_rows_per_block']):
            parent=self.learn(upstream,T.iloc[fi]);ev=T.iloc[ei].copy();p=self.emit(parent,ev)
            ev['parent_prediction']=p;ev['residual']=ev.y_true.to_numpy(float)-p
            if upstream=='C':
                q,native,coordinate=self.response_coordinate(parent,ev)
                ev['response_coordinate']=q;ev['coordinate_available']=native
            ev['residual_source_role']='CROSS_FITTED_PREQUENTIAL';ev['_position']=ei
            ev['CF_fold']=-1;ev.loc[ev.index.isin(T.iloc[score].index),'CF_fold']=j
            overlap,domains=self.overlap_components(parent,ev)
            for col in overlap:ev[col]=overlap[col].to_numpy()
            if upstream=='W':ev['up_w']=p-self.emit(parent['parent'],ev)
            records.append(ev)
            proof.append(dict(fold=j,fit_rows=len(fi),fit_support_hash=self.partition_id(T.iloc[fi]),
                context_n=len(context),context_support_hash=self.partition_id(T.iloc[context]),
                score_n=len(score),score_support_hash=self.partition_id(T.iloc[score]),
                upstream_partition_model_id=parent['partition_model_id'],component_domains=domains,
                target_used_to_fit_own_predictor=False))
        source=pd.concat(records,ignore_index=True)
        # Context windows may overlap prior score windows. Keep a single lawful
        # CF prediction, but preserve the explicitly registered score role.
        roles={int(position):j for j,(_,ei) in enumerate(folds(T,self.protocol['geometry']['inner'])) for position in ei}
        source=source.drop_duplicates(['entity_id','origin'],keep='first').sort_values('_position').reset_index(drop=True)
        source['CF_fold']=[roles.get(int(p),-1) for p in source._position]
        path=self.root/'selection_streams'/upstream/key/'prequential.parquet';path.parent.mkdir(parents=True,exist_ok=True)
        source.to_parquet(path,index=False);self.artifacts.append(str(path.resolve()))
        self._save(path.with_suffix('.json'),dict(role='INNER_SELECTION_CF',provenance=proof,
            source_hash=frame_hash(source),file_sha256=file_sha(path)))
        self.stream_cache[key]=(source,proof,str(path));return self.stream_cache[key]

    def _w_domain(self,q):
        if len(q)<2:return dict(pass_=False,reason='INSUFFICIENT_CF_COORDINATE_ROWS')
        cfg=self.v211['W'];audit=self.w.latent_fold_usable(np.asarray(q,float),
            minimum_distinct_values=int(cfg['minimum_distinct_latent_values_per_fold']),
            minimum_rank=int(cfg['minimum_rank_of_intercept_and_latent']),
            scale_floor_multiplier=float(cfg['relative_scale_floor_multiplier_float64_eps']))
        return dict(audit,pass_=bool(audit['pass']))

    def fit_residual_recipe(self,stage,spec,fit,ev,source,*,comparison=False):
        self._fit_guard()
        mask=np.ones(len(ev),bool);mean=0.;audit={}
        if stage=='A':
            state=build_a_training_state_from_cf_residual(source,policy_id=self.policy_id,h_steps=1)
            _,native,_=feature_rows(fit,state,spec[0],self.A_max,0.,expected_role=TRAINING_CF)
            if native.sum()<2:return None,np.zeros(len(ev)),np.zeros(len(ev),bool),{}
            fit=fit.loc[native].copy();mean=float(fit.residual.mean())
            x,_,fa=feature_rows(fit,state,spec[0],self.A_max,mean,expected_role=TRAINING_CF)
            z,mask,audit=feature_rows(ev,state,spec[0],self.A_max,mean,expected_role=TRAINING_CF)
            if comparison:mask &= common_scoring_mask(ev,state,self.A_profiles,self.A_max)
            # z is in the native order; reconstruct only the common subset.
            if comparison:z=feature_rows(ev.loc[mask],state,spec[0],self.A_max,mean,expected_role=TRAINING_CF)[0]
            if not mask.any():return None,np.zeros(len(ev)),mask,dict(residual_mean=mean)
        else:
            fit=fit.loc[fit.coordinate_available].copy()
            mask=ev.coordinate_available.to_numpy(bool)
            if not self._w_domain(fit.response_coordinate)['pass_'] or not mask.any():
                return None,np.zeros(len(ev)),np.zeros(len(ev),bool),{}
        up=fit[[c for c in fit if c.startswith('up_')]].to_numpy(float)
        # Explicit N/A representation coordinates are identity increments;
        # constant overlap columns convey no identifiable overlap direction.
        if up.shape[1]:up=up[:,np.ptp(up,axis=0)>0]
        self.fit_calls+=1
        if stage=='W':
            family,knots,smoothness,mu,direction=spec
            p,ct=self.w.fit_w_correction(fit.response_coordinate.to_numpy(float),fit.residual.to_numpy(float),
                ev.loc[mask,'response_coordinate'].to_numpy(float),family=family,knot_count=int(knots),
                smoothness=float(smoothness),mu=float(mu),upstream_predictions=up,direction=int(direction))
            output=np.zeros(len(ev));output[mask]=p
        else:
            p,ct=self.a.fit_mature_residual_ar(x,fit.residual.to_numpy(float),z,
                alpha=float(spec[1]),mu=float(spec[2]),upstream_predictions=up)
            output=np.zeros(len(ev));output[mask]=p
        if not self.cert(ct):raise NumericallyInvalid(stage+' residual numerical certificate failed')
        return ct,output,mask,dict(residual_mean=mean,feature_audit=audit)

    def tune_residual(self,stage,T,parent,destination):
        self._fit_guard()
        upstream=UPSTREAM[stage]
        if stage=='W':
            q,native,_=self.response_coordinate(parent,T)
            if not self._w_domain(q[native])['pass_']:
                return self._na(stage,parent,'W_FROZEN_RESPONSE_COORDINATE_UNAVAILABLE')
        source,provenance,path=self.stream(upstream,T)
        specs=self.w._w_candidates(self.v211,self.v21,self.v2,1,False) if stage=='W' else [
            (pr,float(alpha),float(mu)) for pr in self.A_profiles
            for alpha in self.v2['A_module']['ridge_alpha_grid'] for mu in self.v21['A']['soft_overlap_mu']]
        if self.mode=='solver_smoke':
            seen=set();debug=[]
            for spec in specs:
                group=(spec[0],spec[-1]) if stage=='W' else spec[0]
                if group not in seen:seen.add(group);debug.append(spec)
            specs=debug
        losses={};invalid=[];domains={};fold_support=[]
        for j in range(self.protocol['geometry']['inner']['count']):
            ev=source.loc[source.CF_fold==j].copy()
            # Every score block retains all 320 rows. Fit labels may use only
            # earlier source positions and mature target ends in this entity.
            fit=source.loc[source._position<int(ev._position.min())].copy()
            for entity,g in ev.groupby('entity_id',sort=False):
                fit=fit.loc[(fit.entity_id!=entity)|(fit.target_stop_exclusive-1<=int(g.latest_available_target_index.min()))]
            fold_support.append(dict(fold=j,fit_n=len(fit),score_n=len(ev),
                fit_support_hash=self.partition_id(fit),score_support_hash=self.partition_id(ev)))
            for spec in specs:
                if spec in invalid:continue
                try:
                    ct,p,mask,_=self.fit_residual_recipe(stage,spec,fit,ev,source,comparison=True)
                    losses.setdefault(spec,[]).append(mse_evidence(ev.residual,p))
                    domains.setdefault(spec,[]).append(int(mask.sum()))
                except (ValueError,np.linalg.LinAlgError,NumericallyInvalid):invalid.append(spec);losses.pop(spec,None)
        candidates={s:aggregate(r) for s,r in losses.items() if len(r)==4 and any(domains[s])}
        if not candidates:
            if invalid:raise NumericallyInvalid('All residual MODEL candidates numerically invalid or domain absent')
            return self._na(stage,parent,'NO_COMPLETE_MATURE_CF_MODEL_DOMAIN')
        spec=registered_best(candidates)
        freeze=dict(role='FROZEN_RECIPE_BEFORE_FINAL_REPLAY',policy_id=self.policy_id,
            partition_id=self.partition_id(T),upstream_stage=upstream,selected_spec=spec,
            inner_source_sha256=file_sha(path))
        self._save(destination/'SELECTED_RECIPE_BEFORE_FINAL_REPLAY.json',freeze)
        final,proof,final_path=replay(self,upstream,T,freeze)
        self.artifacts.extend([final_path,str(Path(final_path).with_suffix('.json'))])
        ct,_,mask,audit=self.fit_residual_recipe(stage,spec,final,final,final)
        if ct is None:return self._na(stage,parent,'FINAL_SELECTED_NATIVE_RESIDUAL_DOMAIN_ABSENT')
        self._save(destination/'MODEL_INNER_RISKS.json',dict(selected_spec=spec,
            risks={str(s):r.to_json() for s,r in candidates.items()},
            domain_rows={str(s):r for s,r in domains.items()},fold_support=fold_support,
            invalid_candidates=invalid))
        model=dict(parent=parent,domain=True,recipe=spec,contract=ct,
            source_path=final_path,source_sha256=file_sha(final_path),final_replay_coverage=proof['coverage'],
            inner_selection_source_path=path,selection_CF_provenance=provenance,
            final_fit_rows=int(mask.sum()),uses_in_sample_parent_residual=False)
        if stage=='A':model.update(profile=list(spec[0]),residual_mean=audit['residual_mean'],maximum_lags=self.A_max)
        return model

    def scoring_a(self,model,ev,universe):
        # Compatibility state view contains this scoring interval only.
        # The execution path below actually advances one issue/maturity at a time.
        history=ev.copy();upstream=model['parent']
        model_id=upstream.get('deployment_model_id',upstream.get('partition_model_id'))
        typed=build_a_inference_state_from_deployment_residual(history,self.emit(upstream,history),
            deployment_model_id=model_id,h_steps=1)
        return typed,dict(state_role=INFERENCE_DEPLOYED,
            upstream_deployment_model_id=model_id,
            mature_state_hash=typed.content_hash,initialization='EMPTY_PER_SCORING_BLOCK_OR_ENTITY',
            training_backcast_used=False,pre_score_warmup_used=False)

    def online_a(self,model,ev):
        """Issue predictions first, then append only targets matured subsequently."""
        upstream=model['parent'];model_id=upstream.get('deployment_model_id',upstream.get('partition_model_id'))
        baseline=self.emit(upstream,ev.drop(columns=['y_true']))
        output=baseline.copy();mask=np.zeros(len(ev),bool);ledger=[];feature_hashes=[]
        for entity,g in ev.groupby('entity_id',sort=False):
            mature=[];pending=[];last=None
            for index,row in g.iterrows():
                position=ev.index.get_loc(index);latest=int(row.latest_available_target_index)
                if last is not None and int(row.origin)<=last:raise ProtocolError('Online scoring origins must increase')
                last=int(row.origin)
                ready=[item for item in pending if item['end']<=latest]
                pending=[item for item in pending if item['end']>latest]
                for item in ready:
                    observed=ev.loc[[item['index']]].copy()
                    observed['residual']=observed.y_true.to_numpy(float)-item['W_prediction']
                    observed['residual_source_role']='DEPLOYED_W_ONLINE'
                    observed['prediction_issue_latest_index']=item['issue_latest']
                    mature.append(observed)
                history=pd.concat(mature,ignore_index=True) if mature else ev.iloc[:0].assign(residual=np.empty(0),residual_source_role='DEPLOYED_W_ONLINE')
                from .a_state import _state
                typed=_state(history,INFERENCE_DEPLOYED,model_id,1,1)
                prediction,native,audit=self.predict(model,ev.loc[[index]].drop(columns=['y_true']),a_state=typed)
                output[position]=prediction[0];mask[position]=native[0]
                feature_hashes.append(audit.get('mature_feature_hash',digest('NA_IDENTITY')))
                ledger.append(dict(entity=str(entity),origin=last,issue_latest=latest,
                    mature_rows=len(history),state_hash=typed.content_hash,child_hash=prediction_hash(prediction)))
                end=int(row.target_stop_exclusive)-1
                if latest>=end:raise ProtocolError('Prediction must precede its own target maturity')
                pending.append(dict(index=index,end=end,W_prediction=baseline[position],issue_latest=latest))
        provenance=dict(state_role=INFERENCE_DEPLOYED,upstream_deployment_model_id=model_id,
            upstream_model_scope='DEPLOYMENT' if 'deployment_model_id' in upstream else 'PARTITION',
            mature_state_hash=digest(ledger),mature_feature_hash=digest(feature_hashes),
            initialization='EMPTY_PER_SCORING_BLOCK_OR_ENTITY',pre_score_warmup_used=False,
            training_backcast_used=False,prediction_issue_before_target_maturity=True,
            state_crosses_entity=False,issue_ledger_hash=digest(ledger))
        return output,mask,provenance

    def load_development(self):
        self._fit_guard()
        from .data import load_development
        samples=load_development(self.context)
        if self.mode in {'solver_smoke','full_registry_smoke'}:samples=samples.iloc[:int(self.protocol['solver_smoke']['prefix_rows'])].copy()
        return samples

    def complete_development_artifacts(self,state):
        self._fit_guard()
        from .evidence import report_evidence
        node=state
        while node is not None:
            records=[]
            for proof in node.get('outer_realizations',[]):
                candidate=pd.read_parquet(proof['pointwise_path'])
                parent=candidate.parent_prediction.to_numpy(float)
                child=candidate.MODEL_child_prediction.to_numpy(float)
                mask=(candidate.row_domain=='AVAILABLE').to_numpy()
                if node['routing_status']=='NOT_APPLICABLE':child=parent.copy();mask[:]=False
                deployed=child if node['routing_status']=='ADMITTED' else parent
                record=candidate.copy()
                # Exact fold MODEL identity is immutable after global admission.
                record['global_routing_status']=node['routing_status']
                record['global_deployed_prediction']=deployed
                record['global_residual']=record.y_true.to_numpy(float)-deployed
                audit=dict(proof,pointwise_content_hash=frame_hash(record),
                    global_routing_status=node['routing_status'],
                    global_deployed_prediction_hash=prediction_hash(deployed),
                    original_outer_realization_id_preserved=True)
                path=Path(proof['pointwise_path']).with_name('FROZEN_OOF_'+str(proof['fold'])+'.parquet')
                export_pointwise(path,record,audit,resume=self.context.get('resume',False))
                self.artifacts.extend([str(path),str(path.with_suffix('.parquet.manifest.json'))])
                records.append(record)
            if records:
                pointwise=pd.concat(records,ignore_index=True).rename(columns={
                    'parent_prediction':'parent','MODEL_child_prediction':'MODEL_child'})
                report=report_evidence(pointwise,self.protocol['evidence'])
                self._save(self.root/'reports'/(node['stage']+'_SCIENTIFIC_EVIDENCE.json'),report)
            node=node.get('parent')
        return list(dict.fromkeys(self.artifacts))

    def development_complete(self,state):
        return state['stage']=='A' and state.get('partition_model_id') is not None

    def assert_inference_only(self):
        if self.mode!='inference_only' or self.fit_calls!=0:raise ProtocolError('Inference engine cannot fit/refit')

    def bind_deployment(self,state):
        self.assert_inference_only();self.deployed=state
        self._verify_chain(state)

    def _verify_chain(self,state):
        if state['admission_policy_id']!=self.policy_id:raise ProtocolError('Deployment policy mismatch')
        id_key='deployment_model_id' if 'deployment_model_id' in state else 'partition_model_id'
        if digest({k:v for k,v in state.items() if k!=id_key})!=state[id_key]:
            raise ProtocolError('Deployment model hash mismatch')
        if state.get('parent'):self._verify_chain(state['parent'])

    def infer_and_report(self,operation):
        self.assert_inference_only()
        from .data import load_inference
        ev,observations,accessor_shared,accessor_split=load_inference(self.context,operation)
        self.accessor=self._make_accessor(accessor_shared,accessor_split)
        states=[];node=self.deployed
        while node is not None:states.append(node);node=node.get('parent')
        artifacts=[]
        for state in reversed(states):
            stage=state['stage'];parent=np.zeros(len(ev)) if stage=='K' else self.emit(state['parent'],ev)
            if stage=='A':child,mask,provenance=self.online_a(state,ev)
            else:child,mask,_=self.predict(state,ev);provenance={}
            deployed=child if state['routing_status']=='ADMITTED' else parent
            record,proof=paired_pointwise(ev,stage=stage,kind=operation,parent=parent,
                model_child=child,deployed=deployed,row_domain=mask,routing_status=state['routing_status'],
                policy_id=self.policy_id,partition_id=self.partition_id(ev),parameters=state,
                state_provenance=provenance)
            path=self.root/(stage+'_pointwise.parquet');export_pointwise(path,record,proof)
            artifacts.extend([path,path.with_suffix('.parquet.manifest.json')])
        return artifacts
