"""Versioned integration of the existing canonical engine with frozen H1 K registry.

K retains all certified MODEL representations irrespective of global routing.
Every admission fold calls MODEL tuning on that fold's training partition.
W/A fit prequential parent residuals with full recursive upstream reconstruction.
"""
from __future__ import annotations
import os,sys,json,copy,ast
from pathlib import Path
import numpy as np
import pandas as pd
from canonical_core import *
AUTH=Path(os.environ['CZ_AUTHORITY_ROOT'])
sys.path.insert(0,str(AUTH/'src'))
from prism_benchmark import v211_k as k,v211_w as w,v2_k
from prism_benchmark.cpu_data import BaseAccessor
from prism_benchmark.v211_config import load_v211_configs,REPRESENTATIVE_STAGE1_PROTOCOL
from prism_benchmark.cz_h1_temporal_registry import PROTOCOL, file_sha as registry_file_sha, CONFIG_ROOT, REGISTRY_NAME
from prism_benchmark.v211_support import support_id_hash,native_support_mask,apply_native_support,registered_fold_native_masks
from prism_benchmark.v2_urysohn import fit_contract,predict_contract,basis_from_metadata
from prism_benchmark.v2_c import _ridge_fit,_ridge_predict,_qr_columns
from prism_benchmark.v211_selection import numerical_contract_passes
from prism_benchmark.v21_a import mature_residual_features,fit_mature_residual_ar,predict_mature_residual_ar
from prism_benchmark.v2_basis import causal_geometric_intervals
import math
CHANNELS=['main_heater_power','joint_lift','crystal_rotation','crucible_rotation']
BASE_VALUES=v2_k.profile_values

def profile_values(accessor,frame,ch,profile,m_tau):
    return BASE_VALUES(accessor,frame,ch,profile,m_tau)

class CanonicalPipeline:
    def __init__(self,root,shared,view,H,registration,*,metadata_only=False):
        if H!='H1' or view.head.h_steps!=1:
            raise ValueError('This integration is explicit H1 only; retain the historical H4 engine')
        self.root=Path(root);self.shared=Path(shared);self.view=view;self.H=H;self.registration=registration
        self.metadata_only=metadata_only
        self.policy_id=digest(registration);self.cache={};self.tune_cache={};self.stream_cache={}
        self.v211,self.v21,self.v2=load_v211_configs(AUTH,protocol=PROTOCOL)
        if registration['k_registry_sha256']!=registry_file_sha(AUTH/CONFIG_ROOT/REGISTRY_NAME):
            raise ProtocolSupportError('Canonical policy bound to another K registry')
        if registration['anchor_contract']!='CZ_H1_MAXIMAL_AUTHORITY_PARTITION_ANCHORS_V1':
            raise ProtocolSupportError('Canonical policy needs maximal source-train anchors')
        self.block=int(registration['inner_block_rows']);self.warmup=int(registration['cf_warmup_rows'])
        if not metadata_only:
            from final_development_cf import require_contract
            require_contract(self)
        self.accessor=None if metadata_only else BaseAccessor(shared,view.head.dataset,'train',CHANNELS)
        if self.accessor is not None:self.accessor.warm_prefixes(CHANNELS)
        self.A_profiles=[(1,h) for h in [8,16,32,64,96,128]] if H=='H1' else [(1,8),(1,16),(1,32),(2,8),(2,16),(2,32)]
        self.A_max=128 if H=='H1' else 64;self.A_common=128 if H=='H1' else 32
        self.full_train_hash=registration['training_support_hashes'][H][Path(shared).name]

    def k_grid(self,ch,gid=None):
        if gid is not None:raise ProtocolSupportError('Growth arms are not in the frozen formal H1 registry')
        profiles=v2_k.channel_profiles(self.view,ch,self.v2)
        return profiles,list(self.v21['K_C']['m_tau']),max(p[1] for p in profiles)

    def structural_grid_request(self,T,ch,gid,destination):
        profiles,mts,common=self.k_grid(ch,gid)
        return dict(accessor=self.accessor,train=T,folds=folds(T,self.block),channel=ch,
            profiles=profiles,local_comparison_history=common,m_tau_values=mts,
            m_x_values=self.v21['K_C']['m_x'],lambda_tau_values=self.v2['K_module']['penalties']['lambda_tau'],
            lambda_x_values=self.v2['K_module']['penalties']['lambda_x'],
            pilot_lambda_0=float(self.v2['K_module']['penalties']['pilot']['lambda_0']),v2=self.v2,
            inner_workers=int(self.registration['K_workers']),destination=destination,
            checkpoint_context=dict(policy_id=self.policy_id,partition_id=support_id_hash(T),channel=ch,profiles=profiles))

    def support_for(self,state,T,*,representations=False):
        """Deployment support derives from realized dependencies, never registry Lmax.

        Representation support is a separate C-domain view. It cannot replace
        the parent frame merely because a K route is BYPASSED.
        """
        if state['stage']=='K':
            if representations:
                history=max(r['profile'][1] for r in state['representations'])
            elif state['routing_status'] not in ['MODEL','ADMITTED']:
                return T.copy()
            else:history=state['representations'][state['reference_index']]['profile'][1]
            return apply_native_support(T,history)
        parent=self.support_for(state['parent'],T)
        if state['routing_status'] not in ['MODEL','ADMITTED']:return parent
        if state['stage']=='C':return self.support_for(state['parent'],parent,representations=True)
        return parent

    def key(self,stage,T,gid,kind='POLICY'):
        return digest(dict(policy=self.policy_id,stage=stage,kind=kind,support=support_id_hash(T),growth=gid))

    def emit(self,state,T,source=None):
        """Predict frozen routed stage; this function never fits/selects."""
        stage=state['stage'];parent=np.zeros(len(T),float)
        if stage=='K':
            # BYPASS has no K dependencies; do not even read retained bases.
            if state['routing_status'] not in ['ADMITTED','MODEL']:return parent
            if len(self.support_for(state,T))!=len(T):
                raise ProtocolSupportError('K MODEL prediction needs selected-reference native support')
            r=state['representations'][state['reference_index']]
            values,_=profile_values(self.accessor,T,r['channel'],r['profile'],r['m_tau'])
            child=predict_contract(values,r['contract'])
        else:
            parent=self.emit(state['parent'],T,source)
            if state['routing_status'] not in ['ADMITTED','MODEL']:return parent
            if stage=='C':
                if state.get('domain',True) and 'contract' in state:
                    p,raw=self.features(state['parent'],T)
                    x=p if state['recipe']=='ADDITIVE_COMPRESSED' else np.concatenate([r[:,ix] for r,ix in zip(raw,state['joint_columns'])],axis=1)
                    child=parent+_ridge_predict(x,state['contract'])
                else:child=parent.copy()
            elif stage=='W':
                # Scalar canonical coordinate: q=F_C; baseline W0(q)=q.
                child=parent+w.predict_w_correction(parent,state['contract']) if 'contract' in state else parent.copy()
            elif stage=='A':
                if source is None:raise ValueError('A prediction requires mature residual stream')
                x,_,_=self.a_features(T,source,state['profile'],state['residual_mean'])
                child=parent+predict_mature_residual_ar(x,state['contract']) if 'contract' in state else parent.copy()
            else:raise ValueError(stage)
        if state['routing_status'] not in ['ADMITTED','MODEL']:return parent
        return child

    def features(self,state,T):
        if len(self.support_for(state,T,representations=True))!=len(T):
            raise ProtocolSupportError('C representation view needs selected-channel assembly support')
        scalars=[];bases=[]
        for r in state['representations']:
            values,_=profile_values(self.accessor,T,r['channel'],r['profile'],r['m_tau'])
            scalars.append(predict_contract(values,r['contract']))
            bases.append(basis_from_metadata(r['contract']['basis']).transform(values).reshape(len(T),-1))
        return np.column_stack(scalars),bases

    def learn(self,stage,T,gid=None):
        if self.metadata_only:raise ProtocolSupportError('NO_TRAINING_INTEGRATION_PREFLIGHT')
        key=self.key(stage,T,gid)
        if key in self.cache:return self.cache[key]
        dest=self.root/'models'/stage/key
        if (dest/'state.json').exists():
            state=json.loads((dest/'state.json').read_text());self.verify_state(state,T)
            self.cache[key]=state;return state
        dest.mkdir(parents=True,exist_ok=True)
        # Full-T realization domain is checked independently before MODEL tuning.
        parent=None if stage=='K' else self.learn({'C':'K','W':'C','A':'W'}[stage],T,gid)
        domain=self.domain(stage,T,parent)
        if not domain['pass']:
            state=dict(stage=stage,parent=parent,domain=False,domain_audit=domain,routing_status='NOT_APPLICABLE',selection=None,identity_is_model_candidate=False)
        else:
            ys=[];ps=[];cs=[];evidence=[]
            admission_block=320 if support_id_hash(T)==getattr(self,'full_train_hash',None) else self.block
            for fold,(fi,ei) in enumerate(folds(T,admission_block)):
                fit=T.iloc[fi].copy();ev=T.iloc[ei].copy()
                # Entire stage MODEL tuning is inside this outer-training partition.
                model=self.tune(stage,fit,gid)
                if stage=='A':ev,source=self.scoring_a(model,ev,T)
                else:source=None
                pp=np.zeros(len(ev)) if stage=='K' else self.emit(model['parent'],ev,source)
                cp=self.emit(model,ev,source)
                if not model['domain']:assert np.array_equal(cp,pp)
                ys.extend(ev.y_true);ps.extend(pp);cs.extend(cp)
                record=ev[['base_origin_id','view_sample_id','entity_id','origin','y_true']].copy()
                record['parent']=pp;record['MODEL_child']=cp;record['fold']=fold
                record.to_parquet(dest/f'admission_fold_{fold}_private.parquet',index=False)
                evidence.append(dict(fold=fold,partition_id=support_id_hash(fit),evaluation_support_hash=support_id_hash(ev),score_n=len(ev),outer_realization_id=digest(model),MODEL_recipe=model.get('recipe'),domain=model['domain'],upstream_route=None if stage=='K' else model['parent']['routing_status'],parent_prediction_hash=prediction_hash(pp),child_prediction_hash=prediction_hash(cp)))
            selection=paired_route(ys,ps,cs)
            state=copy.deepcopy(self.tune(stage,T,gid))
            if not state['domain']:selection['routing_status']='NOT_APPLICABLE'
            state.update(routing_status=selection['routing_status'],selection=selection,outer_realizations=evidence)
        state.update(stage=stage,admission_policy_id=self.policy_id,partition_id=support_id_hash(T),support_hash=support_id_hash(T),fit_rows=len(T),growth_arm=gid,identity_is_model_candidate=False)
        if stage in ['K','C','W']:
            prediction_frame=self.support_for(state,T)
            state['prediction_support_hash']=support_id_hash(prediction_frame)
            state['training_prediction_hash']=prediction_hash(self.emit(state,prediction_frame))
        state['deployment_model_id']=digest(state)
        dump(dest/'state.json',state)
        self.cache[key]=state
        print('CANONICAL_STAGE_FROZEN',self.H,stage,len(T),state['routing_status'],key[:12],flush=True)
        return state

    def verify_state(self,state,T):
        assert state['admission_policy_id']==self.policy_id
        assert state['partition_id']==state['support_hash']==support_id_hash(T)
        check={k:v for k,v in state.items() if k!='deployment_model_id'}
        assert digest(check)==state['deployment_model_id'],'model hash mismatch'
        if state['stage'] in ['K','C','W']:
            prediction_frame=self.support_for(state,T)
            assert support_id_hash(prediction_frame)==state['prediction_support_hash']
            assert prediction_hash(self.emit(state,prediction_frame))==state['training_prediction_hash']
        # Every retained ancestor is in this immutable policy; no R2 cache objects.
        if state.get('parent'):
            assert state['parent']['admission_policy_id']==self.policy_id
            assert state['parent']['partition_id']==support_id_hash(T)

    def domain(self,stage,T,parent):
        if stage=='K':return {'pass':True,'reason':'REGISTERED_CHANNEL_MODEL_REPRESENTATIONS'}
        if stage=='C':return dict(**{'pass':len(parent['representations'])>=2},reason='LEGAL_REPRESENTATION_COUNT',count=len(parent['representations']))
        if stage=='W':
            rule=self.v211['W'];q=self.emit(parent,self.support_for(parent,T))
            audit=w.latent_fold_usable(q,minimum_distinct_values=int(rule['minimum_distinct_latent_values_per_fold']),minimum_rank=int(rule['minimum_rank_of_intercept_and_latent']),scale_floor_multiplier=float(rule['relative_scale_floor_multiplier_float64_eps']))
            return dict(**audit,reason='W_DOMAIN_ACTUAL_CURRENT_T_Q',q_hash=prediction_hash(q),q_interface='q=F_C; W0(q)=q')
        return {'pass':True,'reason':'A_RESIDUAL_DOMAIN_CHECKED_ON_GENERATED_CF_STREAM'}

    def tune(self,stage,T,gid):
        if self.metadata_only:raise ProtocolSupportError('NO_TRAINING_INTEGRATION_PREFLIGHT')
        key=self.key(stage,T,gid,'MODEL_TUNING')
        if key in self.tune_cache:return self.tune_cache[key]
        dest=self.root/'tuning'/stage/key;dest.mkdir(parents=True,exist_ok=True)
        cached=dest/'model.json'
        if cached.exists():
            model=json.loads(cached.read_text());assert model['tuning_partition_id']==support_id_hash(T) and model['policy_id']==self.policy_id
            assert digest({k:v for k,v in model.items() if k!='MODEL_hash'})==model['MODEL_hash']
            self.tune_cache[key]=model;return model
        if stage=='K':model=self.tune_k(T,gid,dest)
        elif stage=='C':model=self.tune_c(T,gid,dest)
        else:model=self.tune_residual(stage,T,gid,dest)
        model.update(stage=stage,policy_id=self.policy_id,tuning_partition_id=support_id_hash(T),routing_status='MODEL' if model['domain'] else 'NOT_APPLICABLE',identity_is_model_candidate=False)
        model['MODEL_hash']=digest(model)
        dump(cached,model);self.tune_cache[key]=model
        return model

    def tune_k(self,T,gid,dest):
        if self.metadata_only:raise ProtocolSupportError('NO_TRAINING_INTEGRATION_PREFLIGHT')
        reps=[];comparison=[];all_folds=folds(T,self.block)
        for ch in CHANNELS:
            channel_key=digest(dict(policy=self.policy_id,support=support_id_hash(T),channel=ch,growth=gid if ch=='main_heater_power' else None))
            channel_dest=self.root/'channel_MODEL_cache'/ch/channel_key
            record_file=channel_dest/'MODEL.json'
            if record_file.exists():
                saved=json.loads(record_file.read_text())
                assert saved['partition_id']==support_id_hash(T) and saved['policy_id']==self.policy_id
                assert digest(saved['representation'])==saved['model_hash']
                rr=saved['representation'];vv,_=profile_values(self.accessor,apply_native_support(T,rr['profile'][1]),ch,rr['profile'],rr['m_tau'])
                assert prediction_hash(predict_contract(vv,rr['contract']))==saved['prediction_hash']
                if comparison:assert saved['assembly']['fold_support_hashes']==comparison[0]['fold_support_hashes']
                reps.append(rr);comparison.append(saved['assembly']);continue
            channel_dest.mkdir(parents=True,exist_ok=True)
            profiles,mts,common=self.k_grid(ch,gid)
            print('CANONICAL_K_GRID',self.H,ch,len(T),gid,flush=True)
            losses,audit=k._run_structural_grid(**self.structural_grid_request(T,ch,gid,channel_dest))
            # No per-channel neutral/admission decision and no representation deletion.
            means={spec:float(np.mean(v)) if np.isfinite(v).all() else float('inf') for spec,v in losses.items()}
            spec=deterministic_best(means);profile,m_tau,family,m_x,lt,lx=spec
            native=apply_native_support(T,int(profile[1]));values,intervals=profile_values(self.accessor,native,ch,profile,m_tau)
            fold_records=registered_fold_native_masks(T,all_folds,fit_history_steps=int(profile[1]),scoring_history_steps=common,fit_cap=int(self.v2['row_caps']['single_channel_k_fit']),evaluation_cap=int(self.v2['row_caps']['validation_selection_per_fold']))
            def full(l0):return fit_contract(values,native.y_true.to_numpy(float),family,m_x,(l0,lt,lx),**v2_k._als_kwargs(self.v2))
            def folded(l0):
                result=[]
                for j,r in enumerate(fold_records):
                    f=r['fit'];e=r['evaluation'];x,_=profile_values(self.accessor,f,ch,profile,m_tau);z,_=profile_values(self.accessor,e,ch,profile,m_tau)
                    ct=fit_contract(x,f.y_true.to_numpy(float),family,m_x,(l0,lt,lx),**v2_k._als_kwargs(self.v2))
                    p=predict_contract(z,ct)
                    result.append(dict(fold=j,contract=ct,fit_rows=len(f),prediction=p,loss=risk(e.y_true,p)))
                return result
            l0,ct,replay,stability=k.select_smallest_stable_full_and_folds(self.v2['K_module']['penalties']['lambda_0'],full,folded,valid_full=lambda c:k._candidate_valid(c,len(native),self.v2),valid_fold=lambda r:k._candidate_valid(r['contract'],r['fit_rows'],self.v2) and np.isfinite(r['prediction']).all(),parallel_workers=1)
            # Re-score the selected, certified frozen recipe on common assembly rows.
            score_ids=[support_id_hash(r['evaluation']) for r in fold_records]
            if comparison:assert score_ids==comparison[0]['fold_support_hashes']
            assembly_loss=float(np.average([r['loss'] for r in replay],weights=[len(r['evaluation']) for r in fold_records]))
            reps.append(dict(channel=ch,profile=list(profile),m_tau=m_tau,family=family,m_x=m_x,lambdas=[l0,lt,lx],contract=ct,intervals=intervals,native_fit_support_hash=support_id_hash(native),numerical_ridge_audit=stability,representation_preserved=True))
            comparison.append(dict(channel=ch,inner_MSE=assembly_loss,fold_support_hashes=score_ids,selection_inner_MSE=means[spec],candidate_count=len(losses)))
            dump(channel_dest/'MODEL_SELECTION.json',dict(selected_spec=spec,selected_inner_risk=means[spec],assembly=comparison[-1],all_candidate_risks={str(s):v for s,v in means.items()},numerical_invalid_candidates=[str(s) for s,v in means.items() if not np.isfinite(v)]))
            pv,_=profile_values(self.accessor,native,ch,profile,m_tau)
            dump(record_file,dict(policy_id=self.policy_id,partition_id=support_id_hash(T),support_hash=support_id_hash(T),prediction_support_hash=support_id_hash(native),route_state='MODEL_REPRESENTATION_PRESERVED_NOT_INDIVIDUALLY_GATED',representation=reps[-1],model_hash=digest(reps[-1]),prediction_hash=prediction_hash(predict_contract(pv,ct)),assembly=comparison[-1]))
        reference=int(np.argmin([x['inner_MSE'] for x in comparison]))
        return dict(domain=True,representations=reps,reference_index=reference,reference_channel=CHANNELS[reference],assembly_comparison=comparison,recipe='K_RESPONSE_POLICY_ALL_REGISTERED_CHANNELS',neutral_parent='B=0 registered H1/H4 change baseline; no implicit mean',reference_uses_training_only=True)

    def c_design(self,state,fit,ev):
        x,raw=self.features(state,fit);z,rawe=self.features(state,ev);columns=[];xf=[];xe=[]
        cfg=self.v2['C_module']['joint_basis']
        for f,e in zip(raw,rawe):
            ix=_qr_columns(f,int(cfg['per_channel_max_columns']),float(cfg['per_channel_qr_relative_diagonal_floor']))
            columns.append(ix.tolist());xf.append(f[:,ix]);xe.append(e[:,ix])
        return {'ADDITIVE_COMPRESSED':(x,z),'ADDITIVE_JOINT_BASIS':(np.concatenate(xf,axis=1),np.concatenate(xe,axis=1))},columns

    def tune_c(self,T,gid,dest):
        parent=self.learn('K',T,gid);domain=self.domain('C',T,parent)
        if not domain['pass']:return dict(parent=parent,domain=False,domain_audit=domain)
        families=['ADDITIVE_COMPRESSED','ADDITIVE_JOINT_BASIS']
        # Canonical numerical-only ridge: zero is tried first; predictive grids unchanged.
        alphas=sorted(set([0.]+list(map(float,self.v2['C_module']['joint_basis']['ridge_alpha_grid']))))
        assembly=self.support_for(parent,T,representations=True)
        designs,columns=self.c_design(parent,assembly,assembly);kp=self.emit(parent,assembly);records=[]
        for j,(fi,ei) in enumerate(folds(T,self.block)):
            fit=T.iloc[fi].copy();ev=T.iloc[ei].copy();ks=self.learn('K',fit,gid)
            fit=self.support_for(ks,fit,representations=True);ev=self.support_for(ks,ev,representations=True)
            dd,_=self.c_design(ks,fit,ev);pf=self.emit(ks,fit);pe=self.emit(ks,ev)
            records.append((fit,ev,ks,dd,pf,pe))
        candidates={};contracts={};cert_audit={}
        for fam in families:
            audit=[]
            for alpha in alphas:
                try:
                    _,full=_ridge_fit(designs[fam][0],assembly.y_true.to_numpy(float)-kp,designs[fam][0],alpha)
                    vals=[];valid=numerical_contract_passes(full)
                    for fit,ev,ks,dd,pf,pe in records:
                        if len(ks['representations'])<2:
                            vals.append(risk(ev.y_true,pe));continue
                        p,ct=_ridge_fit(dd[fam][0],fit.y_true.to_numpy(float)-pf,dd[fam][1],alpha)
                        valid&=numerical_contract_passes(ct);vals.append(risk(ev.y_true,pe+p))
                    audit.append(dict(alpha=alpha,all_certificates_pass=bool(valid)))
                    if valid:
                        candidates[fam]=float(np.mean(vals));contracts[fam]=(alpha,full);break
                except (ValueError,np.linalg.LinAlgError) as exc:audit.append(dict(alpha=alpha,error=str(exc),all_certificates_pass=False))
            cert_audit[fam]=audit
        fam=deterministic_best(candidates);alpha,ct=contracts[fam]
        dump(dest/'C_INNER_AND_NUMERICAL_EVIDENCE.json',dict(inner_risk=candidates,numerical_audit=cert_audit,zero_ridge_trial_required_by='v2.2 section 13.1',target='y_true - frozen routed F_K'))
        return dict(parent=parent,domain=True,recipe=fam,selected_alpha=alpha,contract=ct,joint_columns=columns,inner_risk=candidates,domain_audit=domain,fit_target='y_true-F_K',additive_parent_preservation=True)

    def stream(self,stage,T,gid):
        key=self.key(stage,T,gid,'PREQUENTIAL_CF')
        if key in self.stream_cache:return self.stream_cache[key]
        blocks=[];provenance=[]
        for fi,ei,j in cf_blocks(T,self.block,self.warmup):
            fit=T.iloc[fi].copy();ev=T.iloc[ei].copy();parent=self.learn(stage,fit,gid)
            p=self.emit(parent,ev);g=ev.copy();g['parent_prediction']=p;g['residual']=g.y_true.to_numpy(float)-p;g['CF_fold']=j
            ks=parent if stage=='K' else parent['parent'] if stage=='C' else parent['parent']['parent']
            comp,_=self.features(ks,ev)
            for index,ch in enumerate(CHANNELS):g['up_'+ch]=comp[:,index]
            # A overlap penalization retains actual W correction, as in registered family.
            if stage=='W':g['up_w']=p-self.emit(parent['parent'],ev)
            if j==0:g.loc[g.index[:-self.block],'CF_fold']=-1
            blocks.append(g)
            provenance.append(dict(fold=j,fit_support_hash=support_id_hash(fit),score_support_hash=support_id_hash(ev),score_n=len(ev),fit_dependency_stop_exclusive=int(fit.dependency_stop_exclusive.max()),evaluation_dependency_start=int(ev.dependency_start.min()),upstream_model_id=parent['deployment_model_id'],prediction_hash=prediction_hash(p),target_used_to_fit_own_predictor=False,raw_purge_preserved=True))
        source=pd.concat(blocks,ignore_index=True)
        assert not source[['entity_id','origin']].duplicated().any()
        source=source.sort_values(['entity_id','origin']).reset_index(drop=True)
        path=self.root/'streams'/stage/key/'prequential.parquet';path.parent.mkdir(parents=True,exist_ok=True)
        source.to_parquet(path,index=False);dump(path.with_suffix('.json'),dict(provenance=provenance,support_hash=support_id_hash(source),file_sha256=file_sha(path),stage=stage))
        self.stream_cache[key]=(source,provenance,str(path));return self.stream_cache[key]

    def a_features(self,T,source,profile,mean):
        return mature_residual_features(T,source,h_steps=self.view.head.h_steps,w_steps=self.view.head.w_steps,delta=profile[0],history=profile[1],maximum_lags=self.A_max,residual_mean=mean)

    def mature(self,T,source):
        return common_mature(T,source,self.A_common,self.view.head.h_steps,self.view.head.w_steps,native_support_mask)

    def final_development_stream(self,upstream,T,gid,selection_freeze):
        from final_development_cf import replay
        return replay(self,upstream,T,gid,selection_freeze,CHANNELS)

    def tune_residual(self,stage,T,gid,dest):
        upstream='C' if stage=='W' else 'W';parent=self.learn(upstream,T,gid);domain=self.domain(stage,T,parent)
        if not domain['pass']:return dict(parent=parent,domain=False,domain_audit=domain,recipe=None,selection=None)
        source,provenance,path=self.stream(upstream,T,gid)
        if stage=='A' and not self.mature(source,source).any():
            return dict(parent=parent,domain=False,domain_audit={'pass':False,'reason':'NO_LEGAL_MATURE_CF_RESIDUAL_HISTORY'},source_path=path,provenance=provenance)
        if stage=='W':
            rule=self.v211['W']
            cf_domain=w.latent_fold_usable(source.parent_prediction.to_numpy(float),minimum_distinct_values=int(rule['minimum_distinct_latent_values_per_fold']),minimum_rank=int(rule['minimum_rank_of_intercept_and_latent']),scale_floor_multiplier=float(rule['relative_scale_floor_multiplier_float64_eps']))
            if not cf_domain['pass']:
                return dict(parent=parent,domain=False,domain_audit={'pass':False,'reason':'W_CF_COORDINATE_NOT_CONSTRUCTABLE','full_T_domain':domain,'CF_domain':cf_domain},source_path=path,provenance=provenance)
        specs=w._w_candidates(self.v211,self.v21,self.v2,1,False) if stage=='W' else [(pr,float(a),float(mu)) for pr in self.A_profiles for a in self.v2['A_module']['ridge_alpha_grid'] for mu in self.v21['A']['soft_overlap_mu']]
        losses={spec:[] for spec in specs};fold_support=[];na_folds=[]
        for j in range(4):
            fit=source[source.CF_fold<j].copy();ev=source[source.CF_fold==j].copy()
            if stage=='A':ev=ev.loc[self.mature(ev,source)].copy()
            if not len(ev):raise ProtocolSupportError(f'{stage} inner common mature scoring support empty')
            # Fit-side residual labels must be mature before the first scoring origin.
            max_origin=int(ev.latest_available_target_index.min())-self.view.head.h_steps-self.view.head.w_steps+1
            fit=fit.loc[fit.origin<=max_origin].copy()
            fold_support.append(dict(fold=j,fit_n=len(fit),score_n=len(ev),fit_support_hash=support_id_hash(fit),score_support_hash=support_id_hash(ev)))
            if stage=='W':
                rule=self.v211['W'];usable=w.latent_fold_usable(fit.parent_prediction.to_numpy(float),minimum_distinct_values=int(rule['minimum_distinct_latent_values_per_fold']),minimum_rank=int(rule['minimum_rank_of_intercept_and_latent']),scale_floor_multiplier=float(rule['relative_scale_floor_multiplier_float64_eps']))
                if not usable['pass']:
                    for spec in specs:losses[spec].append(risk(ev.residual,np.zeros(len(ev))))
                    na_folds.append(dict(fold=j,domain=usable,child_equals_parent=True,rows_retained=len(ev)));continue
            for spec in specs:
                try:
                    ct,p=self.fit_residual_recipe(stage,spec,fit,ev,source)
                    loss=risk(ev.residual,p) if numerical_contract_passes(ct) else float('inf')
                except (ValueError,np.linalg.LinAlgError):loss=float('inf')
                losses[spec].append(loss)
        means={spec:float(np.average(v,weights=[r['score_n'] for r in fold_support])) if np.isfinite(v).all() else float('inf') for spec,v in losses.items()}
        spec=deterministic_best(means)
        from final_development_cf import require_contract
        selection_freeze=dict(role='FROZEN_RESIDUAL_RECIPE_BEFORE_FINAL_REPLAY',
            stage=stage,upstream_stage=upstream,policy_id=self.policy_id,
            partition_id=support_id_hash(T),selected_spec=spec,
            inner_source_sha256=file_sha(path),inner_source_support_hash=support_id_hash(source),
            replay_contract=require_contract(self))
        dump(dest/'SELECTED_RECIPE_BEFORE_FINAL_REPLAY.json',selection_freeze)
        final_source,final_audit,final_path=self.final_development_stream(upstream,T,gid,selection_freeze)
        ct,_=self.fit_residual_recipe(stage,spec,final_source,final_source,final_source)
        if not numerical_contract_passes(ct):raise NumericallyInvalid('Selected residual MODEL final refit failed numerical certificate')
        model=dict(parent=parent,domain=True,domain_audit=domain,recipe=spec,contract=ct,inner_risk=means[spec],source_path=final_path,source_sha256=file_sha(final_path),provenance=final_audit['provenance'],fold_support=fold_support,na_inner_folds=na_folds,CF_training_target='y_true - per-row recursive held-out upstream prediction',uses_in_sample_parent_residual=False,
            inner_selection_source_path=path,inner_selection_source_sha256=file_sha(path),
            inner_selection_provenance=provenance,selection_freeze_hash=digest(selection_freeze),
            final_replay_coverage=final_audit['coverage'],final_replay_binding=final_audit['binding'])
        if stage=='A':model.update(profile=list(spec[0]),residual_mean=float(apply_native_support(final_source,spec[0][1]).residual.mean()),maximum_lags=self.A_max)
        dump(dest/'MODEL_INNER_RISKS.json',dict(selected_spec=spec,all_risks={str(s):v for s,v in means.items()},fold_support=fold_support,domain_absent_folds=na_folds))
        return model

    def fit_residual_recipe(self,stage,spec,fit,ev,source):
        up=fit[[c for c in fit if c.startswith('up_')]].to_numpy(float)
        if stage=='W':
            family,knots,smoothness,mu,direction=spec
            p,ct=w.fit_w_correction(fit.parent_prediction.to_numpy(float),fit.residual.to_numpy(float),ev.parent_prediction.to_numpy(float),family=family,knot_count=int(knots),smoothness=float(smoothness),mu=float(mu),upstream_predictions=up,direction=int(direction))
        else:
            profile,alpha,mu=spec;fit=apply_native_support(fit,profile[1]);mean=float(fit.residual.mean())
            x,_,_=self.a_features(fit,source,profile,mean);z,_,_=self.a_features(ev,source,profile,mean)
            up=fit[[c for c in fit if c.startswith('up_')]].to_numpy(float)
            p,ct=fit_mature_residual_ar(x,fit.residual.to_numpy(float),z,alpha=float(alpha),mu=float(mu),upstream_predictions=up)
        return ct,p

    def scoring_a(self,model,ev,universe):
        """Observed warmup inference only, never fitted into the scoring model."""
        previous=pd.read_parquet(model['source_path']) if 'source_path' in model else pd.DataFrame(columns=['entity_id','origin','residual'])
        histories=[]
        for entity,g in ev.groupby('entity_id',sort=False):
            start=int(g.origin.min())-self.A_common-self.view.head.h_steps-self.view.head.w_steps
            end=int(g.origin.max())
            h=universe.loc[(universe.entity_id==entity)&(universe.origin>=start)&(universe.origin<=end)].copy()
            h['residual']=h.y_true.to_numpy(float)-self.emit(model['parent'],h)
            histories.append(h[['entity_id','origin','residual']])
        source=pd.concat([previous,*histories],ignore_index=True)
        source=source.drop_duplicates(['entity_id','origin'],keep='first')
        scored=ev.loc[self.mature(ev,source)].copy()
        if len(scored)!=len(ev):raise ProtocolSupportError('Registered admission rows lost to A support; need lawful warmup')
        return scored,source
