"""Finite stagewise adaptation of GitHub 6ba6f1b, with outer holdouts isolated.

Upstream recipes are fixed within a parent training partition. Parameters,
PCA and C column selection are refitted on each lawful past-only subpartition.
This is parameter-CF; it does not claim fully nested hyperparameter-CF.
"""
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .canonical_engine import CanonicalPipeline
from .invariants import ProtocolError,NumericallyInvalid,digest,frame_hash,file_sha
from .fold_geometry import folds,selection_blocks
from .partition import ordered_ids
from .risk import paired_route
from .pointwise import paired_pointwise,export_pointwise
from .a_state import build_a_training_state_from_cf_residual,feature_rows,common_scoring_mask,TRAINING_CF
from .candidate_status import SupportLimited,DomainNotApplicable,require_k_support,classify_exception


class StagewisePipeline(CanonicalPipeline):
    def __init__(self,context,root,*,mode):
        super().__init__(context,root,mode=mode)
        self.fixed_cache={};self.fixed_stream_cache={};self.local_chain_cache={};self.active_parents={};self.import_count=0
        self.a_input_cache={}
        self.migration=self.protocol['stagewise_migration']
        self.legacy=Path(self.migration['legacy_root'])
        self.reuse=json.loads(Path(self.migration['K_reuse_receipt']).read_text())
        if file_sha(self.migration['K_reuse_receipt'])!=self.migration['K_reuse_receipt_sha256']:
            raise ProtocolError('K compatibility receipt changed')

    def tune_k(self,T,destination):
        # K has no upstream dependency. Its registry, math, data, geometry,
        # environment and complete partition are all independently verified.
        if self.mode!='formal_development':return super().tune_k(T,destination)
        oldkey=digest(dict(policy=self.migration['legacy_policy_id'],mode=self.mode,stage='K',role='MODEL',frame=frame_hash(T)))
        permitted={r['key']:r for r in self.reuse[self.context['case_id']]['imports']}
        if oldkey not in permitted:
            raise ProtocolError('Missing exact K reuse: refusing an unregistered long search '+oldkey)
        path=self.legacy/'formal_actual'/self.context['case_id']/'checkpoints/MODEL_K'/(oldkey+'.json')
        if file_sha(path)!=permitted[oldkey]['sha256']:raise ProtocolError('Imported K checkpoint changed')
        saved=json.loads(path.read_text());binding=saved['binding']
        if (saved['key']!=oldkey or saved['role']!='MODEL_K' or digest(saved['value'])!=saved['value_hash']
            or binding['policy_id']!=self.migration['legacy_policy_id'] or binding['release_sha']!=self.migration['legacy_release_sha256']
            or binding['case_id']!=self.context['case_id'] or binding['environment']!=self.checkpoints.binding['environment']):
            raise ProtocolError('K source identity/content/environment mismatch')
        for p,sha in saved['artifact_sha256'].items():
            if file_sha(p)!=sha:raise ProtocolError('K numerical model artifact changed')
        model=copy.deepcopy(saved['value']['model'])
        if model['tuning_partition_id']!=self.partition_id(T):raise ProtocolError('K fit rows differ')
        for key in ('stage','parent','routing_status','policy_id','tuning_partition_id','identity_is_model_candidate','MODEL_hash'):
            model.pop(key,None)
        self.import_count+=1
        self._save(destination/'K_EXACT_REUSE.json',dict(source=str(path),sha256=file_sha(path),old_key=oldkey,
            new_policy_id=self.policy_id,partition_hash=frame_hash(T),rows=len(T),numerical_parameters_unchanged=True))
        print('K_EXACT_REUSE',self.context['case_id'],len(T),oldkey,flush=True)
        return model

    def tune_c(self,T,parent,destination):
        from .stagewise_tuning import tune_c
        return tune_c(self,T,parent,destination)

    def tune_residual(self,stage,T,parent,destination):
        from .stagewise_tuning import tune_residual
        return tune_residual(self,stage,T,parent,destination)

    def local_chain(self,stage,T):
        """One stagewise recipe chain inside one EXTERNAL outer fit partition.

        Inner stage routes use fixed recipes and fold-local parameters. No
        inner held-out row is used to fit its own parameters; recipe selection
        is local to the enclosing external training partition.
        """
        key=self._key(stage,T,'LOCAL_STAGEWISE_CHAIN')
        if key in self.local_chain_cache:return self.local_chain_cache[key]
        if stage=='K':state=self.learn('K',T)
        else:
            upstream={'C':'K','W':'C','A':'W'}[stage]
            parent=self.local_chain(upstream,T)
            state=copy.deepcopy(self.build_model(stage,T,parent))
            ys=[];pp=[];cp=[]
            for fi,ei in folds(T,self.protocol['geometry']['inner']):
                model=self.refit_fixed(state,T.iloc[fi]);ev=T.iloc[ei].copy()
                p=self.emit(model['parent'],ev.drop(columns=['y_true']))
                c,_,_=self.online_a(model,ev) if stage=='A' else self.predict(model,ev.drop(columns=['y_true']))
                ys.extend(ev.y_true);pp.extend(p);cp.extend(c)
            if ys:
                ids=[sample for _,ei in folds(T,self.protocol['geometry']['inner']) for sample in ordered_ids(T.iloc[ei])]
                selection=paired_route(ys,pp,cp,score_ids=ids)
                if not state['domain']:selection=dict(selection,routing_status='NOT_APPLICABLE')
            else:selection=dict(routing_status='BYPASSED',reason='LOCAL_STAGEWISE_GEOMETRY_UNAVAILABLE')
            state.update(routing_status=selection['routing_status'],selection=selection,stage=stage,
                admission_policy_id=self.policy_id,partition_id=self.partition_id(T),fit_rows=len(T),
                execution_mode=self.mode,formal_eligible=self.mode=='formal_development',
                route_scope='WITHIN_EXTERNAL_OUTER_FIT',fully_nested_hyperparameter_CF=False)
            state['partition_model_id']=digest(state)
            self._save(self.root/'local_chains'/stage/(key+'.json'),state)
        self.local_chain_cache[key]=state;return state

    def build_model(self,stage,T,parent):
        key=self._key(stage,T,'MODEL');T=T.reset_index(drop=True)
        if key in self.tune_cache:
            model=self.tune_cache[key]
            if digest(model['parent'])!=digest(parent):raise ProtocolError('Stagewise MODEL parent changed')
            return model
        saved=self.checkpoints.load('MODEL_'+stage,key)
        if saved is not None:
            model=saved['model']
            if digest(model['parent'])!=digest(parent):raise ProtocolError('Resumed stagewise MODEL parent changed')
            self.artifacts.extend(saved['artifacts']);self.tune_cache[key]=model;return model
        destination=self.root/'tuning'/stage/key
        if not folds(T,self.protocol['geometry']['inner']):model=self._base_case(stage,parent)
        else:
            self.active_parents[(parent['stage'],frame_hash(T))]=parent
            model=self.tune_c(T,parent,destination) if stage=='C' else self.tune_residual(stage,T,parent,destination)
        model.update(stage=stage,parent=parent,routing_status='MODEL' if model['domain'] else model['routing_status'],
            policy_id=self.policy_id,tuning_partition_id=self.partition_id(T),identity_is_model_candidate=False)
        model['MODEL_hash']=digest(model);self._save(destination/'model.json',model)
        self.tune_cache[key]=model
        artifacts=self._references(model)+[str((destination/'model.json').resolve())]
        self.checkpoints.save('MODEL_'+stage,key,dict(model=model,artifacts=artifacts),artifacts)
        return model

    def tune(self,stage,T):
        if stage=='K':return super().tune(stage,T)
        upstream={'C':'K','W':'C','A':'W'}[stage]
        parent_key=self._key(upstream,T,'ADMISSION')
        parent=self.cache.get(parent_key)
        if parent is None:parent=self.local_chain(upstream,T)
        return self.build_model(stage,T,parent)

    def refit_fixed(self,template,fit):
        """Fit only frozen recipes; no call to learn/tune/recipe enumeration."""
        self._fit_guard();fit=fit.reset_index(drop=True)
        key=digest(dict(template=template.get('partition_model_id',template.get('MODEL_hash',digest(template))),fit=frame_hash(fit)))
        if key in self.fixed_cache:return self.fixed_cache[key]
        stage=template['stage']
        parent=None if stage=='K' else self.refit_fixed(template['parent'],fit)
        out=dict(stage=stage,parent=parent,domain=False,routing_status=template['routing_status'],
            policy_id=self.policy_id,admission_policy_id=self.policy_id,partition_id=self.partition_id(fit),
            fixed_recipe_source_partition=template.get('partition_id',template.get('tuning_partition_id')),
            fit_rows=len(fit),fixed_recipe_no_search=True,identity_is_model_candidate=False)
        if not len(fit):
            if stage=='K':out.update(representations=[],reference_index=0,reference_channel=template.get('reference_channel'))
            out.update(routing_status='BYPASSED',realization_status='SUPPORT_LIMITED',identity_reason='EMPTY_FIXED_REFIT_PARTITION')
            out['partition_model_id']=digest(out);self.fixed_cache[key]=out
            return out
        if stage=='K':
            reps=[];statuses={}
            for recipe in template.get('representations') or []:
                r=copy.deepcopy(recipe);native=fit.loc[self.native(fit,int(r['profile'][1]))]
                if len(native)<int(self.v2['row_caps']['minimum_rows_for_any_fitted_candidate']):
                    statuses[r['channel']]='SUPPORT_LIMITED';continue
                accessor=self.fold_accessor(native)
                x,intervals=self.v2_k.profile_values(accessor,native,r['channel'],tuple(r['profile']),int(r['m_tau']))
                if not np.any(np.ptp(x,axis=0)>0):statuses[r['channel']]='DOMAIN_NOT_APPLICABLE';continue
                try:
                    self.fit_calls+=1
                    ct=self.u.fit_contract(x,native.y_true.to_numpy(float),r['family'],r['m_x'],tuple(r['lambdas']),**self.v2_k._als_kwargs(self.v2))
                    require_k_support(ct,len(native),self.v2)
                    if not self.k._candidate_valid(ct,len(native),self.v2):raise NumericallyInvalid('Fixed K numerical certificate failed')
                except (SupportLimited,DomainNotApplicable) as e:
                    statuses[r['channel']]=classify_exception(e);continue
                r.update(contract=ct,intervals=intervals,joint_lift_pca=accessor.pca,native_fit_support_hash=self.partition_id(native))
                reps.append(r);statuses[r['channel']]='AVAILABLE'
            reference=template.get('reference_channel')
            indices=[i for i,r in enumerate(reps) if r['channel']==reference]
            out.update(representations=reps,channel_status=statuses,reference_channel=reference,
                reference_index=indices[0] if indices else 0,domain=bool(template.get('domain') and indices))
        elif stage=='C':
            if template.get('domain') and len(parent.get('representations') or [])>=2:
                dd,columns,mf,_=self._c_design(parent,fit,fit);family=template['recipe']
                design=dd[family][0][mf]
                if mf.sum()>=2 and design.shape[1] and np.any(np.ptp(design,axis=0)>0):
                    self.fit_calls+=1
                    _,ct=self.c._ridge_fit(design,fit.y_true.to_numpy(float)[mf]-self.emit(parent,fit)[mf],design,template['selected_alpha'])
                    if not self.cert(ct):raise NumericallyInvalid('Fixed C numerical certificate failed')
                    out.update(domain=True,recipe=family,selected_alpha=template['selected_alpha'],contract=ct,joint_columns=columns)
        elif stage in {'W','A'}:
            if template.get('domain') and folds(fit,self.protocol['geometry']['inner']):
                upstream='C' if stage=='W' else 'W'
                source,_,_=self.fixed_stream(upstream,fit,template['parent'])
                ct,_,mask,audit=self.fit_residual_recipe(stage,template['recipe'],source,source,source)
                if ct is not None:
                    out.update(domain=True,recipe=template['recipe'],contract=ct,final_fit_rows=int(mask.sum()))
                    if stage=='A':out.update(profile=template['profile'],residual_mean=audit['residual_mean'],maximum_lags=self.A_max)
        else:raise ProtocolError('Only registered K/C/W/A fixed refits are permitted')
        if not out['domain']:
            out.update(routing_status='BYPASSED',realization_status='SUPPORT_OR_DOMAIN_ABSENT',identity_reason='FIXED_RECIPE_NATIVE_DOMAIN_ABSENT')
        out['partition_model_id']=digest(out);self.fixed_cache[key]=out
        self._save(self.root/'fixed_refits'/stage/(key+'.json'),dict(stage=stage,partition_id=out['partition_id'],
            rows=len(fit),model_id=out['partition_model_id'],recipe_source=out['fixed_recipe_source_partition'],
            no_candidate_search=True,fold_local_PCA=True,domain=out['domain']))
        return out

    def fixed_stream(self,upstream,T,template):
        self._fit_guard()
        key=digest(dict(policy=self.policy_id,stage=upstream,frame=frame_hash(T),template=template.get('partition_model_id',digest(template))))
        if key in self.fixed_stream_cache:return self.fixed_stream_cache[key]
        records=[];proof=[]
        blocks=selection_blocks(T,self.protocol['geometry']['inner'],self.protocol['geometry']['selection_cf_context_rows_per_block'])
        if not blocks:raise SupportLimited('Finite stagewise CF geometry unavailable')
        for fi,ei,context,score,j in blocks:
            parent=self.refit_fixed(template,T.iloc[fi]);ev=T.iloc[ei].copy()
            p=self.emit(parent,ev.drop(columns=['y_true']))
            ev['parent_prediction']=p;ev['residual']=ev.y_true.to_numpy(float)-p
            if upstream=='C':
                q,native,_=self.response_coordinate(parent,ev.drop(columns=['y_true']))
                ev['response_coordinate']=q;ev['coordinate_available']=native
            ev['residual_source_role']='CROSS_FITTED_PREQUENTIAL';ev['_position']=ei
            ev['CF_fold']=-1;ev.loc[ev.index.isin(T.iloc[score].index),'CF_fold']=j
            overlap,domains=self.overlap_components(parent,ev.drop(columns=['y_true']))
            for col in overlap:ev[col]=overlap[col].to_numpy()
            if upstream=='W':ev['up_w']=p-self.emit(parent['parent'],ev.drop(columns=['y_true']))
            records.append(ev)
            proof.append(dict(fold=j,fit_rows=len(fi),fit_support_hash=self.partition_id(T.iloc[fi]),
                context_n=len(context),score_n=len(score),upstream_partition_model_id=parent['partition_model_id'],
                component_domains=domains,target_used_to_fit_own_parameters=False,
                upstream_recipe_scope='FIXED_WITHIN_PARENT_TRAINING_PARTITION',fully_nested_hyperparameter_CF=False,
                score_support_hash=self.partition_id(T.iloc[score])))
        source=pd.concat(records,ignore_index=True)
        roles={int(pos):j for j,(_,ei) in enumerate(folds(T,self.protocol['geometry']['inner'])) for pos in ei}
        source=source.drop_duplicates(['entity_id','origin'],keep='first').sort_values('_position').reset_index(drop=True)
        source['CF_fold']=[roles.get(int(p),-1) for p in source._position]
        path=self.root/'stagewise_streams'/upstream/key/'prequential.parquet';path.parent.mkdir(parents=True,exist_ok=True)
        source.to_parquet(path,index=False,compression='zstd');self.artifacts.append(str(path.resolve()))
        self._save(path.with_suffix('.json'),dict(role='FINITE_STAGEWISE_PARAMETER_CF',provenance=proof,
            source_hash=frame_hash(source),file_sha256=file_sha(path),rows=len(source),universe_rows=len(T),
            all_development_anchors_covered=len(source)==len(T),fully_nested_hyperparameter_CF=False))
        self.fixed_stream_cache[key]=(source,proof,str(path));return self.fixed_stream_cache[key]

    def stream(self,upstream,T):
        parent=self.active_parents.get((upstream,frame_hash(T)))
        if parent is None:raise ProtocolError('CF stream lacks a frozen parent recipe')
        return self.fixed_stream(upstream,T,parent)

    def online_a(self,model,ev):
        child,mask,proof=super().online_a(model,ev)
        # One-row BLAS evaluation may differ in the final bit from a batched
        # upstream evaluation. An unavailable A row is EXACT parent identity,
        # so keep the single frozen batched baseline on that support.
        baseline=self.emit(model['parent'],ev.drop(columns=['y_true']))
        child[~mask]=baseline[~mask]
        proof=dict(proof,unavailable_A_rows_exact_parent_identity=True)
        return child,mask,proof

    def fit_residual_recipe(self,stage,spec,fit,ev,source,*,comparison=False):
        if stage!='A':return super().fit_residual_recipe(stage,spec,fit,ev,source,comparison=comparison)
        self._fit_guard()
        # Keep owning references, so object IDs cannot be recycled. All these
        # frames are pipeline-owned immutable CF partitions after construction.
        key=(id(fit),id(ev),id(source),tuple(spec[0]),bool(comparison))
        if key not in self.a_input_cache:
            state=build_a_training_state_from_cf_residual(source,policy_id=self.policy_id,h_steps=1)
            _,native,_=feature_rows(fit,state,spec[0],self.A_max,0.,expected_role=TRAINING_CF)
            if native.sum()<2:return None,np.zeros(len(ev)),np.zeros(len(ev),bool),dict(candidate_status='DOMAIN_NOT_APPLICABLE')
            selected=fit.loc[native].copy();mean=float(selected.residual.mean())
            x,_,_=feature_rows(selected,state,spec[0],self.A_max,mean,expected_role=TRAINING_CF)
            z,mask,audit=feature_rows(ev,state,spec[0],self.A_max,mean,expected_role=TRAINING_CF)
            if comparison:
                mask &= common_scoring_mask(ev,state,self.A_profiles,self.A_max)
                z=feature_rows(ev.loc[mask],state,spec[0],self.A_max,mean,expected_role=TRAINING_CF)[0]
            if not mask.any():return None,np.zeros(len(ev)),mask,dict(residual_mean=mean)
            up=selected[[c for c in selected if c.startswith('up_')]].to_numpy(float)
            if up.shape[1]:up=up[:,np.ptp(up,axis=0)>0]
            self.a_input_cache[key]=(fit,ev,source,x,z,mask.copy(),mean,audit,up,selected.residual.to_numpy(float))
        _,_,_,x,z,mask,mean,audit,up,target=self.a_input_cache[key]
        self.fit_calls+=1
        p,ct=self.a.fit_mature_residual_ar(x,target,z,alpha=float(spec[1]),mu=float(spec[2]),upstream_predictions=up)
        output=np.zeros(len(ev));output[mask]=p
        if not self.cert(ct):raise NumericallyInvalid('A residual numerical certificate failed')
        return ct,output,mask.copy(),dict(residual_mean=mean,feature_audit=audit)

    def development_complete(self,state):
        complete=super().development_complete(state)
        if complete:
            self._save(self.root/'STAGEWISE_COMPLETION.json',dict(status='DEVELOPMENT_COMPLETE',K_models_imported=self.import_count,
                fit_calls=self.fit_calls,per_320_prefix_recursive_search=False,
                final_residual_fit='SAVED_FINITE_STAGEWISE_CF',fully_nested_hyperparameter_CF=False))
        return complete
