"""Fixed-route, past-only selection/refit policy for development replay.

No admission function is called here. Recipes are selected locally at a refresh
checkpoint, then parameters alone are refitted on each intervening prefix.
"""
import copy
from pathlib import Path
import numpy as np
from .invariants import ProtocolError,NumericallyInvalid,digest,frame_hash,dump,file_sha
from .candidate_status import SupportLimited,DomainNotApplicable,require_k_support
from .fold_geometry import folds

UP={'C':'K','W':'C'}

def routes_from_state(state):
    result={}
    while state is not None:
        result[state['stage']]=state['routing_status'];state=state.get('parent')
    return result

def cadence(frame):
    from .temporal_support import final_replay_blocks
    local={}
    for global_index,(fi,ei) in enumerate(final_replay_blocks(frame,320)):
        entity=str(frame.iloc[ei[0]].entity_id);index=local.get(entity,0);local[entity]=index+1
        yield global_index,fi,ei,entity,index,index%4==0

class ReplaySession:
    def __init__(self,pipe,routes):
        self.pipe=pipe;self.routes=dict(routes);self.selected={};self.epoch={};self.events=[]
        self.previous=None
    def identity(self):
        return dict(role='FIXED_ROUTE_PAST_ONLY_MODEL_SELECTION',routes=self.routes,
            policy=self.pipe.protocol['revision'],cadence=[1280,320])
    def __enter__(self):
        self.previous=self.pipe.replay_session;self.pipe.replay_session=self;return self
    def __exit__(self,*args):self.pipe.replay_session=self.previous
    def _identity(self,stage,T,parent,reason):
        model=self.pipe._unavailable_model(stage,parent,'SUPPORT_LIMITED',reason=reason)
        return self._finish(stage,T,model)
    def _finish(self,stage,T,model):
        model=copy.deepcopy(model)
        ancestors={'K':['K'],'C':['K','C'],'W':['K','C','W']}[stage]
        model.update(stage=stage,admission_policy_id=self.pipe.policy_id,partition_id=self.pipe.partition_id(T),
            fit_rows=len(T),routing_status=self.routes[stage],admission_vote_produced=False,
            replay_route_frozen=True,replay_route_vector={k:self.routes[k] for k in ancestors},policy_id=self.pipe.policy_id)
        model['partition_model_id']=digest(model)
        return model
    def select(self,stage,T):
        if stage not in self.routes:raise ProtocolError('Missing frozen replay route for '+stage)
        T=T.reset_index(drop=True);pipe=self.pipe
        needed={'K':['K'],'C':['K','C'],'W':['K','C','W']}[stage]
        key=digest(dict(model_key=pipe._key(stage,T,'MODEL'),role='REPLAY_SELECT_FIXED_ROUTE',
            routes={k:self.routes[k] for k in needed}))
        saved=pipe.checkpoints.load('REPLAY_SELECT_'+stage,key)
        if saved is not None:return saved['state']
        parent=None if stage=='K' else self.select(UP[stage],T)
        if stage!='K' and self.routes[stage]!='ADMITTED':
            model=pipe._unavailable_model(stage,parent,'DOMAIN_NOT_APPLICABLE',reason='GLOBAL_BYPASS_SKIP_SEARCH')
        elif not folds(T,pipe.protocol['geometry']['inner']):model=pipe._base_case(stage,parent)
        else:
            # tune() obtains its parent through intercepted learn(), which
            # selects a lawful MODEL on that inner prefix without re-voting.
            model=pipe.tune(stage,T)
        state=self._finish(stage,T,model)
        pipe.checkpoints.save('REPLAY_SELECT_'+stage,key,dict(state=state),pipe._references(state))
        return state
    def at_block(self,upstream,fit,*,entity,local_index,reselect,refit_source=None):
        pipe=self.pipe;chain=['K','C']+(['W'] if upstream=='W' else [])
        if reselect:
            state=self.select(upstream,fit)
            for node in chain:
                current=state
                while current['stage']!=node:current=current['parent']
                self.selected[entity,node]=current
                self.epoch[entity,node]=dict(entity=entity,local_block=local_index,
                    fit_hash=frame_hash(fit),selected_model_hash=current.get('MODEL_hash'),
                    recipe=recipe_binding(current))
        else:
            state=None
            for stage in chain:
                prior=self.selected.get((entity,stage))
                state=self.refit(stage,fit,state,prior,self.epoch.get((entity,stage)),refit_source=refit_source)
        self.events.append(dict(entity=entity,local_block=local_index,reselect=reselect,
            refit=True,fit_rows=len(fit),fit_hash=frame_hash(fit),routes=self.routes,
            upstream_realization_id=state['partition_model_id'],admission_vote_produced=False))
        return state
    def refit(self,stage,T,parent,selected,epoch,*,refit_source=None):
        pipe=self.pipe;T=T.reset_index(drop=True)
        needed={'K':['K'],'C':['K','C'],'W':['K','C','W']}[stage]
        key=digest(dict(role='REPLAY_PARAMETER_REFIT',stage=stage,frame=frame_hash(T),
            route={k:self.routes[k] for k in needed},epoch=epoch,recipe=recipe_binding(selected),
            parent_id=None if parent is None else parent['partition_model_id'],
            registries=pipe.registration['canonical_protocol_sha256'],
            authority=pipe.registration['authority_bindings_sha256_lf']))
        saved=pipe.checkpoints.load('REPLAY_REFIT_'+stage,key)
        if saved is not None:return saved['state']
        if stage!='K' and self.routes[stage]!='ADMITTED':
            model=pipe._unavailable_model(stage,parent,'DOMAIN_NOT_APPLICABLE',reason='GLOBAL_BYPASS_SKIP_SEARCH')
        elif not selected or not selected.get('domain'):
            model=pipe._unavailable_model(stage,parent,'SUPPORT_LIMITED',reason='NO_RECIPE_UNTIL_NEXT_REGISTERED_RESELECTION')
        elif stage=='K':model=refit_k(pipe,T,selected)
        elif stage=='C':model=refit_c(pipe,T,parent,selected)
        else:
            # The enclosing replay has already emitted every past C residual.
            # A refit-only block never recursively opens a historical selection.
            if refit_source is None:raise ProtocolError('W refit needs existing past-only C replay history')
            source=refit_source
            if set(source.sample_id)-set(T.sample_id):raise ProtocolError('W refit history includes current/future rows')
            ct,_,mask,audit=pipe.fit_residual_recipe('W',selected['recipe'],source,source,source)
            if ct is None:model=pipe._unavailable_model(stage,parent,audit.get('candidate_status','DOMAIN_NOT_APPLICABLE'))
            else:model=dict(domain=True,parent=parent,recipe=selected['recipe'],contract=ct,
                refit_source_hash=frame_hash(source),final_fit_rows=int(mask.sum()),uses_in_sample_parent_residual=False)
        state=self._finish(stage,T,model)
        state['reselection_epoch']=epoch;state['parameter_refit_only']=True
        state['partition_model_id']=digest({k:v for k,v in state.items() if k!='partition_model_id'})
        pipe.checkpoints.save('REPLAY_REFIT_'+stage,key,dict(state=state),pipe._references(state))
        return state

def recipe_binding(state):
    if not state:return None
    if state['stage']=='K':
        return dict(reference_channel=state.get('reference_channel'),representations=[
            {k:r[k] for k in ('channel','profile','m_tau','family','m_x')}|
            dict(lambda_tau=r['lambdas'][1],lambda_x=r['lambdas'][2]) for r in state.get('representations') or []])
    return dict(recipe=state.get('recipe'),alpha=state.get('selected_alpha'))

def refit_k(pipe,T,selected):
    reps=[];statuses={}
    for prior in selected['representations']:
        r=copy.deepcopy(prior);native=T.loc[pipe.native(T,r['profile'][1])]
        try:
            if len(native)<int(pipe.v2['row_caps']['minimum_rows_for_any_fitted_candidate']):raise SupportLimited('K prefix native rows below minimum')
            x,intervals=pipe._values(native,r)
            if not np.any(np.ptp(x,axis=0)>0):raise DomainNotApplicable('K prefix input domain absent')
            numerical_key=digest(dict(role='K_FIXED_RECIPE_NUMERICAL_REFIT',frame=frame_hash(native),
                recipe={k:r[k] for k in ('channel','profile','m_tau','family','m_x')},
                penalties=r['lambdas'][1:],policy=pipe.policy_id,
                source=pipe.registration['authority_bindings_sha256_lf']))
            saved=pipe.checkpoints.load('K_PARAMETER_CONTRACT',numerical_key)
            if saved is not None:
                r.update(contract=saved['contract'],intervals=intervals,lambdas=saved['lambdas'],
                    native_fit_support_hash=pipe.partition_id(native),numerical_ridge_audit=saved['stability'])
                reps.append(r);statuses[r['channel']]='AVAILABLE';continue
            # Numerical lambda_0 is a stability parameter, not MODEL reselection.
            ct=None;stability=[]
            for l0 in pipe.v2['K_module']['penalties']['lambda_0']:
                from .feature_cache import prepare
                width=1 if r['family']=='LINEAR_DISTRIBUTED_LAG' else r['m_x']
                prep=prepare(pipe,channel_name=r['channel'],profile=tuple(r['profile']),mt=r['m_tau'],
                    fit=native,values=x,width=width,family=r['family'])
                from .fit_contract_cache import fit_fixed_recipe
                attempt=fit_fixed_recipe(pipe,prep,r['family'],r['m_x'],
                    (float(l0),*r['lambdas'][1:]),**pipe.v2_k._als_kwargs(pipe.v2))
                require_k_support(attempt,len(native),pipe.v2)
                valid=pipe.k._candidate_valid(attempt,len(native),pipe.v2)
                stability.append(dict(lambda_0=float(l0),valid=bool(valid)))
                if valid:ct=attempt;break
            if ct is None:raise NumericallyInvalid('Fixed K recipe parameter refit failed all numerical lambdas')
            pipe.checkpoints.save('K_PARAMETER_CONTRACT',numerical_key,
                dict(contract=ct,lambdas=[float(l0),*r['lambdas'][1:]],stability=stability))
            r.update(contract=ct,intervals=intervals,lambdas=[float(l0),*r['lambdas'][1:]],
                native_fit_support_hash=pipe.partition_id(native),numerical_ridge_audit=stability)
            reps.append(r);statuses[r['channel']]='AVAILABLE'
        except (SupportLimited,DomainNotApplicable) as exc:
            statuses[r['channel']]=type(exc).__name__
    channel=selected.get('reference_channel')
    # A missing fixed reference realizes identity; it is never reselected here.
    reference=next((i for i,r in enumerate(reps) if r['channel']==channel),None)
    return dict(stage='K',parent=None,domain=reference is not None,representations=reps,
        reference_channel=channel,reference_index=reference,channel_status=statuses,
        recipe=selected.get('recipe'),representation_preserved=True)

def refit_c(pipe,T,parent,selected):
    if len(parent.get('representations') or [])<2:
        return pipe._unavailable_model('C',parent,'DOMAIN_NOT_APPLICABLE')
    dd,columns,mask,_=pipe._c_design(parent,T,T)
    if mask.sum()<2:return pipe._unavailable_model('C',parent,'SUPPORT_LIMITED')
    family=selected['recipe'];design=dd[family][0][mask]
    if not design.shape[1] or not np.any(np.ptp(design,axis=0)>0):
        return pipe._unavailable_model('C',parent,'DOMAIN_NOT_APPLICABLE')
    pipe.fit_calls+=1
    _,ct=pipe.c._ridge_fit(design,T.y_true.to_numpy(float)[mask]-pipe.emit(parent,T)[mask],design,selected['selected_alpha'])
    if not pipe.cert(ct):raise NumericallyInvalid('Fixed C recipe refit certificate failed')
    return dict(stage='C',parent=parent,domain=True,recipe=family,selected_alpha=selected['selected_alpha'],
        contract=ct,joint_columns=columns,fit_target='y_true-F_K')
