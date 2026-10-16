"""K channel extraction preserving the accepted registration/solver order."""
import numpy as np
from .model_tuning import candidate
from .invariants import NumericallyInvalid,ProtocolError
from .candidate_status import SupportLimited,DomainNotApplicable,classify_exception,require_k_support,no_available_model
from .risk import RiskEvidence,mse_evidence,aggregate,registered_best
from .partition import ordered_ids,support_hash
from .fold_evidence_cache import fold_binding

def _compute_fold(self,prepared,profile,mt,j,fit,ev,mask,x,z,family,mx,lt,lx):
    width=1 if family=='LINEAR_DISTRIBUTED_LAG' else mx;pk=(profile,mt,j,width)
    if pk not in prepared:prepared[pk]=self.u.prepare_contract_fit(x,fit.y_true.to_numpy(float),width)
    prep=prepared[pk];dims=prep.phi.shape[1:];rank=int(family.split('_')[1]) if family.startswith('RANK_') else None
    count=1+(rank*sum(dims) if rank is not None else int(np.prod(dims)))
    require_k_support(dict(parameter_count=count,family=family),len(fit),self.v2)
    self.fit_calls+=1
    ct=self.u.fit_prepared_contract(prep,family,mx,(float(self.v2['K_module']['penalties']['pilot']['lambda_0']),lt,lx),**self.v2_k._als_kwargs(self.v2))
    require_k_support(ct,len(fit),self.v2)
    if not self.k._candidate_valid(ct,len(fit),self.v2):raise NumericallyInvalid('K numerical certificate failed')
    p=self.u.predict_contract(z,ct)
    return mse_evidence(ev.y_true.to_numpy(float)[mask],p,ordered_ids(ev.loc[mask]))

def channel(self,T,destination,channel,fold_cache=None):
    if self.mode!='accelerated_benchmark':raise ProtocolError('Exhaustive is reporting-only')
    from .fold_geometry import folds
    geometry=folds(T,self.protocol['geometry']['inner']);reps=[];statuses={};selected_folds={};selections={}
    partition=self._key('K',T,'MODEL');group_bindings={}
    for channel in (channel,):
        profiles,mts,maximum=self.k_grid(channel)
        specs=list(self.K_specs)
        if self.mode=='solver_smoke':specs=[s for s in specs if s[0][0]==2 and s[1]==4 and s[2]=='LINEAR_DISTRIBUTED_LAG' and s[4]==float(self.v2['K_module']['penalties']['lambda_tau'][0])]
        if not self.accessor.has_columns([channel]):statuses[channel]='DOMAIN_NOT_APPLICABLE';continue
        fold_data={};prepared={};risks={};ledger=[]
        common_masks=[self.native(T.iloc[ei],maximum) for _,ei in geometry]
        supports=[dict(fold=j,ordered_score_ids=ordered_ids(T.iloc[ei].loc[mask]),ordered_score_support_hash=support_hash(T.iloc[ei].loc[mask]),rows=int(mask.sum())) for j,((_,ei),mask) in enumerate(zip(geometry,common_masks))]
        def records(group):
            if group not in fold_data:
                profile,mt=group;data=[]
                for (fi,ei),mask in zip(geometry,common_masks):
                    if len(fi)<int(self.v2['row_caps']['minimum_rows_for_any_fitted_candidate']):raise SupportLimited('K empty/insufficient fold-training partition')
                    fit=self.cap(T.iloc[fi].loc[self.native(T.iloc[fi],profile[1])],int(self.v2['row_caps']['single_channel_k_fit']))
                    if len(fit)<int(self.v2['row_caps']['minimum_rows_for_any_fitted_candidate']):raise SupportLimited('K native fit rows below minimum')
                    if not mask.any():raise SupportLimited('K common scoring support empty')
                    x,_=self.v2_k.profile_values(self.accessor,fit,channel,profile,mt)
                    if not np.any(np.ptp(x,axis=0)>0):raise DomainNotApplicable('K profile has no fit variation')
                    ev=T.iloc[ei];z,_=self.v2_k.profile_values(self.accessor,ev.loc[mask],channel,profile,mt)
                    data.append((fit,ev,mask,x,z))
                fold_data[group]=data
            return fold_data[group]
        for spec in specs:
            profile,mt,family,mx,lt,lx=spec
            def compute():
                evidence=[]
                for j,(fit,ev,mask,x,z) in enumerate(records((profile,mt))):
                    def compute_fold():
                        return _compute_fold(self,prepared,profile,mt,j,fit,ev,mask,x,z,family,mx,lt,lx)
                    if fold_cache is None:e=compute_fold()
                    else:
                        group=(profile,mt,j)
                        if group not in group_bindings:
                            group_bindings[group]=fold_binding(self,channel,profile,mt,fit,ev.loc[mask],x,z)
                        e=fold_cache.evaluate(group_bindings[group],spec,compute_fold)
                    evidence.append(e)
                return aggregate(evidence)
            result=candidate(self,'K',partition,(channel,)+spec,compute)
            # Family is corrected explicitly for the channel-prefixed checkpoint recipe.
            ledger.append(dict(spec=spec,status=result['status'],error=result.get('error')))
            if result['status']=='AVAILABLE':risks[spec]=RiskEvidence(**result['risk'])
        if not risks:
            statuses[channel]=no_available_model(r['status'] for r in ledger)
            self._save(destination/(channel+'_selection.json'),dict(candidate_status=ledger,common_support=supports));continue
        spec=registered_best(risks);profile,mt,family,mx,lt,lx=spec
        native=T.loc[self.native(T,profile[1])];x,intervals=self.v2_k.profile_values(self.accessor,native,channel,profile,mt)
        if not np.any(np.ptp(x,axis=0)>0):statuses[channel]='DOMAIN_NOT_APPLICABLE';continue
        data=records((profile,mt))
        def full(l0):
            self.fit_calls+=1;ct=self.u.fit_contract(x,native.y_true.to_numpy(float),family,mx,(l0,lt,lx),**self.v2_k._als_kwargs(self.v2))
            require_k_support(ct,len(native),self.v2);return ct
        def folded(l0):
            out=[]
            for j,(fit,ev,mask,xx,z) in enumerate(data):
                self.fit_calls+=1;ct=self.u.fit_contract(xx,fit.y_true.to_numpy(float),family,mx,(l0,lt,lx),**self.v2_k._als_kwargs(self.v2))
                require_k_support(ct,len(fit),self.v2)
                out.append(dict(fold=j,contract=ct,fit_rows=len(fit),domain=True))
            return out
        try:
            l0,ct,replayed,stability=self.k.select_smallest_stable_full_and_folds(self.v2['K_module']['penalties']['lambda_0'],full,folded,
                valid_full=lambda c:self.k._candidate_valid(c,len(native),self.v2),
                valid_fold=lambda r:self.k._candidate_valid(r['contract'],r['fit_rows'],self.v2),parallel_workers=1)
        except (RuntimeError,ValueError,np.linalg.LinAlgError) as exc:
            if isinstance(exc,(SupportLimited,DomainNotApplicable)):statuses[channel]=classify_exception(exc);continue
            raise NumericallyInvalid('Selected K numerical-only refit failed: '+str(exc)) from exc
        reps.append(dict(channel=channel,profile=list(profile),m_tau=mt,family=family,m_x=mx,lambdas=[l0,lt,lx],contract=ct,intervals=intervals,
            native_fit_support_hash=self.partition_id(native),numerical_ridge_audit=stability,representation_preserved=True))
        selected_folds[channel]=replayed;statuses[channel]='AVAILABLE';selections[channel]=spec
        self._save(destination/(channel+'_selection.json'),dict(selected_spec=spec,selected_risk=risks[spec].to_json(),
            candidate_count=len(specs),candidate_status=ledger,common_support=supports,registered_profiles=len(profiles),maximum_history_steps=maximum))
    return dict(reps=reps,statuses=statuses,selected_folds=selected_folds,selections=selections)
