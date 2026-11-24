"""Original K reference assembly, collected in frozen channel order."""
import numpy as np
def assemble(self,T,destination,outputs):
    from .canonical_engine import CHANNELS
    from .fold_geometry import folds
    from .invariants import NumericallyInvalid
    from .candidate_status import SupportLimited,no_available_model
    from .risk import aggregate,registered_best,equivalence_bound,mse_evidence
    from .partition import ordered_ids,support_hash
    geometry=folds(T,self.protocol['geometry']['inner']);reps=[];statuses={};selected_folds={}
    for output in outputs:
        reps.extend(output['reps']);statuses.update(output['statuses']);selected_folds.update(output['selected_folds'])
    if not reps:
        reason=no_available_model(statuses.values());return self._unavailable_model('K',None,reason,representations=[],channel_status=statuses)
    reference_risks={};assembly=[]
    for j,(_,ei) in enumerate(geometry):
        ev=T.iloc[ei];mask=np.ones(len(ev),bool)
        for r in reps:mask &= self.native(ev,r['profile'][1])
        if not mask.any():raise SupportLimited('Selected K assembly has no common scoring support')
        ids=ordered_ids(ev.loc[mask]);assembly.append(dict(fold=j,rows=len(ids),ordered_score_ids=ids,ordered_score_support_hash=support_hash(ev.loc[mask])))
        for r in reps:
            xx,_=self._values(ev.loc[mask],r);ct=selected_folds[r['channel']][j]['contract']
            p=self.u.predict_contract(xx,ct)
            reference_risks.setdefault(r['channel'],[]).append(mse_evidence(ev.y_true.to_numpy(float)[mask],p,ids))
    risks={channel:aggregate(values) for channel,values in reference_risks.items()};channel=registered_best(risks)
    audit=dict(selected_representation_history_by_channel={r['channel']:r['profile'][1] for r in reps},
        assembly_support_by_fold=assembly,reference_risk_by_channel={c:r.to_json() for c,r in risks.items()},reference_channel=channel,
        reference_tie_bound={c:equivalence_bound(r,risks[channel]) for c,r in risks.items()},outer_held_out_used=False)
    self._save(destination/'K_REFERENCE_ASSEMBLY_AUDIT.json',audit)
    return dict(domain=True,representations=reps,reference_index=next(i for i,r in enumerate(reps) if r['channel']==channel),
        reference_channel=channel,channel_status=statuses,assembly_risks=[r.to_json() for r in risks.values()],reference_assembly_audit=audit,recipe='REGISTERED_K_REPRESENTATIONS')
