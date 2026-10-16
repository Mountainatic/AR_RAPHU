"""Equivalent mature A feature gathering, without growing pandas histories."""
import json
from pathlib import Path
import numpy as np
from canonical.cz_crossrod_l128_20261008.stagewise_engine import StagewisePipeline
from canonical.cz_crossrod_l128_20261008.invariants import ProtocolError,digest,file_sha,prediction_hash
from canonical.cz_crossrod_l128_20261008.partition import validate_partition
from canonical.cz_crossrod_l128_20261008.a_state import offsets,INFERENCE_DEPLOYED,cz_target_end
ROOT=Path(__file__).resolve().parent

class FastInferencePipeline(StagewisePipeline):
    def __init__(self,context,root,*,mode):
        if mode!='inference_only':raise ProtocolError('Fast inference adapter cannot fit')
        super().__init__(context,root,mode=mode)
        seal=ROOT/'INFERENCE_EXTENSION_SEAL.json'
        self.extension_pin=file_sha(seal)
        frozen=json.loads(seal.read_text())
        for p,h in frozen['file_sha256'].items():
            if file_sha(p)!=h:raise ProtocolError('Inference extension source changed')
        if frozen['main_release_sha256']!=context['release_sha']:raise ProtocolError('Inference extension belongs to another model release')

    def online_a(self,model,ev):
        self.assert_inference_only();validate_partition(ev)
        upstream=model['parent'];model_id=upstream.get('deployment_model_id',upstream.get('partition_model_id'))
        baseline=self.emit(upstream,ev.drop(columns=['y_true']));output=baseline.copy();native=np.zeros(len(ev),bool)
        ledger=[];features=[]
        lags=offsets(model['profile'],self.A_max) if model.get('domain') else np.empty(0,dtype=int)
        for entity,g in ev.groupby('entity_id',sort=False):
            indices=ev.index.get_indexer(g.index);origins=g.origin.to_numpy(np.int64)
            latest=g.latest_available_target_index.to_numpy(np.int64);ends=cz_target_end(g,1,1)
            if np.any(np.diff(origins)<=0) or np.any(latest>=ends):raise ProtocolError('Prediction issue/maturity contract changed')
            for row,position in enumerate(indices):
                mature_count=min(row,int(np.searchsorted(ends,latest[row],side='right')))
                feature_hash=digest('NA_IDENTITY');used=[]
                if len(lags):
                    query=latest[row]-lags;source=np.searchsorted(origins,query)
                    safe=np.minimum(source,max(0,len(origins)-1))
                    mask=(source<len(origins))&(origins[safe]==query)&(source<row)&(ends[safe]<=latest[row])
                    if np.all(mask):
                        # Read ONLY previously issued targets that are mature
                        # now. No future residual vector is precomputed.
                        true=g.iloc[source].y_true.to_numpy(float)
                        x=((true-baseline[indices[source]])-float(model['residual_mean'])).reshape(1,-1)
                        correction=self.a.predict_mature_residual_ar(x,model['contract'])[0]
                        output[position]=baseline[position]+correction;native[position]=True
                        used=origins[source].tolist();feature_hash=prediction_hash(x)
                features.append(feature_hash)
                ledger.append(dict(entity=str(entity),origin=int(origins[row]),issue_latest=int(latest[row]),
                    mature_rows=mature_count,used_residual_origins=used,feature_hash=feature_hash,
                    child_hash=prediction_hash(output[position:position+1])))
        proof=dict(state_role=INFERENCE_DEPLOYED,upstream_deployment_model_id=model_id,
            upstream_model_scope='DEPLOYMENT' if 'deployment_model_id' in upstream else 'PARTITION',
            mature_state_hash=digest(ledger),mature_feature_hash=digest(features),
            initialization='EMPTY_PER_SCORING_BLOCK_OR_ENTITY',pre_score_warmup_used=False,training_backcast_used=False,
            prediction_issue_before_target_maturity=True,state_crosses_entity=False,issue_ledger_hash=digest(ledger),
            unavailable_A_rows_exact_parent_identity=True,feature_gathering='EXACT_MATURE_ORIGIN_QUERY_NO_GROWING_HISTORY',
            inference_extension_sha256=self.extension_pin)
        return output,native,proof
