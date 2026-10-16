"""Label-free conservative call DAG with symbolic frozen routes and recipes."""
from collections import Counter
import numpy as np
from .invariants import digest
from .partition import ordered_ids
from .fold_geometry import folds,selection_blocks
from .replay_policy import cadence

UP={'C':'K','W':'C','A':'W'}

def plan(frame,protocol,*,bindings):
    if any(c in frame for c in ('y_true','residual')):raise ValueError('DAG planning is metadata-only')
    pending=[('LEARN','A',np.arange(len(frame)),None,None)];seen=set();nodes=[];requested=0
    while pending:
        role,stage,pos,epoch,score=pending.pop();requested+=1
        T=frame.iloc[pos].reset_index(drop=True)
        identity=dict(stage=stage,role=role,ordered_fit_ids=digest(ordered_ids(T)),
            entity_structure=[(str(e),len(g)) for e,g in T.groupby('entity_id',sort=False)],
            policy=protocol['revision'],support=protocol['geometry']['purge'],bindings_sha256=digest(bindings),
            upstream_route='SYMBOLIC_FROZEN_ROUTE_VECTOR' if role.startswith('REPLAY') else 'DETERMINISTIC_NESTED_ADMISSION',
            epoch=epoch,selected_recipe='SYMBOLIC_EPOCH_RECIPE' if role=='REPLAY_REFIT' else None,
            score_ids=None if score is None else ordered_ids(frame.iloc[score]))
        key=digest(identity)
        if key in seen:continue
        seen.add(key)
        geometry=folds(T,protocol['geometry']['outer' if role=='LEARN' else 'inner'])
        eligible=bool(geometry) if role in {'LEARN','MODEL','REPLAY_SELECT'} else len(T)>0
        nodes.append(dict(id=key,role=role,stage=stage,fit_rows=len(T),fit_ids_sha256=identity['ordered_fit_ids'],
            entity_structure=identity['entity_structure'],epoch=epoch,status='AVAILABLE' if eligible else 'SUPPORT_LIMITED',
            identity=identity))
        def add(r,s,local,ep=None,sc=None):pending.append((r,s,pos[local],ep,None if sc is None else pos[sc]))
        if role=='REPLAY_SELECT' and stage=='K':
            add('MODEL','K',np.arange(len(T)));continue
        up=UP.get(stage)
        if role=='REPLAY_REFIT':
            if up:add(role,up,np.arange(len(T)),epoch)
            continue
        if up:add('LEARN' if role in {'LEARN','MODEL'} else 'REPLAY_SELECT',up,np.arange(len(T)))
        if not geometry:continue
        if role=='LEARN':
            add('MODEL',stage,np.arange(len(T)))
            for fit,_ in geometry:add('MODEL',stage,fit)
        elif stage=='C':
            for fit,_ in geometry:add('LEARN' if role=='MODEL' else 'REPLAY_SELECT','K',fit)
        elif stage in {'W','A'}:
            for fit,*_ in selection_blocks(T,protocol['geometry']['inner'],protocol['geometry']['selection_cf_context_rows_per_block']):
                add('LEARN' if role=='MODEL' else 'REPLAY_SELECT',up,fit)
            epochs={}
            for index,fit,ev,entity,local,reselect in cadence(T):
                if reselect:epochs[entity]=dict(entity=entity,local_block=local,fit_ids_sha256=digest(ordered_ids(T.iloc[fit])))
                add('REPLAY_SELECT' if reselect else 'REPLAY_REFIT',up,fit,None if reselect else epochs[entity])
                # Prediction blocks include their distinct score metadata.
                identity_block=dict(role='REPLAY_BLOCK',stage=up,fit_ids=digest(ordered_ids(T.iloc[fit])),
                    score_ids=digest(ordered_ids(T.iloc[ev])),epoch=epochs[entity],bindings_sha256=digest(bindings),policy=protocol['revision'])
                bk=digest(identity_block);requested+=1
                if bk not in seen:
                    seen.add(bk);nodes.append(dict(id=bk,role='REPLAY_BLOCK',stage=up,fit_rows=len(fit),
                        fit_ids_sha256=identity_block['fit_ids'],score_rows=len(ev),epoch=epochs[entity],status='AVAILABLE',identity=identity_block))
    counts=Counter(n['role']+'_'+n['stage'] for n in nodes)
    classes=Counter(n['role']+'_'+n['stage']+'_'+str((n['fit_rows']//1000)*1000) for n in nodes)
    availability=Counter(n['role']+'_'+n['stage']+'_'+n['status'] for n in nodes)
    return dict(schema='CZ_ACCEL12H_METADATA_DAG_V1',status='COMPLETED',metadata_only=True,target_bytes_accessed=False,
        requested_nodes=requested,unique_nodes=len(nodes),dedup_ratio=1-len(nodes)/max(requested,1),
        bindings=bindings,bindings_sha256=digest(bindings),nodes_by_stage=dict(counts),
        nodes_by_partition_size=dict(classes),nodes_by_metadata_status=dict(availability),estimated_cache_reuse=requested-len(nodes),
        route_enumeration='SYMBOLIC_FULLY_ADMITTED_WORST_PATH; RUNTIME_BINDS_EXACT_VECTOR',
        symbolic_recipe_binding='EXACT_SELECTED_RECIPE_RESOLVED_AT_PAST_ONLY_EPOCH',
        route_state_safety_multiplier=8,all_route_vectors_superset=['000','001','010','011','100','101','110','111'],nodes=nodes)

