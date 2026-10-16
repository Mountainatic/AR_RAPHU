"""Materialize the unchanged effective authority grids before release freeze."""
import json,sys
from pathlib import Path
from .invariants import digest,dump,source_sha,ProtocolError

CHANNELS=['main_heater_power','joint_lift','crystal_rotation','crucible_rotation']

def materialize(authority,destination):
    authority=Path(authority);destination=Path(destination)
    sys.path.insert(0,str(authority/'src'))
    from prism_benchmark import v211_k,v211_w,v2_k
    from prism_benchmark.v211_config import load_v211_configs
    from prism_benchmark.cz_l256_nowcast import view
    v211,v21,v2=load_v211_configs(authority,protocol='cz_h1_k_temporal_registry_v2')
    task=view('dynamic',h_steps=1,task_id='CZ_DIAM_RAW2S_CURRENT_L256_H1')
    profiles=[v2_k.channel_profiles(task,ch,v2) for ch in CHANNELS]
    if any(p!=profiles[0] for p in profiles):raise ProtocolError('Channel registry drift')
    k=v211_k._structural_specs(profiles[0],list(v21['K_C']['m_tau']),v2['K_module']['penalties']['lambda_tau'],v21['K_C']['m_x'],v2['K_module']['penalties']['lambda_x'])
    c=[(family,float(alpha)) for family in ('ADDITIVE_COMPRESSED','ADDITIVE_JOINT_BASIS') for alpha in v2['C_module']['joint_basis']['ridge_alpha_grid']]
    w=v211_w._w_candidates(v211,v21,v2,1,False)
    profiles_a=json.loads((destination/'A_TEMPORAL_REGISTRY.json').read_text())['profiles_in_registered_order']
    a=[(profile,float(alpha),float(mu)) for profile in profiles_a for alpha in v2['A_module']['ridge_alpha_grid'] for mu in v21['A']['soft_overlap_mu']]
    grids={'K_MODEL_REGISTRY.json':dict(candidates=k,channels=CHANNELS,profiles=profiles[0],m_tau=v21['K_C']['m_tau'],
        support_role='CANDIDATE_NATIVE_FIT_EXACT_REGISTERED_COMMON_SCORE',numerical_only_parameters={'lambda_0':v2['K_module']['penalties']['lambda_0']},
        predictive_parameters=['profile','m_tau','family','m_x','lambda_tau','lambda_x']),
        'C_MODEL_REGISTRY.json':dict(candidates=c,families=['ADDITIVE_COMPRESSED','ADDITIVE_JOINT_BASIS'],
        alpha_grid=v2['C_module']['joint_basis']['ridge_alpha_grid'],fit_target='y-F_K',
        support_role='FROZEN_LEGAL_K_ASSEMBLY_EXACT_COMMON_SCORE',predictive_parameters=['family','alpha'],numerical_only_parameters=[]),
        'W_MODEL_REGISTRY.json':dict(candidates=w,families=list(dict.fromkeys(s[0] for s in w)),
        support_role='AVAILABLE_FROZEN_CURRENT_Q_EXACT_COMMON_SCORE',predictive_parameters=['family','knots','smoothness','mu','direction'],numerical_only_parameters=[]),
        'A_MODEL_REGISTRY.json':dict(candidates=a,families=['MATURE_RESIDUAL_AR'],profiles=profiles_a,
        support_role='CANDIDATE_NATIVE_MATURE_FIT_REGISTERED_COMPLETE_HISTORY_COMMON_SCORE',predictive_parameters=['profile','alpha','mu'],numerical_only_parameters=[])}
    source={p.relative_to(authority).as_posix():source_sha(p) for p in (authority/'configs/cz_h1_k_temporal_registry_v2').glob('*.json')}
    for name,body in grids.items():
        body.update(schema='CZ_EFFECTIVE_MODEL_REGISTRY_V1',registration_order='EXACT_CANDIDATES_LIST_ORDER',source_authority_binding=source,identity_is_candidate=False)
        body['payload_sha256']=digest(body)
        p=destination/name
        if p.exists():
            if json.loads(p.read_text())!=json.loads(json.dumps(body)):raise ProtocolError('Effective registry is immutable')
        else:dump(p,body)
    return grids

def load_registry(path):
    body=json.loads(Path(path).read_text())
    if digest({k:v for k,v in body.items() if k!='payload_sha256'})!=body['payload_sha256']:raise ProtocolError('Registry payload SHA mismatch')
    def frozen(v):return tuple(frozen(x) for x in v) if isinstance(v,list) else v
    return body,[frozen(v) for v in body['candidates']]
