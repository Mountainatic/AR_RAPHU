# Appended to the repaired model in isolated E experiment copies.
_prior_design=design
_prior_configs=configs

def e_intervals(history):
    edges=[0,1,2,4,8,16,32,64,128]
    edges=[v for v in edges if v<=history]
    assert edges[-1]==history
    return list(zip(edges[:-1],edges[1:]))

def design(D,V,config,recipe=None):
    kind=config['kind']
    if kind not in ('LINEAR','URYSOHN'):return _prior_design(D,V,config,recipe)
    assert L==E_HISTORY and config['memory']
    own,u=raw(D,V)
    blocks=np.stack([u[:,lo:hi].mean(1) for lo,hi in e_intervals(L)],axis=1)
    states=own[:,[0,L-1]] if config.get('own',True) else np.empty((len(V),0))
    if kind=='LINEAR':return np.c_[states,blocks.reshape(len(V),-1)],None
    if recipe is None:
        recipe=dict(spline=SplineTransformer(n_knots=3,degree=1,include_bias=False,extrapolation='linear').fit(blocks.reshape(-1,blocks.shape[-1])))
    basis=recipe['spline'].transform(blocks.reshape(-1,blocks.shape[-1])).reshape(len(V),len(e_intervals(L)),-1)
    return np.c_[states,basis.reshape(len(V),-1)],recipe

def configs(variant,family):
    return [c for c in _prior_configs(variant,family) if c['kind'] not in ('LINEAR','URYSOHN') or c.get('memory')]
