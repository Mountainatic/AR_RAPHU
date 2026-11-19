"""Load a compact head from legal process and mature-target history only."""
import pickle
from pathlib import Path
import numpy as np
from streaming_balance import StreamingPRISM


def load_predictor(path, process_history, mature_target_history, first_origin):
    """Histories are oldest to newest, each exactly L+63 rows.

    Process history ends at first_origin-1. Mature target history ends at
    first_origin-delay-1. Each subsequent step receives X(t) and y(t-delay).
    Use only trusted model pickle files.
    """
    cfg=pickle.loads(Path(path).read_bytes());obj=cfg['model'];L=cfg['L'];d=cfg['delay']
    x=np.asarray(process_history,dtype=np.float64);y=np.asarray(mature_target_history,dtype=np.float64)
    if x.ndim!=2 or y.ndim!=1 or len(x)!=L+63 or len(y)!=L+63:
        raise ValueError('Both histories must have exactly L+63 ordered observations')
    if not np.isfinite(x).all() or not np.isfinite(y).all():raise ValueError('Nonfinite history')
    if first_origin-64<=cfg['parent_fit_last_origin']:
        raise ValueError('Upstream head was not frozen before the history issues')
    args=dict(temporal_bins=cfg['temporal_bins'],delta_target=cfg['delta_target'],target_history_mode=cfg['target_history_mode'])
    # Only the already frozen upper model issues historical q; final A is never
    # used retroactively before its initializer labels become available.
    upper=StreamingPRISM(dict(upper=obj['upper'],A=None),L,d,x[:L-1],y[:L-1],first_origin-64,**args)
    q={i-64:upper.upper(x[L-1+i],y[L-1+i]) for i in range(64)}
    residual={i:float(y[i+d+len(y)]-qi) for i,qi in q.items() if i < -d}
    return StreamingPRISM(obj,L,d,x[-L+1:],y[-L+1:],first_origin,
                          issued_q_past=q,mature_residual_past=residual,**args)
