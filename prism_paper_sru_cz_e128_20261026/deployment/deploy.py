"""Load an exported head. Caller supplies only lawful history and mature labels."""
import pickle
from pathlib import Path
from streaming import StreamingPRISM

def load_predictor(path,x_past,y_mature_past,first_origin):
    with Path(path).open('rb') as f:bundle=pickle.load(f)
    assert bundle['schema']=='PRISM_EXPLORATORY_DEPLOYMENT_20261026'
    return StreamingPRISM(bundle['model'],bundle['L'],bundle['delay'],x_past,y_mature_past,first_origin,bundle['temporal_bins'])
