"""Replay original frozen PRISM and no_A at the full-PRISM selected window."""
from pathlib import Path
import os,sys,json,hashlib,pickle
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'prism'))
import model as m
WINDOWS={4:8,8:16,16:16}
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def run(delay):
    r=read(ROOT/'REGISTRATION.json');gate=read(ROOT/'ALL_FITS_FROZEN.json')
    assert gate['all_fits_frozen_before_evaluation'] and all(sha(ROOT/k)==v for k,v in gate['receipts'].items())
    assert all(sha(ROOT/k)==v for k,v in r['code_sha256'].items()) and all(sha(ROOT/k)==v for k,v in r['reference_sha256'].items())
    a=np.vstack([np.load(ROOT/'data/train.npz')['values'],np.load(ROOT/'data/development.npz')['values']]);assert a.shape==(8064,6)
    m.L=WINDOWS[delay];m.configure(delay);m.SUPPORT=127+delay;m.old.L=m.SUPPORT;m.mature_a.configure(delay,m.SUPPORT)
    case=f'SRU_H2S_D{delay}';D={case:a};V=m.frame(D,6048,8064);before=m.FITS
    f=pd.DataFrame(dict(origin=V.origin,y_true=V.y,label_delay=delay));audits=[]
    dest=ROOT/'prism_reference'/f'D{delay}';receipt=read(dest/'FROZEN.json');assert receipt['window']==WINDOWS[delay]
    assert all(sha(dest/k)==v for k,v in receipt['files'].items())
    prior=pd.read_parquet(ROOT/'previous'/f'prism_D{delay}.parquet')
    for variant,col in [('full','PRISM'),('no_A','PRISM_no_A')]:
        obj=pickle.loads((dest/f'{variant}.pkl').read_bytes());q=m.predict_kcw(D,V,obj['upper']);stream=m.source(V,q,'matched_window_replay')
        correction=np.zeros(len(V)) if obj['A'] is None else m.a_predict(stream,obj['A'])[0];p=q+correction;f[col]=p
        for n in [1,delay,delay+1,33,65,257,2016]:
            cut=6048+n-1;poison={case:a.copy()};poison[case][cut-delay+1:,0]+=1234;poison[case][cut+1:,1:]-=1234
            W=m.frame(poison,6048,cut+1);qq=m.predict_kcw(poison,W,obj['upper']);s=m.source(W,qq,'matched_window_replay')
            aa=np.zeros(n) if obj['A'] is None else m.a_predict(s,obj['A'])[0]
            np.testing.assert_allclose(qq+aa,p[:n],rtol=1e-12,atol=1e-12)
        if variant=='full' or delay!=4:np.testing.assert_allclose(p,prior[col],rtol=1e-12,atol=1e-12)
        audits.append(dict(variant=variant,direct_window=WINDOWS[delay],poison_prefix_checks=True,previous_selected_prediction_equal=variant=='full' or delay!=4))
    assert before==m.FITS
    f.to_parquet(ROOT/'data'/f'prism_D{delay}.parquet',index=False)
    out=ROOT/'audits';out.mkdir(exist_ok=True)
    (out/f'PRISM_D{delay}_CAUSAL_REPLAY.json').write_text(json.dumps(dict(status='COMPLETED',checks=audits,new_fits=0,reserved_tail_read=False),indent=2))
    print('PRISM_MATCHED_REPLAYED',delay,WINDOWS[delay],flush=True)
if __name__=='__main__':run(int(sys.argv[1]))
