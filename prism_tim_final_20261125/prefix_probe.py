"""Compare native process-window means with independent long-double sums."""

import ctypes
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import core as c
from e4 import CORE, LONG


def stress():
    os.sched_setaffinity(0,{CORE})
    path=c.ROOT/"E4"
    c.x.bind(path/"PREFIX_PROBE_REGISTRATION.json",dict(code_lock=c.verify_code(),rows=LONG,
        sources={name:c.sha(c.ROOT/name) for name in ("prefix_probe.py","prefix_probe.c")},
        reference="Independent C long-double sum of transformed raw process values, including delayed subtraction",
        timestamps="Every 1024 steps, and at the first step after the production reanchor",
        tolerance=1e-7))
    rows=[]
    for name in c.MAIN:
        q,k,fit,chain,_=c.load(name)
        a=c.np.ascontiguousarray(c.data(q)[:4096])
        with TemporaryDirectory(prefix="prism-native-prefix-") as p:
            for period in (c.REANCHOR,0):
                work=Path(p)/str(period)
                *_,flags=c.compile_kernel(q,k,fit,chain,work,period=period)
                result=work/"probe.so"
                subprocess.run(["gcc",*flags,"-I",str(work),str(c.ROOT/"prefix_probe.c"),"-o",str(result)],check=True)
                probe=ctypes.CDLL(str(result)).probe
                fp=c.p.native.legacy.F64
                probe.argtypes=(fp,ctypes.c_size_t,ctypes.c_size_t,fp)
                probe.restype=None
                maxima=c.np.zeros(32)
                probe(a,len(a),LONG,maxima)
                if period:
                    assert c.np.max(maxima)<1e-7
                rows.append(dict(data=name,period=period,rows=LONG,max_raw_mean_absolute_error=float(maxima.max())))
    c.csv_table(path/"PREFIX_PROBE.csv",rows)
    c.write(path/"PREFIX_PROBE.json",rows)


if __name__=="__main__":
    stress()
