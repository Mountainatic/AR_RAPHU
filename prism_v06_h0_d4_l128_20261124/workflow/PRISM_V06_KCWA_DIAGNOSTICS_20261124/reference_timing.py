"""Measure the historical Python one-row replay on increasing score blocks."""

import os
import time

from ablation import ROOT, h, np, old, prepare, write


def run():
    cores = os.sched_getaffinity(0)
    core = 12 if 12 in cores else min(cores)
    os.sched_setaffinity(0,{core})
    entity = "SRU_H2S"
    geo, pipe, data, _, model = prepare(entity)
    repetitions = 3
    start = geo["development"][0]
    rows = []
    for n in (128,256,512,1024):
        f = h.frame(entity,start,start+n,"development",y=data[:,0])
        anchor = data[f.origin.to_numpy()-h.D,0]
        expected, _, ready = old.prediction(pipe,model,f,anchor)
        samples = []
        for _ in range(repetitions):
            before = time.perf_counter_ns()
            observed, native, _ = pipe.online_a(model,f)
            samples.append((time.perf_counter_ns()-before)/1e6)
            np.testing.assert_array_equal(ready,native)
            np.testing.assert_allclose(expected,observed+anchor,atol=1e-10,rtol=1e-12)
        rows.append(dict(N=n, median_ms=float(np.median(samples)), samples_ms=samples))
        print("ORIGINAL_PYTHON",n,rows[-1]["median_ms"],flush=True)
    write(ROOT/"REFERENCE_COMPLEXITY.json",dict(entity=entity,core=core,L=h.L,repeats=repetitions,
          scope="October StagewisePipeline.online_a with incoming warm history for process inputs and cold-start residual state.",rows=rows))


if __name__ == "__main__":
    run()
