"""Rank, embargo, and streaming regression checks for the dual-view readout."""

import unittest

import numpy as np

from laguerre_anchored import X_BANKS, cuts
from laguerre_dual import WINDOW, late_folds
from laguerre_dual_views import DualModel, MatureState, independent_taps, prepare, readout
from laguerre_ecm import Dynamics
from laguerre_screen import CONFIGS, Series, lag, train_rows


class DualViewTest(unittest.TestCase):
    def test_mature_increment_clock_and_level_center(self):
        y = np.arange(100, dtype=float)**2
        for horizon, delay in ((0, 1), (0, 8), (16, 0), (64, 0)):
            span = horizon + delay
            for near in (4, 8, 16):
                anchor = lag(y, delay)
                state = MatureState(near, span, delay, 20.)
                expected = np.zeros((len(y), near+1))
                expected[delay:, -1] = y[:len(y)-delay] - 20.
                for k in range(near):
                    first = delay + span + k
                    expected[first:, k] = y[span:len(y)-delay-k] - y[:len(y)-first]
                np.testing.assert_array_equal(state.batch(anchor), expected)
                stream = state.stream(())
                np.testing.assert_array_equal([stream.step(value) for value in anchor], expected)
                shifted = MatureState(near, span, delay, 120.)
                np.testing.assert_array_equal(shifted.batch(lag(y+100, delay)), expected)
                np.testing.assert_array_equal(state.batch(anchor[:3]), expected[:3])

    def test_exact_taps_span_original_lags_and_tail_selection_is_independent(self):
        for d in (1,2,8,15):
            for r in (4,8,12):
                a = Dynamics(r,X_BANKS).batch(np.r_[1.,np.zeros(550)])
                b = a-lag(a,d)
                mask = independent_taps(r,d)
                left = min(r,d)
                near = np.column_stack((a[:,:left],b[:,:r]))
                self.assertEqual(np.linalg.matrix_rank(near),left+r)
                if r>=d:
                    self.assertEqual(np.linalg.matrix_rank(near[:r+d]),r+d)
                selected = np.column_stack((a[:,mask[0]],b[:,mask[1]]))
                self.assertEqual(np.linalg.matrix_rank(selected,tol=1e-10),selected.shape[1])

    def test_nested_label_cutoffs_and_stream(self):
        rng=np.random.default_rng(915)
        x=rng.normal(size=(650,2))
        y=.18*x[:,0]+.12*lag(x[:,1],8)+.03*rng.standard_normal(len(x))
        d=8
        data=Series(x,y,y,d)
        for outer,_ in cuts(data)[2]:
            for stop,test in late_folds(data.through(outer),WINDOW):
                f=train_rows(d,stop)
                self.assertLess(f[-1],test[0]-WINDOW-d)
                self.assertLess(test[-1],outer-d)
        f=prepare(data,520,12,dict(near=4,alpha=.018,admitted=True),CONFIGS["sru"][0])
        head,_=readout(f,.003,.2,.04)
        model=DualModel(f.safe,f.mapping,head,d,True)
        unsafe=f.design.trust.copy()
        unsafe[:]=0
        fallback,_=head.mix(f.design.raw,unsafe,f.state,f.safe_hat)
        np.testing.assert_array_equal(fallback,f.safe_hat)
        online=model.stream(x.shape[1])
        expected=model.predict(data)
        reading=np.array([online.step(v,a) for v,a in zip(x,lag(y,d))])
        np.testing.assert_allclose(reading,np.c_[expected["prediction"],expected["endo"],expected["trust"]],
                                   atol=2e-12,rtol=0)


if __name__ == "__main__":
    unittest.main()

