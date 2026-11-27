"""Causal timing checks for transfer to direct, zero-observation-delay CZ."""

import unittest

import numpy as np
from unittest.mock import patch

import laguerre_dual as algorithm
from laguerre_blind_cz import Direct, WINDOW, direct_rows, prepare
from laguerre_dual_views import readout
from laguerre_joint import LaguerreARX
from laguerre_screen import CONFIGS


class DirectTransferTest(unittest.TestCase):
    def test_calendar_and_sensor_only_cutoff(self):
        rng=np.random.default_rng(520)
        data=rng.normal(size=(760,4))
        h=16
        a=Direct.from_values(data,h)
        np.testing.assert_array_equal(a.anchor(),data[:,0])
        with patch.object(algorithm,"train_rows",direct_rows):
            folds=algorithm.late_folds(a.through(680),WINDOW)
        for stop,t in folds:
            selected=direct_rows(h,stop)
            self.assertLess(selected[-1]+h,t[0]-WINDOW)
            self.assertLess(t[-1]+h,680)
        f=prepare(a,540,12,dict(near=4,alpha=.018,admitted=True),CONFIGS["sru"][0])
        self.assertTrue(f.mask.all())
        j,_=readout(f,.003,.2)
        model=LaguerreARX(f.safe,f.mapping,j,True)
        clean=model.predict(a)["prediction"]
        edited=data.copy()
        edited[650:]+=200.
        changed=model.predict(Direct.from_values(edited,h))["prediction"]
        np.testing.assert_array_equal(changed[:650],clean[:650])
        live=model.stream()
        replay=np.array([live.step(x,y) for x,y in zip(a.x,a.lab)])
        np.testing.assert_allclose(replay[:,0],clean,atol=2e-12,rtol=0)


if __name__=="__main__":
    unittest.main()

