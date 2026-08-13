import json
from pathlib import Path

import numpy as np

from experiments.neurobem_sampling_rate_scaling.path_excursion import (
    PathReplay, excursion_metrics, inside_envelope, transient_order_class,
)


ROOT = Path(__file__).parents[1]


def fixture():
    return PathReplay({"full_state": np.array([.5, 2., .5]), "velocity": np.array([.5, 2., .5]),
                       "attitude": np.zeros(3), "body_rate": np.zeros(3)}, np.ones(3,dtype=bool))


def test_frozen_channel_envelope_is_used_without_scalar_redefinition():
    path=fixture(); bounds={"velocity":1.,"attitude":1.,"body_rate":1.}
    assert inside_envelope(path,bounds,"full_state").tolist()==[True,False,True]


def test_crossing_reentry_and_path_metrics():
    value=excursion_metrics(fixture(),{"velocity":1.,"attitude":1.,"body_rate":1.},"velocity",3,100)
    assert value["terminal_error"]==.5 and value["max_path_error"]==2.
    assert value["envelope_crossing_count"]==1 and value["reentry_count"]==1
    assert value["first_cross_ms"]==20.


def test_registered_order_class():
    assert transient_order_class({100:1.,200:2.,400:3.}).endswith("SUPPORTED")
    assert transient_order_class({100:3.,200:2.,400:1.}).endswith("NOT_SUPPORTED")


def test_protocol_is_deterministic_replay_not_new_test_decision():
    cfg=json.loads((ROOT/"configs"/"path_excursion.yaml").read_text())
    assert cfg["execution_mode"]=="DETERMINISTIC_REPRODUCTION_REPLAY_EXISTING_FORMAL_TEST"
    source=(ROOT/"run_path_excursion.py").read_text()
    assert '"new_test_decision_access": False' in source
    assert "fit_local_adapter" not in source


def test_frozen_bounds_are_read_from_sampling_freeze():
    source=(ROOT/"run_path_excursion.py").read_text()
    assert 'freeze["reliability_bounds"]' in source
    assert '"threshold_changed": False' in source


def test_no_clipping_stabilization_or_threshold_search():
    source=((ROOT/"run_path_excursion.py").read_text()+(ROOT/"path_excursion.py").read_text()).lower()
    for value in ("clip(","spectral_radius","lyapunov_penalty","threshold_grid","fit_local_adapter"):
        assert value not in source
