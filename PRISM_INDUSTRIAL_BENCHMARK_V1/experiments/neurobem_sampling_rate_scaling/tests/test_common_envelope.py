import json
from pathlib import Path
ROOT=Path(__file__).parents[1]
def test_common_bound_is_frozen_100hz_bound():
 s=(ROOT/"run_common_envelope.py").read_text(); assert 'common= bounds["hz100_h20"][route]' in s
def test_full_state_has_no_scalar_threshold():
 s=(ROOT/"path_excursion.py").read_text(); assert 'for name in ("velocity", "attitude", "body_rate")' in s
def test_original_nested_90pct_rule_retained():
 c=json.loads((ROOT/"configs"/"common_envelope.yaml").read_text()); assert c["reliability_probability_minimum"]==.9; assert c["reliability_candidate_ms"]==[10,50,100,200,500,1000]
def test_diagnostic_is_replay_without_fit_or_tuning():
 s=(ROOT/"run_common_envelope.py").read_text(); assert "fit_local_adapter" not in s; assert '"new_test_decision_access":False' in s; assert '"threshold_changed":False' in s
def test_terminal_summary_uses_safe_json_scalars():
 s=(ROOT/"run_common_envelope.py").read_text(); assert "json.dumps(_safe(meta)" in s
