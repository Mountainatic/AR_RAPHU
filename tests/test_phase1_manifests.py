import importlib.util
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
MODULE_PATH = PROJECT_ROOT / "tools" / "make_phase1_scheme_a_manifest.py"
SPEC = importlib.util.spec_from_file_location("phase1_manifest", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_manifest_counts_and_no_test_aggregate_commands(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(MODULE, "PROJECT_ROOT", tmp_path)
    # Keep the real config and runner resolution available to the loader.
    monkeypatch.setattr(
        MODULE,
        "load_protocol_config",
        lambda require_phase1_frozen: {
            "training": {
                "seeds": {
                    "screening": list(range(10)),
                    "critical": list(range(30)),
                },
                "source_backed_v20_reference": {
                    "pruning_scales": [0.003, 0.006]
                },
            }
        },
    )
    warmup = MODULE.make_manifest(
        "AR-S0", "warmup", device="cuda", track="XAR"
    )
    fork = MODULE.make_manifest("AR-S0", "fork", device="cuda", track="X")
    dense = MODULE.make_manifest(
        "AR-S0", "dense_ar", device="cuda", track="XAR"
    )
    critical = MODULE.make_manifest(
        "AR-S0",
        "warmup",
        device="cuda",
        track="XAR",
        replicate_profile="critical",
    )
    import json

    warmup_payload = json.loads(warmup.read_text())
    fork_payload = json.loads(fork.read_text())
    dense_payload = json.loads(dense.read_text())
    critical_payload = json.loads(critical.read_text())
    assert warmup_payload["job_count"] == 10
    assert fork_payload["job_count"] == 20
    assert dense_payload["job_count"] == 10
    assert critical_payload["job_count"] == 30
    assert critical_payload["replicate_profile"] == "critical"
    assert warmup_payload["track"] == "XAR"
    assert fork_payload["track"] == "X"
    assert dense_payload["track"] == "AR"
    assert dense.name == "AR-S0_G2_AR_dense_ar.json"
    for payload in (warmup_payload, fork_payload):
        assert payload["test_access"] is False
        assert all("aggregate" not in job["command"] for job in payload["jobs"])
