import importlib.util
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
MODULE_PATH = PROJECT_ROOT / "tools" / "make_phase1_m7_manifest.py"
SPEC = importlib.util.spec_from_file_location("phase1_m7_manifest", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_m7_manifest_is_validation_only_and_complete(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(MODULE, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(
        MODULE,
        "load_protocol_config",
        lambda require_phase1_frozen: {
            "training": {"seeds": {"screening": list(range(10))}},
            "phase1_model_selection": {
                "M7": {
                    "amplitude_spline_grid_sizes": [8, 12, 16],
                    "smoothness_weights": [0.0, 1.0e-4, 1.0e-3],
                }
            },
        },
    )
    path = MODULE.make_manifest()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["job_count"] == 90
    assert payload["test_access"] is False
    assert len({job["job_id"] for job in payload["jobs"]}) == 90
    assert all("aggregate" not in job["command"] for job in payload["jobs"])
