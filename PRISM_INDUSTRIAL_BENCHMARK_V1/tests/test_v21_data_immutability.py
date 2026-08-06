from prism_benchmark.v21_audit import compare_data_base_audits


def _audit(digest="abc"):
    return {"files": [{"path": "base_data/non_sru/train.parquet", "bytes": 10, "sha256": digest}]}


def test_non_sru_data_base_mutation_is_detected():
    assert compare_data_base_audits(_audit(), _audit())["status"] == "PASS"
    changed = compare_data_base_audits(_audit(), _audit("changed"))
    assert changed["status"] == "STOP_DATA_BASE_MUTATED"
    assert changed["changed"] == ["base_data/non_sru/train.parquet"]
