from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "实验数据1.xlsx"
MANIFEST_DIR = PROJECT_ROOT / "data_manifests" / "cz"
EXPECTED_SOURCE_SHA256 = (
    "c46e0d35d26903386fd80408f36660c4f8925a5dbc56c92527f020e433ef04de"
)
EXPECTED_INPUTS = [
    "加热元件温度",
    "主加热功率",
    "晶升速度",
    "晶转速度",
    "埚升速度",
    "埚转速度",
    "氩气流量设定",
    "晶体长度",
    "炉压",
]
AUTHORITATIVE_DOCUMENTS = [
    "AR_RAPHU_method_v2.md",
    "AR_RAPHU_three_layer_validation_plan_v2.md",
    "AR_RAPHU_v2_revision_notes.md",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(name: str) -> dict:
    with (MANIFEST_DIR / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def test_private_source_integrity_and_structural_contract() -> None:
    manifest = load_json("data_manifest.json")
    sheet = manifest["workbook"]["sheets"][0]

    assert sha256_file(SOURCE) == EXPECTED_SOURCE_SHA256
    assert manifest["source"]["sha256"] == EXPECTED_SOURCE_SHA256
    assert manifest["privacy"]["classification"] == "PRIVATE_CZ"
    assert manifest["privacy"]["upload_permitted"] is False
    assert manifest["audit_scope"]["numeric_distribution_profiled"] is False
    assert sheet["data_row_count"] == 20103
    assert sheet["column_count"] == 10
    assert sheet["formula_cell_count"] == 0
    assert all(column["missing_count"] == 0 for column in sheet["columns"])
    argon = next(
        column for column in sheet["columns"] if column["header"] == "氩气流量设定"
    )
    assert argon["constant_over_all_rows"] is True


def test_confirmed_variable_roles_are_p9_plus_strictly_lagged_target() -> None:
    semantics = load_json("model_semantics.json")
    with (MANIFEST_DIR / "variable_dictionary.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        variables = list(csv.DictReader(handle))

    data = semantics["current_data_semantics"]
    assert data["external_process_channels"] == 9
    assert data["historical_output_channels"] == 1
    assert data["model_input_channels"] == 10
    assert data["external_process_columns"] == EXPECTED_INPUTS
    assert semantics["current_data_semantics"]["target_column"] == "晶体直径"
    assert data["current_target_used_as_input"] is False
    assert data["strictly_lagged_target_used_as_input"] is True
    assert data["historical_output_minimum_lag"] == 1
    assert data["crystal_count"] == 1
    assert data["stage"] == "constant_diameter_only"
    assert [row["canonical_name"] for row in variables[:-1]] == EXPECTED_INPUTS
    assert all(row["role"] == "input" for row in variables[:-1])
    assert variables[-1]["canonical_name"] == "晶体直径"
    assert variables[-1]["role"] == "target_and_strictly_lagged_AR_source"
    assert variables[-1]["same_index_as_target_used_as_input"] == "no"
    assert variables[-1]["prediction_origin_t_value_used_as_input"] == "yes"
    argon = next(row for row in variables if row["canonical_name"] == "氩气流量设定")
    assert argon["scientifically_trainable_current_crystal"] == (
        "no_constant_unidentifiable"
    )


def test_four_fold_direct_forecast_protocol_is_frozen() -> None:
    split = load_json("split_manifest.json")

    assert split["protocol_status"] == "FROZEN"
    assert split["fold_boundaries_frozen"] is True
    assert split["sample_window_indices_materialized"] is False
    assert split["sequence_length"] == 20103
    assert len(split["folds"]) == 4
    assert split["folds"][0] == {
        "fold": 1,
        "role": "development",
        "train": [0, 10051],
        "validation": [10051, 12061],
        "test": [12061, 14072],
    }
    assert split["folds"][-1]["test"] == [18092, 20103]
    assert split["window_protocol"]["legacy_reproduction"] == {"L_x": 32, "L_y": 32}
    assert split["window_protocol"]["formal_L_y_candidates"] == [1, 4, 8, 16, 32, 64]
    assert split["window_protocol"]["formal_L_x_candidates"] == [32, 64, 128, 256]
    assert split["forecast_protocol"]["prediction_horizons_samples"] == [
        1,
        5,
        10,
        30,
        60,
    ]
    assert split["forecast_protocol"]["split_membership_by"] == (
        "target_index_t_plus_h"
    )
    assert split["forecast_protocol"]["future_X_after_t_allowed"] is False
    assert split["trajectory"]["crystal_count"] == 1
    assert split["trajectory"]["stage"] == "constant_diameter_only"
    assert split["required_policy"][
        "historical_inputs_may_precede_target_partition_start"
    ] is True
    assert split["test_access"]["Fold_4_test_locked"] is True
    assert split["not_selected"]["random_row_split"] == "PROHIBITED"
    assert split["blocking_fields"] == []


def test_v2_model_scope_and_tracks_are_frozen() -> None:
    semantics = load_json("model_semantics.json")

    assert semantics["method_acronym"] == "AR-RAPHU"
    recorded_documents = semantics["authoritative_documents"]
    assert [
        document["path_relative_to_project"] for document in recorded_documents
    ] == AUTHORITATIVE_DOCUMENTS
    for document in recorded_documents:
        path = PROJECT_ROOT / document["path_relative_to_project"]
        assert document["size_bytes"] == path.stat().st_size
        assert document["sha256"] == sha256_file(path)
    assert set(semantics["task_tracks"]) >= {"Track-X", "Track-AR", "Track-XAR"}
    assert semantics["model_a"]["historical_output_branch"]["enabled"] is True
    assert semantics["model_a"]["historical_output_branch"]["minimum_lag"] == 1
    assert semantics["model_b"]["historical_output_residual_enabled"] is False
    assert semantics["model_b"]["scope"] == (
        "external_process_branches_only_in_v2_first_version"
    )
    assert semantics["forecast_protocol"]["primary_mode"] == "direct_multi_horizon"
    assert semantics["forecast_protocol"]["future_X_after_t_allowed"] is False
    assert semantics["window_selection_protocol"][
        "global_pair_frozen_before_Fold_4_test"
    ] is True
    assert "exact_Gamma_discretization" not in semantics["blocking_fields"]
    assert "KAN_architecture" not in semantics["blocking_fields"]
    assert semantics["blocking_fields"] == []
    assert semantics["model_a"]["external_lag_kernel_exact_discretization"] == (
        "point_value_then_softmax"
    )
    assert semantics["model_a"]["gamma_shape_rate_or_scale_convention"] == (
        "shape_scale"
    )
    assert semantics["model_a"]["gamma_zero_offset_delta"] == 0.001


def test_v20_reference_contract_is_audited_and_scoped() -> None:
    audit = load_json("v20_implementation_audit.json")
    semantics = load_json("model_semantics.json")

    assert audit["integrity"]["status"] == "VERIFIED"
    assert audit["integrity"]["verified_file_count"] == 64
    assert audit["integrity"]["all_listed_files_match"] is True
    gamma = audit["historical_v20_external_gamma_contract"]
    assert gamma["discretization"] == "point_value_then_softmax"
    assert gamma["beta_convention"] == "scale"
    assert gamma["delta"] == 0.001
    assert gamma["learned_parametrization"] == "bounded_mean_std"
    kan = audit["historical_v20_kan_contract"]
    assert kan["per_variable_layers"] == [1, 8, 1]
    assert kan["grid_size"] == 7
    assert kan["spline_order"] == 3
    assert "process_data.py" in audit["missing_from_bundle"]
    assert semantics["historical_v20_reference"]["integrity"]["status"] == "VERIFIED"


def test_v2_dual_branch_adapter_is_frozen_by_hash() -> None:
    semantics = load_json("model_semantics.json")
    adapter = semantics["v2_adapter"]

    assert adapter["status"] == "STRUCTURAL_ADAPTER_IMPLEMENTED_AND_UNIT_TESTED"
    assert adapter["implementation"]["separate_external_and_AR_branches"] is True
    assert adapter["implementation"]["unequal_L_x_L_y_supported"] is True
    assert adapter["implementation"]["constant_argon_scientifically_masked"] is True
    assert adapter["implementation"][
        "train_only_first_layer_grid_ranges_required_at_construction"
    ] is True
    assert adapter["phase1_gate"]["status"] == "PHASE1_PREREGISTERED"
    protocol = json.loads(
        (PROJECT_ROOT / "configs" / "protocol_v2.yaml").read_text(
            encoding="utf-8"
        )
    )
    if protocol["execution_scope"]["private_CZ"].startswith("EXCLUDED_BY_USER"):
        # The CZ manifest is an immutable archival snapshot. Public/synthetic
        # implementation work after exclusion must not mutate or regenerate it
        # by reopening the private workbook.
        assert adapter["file_hashes"]
        return
    for relative_path, expected in adapter["file_hashes"].items():
        assert sha256_file(PROJECT_ROOT / relative_path) == expected


def test_generated_artifact_checksums_are_current() -> None:
    checksums = load_json("sha256sums.json")

    assert checksums["source"][SOURCE.name] == EXPECTED_SOURCE_SHA256
    for name, expected in checksums["artifacts"].items():
        assert sha256_file(MANIFEST_DIR / name) == expected
