from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from lspgmoe.recuration import (
    _angle_type,
    _portable_manifest_path,
    _stable_id,
    load_reextraction_files,
    validate_decision_state,
)


def _decisions(decision: str) -> pd.DataFrame:
    return pd.DataFrame({
        "source_group_id": ["SRC001"],
        "decision": [decision],
    })


def test_stable_id_is_deterministic_and_case_insensitive():
    assert _stable_id("SFC", "SRC001", "Sample A") == _stable_id(
        "SFC", "src001", "sample a"
    )


def test_manifest_paths_are_workspace_relative(tmp_path: Path):
    root = tmp_path / "workspace"
    data = root / "data" / "processed_v4_3"
    assert _portable_manifest_path(data, root) == "data/processed_v4_3"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("reported contact angle; static/advancing/receding mode not specified", "reported_unspecified"),
        ("capillary_rise_washburn_derived", "capillary_rise_derived"),
        ("equilibrium sessile drop", "equilibrium"),
        ("advancing angle", "advancing"),
    ],
)
def test_angle_type_normalization(raw: str, expected: str):
    assert _angle_type(raw) == expected


def test_pending_source_blocks_dataset_lock():
    with pytest.raises(RuntimeError, match="remain pending"):
        validate_decision_state(
            _decisions("pending_reextract"),
            pd.DataFrame({"source_group_id": ["SRC001"]}),
        )


def test_reextracted_source_requires_replacement_rows():
    with pytest.raises(ValueError, match="no replacement rows"):
        validate_decision_state(
            _decisions("retain_reextracted"),
            pd.DataFrame({"source_group_id": []}),
        )


def test_reextraction_loader_normalizes_locator_alias(tmp_path: Path):
    path = tmp_path / "records_A.csv"
    pd.DataFrame({
        "source_group_id": ["SRC001"],
        "原表位置": ["Table 2"],
        "surface_label": ["sample"],
        "liquid": ["water"],
        "contact_angle_deg": [90.0],
        "contact_angle_type": ["static"],
        "doi": ["10.example/test"],
    }).to_csv(path, index=False, encoding="utf-8-sig")
    frame = load_reextraction_files([path])
    assert frame.loc[0, "source_locator"] == "Table 2"


def test_reextraction_loader_normalizes_agent_a_schema(tmp_path: Path):
    path = tmp_path / "records_A.csv"
    pd.DataFrame({
        "source_group_id": ["SRC001"],
        "original_table_location": ["Table 2"],
        "surface_label": ["sample"],
        "liquid": ["water"],
        "contact_angle_deg": [90.0],
        "contact_angle_type": ["static"],
        "std_deg": [2.0],
        "replicate_n": [3],
        "DOI": ["10.example/test"],
    }).to_csv(path, index=False, encoding="utf-8-sig")
    frame = load_reextraction_files([path])
    assert frame.loc[0, "source_locator"] == "Table 2"
    assert frame.loc[0, "contact_angle_std_deg"] == 2.0
    assert frame.loc[0, "replicates_n"] == 3
    assert frame.loc[0, "doi"] == "10.example/test"


def test_reextraction_loader_rejects_missing_angle_type(tmp_path: Path):
    path = tmp_path / "records_A.csv"
    pd.DataFrame({
        "source_group_id": ["SRC001"],
        "source_locator": ["Table 2"],
        "surface_label": ["sample"],
        "liquid": ["water"],
        "contact_angle_deg": [90.0],
        "doi": ["10.example/test"],
    }).to_csv(path, index=False)
    with pytest.raises(ValueError, match="contact_angle_type"):
        load_reextraction_files([path])
