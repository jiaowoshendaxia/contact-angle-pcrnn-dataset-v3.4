from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lspgmoe.revision import (
    _nested_oof_path,
    _novelty_label,
    _probe_liquids,
    validate_provenance_decisions,
)


def test_probe_liquid_parser_ignores_empty_and_nan_values():
    assert _probe_liquids("A;B;;nan") == {"A", "B"}


def test_nested_oof_path_supports_current_v41_filename(tmp_path: Path):
    current = tmp_path / "nested_oof_predictions_v4_1.csv"
    current.write_text("sample_id\nA\n", encoding="utf-8")
    assert _nested_oof_path(tmp_path) == current


@pytest.mark.parametrize(
    ("material_known", "liquid_known", "expected"),
    [
        (True, True, "known_material_known_target_liquid"),
        (False, True, "new_material_family"),
        (True, False, "new_target_liquid"),
        (False, False, "new_material_family_and_target_liquid"),
    ],
)
def test_novelty_labels(material_known, liquid_known, expected):
    assert _novelty_label(material_known, liquid_known) == expected


def test_provenance_gate_rejects_unreviewed_source():
    primary = pd.DataFrame({
        "source_group_id": ["SRC001", "SRC002"],
        "v4_split": ["train", "validation"],
    })
    decisions = pd.DataFrame({
        "source_group_id": ["SRC001"],
        "decision": ["retain"],
        "bibliographic_status": ["verified"],
        "location_status": ["verified"],
        "license_status": ["citation_use"],
        "evidence_url": ["https://doi.org/example"],
        "audit_note": ["checked"],
    })
    with pytest.raises(ValueError, match="SRC002"):
        validate_provenance_decisions(primary, decisions, ["retain"])


def test_provenance_gate_rejects_excluded_source():
    primary = pd.DataFrame({
        "source_group_id": ["SRC001"],
        "v4_split": ["train"],
    })
    decisions = pd.DataFrame({
        "source_group_id": ["SRC001"],
        "decision": ["exclude"],
        "bibliographic_status": ["verified"],
        "location_status": ["unverifiable"],
        "license_status": ["unknown"],
        "evidence_url": ["https://doi.org/example"],
        "audit_note": ["not located"],
    })
    with pytest.raises(RuntimeError, match="cleaned-data rerun"):
        validate_provenance_decisions(primary, decisions, ["retain"])
