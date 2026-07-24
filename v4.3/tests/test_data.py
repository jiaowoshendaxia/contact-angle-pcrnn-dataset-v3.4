from __future__ import annotations

import pandas as pd

from lspgmoe.data import V4Tables, build_samples, canonical_liquid_name, stable_id, validate_v4_tables


def make_tables() -> V4Tables:
    sources = pd.DataFrame([{"source_group_id": "SRC1"}])
    surfaces = pd.DataFrame([{"surface_group_id": "SFC1", "source_group_id": "SRC1"}])
    liquids = pd.DataFrame([
        {"liquid_id": "L1", "liquid_name": "water"},
        {"liquid_id": "L2", "liquid_name": "diiodomethane"},
        {"liquid_id": "L3", "liquid_name": "formamide"},
    ])
    measurements = pd.DataFrame([
        {"measurement_id": "M1", "record_id": "M1", "surface_group_id": "SFC1", "source_group_id": "SRC1", "liquid_id": "L1", "contact_angle_deg": 80.0, "solid_surface_energy_source_type": "inferred_from_contact_angles", "solid_dispersion_mJ_m2": 20.0, "solid_polar_mJ_m2": 10.0, "target_eligible": "yes"},
        {"measurement_id": "M2", "record_id": "M2", "surface_group_id": "SFC1", "source_group_id": "SRC1", "liquid_id": "L2", "contact_angle_deg": 45.0, "solid_surface_energy_source_type": "inferred_from_contact_angles", "solid_dispersion_mJ_m2": 20.0, "solid_polar_mJ_m2": 10.0, "target_eligible": "yes"},
        {"measurement_id": "M3", "record_id": "M3", "surface_group_id": "SFC1", "source_group_id": "SRC1", "liquid_id": "L3", "contact_angle_deg": 60.0, "solid_surface_energy_source_type": "literature_reported", "solid_dispersion_mJ_m2": 20.0, "solid_polar_mJ_m2": 10.0, "target_eligible": "yes"},
    ])
    splits = pd.DataFrame([
        {"measurement_id": item, "surface_group_id": "SFC1", "source_group_id": "SRC1", "v4_split": "train"}
        for item in ["M1", "M2", "M3"]
    ])
    return V4Tables(sources, surfaces, liquids, measurements, splits)


def test_stable_id_is_deterministic_and_case_normalized() -> None:
    assert stable_id("SFC", [" DOI ", "PTFE"]) == stable_id("SFC", ["doi", "ptfe"])


def test_liquid_aliases_share_canonical_identity() -> None:
    assert canonical_liquid_name("glycerin") == "glycerol"


def test_target_liquid_is_masked_from_probe_set() -> None:
    tables = make_tables()
    samples = build_samples(tables)
    probe_samples = samples[samples.prediction_mode == "probe_assisted"]
    assert len(probe_samples) == 3
    for row in probe_samples.itertuples(index=False):
        assert row.target_liquid_id not in row.probe_liquid_ids.split(";")
        assert row.n_probes == 2
    assert validate_v4_tables(tables, samples)["status"] == "pass"


def test_zero_shot_hides_angle_derived_sfe() -> None:
    samples = build_samples(make_tables())
    inferred = samples[(samples.prediction_mode == "zero_shot") & (samples.record_id == "M1")].iloc[0]
    reported = samples[(samples.prediction_mode == "zero_shot") & (samples.record_id == "M3")].iloc[0]
    assert inferred.has_independent_sfe == "no"
    assert pd.isna(inferred.independent_sfe_dispersion_mj_m2)
    assert reported.has_independent_sfe == "yes"


def test_independent_sfe_requires_both_finite_components() -> None:
    tables = make_tables()
    tables.measurements.loc[tables.measurements.measurement_id == "M3", "solid_polar_mJ_m2"] = float("nan")
    samples = build_samples(tables)
    reported = samples[(samples.prediction_mode == "zero_shot") & (samples.record_id == "M3")].iloc[0]
    assert reported.has_independent_sfe == "no"
    assert pd.isna(reported.independent_sfe_polar_mj_m2)
