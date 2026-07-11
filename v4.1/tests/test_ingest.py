import pandas as pd

from lspgmoe.ingest import validate_open_source_frame


def _registry():
    return {
        "doi": "10.1234/example",
        "url": "https://example.org/data",
        "verification_status": "verified",
        "license_or_access": "CC BY 4.0",
    }


def _valid_frame():
    return pd.DataFrame([
        {
            "solid_name": "PTFE", "solid_family": "polymer", "solid_substrate": "sheet",
            "surface_treatment": "untreated", "surface_state": "smooth",
            "liquid_name": "water", "liquid_total_surface_tension_mN_m": 72.8,
            "liquid_dispersion_mN_m": 21.8, "liquid_polar_mN_m": 51.0,
            "contact_angle_deg": 110.0, "contact_angle_type": "static",
            "measurement_method": "sessile_drop",
        },
        {
            "solid_name": "PTFE", "solid_family": "polymer", "solid_substrate": "sheet",
            "surface_treatment": "untreated", "surface_state": "smooth",
            "liquid_name": "formamide", "liquid_total_surface_tension_mN_m": 58.0,
            "liquid_dispersion_mN_m": 39.0, "liquid_polar_mN_m": 19.0,
            "contact_angle_deg": 90.0, "contact_angle_type": "static",
            "measurement_method": "sessile_drop",
        },
        {
            "solid_name": "PTFE", "solid_family": "polymer", "solid_substrate": "sheet",
            "surface_treatment": "untreated", "surface_state": "smooth",
            "liquid_name": "diiodomethane", "liquid_total_surface_tension_mN_m": 50.8,
            "liquid_dispersion_mN_m": 50.8, "liquid_polar_mN_m": 0.0,
            "contact_angle_deg": 70.0, "contact_angle_type": "static",
            "measurement_method": "sessile_drop",
        },
    ])


def test_valid_import_is_staged_and_counts_probe_surface():
    staged, report = validate_open_source_frame(_valid_frame(), "OPEN_TEST", _registry())
    assert report["status"] == "pass"
    assert report["n_surfaces_with_at_least_3_liquids"] == 1
    assert set(staged["source_group_id"]) == {"OPEN_TEST"}
    assert set(staged["import_status"]) == {"validated_staged"}


def test_invalid_angle_and_unverified_license_are_rejected():
    frame = _valid_frame()
    frame.loc[0, "contact_angle_deg"] = 181.0
    registry = _registry()
    registry["verification_status"] = "to_verify"
    _, report = validate_open_source_frame(frame, "OPEN_TEST", registry)
    assert report["status"] == "fail"
    assert any("angle_out_of_range" in error for error in report["errors"])
    assert "registry_license_or_access_not_verified" in report["errors"]
