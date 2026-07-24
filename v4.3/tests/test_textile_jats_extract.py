from pathlib import Path

from lspgmoe.ingest import extract_textile_surface_jats


def test_textile_table_extracts_static_ca_only(tmp_path: Path):
    report = extract_textile_surface_jats(
        Path("data/raw/textile_surfaces_PMC6473839.xml"),
        tmp_path / "candidate.csv",
        "OPEN_TEXTILE_ANTI_WETTING_2019",
    )
    assert report["n_rows"] == 48
    assert report["n_surfaces"] == 16
    assert report["n_liquids"] == 3
    assert report["excluded_shedding_angle_columns"] is True

    import pandas as pd

    frame = pd.read_csv(tmp_path / "candidate.csv", encoding="utf-8-sig")
    assert frame["contact_angle_deg"].max() == 166.0
    assert 12.8 not in set(frame["contact_angle_deg"])
    assert set(frame["contact_angle_type"]) == {"static"}
    assert frame["surface_group_id"].isna().all() if "surface_group_id" in frame else True
