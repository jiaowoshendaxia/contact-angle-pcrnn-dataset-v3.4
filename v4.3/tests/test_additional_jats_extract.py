from pathlib import Path

import pandas as pd

from lspgmoe.ingest import (
    extract_carbon_surface_jats,
    extract_chitosan_gelatin_jats,
    extract_facemask_surface_jats,
)


def test_carbon_table_extracts_six_surfaces_and_three_liquids(tmp_path: Path):
    report = extract_carbon_surface_jats(
        Path("data/raw/carbon_surfaces_PMC4843010.xml"),
        tmp_path / "carbon.csv",
        "OPEN_CARBON_GLYCAN_2016",
    )
    assert report["n_rows"] == 18
    assert report["n_surfaces"] == 6
    assert report["n_liquids"] == 3


def test_facemask_table_extracts_triplicate_static_angles(tmp_path: Path):
    report = extract_facemask_surface_jats(
        Path("data/raw/facemask_surfaces_PMC10183065.xml"),
        tmp_path / "facemask.csv",
        "OPEN_FACEMASK_2023",
    )
    assert report["n_rows"] == 21
    assert report["n_surfaces"] == 7
    frame = pd.read_csv(tmp_path / "facemask.csv", encoding="utf-8-sig")
    assert set(frame["contact_angle_type"]) == {"static"}
    assert set(frame["replicates_n"]) == {3}


def test_chitosan_table_canonicalizes_two_liquid_surface_rows(tmp_path: Path):
    report = extract_chitosan_gelatin_jats(
        Path("data/raw/chitosan_gelatin_films_PMC9456065.xml"),
        tmp_path / "chitosan.csv",
        "OPEN_CHITOSAN_GELATIN_2022",
    )
    assert report["n_rows"] == 14
    assert report["n_surfaces"] == 7
    frame = pd.read_csv(tmp_path / "chitosan.csv", encoding="utf-8-sig")
    assert set(frame["liquid_name"]) == {"glycerol", "diiodomethane"}
    assert set(frame["replicates_n"]) == {5}
