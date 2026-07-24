from pathlib import Path

from lspgmoe.ingest import extract_pla_films_jats


def test_pla_table_extracts_four_conditions_and_three_liquids(tmp_path: Path):
    report = extract_pla_films_jats(
        Path("data/raw/pla_films_PMC8707572.xml"),
        tmp_path / "candidate.csv",
        "OPEN_PLA_FILMS_2021",
    )
    assert report["n_rows"] == 48
    assert report["n_surfaces"] == 16
    assert report["n_liquids"] == 3
    assert report["conditions"] == ["Control", "PLA-C", "PLA-A", "PLA-T"]

    import pandas as pd

    frame = pd.read_csv(tmp_path / "candidate.csv", encoding="utf-8-sig")
    assert set(frame["contact_angle_type"]) == {"static"}
    assert set(frame["replicates_n"]) == {15}
    assert frame["contact_angle_std_deg"].notna().all()
