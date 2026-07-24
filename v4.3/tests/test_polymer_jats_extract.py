from pathlib import Path

from lspgmoe.ingest import extract_polymer_contact_angle_jats


def test_polymer_table_extraction_has_51_values(tmp_path: Path):
    report = extract_polymer_contact_angle_jats(
        Path("data/raw/polymer_surfaces_PMC10504009.xml"),
        tmp_path / "candidate.csv",
        "OPEN_POLYMER_SURFACES_2023",
    )
    assert report["n_rows"] == 54
    assert report["n_surfaces"] == 18
