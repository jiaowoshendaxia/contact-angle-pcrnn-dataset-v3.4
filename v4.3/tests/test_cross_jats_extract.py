from pathlib import Path

from lspgmoe.ingest import extract_cross_material_jats


def test_cross_material_extraction_excludes_ranges(tmp_path: Path):
    source = Path("data/raw/cross_material_PMC13031877.xml")
    output = tmp_path / "candidate.csv"
    report = extract_cross_material_jats(source, output, "OPEN_CROSS_MATERIAL_2026")
    assert report["n_rows"] == 91
    assert report["n_skipped"] == 19
    assert report["n_liquids"] == 5
