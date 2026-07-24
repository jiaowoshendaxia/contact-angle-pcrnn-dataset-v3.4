from pathlib import Path

from lspgmoe.ingest import extract_cellulose_ester_jats


def test_cellulose_jats_extraction_has_12_rows(tmp_path: Path):
    source = Path("data/raw/cellulose_ester_PMC9993463.xml")
    output = tmp_path / "candidate.csv"
    report = extract_cellulose_ester_jats(source, output, "OPEN_CELLULOSE_ESTER")
    assert report["n_rows"] == 12
    assert report["n_surfaces"] == 4
    assert report["n_liquids"] == 3
