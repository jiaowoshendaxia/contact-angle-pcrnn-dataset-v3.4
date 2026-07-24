from __future__ import annotations

import csv
import hashlib
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw_sources"

FIELDS = [
    "record_status",
    "source_group_id",
    "source_title",
    "original_table_location",
    "surface_label",
    "material",
    "treatment",
    "state",
    "liquid",
    "contact_angle_deg",
    "contact_angle_type",
    "std_deg",
    "range_deg",
    "reported_uncertainty_deg",
    "uncertainty_type",
    "replicate_n",
    "measurement_method",
    "droplet_volume",
    "license",
    "DOI",
    "keep_reason",
    "extraction_note",
]

rows: list[dict[str, object]] = []


def add(
    *,
    status: str,
    source_id: str,
    title: str,
    location: str,
    surface: str,
    material: str,
    treatment: str,
    state: str,
    liquid: str,
    angle: float,
    angle_type: str,
    uncertainty: float | None,
    uncertainty_type: str,
    replicate_n: int | None,
    method: str,
    volume: str,
    license_text: str,
    doi: str,
    reason: str,
    note: str,
    std: float | None = None,
    range_text: str = "",
) -> None:
    rows.append(
        {
            "record_status": status,
            "source_group_id": source_id,
            "source_title": title,
            "original_table_location": location,
            "surface_label": surface,
            "material": material,
            "treatment": treatment,
            "state": state,
            "liquid": liquid,
            "contact_angle_deg": angle,
            "contact_angle_type": angle_type,
            "std_deg": "" if std is None else std,
            "range_deg": range_text,
            "reported_uncertainty_deg": "" if uncertainty is None else uncertainty,
            "uncertainty_type": uncertainty_type,
            "replicate_n": "" if replicate_n is None else replicate_n,
            "measurement_method": method,
            "droplet_volume": volume,
            "license": license_text,
            "DOI": doi,
            "keep_reason": reason,
            "extraction_note": note,
        }
    )


# SRC001: Tables 2 and 3 are the primary numeric source. The paper reports
# capillary-rise/Washburn contact angles, not sessile-drop static angles.
src001_title = "Wettability Modification of Nanomaterials by Low-Energy Electron Flux"
src001_license = (
    "Copyright 2009 to the authors; PMC OA API license=none; no machine-readable "
    "Creative Commons license in the retrieved article XML."
)
src001_method = (
    "Packed-bed capillary-rise method; liquid-rise height versus time converted "
    "with the Washburn equation; n-hexane used to determine effective capillary radius."
)
src001_liquids = [
    "1-bromonaphthalene",
    "ethylene glycol",
    "diiodomethane",
    "formamide",
    "water",
]
src001_surfaces = [
    (
        "Table 2",
        "As-prepared diamond powder",
        "diamond powder (average particle size 1 um)",
        "untreated/as prepared",
        [35, 54, 76, 62, 75],
        [3, 2, 1, 2, 1],
    ),
    (
        "Table 2",
        "UV-illuminated diamond powder",
        "diamond powder (average particle size 1 um)",
        "Hg-Xe UV illumination for approximately 5 min",
        [38, 54, 69, 52, 70],
        [2, 2, 3, 2, 1],
    ),
    (
        "Table 2",
        "E-beam-irradiated diamond powder",
        "diamond powder (average particle size 1 um)",
        "300 eV electron irradiation, 360 uC/cm2, 1e-7 Torr, room temperature",
        [35, 70, 80, 80, 84],
        [2, 1, 2, 3, 2],
    ),
    (
        "Table 3",
        "As-prepared ZnO nanomaterial",
        "ZnO powder (99.4% purity; average particle size 200 nm)",
        "untreated/as prepared",
        [9, 63, 66, 52, 60],
        [1, 1, 2, 3, 4],
    ),
    (
        "Table 3",
        "UV-illuminated ZnO nanomaterial",
        "ZnO powder (99.4% purity; average particle size 200 nm)",
        "Hg-Xe UV illumination for approximately 5 min",
        [8, 14, 64, 20, 3],
        [2, 2, 1, 2, 1],
    ),
    (
        "Table 3",
        "E-beam-irradiated ZnO nanomaterial",
        "ZnO powder (99.4% purity; average particle size 200 nm)",
        "300 eV electron irradiation, 360 uC/cm2, 1e-7 Torr, room temperature",
        [8, 72, 70, 73, 85],
        [1, 1, 2, 3, 2],
    ),
]
for table, surface, material, treatment, angles, uncertainties in src001_surfaces:
    for liquid, angle, uncertainty in zip(src001_liquids, angles, uncertainties):
        add(
            status="keep",
            source_id="SRC001",
            title=src001_title,
            location=f"{table}, row '{surface}', liquid column '{liquid}'",
            surface=surface,
            material=material,
            treatment=treatment,
            state="powder manually packed into a glass tube; particles shaken during treatment",
            liquid=liquid,
            angle=angle,
            angle_type="capillary_rise_washburn_derived",
            uncertainty=uncertainty,
            uncertainty_type="reported_plus_minus_type_not_defined",
            replicate_n=None,
            method=src001_method,
            volume="not applicable (capillary-rise measurement)",
            license_text=src001_license,
            doi="10.1007/s11671-009-9380-0",
            reason="keep_primary_table_value",
            note=(
                "Re-extracted directly from the article XML. Do not map the treatment "
                "to the legacy 100 h/200 h labels; those durations are not reported here."
            ),
        )


# SRC004: Supplementary Table S2 explicitly labels all three-liquid values as
# advancing contact angles. Water hysteresis is a different quantity and is not imported.
src004_title = "Hydrophobicity of Rare-Earth Oxide Ceramics"
src004_license = "Copyright 2013 Macmillan Publishers Limited. All rights reserved."
src004_method = (
    "Rame-Hart M500 advanced goniometer; sintered pellets polished to a mirror "
    "finish down to 0.03 um alumina, solvent/water cleaned, and vacuum-desiccated."
)
src004_liquids = ["water", "ethylene glycol", "diiodomethane"]
src004_surfaces = [
    ("Sintered CeO2", "CeO2", [103, 78, 55], [2, 2, 2]),
    ("Sintered Pr6O11", "Pr6O11", [102, 80, 64], [3, 3, 4]),
    ("Sintered Nd2O3", "Nd2O3", [101, 76, 58], [3, 4, 3]),
    ("Sintered Sm2O3", "Sm2O3", [107, 84, 58], [2, 3, 2]),
    ("Sintered Eu2O3", "Eu2O3", [104, 78, 55], [4, 4, 3]),
    ("Sintered Gd2O3", "Gd2O3", [109, 85, 67], [2, 5, 3]),
    ("Sintered Tb7O12", "Tb7O12", [107, 69, 45], [3, 4, 5]),
    ("Sintered Dy2O3", "Dy2O3", [105, 77, 55], [5, 3, 3]),
    ("Sintered Ho2O3", "Ho2O3", [115, 88, 60], [3, 2, 5]),
    ("Sintered Er2O3", "Er2O3", [108, 82, 58], [5, 4, 5]),
    ("Sintered Tm2O3", "Tm2O3", [112, 87, 60], [4, 2, 2]),
    ("Sintered Yb2O3", "Yb2O3", [100, 73, 56], [5, 4, 5]),
    ("Sintered Lu2O3", "Lu2O3", [98, 79, 55], [3, 5, 4]),
    (
        "CeO2 sputtered on smooth silicon",
        "CeO2 thin film on silicon",
        [109, 80, 63],
        [3, 3, 2],
    ),
    (
        "Er2O3 sputtered on smooth silicon",
        "Er2O3 thin film on silicon",
        [110, 81, 65],
        [2, 2, 2],
    ),
]
for surface, material, angles, uncertainties in src004_surfaces:
    sputtered = "sputtered" in surface.lower()
    treatment = (
        "rare-earth oxide sputtered on smooth silicon"
        if sputtered
        else "sintered pellet; mirror polished to 0.03 um before measurement"
    )
    state = (
        "smooth silicon-supported film; cleaned and vacuum-desiccated"
        if sputtered
        else "polished sintered ceramic; cleaned and vacuum-desiccated"
    )
    for liquid, angle, uncertainty in zip(src004_liquids, angles, uncertainties):
        add(
            status="keep",
            source_id="SRC004",
            title=src004_title,
            location=f"Supplementary Table S2, row '{surface}', {liquid} theta_adv column",
            surface=surface,
            material=material,
            treatment=treatment,
            state=state,
            liquid=liquid,
            angle=angle,
            angle_type="advancing",
            uncertainty=uncertainty,
            uncertainty_type="reported_plus_minus_type_not_defined",
            replicate_n=None,
            method=src004_method,
            volume="not reported",
            license_text=src004_license,
            doi="10.1038/NMAT3545",
            reason="keep_primary_supplementary_table_value",
            note=(
                "Table S2 labels theta_adv. The separate water contact-angle-hysteresis "
                "column was intentionally not imported as a contact-angle target."
            ),
        )


# SRC012: Main-text Table 1 gives two explicitly identified surfaces and four
# organic probe liquids. Other water angles in figures/text refer to different
# deposition charges or durability states and are not merged into these surfaces.
src012_title = (
    "Transparent, Thermally and Mechanically Stable Superhydrophobic Coating "
    "Prepared by an Electrochemical Template Strategy"
)
src012_license = (
    "Copyright The Royal Society of Chemistry 2015; author version and ESI are "
    "publicly accessible, but no Creative Commons license is stated."
)
src012_method = (
    "OCA 20 contact-angle system; sessile-drop measurement in air; reported "
    "values are means from three positions on each sample."
)
src012_liquids = [
    ("diiodomethane", 101.1, 161.1, 2, 2),
    ("ethylene glycol", 93.2, 159.4, 2, 2),
    ("peanut oil", 78.7, 150.7, 3, 3),
    ("hexadecane", 71.8, 128.6, 3, 4),
]
src012_surface_defs = [
    (
        "Flat fluorinated ITO glass",
        "fluorinated ITO-coated glass",
        "flat fluorinated reference surface",
        "flat reference surface measured in air",
    ),
    (
        "Porous superhydrophobic silica-coated ITO glass (PEDOT EC 47.7 mC/cm2)",
        "POTS-fluorinated porous silica coating on ITO glass",
        (
            "PEDOT template electrodeposited at 47.7 mC/cm2; TEOS CVD 30 h; "
            "calcined at 500 C for 2 h; POTS CVD 12 h"
        ),
        "porous Cassie-state coating measured in air",
    ),
]
for liquid, flat_angle, coated_angle, flat_u, coated_u in src012_liquids:
    for (surface, material, treatment, state), angle, uncertainty in [
        (src012_surface_defs[0], flat_angle, flat_u),
        (src012_surface_defs[1], coated_angle, coated_u),
    ]:
        add(
            status="keep",
            source_id="SRC012",
            title=src012_title,
            location=f"Main-text Table 1, row '{liquid}', {'flat' if angle == flat_angle else 'coated'} surface CA column",
            surface=surface,
            material=material,
            treatment=treatment,
            state=state,
            liquid=liquid,
            angle=angle,
            angle_type="static_sessile_drop",
            uncertainty=uncertainty,
            uncertainty_type="reported_plus_minus_type_not_defined",
            replicate_n=3,
            method=src012_method,
            volume="5 uL organic droplet",
            license_text=src012_license,
            doi="10.1039/C4TA06944G",
            reason="keep_primary_main_table_value",
            note=(
                "Re-extracted from Table 1 and cross-checked against the ESI method. "
                "Untabulated water angles and post-durability values are not joined to this surface."
            ),
        )


# SRC014: Article Tables 2 and 7 explicitly report average, SD, and n.
# This becomes the one canonical re-extraction for DOI D2RA08165B.
src014_title = (
    "Exploration of Advanced Cellulosic Material for Membrane Filtration "
    "with Outstanding Antifouling Property"
)
src014_license = "CC BY (PMC Open Access API metadata); Royal Society of Chemistry article."
src014_method = (
    "DropMaster 700 (Kyowa Interface Science), 20 C; sessile drop on dried "
    "polymer film formed on glass."
)
src014_liquids = ["formamide", "diiodomethane", "water"]
src014_table2 = [
    (
        "CTA(a)",
        "cellulose triacetate",
        "acetyl substitution, DS 3.0",
        [50.7, 29.0, 61.7],
        [1.8, 1.2, 0.9],
        [28, 30, 30],
    ),
    (
        "CTP",
        "cellulose tripropionate",
        "propionyl substitution, DS 2.9",
        [61.3, 35.8, 70.3],
        [1.1, 0.9, 1.0],
        [29, 30, 38],
    ),
    (
        "CTB",
        "cellulose tributyrate",
        "butyryl substitution, DS 3.1",
        [71.2, 41.0, 80.2],
        [0.6, 0.6, 0.8],
        [29, 30, 38],
    ),
    (
        "CTV",
        "cellulose trivalerate",
        "valeryl substitution, DS 3.1",
        [78.7, 45.2, 86.9],
        [0.8, 0.6, 0.9],
        [30, 30, 39],
    ),
]
src014_table7 = [
    (
        "CTL",
        "cellulose trilaurate",
        "lauroyl substitution, DS 3.0",
        [87.8, 54.3, 102.9],
        [1.1, 1.0, 0.4],
        [10, 10, 10],
    ),
    (
        "CLTOD(a)",
        "cellulose laurate trioxadecanoate",
        "lauroyl DS 1.7 and trioxadecanoyl DS 1.3",
        [84.7, 46.0, 97.4],
        [1.5, 1.1, 0.4],
        [10, 10, 10],
    ),
    (
        "CLTOD(b)",
        "cellulose laurate trioxadecanoate",
        "lauroyl DS 1.1 and trioxadecanoyl DS 1.9",
        [77.3, 47.5, 90.3],
        [1.6, 3.2, 0.8],
        [10, 10, 10],
    ),
    (
        "PES",
        "polyethersulfone reference film",
        "PES dissolved in DMAC and dried at 100 C",
        [57.2, 28.0, 81.2],
        [2.6, 1.3, 1.3],
        [10, 10, 10],
    ),
    (
        "CTA(b)",
        "cellulose triacetate reference film",
        "acetyl substitution, DS 2.9",
        [51.9, 36.2, 62.1],
        [1.8, 1.3, 1.0],
        [10, 10, 10],
    ),
]
for table, surface_defs in [("Table 2", src014_table2), ("Table 7", src014_table7)]:
    for surface, material, treatment, angles, sds, ns in surface_defs:
        for liquid, angle, sd, n in zip(src014_liquids, angles, sds, ns):
            add(
                status="keep",
                source_id="SRC014",
                title=src014_title,
                location=f"{table}, row '{surface}', {liquid} average/SD/n columns",
                surface=surface,
                material=material,
                treatment=treatment,
                state="dried polymer film on glass; measured at 20 C",
                liquid=liquid,
                angle=angle,
                angle_type="sessile_drop_on_dried_film",
                uncertainty=sd,
                uncertainty_type="standard_deviation",
                replicate_n=n,
                method=src014_method,
                volume="not reported",
                license_text=src014_license,
                doi="10.1039/D2RA08165B",
                reason=(
                    "keep_canonical_reextraction; supersedes duplicate legacy copies "
                    "SRC014 and OPEN_CELLULOSE_ESTER"
                ),
                note=(
                    "Average, SD, and n were transcribed from the same row. Captive-bubble "
                    "Table 12 values were excluded because they are a different hydrated-air-bubble task."
                ),
                std=sd,
            )


# SRC016: Table 2 values for the four coated papers are primary to this
# article. PS and PDMS values carry footnote b to Juvonen et al. (2013) and
# remain in the audit CSV as explicit exclusions.
src016_title = "Printed Paper-Based Arrays as Substrates for Biofilm Formation"
src016_license = (
    "Copyright 2014 Maattaenen et al.; licensee Springer; PMC OA API license=none; "
    "no machine-readable Creative Commons license in the retrieved article XML."
)
src016_method = (
    "CAM 200 goniometer; 1-2 uL droplets in RH 15+/-5% and 24+/-1 C; "
    "apparent static angle averaged over three stabilized measurements; Table 2 "
    "reports Wenzel roughness-corrected theta_r unless footnoted otherwise."
)
src016_liquids = ["water", "diiodomethane", "ethylene glycol"]
src016_surfaces = [
    (
        "Latex 1 coated paper",
        "paper with latex-blend barrier coating",
        "reverse-gravure coated, IR dried, calendered at 70 bar/35 C, washed and IR cured",
        [84, 52, 73],
        [1, 1, 1],
        "keep",
        "keep_primary_roughness_corrected_table_value",
    ),
    (
        "Latex 2 coated paper",
        "paper with latex-blend barrier coating",
        "reverse-gravure coated, IR dried, calendered at 70 bar/35 C, washed and IR cured",
        [74, 47, 59],
        [3, 2, 2],
        "keep",
        "keep_primary_roughness_corrected_table_value",
    ),
    (
        "Kaolin coated paper",
        "paper with kaolin-pigment barrier coating",
        "reverse-gravure coated, IR dried, calendered at 50 bar/70 C",
        [64, 50, 48],
        [2, 1, 1],
        "keep",
        "keep_primary_roughness_corrected_table_value",
    ),
    (
        "PCC coated paper",
        "paper with precipitated-calcium-carbonate barrier coating",
        "reverse-gravure coated, IR dried, calendered at 50 bar/70 C",
        [92, 52, 52],
        [3, 1, 2],
        "keep",
        "keep_primary_roughness_corrected_table_value",
    ),
    (
        "PS (96-well) reference",
        "polystyrene 96-well microplate",
        "reference substrate",
        [80, 37, 53],
        [1, 2, 2],
        "exclude",
        "exclude_secondary_values_footnoted_to_Juvonen_2013",
    ),
    (
        "PDMS reference",
        "polydimethylsiloxane printed reference surface",
        "flexographically printed PDMS-based ink",
        [114, 92, 96],
        [1, 2, 1],
        "exclude",
        "exclude_secondary_values_footnoted_to_Juvonen_2013",
    ),
]
for surface, material, treatment, angles, uncertainties, status, reason in src016_surfaces:
    for liquid, angle, uncertainty in zip(src016_liquids, angles, uncertainties):
        angle_type = (
            "secondary_reported_static_not_roughness_corrected"
            if surface.startswith("PDMS")
            else (
                "secondary_reported_static"
                if status == "exclude"
                else "roughness_corrected_static_theta_r"
            )
        )
        add(
            status=status,
            source_id="SRC016",
            title=src016_title,
            location=f"Table 2, row '{surface}', {liquid} theta_r column",
            surface=surface,
            material=material,
            treatment=treatment,
            state="ambient RH 15+/-5%, 24+/-1 C",
            liquid=liquid,
            angle=angle,
            angle_type=angle_type,
            uncertainty=uncertainty,
            uncertainty_type="reported_plus_minus_type_not_defined",
            replicate_n=3 if status == "keep" else None,
            method=src016_method,
            volume="1-2 uL",
            license_text=src016_license,
            doi="10.1186/s13568-014-0032-0",
            reason=reason,
            note=(
                "The article title was corrected. PS/PDMS rows are retained only for "
                "audit visibility because footnote b attributes the values to Juvonen et al. (2013)."
            ),
        )


def write_csv() -> None:
    output = ROOT / "records_A.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def write_hashes() -> None:
    output = ROOT / "source_file_hashes_A.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.writer(handle)
        writer.writerow(["file_name", "sha256", "bytes"])
        for path in sorted(RAW.glob("*")):
            if not path.is_file():
                continue
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            writer.writerow([path.name, digest, path.stat().st_size])


def write_source_manifest() -> None:
    manifest_rows = [
        {
            "source_group_id": "SRC001",
            "DOI": "10.1007/s11671-009-9380-0",
            "primary_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC2894349/",
            "supplementary_url": "",
            "local_evidence": "raw_sources/SRC001_fulltext.xml",
            "access_note": "Europe PMC full-text XML retrieved successfully.",
            "retained_rows": 30,
            "excluded_rows": 0,
        },
        {
            "source_group_id": "SRC004",
            "DOI": "10.1038/NMAT3545",
            "primary_url": "https://doi.org/10.1038/NMAT3545",
            "supplementary_url": (
                "https://static-content.springer.com/esm/art%3A10.1038%2F"
                "nmat3545/MediaObjects/41563_2013_BFnmat3545_MOESM8_ESM.pdf"
            ),
            "local_evidence": (
                "raw_sources/SRC004_supplement.pdf; "
                "raw_sources/SRC004_supplement.txt"
            ),
            "access_note": "Official Nature supplementary PDF retrieved successfully.",
            "retained_rows": 45,
            "excluded_rows": 0,
        },
        {
            "source_group_id": "SRC012",
            "DOI": "10.1039/C4TA06944G",
            "primary_url": (
                "https://pubs.rsc.org/en/content/getauthorversionpdf/C4TA06944G"
            ),
            "supplementary_url": (
                "https://www.rsc.org/suppdata/ta/c4/c4ta06944g/c4ta06944g6.pdf"
            ),
            "local_evidence": (
                "Official searchable RSC PDF render inspected online; local automated "
                "requests were blocked/returned 404 and are retained as diagnostic HTML."
            ),
            "access_note": (
                "Values came from official author-version Table 1; measurement details "
                "came from official ESI section 3."
            ),
            "retained_rows": 8,
            "excluded_rows": 0,
        },
        {
            "source_group_id": "SRC014",
            "DOI": "10.1039/D2RA08165B",
            "primary_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9993463/",
            "supplementary_url": "",
            "local_evidence": "raw_sources/SRC014_fulltext.xml",
            "access_note": "Europe PMC full-text XML retrieved successfully.",
            "retained_rows": 27,
            "excluded_rows": 0,
        },
        {
            "source_group_id": "SRC016",
            "DOI": "10.1186/s13568-014-0032-0",
            "primary_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4077624/",
            "supplementary_url": "",
            "local_evidence": "raw_sources/SRC016_fulltext.xml",
            "access_note": "Europe PMC full-text XML retrieved successfully.",
            "retained_rows": 12,
            "excluded_rows": 6,
        },
    ]
    output = ROOT / "source_manifest_A.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)


def write_audit() -> None:
    by_source = Counter(row["source_group_id"] for row in rows)
    kept_by_source = Counter(
        row["source_group_id"] for row in rows if row["record_status"] == "keep"
    )
    excluded_by_source = Counter(
        row["source_group_id"] for row in rows if row["record_status"] == "exclude"
    )
    kept = sum(1 for row in rows if row["record_status"] == "keep")
    excluded = len(rows) - kept

    audit = f"""# v4.3 Legacy Re-extraction Group A Audit

## Scope and Rules

- Sources: `SRC001`, `SRC004`, `SRC012`, `SRC014`, and `SRC016`.
- Existing measurement values in directory `63_...` were not used as numeric input.
- A record was retained only when the value, surface, liquid, and angle definition were visible in an original article table or supplement.
- A reported `+/-` value was not silently relabeled as a standard deviation. `std_deg` is populated only when the article explicitly labels the column `SD`.
- This directory is independent and does not modify the existing v4.0/v4.1 data or models.

## Outcome

- Total audit rows: **{len(rows)}**
- Candidate retained rows: **{kept}**
- Explicitly excluded numeric rows: **{excluded}**
- Source-level exclusions: **0** (all five sources had at least one directly traceable table).

| source | audit rows | retained | excluded | primary location |
|---|---:|---:|---:|---|
| SRC001 | {by_source['SRC001']} | {kept_by_source['SRC001']} | {excluded_by_source['SRC001']} | Article Tables 2-3 |
| SRC004 | {by_source['SRC004']} | {kept_by_source['SRC004']} | {excluded_by_source['SRC004']} | Supplementary Table S2 |
| SRC012 | {by_source['SRC012']} | {kept_by_source['SRC012']} | {excluded_by_source['SRC012']} | Main-text Table 1 |
| SRC014 | {by_source['SRC014']} | {kept_by_source['SRC014']} | {excluded_by_source['SRC014']} | Article Tables 2 and 7 |
| SRC016 | {by_source['SRC016']} | {kept_by_source['SRC016']} | {excluded_by_source['SRC016']} | Article Table 2 |

## Source Decisions

### SRC001

- **Decision:** retain 30 primary table values.
- **Correction:** these are packed-powder capillary-rise/Washburn contact angles, not sessile-drop static angles.
- **Correction:** the table treatments are `as prepared`, `UV illuminated`, and `E-beam irradiated`; the legacy `100 h/200 h` labels are unsupported and were not reused.
- **Traceability:** diamond is in Table 2 and ZnO is in Table 3. Each has five probe liquids and a reported `+/-` value of unspecified type.
- **Method:** n-hexane calibration of the effective capillary radius, followed by the Washburn equation.
- **License note:** the retrieved XML states copyright to the authors; the PMC OA API reports `license=none`.

### SRC004

- **Decision:** retain all 45 advancing-angle values in Supplementary Table S2.
- **Correction:** every imported value is `theta_adv`; none is labeled static.
- **Correction:** the table formula is `Tb7O12`, not the legacy `Tb4O7`.
- **Exclusion:** the water contact-angle-hysteresis column was not imported because hysteresis is not a contact-angle target.
- **Method:** Rame-Hart M500 goniometer. Sintered pellets were mirror polished, cleaned, and vacuum desiccated.
- **License note:** the supplement states Macmillan copyright and all rights reserved.

### SRC012

- **Decision:** retain the eight static organic-liquid angles from main-text Table 1.
- **Surfaces:** the table directly compares a flat fluorinated ITO surface with a porous fluorinated silica-coated ITO surface made using a PEDOT template charge of 47.7 mC/cm2.
- **Method:** OCA 20 sessile-drop system, 5 uL organic droplets, mean of three positions.
- **Conservative exclusion:** water angles and durability-test angles appearing in prose/figure captions were not merged into Table 1 surfaces because they refer to different template charges, controls, or post-treatment states.
- **License note:** the author version and ESI are accessible, but the files do not state a Creative Commons license.

### SRC014

- **Decision:** retain 27 primary measurements from Tables 2 and 7.
- **Traceability:** each row contains the article's average, explicitly labeled SD, and sample count `n`.
- **Method:** DropMaster 700 at 20 C on dried polymer films formed on glass.
- **Duplicate control:** this re-extraction is the sole canonical version for DOI `10.1039/D2RA08165B`; legacy copies under both `SRC014` and `OPEN_CELLULOSE_ESTER` must not coexist in a split.
- **Exclusion:** captive-bubble Table 12 was not mixed with the sessile-drop task.
- **License note:** the PMC OA API identifies this article as CC BY.

### SRC016

- **Decision:** retain 12 primary values for Latex 1, Latex 2, Kaolin, and PCC coated-paper substrates.
- **Correction:** the article title is **Printed Paper-Based Arrays as Substrates for Biofilm Formation**.
- **Angle definition:** Table 2 reports Wenzel roughness-corrected `theta_r`; the underlying apparent static angles were measured after stabilization with three 1-2 uL droplets.
- **Explicit exclusion:** six PS/PDMS values are present in `records_A.csv` with `record_status=exclude` because Table 2 footnote `b` attributes them to Juvonen et al. (2013), not to primary measurements in SRC016. PDMS is additionally footnoted as not corrected for roughness.
- **License note:** the XML states copyright to the authors/licensee Springer; the PMC OA API reports `license=none`.

## Integration Requirements

1. Never concatenate these rows with the legacy rows for the same source.
2. Remove the duplicate `OPEN_CELLULOSE_ESTER` copy before any source-group split.
3. Preserve `contact_angle_type`; do not mix capillary-rise, advancing, roughness-corrected, and sessile-drop values without an explicit task definition.
4. Rebuild `surface_group_id` from source, material, treatment, and state after all re-extraction groups are complete.
5. Re-run split leakage checks before model fitting.

## Files

- `records_A.csv`: row-level values and decisions.
- `source_manifest_A.csv`: DOI, official source URLs, evidence files, and counts.
- `source_file_hashes_A.csv`: hashes for the retrieved XML/PDF/text evidence cache.
- `raw_sources/`: local evidence cache used during extraction.
"""
    (ROOT / "audit_A.md").write_text(audit, encoding="utf-8")


def validate_and_write_qa() -> None:
    expected = {
        "SRC001": (30, 30, 0),
        "SRC004": (45, 45, 0),
        "SRC012": (8, 8, 0),
        "SRC014": (27, 27, 0),
        "SRC016": (18, 12, 6),
    }
    required = [
        "source_group_id",
        "original_table_location",
        "surface_label",
        "material",
        "treatment",
        "state",
        "liquid",
        "contact_angle_deg",
        "contact_angle_type",
        "measurement_method",
        "license",
        "DOI",
        "keep_reason",
    ]
    checks: list[str] = []

    assert len(rows) == 128
    checks.append("PASS total_rows=128")
    for source, (total, kept, excluded) in expected.items():
        source_rows = [row for row in rows if row["source_group_id"] == source]
        assert len(source_rows) == total
        assert sum(row["record_status"] == "keep" for row in source_rows) == kept
        assert sum(row["record_status"] == "exclude" for row in source_rows) == excluded
        checks.append(
            f"PASS {source} total={total} keep={kept} exclude={excluded}"
        )

    for field in required:
        assert all(str(row[field]).strip() for row in rows)
    checks.append("PASS required_fields_nonempty")

    assert all(0.0 <= float(row["contact_angle_deg"]) <= 180.0 for row in rows)
    checks.append("PASS contact_angle_range_0_to_180")

    keys = [
        (row["source_group_id"], row["surface_label"], row["liquid"])
        for row in rows
    ]
    assert len(keys) == len(set(keys))
    checks.append("PASS no_duplicate_source_surface_liquid")

    assert all(
        row["contact_angle_type"] == "advancing"
        for row in rows
        if row["source_group_id"] == "SRC004"
    )
    checks.append("PASS SRC004_all_advancing")

    assert all(
        row["std_deg"] != ""
        for row in rows
        if row["source_group_id"] == "SRC014"
    )
    assert all(
        row["std_deg"] == ""
        for row in rows
        if row["source_group_id"] != "SRC014"
    )
    checks.append("PASS std_only_when_explicitly_labeled")

    assert not any(
        "100 h" in str(row["treatment"]) or "200 h" in str(row["treatment"])
        for row in rows
        if row["source_group_id"] == "SRC001"
    )
    checks.append("PASS SRC001_unsupported_duration_labels_absent")

    excluded_016 = [
        row
        for row in rows
        if row["source_group_id"] == "SRC016"
        and row["record_status"] == "exclude"
    ]
    assert len(excluded_016) == 6
    assert {row["surface_label"] for row in excluded_016} == {
        "PS (96-well) reference",
        "PDMS reference",
    }
    checks.append("PASS SRC016_secondary_rows_explicitly_excluded")

    assert all(
        "OPEN_CELLULOSE_ESTER" in str(row["keep_reason"])
        for row in rows
        if row["source_group_id"] == "SRC014"
    )
    checks.append("PASS SRC014_duplicate_control_marked")

    (ROOT / "qa_A.txt").write_text("\n".join(checks) + "\n", encoding="utf-8")


if __name__ == "__main__":
    validate_and_write_qa()
    write_csv()
    write_source_manifest()
    write_hashes()
    write_audit()
    print(f"wrote {len(rows)} rows: {sum(r['record_status'] == 'keep' for r in rows)} keep, "
          f"{sum(r['record_status'] == 'exclude' for r in rows)} exclude")
