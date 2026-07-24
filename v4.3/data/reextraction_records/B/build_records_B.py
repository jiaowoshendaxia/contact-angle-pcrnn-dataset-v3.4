from __future__ import annotations

import csv
from pathlib import Path


OUT_DIR = Path(__file__).resolve().parent

FIELDS = [
    "source_group_id",
    "source_title",
    "title_correction",
    "original_table_location",
    "surface_label",
    "material",
    "treatment",
    "state",
    "liquid",
    "contact_angle_deg",
    "reported_value_text",
    "censor_type",
    "contact_angle_type",
    "contact_angle_std_deg",
    "contact_angle_range",
    "replicate_n",
    "measurement_method",
    "droplet_volume_uL",
    "license",
    "DOI",
    "keep_reason",
    "exclusion_reason",
]


def base_record(
    *,
    source_group_id: str,
    source_title: str,
    title_correction: str,
    original_table_location: str,
    surface_label: str,
    material: str,
    treatment: str,
    state: str,
    liquid: str,
    angle: float | None,
    reported: str,
    censor_type: str,
    angle_type: str,
    std: float | None,
    replicate_n: str,
    method: str,
    volume: str,
    doi: str,
    keep_reason: str,
    exclusion_reason: str = "",
) -> dict[str, str]:
    return {
        "source_group_id": source_group_id,
        "source_title": source_title,
        "title_correction": title_correction,
        "original_table_location": original_table_location,
        "surface_label": surface_label,
        "material": material,
        "treatment": treatment,
        "state": state,
        "liquid": liquid,
        "contact_angle_deg": "" if angle is None else f"{angle:g}",
        "reported_value_text": reported,
        "censor_type": censor_type,
        "contact_angle_type": angle_type,
        "contact_angle_std_deg": "" if std is None else f"{std:g}",
        "contact_angle_range": "",
        "replicate_n": replicate_n,
        "measurement_method": method,
        "droplet_volume_uL": volume,
        "license": "CC BY 4.0",
        "DOI": doi,
        "keep_reason": keep_reason,
        "exclusion_reason": exclusion_reason,
    }


def add_src018(records: list[dict[str, str]]) -> None:
    title = (
        "Robust Superhydrophobic and Repellent Coatings Based on Micro/Nano "
        "SiO2 and Fluorinated Epoxy"
    )
    title_correction = (
        "Corrected from the unrelated local title referring to FEP/SiO2 "
        "oil/water-separation coatings."
    )
    values = {
        "micro:nano = 1:0": {
            "water": (123.4, 2.3),
            "glycerol": (105.8, 1.4),
            "ethylene glycol": (127.3, 1.1),
            "diiodomethane": (113.6, 1.3),
        },
        "micro:nano = 2:1": {
            "water": (135.3, 0.8),
            "glycerol": (110.4, 1.3),
            "ethylene glycol": (128.6, 1.5),
            "diiodomethane": (118.5, 2.1),
        },
        "micro:nano = 1:1 (FEP-S)": {
            "water": (158.6, 1.1),
            "glycerol": (140.7, 0.9),
            "ethylene glycol": (152.4, 0.9),
            "diiodomethane": (153.4, 1.3),
        },
        "micro:nano = 1:2": {
            "water": (125.2, 1.5),
            "glycerol": (120.5, 1.5),
            "ethylene glycol": (132.7, 1.4),
            "diiodomethane": (120.8, 1.8),
        },
        "micro:nano = 0:1": {
            "water": (120.7, 2.7),
            "glycerol": (114.4, 1.8),
            "ethylene glycol": (118.5, 1.7),
            "diiodomethane": (108.5, 2.4),
        },
    }
    for ratio, liquid_values in values.items():
        ratio_only = ratio.split(" (")[0].split("=")[1].strip()
        for liquid, (angle, std) in liquid_values.items():
            records.append(
                base_record(
                    source_group_id="SRC018",
                    source_title=title,
                    title_correction=title_correction,
                    original_table_location=(
                        "Table 2, PDF page 8; measurement method in Section 2.4, "
                        "PDF page 5"
                    ),
                    surface_label=f"fluorinated epoxy coating, {ratio}",
                    material=(
                        "fluorinated epoxy with micro/nano SiO2, spray-coated "
                        "on a glass slide"
                    ),
                    treatment=(
                        "total SiO2 = 10 wt% of fluorinated epoxy; "
                        f"micro:nano SiO2 ratio {ratio_only}; heated at 80 C for "
                        "5 min and cured at 120 C for 2 h"
                    ),
                    state="as-prepared rough micro/nanostructured coating",
                    liquid=liquid,
                    angle=angle,
                    reported=f"{angle:g} +/- {std:g}",
                    censor_type="none",
                    angle_type=(
                        "reported contact angle; static/advancing/receding mode "
                        "not specified"
                    ),
                    std=std,
                    replicate_n="5 surface positions",
                    method=(
                        "JC 2000D2 contact-angle tester at 25 C; mean of five "
                        "positions"
                    ),
                    volume="not reported",
                    doi="10.3390/coatings11060663",
                    keep_reason=(
                        "Direct Table 2 extraction. The abstract assigns the four "
                        "FEP-S values to liquids in a conflicting order; Table 2 "
                        "is retained as the row/column source. Main-analysis use "
                        "remains conditional on the angle-type policy."
                    ),
                )
            )


def add_src019(records: list[dict[str, str]]) -> None:
    title = (
        "Plant-Origin Stabilizer as an Alternative of Natural Additive to "
        "Polymers Used in Packaging Materials"
    )
    values = {
        "Topas": {
            "water": [(87.6, 0.79), (90.6, 6.61), (98.2, 1.01), (97.8, 1.73), (97.6, 0.97)],
            "diiodomethane": [(61.6, 1.43), (59.9, 0.74), (55.4, 1.89), (62.5, 1.73), (53.8, 1.68)],
            "ethylene glycol": [(63.3, 1.72), (65.4, 1.79), (74.9, 1.04), (75.0, 0.66), (69.5, 2.05)],
        },
        "Topas/CBD": {
            "water": [(71.2, 0.92), (97.3, 1.08), (85.9, 1.08), (92.9, 1.04), (97.1, 0.99)],
            "diiodomethane": [(55.4, 0.95), (51.1, 0.74), (47.0, 0.56), (48.6, 0.96), (49.5, 1.28)],
            "ethylene glycol": [(48.1, 2.29), (71.9, 1.88), (58.0, 1.32), (65.9, 1.17), (69.4, 1.09)],
        },
        "PLA": {
            "water": [(82.1, 0.84), (86.1, 0.54), (76.0, 0.95), (75.0, 1.25), (65.1, 0.73)],
            "diiodomethane": [(54.2, 1.17), (51.6, 1.74), (44.0, 1.32), (46.9, 2.50), (41.3, 3.50)],
            "ethylene glycol": [(57.8, 1.47), (61.2, 1.45), (49.5, 0.85), (48.0, 1.10), (42.7, 1.65)],
        },
        "PLA/CBD": {
            "water": [(87.8, 1.44), (78.9, 1.62), (76.8, 1.76), (71.6, 1.01), (67.9, 2.04)],
            "diiodomethane": [(47.6, 2.23), (46.8, 1.46), (51.4, 2.68), (40.3, 1.23), (33.8, 2.43)],
            "ethylene glycol": [(58.9, 1.34), (51.1, 1.28), (48.2, 0.84), (50.3, 1.84), (40.7, 3.32)],
        },
    }
    times = [0, 100, 200, 300, 400]
    for composition, liquid_values in values.items():
        for liquid, sequence in liquid_values.items():
            for hours, (angle, std) in zip(times, sequence, strict=True):
                polymer = "ethylene-norbornene copolymer" if composition.startswith("Topas") else "polylactide"
                additive = " with 1 phr cannabidiol extract" if "/CBD" in composition else ""
                state = "reference, before weather aging" if hours == 0 else f"after {hours} h weather aging"
                records.append(
                    base_record(
                        source_group_id="SRC019",
                        source_title=title,
                        title_correction="No title correction required.",
                        original_table_location=(
                            "Table 1, PDF page 4; measurement method in Section "
                            "3.4.1, PDF page 13"
                        ),
                        surface_label=f"{composition}, {state}",
                        material=f"{polymer}{additive}",
                        treatment=(
                            "controlled weather aging; reference/0 h, 100 h, "
                            "200 h, 300 h, or 400 h as encoded in the state"
                        ),
                        state=state,
                        liquid=liquid,
                        angle=angle,
                        reported=f"{angle:g} +/- {std:g}",
                        censor_type="none",
                        angle_type=(
                            "reported contact angle; static/advancing/receding "
                            "mode not specified"
                        ),
                        std=std,
                        replicate_n="10 contact-angle determinations",
                        method=(
                            "DataPhysics OCA 15EC goniometer with SCA 20 "
                            "software; drop volume and fitting details not reported"
                        ),
                        volume="not reported",
                        doi="10.3390/ijms22084012",
                        keep_reason=(
                            "Direct Table 1 extraction with mean, standard "
                            "deviation, composition, and aging time. Main-analysis "
                            "use remains conditional on the angle-type policy."
                        ),
                    )
                )


def add_src020(records: list[dict[str, str]]) -> None:
    title = (
        "The Effect of Ultraviolet Treatment on TiO2 Nanotubes: A Study of "
        "Surface Characteristics, Bacterial Adhesion, and Gingival Fibroblast Response"
    )
    title_correction = (
        "Corrected from the unrelated local title about anodizing treatment in "
        "various electrolytes."
    )
    values = {
        ("Grade I Ti control", "before UV"): {
            "water": (89.2, 2.1),
            "diiodomethane": (52.1, 3.4),
            "formamide": (57.9, 8.0),
        },
        ("Grade I Ti control", "after UV"): {
            "water": (83.0, 1.7),
            "diiodomethane": (48.1, 1.6),
            "formamide": (59.2, 4.5),
        },
        ("TiO2 nanotube 10 V", "before UV"): {
            "water": (16.2, 2.1),
            "diiodomethane": (9.4, 1.4),
            "formamide": (11.3, 0.6),
        },
        ("TiO2 nanotube 15 V", "before UV"): {
            "water": (22.4, 2.6),
            "diiodomethane": (12.7, 1.2),
            "formamide": (17.5, 1.7),
        },
        ("TiO2 nanotube 20 V", "before UV"): {
            "water": (16.0, 2.0),
            "diiodomethane": (8.9, 1.9),
            "formamide": (12.8, 3.1),
        },
    }
    for (surface, state), liquid_values in values.items():
        for liquid, (angle, std) in liquid_values.items():
            records.append(
                src020_record(
                    title=title,
                    title_correction=title_correction,
                    surface=surface,
                    state=state,
                    liquid=liquid,
                    angle=angle,
                    std=std,
                    reported=f"{angle:g} +/- {std:g}",
                    censor_type="none",
                    keep_reason="Direct numeric extraction from Table 4.",
                )
            )

    for voltage in (10, 15, 20):
        surface = f"TiO2 nanotube {voltage} V"
        for liquid in ("water", "diiodomethane", "formamide"):
            records.append(
                src020_record(
                    title=title,
                    title_correction=title_correction,
                    surface=surface,
                    state="after UV",
                    liquid=liquid,
                    angle=None,
                    std=None,
                    reported="<0.1",
                    censor_type="left_censored_lt_0.1_deg",
                    keep_reason=(
                        "The source reports a left-censored value (<0.1 deg). "
                        "The text is retained without conversion to a point value."
                    ),
                    exclusion_reason=(
                        "Exclude from point-valued model training unless the "
                        "ingestion and loss are explicitly censor-aware."
                    ),
                )
            )


def src020_record(
    *,
    title: str,
    title_correction: str,
    surface: str,
    state: str,
    liquid: str,
    angle: float | None,
    std: float | None,
    reported: str,
    censor_type: str,
    keep_reason: str,
    exclusion_reason: str = "",
) -> dict[str, str]:
    if surface == "Grade I Ti control":
        material = "Grade I commercially pure titanium"
        preparation = "cleaned Ti control; no anodization"
    else:
        voltage = surface.split()[-2]
        material = "TiO2 nanotube array on Grade I titanium"
        preparation = (
            "anodized at "
            f"{voltage} V for 40 min in 0.5 vol% HF in deionized water at 22 C"
        )
    uv = (
        "no UV treatment"
        if state == "before UV"
        else "UV-A 300-400 nm, 15 cm distance, 24 h"
    )
    return base_record(
        source_group_id="SRC020",
        source_title=title,
        title_correction=title_correction,
        original_table_location=(
            "Table 4, PDF page 6; specimen preparation in Section 2.1 and "
            "measurement method in Section 2.4, PDF page 3"
        ),
        surface_label=f"{surface}, {state}",
        material=material,
        treatment=f"{preparation}; {uv}",
        state=state,
        liquid=liquid,
        angle=angle,
        reported=reported,
        censor_type=censor_type,
        angle_type="equilibrium contact angle",
        std=std,
        replicate_n="mean of at least 6 drops per specimen",
        method=(
            "sessile drop, KSVCAM100; images every 2 s for 20 s; "
            "Young-Laplace fit and mean of both droplet sides"
        ),
        volume="not reported",
        doi="10.3390/met12010080",
        keep_reason=keep_reason,
        exclusion_reason=exclusion_reason,
    )


def add_src021(records: list[dict[str, str]]) -> None:
    title = (
        "Hydrophobic/Oleophilic Structures Based on MacroPorous Silicon: "
        "Effect of Topography and Fluoroalkyl Silane Functionalization on Wettability"
    )
    sample_meta = {
        "MacroPSi-1": ("6.2 ohm cm", "10 mA/cm2", "74.9 +/- 6.5%"),
        "MacroPSi-2": ("6.2 ohm cm", "20 mA/cm2", "86.2 +/- 6.2%"),
        "MacroPSi-3": ("10 ohm cm", "10 mA/cm2", "30.2 +/- 1.4%"),
        "MacroPSi-4": ("10 ohm cm", "20 mA/cm2", "27.5 +/- 0.4%"),
    }
    table4 = {
        "MacroPSi-1": {
            "water": (148.4, 5.8),
            "ethylene glycol": (13.7, 2.8),
            "diiodomethane": ("<10", None),
            "formamide": (88.0, 0.7),
        },
        "MacroPSi-2": {
            "water": (157.0, 2.5),
            "ethylene glycol": (14.2, 5.0),
            "diiodomethane": ("<10", None),
            "formamide": (94.5, 4.0),
        },
        "MacroPSi-3": {
            "water": (129.3, 4.0),
            "ethylene glycol": (59.1, 1.8),
            "diiodomethane": (30.0, 4.3),
            "formamide": (63.5, 9.7),
        },
        "MacroPSi-4": {
            "water": (117.9, 4.3),
            "ethylene glycol": (61.6, 3.0),
            "diiodomethane": (55.5, 1.0),
            "formamide": (82.2, 7.0),
        },
    }
    table5 = {
        "MacroPSi-1": {
            "water": (158.5, 2.1),
            "ethylene glycol": (136.1, 4.7),
            "diiodomethane": (148.7, 8.1),
            "formamide": (146.2, 3.9),
        },
        "MacroPSi-2": {
            "water": (163.2, 2.9),
            "ethylene glycol": (142.4, 5.1),
            "diiodomethane": (152.3, 8.6),
            "formamide": (145.3, 1.5),
        },
        "MacroPSi-3": {
            "water": (148.0, 4.2),
            "ethylene glycol": (129.7, 2.2),
            "diiodomethane": (107.7, 6.3),
            "formamide": (119.0, 7.8),
        },
        "MacroPSi-4": {
            "water": (147.0, 3.0),
            "ethylene glycol": (122.6, 3.9),
            "diiodomethane": (97.1, 1.5),
            "formamide": (124.1, 4.2),
        },
    }
    add_src021_table(records, title, sample_meta, table4, fots=False)
    add_src021_table(records, title, sample_meta, table5, fots=True)


def add_src021_table(
    records: list[dict[str, str]],
    title: str,
    sample_meta: dict[str, tuple[str, str, str]],
    values: dict[str, dict[str, tuple[float | str, float | None]]],
    *,
    fots: bool,
) -> None:
    table = "Table 5" if fots else "Table 4"
    state = "FOTS-functionalized" if fots else "as-fabricated"
    for sample, liquid_values in values.items():
        resistivity, current, porosity = sample_meta[sample]
        base_treatment = (
            "p-type Si anodized for 90 min in HF:DMF 1:3; "
            f"wafer resistivity {resistivity}; current density {current}; "
            f"surface porosity {porosity}"
        )
        if fots:
            base_treatment += (
                "; FOTS physical-vapor deposition for 1 h under low pressure "
                "at room temperature, followed by 120 C annealing for 2 h"
            )
        for liquid, (value, std) in liquid_values.items():
            censored = isinstance(value, str)
            angle = None if censored else float(value)
            reported = value if censored else f"{float(value):g} +/- {float(std):g}"
            records.append(
                base_record(
                    source_group_id="SRC021",
                    source_title=title,
                    title_correction="No title correction required.",
                    original_table_location=(
                        f"{table}, PDF page 8; sample metadata in Table 1, PDF "
                        "page 4; measurement method in Section 2.4, PDF page 3"
                    ),
                    surface_label=f"{sample}, {state}",
                    material="macroporous p-type silicon",
                    treatment=base_treatment,
                    state=state,
                    liquid=liquid,
                    angle=angle,
                    reported=reported,
                    censor_type="left_censored_lt_10_deg" if censored else "none",
                    angle_type=(
                        "reported apparent contact angle; dynamic mode not specified"
                    ),
                    std=std,
                    replicate_n="9 measurements at 3 positions",
                    method=(
                        "drop-shape analysis under ambient conditions; direct "
                        "tangent-angle measurement"
                    ),
                    volume="approximately 3",
                    doi="10.3390/nano11030670",
                    keep_reason=(
                        "The source reports a left-censored value (<10 deg); "
                        "retained as text without conversion to 10.0 or 10.1."
                        if censored
                        else f"Direct numeric extraction from {table}."
                    ),
                    exclusion_reason=(
                        "Exclude from point-valued model training unless the "
                        "ingestion and loss are explicitly censor-aware."
                        if censored
                        else ""
                    ),
                )
            )


def write_audit(records: list[dict[str, str]]) -> None:
    counts: dict[str, dict[str, int]] = {}
    for row in records:
        item = counts.setdefault(row["source_group_id"], {"total": 0, "numeric": 0, "censored": 0})
        item["total"] += 1
        if row["censor_type"] == "none":
            item["numeric"] += 1
        else:
            item["censored"] += 1

    audit = f"""# v4.3 Legacy Re-extraction Audit - Group B

## Scope and decision rule

This package re-extracts only `SRC018`, `SRC019`, `SRC020`, and `SRC021`. No value from the legacy v4 tables was copied into `records_B.csv`. Values were transcribed from the original article tables and checked against rendered source pages. A reported inequality is stored as a censored observation, never as an invented point value.

## Output summary

| Source | Total rows | Numeric rows | Censored rows | Primary evidence |
|---|---:|---:|---:|---|
| SRC018 | {counts['SRC018']['total']} | {counts['SRC018']['numeric']} | {counts['SRC018']['censored']} | Coatings 2021, Table 2, PDF p. 8 |
| SRC019 | {counts['SRC019']['total']} | {counts['SRC019']['numeric']} | {counts['SRC019']['censored']} | IJMS 2021, Table 1, PDF p. 4 |
| SRC020 | {counts['SRC020']['total']} | {counts['SRC020']['numeric']} | {counts['SRC020']['censored']} | Metals 2022, Table 4, PDF p. 6 |
| SRC021 | {counts['SRC021']['total']} | {counts['SRC021']['numeric']} | {counts['SRC021']['censored']} | Nanomaterials 2021, Tables 4-5, PDF p. 8 |
| **Total** | **{len(records)}** | **{sum(x['numeric'] for x in counts.values())}** | **{sum(x['censored'] for x in counts.values())}** | |

## Source-by-source audit

### SRC018

- Correct title: *Robust Superhydrophobic and Repellent Coatings Based on Micro/Nano SiO2 and Fluorinated Epoxy*.
- DOI and license: `10.3390/coatings11060663`, CC BY 4.0.
- Evidence: Table 2 on PDF page 8; contact-angle method in Section 2.4.
- Extracted: five micro:nano-SiO2 ratios x four liquids = 20 numeric records.
- Important internal conflict: the article abstract maps the four FEP-S values to liquids in an order that conflicts with Table 2. `records_B.csv` uses Table 2 because it is the explicit row/column table. The abstract values were not merged or guessed.
- Angle type: the paper reports contact angles measured with a JC 2000D2 instrument at 25 C, but does not classify them as static, advancing, receding, or equilibrium. The records therefore remain conditional for any strict angle-type main analysis.

### SRC019

- Title: *Plant-Origin Stabilizer as an Alternative of Natural Additive to Polymers Used in Packaging Materials*.
- DOI and license: `10.3390/ijms22084012`, CC BY 4.0.
- Evidence: Table 1 on PDF page 4; instrument and replicate description in Section 3.4.1.
- Extracted: four compositions x five weather-aging states x three liquids = 60 numeric records.
- Replication: each value is the mean of 10 contact-angle determinations.
- Angle type and droplet volume are not reported. The rows are traceable but remain conditional for a strict static-only analysis.

### SRC020

- Correct title: *The Effect of Ultraviolet Treatment on TiO2 Nanotubes: A Study of Surface Characteristics, Bacterial Adhesion, and Gingival Fibroblast Response*.
- DOI and license: `10.3390/met12010080`, CC BY 4.0.
- Evidence: Table 4 on PDF page 6; preparation in Section 2.1 and equilibrium sessile-drop method in Section 2.4.
- Extracted: four surface groups x before/after UV x three liquids = 24 records.
- Nine UV-treated nanotube observations are reported as `<0.1 deg`. They have blank `contact_angle_deg`, preserve the exact report in `reported_value_text`, and are excluded from ordinary point-valued model training unless a censor-aware pipeline is implemented.
- The original local title referred to a different anodizing/electrolyte study and must not be reused.

### SRC021

- Title: *Hydrophobic/Oleophilic Structures Based on MacroPorous Silicon: Effect of Topography and Fluoroalkyl Silane Functionalization on Wettability*.
- DOI and license: `10.3390/nano11030670`, CC BY 4.0.
- Evidence: Tables 4 and 5 on PDF page 8; surface metadata in Table 1; measurement method in Section 2.4.
- Extracted: four untreated MacroPSi surfaces x four liquids plus four FOTS-functionalized surfaces x four liquids = 32 records.
- Two untreated diiodomethane observations are reported as `<10 deg`. They are retained as left-censored values and are not converted to 10.0 or 10.1.
- The initial water values also appear in Table 2. They were not duplicated. The later four-month water-only observations in Table 2 were not included because they do not form a complete multi-probe state and would require a separate longitudinal schema.

## Explicit exclusions and integration notes

1. Do not ingest the old `SRC018`/`SRC019`/`SRC020`/`SRC021` rows from the v4 legacy table.
2. Do not treat `reported_value_text` values beginning with `<` as numeric.
3. Do not label SRC018, SRC019, or SRC021 as `static` without a documented source. Their dynamic angle mode is unspecified.
4. SRC020 is explicitly an equilibrium sessile-drop dataset and can be typed accordingly.
5. For a main analysis restricted to a single contact-angle type, the integrator must either:
   - retain only explicitly compatible records, or
   - define and disclose a broader apparent/equilibrium policy before rebuilding splits.
6. All four source PDFs and rendered evidence pages are stored under `raw/` and `evidence_pages/` for local verification. The CC BY 4.0 source and DOI must remain attached to redistributed rows.

## Reproducibility

Run `python build_records_B.py` in this directory. The script deterministically regenerates `records_B.csv` and this audit summary from hard-coded, visually checked table transcriptions.
"""
    (OUT_DIR / "audit_B.md").write_text(audit, encoding="utf-8")


def main() -> None:
    records: list[dict[str, str]] = []
    add_src018(records)
    add_src019(records)
    add_src020(records)
    add_src021(records)

    with (OUT_DIR / "records_B.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)

    write_audit(records)
    print(f"Wrote {len(records)} rows to {OUT_DIR / 'records_B.csv'}")


if __name__ == "__main__":
    main()
