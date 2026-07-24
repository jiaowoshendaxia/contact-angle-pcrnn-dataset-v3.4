# v4.3 Legacy Re-extraction Audit - Group B

## Scope and decision rule

This package re-extracts only `SRC018`, `SRC019`, `SRC020`, and `SRC021`. No value from the legacy v4 tables was copied into `records_B.csv`. Values were transcribed from the original article tables and checked against rendered source pages. A reported inequality is stored as a censored observation, never as an invented point value.

## Output summary

| Source | Total rows | Numeric rows | Censored rows | Primary evidence |
|---|---:|---:|---:|---|
| SRC018 | 20 | 20 | 0 | Coatings 2021, Table 2, PDF p. 8 |
| SRC019 | 60 | 60 | 0 | IJMS 2021, Table 1, PDF p. 4 |
| SRC020 | 24 | 15 | 9 | Metals 2022, Table 4, PDF p. 6 |
| SRC021 | 32 | 30 | 2 | Nanomaterials 2021, Tables 4-5, PDF p. 8 |
| **Total** | **136** | **125** | **11** | |

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
