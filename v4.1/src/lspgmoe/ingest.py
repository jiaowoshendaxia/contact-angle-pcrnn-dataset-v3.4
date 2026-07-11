"""Validation-only importer for reviewed open contact-angle tables.

The importer creates a staged normalized file. It never mutates the five processed
tables or assigns a train/test split; those actions happen only after review.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import (
    LIQUID_ALIASES, V4Tables, build_nnls_audit, build_samples, canonical_liquid_name,
    clean_text, stable_id, validate_v4_tables,
)


REQUIRED_COLUMNS = {
    "solid_name", "solid_family", "liquid_name", "contact_angle_deg",
    "contact_angle_type", "measurement_method",
}
ANGLE_TYPES = {"static", "advancing", "receding"}
REQUIRED_NUMERIC = [
    "contact_angle_deg", "liquid_total_surface_tension_mN_m",
    "liquid_dispersion_mN_m", "liquid_polar_mN_m",
]
SURFACE_KEY_COLUMNS = [
    "solid_name", "solid_family", "solid_substrate", "surface_treatment",
    "surface_treatment_detail", "coating_or_layer", "surface_state",
]


def read_tabular(path: Path) -> pd.DataFrame:
    suffix = path.suffix.casefold()
    if suffix == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported import format: {path.suffix}; use CSV/XLSX/XLS")


def _registry_row(registry_path: Path, source_id: str) -> dict[str, Any]:
    registry = pd.read_csv(registry_path, encoding="utf-8-sig")
    selected = registry.loc[registry["source_id"].astype(str) == str(source_id)]
    if selected.empty:
        raise ValueError(f"source_id {source_id!r} is absent from {registry_path}")
    return {str(key): value for key, value in selected.iloc[0].to_dict().items()}


def _missing_rows(frame: pd.DataFrame, column: str) -> list[int]:
    if column not in frame:
        return []
    missing = frame[column].isna() | frame[column].astype(str).str.strip().eq("")
    return [int(index) + 2 for index in frame.index[missing][:20]]


def validate_open_source_frame(
    frame: pd.DataFrame,
    source_id: str,
    registry: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    errors: list[str] = []
    warnings: list[str] = []
    missing_columns = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing_columns:
        errors.append("missing_required_columns:" + ",".join(missing_columns))
        return frame.copy(), {
            "status": "fail", "source_id": source_id, "n_rows": int(len(frame)),
            "errors": errors, "warnings": warnings,
        }

    staged = frame.copy()
    staged["source_group_id"] = source_id
    for column, registry_key in [("reference_doi", "doi"), ("reference_url", "url")]:
        if column not in staged:
            staged[column] = registry.get(registry_key, "")
        else:
            staged[column] = staged[column].replace({np.nan: ""})
            staged[column] = staged[column].map(clean_text)
            staged.loc[staged[column].eq(""), column] = clean_text(registry.get(registry_key, ""))
    staged["license_verified"] = "yes" if str(registry.get("verification_status", "")).casefold() == "verified" else staged.get("license_verified", "no")
    staged["new_external_source_flag"] = "yes"
    staged["liquid_name"] = staged["liquid_name"].map(canonical_liquid_name)

    for column in ["solid_name", "solid_family", "liquid_name", "contact_angle_type", "measurement_method"]:
        rows = _missing_rows(staged, column)
        if rows:
            errors.append(f"missing_values:{column}:rows={rows}")
    for column in REQUIRED_NUMERIC:
        if column not in staged:
            errors.append(f"missing_required_numeric_column:{column}")
            continue
        staged[column] = pd.to_numeric(staged[column], errors="coerce")
        if staged[column].isna().any():
            errors.append(f"non_numeric_or_missing:{column}")
    if "contact_angle_deg" in staged:
        invalid_angle = ~staged["contact_angle_deg"].between(0.0, 180.0, inclusive="both")
        if invalid_angle.any():
            errors.append(f"angle_out_of_range:rows={[int(i) + 2 for i in staged.index[invalid_angle][:20]]}")
    for column in ["liquid_total_surface_tension_mN_m", "liquid_dispersion_mN_m", "liquid_polar_mN_m"]:
        if column in staged and (staged[column] < 0).any():
            errors.append(f"negative_liquid_property:{column}")
    angle_tokens = staged["contact_angle_type"].map(lambda value: clean_text(value).casefold())
    unknown_angle_type = ~angle_tokens.isin(ANGLE_TYPES)
    if unknown_angle_type.any():
        errors.append(f"unsupported_angle_type:values={sorted(angle_tokens[unknown_angle_type].unique().tolist())}")
    if str(registry.get("verification_status", "")).casefold() != "verified":
        errors.append("registry_license_or_access_not_verified")
    if not clean_text(registry.get("doi")) and not clean_text(registry.get("url")):
        errors.append("registry_has_no_doi_or_url")

    surface_columns = [column for column in SURFACE_KEY_COLUMNS if column in staged]
    key_columns = surface_columns + [
        column for column in ["liquid_name", "contact_angle_type"] if column in staged
    ]
    duplicate_count = int(staged.duplicated(subset=key_columns, keep=False).sum()) if key_columns else 0
    if duplicate_count:
        warnings.append(f"duplicate_surface_liquid_angle_type_rows:{duplicate_count}")
    surface_keys = staged[surface_columns].fillna("").astype(str).agg("|".join, axis=1)
    probe_counts = staged.assign(_surface_key=surface_keys).groupby("_surface_key")["liquid_name"].nunique()
    surface_count = int(surface_keys.nunique())
    feasible_surface_count = int((probe_counts >= 3).sum())
    staged["surface_group_id"] = [
        stable_id("SFC", [source_id, *[row.get(column, "") for column in SURFACE_KEY_COLUMNS]])
        for _, row in staged.iterrows()
    ]
    staged["liquid_id"] = staged["liquid_name"].map(lambda value: stable_id("LIQ", [value], length=10))
    staged["source_row_id"] = [stable_id("RAW", [source_id, index + 2]) for index in range(len(staged))]
    staged["applsci_split"] = "pending_review"
    staged["import_status"] = "validated_staged" if not errors else "rejected"
    report = {
        "status": "pass" if not errors else "fail",
        "source_id": source_id,
        "n_rows": int(len(staged)),
        "n_surfaces": surface_count,
        "n_liquids": int(staged["liquid_name"].nunique()),
        "n_surfaces_with_at_least_3_liquids": feasible_surface_count,
        "duplicate_key_rows": duplicate_count,
        "errors": errors,
        "warnings": warnings,
        "registry_license_or_access": registry.get("license_or_access", ""),
    }
    return staged, report


def validate_and_stage(
    input_path: Path,
    source_id: str,
    registry_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    registry = _registry_row(registry_path, source_id)
    frame = read_tabular(input_path)
    staged, report = validate_open_source_frame(frame, source_id, registry)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"{source_id}_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if report["status"] == "pass":
        staged_path = output_dir / f"{source_id}_staged.csv"
        staged.to_csv(staged_path, index=False, encoding="utf-8-sig")
        report["staged_path"] = str(staged_path)
    return report


def extract_cellulose_ester_jats(xml_path: Path, output_path: Path, source_id: str) -> dict[str, Any]:
    """Extract the contact-angle table from the JATS source into a review candidate."""
    root = ET.parse(xml_path).getroot()
    selected = None
    for table in root.findall(".//table-wrap"):
        caption = table.find("caption")
        title = " ".join("".join(caption.itertext()).split()) if caption is not None else ""
        if "contact angles" in title.casefold() and "average" in title.casefold():
            selected = (table, title)
            break
    if selected is None:
        raise ValueError("No average contact-angle table was found in the JATS XML")
    table, title = selected
    rows: list[dict[str, Any]] = []
    liquids = [
        ("formamide", 58.0, 39.0, 19.0),
        ("diiodomethane", 50.8, 50.8, 0.0),
        ("water", 72.8, 21.8, 51.0),
    ]
    for tr in table.findall(".//tr"):
        cells = [" ".join("".join(cell.itertext()).split()) for cell in tr.findall("./th") + tr.findall("./td")]
        if len(cells) < 11 or cells[0].casefold() in {"cellulose esters", "average (deg)"}:
            continue
        name = cells[0]
        if not re.search(r"\d", cells[1]):
            continue
        for liquid_index, (liquid_name, total, dispersion, polar) in enumerate(liquids):
            offset = 2 + liquid_index * 3
            angle_match = re.search(r"\d+(?:\.\d+)?", cells[offset])
            std_match = re.search(r"\d+(?:\.\d+)?", cells[offset + 1])
            if not angle_match or not std_match:
                raise ValueError(f"Could not parse angle/SD in row {cells[0]!r}: {cells[offset:offset + 3]}")
            rows.append({
                "source_group_id": source_id,
                "reference_doi": "10.1039/d2ra08165b",
                "reference_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9993463/",
                "solid_name": name,
                "solid_family": "cellulose_ester",
                "solid_substrate": "glass-supported film",
                "surface_treatment": "ester substitution",
                "surface_treatment_detail": f"cellulose ester; substituent carbon number {cells[1]}",
                "surface_state": "dried film",
                "liquid_name": liquid_name,
                "liquid_total_surface_tension_mN_m": total,
                "liquid_dispersion_mN_m": dispersion,
                "liquid_polar_mN_m": polar,
                "contact_angle_deg": float(angle_match.group()),
                "contact_angle_type": "static",
                "measurement_method": "sessile_drop",
                "temperature_K": 293.15,
                "replicates_n": int(float(cells[offset + 2])),
                "contact_angle_std_deg": float(std_match.group()),
                "solid_surface_energy_source_type": "not_reported",
                "table_or_figure_locator": f"JATS table {table.attrib.get('id', '')}; {title}",
                "extraction_note": "Candidate extracted from the article's average/SD/n contact-angle table; license review required.",
                "license_verified": "no",
                "new_external_source_flag": "yes",
            })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidate = pd.DataFrame(rows)
    candidate.to_csv(output_path, index=False, encoding="utf-8-sig")
    return {
        "status": "candidate_extracted",
        "source_id": source_id,
        "table_id": table.attrib.get("id", ""),
        "table_title": title,
        "n_rows": len(candidate),
        "n_surfaces": int(candidate["solid_name"].nunique()),
        "n_liquids": int(candidate["liquid_name"].nunique()),
        "output_path": str(output_path),
    }


def extract_cross_material_jats(xml_path: Path, output_path: Path, source_id: str) -> dict[str, Any]:
    """Extract exact/static rows from the cross-material Table 1.

    Range-only, transition, mixture, and unknown-liquid rows are deliberately excluded
    and reported so they cannot silently become fabricated point measurements.
    """
    root = ET.parse(xml_path).getroot()
    table = root.find(".//table-wrap[@id='Tab1']")
    if table is None:
        raise ValueError("JATS Table 1 was not found")
    title = " ".join("".join(table.find("caption").itertext()).split())
    liquid_properties = {
        "water": ("water", 72.8, 21.8, 51.0),
        "distilled water": ("water", 72.8, 21.8, 51.0),
        "ethylene glycol": ("ethylene glycol", 48.0, 29.0, 19.0),
        "formamide": ("formamide", 58.0, 39.0, 19.0),
        "diiodomethane": ("diiodomethane", 50.8, 50.8, 0.0),
        "glycerol": ("glycerol", 63.4, 37.0, 26.4),
    }
    rows: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}
    number_pattern = re.compile(r"^(\d+(?:\.\d+)?)(?:\s*±\s*(\d+(?:\.\d+)?))?$")
    for tr in table.findall(".//tr")[1:]:
        cells = [" ".join("".join(cell.itertext()).split()) for cell in tr.findall("./th") + tr.findall("./td")]
        if len(cells) < 4:
            continue
        surface_name, liquid_raw, angle_raw, notes = cells[:4]
        liquid_key = liquid_raw.casefold()
        match = number_pattern.fullmatch(angle_raw)
        if match is None:
            skipped["range_or_transition_angle"] = skipped.get("range_or_transition_angle", 0) + 1
            continue
        if liquid_key not in liquid_properties:
            skipped["unsupported_liquid"] = skipped.get("unsupported_liquid", 0) + 1
            continue
        liquid_name, total, dispersion, polar = liquid_properties[liquid_key]
        rows.append({
            "source_group_id": source_id,
            "reference_doi": "10.1038/s41598-026-40965-x",
            "reference_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC13031877/",
            "solid_name": surface_name,
            "solid_family": "other",
            "solid_substrate": "",
            "surface_treatment": "literature-described condition",
            "surface_treatment_detail": notes,
            "surface_state": "static literature condition",
            "liquid_name": liquid_name,
            "liquid_total_surface_tension_mN_m": total,
            "liquid_dispersion_mN_m": dispersion,
            "liquid_polar_mN_m": polar,
            "contact_angle_deg": float(match.group(1)),
            "contact_angle_type": "static",
            "measurement_method": "sessile_drop" if "sessile" in notes.casefold() else "literature_reported",
            "temperature_K": np.nan,
            "replicates_n": np.nan,
            "contact_angle_std_deg": float(match.group(2)) if match.group(2) else np.nan,
            "solid_surface_energy_source_type": "not_reported",
            "table_or_figure_locator": f"JATS table {table.attrib.get('id', 'Tab1')}; {title}",
            "extraction_note": "Exact/static row extracted; range-only and unsupported-liquid rows were excluded.",
            "license_verified": "yes",
            "new_external_source_flag": "yes",
        })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    candidate = pd.DataFrame(rows)
    candidate.to_csv(output_path, index=False, encoding="utf-8-sig")
    return {
        "status": "candidate_extracted", "source_id": source_id,
        "table_id": table.attrib.get("id", "Tab1"), "table_title": title,
        "n_rows": len(candidate), "n_skipped": int(110 - len(candidate)),
        "skipped_reasons": skipped, "n_surfaces": int(candidate["solid_name"].nunique()),
        "n_liquids": int(candidate["liquid_name"].nunique()), "output_path": str(output_path),
    }


def extract_polymer_contact_angle_jats(xml_path: Path, output_path: Path, source_id: str) -> dict[str, Any]:
    """Extract the three-liquid polymer contact-angle table as a diagnostic source."""
    root = ET.parse(xml_path).getroot()
    selected = None
    for table in root.findall(".//table-wrap"):
        caption = table.find("caption")
        title = " ".join("".join(caption.itertext()).split()) if caption is not None else ""
        if "contact angle results of various polymer surfaces" in title.casefold():
            selected = (table, title)
            break
    if selected is None:
        raise ValueError("Polymer contact-angle table was not found")
    table, title = selected
    liquids = [
        ("water", 72.8, 21.8, 51.0),
        ("formamide", 58.0, 39.0, 19.0),
        ("diiodomethane", 50.8, 50.8, 0.0),
    ]
    rows: list[dict[str, Any]] = []
    for tr in table.findall(".//tr")[1:]:
        cells = [" ".join("".join(cell.itertext()).split()) for cell in tr.findall("./th") + tr.findall("./td")]
        if len(cells) < 4:
            continue
        values = [re.fullmatch(r"\d+(?:\.\d+)?", cells[index]) for index in [1, 2, 3]]
        if not all(values):
            continue
        for index, (liquid_name, total, dispersion, polar) in enumerate(liquids, start=1):
            rows.append({
                "source_group_id": source_id,
                "reference_doi": "10.55730/1300-0527.3518",
                "reference_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10504009/",
                "solid_name": cells[0], "solid_family": "polymer",
                "solid_substrate": "polymer surface", "surface_treatment": "literature-reported",
                "surface_treatment_detail": "Table-reported polymer surface; treatment detail not available in this table",
                "surface_state": "literature-reported surface", "liquid_name": liquid_name,
                "liquid_total_surface_tension_mN_m": total,
                "liquid_dispersion_mN_m": dispersion, "liquid_polar_mN_m": polar,
                "contact_angle_deg": float(cells[index]), "contact_angle_type": "static",
                "measurement_method": "literature_reported", "temperature_K": np.nan,
                "replicates_n": np.nan, "contact_angle_std_deg": np.nan,
                "solid_surface_energy_source_type": "not_reported",
                "table_or_figure_locator": f"JATS table {table.attrib.get('id', '')}; {title}",
                "extraction_note": "Diagnostic literature-reported angle; advancing/receding status is not specified in the source table.",
                "license_verified": "yes", "new_external_source_flag": "yes",
            })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
    return {
        "status": "candidate_extracted", "source_id": source_id,
        "table_id": table.attrib.get("id", ""), "table_title": title,
        "n_rows": len(rows), "n_surfaces": len({row["solid_name"] for row in rows}),
        "n_liquids": len(liquids), "output_path": str(output_path),
    }


def extract_textile_surface_jats(xml_path: Path, output_path: Path, source_id: str) -> dict[str, Any]:
    """Extract static contact angles from the textile anti-wetting Table 6.

    The source table places static CA and shedding angle (ShA) side by side. Only
    the three CA columns are imported; ShA is deliberately excluded because it is
    a different target property.
    """
    root = ET.parse(xml_path).getroot()
    table = root.find(".//table-wrap[@id='polymers-11-00498-t006']")
    if table is None:
        raise ValueError("Textile contact-angle Table 6 was not found")
    caption = table.find("caption")
    title = " ".join("".join(caption.itertext()).split()) if caption is not None else ""
    liquids = [
        ("water", 72.8, 21.8, 51.0, 1),
        ("formamide", 58.0, 39.0, 19.0, 3),
        ("diiodomethane", 50.8, 50.8, 0.0, 5),
    ]
    rows: list[dict[str, Any]] = []

    def surface_description(specimen: str) -> tuple[str, str, str]:
        is_nonwoven = "(SL)" in specimen or "(SB" in specimen
        substrate = "polymeric nonwoven textile" if is_nonwoven else "polymeric film"
        state = "spunlace nonwoven" if "(SL)" in specimen else (
            "spunbond nonwoven" if "(SB" in specimen else "flat film"
        )
        treatments: list[str] = []
        if "-f" in specimen:
            treatments.append("fluorinated anti-wetting coating")
        if "-Si" in specimen:
            treatments.append("silane anti-wetting coating")
        if "-etch" in specimen:
            treatments.append("alkaline etching")
        treatment = "; ".join(treatments) if treatments else "untreated"
        detail = f"Specimen label {specimen}; {state}; {treatment}."
        return substrate, state, detail

    number_pattern = re.compile(r"^(\d+(?:\.\d+)?)$")
    for tr in table.findall(".//tr")[2:]:
        cells = [" ".join("".join(cell.itertext()).split()) for cell in tr.findall("./th") + tr.findall("./td")]
        if len(cells) < 7:
            continue
        specimen = cells[0]
        if not specimen or not re.search(r"[A-Za-z]", specimen):
            continue
        substrate, state, treatment_detail = surface_description(specimen)
        for liquid_name, total, dispersion, polar, angle_offset in liquids:
            angle_match = number_pattern.fullmatch(cells[angle_offset])
            sd_match = re.search(r"(\d+(?:\.\d+)?)", cells[angle_offset + 1])
            if angle_match is None or sd_match is None:
                raise ValueError(
                    f"Could not parse static CA/SD in textile row {specimen!r}: "
                    f"{cells[angle_offset:angle_offset + 2]}"
                )
            rows.append({
                "source_group_id": source_id,
                "reference_doi": "10.3390/polym11030498",
                "reference_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC6473839/",
                "solid_name": specimen,
                "solid_family": "polymer_textile",
                "solid_substrate": substrate,
                "surface_treatment": treatment_detail.split("; ", 1)[-1].rstrip("."),
                "surface_treatment_detail": treatment_detail,
                "surface_state": state,
                "liquid_name": liquid_name,
                "liquid_total_surface_tension_mN_m": total,
                "liquid_dispersion_mN_m": dispersion,
                "liquid_polar_mN_m": polar,
                "contact_angle_deg": float(angle_match.group(1)),
                "contact_angle_type": "static",
                "measurement_method": "sessile_drop",
                "temperature_K": np.nan,
                "replicates_n": np.nan,
                "contact_angle_std_deg": float(sd_match.group(1)),
                "solid_surface_energy_source_type": "not_reported",
                "table_or_figure_locator": f"JATS table {table.attrib.get('id', '')}; {title}; CA columns",
                "extraction_note": (
                    "Static contact angle and SD extracted from CA columns only; "
                    "shedding-angle columns were intentionally excluded."
                ),
                "license_verified": "yes",
                "new_external_source_flag": "yes",
            })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
    return {
        "status": "candidate_extracted", "source_id": source_id,
        "table_id": table.attrib.get("id", ""), "table_title": title,
        "n_rows": len(rows), "n_surfaces": len({row["solid_name"] for row in rows}),
        "n_liquids": len(liquids), "excluded_shedding_angle_columns": True,
        "output_path": str(output_path),
    }


def extract_pla_films_jats(xml_path: Path, output_path: Path, source_id: str) -> dict[str, Any]:
    """Extract the three-liquid PLA film table with substrate and solvent factors."""
    root = ET.parse(xml_path).getroot()
    table = root.find(".//table-wrap[@id='polymers-13-04289-t001']")
    if table is None:
        raise ValueError("PLA film contact-angle Table 1 was not found")
    caption = table.find("caption")
    title = " ".join("".join(caption.itertext()).split()) if caption is not None else ""
    liquids = [
        ("water", 72.8, 21.8, 51.0, 1),
        ("formamide", 58.0, 39.0, 19.0, 5),
        ("diiodomethane", 50.8, 50.8, 0.0, 9),
    ]
    conditions = [
        ("Control", "substrate control; no PLA film"),
        ("PLA-C", "PLA film cast from chloroform"),
        ("PLA-A", "PLA film cast from acetone"),
        ("PLA-T", "PLA film cast from tetrahydrofuran"),
    ]
    rows: list[dict[str, Any]] = []
    number_pattern = re.compile(r"^(\d+(?:\.\d+)?)\s*[±+−-]\s*(\d+(?:\.\d+)?)$")
    for tr in table.findall(".//tr")[2:]:
        cells = [" ".join("".join(cell.itertext()).split()) for cell in tr.findall("./th") + tr.findall("./td")]
        if len(cells) != 13:
            continue
        substrate = cells[0]
        if not substrate:
            continue
        for liquid_name, total, dispersion, polar, liquid_offset in liquids:
            for condition_index, (condition, detail) in enumerate(conditions):
                value = cells[liquid_offset + condition_index]
                match = number_pattern.fullmatch(value)
                if match is None:
                    raise ValueError(
                        f"Could not parse PLA angle/SD for {substrate}/{condition}/{liquid_name}: {value!r}"
                    )
                solid_name = f"{condition} on {substrate}"
                rows.append({
                    "source_group_id": source_id,
                    "reference_doi": "10.3390/polym13244289",
                    "reference_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC8707572/",
                    "solid_name": solid_name,
                    "solid_family": "polymer_film",
                    "solid_substrate": substrate,
                    "surface_treatment": detail,
                    "surface_treatment_detail": (
                        f"{detail}; source Table 1 condition {condition}; "
                        "film preparation and substrate are retained as separate fields."
                    ),
                    "surface_state": "flat film" if condition != "Control" else "bare substrate control",
                    "liquid_name": liquid_name,
                    "liquid_total_surface_tension_mN_m": total,
                    "liquid_dispersion_mN_m": dispersion,
                    "liquid_polar_mN_m": polar,
                    "contact_angle_deg": float(match.group(1)),
                    "contact_angle_type": "static",
                    "measurement_method": "sessile_drop",
                    "temperature_K": np.nan,
                    "replicates_n": 15,
                    "contact_angle_std_deg": float(match.group(2)),
                    "solid_surface_energy_source_type": "not_reported",
                    "table_or_figure_locator": f"JATS table {table.attrib.get('id', '')}; {title}",
                    "extraction_note": (
                        "Average +/- standard deviation extracted from Table 1. "
                        "The article reports at least fifteen drops on three samples; replicates_n=15 is a conservative lower bound."
                    ),
                    "license_verified": "yes",
                    "new_external_source_flag": "yes",
                })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
    return {
        "status": "candidate_extracted", "source_id": source_id,
        "table_id": table.attrib.get("id", ""), "table_title": title,
        "n_rows": len(rows), "n_surfaces": len({row["solid_name"] for row in rows}),
        "n_liquids": len(liquids), "conditions": [condition for condition, _ in conditions],
        "output_path": str(output_path),
    }


def extract_carbon_surface_jats(xml_path: Path, output_path: Path, source_id: str) -> dict[str, Any]:
    """Extract static water/glycerol/diiodomethane angles from carbon Table 4."""
    root = ET.parse(xml_path).getroot()
    table = root.find(".//table-wrap[@id='t4']")
    if table is None:
        raise ValueError("Carbon surface contact-angle Table 4 was not found")
    caption = table.find("caption")
    title = " ".join("".join(caption.itertext()).split()) if caption is not None else ""
    liquids = [
        ("water", 72.8, 21.8, 51.0, 1),
        ("glycerol", 63.4, 37.0, 26.4, 2),
        ("diiodomethane", 50.8, 50.8, 0.0, 3),
    ]
    rows: list[dict[str, Any]] = []
    number_pattern = re.compile(r"^(\d+(?:\.\d+)?)\s*[±]\s*(\d+(?:\.\d+)?)$")
    for tr in table.findall(".//tr")[2:]:
        cells = [" ".join("".join(cell.itertext()).split()) for cell in tr.findall("./th") + tr.findall("./td")]
        if len(cells) < 4 or not cells[0]:
            continue
        for liquid_name, total, dispersion, polar, offset in liquids:
            match = number_pattern.fullmatch(cells[offset])
            if match is None:
                raise ValueError(f"Could not parse carbon angle/SD for {cells[0]!r}: {cells[offset]!r}")
            rows.append({
                "source_group_id": source_id,
                "reference_doi": "10.1038/srep24840",
                "reference_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4843010/",
                "solid_name": cells[0], "solid_family": "carbon_surface",
                "solid_substrate": "carbon substrate",
                "surface_treatment": "glycan functionalization condition",
                "surface_treatment_detail": f"Surface label {cells[0]} from source Table 4; SFE is angle-derived and not used as an input.",
                "surface_state": "solid carbon surface",
                "liquid_name": liquid_name,
                "liquid_total_surface_tension_mN_m": total,
                "liquid_dispersion_mN_m": dispersion,
                "liquid_polar_mN_m": polar,
                "contact_angle_deg": float(match.group(1)),
                "contact_angle_type": "static", "measurement_method": "sessile_drop",
                "temperature_K": 298.15, "replicates_n": 3,
                "contact_angle_std_deg": float(match.group(2)),
                "solid_surface_energy_source_type": "angle_derived",
                "table_or_figure_locator": f"JATS table {table.attrib.get('id', '')}; {title}",
                "extraction_note": "Mean +/- SD extracted from the three static contact-angle columns; reported surface energy is not imported as independent SFE.",
                "license_verified": "yes", "new_external_source_flag": "yes",
            })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
    return {
        "status": "candidate_extracted", "source_id": source_id,
        "table_id": table.attrib.get("id", ""), "table_title": title,
        "n_rows": len(rows), "n_surfaces": len({row["solid_name"] for row in rows}),
        "n_liquids": len(liquids), "output_path": str(output_path),
    }


def extract_facemask_surface_jats(xml_path: Path, output_path: Path, source_id: str) -> dict[str, Any]:
    """Extract static water/formamide/diiodomethane angles from facemask Table 2."""
    root = ET.parse(xml_path).getroot()
    table = root.find(".//table-wrap[@id='T2']")
    if table is None:
        raise ValueError("Facemask contact-angle Table 2 was not found")
    caption = table.find("caption")
    title = " ".join("".join(caption.itertext()).split()) if caption is not None else ""
    liquids = [
        ("diiodomethane", 50.8, 50.8, 0.0, 1),
        ("formamide", 58.0, 39.0, 19.0, 2),
        ("water", 72.8, 21.8, 51.0, 3),
    ]
    rows: list[dict[str, Any]] = []
    number_pattern = re.compile(r"^(\d+(?:\.\d+)?)\s*[\(\[]\s*(\d+(?:\.\d+)?)\s*[\)\]]$")
    for tr in table.findall(".//tr")[3:]:
        cells = [" ".join("".join(cell.itertext()).split()) for cell in tr.findall("./th") + tr.findall("./td")]
        if len(cells) < 4 or not cells[0]:
            continue
        for liquid_name, total, dispersion, polar, offset in liquids:
            match = number_pattern.fullmatch(cells[offset])
            if match is None:
                raise ValueError(f"Could not parse facemask angle/SD for {cells[0]!r}: {cells[offset]!r}")
            rows.append({
                "source_group_id": source_id,
                "reference_doi": "10.18502/ijm.v15i2.12482",
                "reference_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC10183065/",
                "solid_name": f"facemask_{cells[0]}", "solid_family": "textile_composite",
                "solid_substrate": "reusable facemask fabric",
                "surface_treatment": "commercial facemask construction",
                "surface_treatment_detail": f"Commercial reusable facemask species {cells[0]}; composition details are not expanded in Table 2.",
                "surface_state": "fabric surface",
                "liquid_name": liquid_name,
                "liquid_total_surface_tension_mN_m": total,
                "liquid_dispersion_mN_m": dispersion,
                "liquid_polar_mN_m": polar,
                "contact_angle_deg": float(match.group(1)),
                "contact_angle_type": "static", "measurement_method": "sessile_drop",
                "temperature_K": np.nan, "replicates_n": 3,
                "contact_angle_std_deg": float(match.group(2)),
                "solid_surface_energy_source_type": "angle_derived",
                "table_or_figure_locator": f"JATS table {table.attrib.get('id', '')}; {title}",
                "extraction_note": "Mean (SD) extracted from the three static contact-angle columns; the article reports triplicate measurements and calculated SFE is not imported as independent SFE.",
                "license_verified": "yes", "new_external_source_flag": "yes",
            })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
    return {
        "status": "candidate_extracted", "source_id": source_id,
        "table_id": table.attrib.get("id", ""), "table_title": title,
        "n_rows": len(rows), "n_surfaces": len({row["solid_name"] for row in rows}),
        "n_liquids": len(liquids), "output_path": str(output_path),
    }


def extract_chitosan_gelatin_jats(xml_path: Path, output_path: Path, source_id: str) -> dict[str, Any]:
    """Extract static glycerol/diiodomethane angles from chitosan-gelatin Table 3."""
    root = ET.parse(xml_path).getroot()
    table = root.find(".//table-wrap[@id='ijms-23-09700-t003']")
    if table is None:
        raise ValueError("Chitosan-gelatin contact-angle Table 3 was not found")
    caption = table.find("caption")
    title = " ".join("".join(caption.itertext()).split()) if caption is not None else ""
    liquids = [
        ("glycerol", 63.4, 37.0, 26.4, 1),
        ("diiodomethane", 50.8, 50.8, 0.0, 2),
    ]
    rows: list[dict[str, Any]] = []
    for tr in table.findall(".//tr")[3:]:
        cells = [" ".join("".join(cell.itertext()).split()) for cell in tr.findall("./th") + tr.findall("./td")]
        if len(cells) < 3 or not cells[0]:
            continue
        for liquid_name, total, dispersion, polar, offset in liquids:
            value = re.fullmatch(r"(\d+(?:\.\d+)?)", cells[offset])
            if value is None:
                raise ValueError(f"Could not parse chitosan-gelatin angle for {cells[0]!r}: {cells[offset]!r}")
            rows.append({
                "source_group_id": source_id,
                "reference_doi": "10.3390/ijms23179700",
                "reference_url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC9456065/",
                "solid_name": cells[0], "solid_family": "biopolymer_film",
                "solid_substrate": "chitosan-gelatin film",
                "surface_treatment": "cellulose-nanocrystal crosslinking",
                "surface_treatment_detail": f"Sample formulation {cells[0]}; crosslinker and concentration are encoded in the source sample label.",
                "surface_state": "dried biopolymer film",
                "liquid_name": liquid_name,
                "liquid_total_surface_tension_mN_m": total,
                "liquid_dispersion_mN_m": dispersion,
                "liquid_polar_mN_m": polar,
                "contact_angle_deg": float(value.group(1)),
                "contact_angle_type": "static", "measurement_method": "drop_shape_analysis",
                "temperature_K": 293.15, "replicates_n": 5,
                "contact_angle_std_deg": np.nan,
                "solid_surface_energy_source_type": "angle_derived",
                "table_or_figure_locator": f"JATS table {table.attrib.get('id', '')}; {title}",
                "extraction_note": "Average static contact angles extracted from Table 3; article methods report five measurements per angle and calculated SFE is not imported as independent SFE.",
                "license_verified": "yes", "new_external_source_flag": "yes",
            })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False, encoding="utf-8-sig")
    return {
        "status": "candidate_extracted", "source_id": source_id,
        "table_id": table.attrib.get("id", ""), "table_title": title,
        "n_rows": len(rows), "n_surfaces": len({row["solid_name"] for row in rows}),
        "n_liquids": len(liquids), "output_path": str(output_path),
    }


def _rebuild_derived_tables(data_dir: Path, audit_dir: Path) -> dict[str, Any]:
    tables = V4Tables.load(data_dir)
    samples = build_samples(tables)
    nnls_audit = build_nnls_audit(tables, samples)
    audit_fields = nnls_audit[[
        "sample_id", "fit_status", "nnls_dispersion_mj_m2", "nnls_polar_mj_m2",
        "nnls_physical_prediction_deg",
    ]].copy()
    audit_fields["loo_sfe_feasible_audit"] = np.where(
        audit_fields["fit_status"].isin(["interior_fit", "boundary_fit"]), "yes", "no"
    )
    audit_fields["loo_sfe_failure_reason_audit"] = np.where(
        audit_fields["loo_sfe_feasible_audit"] == "yes", "", audit_fields["fit_status"]
    )
    samples = samples.merge(
        audit_fields[[
            "sample_id", "loo_sfe_feasible_audit", "loo_sfe_failure_reason_audit",
            "nnls_dispersion_mj_m2", "nnls_polar_mj_m2", "nnls_physical_prediction_deg",
        ]], on="sample_id", how="left", validate="one_to_one"
    )
    samples["nnls_dispersion_mj_m2"] = samples["nnls_dispersion_mj_m2"].fillna(0.0)
    samples["nnls_polar_mj_m2"] = samples["nnls_polar_mj_m2"].fillna(0.0)
    samples["nnls_physical_prediction_deg"] = samples["nnls_physical_prediction_deg"].fillna(0.0)
    samples["loo_sfe_feasible"] = samples["loo_sfe_feasible_audit"].fillna(samples["loo_sfe_feasible"])
    samples["loo_sfe_failure_reason"] = samples["loo_sfe_failure_reason_audit"].fillna(samples["loo_sfe_failure_reason"])
    samples = samples.drop(columns=["loo_sfe_feasible_audit", "loo_sfe_failure_reason_audit"])
    samples.to_csv(data_dir / "samples_v4.csv", index=False, encoding="utf-8-sig")
    audit_dir.mkdir(parents=True, exist_ok=True)
    nnls_audit.to_csv(audit_dir / "loo_nnls_audit_v4.csv", index=False, encoding="utf-8-sig")
    validation = validate_v4_tables(tables, samples)
    validation["nnls_fit_status_counts"] = {
        str(k): int(v) for k, v in nnls_audit["fit_status"].value_counts(dropna=False).items()
    }
    validation["legacy_negative_coefficient_rows"] = int(
        (nnls_audit["legacy_negative_coefficient"] == "yes").sum()
    )
    (audit_dir / "v4_migration_audit.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return validation


def merge_staged_source(
    staged_path: Path,
    data_dir: Path,
    registry_path: Path,
    audit_dir: Path,
    source_id: str,
    seed: int,
) -> dict[str, Any]:
    """Merge a validated staged source as a frozen prospective external source."""
    registry = _registry_row(registry_path, source_id)
    if str(registry.get("verification_status", "")).casefold() != "verified":
        raise ValueError(f"Cannot merge unverified source {source_id}")
    staged = pd.read_csv(staged_path, encoding="utf-8-sig")
    if not len(staged) or (staged.get("import_status", "validated_staged") == "rejected").any():
        raise ValueError("Staged source is empty or contains rejected rows")
    tables = V4Tables.load(data_dir)
    existing_measurements = tables.measurements.copy()
    existing_keys = set(zip(
        existing_measurements["surface_group_id"].astype(str),
        existing_measurements["liquid_id"].astype(str),
        existing_measurements["contact_angle_type"].astype(str),
    ))
    source_row = {
        "source_group_id": source_id,
        "reference_doi": registry.get("doi", ""),
        "reference_url": registry.get("url", ""),
        "source_type": "open_literature_table",
        "reference_title": registry.get("title", ""),
        "reference_authors": "",
        "reference_year": registry.get("publication_year", ""),
        "collection_status": "verified",
        "license": registry.get("license_or_access", ""),
        "extraction_status": "validated_staged",
        "new_external_source_flag": "yes",
        "applsci_split": "prospective_open_external",
    }
    sources = tables.sources.loc[tables.sources["source_group_id"].astype(str) != source_id].copy()
    tables.sources = pd.concat([sources, pd.DataFrame([source_row])], ignore_index=True)

    surface_columns = [
        "surface_group_id", "source_group_id", "solid_name", "solid_family", "solid_substrate",
        "surface_treatment", "surface_treatment_detail", "coating_or_layer", "surface_state",
        "roughness_Ra_nm", "roughness_Rq_nm", "roughness_r_factor", "sample_preparation_notes",
    ]
    new_surfaces = []
    for surface_id, group in staged.groupby("surface_group_id", sort=False):
        first = group.iloc[0]
        new_surfaces.append({
            "surface_group_id": surface_id, "source_group_id": source_id,
            "solid_name": first.get("solid_name", ""), "solid_family": first.get("solid_family", ""),
            "solid_substrate": first.get("solid_substrate", ""),
            "surface_treatment": first.get("surface_treatment", ""),
            "surface_treatment_detail": first.get("surface_treatment_detail", ""),
            "coating_or_layer": first.get("coating_or_layer", ""),
            "surface_state": first.get("surface_state", ""),
            "roughness_Ra_nm": first.get("roughness_Ra_nm", np.nan),
            "roughness_Rq_nm": first.get("roughness_Rq_nm", np.nan),
            "roughness_r_factor": first.get("roughness_r_factor", np.nan),
            "sample_preparation_notes": first.get("extraction_note", ""),
        })
    tables.surfaces = pd.concat([
        tables.surfaces, pd.DataFrame(new_surfaces, columns=surface_columns)
    ], ignore_index=True).drop_duplicates(subset=["surface_group_id"], keep="last")

    liquid_columns = [
        "liquid_id", "liquid_name", "liquid_family", "liquid_total_surface_tension_mN_m",
        "liquid_dispersion_mN_m", "liquid_polar_mN_m", "liquid_LW_mN_m",
        "liquid_acid_plus_mN_m", "liquid_base_minus_mN_m", "liquid_viscosity_mPa_s",
        "liquid_dipole_moment_D", "liquid_dielectric_constant", "liquid_property_source",
    ]
    new_liquids = []
    for liquid_id, group in staged.groupby("liquid_id", sort=False):
        first = group.iloc[0]
        new_liquids.append({
            "liquid_id": liquid_id, "liquid_name": canonical_liquid_name(first["liquid_name"]),
            "liquid_family": "polar" if float(first["liquid_polar_mN_m"]) > 0 else "nonpolar",
            "liquid_total_surface_tension_mN_m": first["liquid_total_surface_tension_mN_m"],
            "liquid_dispersion_mN_m": first["liquid_dispersion_mN_m"],
            "liquid_polar_mN_m": first["liquid_polar_mN_m"],
            "liquid_LW_mN_m": first["liquid_dispersion_mN_m"],
            "liquid_acid_plus_mN_m": np.nan, "liquid_base_minus_mN_m": np.nan,
            "liquid_viscosity_mPa_s": np.nan, "liquid_dipole_moment_D": np.nan,
            "liquid_dielectric_constant": np.nan,
            "liquid_property_source": "staged source row; standard components recorded in import table",
        })
    tables.liquids = pd.concat([
        tables.liquids, pd.DataFrame(new_liquids, columns=liquid_columns)
    ], ignore_index=True).drop_duplicates(subset=["liquid_id"], keep="last")

    new_measurements = []
    skipped_duplicates = 0
    for row in staged.itertuples(index=False):
        key = (str(row.surface_group_id), str(row.liquid_id), str(row.contact_angle_type))
        if key in existing_keys:
            skipped_duplicates += 1
            continue
        measurement_id = stable_id("EXT", [source_id, row.source_row_id], length=12)
        new_measurements.append({
            "measurement_id": measurement_id, "record_id": measurement_id,
            "surface_group_id": row.surface_group_id, "source_group_id": source_id,
            "liquid_id": row.liquid_id, "contact_angle_deg": row.contact_angle_deg,
            "contact_angle_type": row.contact_angle_type, "measurement_method": row.measurement_method,
            "temperature_K": row.temperature_K, "humidity_percent": np.nan,
            "pressure_atm": np.nan, "droplet_volume_uL": np.nan, "replicates_n": row.replicates_n,
            "contact_angle_std_deg": row.contact_angle_std_deg,
            "contact_angle_min_deg": np.nan, "contact_angle_max_deg": np.nan,
            "quality_grade": "A_high", "conflict_flag": "no",
            "solid_total_surface_energy_mJ_m2": np.nan, "solid_dispersion_mJ_m2": np.nan,
            "solid_polar_mJ_m2": np.nan, "solid_surface_energy_source": "",
            "solid_surface_energy_source_type": "not_reported",
            "data_extraction_note": f"{row.table_or_figure_locator}; {row.extraction_note}",
            "target_eligible": "yes", "sfe_source_type": "not_reported",
            "new_external_source_flag": "yes",
        })
        existing_keys.add(key)
    tables.measurements = pd.concat([
        tables.measurements, pd.DataFrame(new_measurements, columns=tables.measurements.columns)
    ], ignore_index=True)

    split_rows = []
    for measurement in new_measurements:
        split_rows.append({
            "measurement_id": measurement["measurement_id"], "record_id": measurement["record_id"],
            "source_group_id": source_id, "surface_group_id": measurement["surface_group_id"],
            "legacy_analysis_split": "prospective_open_external",
            "v4_split": "prospective_open_external", "applsci_split": "prospective_open_external",
            "allow_training": "no", "allow_tuning": "no", "allow_evaluation": "yes",
            "freeze_status": "provisional_open_external_pending_model_run",
        })
    tables.splits = pd.concat([
        tables.splits, pd.DataFrame(split_rows, columns=tables.splits.columns)
    ], ignore_index=True)
    for name, frame in [
        ("sources", tables.sources), ("surfaces", tables.surfaces), ("liquids", tables.liquids),
        ("measurements", tables.measurements), ("splits", tables.splits),
    ]:
        frame.to_csv(data_dir / f"{name}_v4.csv", index=False, encoding="utf-8-sig")
    (data_dir / "splits_v4.sha256").write_text(
        __import__("hashlib").sha256((data_dir / "splits_v4.csv").read_bytes()).hexdigest() + "\n",
        encoding="ascii",
    )
    validation = _rebuild_derived_tables(data_dir, audit_dir)
    return {
        "status": "complete" if validation["status"] == "pass" else "fail",
        "source_id": source_id, "added_measurements": len(new_measurements),
        "skipped_duplicates": skipped_duplicates, "added_surfaces": len(new_surfaces),
        "split": "prospective_open_external", "validation": validation,
    }


def audit_semantic_duplicates(data_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Report exact and near duplicate measurements across source boundaries."""
    tables = V4Tables.load(data_dir)
    merged = tables.measurements.merge(
        tables.surfaces[[
            "surface_group_id", "solid_name", "solid_family", "solid_substrate",
            "surface_treatment", "surface_treatment_detail", "surface_state",
        ]], on="surface_group_id", how="left", validate="many_to_one",
    ).merge(
        tables.liquids[["liquid_id", "liquid_name"]], on="liquid_id", how="left", validate="many_to_one",
    )

    def token(value: Any) -> str:
        return re.sub(r"[^a-z0-9]+", "", clean_text(value).casefold())

    surface_columns = [
        "solid_name", "solid_family", "solid_substrate", "surface_treatment",
        "surface_treatment_detail", "surface_state",
    ]
    merged["semantic_surface_key"] = merged[surface_columns].fillna("").astype(str).agg("|".join, axis=1).map(token)
    merged["semantic_liquid_key"] = merged["liquid_name"].map(canonical_liquid_name).map(token)
    exact_key = ["semantic_surface_key", "semantic_liquid_key", "contact_angle_type", "contact_angle_deg"]
    near_key = ["semantic_surface_key", "semantic_liquid_key", "contact_angle_type"]
    exact_counts = merged.groupby(exact_key, dropna=False)["source_group_id"].transform("nunique")
    near = merged.loc[exact_counts > 1].copy()
    near["duplicate_kind"] = "exact_surface_liquid_type_angle_across_sources"
    if len(near):
        near_counts = near.groupby(near_key, dropna=False)["contact_angle_deg"].transform("count")
        near.loc[near_counts > 1, "duplicate_kind"] = "near_surface_liquid_type_candidate"
        report = near[[
            "duplicate_kind", "measurement_id", "source_group_id", "surface_group_id",
            "solid_name", "liquid_name", "contact_angle_type", "contact_angle_deg",
        ]].copy()
    else:
        report = pd.DataFrame(columns=[
            "duplicate_kind", "measurement_id", "source_group_id", "surface_group_id",
            "solid_name", "liquid_name", "contact_angle_type", "contact_angle_deg",
        ])
    output_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_dir / "semantic_duplicate_audit_v4.csv", index=False, encoding="utf-8-sig")
    result = {
        "status": "complete", "n_measurements": int(len(merged)),
        "n_candidate_rows": int(len(report)),
        "n_candidate_groups": int(
            report.groupby(["duplicate_kind", "solid_name", "liquid_name", "contact_angle_type"]).ngroups
        ) if len(report) else 0,
        "note": "Candidates require manual review; no rows are removed automatically.",
    }
    (output_dir / "semantic_duplicate_audit_v4.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return result
