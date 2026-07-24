"""Build a provenance-locked v4.3 dataset without modifying the submitted data."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

from .data import V4Tables, build_liquid_id, canonical_liquid_name
from .ingest import _rebuild_derived_tables


FINAL_RETAIN_DECISIONS = {
    "retain_verified",
    "retain_corrected",
    "retain_reextracted",
}
EXCLUDE_PREFIX = "exclude_"
PENDING_PREFIX = "pending_"

REEXTRACTION_ALIASES = {
    "原表位置": "source_locator",
    "table_or_figure_locator": "source_locator",
    "original_table_location": "source_locator",
    "liquid_name": "liquid",
    "angle_deg": "contact_angle_deg",
    "angle_type": "contact_angle_type",
    "standard_deviation": "contact_angle_std_deg",
    "std_deg": "contact_angle_std_deg",
    "replicates": "replicates_n",
    "replicate_n": "replicates_n",
    "droplet_volume_uL": "droplet_volume",
    "censor_type": "censoring",
    "reference_doi": "doi",
    "DOI": "doi",
}

REQUIRED_REEXTRACTION_COLUMNS = {
    "source_group_id",
    "source_locator",
    "surface_label",
    "liquid",
    "contact_angle_deg",
    "contact_angle_type",
    "doi",
}


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def _portable_manifest_path(path: Path, root: Path) -> str:
    try:
        return Path(os.path.relpath(path, root)).as_posix()
    except ValueError:
        return path.name


def _clean(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def _number(value: Any) -> float | None:
    text = _clean(value)
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        import re

        match = re.match(r"^\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))", text)
        return float(match.group(1)) if match else None


def _stable_id(prefix: str, *parts: Any) -> str:
    payload = "|".join(_clean(part).casefold() for part in parts)
    return f"{prefix}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:12].upper()}"


def load_reextraction_files(paths: Iterable[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path, encoding="utf-8-sig")
        frame = frame.rename(columns={
            column: REEXTRACTION_ALIASES.get(column, column)
            for column in frame.columns
        })
        missing = REQUIRED_REEXTRACTION_COLUMNS - set(frame.columns)
        if missing:
            raise ValueError(f"{path.name} is missing re-extraction columns: {sorted(missing)}")
        frame["reextraction_file"] = path.name
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=sorted(REQUIRED_REEXTRACTION_COLUMNS))
    combined = pd.concat(frames, ignore_index=True)
    combined["source_group_id"] = combined["source_group_id"].astype(str).str.strip()
    return combined


def validate_decision_state(decisions: pd.DataFrame, reextracted: pd.DataFrame) -> None:
    if decisions["source_group_id"].astype(str).duplicated().any():
        raise ValueError("Duplicate source_group_id in provenance decisions")
    pending = decisions.loc[
        decisions["decision"].astype(str).str.startswith(PENDING_PREFIX),
        "source_group_id",
    ].astype(str).tolist()
    if pending:
        raise RuntimeError(
            "The v4.3 dataset cannot be locked while source decisions remain pending: "
            f"{pending}"
        )
    permitted = decisions["decision"].astype(str).isin(FINAL_RETAIN_DECISIONS)
    excluded = decisions["decision"].astype(str).str.startswith(EXCLUDE_PREFIX)
    unknown = decisions.loc[~(permitted | excluded), ["source_group_id", "decision"]]
    if not unknown.empty:
        raise ValueError(f"Unknown provenance decisions: {unknown.to_dict('records')}")
    expected = set(
        decisions.loc[
            decisions["decision"].astype(str).eq("retain_reextracted"),
            "source_group_id",
        ].astype(str)
    )
    present = set(reextracted.get("source_group_id", pd.Series(dtype=str)).astype(str))
    missing = sorted(expected - present)
    if missing:
        raise ValueError(f"Retained re-extracted sources have no replacement rows: {missing}")


def _source_splits(tables: V4Tables) -> dict[str, str]:
    active = tables.splits.loc[
        ~tables.splits["v4_split"].astype(str).eq("excluded_review")
    ]
    mapping: dict[str, str] = {}
    for source, frame in active.groupby("source_group_id"):
        counts = frame["v4_split"].astype(str).value_counts()
        if len(counts) != 1:
            raise ValueError(
                f"Source {source} crosses active splits and cannot be re-curated: "
                f"{counts.to_dict()}"
            )
        mapping[str(source)] = str(counts.index[0])
    return mapping


def _family(value: str) -> str:
    text = value.casefold()
    if any(token in text for token in ["poly", "pla", "ptfe", "fep", "topas"]):
        return "polymer"
    if any(token in text for token in ["paper", "cellulose"]):
        return "cellulosic"
    if any(token in text for token in ["tio2", "titan", "oxide", "glass", "silicon"]):
        return "inorganic"
    if any(token in text for token in ["bio", "bacter", "microb", "fung"]):
        return "biological"
    return "other"


def _angle_type(value: Any) -> str:
    text = _clean(value).casefold()
    if not text or "not specified" in text or "unspecified" in text:
        return "reported_unspecified"
    if "advanc" in text:
        return "advancing"
    if "reced" in text:
        return "receding"
    if "equilibrium" in text:
        return "equilibrium"
    if "capillary" in text or "washburn" in text:
        return "capillary_rise_derived"
    if "roughness" in text and "correct" in text:
        return "roughness_corrected_apparent"
    if "apparent" in text:
        return "apparent"
    if "static" in text:
        return "static"
    return text


def _allow_fields(split: str) -> tuple[str, str, str]:
    if split == "train":
        return "yes", "yes", "no"
    if split == "validation":
        return "no", "yes", "no"
    return "no", "no", "yes"


def _replacement_rows(
    rows: pd.DataFrame,
    source_id: str,
    split: str,
    tables: V4Tables,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    liquid_lookup = {
        canonical_liquid_name(row.liquid_name): str(row.liquid_id)
        for row in tables.liquids.itertuples(index=False)
    }
    surfaces: dict[str, dict[str, Any]] = {}
    measurements: list[dict[str, Any]] = []
    splits: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    measurement_template = {column: np.nan for column in tables.measurements.columns}
    split_template = {column: "" for column in tables.splits.columns}
    seen_keys: set[tuple[str, str]] = set()

    rows = rows.copy()
    if "record_status" in rows:
        rows = rows.loc[~rows["record_status"].astype(str).str.startswith("exclude")]
    for row_number, row in enumerate(rows.to_dict("records"), start=1):
        angle = _number(row.get("contact_angle_deg"))
        censoring = _clean(row.get("censoring")).casefold()
        censored = censoring not in {"", "none", "no", "nan", "not_censored"}
        if angle is None and not censored:
            continue
        liquid_name = canonical_liquid_name(row.get("liquid", ""))
        if liquid_name not in liquid_lookup:
            candidate = build_liquid_id(liquid_name)
            if candidate not in set(tables.liquids["liquid_id"].astype(str)):
                excluded_rows.append({
                    "source_group_id": source_id,
                    "source_locator": _clean(row.get("source_locator")),
                    "surface_label": _clean(row.get("surface_label")),
                    "liquid": liquid_name,
                    "reason": "missing_locked_liquid_surface_tension_components",
                })
                continue
            liquid_lookup[liquid_name] = candidate
        material = _clean(row.get("material")) or _clean(row.get("solid_name"))
        treatment = _clean(row.get("treatment")) or _clean(row.get("surface_treatment"))
        state = _clean(row.get("state")) or _clean(row.get("surface_state"))
        label = _clean(row.get("surface_label"))
        surface_id = _stable_id("SFC", source_id, label, material, treatment, state)
        if surface_id not in surfaces:
            surface_template = {column: np.nan for column in tables.surfaces.columns}
            surface_template.update({
                "surface_group_id": surface_id,
                "source_group_id": source_id,
                "solid_name": label or material,
                "solid_family": _clean(row.get("solid_family")) or _family(material),
                "solid_substrate": _clean(row.get("substrate")),
                "surface_treatment": treatment or "reported_state",
                "surface_treatment_detail": _clean(row.get("treatment_detail")) or treatment,
                "coating_or_layer": _clean(row.get("coating_or_layer")),
                "surface_state": state or "reported",
                "roughness_Ra_nm": _number(row.get("roughness_Ra_nm")),
                "roughness_Rq_nm": _number(row.get("roughness_Rq_nm")),
                "roughness_r_factor": _number(row.get("roughness_r_factor")),
                "sample_preparation_notes": (
                    f"{_clean(row.get('source_locator'))}; "
                    f"{_clean(row.get('keep_reason'))}"
                ).strip("; "),
            })
            surfaces[surface_id] = surface_template
        liquid_id = liquid_lookup[liquid_name]
        duplicate_key = (surface_id, liquid_id)
        if duplicate_key in seen_keys:
            raise ValueError(
                f"Duplicate surface/liquid replacement row for {source_id}: {label}, {liquid_name}"
            )
        seen_keys.add(duplicate_key)
        measurement_id = _stable_id(
            "MEA", source_id, surface_id, liquid_id, row_number, row.get("source_locator")
        )
        target_eligible = "yes" if angle is not None and not censored else "no"
        measurement = measurement_template.copy()
        measurement.update({
            "measurement_id": measurement_id,
            "surface_group_id": surface_id,
            "source_group_id": source_id,
            "liquid_id": liquid_id,
            "contact_angle_deg": angle,
            "contact_angle_type": _angle_type(row.get("contact_angle_type")),
            "measurement_method": _clean(row.get("measurement_method")) or "sessile_drop",
            "temperature_K": _number(row.get("temperature_K")),
            "humidity_percent": _number(row.get("humidity_percent")),
            "pressure_atm": _number(row.get("pressure_atm")),
            "droplet_volume_uL": _number(row.get("droplet_volume")),
            "replicates_n": _number(row.get("replicates_n")),
            "contact_angle_std_deg": _number(row.get("contact_angle_std_deg")),
            "contact_angle_min_deg": _number(row.get("contact_angle_min_deg")),
            "contact_angle_max_deg": _number(row.get("contact_angle_max_deg")),
            "quality_grade": "A" if target_eligible == "yes" else "B",
            "conflict_flag": "no",
            "solid_surface_energy_source": "",
            "solid_surface_energy_source_type": "",
            "data_extraction_note": (
                f"{_clean(row.get('source_locator'))}; "
                f"re-extracted for v4.3 from {row.get('reextraction_file', '')}"
            ).strip("; "),
            "record_id": _stable_id("REC", measurement_id),
            "sfe_source_type": "none",
            "new_external_source_flag": "no",
            "target_eligible": target_eligible,
        })
        measurements.append(measurement)
        allow_training, allow_tuning, allow_evaluation = _allow_fields(split)
        split_row = split_template.copy()
        split_row.update({
            "measurement_id": measurement_id,
            "record_id": measurement["record_id"],
            "source_group_id": source_id,
            "surface_group_id": surface_id,
            "legacy_analysis_split": split,
            "v4_split": split,
            "applsci_split": split,
            "allow_training": allow_training,
            "allow_tuning": allow_tuning,
            "allow_evaluation": allow_evaluation,
            "freeze_status": "v4.3_source_audit_locked",
        })
        splits.append(split_row)
    if not measurements:
        raise ValueError(f"No eligible replacement measurements were supplied for {source_id}")
    return list(surfaces.values()), measurements, splits, excluded_rows


def build_recurated_dataset(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    root = config_path.parent.parent
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    project = config["project"]
    base_dir = _resolve(root, project["base_data_dir"])
    output_dir = _resolve(root, project["output_data_dir"])
    decisions_path = _resolve(root, project["provenance_review_file"])
    reextraction_dir = _resolve(root, project["reextraction_dir"])
    decisions = pd.read_csv(decisions_path, encoding="utf-8-sig")
    reextracted = load_reextraction_files(sorted(reextraction_dir.glob("**/records_*.csv")))
    validate_decision_state(decisions, reextracted)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    shutil.copytree(base_dir, output_dir)
    tables = V4Tables.load(output_dir)
    source_splits = _source_splits(tables)
    decision_map = decisions.set_index("source_group_id")["decision"].astype(str).to_dict()
    audited_sources = set(decision_map)
    alias_map = {
        str(source): [str(alias) for alias in aliases]
        for source, aliases in project.get("duplicate_source_aliases", {}).items()
    }
    replace_sources = {
        source for source, decision in decision_map.items()
        if decision == "retain_reextracted"
    }
    remove_sources = {
        source for source, decision in decision_map.items()
        if decision.startswith(EXCLUDE_PREFIX) or decision == "retain_reextracted"
    }
    duplicate_aliases = {
        alias
        for source in replace_sources
        for alias in alias_map.get(source, [])
    }
    remove_sources |= duplicate_aliases

    tables.sources = tables.sources.loc[
        ~tables.sources["source_group_id"].astype(str).isin(
            {
                source for source, decision in decision_map.items()
                if decision.startswith(EXCLUDE_PREFIX)
            }
            | duplicate_aliases
        )
    ].copy()
    tables.surfaces = tables.surfaces.loc[
        ~tables.surfaces["source_group_id"].astype(str).isin(remove_sources)
    ].copy()
    tables.measurements = tables.measurements.loc[
        ~tables.measurements["source_group_id"].astype(str).isin(remove_sources)
    ].copy()
    tables.splits = tables.splits.loc[
        ~tables.splits["source_group_id"].astype(str).isin(remove_sources)
    ].copy()
    excluded_reextraction_rows: list[dict[str, Any]] = []

    # Two sources are numerically verified but require angle-type metadata correction.
    if decision_map.get("SRC002") == "retain_corrected":
        source_mask = tables.measurements["source_group_id"].astype(str).eq("SRC002")
        liquid_names = tables.measurements.loc[source_mask, ["measurement_id", "liquid_id"]].merge(
            tables.liquids[["liquid_id", "liquid_name"]], on="liquid_id", how="left"
        ).set_index("measurement_id")["liquid_name"]
        water_ids = liquid_names.loc[
            liquid_names.astype(str).str.casefold().eq("water")
        ].index
        tables.measurements.loc[
            tables.measurements["measurement_id"].isin(water_ids), "contact_angle_type"
        ] = "advancing"
        tables.measurements.loc[
            source_mask & ~tables.measurements["measurement_id"].isin(water_ids),
            "contact_angle_type",
        ] = "equilibrium"
    if decision_map.get("SDX010") == "retain_corrected":
        source_mask = tables.measurements["source_group_id"].astype(str).eq("SDX010")
        tables.measurements.loc[source_mask, "contact_angle_type"] = "apparent"
        tables.measurements.loc[source_mask, "data_extraction_note"] = (
            tables.measurements.loc[source_mask, "data_extraction_note"].astype(str)
            + "; angle type corrected to apparent in v4.3 source audit"
        )

    for source_id in sorted(replace_sources):
        if source_id not in source_splits:
            raise ValueError(f"No frozen active split exists for re-extracted source {source_id}")
        rows = reextracted.loc[
            reextracted["source_group_id"].astype(str).eq(source_id)
        ]
        new_surfaces, new_measurements, new_splits, skipped_rows = _replacement_rows(
            rows, source_id, source_splits[source_id], tables
        )
        excluded_reextraction_rows.extend(skipped_rows)
        tables.surfaces = pd.concat(
            [tables.surfaces, pd.DataFrame(new_surfaces)], ignore_index=True
        )
        tables.measurements = pd.concat(
            [tables.measurements, pd.DataFrame(new_measurements)], ignore_index=True
        )
        tables.splits = pd.concat(
            [tables.splits, pd.DataFrame(new_splits)], ignore_index=True
        )
        source_rows = reextracted.loc[
            reextracted["source_group_id"].astype(str).eq(source_id)
        ]
        source_mask = tables.sources["source_group_id"].astype(str).eq(source_id)
        if source_mask.any():
            if "doi" in source_rows and source_rows["doi"].fillna("").astype(str).str.strip().ne("").any():
                tables.sources.loc[source_mask, "reference_doi"] = (
                    source_rows.loc[
                        source_rows["doi"].fillna("").astype(str).str.strip().ne(""),
                        "doi",
                    ].iloc[0]
                )
            if "source_title" in source_rows and source_rows["source_title"].fillna("").astype(str).str.strip().ne("").any():
                tables.sources.loc[source_mask, "reference_title"] = (
                    source_rows.loc[
                        source_rows["source_title"].fillna("").astype(str).str.strip().ne(""),
                        "source_title",
                    ].iloc[0]
                )
            tables.sources.loc[source_mask, "extraction_status"] = "v4.3_row_level_reextracted"
            tables.sources.loc[source_mask, "collection_status"] = "verified_v4.3"

    # Source-level DOI identity must be unique among active measurements.
    active_sources = set(tables.measurements["source_group_id"].astype(str))
    source_view = tables.sources.loc[
        tables.sources["source_group_id"].astype(str).isin(active_sources)
    ].copy()
    normalized_doi = source_view["reference_doi"].fillna("").astype(str).str.casefold().str.strip()
    duplicated_doi = source_view.loc[
        normalized_doi.ne("") & normalized_doi.duplicated(keep=False),
        ["source_group_id", "reference_doi"],
    ]
    if not duplicated_doi.empty:
        raise ValueError(
            "Active source IDs still share a DOI after re-curation: "
            f"{duplicated_doi.to_dict('records')}"
        )

    for name, frame in [
        ("sources", tables.sources),
        ("surfaces", tables.surfaces),
        ("liquids", tables.liquids),
        ("measurements", tables.measurements),
        ("splits", tables.splits),
    ]:
        frame.to_csv(output_dir / f"{name}_v4.csv", index=False, encoding="utf-8-sig")
    split_hash = hashlib.sha256((output_dir / "splits_v4.csv").read_bytes()).hexdigest()
    (output_dir / "splits_v4.sha256").write_text(split_hash + "\n", encoding="ascii")
    audit_dir = _resolve(root, project["audit_dir"])
    validation = _rebuild_derived_tables(output_dir, audit_dir)
    manifest = {
        "status": "complete",
        "model_data_version": "v4.3",
        "base_data_dir": _portable_manifest_path(base_dir, root),
        "output_data_dir": _portable_manifest_path(output_dir, root),
        "audited_sources": sorted(audited_sources),
        "reextracted_sources": sorted(replace_sources),
        "removed_duplicate_aliases": sorted(duplicate_aliases),
        "excluded_reextraction_rows": excluded_reextraction_rows,
        "excluded_sources": sorted(
            source for source, decision in decision_map.items()
            if decision.startswith(EXCLUDE_PREFIX)
        ),
        "validation": validation,
        "split_sha256": split_hash,
    }
    manifest_path = output_dir / "recuration_manifest_v4_3.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pd.DataFrame(excluded_reextraction_rows).to_csv(
        output_dir / "excluded_reextraction_rows_v4_3.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return manifest
