"""v4 scientific tables, grouped splits, target masking, and migration audits."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .physics import fit_owrk_nnls, legacy_unconstrained_coefficients, owens_wendt_angle
from .schema import LiquidDescriptor, ProbeMeasurement


INDEPENDENT_SFE_TYPES = {"literature_reported", "independently_measured"}
ACTIVE_SPLITS = {"train", "validation", "internal_test", "legacy_external", "prospective_open_external"}
LIQUID_ALIASES = {
    "methylene iodide": "diiodomethane",
    "methylene iodine": "diiodomethane",
    "diiodo methane": "diiodomethane",
    "glycerin": "glycerol",
    "di water": "water",
    "deionized water": "water",
    "distilled water": "water",
}


def clean_text(value: Any) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return " ".join(str(value).strip().split())


def normalized_token(value: Any) -> str:
    return clean_text(value).casefold()


def stable_id(prefix: str, values: Iterable[Any], length: int = 12) -> str:
    payload = "\x1f".join(normalized_token(value) for value in values)
    return f"{prefix}_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:length].upper()}"


def canonical_liquid_name(value: Any) -> str:
    name = normalized_token(value)
    return LIQUID_ALIASES.get(name, name)


def safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def build_surface_id(row: pd.Series) -> str:
    source_identity = (
        clean_text(row.get("reference_doi"))
        or clean_text(row.get("reference_url"))
        or clean_text(row.get("source_group_id"))
    )
    return stable_id(
        "SFC",
        [
            source_identity,
            row.get("solid_name"),
            row.get("solid_substrate"),
            row.get("surface_treatment"),
            row.get("surface_treatment_detail"),
            row.get("coating_or_layer"),
            row.get("surface_state"),
        ],
    )


def build_liquid_id(name: str) -> str:
    return stable_id("LIQ", [canonical_liquid_name(name)], length=10)


def _source_level_split(source_counts: pd.Series, seed: int) -> dict[str, str]:
    if not source_counts.size:
        return {}
    fractions = {"train": 0.70, "validation": 0.15, "internal_test": 0.15}
    rng = np.random.default_rng(seed)
    order = pd.DataFrame({"source": source_counts.index, "count": source_counts.values})
    order["jitter"] = rng.random(len(order))
    order = order.sort_values(["count", "jitter"], ascending=[False, True])
    targets = {name: float(source_counts.sum()) * fraction for name, fraction in fractions.items()}
    assigned = {name: 0 for name in fractions}
    mapping: dict[str, str] = {}
    for row in order.itertuples(index=False):
        split = max(fractions, key=lambda name: (targets[name] - assigned[name], fractions[name]))
        mapping[str(row.source)] = split
        assigned[split] += int(row.count)
    return mapping


def build_v4_splits(raw: pd.DataFrame, seed: int) -> pd.DataFrame:
    active = raw.loc[raw["analysis_split"] != "excluded_review"].copy()
    external_sources = set(
        active.loc[active["analysis_split"] == "source_disjoint_external", "source_group_id"].astype(str)
    )
    development = active.loc[~active["source_group_id"].astype(str).isin(external_sources)]
    mapping = _source_level_split(development.groupby("source_group_id").size(), seed)
    mapping.update({source: "legacy_external" for source in external_sources})
    rows = []
    for row in raw.itertuples(index=False):
        legacy_split = clean_text(getattr(row, "analysis_split"))
        v4_split = "excluded_review" if legacy_split == "excluded_review" else mapping[str(row.source_group_id)]
        rows.append(
            {
                "measurement_id": str(row.record_id),
                "record_id": str(row.record_id),
                "source_group_id": str(row.source_group_id),
                "surface_group_id": str(row.surface_group_id),
                "legacy_analysis_split": legacy_split,
                "v4_split": v4_split,
                "applsci_split": v4_split,
                "allow_training": "yes" if v4_split == "train" else "no",
                "allow_tuning": "yes" if v4_split == "validation" else "no",
                "allow_evaluation": "yes" if v4_split in {"internal_test", "legacy_external", "prospective_open_external"} else "no",
                "freeze_status": "provisional_v4_bootstrap",
            }
        )
    return pd.DataFrame(rows)


def _first_nonempty(series: pd.Series) -> Any:
    for value in series.dropna():
        if clean_text(value):
            return value
    return ""


def migrate_legacy_dataset(legacy_csv: Path, output_dir: Path, seed: int = 20260710) -> dict[str, Path]:
    raw = pd.read_csv(legacy_csv, encoding="utf-8-sig")
    raw["source_group_id"] = raw["source_group_id"].astype(str)
    raw["surface_group_id"] = raw.apply(build_surface_id, axis=1)
    raw["canonical_liquid_name"] = raw["liquid_name"].map(canonical_liquid_name)
    raw["liquid_id"] = raw["canonical_liquid_name"].map(build_liquid_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_fields = [
        "reference_doi", "reference_url", "source_type", "reference_title",
        "reference_authors", "reference_year", "collection_status",
    ]
    sources = raw.groupby("source_group_id", as_index=False)[source_fields].agg(_first_nonempty)
    sources["license"] = "source_license_to_verify"
    sources["extraction_status"] = "legacy_curated_requires_v4_review"
    sources["new_external_source_flag"] = "no"
    sources["applsci_split"] = "provisional_legacy_migration"

    surface_fields = [
        "source_group_id", "solid_name", "solid_family", "solid_substrate",
        "surface_treatment", "surface_treatment_detail", "coating_or_layer", "surface_state",
        "roughness_Ra_nm", "roughness_Rq_nm", "roughness_r_factor", "sample_preparation_notes",
    ]
    surfaces = raw.groupby("surface_group_id", as_index=False)[surface_fields].agg(_first_nonempty)

    liquid_fields = [
        "canonical_liquid_name", "liquid_family", "liquid_total_surface_tension_mN_m",
        "liquid_dispersion_mN_m", "liquid_polar_mN_m", "liquid_LW_mN_m",
        "liquid_acid_plus_mN_m", "liquid_base_minus_mN_m", "liquid_viscosity_mPa_s",
        "liquid_dipole_moment_D", "liquid_dielectric_constant", "liquid_property_source",
    ]
    liquids = raw.groupby("liquid_id", as_index=False)[liquid_fields].agg(_first_nonempty)
    liquids = liquids.rename(columns={"canonical_liquid_name": "liquid_name"})

    measurement_fields = [
        "record_id", "surface_group_id", "source_group_id", "liquid_id", "contact_angle_deg",
        "contact_angle_type", "measurement_method", "temperature_K", "humidity_percent",
        "pressure_atm", "droplet_volume_uL", "replicates_n", "contact_angle_std_deg",
        "contact_angle_min_deg", "contact_angle_max_deg", "quality_grade", "conflict_flag",
        "solid_total_surface_energy_mJ_m2", "solid_dispersion_mJ_m2", "solid_polar_mJ_m2",
        "solid_surface_energy_source", "solid_surface_energy_source_type", "data_extraction_note",
    ]
    measurements = raw[measurement_fields].rename(columns={"record_id": "measurement_id"}).copy()
    measurements["record_id"] = measurements["measurement_id"]
    measurements["sfe_source_type"] = measurements["solid_surface_energy_source_type"]
    measurements["new_external_source_flag"] = "no"
    measurements["target_eligible"] = np.where(
        measurements["quality_grade"].astype(str).str.casefold().eq("c_low"), "diagnostic_only", "yes"
    )
    splits = build_v4_splits(raw, seed)

    paths = {name: output_dir / f"{name}_v4.csv" for name in ["sources", "surfaces", "liquids", "measurements", "splits"]}
    for name, frame in [
        ("sources", sources), ("surfaces", surfaces), ("liquids", liquids),
        ("measurements", measurements), ("splits", splits),
    ]:
        frame.to_csv(paths[name], index=False, encoding="utf-8-sig")
    (output_dir / "splits_v4.sha256").write_text(
        hashlib.sha256(paths["splits"].read_bytes()).hexdigest() + "\n", encoding="ascii"
    )
    return paths


@dataclass
class V4Tables:
    sources: pd.DataFrame
    surfaces: pd.DataFrame
    liquids: pd.DataFrame
    measurements: pd.DataFrame
    splits: pd.DataFrame

    @classmethod
    def load(cls, data_dir: Path) -> "V4Tables":
        filenames = ["sources_v4.csv", "surfaces_v4.csv", "liquids_v4.csv", "measurements_v4.csv", "splits_v4.csv"]
        return cls(*[pd.read_csv(data_dir / name, encoding="utf-8-sig") for name in filenames])


def build_samples(tables: V4Tables, independent_sfe_types: set[str] | None = None) -> pd.DataFrame:
    independent_sfe_types = independent_sfe_types or INDEPENDENT_SFE_TYPES
    measurements = tables.measurements.merge(
        tables.splits[["measurement_id", "v4_split"]], on="measurement_id", how="left", validate="one_to_one"
    )
    by_surface = {key: group.copy() for key, group in measurements.groupby("surface_group_id")}
    rows: list[dict[str, Any]] = []
    for target in measurements.itertuples(index=False):
        sfe_type = clean_text(target.solid_surface_energy_source_type)
        sfe_dispersion = safe_float(target.solid_dispersion_mJ_m2)
        sfe_polar = safe_float(target.solid_polar_mJ_m2)
        independent = (
            sfe_type in independent_sfe_types
            and sfe_dispersion is not None and sfe_dispersion >= 0.0
            and sfe_polar is not None and sfe_polar >= 0.0
        )
        base = {
            "target_measurement_id": target.measurement_id,
            "record_id": target.record_id,
            "source_group_id": target.source_group_id,
            "surface_group_id": target.surface_group_id,
            "target_liquid_id": target.liquid_id,
            "target_contact_angle_deg": target.contact_angle_deg,
            "v4_split": target.v4_split,
            "has_independent_sfe": "yes" if independent else "no",
            "independent_sfe_dispersion_mj_m2": sfe_dispersion if independent else np.nan,
            "independent_sfe_polar_mj_m2": sfe_polar if independent else np.nan,
        }
        rows.append({
            **base, "sample_id": f"ZS_{target.measurement_id}", "prediction_mode": "zero_shot",
            "probe_measurement_ids": "", "probe_liquid_ids": "", "n_probes": 0,
            "target_liquid_removed": "no", "loo_sfe_feasible": "no",
            "loo_sfe_failure_reason": "zero_shot_not_applicable",
        })
        candidates = by_surface[target.surface_group_id]
        probes = candidates.loc[
            (candidates["liquid_id"] != target.liquid_id)
            & (candidates["v4_split"] == target.v4_split)
            & (candidates["target_eligible"] == "yes")
        ]
        if len(probes):
            rows.append({
                **base, "sample_id": f"PA_{target.measurement_id}", "prediction_mode": "probe_assisted",
                "probe_measurement_ids": ";".join(sorted(probes["measurement_id"].astype(str))),
                "probe_liquid_ids": ";".join(sorted(set(probes["liquid_id"].astype(str)))),
                "n_probes": len(probes), "target_liquid_removed": "yes",
                "loo_sfe_feasible": "pending", "loo_sfe_failure_reason": "pending_nnls_audit",
            })
    return pd.DataFrame(rows)


def liquid_from_row(row: pd.Series) -> LiquidDescriptor:
    return LiquidDescriptor(
        liquid_id=str(row["liquid_id"]), name=clean_text(row["liquid_name"]),
        total_surface_tension=float(row["liquid_total_surface_tension_mN_m"]),
        dispersion_component=float(row["liquid_dispersion_mN_m"]),
        polar_component=float(row["liquid_polar_mN_m"]),
        viscosity_mpa_s=safe_float(row.get("liquid_viscosity_mPa_s")),
        dipole_moment_d=safe_float(row.get("liquid_dipole_moment_D")),
        dielectric_constant=safe_float(row.get("liquid_dielectric_constant")),
    )


def build_nnls_audit(tables: V4Tables, samples: pd.DataFrame) -> pd.DataFrame:
    liquids = {str(row.liquid_id): liquid_from_row(row) for _, row in tables.liquids.iterrows()}
    measurements = tables.measurements.set_index("measurement_id", drop=False)
    rows: list[dict[str, Any]] = []
    for sample in samples.loc[samples["prediction_mode"] == "probe_assisted"].itertuples(index=False):
        probe_ids = [item for item in str(sample.probe_measurement_ids).split(";") if item and item != "nan"]
        probes = []
        for measurement_id in probe_ids:
            row = measurements.loc[measurement_id]
            probes.append(ProbeMeasurement(
                measurement_id=measurement_id, liquid=liquids[str(row.liquid_id)],
                contact_angle_deg=float(row.contact_angle_deg),
                contact_angle_std_deg=safe_float(row.contact_angle_std_deg),
                replicates_n=int(row.replicates_n) if safe_float(row.replicates_n) is not None else None,
            ))
        fit = fit_owrk_nnls(probes)
        legacy = legacy_unconstrained_coefficients(probes)
        target_row = measurements.loc[sample.target_measurement_id]
        prediction: float | str = ""
        legacy_prediction: float | str = ""
        legacy_dispersion: float | str = ""
        legacy_polar: float | str = ""
        if fit.dispersion_mj_m2 is not None and fit.polar_mj_m2 is not None:
            prediction = owens_wendt_angle(
                fit.dispersion_mj_m2, fit.polar_mj_m2, liquids[str(target_row.liquid_id)]
            )
        if legacy:
            legacy_dispersion = legacy[0] ** 2
            legacy_polar = legacy[1] ** 2
            legacy_prediction = owens_wendt_angle(
                legacy_dispersion, legacy_polar, liquids[str(target_row.liquid_id)]
            )
        rows.append({
            "sample_id": sample.sample_id, "record_id": sample.record_id,
            "surface_group_id": sample.surface_group_id, "source_group_id": sample.source_group_id,
            "v4_split": sample.v4_split, "target_liquid_id": sample.target_liquid_id,
            "n_probes": fit.n_probes, "n_unique_liquids": fit.n_unique_liquids,
            "fit_status": fit.status, "boundary_fit": "yes" if fit.boundary_fit else "no",
            "nnls_dispersion_mj_m2": fit.dispersion_mj_m2, "nnls_polar_mj_m2": fit.polar_mj_m2,
            "nnls_total_mj_m2": fit.total_mj_m2, "nnls_residual_norm": fit.residual_norm,
            "legacy_sqrt_dispersion": legacy[0] if legacy else "",
            "legacy_sqrt_polar": legacy[1] if legacy else "",
            "legacy_squared_dispersion_mj_m2": legacy_dispersion,
            "legacy_squared_polar_mj_m2": legacy_polar,
            "legacy_negative_coefficient": "yes" if legacy and min(legacy) < 0 else "no",
            "target_contact_angle_deg": sample.target_contact_angle_deg,
            "nnls_physical_prediction_deg": prediction,
            "legacy_squared_physical_prediction_deg": legacy_prediction,
        })
    return pd.DataFrame(rows)


def validate_v4_tables(tables: V4Tables, samples: pd.DataFrame) -> dict[str, Any]:
    errors: list[str] = []
    active = tables.splits.loc[tables.splits["v4_split"].isin(ACTIVE_SPLITS)]
    for field in ["source_group_id", "surface_group_id"]:
        overlaps = active.groupby(field)["v4_split"].nunique()
        if (overlaps > 1).any():
            errors.append(f"{field} crosses active v4 splits: {int((overlaps > 1).sum())}")
    for sample in samples.itertuples(index=False):
        probe_liquids = {item for item in str(sample.probe_liquid_ids).split(";") if item and item != "nan"}
        if str(sample.target_liquid_id) in probe_liquids:
            errors.append(f"Target liquid leaked into probes for {sample.sample_id}")
        if sample.prediction_mode == "zero_shot" and int(sample.n_probes) != 0:
            errors.append(f"Zero-shot sample contains probes: {sample.sample_id}")
    return {
        "status": "pass" if not errors else "fail", "errors": errors,
        "n_sources": int(len(tables.sources)), "n_surfaces": int(len(tables.surfaces)),
        "n_liquids": int(len(tables.liquids)), "n_measurements": int(len(tables.measurements)),
        "n_zero_shot_samples": int((samples["prediction_mode"] == "zero_shot").sum()),
        "n_probe_assisted_samples": int((samples["prediction_mode"] == "probe_assisted").sum()),
        "split_counts": {str(k): int(v) for k, v in tables.splits["v4_split"].value_counts().items()},
    }


def write_migration_artifacts(legacy_csv: Path, data_dir: Path, audit_dir: Path, seed: int) -> dict[str, Any]:
    migrate_legacy_dataset(legacy_csv, data_dir, seed)
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
            "nnls_dispersion_mj_m2", "nnls_polar_mj_m2", "nnls_physical_prediction_deg"
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
    if validation["status"] != "pass":
        raise RuntimeError("v4 migration validation failed: " + "; ".join(validation["errors"]))
    return validation
