"""Reviewer-driven v4.3 analyses for the locked LS-PSRMoE task."""

from __future__ import annotations

import hashlib
import itertools
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupKFold

from .metrics import paired_cluster_bootstrap, regression_metrics
from .robustness import _xgb_ensemble_prediction, clip_angle
from .v41 import V41Preprocessor, _json_default, _load_primary_samples


MODEL_COLUMNS = {
    "nnls_physics": "physics_prediction",
    "physics_residual_xgboost": "xgboost_prediction",
    "physics_residual_random_forest": "random_forest_prediction",
    "direct_xgboost": "xgboost_direct_prediction",
    "ls_psrmoe_fusion": "theta_pred_deg",
}


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path).resolve()


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _nested_oof_path(v41_dir: Path) -> Path:
    """Resolve the locked nested OOF artifact across v4.1 filename revisions."""
    candidates = [
        v41_dir / "nested_oof_evaluated_v4_1.csv",
        v41_dir / "nested_oof_predictions_v4_1.csv",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Missing locked nested OOF predictions; checked: "
        + ", ".join(str(path) for path in candidates)
    )


def _probe_liquids(value: Any) -> set[str]:
    return {
        item.strip()
        for item in str(value).split(";")
        if item.strip() and item.strip().casefold() != "nan"
    }


def validate_provenance_decisions(
    primary: pd.DataFrame,
    decisions: pd.DataFrame,
    permitted: Sequence[str],
) -> pd.DataFrame:
    """Require an explicit audit decision for every legacy source used by the paper."""
    legacy_splits = {"train", "validation", "internal_test", "legacy_external"}
    required = sorted(
        primary.loc[primary["v4_split"].isin(legacy_splits), "source_group_id"]
        .astype(str)
        .unique()
    )
    required_columns = {
        "source_group_id", "decision", "bibliographic_status", "location_status",
        "license_status", "evidence_url", "audit_note",
    }
    missing_columns = required_columns - set(decisions.columns)
    if missing_columns:
        raise ValueError(f"Provenance review is missing columns: {sorted(missing_columns)}")
    if decisions["source_group_id"].astype(str).duplicated().any():
        raise ValueError("Provenance review contains duplicate source_group_id values")
    indexed = decisions.set_index(decisions["source_group_id"].astype(str), drop=False)
    missing = sorted(set(required) - set(indexed.index))
    if missing:
        raise ValueError(f"Missing provenance decisions for main-analysis sources: {missing}")
    reviewed = indexed.loc[required].reset_index(drop=True)
    invalid = reviewed.loc[~reviewed["decision"].astype(str).isin(set(permitted))]
    if not invalid.empty:
        sources = invalid["source_group_id"].astype(str).tolist()
        raise RuntimeError(
            "Main-analysis sources failed the provenance gate and require a cleaned-data rerun: "
            f"{sources}"
        )
    if reviewed["evidence_url"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("Every retained source requires a non-empty evidence_url")
    return reviewed


def build_dataset_profiles(tables: Any, primary: pd.DataFrame) -> dict[str, pd.DataFrame]:
    samples = primary.copy()
    measurements = tables.measurements.rename(
        columns={"measurement_id": "target_measurement_id"}
    )
    frame = (
        samples.merge(
            measurements,
            on=["target_measurement_id", "source_group_id", "surface_group_id"],
            how="left",
            validate="many_to_one",
            suffixes=("", "_measurement"),
        )
        .merge(
            tables.surfaces,
            on=["surface_group_id", "source_group_id"],
            how="left",
            validate="many_to_one",
            suffixes=("", "_surface"),
        )
        .merge(
            tables.liquids,
            left_on="target_liquid_id",
            right_on="liquid_id",
            how="left",
            validate="many_to_one",
        )
    )
    cohort = (
        frame.groupby("v4_split", sort=True)
        .agg(
            measurements=("sample_id", "size"),
            sources=("source_group_id", "nunique"),
            surfaces=("surface_group_id", "nunique"),
            liquids=("target_liquid_id", "nunique"),
        )
        .reset_index()
    )
    liquid = (
        frame.groupby(["v4_split", "liquid_name"], sort=True)
        .agg(
            n=("sample_id", "size"),
            sources=("source_group_id", "nunique"),
            surfaces=("surface_group_id", "nunique"),
        )
        .reset_index()
    )
    material = (
        frame.groupby(["v4_split", "solid_family"], sort=True)
        .agg(
            n=("sample_id", "size"),
            sources=("source_group_id", "nunique"),
            surfaces=("surface_group_id", "nunique"),
        )
        .reset_index()
    )
    categorical_rows = []
    for field in [
        "contact_angle_type", "measurement_method", "quality_grade", "conflict_flag",
    ]:
        values = frame[field].fillna("<missing>").astype(str)
        for value, count in values.value_counts(dropna=False).items():
            categorical_rows.append({"field": field, "value": value, "n": int(count)})
    metadata_rows = []
    for field in [
        "temperature_K", "humidity_percent", "pressure_atm", "droplet_volume_uL",
        "replicates_n", "contact_angle_std_deg", "roughness_Ra_nm",
        "roughness_Rq_nm", "roughness_r_factor",
    ]:
        present = int(frame[field].notna().sum())
        metadata_rows.append({
            "field": field,
            "present_n": present,
            "total_n": int(len(frame)),
            "completeness_fraction": present / len(frame),
        })
    frame["has_quantitative_roughness"] = frame[
        ["roughness_Ra_nm", "roughness_Rq_nm", "roughness_r_factor"]
    ].notna().any(axis=1)
    roughness = (
        frame.groupby(["v4_split", "has_quantitative_roughness"], sort=True)
        .agg(
            n=("sample_id", "size"),
            sources=("source_group_id", "nunique"),
            surfaces=("surface_group_id", "nunique"),
        )
        .reset_index()
    )
    return {
        "cohorts": cohort,
        "liquids": liquid,
        "materials": material,
        "categorical": pd.DataFrame(categorical_rows),
        "metadata_completeness": pd.DataFrame(metadata_rows),
        "roughness_coverage": roughness,
    }


def _fit_residual_xgb(
    tables: Any,
    train_samples: pd.DataFrame,
    test_samples: pd.DataFrame,
    seeds: Sequence[int],
    config: dict[str, Any],
) -> pd.DataFrame:
    if set(train_samples["source_group_id"]) & set(test_samples["source_group_id"]):
        source_disjoint = False
    else:
        source_disjoint = True
    preprocessor = V41Preprocessor().fit(tables, train_samples)
    train = preprocessor.transform(tables, train_samples)
    test = preprocessor.transform(tables, test_samples)
    residual = train.target_angle.astype(float) - train.physics_angle.astype(float)
    correction = _xgb_ensemble_prediction(
        train.tabular, residual, test.tabular, seeds, config
    )
    prediction = clip_angle(test.physics_angle, correction)
    return pd.DataFrame({
        "sample_id": test.sample_ids,
        "source_group_id": test.source_group_ids,
        "surface_group_id": test.surface_group_ids,
        "target_liquid_id": test.target_liquid_ids,
        "solid_family": test.solid_families,
        "theta_observed_deg": test.target_angle,
        "physics_prediction": test.physics_angle,
        "xgboost_prediction": prediction,
        "source_disjoint": np.where(source_disjoint, "yes", "no"),
    })


def source_learning_curve(
    tables: Any,
    development: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    revision = config["revision"]
    all_sources = sorted(development["source_group_id"].astype(str).unique())
    rng = np.random.default_rng(int(config["project"]["seed"]))
    frames = []
    for n_sources in revision["learning_curve_source_counts"]:
        combinations = list(itertools.combinations(all_sources, int(n_sources)))
        maximum = int(revision["learning_curve_max_subsets_per_count"])
        if len(combinations) > maximum:
            selected = np.sort(rng.choice(len(combinations), size=maximum, replace=False))
            combinations = [combinations[index] for index in selected]
        for subset_index, training_sources in enumerate(combinations):
            train = development.loc[
                development["source_group_id"].astype(str).isin(training_sources)
            ].reset_index(drop=True)
            test = development.loc[
                ~development["source_group_id"].astype(str).isin(training_sources)
            ].reset_index(drop=True)
            predicted = _fit_residual_xgb(
                tables, train, test,
                [int(value) for value in revision["learning_curve_seeds"]],
                config,
            )
            predicted["n_training_sources"] = int(n_sources)
            predicted["subset_index"] = subset_index
            predicted["training_sources"] = ";".join(training_sources)
            frames.append(predicted)
    predictions = pd.concat(frames, ignore_index=True)
    rows = []
    for (n_sources, subset_index), frame in predictions.groupby(
        ["n_training_sources", "subset_index"], sort=True
    ):
        for model, column in {
            "nnls_physics": "physics_prediction",
            "physics_residual_xgboost": "xgboost_prediction",
        }.items():
            rows.append({
                "n_training_sources": int(n_sources),
                "subset_index": int(subset_index),
                "model": model,
                "test_sources": int(frame["source_group_id"].nunique()),
                **regression_metrics(
                    frame["theta_observed_deg"].to_numpy(float),
                    frame[column].to_numpy(float),
                ),
            })
    return predictions, pd.DataFrame(rows)


def strict_leave_one_liquid(
    tables: Any,
    development: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    name_to_id = {
        str(row.liquid_name).casefold(): str(row.liquid_id)
        for row in tables.liquids.itertuples(index=False)
    }
    frames = []
    diagnostic_rows = []
    for name in config["revision"]["strict_leave_one_liquid_names"]:
        liquid_id = name_to_id[str(name).casefold()]
        contains = development["probe_liquid_ids"].map(
            lambda value: liquid_id in _probe_liquids(value)
        )
        train = development.loc[
            development["target_liquid_id"].astype(str).ne(liquid_id) & ~contains
        ].reset_index(drop=True)
        test = development.loc[
            development["target_liquid_id"].astype(str).eq(liquid_id)
        ].reset_index(drop=True)
        if test.empty or train["source_group_id"].nunique() < 2:
            diagnostic_rows.append({
                "held_out_liquid": name,
                "status": "infeasible_after_strict_target_and_probe_exclusion",
                "model": "not_fitted",
                "n_training_samples": int(len(train)),
                "n_training_sources": int(train["source_group_id"].nunique()),
                "n_test_samples": int(len(test)),
                "n_test_sources": int(test["source_group_id"].nunique()),
            })
            continue
        predicted = _fit_residual_xgb(
            tables, train, test,
            [int(value) for value in config["robustness"]["seeds"]],
            config,
        )
        predicted["held_out_liquid"] = name
        predicted["n_training_samples"] = len(train)
        predicted["n_training_sources"] = int(train["source_group_id"].nunique())
        predicted["held_out_liquid_seen_as_target_or_probe_in_training"] = "no"
        frames.append(predicted)
    predictions = pd.concat(frames, ignore_index=True)
    rows = list(diagnostic_rows)
    for liquid, frame in predictions.groupby("held_out_liquid", sort=True):
        for model, column in {
            "nnls_physics": "physics_prediction",
            "physics_residual_xgboost": "xgboost_prediction",
        }.items():
            rows.append({
                "held_out_liquid": liquid,
                "status": "evaluated",
                "model": model,
                "sources": int(frame["source_group_id"].nunique()),
                **regression_metrics(
                    frame["theta_observed_deg"].to_numpy(float),
                    frame[column].to_numpy(float),
                ),
            })
    return predictions, pd.DataFrame(rows)


def _novelty_label(material_known: bool, liquid_known: bool) -> str:
    if material_known and liquid_known:
        return "known_material_known_target_liquid"
    if not material_known and liquid_known:
        return "new_material_family"
    if material_known and not liquid_known:
        return "new_target_liquid"
    return "new_material_family_and_target_liquid"


def generalization_strata(
    development: pd.DataFrame,
    nested: pd.DataFrame,
    confirmation: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    annotated = []
    for fold, frame in nested.groupby("outer_fold", sort=True):
        held_sources = set(frame["source_group_id"].astype(str))
        train = development.loc[
            ~development["source_group_id"].astype(str).isin(held_sources)
        ]
        known_materials = set(train["solid_family"].astype(str))
        known_liquids = set(train["target_liquid_id"].astype(str))
        copy = frame.copy()
        copy["novelty_stratum"] = [
            _novelty_label(
                str(material) in known_materials,
                str(liquid) in known_liquids,
            )
            for material, liquid in zip(copy["solid_family"], copy["target_liquid_id"])
        ]
        copy["evaluation"] = "nested_source_cv"
        annotated.append(copy)
    known_materials = set(development["solid_family"].astype(str))
    known_liquids = set(development["target_liquid_id"].astype(str))
    external = confirmation.copy()
    external["novelty_stratum"] = [
        _novelty_label(
            str(material) in known_materials,
            str(liquid) in known_liquids,
        )
        for material, liquid in zip(external["solid_family"], external["target_liquid_id"])
    ]
    external["evaluation"] = external["v4_split"].astype(str)
    annotated.append(external)
    output = pd.concat(annotated, ignore_index=True)
    rows = []
    for (evaluation, stratum), frame in output.groupby(
        ["evaluation", "novelty_stratum"], sort=True
    ):
        for model, column in MODEL_COLUMNS.items():
            if column not in frame:
                continue
            rows.append({
                "evaluation": evaluation,
                "novelty_stratum": stratum,
                "model": model,
                "sources": int(frame["source_group_id"].nunique()),
                **regression_metrics(
                    frame["theta_observed_deg"].to_numpy(float),
                    frame[column].to_numpy(float),
                ),
            })
    return output, pd.DataFrame(rows)


def bootstrap_comparisons(
    predictions: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    rows = []
    comparisons = [
        ("physics_residual_xgboost", "nnls_physics"),
        ("physics_residual_xgboost", "direct_xgboost"),
        ("physics_residual_xgboost", "physics_residual_random_forest"),
        ("physics_residual_xgboost", "ls_psrmoe_fusion"),
    ]
    for evaluation, frame in predictions.groupby("evaluation", sort=True):
        for model_a, model_b in comparisons:
            column_a, column_b = MODEL_COLUMNS[model_a], MODEL_COLUMNS[model_b]
            if column_a not in frame or column_b not in frame:
                continue
            for cluster in ["surface_group_id", "source_group_id"]:
                rows.append({
                    "evaluation": evaluation,
                    "model_a": model_a,
                    "model_b": model_b,
                    "cluster": cluster,
                    **paired_cluster_bootstrap(
                        frame["theta_observed_deg"].to_numpy(float),
                        frame[column_a].to_numpy(float),
                        frame[column_b].to_numpy(float),
                        frame[cluster].astype(str).to_numpy(),
                        int(config["robustness"]["bootstrap_resamples"]),
                        int(config["project"]["seed"]),
                    ),
                })
    return pd.DataFrame(rows)


def practical_accuracy(
    predictions: pd.DataFrame,
    thresholds: Iterable[float],
) -> pd.DataFrame:
    rows = []
    for evaluation, frame in predictions.groupby("evaluation", sort=True):
        observed = frame["theta_observed_deg"].to_numpy(float)
        for model, column in MODEL_COLUMNS.items():
            if column not in frame:
                continue
            absolute = np.abs(observed - frame[column].to_numpy(float))
            row = {
                "evaluation": evaluation,
                "model": model,
                **regression_metrics(observed, frame[column].to_numpy(float)),
            }
            for threshold in thresholds:
                row[f"fraction_abs_error_le_{float(threshold):g}_deg"] = float(
                    np.mean(absolute <= float(threshold))
                )
            rows.append(row)
    return pd.DataFrame(rows)


def run_revision(config_path: Path) -> dict[str, Any]:
    started = time.time()
    config_path = config_path.resolve()
    project_root = config_path.parent.parent
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = _resolve(project_root, config["project"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    tables, primary, _ = _load_primary_samples(project_root, config)
    development = primary.loc[
        primary["v4_split"].isin(config["sample"]["development_splits"])
    ].reset_index(drop=True)

    provenance_path = _resolve(project_root, config["project"]["provenance_review_file"])
    decisions = pd.read_csv(provenance_path, encoding="utf-8-sig")
    reviewed = validate_provenance_decisions(
        primary, decisions, config["revision"]["permitted_provenance_decisions"]
    )
    reviewed.to_csv(
        output_dir / "source_provenance_audit_v4_3.csv",
        index=False, encoding="utf-8-sig",
    )

    profiles = build_dataset_profiles(tables, primary)
    with pd.ExcelWriter(output_dir / "dataset_distribution_v4_3.xlsx") as writer:
        for name, frame in profiles.items():
            frame.to_excel(writer, sheet_name=name[:31], index=False)
            frame.to_csv(
                output_dir / f"dataset_{name}_v4_3.csv",
                index=False, encoding="utf-8-sig",
            )

    learning_predictions, learning_metrics = source_learning_curve(
        tables, development, config
    )
    learning_predictions.to_csv(
        output_dir / "source_learning_curve_predictions_v4_3.csv",
        index=False, encoding="utf-8-sig",
    )
    learning_metrics.to_csv(
        output_dir / "source_learning_curve_metrics_v4_3.csv",
        index=False, encoding="utf-8-sig",
    )

    liquid_predictions, liquid_metrics = strict_leave_one_liquid(
        tables, development, config
    )
    liquid_predictions.to_csv(
        output_dir / "strict_leave_one_liquid_predictions_v4_3.csv",
        index=False, encoding="utf-8-sig",
    )
    liquid_metrics.to_csv(
        output_dir / "strict_leave_one_liquid_metrics_v4_3.csv",
        index=False, encoding="utf-8-sig",
    )

    v41_dir = _resolve(project_root, config["project"]["v41_output_dir"])
    nested_path = _nested_oof_path(v41_dir)
    nested = pd.read_csv(nested_path, encoding="utf-8-sig")
    confirmation = pd.read_csv(
        v41_dir / "predictions_v4_1.csv", encoding="utf-8-sig"
    )
    confirmation = confirmation.loc[
        confirmation["v4_split"].isin(config["sample"]["confirmation_splits"])
    ].reset_index(drop=True)
    development_context = development.merge(
        tables.surfaces[["surface_group_id", "solid_family"]],
        on="surface_group_id",
        how="left",
        validate="many_to_one",
    )
    annotated, strata = generalization_strata(
        development_context, nested, confirmation
    )
    annotated.to_csv(
        output_dir / "generalization_annotated_predictions_v4_3.csv",
        index=False, encoding="utf-8-sig",
    )
    strata.to_csv(
        output_dir / "generalization_strata_metrics_v4_3.csv",
        index=False, encoding="utf-8-sig",
    )

    bootstrap = bootstrap_comparisons(annotated, config)
    bootstrap.to_csv(
        output_dir / "paired_bootstrap_v4_3.csv",
        index=False, encoding="utf-8-sig",
    )
    practical = practical_accuracy(
        annotated, config["revision"]["practical_error_thresholds_deg"]
    )
    practical.to_csv(
        output_dir / "practical_accuracy_v4_3.csv",
        index=False, encoding="utf-8-sig",
    )

    calibration = json.loads(
        (v41_dir / "calibration_v4_1.json").read_text(encoding="utf-8")
    )
    applicability = {
        "hard_eligibility": {
            "minimum_distinct_non_target_probes": 2,
            "target_liquid_must_be_removed": True,
            "accepted_nnls_statuses": ["interior_fit", "boundary_fit"],
        },
        "development_oof_thresholds": {
            "ood_distance_p95": calibration["ood_raw_distance_threshold"],
            "ensemble_uncertainty_p90_deg": calibration[
                "abstention_score_threshold_deg"
            ],
        },
        "unvalidated_domains": [
            "highly rough surfaces",
            "porous surfaces",
            "anisotropic surfaces",
            "chemically heterogeneous surfaces",
            "dynamic wetting conditions",
        ],
        "roughness_threshold_available": False,
        "reason": (
            "Quantitative roughness is source-confounded and too sparsely reported "
            "for a defensible numerical cutoff."
        ),
    }
    (output_dir / "applicability_rules_v4_3.json").write_text(
        json.dumps(applicability, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    outputs = sorted(path for path in output_dir.iterdir() if path.is_file())
    manifest = {
        "status": "complete",
        "model_version": config["project"]["model_version"],
        "python": sys.version,
        "platform": platform.platform(),
        "runtime_seconds": time.time() - started,
        "n_primary_samples": len(primary),
        "n_development_samples": len(development),
        "n_development_sources": int(development["source_group_id"].nunique()),
        "n_provenance_decisions": int(decisions["source_group_id"].nunique()),
        "n_provenance_reviewed_active_sources": len(reviewed),
        "confirmation_used_for_model_selection": False,
        "input_hashes": {
            provenance_path.name: _hash(provenance_path),
            nested_path.name: _hash(nested_path),
            "predictions_v4_1.csv": _hash(v41_dir / "predictions_v4_1.csv"),
        },
        "output_hashes": {
            path.name: _hash(path)
            for path in outputs
            if path.name != "run_manifest_v4_3.json"
        },
    }
    (output_dir / "run_manifest_v4_3.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    return manifest
