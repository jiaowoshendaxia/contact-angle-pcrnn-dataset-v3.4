"""Leakage-safe feature fitting and dual-mode sample encoding."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .data import V4Tables, clean_text


SURFACE_NUMERIC = ["roughness_Ra_nm", "roughness_Rq_nm", "roughness_r_factor"]
SURFACE_CATEGORICAL = [
    "solid_family",
    "solid_substrate",
    "surface_treatment",
    "coating_or_layer",
    "surface_state",
]
LIQUID_NUMERIC = [
    "liquid_total_surface_tension_mN_m",
    "liquid_dispersion_mN_m",
    "liquid_polar_mN_m",
    "liquid_viscosity_mPa_s",
    "liquid_dipole_moment_D",
    "liquid_dielectric_constant",
]
CONDITION_NUMERIC = ["temperature_K", "humidity_percent", "pressure_atm", "droplet_volume_uL"]
CONDITION_CATEGORICAL = ["contact_angle_type", "measurement_method"]


@dataclass
class NumericScaler:
    means: dict[str, float]
    scales: dict[str, float]

    @classmethod
    def fit(cls, frame: pd.DataFrame, columns: Iterable[str]) -> "NumericScaler":
        means: dict[str, float] = {}
        scales: dict[str, float] = {}
        for column in columns:
            values = pd.to_numeric(frame.get(column, pd.Series(dtype=float)), errors="coerce")
            finite = values[np.isfinite(values)]
            mean = float(finite.mean()) if len(finite) else 0.0
            scale = float(finite.std(ddof=0)) if len(finite) else 1.0
            means[column] = mean
            scales[column] = scale if scale > 1e-8 else 1.0
        return cls(means=means, scales=scales)

    def transform_row(self, row: pd.Series, columns: Iterable[str]) -> list[float]:
        standardized: list[float] = []
        missing: list[float] = []
        for column in columns:
            value = pd.to_numeric(pd.Series([row.get(column)]), errors="coerce").iloc[0]
            is_missing = not np.isfinite(value)
            value = self.means[column] if is_missing else float(value)
            standardized.append((value - self.means[column]) / self.scales[column])
            missing.append(float(is_missing))
        return standardized + missing


@dataclass
class CategoryEncoder:
    vocabularies: dict[str, dict[str, int]]

    @staticmethod
    def _token(value: Any) -> str:
        return clean_text(value).casefold() or "<missing>"

    @classmethod
    def fit(cls, frame: pd.DataFrame, columns: Iterable[str]) -> "CategoryEncoder":
        vocabularies: dict[str, dict[str, int]] = {}
        for column in columns:
            tokens = sorted({cls._token(value) for value in frame.get(column, pd.Series(dtype=str))})
            vocabularies[column] = {token: index + 2 for index, token in enumerate(tokens)}
        return cls(vocabularies=vocabularies)

    def transform_row(self, row: pd.Series, columns: Iterable[str]) -> list[int]:
        return [self.vocabularies[column].get(self._token(row.get(column)), 0) for column in columns]

    def one_hot(self, codes: Iterable[int], columns: Iterable[str]) -> list[float]:
        output: list[float] = []
        for code, column in zip(codes, columns):
            size = len(self.vocabularies[column]) + 2
            values = [0.0] * size
            values[int(code)] = 1.0
            output.extend(values)
        return output

    def cardinalities(self, columns: Iterable[str]) -> list[int]:
        return [len(self.vocabularies[column]) + 2 for column in columns]


@dataclass
class EncodedSamples:
    sample_ids: list[str]
    record_ids: list[str]
    source_group_ids: list[str]
    surface_group_ids: list[str]
    modes: list[str]
    splits: list[str]
    target_liquid_ids: list[str]
    surface_numeric: np.ndarray
    categorical: np.ndarray
    target_liquid_numeric: np.ndarray
    target_liquid_physical: np.ndarray
    condition_numeric: np.ndarray
    nnls_sfe: np.ndarray
    nnls_sfe_mask: np.ndarray
    nnls_physics_prediction: np.ndarray
    probes: np.ndarray
    probe_mask: np.ndarray
    independent_sfe: np.ndarray
    independent_sfe_mask: np.ndarray
    target_angle: np.ndarray
    target_cosine: np.ndarray
    tabular: np.ndarray
    gate_context: np.ndarray

    def __len__(self) -> int:
        return len(self.sample_ids)

    def subset(self, indices: np.ndarray | list[int]) -> "EncodedSamples":
        indices = np.asarray(indices, dtype=int)
        list_fields = [
            "sample_ids", "record_ids", "source_group_ids", "surface_group_ids",
            "modes", "splits", "target_liquid_ids",
        ]
        kwargs: dict[str, Any] = {}
        for field in list_fields:
            values = getattr(self, field)
            kwargs[field] = [values[index] for index in indices]
        for field in [
            "surface_numeric", "categorical", "target_liquid_numeric", "target_liquid_physical", "condition_numeric",
            "nnls_sfe", "nnls_sfe_mask", "nnls_physics_prediction",
            "probes", "probe_mask", "independent_sfe", "independent_sfe_mask",
            "target_angle", "target_cosine", "tabular", "gate_context",
        ]:
            kwargs[field] = getattr(self, field)[indices]
        return EncodedSamples(**kwargs)


class FeaturePreprocessor:
    """Fits all imputers, scalers, and vocabularies on training sources only."""

    def __init__(self) -> None:
        self.surface_scaler: NumericScaler | None = None
        self.liquid_scaler: NumericScaler | None = None
        self.condition_scaler: NumericScaler | None = None
        self.probe_aux_scaler: NumericScaler | None = None
        self.category_encoder: CategoryEncoder | None = None
        self.max_probes = 1

    @staticmethod
    def _lookups(tables: V4Tables) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return (
            tables.surfaces.set_index("surface_group_id", drop=False),
            tables.liquids.set_index("liquid_id", drop=False),
            tables.measurements.set_index("measurement_id", drop=False),
        )

    def fit(self, tables: V4Tables, samples: pd.DataFrame) -> "FeaturePreprocessor":
        surfaces, liquids, measurements = self._lookups(tables)
        surface_rows = surfaces.loc[sorted(set(samples["surface_group_id"].astype(str)))]
        liquid_ids = set(samples["target_liquid_id"].astype(str))
        probe_ids: set[str] = set()
        for value in samples["probe_measurement_ids"].fillna(""):
            probe_ids.update(item for item in str(value).split(";") if item)
        if probe_ids:
            liquid_ids.update(measurements.loc[sorted(probe_ids), "liquid_id"].astype(str))
        liquid_rows = liquids.loc[sorted(liquid_ids)]
        target_rows = measurements.loc[samples["target_measurement_id"].astype(str)]
        probe_rows = measurements.loc[sorted(probe_ids)] if probe_ids else measurements.iloc[0:0]

        self.surface_scaler = NumericScaler.fit(surface_rows, SURFACE_NUMERIC)
        self.liquid_scaler = NumericScaler.fit(liquid_rows, LIQUID_NUMERIC)
        self.condition_scaler = NumericScaler.fit(target_rows, CONDITION_NUMERIC)
        self.probe_aux_scaler = NumericScaler.fit(
            probe_rows.rename(columns={
                "contact_angle_deg": "probe_angle",
                "contact_angle_std_deg": "probe_std",
                "replicates_n": "probe_replicates",
            }),
            ["probe_angle", "probe_std", "probe_replicates"],
        )
        category_frame = pd.concat(
            [
                surface_rows[SURFACE_CATEGORICAL].reset_index(drop=True),
                target_rows[CONDITION_CATEGORICAL].reset_index(drop=True),
            ],
            axis=1,
        )
        self.category_encoder = CategoryEncoder.fit(
            category_frame, SURFACE_CATEGORICAL + CONDITION_CATEGORICAL
        )
        # A fixed lower capacity keeps evaluation-set probe counts from changing the fitted schema.
        self.max_probes = max(8, int(samples["n_probes"].max()))
        return self

    def _require_fit(self) -> None:
        if any(item is None for item in [
            self.surface_scaler, self.liquid_scaler, self.condition_scaler,
            self.probe_aux_scaler, self.category_encoder,
        ]):
            raise RuntimeError("FeaturePreprocessor must be fitted on training data first")

    @property
    def categorical_cardinalities(self) -> list[int]:
        self._require_fit()
        assert self.category_encoder is not None
        return self.category_encoder.cardinalities(SURFACE_CATEGORICAL + CONDITION_CATEGORICAL)

    def transform(self, tables: V4Tables, samples: pd.DataFrame) -> EncodedSamples:
        self._require_fit()
        assert self.surface_scaler and self.liquid_scaler and self.condition_scaler
        assert self.probe_aux_scaler and self.category_encoder
        surfaces, liquids, measurements = self._lookups(tables)

        sample_ids: list[str] = []
        record_ids: list[str] = []
        source_ids: list[str] = []
        surface_ids: list[str] = []
        modes: list[str] = []
        splits: list[str] = []
        target_liquid_ids: list[str] = []
        surface_numeric: list[list[float]] = []
        categorical: list[list[int]] = []
        liquid_numeric: list[list[float]] = []
        liquid_physical: list[list[float]] = []
        condition_numeric: list[list[float]] = []
        nnls_sfe: list[list[float]] = []
        nnls_sfe_mask: list[float] = []
        nnls_predictions: list[float] = []
        probes_all: list[np.ndarray] = []
        probe_masks: list[np.ndarray] = []
        independent_sfe: list[list[float]] = []
        independent_sfe_mask: list[float] = []
        targets: list[float] = []
        tabular: list[list[float]] = []
        gate_context: list[list[float]] = []

        for sample in samples.itertuples(index=False):
            surface = surfaces.loc[str(sample.surface_group_id)]
            target_liquid = liquids.loc[str(sample.target_liquid_id)]
            target_measurement = measurements.loc[str(sample.target_measurement_id)]
            surface_values = self.surface_scaler.transform_row(surface, SURFACE_NUMERIC)
            liquid_values = self.liquid_scaler.transform_row(target_liquid, LIQUID_NUMERIC)
            physical_values = [
                float(target_liquid["liquid_total_surface_tension_mN_m"]),
                float(target_liquid["liquid_dispersion_mN_m"]),
                float(target_liquid["liquid_polar_mN_m"]),
            ]
            condition_values = self.condition_scaler.transform_row(target_measurement, CONDITION_NUMERIC)
            nnls_available = (
                clean_text(getattr(sample, "loo_sfe_feasible", "no")).casefold() == "yes"
                and str(sample.prediction_mode) == "probe_assisted"
            )
            nnls_d = float(getattr(sample, "nnls_dispersion_mj_m2", 0.0)) if nnls_available else 0.0
            nnls_p = float(getattr(sample, "nnls_polar_mj_m2", 0.0)) if nnls_available else 0.0
            nnls_angle = float(getattr(sample, "nnls_physical_prediction_deg", 0.0)) if nnls_available else 0.0
            category_row = pd.concat([surface, target_measurement])
            category_values = self.category_encoder.transform_row(
                category_row, SURFACE_CATEGORICAL + CONDITION_CATEGORICAL
            )

            probe_ids = [
                item for item in clean_text(sample.probe_measurement_ids).split(";") if item
            ]
            target_liquid_id = str(sample.target_liquid_id)
            probe_items: list[list[float]] = []
            raw_probe_angles: list[float] = []
            for probe_id in probe_ids:
                probe = measurements.loc[probe_id]
                if str(probe.liquid_id) == target_liquid_id:
                    raise ValueError(f"Target liquid leaked into probe set for {sample.sample_id}")
                probe_liquid = liquids.loc[str(probe.liquid_id)]
                liquid_part = self.liquid_scaler.transform_row(probe_liquid, LIQUID_NUMERIC)
                aux = pd.Series({
                    "probe_angle": probe.contact_angle_deg,
                    "probe_std": probe.contact_angle_std_deg,
                    "probe_replicates": probe.replicates_n,
                })
                aux_part = self.probe_aux_scaler.transform_row(
                    aux, ["probe_angle", "probe_std", "probe_replicates"]
                )
                probe_items.append(liquid_part + aux_part)
                raw_probe_angles.append(float(probe.contact_angle_deg))

            probe_array = np.zeros((self.max_probes, 18), dtype=np.float32)
            probe_mask = np.zeros(self.max_probes, dtype=bool)
            if len(probe_items) > self.max_probes:
                raise ValueError(f"Sample {sample.sample_id} exceeds fitted maximum probe count")
            if probe_items:
                probe_array[:len(probe_items)] = np.asarray(probe_items, dtype=np.float32)
                probe_mask[:len(probe_items)] = True

            has_sfe = clean_text(sample.has_independent_sfe).casefold() == "yes"
            if has_sfe:
                sfe_d = float(sample.independent_sfe_dispersion_mj_m2)
                sfe_p = float(sample.independent_sfe_polar_mj_m2)
                if not np.isfinite([sfe_d, sfe_p]).all() or min(sfe_d, sfe_p) < 0.0:
                    raise ValueError(f"Invalid independent SFE supervision for {sample.sample_id}")
            else:
                sfe_d = 0.0
                sfe_p = 0.0
            category_one_hot = self.category_encoder.one_hot(
                category_values, SURFACE_CATEGORICAL + CONDITION_CATEGORICAL
            )
            n_probes = len(probe_items)
            probe_summary = [
                float(n_probes), float(n_probes > 0),
                float(np.mean(raw_probe_angles)) / 180.0 if raw_probe_angles else 0.0,
                float(np.std(raw_probe_angles)) / 180.0 if raw_probe_angles else 0.0,
                float(np.min(raw_probe_angles)) / 180.0 if raw_probe_angles else 0.0,
                float(np.max(raw_probe_angles)) / 180.0 if raw_probe_angles else 0.0,
            ]
            sfe_features = [sfe_d / 100.0, sfe_p / 100.0, float(has_sfe)]
            nnls_features = [nnls_d / 100.0, nnls_p / 100.0, float(nnls_available), nnls_angle / 180.0]
            tabular_values = (
                surface_values + category_one_hot + liquid_values + condition_values
                + probe_summary + sfe_features + nnls_features
            )

            sample_ids.append(str(sample.sample_id))
            record_ids.append(str(sample.record_id))
            source_ids.append(str(sample.source_group_id))
            surface_ids.append(str(sample.surface_group_id))
            modes.append(str(sample.prediction_mode))
            splits.append(str(sample.v4_split))
            target_liquid_ids.append(target_liquid_id)
            surface_numeric.append(surface_values)
            categorical.append(category_values)
            liquid_numeric.append(liquid_values)
            liquid_physical.append(physical_values)
            condition_numeric.append(condition_values)
            nnls_sfe.append([nnls_d, nnls_p])
            nnls_sfe_mask.append(float(nnls_available and not has_sfe))
            nnls_predictions.append(nnls_angle)
            probes_all.append(probe_array)
            probe_masks.append(probe_mask)
            independent_sfe.append([sfe_d, sfe_p])
            independent_sfe_mask.append(float(has_sfe))
            targets.append(float(sample.target_contact_angle_deg))
            tabular.append(tabular_values)
            gate_context.append([
                float(n_probes) / max(self.max_probes, 1), float(n_probes > 0), float(has_sfe),
                float(nnls_available), float(liquid_values[0]), float(surface_values[0]), float(surface_values[3]),
            ])

        target_array = np.asarray(targets, dtype=np.float32)
        return EncodedSamples(
            sample_ids=sample_ids,
            record_ids=record_ids,
            source_group_ids=source_ids,
            surface_group_ids=surface_ids,
            modes=modes,
            splits=splits,
            target_liquid_ids=target_liquid_ids,
            surface_numeric=np.asarray(surface_numeric, dtype=np.float32),
            categorical=np.asarray(categorical, dtype=np.int64),
            target_liquid_numeric=np.asarray(liquid_numeric, dtype=np.float32),
            target_liquid_physical=np.asarray(liquid_physical, dtype=np.float32),
            condition_numeric=np.asarray(condition_numeric, dtype=np.float32),
            nnls_sfe=np.asarray(nnls_sfe, dtype=np.float32),
            nnls_sfe_mask=np.asarray(nnls_sfe_mask, dtype=np.float32),
            nnls_physics_prediction=np.asarray(nnls_predictions, dtype=np.float32),
            probes=np.asarray(probes_all, dtype=np.float32),
            probe_mask=np.asarray(probe_masks, dtype=bool),
            independent_sfe=np.asarray(independent_sfe, dtype=np.float32),
            independent_sfe_mask=np.asarray(independent_sfe_mask, dtype=np.float32),
            target_angle=target_array,
            target_cosine=np.cos(np.deg2rad(target_array)).astype(np.float32),
            tabular=np.asarray(tabular, dtype=np.float32),
            gate_context=np.asarray(gate_context, dtype=np.float32),
        )

    def to_dict(self) -> dict[str, Any]:
        self._require_fit()
        assert self.surface_scaler and self.liquid_scaler and self.condition_scaler
        assert self.probe_aux_scaler and self.category_encoder
        return {
            "surface_scaler": self.surface_scaler.__dict__,
            "liquid_scaler": self.liquid_scaler.__dict__,
            "condition_scaler": self.condition_scaler.__dict__,
            "probe_aux_scaler": self.probe_aux_scaler.__dict__,
            "category_encoder": self.category_encoder.vocabularies,
            "max_probes": self.max_probes,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "FeaturePreprocessor":
        payload = json.loads(path.read_text(encoding="utf-8"))
        instance = cls()
        instance.surface_scaler = NumericScaler(**payload["surface_scaler"])
        instance.liquid_scaler = NumericScaler(**payload["liquid_scaler"])
        instance.condition_scaler = NumericScaler(**payload["condition_scaler"])
        instance.probe_aux_scaler = NumericScaler(**payload["probe_aux_scaler"])
        instance.category_encoder = CategoryEncoder(payload["category_encoder"])
        instance.max_probes = int(payload["max_probes"])
        return instance
