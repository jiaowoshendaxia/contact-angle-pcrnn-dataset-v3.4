# LS-PSRMoE v4.1 Reproducibility Package

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21305764.svg)](https://doi.org/10.5281/zenodo.21305764)

This directory contains the public research code, normalized data tables,
frozen configuration, tests, predictions, and statistical outputs for
leakage-safe probe-assisted contact-angle prediction.

The primary v4.1 model converts target-masked probe-liquid measurements into
nonnegative NNLS-OWRK surface-energy summaries and learns an XGBoost correction
to the physical contact-angle prediction. The multi-expert fusion is retained
as a secondary analysis. The optimized neural-expert weight is effectively
zero, so the neural branch is not claimed as the source of the performance
gain.

## Scientific Scope

- Primary task: probe-assisted contact-angle prediction.
- Target leakage control: the target liquid is removed before every physical
  summary and model feature is constructed.
- Primary eligibility: at least two unique non-target probe liquids and an
  `interior_fit` or `boundary_fit` NNLS status.
- Model selection: four-fold outer and three-fold inner source-group cross-
  validation on the development sources only.
- Confirmation cohorts: `internal_test`, `legacy_external`, and
  `prospective_open_external`. The last cohort is described as a fixed
  cross-source external confirmation set, not as a fully blind prospective
  test.

## Locked Result Summary

The primary clean probe-assisted cohort contains 587 samples. Development
model selection uses 240 samples from 8 sources.

| Evaluation | Model | MAE (deg) | RMSE (deg) | R2 |
|---|---|---:|---:|---:|
| Nested source CV | NNLS physics | 16.335 | 23.793 | 0.572 |
| Nested source CV | Physics-residual XGBoost | **14.493** | **19.720** | **0.706** |
| Nested source CV | Direct XGBoost | 18.157 | 23.508 | 0.582 |
| Fixed open external | NNLS physics | 18.560 | 25.801 | 0.569 |
| Fixed open external | Physics-residual XGBoost | 13.439 | 18.393 | 0.781 |
| Fixed open external | Secondary fusion | **13.315** | 18.398 | 0.781 |

The nested-CV MAE improvement over NNLS is 1.842 degrees. The paired
surface-cluster bootstrap interval is [-4.280, 0.486] degrees and crosses zero;
the result is therefore directional rather than statistically definitive.
The 90% conformal interval has 87.9% nested-CV coverage and an 80.03-degree
mean width. Removing the highest-risk 20% reduces nested-CV MAE from 18.69 to
16.27 degrees.

## Directory Layout

```text
v4.1/
  configs/                 Frozen v4.1 main and smoke configurations
  data/processed/          Six normalized tables, NNLS audit, and split hash
  docs/                    Reproduction and interpretation notes
  results/                 Locked predictions, metrics, bootstrap and ablations
  src/lspgmoe/             Training, physics, evaluation and prediction code
  tests/                   Public core leakage and model tests
  pyproject.toml
  requirements-lock.txt
```

Local virtual environments, caches, raw publisher files, runtime logs, and
serialized model binaries are intentionally excluded. The supplied code and
frozen inputs regenerate the trained artifacts.

## Environment

Python 3.11 is recommended. From this `v4.1` directory:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.venv\Scripts\python.exe -m pip install -e .
```

The locked production run used Python 3.11.15, PyTorch 2.10.0 with CUDA 12.8,
and an NVIDIA GeForce RTX 4070 Laptop GPU. CPU execution is supported but will
be slower.

## Verification And Reproduction

Run the public test suite:

```powershell
.venv\Scripts\python.exe -m pytest -q
```

Run a short pipeline check:

```powershell
.venv\Scripts\python.exe -m src.pipeline run --config configs/v4_1_smoke.yaml
```

Run the frozen v4.1 analysis:

```powershell
.venv\Scripts\python.exe -m src.pipeline run --config configs/v4_1_main.yaml
```

The full run performs nested source-group validation, five-seed final fitting,
external confirmation, cluster bootstrap, calibration, rejection analysis,
and ablations. It writes new artifacts under `outputs/v4_1_final` and does not
modify the locked files in this public `results` directory.

## Data Tables

- `sources_v4.csv`: source identifiers, bibliographic provenance, license
  status, extraction status, and frozen split role.
- `surfaces_v4.csv`: stable source-plus-condition surface identities and
  surface descriptors.
- `liquids_v4.csv`: canonical probe-liquid names and liquid properties.
- `measurements_v4.csv`: one normalized surface-liquid measurement per row.
- `splits_v4.csv`: source- and surface-disjoint frozen assignments.
- `samples_v4.csv`: model-ready zero-shot and target-masked probe-assisted
  samples.
- `loo_nnls_audit_v4.csv`: target-masked NNLS fit status and primary-eligibility
  audit used by the frozen v4.1 configuration.

The `license` field in `sources_v4.csv` records source-level status. Some
legacy literature rows remain marked `source_license_to_verify`. This release
does not include article PDFs, publisher figures, or full-text source files.
The repository license covers the authors' compilation, annotations, code,
and derived outputs only; it does not relicense third-party publications or
override source-specific reuse terms. Reusers must cite the corresponding DOI
and check the original source license.

## Claim Boundaries

- Do not describe the neural expert as a performance contribution in v4.1.
- Do not claim statistically significant nested-CV superiority over NNLS; the
  paired cluster-bootstrap confidence interval crosses zero.
- Do not tune or select models using any external confirmation cohort.
- Do not mix primary eligible samples with single-probe, singular, or
  insufficient-probe sensitivity samples.
- Do not interpret angle-derived surface energy as independently measured SFE.

## Citation And License

The version-specific archive DOI is
[10.5281/zenodo.21305764](https://doi.org/10.5281/zenodo.21305764), and the
all-versions concept DOI is
[10.5281/zenodo.20382890](https://doi.org/10.5281/zenodo.20382890). Use the
repository-level `CITATION.cff`. Code is released under the MIT
License. Author-created data compilation, documentation, and derived results
are released under CC BY 4.0, subject to the third-party scope clarification
above and in the repository-level `LICENSE` file.
