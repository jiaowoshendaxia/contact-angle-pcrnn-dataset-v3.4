# LS-PSRMoE Revision v4.3

This directory contains the source-audited data, locked models, reviewer
analyses, and public reproduction code for the Applied Sciences revision:

**Leakage-Safe Probe-Assisted Contact Angle Prediction Using Nonnegative
Surface-Energy Summaries and Physics-Residual Learning**

## Version Semantics

- **v4.0**: normalized five-table data schema.
- **LS-PSRMoE v4.1**: prediction model and public API.
- **v4.2**: post-lock robustness analyses.
- **v4.3**: source-audited revision data and reproducibility package.

The model architecture is still LS-PSRMoE v4.1. The v4.3 label identifies
the audited dataset and the reviewer-driven analyses, not a post-confirmation
model reselection.

## Public Contents

- `data/base_v4_0/`: frozen pre-revision processed tables.
- `data/processed_v4_3/`: source-audited v4.3 tables.
- `data/source_provenance_decisions_v4_3.csv`: all 20 legacy-source decisions.
- `data/reextraction_records/`: row-level replacement records and audit notes.
- `data/raw/`: eight openly accessible PubMed Central JATS XML fixtures used
  by the public extraction tests.
- `src/`: data, model, prediction, uncertainty, recuration, and revision code.
- `results/model_v4_3/`: locked predictions, metrics, models, and manifests.
- `results/reviewer_analyses_v4_3/`: learning curves, novelty strata,
  leave-one-liquid-out diagnostics, bootstrap results, and practical accuracy.
- `figures/`: publication figures generated from the locked outputs.
- `docs/`: acceptance report, reviewer coverage matrix, and literature audit.

Publisher PDFs, downloaded HTML snapshots, and page-level evidence images are
intentionally excluded. The only source documents retained are eight
open-access PubMed Central JATS XML test fixtures. Original DOI and table
locations are recorded in the public audit files.

## Environment

Python 3.11 is recommended.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-lock.txt
```

On Linux or macOS, use `.venv/bin/python`.

## Reproduction

Run commands from this `v4.3` directory:

```bash
python -m src.pipeline recurate-data --config configs/v4_3_recuration.yaml
python -m src.pipeline run --config configs/v4_3_main.yaml
python -m src.pipeline revision --config configs/v4_3_revision.yaml
python -m pytest -q
```

The first command rebuilds the audited tables from the frozen v4.0 tables,
the public source-decision file, and the released replacement-record CSVs.
The second command reruns nested source-group model selection. The third
recomputes all reviewer-requested analyses.

## Locked Primary Evidence

| Evaluation | NNLS physics MAE | Residual XGBoost MAE | Residual RF MAE | OOF fusion MAE |
|---|---:|---:|---:|---:|
| Nested source-group CV | 18.3 | 15.5 | 13.7 | 14.6 |
| Fixed cross-source confirmation | 17.8 | 13.2 | 12.1 | 12.7 |

Residual XGBoost remains the prespecified primary model. Residual Random
Forest is reported as a stronger sensitivity result and was not selected
after inspecting the fixed confirmation set.

The neural expert obtained 17.2 degrees MAE in nested source-group CV and a
consensus fusion weight of 0.080. The revision reports this as a negative
deep-learning result rather than claiming a neural performance advantage.

## Applicability

The primary prediction mode requires:

- at least two distinct non-target probe liquids;
- complete masking of the target liquid from the probe set;
- an NNLS status of `interior_fit` or `boundary_fit`.

Rough, porous, anisotropic, or chemically heterogeneous surfaces remain
outside the validated range because quantitative roughness and related
conditions are too sparsely reported. High OOD distance or excessive
conformal interval width triggers a high-risk or abstention result.

## Data and Model Caveats

- The development evidence contains 223 samples from eight publication
  sources.
- The fixed cross-source confirmation set influenced early architecture
  thinking and is therefore not described as a prospective blind test.
- Strict leave-one-liquid-out analyses are diagnostic.
- Dual novelty is represented by only four internal-test samples and showed
  severe residual-model failure.
- The system supports candidate screening; it is not a replacement for
  high-precision contact-angle measurement.

## License and Citation

Repository code is covered by the root repository license. Source-level data
licenses and provenance fields are provided in `sources_v4.csv`. Cite the
versioned Zenodo record listed in the root `CITATION.cff`.
