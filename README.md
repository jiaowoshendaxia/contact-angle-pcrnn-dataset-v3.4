# Contact Angle Prediction Dataset and Reproducibility Packages

## Current Release: Source-Audited Revision v4.3

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21545090.svg)](https://doi.org/10.5281/zenodo.21545090)

The current research release is available under [`v4.3/`](v4.3/README.md). It
contains the source-audited v4.3 data, row-level provenance decisions,
target-masked NNLS-OWRK summaries, nested source-group validation, locked
LS-PSRMoE v4.1 model artifacts, reviewer-requested robustness analyses,
publication figures, tests, and public reproduction configurations.

The v4.3 audit rebuilt the evidence base instead of preserving the submitted
numbers. It releases 794 measurements from 35 sources, with 490 primary
probe-assisted samples from 20 sources. Residual XGBoost remains the
prespecified primary model; the stronger residual-Random-Forest sensitivity
result is reported without post-confirmation model switching.

Version-specific v4.3.0 DOI:
[10.5281/zenodo.21545090](https://doi.org/10.5281/zenodo.21545090).

The all-versions concept DOI is
[10.5281/zenodo.20382890](https://doi.org/10.5281/zenodo.20382890).

## Archived Release: LS-PSRMoE v4.1 and Robustness v4.2

The original v4.1 research release remains under
[`v4.1/`](v4.1/README.md). It contains the initial leakage-safe
probe-assisted pipeline, normalized v4 data tables, uncertainty analysis,
tests, and locked pre-audit results.

An additive post-lock v4.2 robustness extension is also available inside
[`v4.1/`](v4.1/README.md). It adds leave-one-source-out transfer, simple
residual baselines, a source-weighting stress test, fixed confirmation, and
worked application examples. The archived v4.1 model remains primary because
the source-weighted candidate failed all three fixed confirmation cohorts.

Version-specific archived release:
[Zenodo 10.5281/zenodo.21305764](https://doi.org/10.5281/zenodo.21305764).

The earlier PCRNN v3.4 package is preserved below as a legacy release so that
published hashes, results, and tags remain auditable.

## Legacy Release: PCRNN v3.4

This public package accompanies the study on leakage-aware, physics-constrained residual learning for contact-angle prediction with apparent surface-energy descriptors. It provides the curated v3.4 contact-angle dataset, fixed split manifest, locked result tables, statistical comparison files, and figure-generation inputs needed to audit and reproduce the main reported tables and figures.

The package is designed for transparent reuse. It does not include article PDFs, copyrighted source figures, private project notes, local file paths, or internal review comments.

## Dataset Summary

The v3.4 dataset contains 616 literature-derived solid-liquid contact-angle records. Each row describes one material/liquid measurement pair with a reported contact angle, solid descriptors, liquid descriptors, surface-energy-related descriptors, source metadata, and fixed evaluation labels.

Important points:

- The target variable is `contact_angle_deg`, reported in degrees.
- Surface-energy descriptors may be independently reported, literature-reported, inferred from contact angles, assumed/estimated, or unclear.
- The primary clean external validation in the associated study uses 114 source-disjoint samples after leave-one-liquid-out surface-energy correction (`clean_source_disjoint_external_LOO_SFE_114`).
- The original all-liquid source-disjoint cohort has 148 samples and is used only as leakage-risk sensitivity.
- The 34 LOO-SFE-infeasible or high-risk source-disjoint samples are diagnostic appendix material only.

## Package Structure

Expected release layout:

```text
release_public/
  README.md
  CITATION.cff
  LICENSE
  data/
    contact_angle_dataset_v3.4_public.xlsx
    contact_angle_dataset_v3.4_public.csv
    split_manifest_v3.4.csv
    Dataset_Source_Reference_Map_verified_v3.4.csv
  results/
    main_result_table_v3.4_LOO_corrected_20260519.csv
    combined_pcrnn_vs_baselines_stat_tests_v3.4_LOO_corrected_20260519.csv
    plot_ready_metrics_v3.4_LOO_corrected_20260519.csv
    pcrnn_strict_v3.4_loo_sfe_metrics_20260519.csv
    pcrnn_strict_v3.4_loo_sfe_predictions_20260519.csv
    pcrnn_strict_v3.4_loo_sfe_comparison_20260519.csv
    pcrnn_strict_v3.4_loo_sfe_forward_check_20260519.csv
  code/
    run_all_baselines.mjs
    run_loo_sfe_baselines_v34.mjs
    make_paper_figures_v3.4_LOO_corrected_20260519.py
    requirements_figures_v3.4_LOO_corrected_20260520.txt
  figures/
    png/
    source_600dpi/
  docs/
    DATA_DICTIONARY.md
    DATASET_QA_REPORT.md
    DATA_AVAILABILITY_STATEMENT.md
    AIITA_CMC_DIFFERENCE_NOTE.md
  reproducibility/
    REPRODUCIBILITY_CHECK.md
    check_release_reproducibility.ps1
```

The locked v3.4 public data files and locked v3.4 LOO-corrected result files should be used together.

## Core Files

### Data

- `data/contact_angle_dataset_v3.4_public.xlsx`: public workbook with raw data, processed features, split manifest, field dictionary, controlled vocabulary, source map, and freeze summary.
- `data/contact_angle_dataset_v3.4_public.csv`: public raw data table, one row per contact-angle record.
- `data/split_manifest_v3.4.csv`: fixed split labels and allowed model-use flags.
- `data/Dataset_Source_Reference_Map_verified_v3.4.csv`: DOI/source metadata map for DOI-bearing literature sources.

### Result Tables

- `results/main_result_table_v3.4_LOO_corrected_20260519.csv`: main metric table for PCRNN and baselines, including the corrected clean external cohort.
- `results/combined_pcrnn_vs_baselines_stat_tests_v3.4_LOO_corrected_20260519.csv`: paired bootstrap comparisons between PCRNN and representative baselines.
- `results/plot_ready_metrics_v3.4_LOO_corrected_20260519.csv`: long-format metric table for plotting.
- `results/pcrnn_strict_v3.4_loo_sfe_metrics_20260519.csv`: LOO-SFE corrected, all-liquid sensitivity, and high-risk diagnostic metric rows.
- `results/pcrnn_strict_v3.4_loo_sfe_predictions_20260519.csv`: locked PCRNN/physical/baseline prediction rows used for corrected external analysis.
- `results/pcrnn_strict_v3.4_loo_sfe_comparison_20260519.csv`: clean external paired comparisons on the 114 LOO-SFE feasible samples.
- `results/pcrnn_strict_v3.4_loo_sfe_forward_check_20260519.csv`: forward-check table showing how PCRNN predictions changed after LOO-SFE feature replacement.

## Split Definitions

Use `analysis_split` from `split_manifest_v3.4.csv`; do not create a new random split when reproducing the reported results.

| analysis_split | n | Intended use |
|---|---:|---|
| `internal_train` | 156 | Training only. Fit scalers, encoders, imputers, feature transforms, and model parameters only here. |
| `internal_val` | 33 | Validation, tuning, and early stopping only. |
| `internal_test` | 33 | Internal held-out evaluation only. |
| `balanced_holdout` | 50 | Supporting holdout evaluation only. |
| `hard_external` | 75 | Supporting difficult external evaluation only. |
| `source_disjoint_external` | 148 | Source-disjoint external cohort before LOO-SFE separation. The clean main external conclusion uses only the 114 LOO-SFE feasible samples. |
| `excluded_review` | 121 | Excluded from model training, tuning, and main evaluation. |

The source-disjoint external cohort is further separated in the corrected analysis:

- `clean_source_disjoint_external_LOO_SFE_114`: primary clean external validation cohort.
- `original_source_disjoint_external_all_liquid_SFE_148`: leakage-risk sensitivity only.
- `high_risk_source_disjoint_external_34_original_only`: diagnostic appendix only.

## Reading the Data

Python example:

```python
import pandas as pd

raw = pd.read_csv("data/contact_angle_dataset_v3.4_public.csv")
split = pd.read_csv("data/split_manifest_v3.4.csv")

data = raw.merge(
    split[["record_id", "analysis_split", "allow_train", "allow_validation", "allow_external_eval"]],
    on="record_id",
    how="left",
)

train = data[data["analysis_split"] == "internal_train"]
clean_external_candidates = data[data["analysis_split"] == "source_disjoint_external"]
```

R example:

```r
raw <- read.csv("data/contact_angle_dataset_v3.4_public.csv", stringsAsFactors = FALSE)
split <- read.csv("data/split_manifest_v3.4.csv", stringsAsFactors = FALSE)
data <- merge(raw, split[, c("record_id", "analysis_split", "allow_train")], by = "record_id")
```

## Reproducing the Reported Results

The recommended reproducibility path is to audit the locked result files rather than to create new splits or retrain models from scratch.

Minimal checks:

1. Confirm the public dataset has 616 records.
2. Confirm `split_manifest_v3.4.csv` has 616 records and fixed split counts.
3. Confirm the clean LOO-SFE external cohort has 114 samples.
4. Confirm the high-risk diagnostic cohort has 34 samples.
5. Confirm the locked clean external metrics and paired tests from the result CSV files.

Key clean external results:

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| PCRNN | 19.0381 | 24.8473 | 0.7013 |
| Owens-Wendt | 19.0453 | 24.8253 | 0.7018 |
| XGBoost | 17.0573 | 22.5064 | 0.7549 |
| Random Forest | 24.4038 | 29.6862 | 0.5736 |
| Ordinary MLP | 20.1858 | 23.7907 | 0.7262 |

Paired clean external comparisons use `delta_MAE = MAE(PCRNN) - MAE(comparator)`. Negative values favor PCRNN; positive values favor the comparator.

| Comparison | delta_MAE | p_value | Interpretation |
|---|---:|---:|---|
| PCRNN vs Owens-Wendt | -0.0072 | 0.9080 | Statistically tied. |
| PCRNN vs XGBoost | 1.9808 | 0.0492 | XGBoost has lower MAE. |
| PCRNN vs Random Forest | -5.3657 | 0.0008 | PCRNN has lower MAE. |
| PCRNN vs Ordinary MLP | -1.1477 | 0.3998 | No significant difference. |

The forward check reports that 107 of 114 PCRNN predictions changed after LOO-SFE feature replacement, with mean absolute prediction shift 12.0957 degrees and maximum absolute shift 55.7162 degrees.

### Optional Scripted Reproduction

The recommended public reproducibility path is the read-only check script:

```powershell
powershell -ExecutionPolicy Bypass -File .\reproducibility\check_release_reproducibility.ps1
```

The baseline LOO-SFE script under `code/` is included for audit purposes. It requires the Node.js dependencies used by the project. PCRNN locked predictions and forward-check outputs are provided in `results/`; PCRNN retraining or re-selection is not part of this public release package.

Example baseline sensitivity command:

```powershell
node .\code\run_loo_sfe_baselines_v34.mjs `
  --input .\data\contact_angle_dataset_v3.4_public.xlsx `
  --manifest .\data\split_manifest_v3.4.csv `
  --feasibility .\results\source_disjoint_external_loo_sfe_feasibility_v3.4_20260519.csv `
  --sheet Raw_Data_Public `
  --out .\results\_loo_sfe_baseline_predictions_v3.4_20260519.csv
```

Do not use these scripts to select a different model or to tune on external cohorts. The locked paper-facing conclusions use the fixed split manifest and the locked v3.4 LOO-corrected result files.

## Claim Boundaries

Please keep the following interpretation when citing or reusing this package:

- PCRNN is not the overall best model on the clean LOO-SFE external cohort.
- PCRNN is statistically tied with Owens-Wendt, slightly weaker than XGBoost in MAE, significantly better than Random Forest, and not significantly different from Ordinary MLP on the clean external cohort.
- The original all-liquid apparent-SFE result is leakage-risk sensitivity only.
- The 34 high-risk samples are diagnostic appendix material only.
- Apparent or inferred SFE descriptors should not be treated as equivalent to independently measured SFE descriptors.

## Citation

Please cite this release using `CITATION.cff`. Until the journal article DOI and repository DOI are finalized, use the provisional citation:

> Xia Y, Liu W, Shen M, Xing R, Gao X. Contact Angle Prediction Dataset and Physics-Constrained Residual Learning Results, Version 3.4. 2026.

When reusing source-level data, also cite the relevant original literature sources listed in `data/Dataset_Source_Reference_Map_verified_v3.4.csv`.

## License

This release uses a dual-license scheme:

- Data, documentation, result tables, and figure-generation input files: Creative Commons Attribution 4.0 International (CC BY 4.0).
- Code and scripts: MIT License.

See `LICENSE` for details.
