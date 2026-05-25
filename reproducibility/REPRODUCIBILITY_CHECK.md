# Reproducibility Check: v3.4 LOO-SFE Corrected Public Release

This note documents the read-only verification path for the public release package. It does not retrain PCRNN or select new models. It checks the locked public data files, fixed split manifest, LOO-SFE corrected result tables, paired statistical tests, and forward-check rows shipped in this release.

## Files Checked

- `data/contact_angle_dataset_v3.4_public.csv`
- `data/contact_angle_dataset_v3.4_public.xlsx`
- `data/split_manifest_v3.4.csv`
- `results/source_disjoint_external_loo_sfe_feasibility_v3.4_20260519.csv`
- `results/pcrnn_strict_v3.4_loo_sfe_metrics_20260519.csv`
- `results/pcrnn_strict_v3.4_loo_sfe_predictions_20260519.csv`
- `results/pcrnn_strict_v3.4_loo_sfe_comparison_20260519.csv`
- `results/pcrnn_strict_v3.4_loo_sfe_error_cases_20260519.csv`
- `results/pcrnn_strict_v3.4_loo_sfe_forward_check_20260519.csv`
- `results/pcrnn_strict_v3.4_sfe_leakage_sensitivity_note_20260519.md`
- `results/main_result_table_v3.4_LOO_corrected_20260519.csv`
- `results/plot_ready_metrics_v3.4_LOO_corrected_20260519.csv`
- `results/combined_pcrnn_vs_baselines_stat_tests_v3.4_LOO_corrected_20260519.csv`

## Locked Counts

The public v3.4 dataset and split manifest both contain 616 rows.

| analysis_split | n |
|---|---:|
| internal_train | 156 |
| internal_val | 33 |
| internal_test | 33 |
| balanced_holdout | 50 |
| hard_external | 75 |
| source_disjoint_external | 148 |
| excluded_review | 121 |

The source-disjoint external cohort contains 148 samples. The LOO-SFE audit separates these into 114 feasible clean external samples and 34 high-risk or LOO-SFE-infeasible diagnostic samples.

## Locked Clean External Metrics

Primary clean external cohort: `clean_source_disjoint_external_LOO_SFE_114`.

| model | n | MAE | RMSE | R2 |
|---|---:|---:|---:|---:|
| PCRNN | 114 | 19.0381 | 24.8473 | 0.7013 |
| Owens-Wendt | 114 | 19.0453 | 24.8253 | 0.7018 |
| XGBoost | 114 | 17.0573 | 22.5064 | 0.7549 |
| Random Forest | 114 | 24.4038 | 29.6862 | 0.5736 |
| Ordinary MLP | 114 | 20.1858 | 23.7907 | 0.7262 |

## Forward Check

`results/pcrnn_strict_v3.4_loo_sfe_forward_check_20260519.csv` confirms that 107 of 114 PCRNN predictions changed after LOO-SFE feature replacement. The mean absolute prediction shift is 12.0957 degrees and the maximum absolute prediction shift is 55.7162 degrees.

## How To Run The Check

From the extracted `release_public` folder:

```powershell
powershell -ExecutionPolicy Bypass -File .\reproducibility\check_release_reproducibility.ps1
```

Expected final line:

```text
PUBLIC RELEASE REPRODUCIBILITY CHECK PASSED
```

The script reads only files inside the release folder.
