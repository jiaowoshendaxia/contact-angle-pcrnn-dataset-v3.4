# v3.4 LOO-SFE leakage sensitivity

Protocol: the PCRNN selected model was frozen from the v3.4 strict run. Model selection remains internal_val only. source_disjoint_external was not used for model selection, tuning, threshold selection, or training.

## Cohorts

- source_disjoint_external total: 148
- LOO-SFE feasible: 114
- high-risk / LOO-SFE not feasible: 34

## Key results on LOO-SFE feasible samples

| Model | MAE | RMSE | R2 |
|---|---:|---:|---:|
| PCRNN | 19.0381 | 24.8473 | 0.7013 |
| Owens-Wendt | 19.0453 | 24.8253 | 0.7018 |
| XGBoost | 17.0573 | 22.5064 | 0.7549 |
| Random Forest | 24.4038 | 29.6862 | 0.5736 |
| Ordinary MLP | 20.1858 | 23.7907 | 0.7262 |

PCRNN matched-114 MAE changed from 7.9098 to 19.0381, delta=11.1283. Compared with the original all-148 source_disjoint_external PCRNN MAE 6.1668, the LOO-feasible corrected cohort differs by 12.8713; this latter comparison is not cohort-matched.

PCRNN forward recomputation check: 107/114 LOO-feasible predictions changed after replacing SFE features. Mean absolute prediction shift is 12.0957 deg; max absolute shift is 55.7162 deg.

## Paired tests on corrected 114 samples

- PCRNN vs Owens-Wendt: delta_MAE=-0.0072, 95% CI [-0.1729, 0.1634], p=0.9080
- PCRNN vs XGBoost: delta_MAE=1.9808, 95% CI [0.0099, 3.9264], p=0.0492
- PCRNN vs Random Forest: delta_MAE=-5.3657, 95% CI [-8.6131, -2.0735], p=0.0008
- PCRNN vs Ordinary MLP: delta_MAE=-1.1477, 95% CI [-3.7963, 1.5644], p=0.3998

## Interpretation

The 114 feasible samples can be used as the stronger external-validation evidence because the target liquid is removed from the apparent SFE fit before prediction. The 34 high-risk samples should be reported separately as leakage-risk / infeasible-LOO diagnostics, not mixed into the strong external claim.
