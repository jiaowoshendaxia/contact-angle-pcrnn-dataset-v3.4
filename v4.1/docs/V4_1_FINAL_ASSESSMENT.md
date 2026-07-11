# LS-PSRMoE v4.1 Final Assessment

## Decision

The optimization is conditionally paper-ready, but it does not satisfy every
pre-registered success gate. The recommended primary model for the paper is
the leakage-safe NNLS physics-summary XGBoost residual expert. LS-PSRMoE
fusion should be reported as a secondary experiment, and the neural branch
must not be described as the source of the performance gain.

## Locked Results

The primary clean probe-assisted subset contains 587 samples. Model selection
uses 240 development samples from 8 sources. The fixed confirmation sets
contain 34 internal-test, 107 legacy-external, and 206 open-external samples.

| Evaluation | Model | MAE (deg) | RMSE (deg) | R2 |
|---|---|---:|---:|---:|
| Nested source CV | NNLS physics | 16.335 | 23.793 | 0.572 |
| Nested source CV | Physics-residual XGBoost | **14.493** | **19.720** | **0.706** |
| Nested source CV | Direct XGBoost | 18.157 | 23.508 | 0.582 |
| Fixed open external | NNLS physics | 18.560 | 25.801 | 0.569 |
| Fixed open external | Physics-residual XGBoost | 13.439 | 18.393 | 0.781 |
| Fixed open external | LS-PSRMoE fusion | **13.315** | 18.398 | 0.781 |

The physics-residual tree improves nested-CV MAE over NNLS by 1.842 degrees.
The surface-cluster bootstrap probability of improvement is 0.934, but its
95% interval is [-4.280, 0.486] degrees and therefore crosses zero. This is
directionally strong evidence, not a statistically definitive universal gain.

## Uncertainty

The selected 90% conformal method scales residuals by five-seed ensemble
standard deviation. Nested-CV coverage is 87.9% and mean width is 80.03
degrees. Rejecting the highest-uncertainty 20% reduces nested-CV MAE from
18.69 to 16.27 degrees. OOD distance and the heteroscedastic head are retained
as risk indicators but are not used directly for abstention.

## Paper Positioning

Use these claims:

1. Target-masked probe measurements are converted to nonnegative, auditable
   NNLS-OWRK surface-energy summaries.
2. Predicting the correction to the physical angle is substantially more
   robust than direct XGBoost prediction.
3. Source-disjoint nested validation exposes expert instability that a single
   random split would hide.
4. The fixed external confirmation reaches 13.44-degree MAE for the residual
   tree and 13.32 degrees for the secondary fusion.

Do not claim that the neural expert improves the final model. Its optimized
stacking weight is effectively zero. Do not claim that the bootstrap proves a
significant nested-CV improvement, because the confidence interval crosses
zero. The honest novelty is leakage-safe physical residual learning and
strict cross-source validation, not a deep-network performance breakthrough.

## Locked Artifacts

- `outputs/v4_1_final/predictions_v4_1.csv`
- `outputs/v4_1_final/metrics_v4_1.csv`
- `outputs/v4_1_final/bootstrap_v4_1.csv`
- `outputs/v4_1_final/acceptance_report_v4_1.json`
- `outputs/v4_1_final/model_comparison_v4_1.xlsx`
- `outputs/v4_1_final/selected_config.json`
