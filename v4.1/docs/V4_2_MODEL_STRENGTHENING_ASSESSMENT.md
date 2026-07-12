# V4.2 Model Strengthening Assessment

## Decision

Retain the unweighted physics-residual XGBoost expert as the primary model.
Do not replace it with the source-weighted variant. Source weighting improved
development-only leave-one-source-out (LOSO) performance but degraded all three
fixed confirmation cohorts. No additional weighting variants may be tuned on
the confirmation labels.

## Development-Only LOSO Results

The LOSO analysis covers all 240 eligible development samples from eight
sources. Every source is held out once, all preprocessing is fitted without the
held-out source, and no confirmation cohort is read during fitting or model
comparison.

| Model | MAE (deg) | RMSE (deg) | R2 |
|---|---:|---:|---:|
| NNLS physics | 16.335 | 23.793 | 0.572 |
| Global mean residual | 16.532 | 23.271 | 0.590 |
| Target-liquid mean residual | 16.901 | 22.771 | 0.608 |
| Ridge residual, inner source-CV alpha | 19.417 | 28.560 | 0.383 |
| Direct XGBoost | 16.819 | 22.691 | 0.611 |
| Physics-residual XGBoost | 15.358 | 21.272 | 0.658 |
| Source-weighted physics-residual XGBoost | **14.794** | **20.548** | **0.681** |

The unweighted residual tree improves MAE over NNLS by 0.976 degrees and over
direct XGBoost by 1.460 degrees. Its source-cluster bootstrap interval versus
NNLS is [-8.127, 6.092] degrees and crosses zero. It wins against NNLS on three
of eight held-out sources, showing that the benefit is source-dependent.

Source weighting lowers MAE by 0.564 degrees relative to the unweighted
residual tree and improves seven of eight source-specific MAEs. The paired
source-cluster bootstrap interval for weighted minus unweighted MAE is
[-0.803, -0.184] degrees, with probability of improvement 0.995. This is a
stable development-only result, not sufficient evidence for model promotion.

## One-Time Fixed Confirmation

| Cohort | NNLS | Direct XGBoost | Residual XGBoost | Weighted Residual XGBoost |
|---|---:|---:|---:|---:|
| Internal test, n=34 | 29.686 | 35.899 | **30.086** | 41.512 |
| Legacy external, n=107 | 15.102 | 13.790 | **12.538** | 14.541 |
| Open external, n=206 | 18.560 | 14.990 | **13.439** | 14.499 |

The source-weighted candidate degrades every fixed cohort. On the open external
set it is 1.061 degrees worse than the unweighted residual tree. On the small
internal test it is 11.426 degrees worse, with a source-cluster bootstrap
interval [0.726, 26.307] degrees. This is clear evidence that source weighting
overfits the development-source composition.

## Paper Use

1. Keep the existing four-by-three nested source CV as the main model-selection
   result.
2. Add LOSO as a stricter source-transfer sensitivity analysis.
3. Add the global, target-liquid and Ridge residual baselines to show that the
   XGBoost gain is not a constant offset or a simple linear correction.
4. Report source weighting as an instructive failed robustness intervention:
   development LOSO improves, but external transfer degrades.
5. Place the source-specific LOSO figure and full confirmation bootstrap in the
   Supplementary Materials.
6. Use the three deterministic low-, medium- and high-uncertainty examples as a
   worked application table, not as model-selection evidence.

## Claim Boundaries

- Do not claim source-universal superiority over NNLS; the source-cluster
  interval remains wide and crosses zero.
- Do not claim that source weighting is the improved final model.
- Do claim that residual learning is more robust than direct XGBoost in both
  LOSO and fixed open-external evaluation.
- Do claim that simple global, liquid-specific and linear residual corrections
  do not explain the XGBoost result.
- Do emphasize that strict source-level validation exposed a seemingly stable
  development improvement that failed external confirmation.

## Reproduction

```powershell
.venv\Scripts\python.exe -m src.pipeline robustness --config configs\v4_2_robustness.yaml
```

Locked outputs are under `outputs/v4_2_model_strengthening`.
