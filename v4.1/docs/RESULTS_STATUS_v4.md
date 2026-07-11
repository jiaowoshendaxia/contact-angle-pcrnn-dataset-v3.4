# LS-PGMoE v4 status

## Data and audit

- Current processed tables contain 922 measurements, 42 sources, 407 stable source-conditioned surfaces, and 11 canonical liquids.
- There are 922 zero-shot samples and 714 probe-assisted samples. The fixed split contains 243 train, 52 validation, 52 internal test, 148 legacy external, 306 prospective open external, and 121 excluded-review rows.
- 210 surfaces have at least three liquids in the full table; 68 prospective surfaces have at least three liquids. All processed main-task records are static or legacy apparent angles; no advancing-angle source was mixed into the static task.
- The target-masked NNLS audit yields 447 interior fits, 144 boundary fits, 116 insufficient-probe cases, and 7 singular fits. The 144 legacy negative-coefficient rows are retained only for the legacy algorithm comparison.
- Eight reviewed open-access article tables have passed extraction, license, locator, duplicate, and schema checks and are frozen as evaluation-only prospective data. The facemask source is CC BY-NC and must retain its non-commercial restriction. Mendeley and Droplet Lab remain in the raw ledger but are not processed because their tabular extraction is incomplete.

## Full five-seed result

The current `outputs/experiments` run is `lspgmoe-v4.0` with seeds 7, 19, 42, 67, and 99, five-fold source-group OOF training, and 10,000 bootstrap resamples. The prospective probe-assisted subset contains 220 eligible target-masked samples.

| split/mode | model | MAE (deg) | RMSE (deg) | R2 |
|---|---|---:|---:|---:|
| prospective zero-shot | XGBoost | 30.76 | 40.93 | 0.065 |
| prospective zero-shot | fusion | 30.93 | 39.86 | 0.113 |
| prospective probe-assisted | random forest baseline | 20.70 | 28.04 | not in baseline export |
| prospective probe-assisted | NNLS physics expert | 18.39 | 25.29 | 0.565 |
| prospective probe-assisted | neural expert | 19.62 | 26.85 | 0.509 |
| prospective probe-assisted | global OOF convex fusion | 17.25 | 23.09 | 0.637 |

On the legacy probe-assisted split, fusion is 14.04 degrees MAE versus 13.63 degrees for the neural expert. The global convex gate is a clear improvement over the previous high-capacity context gate, whose archived prospective probe-assisted MAE was 24.54 degrees. The new gate learns one OOF-only simplex weight, approximately physics 0.035, neural 0.663, tree 0.303, and applies it consistently across samples.

The prospective surface-cluster bootstrap gives fusion minus physics MAE difference -1.15 degrees, 95% CI [-2.52, 0.35], probability of improvement 0.939; fusion minus neural is -2.37 degrees, CI [-3.64, -1.09]; fusion minus tree is -5.14 degrees, CI [-8.74, -1.73]. Source-cluster intervals cross zero, so the manuscript should describe the result as robust improvement against neural/tree experts at the surface level, with source-level sensitivity, not as universal superiority.

## Uncertainty and ablation status

The 90% conformal interval coverage on prospective probe-assisted data is 98.2% with mean width 117.6 degrees. After correcting the rejection rule, 88.2% of prospective probe-assisted predictions are retained and their MAE is 17.04 degrees; unknown categories increase risk to medium but no longer automatically discard every new material.

The full `v4_main` ablation is now complete. On prospective probe-assisted data, full fusion is 17.25 degrees MAE; latent-SFE neural-only is 19.62, fixed average is 17.18, no probe encoder is 16.76, no physics expert is 17.33, no XGBoost expert is 19.39, and the DeepSets-without-physics variant is 27.32 degrees. The physical decoder is therefore essential, while the probe encoder does not yet show a stable positive contribution and should be discussed as a limitation or simplified in a follow-up model.

The interpretation pipeline is complete. Permutation importance fallback ranks target-liquid total/dispersion surface tension, probe-angle extrema, and NNLS polar SFE among the strongest tree-expert features. Error stratification is exported by prediction mode, liquid, material family, angle interval, source, and probe count. The semantic duplicate audit reports zero exact cross-source candidates across all 922 processed measurements.

## Remaining gates before writing the paper

1. Add SHAP/permutation interpretation and liquid/material/error stratification.
2. Recalibrate interval width and risk curves; coverage is acceptable, but intervals remain wide.
3. Complete the second-person audit of all eight new source tables, especially the CC BY-NC facemask source, before public release.
4. Add exact cross-source duplicate auditing before the public release; the current merge protects within-source keys, while source-level semantic deduplication still needs a report.
5. Only after these gates pass, replace manuscript tables and figures with locked outputs.
