# Release Notes: v4.3.0

Date: 2026-07-24

## Major Changes

- Audited 20 legacy sources at row level.
- Re-extracted nine sources, corrected two, verified three, and excluded six.
- Rebuilt the dataset to 794 measurements from 35 sources and 338
  source-conditioned surfaces.
- Recomputed 617 target-masked probe-assisted samples; 490 satisfy the
  primary two-probe and feasible-NNLS criteria.
- Reran the locked LS-PSRMoE v4.1 pipeline without using the fixed
  cross-source confirmation set for model or threshold selection.
- Added source-learning curves, novelty-stratified generalization,
  strict leave-one-liquid-out diagnostics, dual surface/source cluster
  bootstrap, practical error thresholds, and quantitative applicability
  rules.
- Added a complete reviewer-comment coverage matrix and independent
  numerical acceptance report.

## Interpretation Changes

- Residual XGBoost remains the prespecified primary model.
- Residual Random Forest is reported as a stronger sensitivity result but
  was not selected after confirmation-set inspection.
- The neural expert is reported as a negative deep-learning result because
  its consensus fusion weight is 0.080.
- The open external cohort is called a fixed cross-source external
  confirmation set, not a prospective blind test.
- The package does not claim validated prediction for highly rough, porous,
  anisotropic, chemically heterogeneous, or dual-novelty surfaces.

## Compatibility

The `predict()` output interface remains compatible with v4.1. The v4.3
label refers to the audited data and revision analyses, while the deployed
model backend remains LS-PSRMoE v4.1.
