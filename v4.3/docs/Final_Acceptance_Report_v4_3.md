# Applied Sciences Revision v4.3 Final Acceptance Report

Date: 2026-07-24

## Scientific and Data Acceptance

- The source-audited v4.3 data contain 794 measurements from 35 sources,
  338 source-conditioned surfaces, and 11 liquids.
- The probe-assisted construction produced 617 candidate samples and 490
  primary-eligible samples from 20 sources and 156 surfaces.
- The development analysis contains 223 samples from eight publication
  sources.
- All 20 legacy-source decisions have a documented retain, correct,
  re-extract, or exclude outcome.
- Target-liquid leakage checks found zero violations.
- Source and surface overlap checks passed.
- The fixed cross-source external confirmation set was not read by model,
  hyperparameter, uncertainty-threshold, or OOD-threshold selection.
- All 140 independent numerical and data-integrity checks passed.

## Locked Results

- Nested source-group cross-validation MAE:
  - NNLS-OWRK physics: 18.3 degrees.
  - Neural residual expert: 17.2 degrees.
  - Residual XGBoost: 15.5 degrees.
  - Residual Random Forest: 13.7 degrees.
  - Direct XGBoost: 17.5 degrees.
  - OOF fusion: 14.6 degrees.
- Fixed cross-source external confirmation MAE:
  - Residual XGBoost: 13.2 degrees.
  - Residual Random Forest: 12.1 degrees.
  - OOF fusion: 12.7 degrees.
- Residual XGBoost remains the prespecified primary model. Residual Random
  Forest is reported as a stronger sensitivity result and was not selected
  post hoc using the confirmation set.
- The 90% conformal interval achieved 88.3% coverage with a mean width of
  78.2 degrees. Risk-based retention was 79.8%, with retained-sample MAE of
  11.9 degrees.
- The neural expert consensus weight was 0.080; the manuscript treats this
  as a negative deep-learning result rather than claiming a neural advantage.

## Document Acceptance

- Clean manuscript: 11 pages.
- Marked manuscript: 11 pages.
- Supplementary Materials: 15 pages.
- Response to Reviewers: 7 pages.
- Every page was rasterized and visually inspected.
- No blank pages, clipped figures, content outside margins, broken tables,
  or missing repeated table headers were found.
- The official Applied Sciences page size, margins, line numbering,
  first-page behavior, and template styles were retained.
- All six main figures and two supplementary figures are within the text
  width and exceed 300 effective dpi.
- Four model equations are editable Word math objects.
- The abstract contains 160 words and two core performance values.

## Reviewer Coverage

- The coverage matrix contains 28 completed rows.
- The response letter contains 27 numbered response items because Reviewer 2
  Comment 10 addresses two linked coverage rows.
- Every response uses Reviewer Comment, Response, Changes Made, and Location
  in Revised Manuscript fields.

## Automated Tests

- `pytest`: 62 passed.
- Document content audit: passed.
- Strict MDPI template and media audit: passed with zero issues.
- Numerical-output validation: passed with 140 checks.

## Remaining Scientific Limitations

- The development evidence comes from only eight publication sources.
- Quantitative roughness is too sparse and source-confounded to support a
  numerical roughness cutoff.
- Strict leave-one-liquid-out results are diagnostic and unstable for some
  liquids.
- Dual novelty (new material family and new target liquid) is supported by
  only four internal-test samples and showed severe residual-model failure.
- The method should be presented as a leakage-safe screening tool, not a
  replacement for high-precision contact-angle measurement.
