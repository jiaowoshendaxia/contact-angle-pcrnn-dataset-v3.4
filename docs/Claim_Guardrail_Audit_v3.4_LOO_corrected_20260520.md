# Claim Guardrail Audit v3.4 LOO-corrected (2026-05-20)

## Guardrail Verdict

PASS after minor cross-reference edits.

## Clean External Claim

Allowed wording:

- PCRNN is statistically tied with Owens-Wendt on the clean LOO-SFE external cohort.
- PCRNN is significantly better than Random Forest.
- PCRNN is not significantly different from Ordinary MLP.
- PCRNN is slightly weaker than XGBoost in MAE.

Disallowed wording:

- PCRNN is best on clean external validation.
- PCRNN universally outperforms all baselines.
- The raw all-liquid source-disjoint result is the main external validation.

## Checked Lines and Numeric Anchors

- PCRNN clean external MAE = 19.0381.
- Owens-Wendt clean external MAE = 19.0453.
- XGBoost clean external MAE = 17.0573.
- Random Forest clean external MAE = 24.4038.
- Ordinary MLP clean external MAE = 20.1858.
- PCRNN vs XGBoost delta MAE = 1.9808, p = 0.0492, meaning PCRNN is weaker on MAE.
- All-liquid 148 PCRNN MAE = 6.1668 is sensitivity only.
- High-risk 34 samples are diagnostic appendix only.

## Automated Overclaim Scan

Possible overclaim hits: 0.

No prohibited clean-external superiority claim was found.
