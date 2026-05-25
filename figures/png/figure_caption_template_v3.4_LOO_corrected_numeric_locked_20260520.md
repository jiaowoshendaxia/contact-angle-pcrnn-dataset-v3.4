# Figure Caption Template v3.4 LOO-SFE Corrected Numeric-Locked 20260520

**Figure 1. Leakage-aware modeling and validation workflow.** The study workflow begins with dataset construction and source-disjoint split design, followed by a surface-energy descriptor leakage audit. LOO-SFE correction removes the target liquid from apparent surface-energy fitting for feasible source-disjoint external samples before PCRNN and baseline prediction. Strict evaluation separates internal/supporting holdouts, the primary clean external cohort `clean_source_disjoint_external_LOO_SFE_114`, leakage-risk sensitivity, and high-risk diagnostic samples.

**Figure 2. Main MAE comparison under the corrected evidence hierarchy.** MAE is shown for PCRNN, Owens-Wendt, XGBoost, Random Forest, and Ordinary MLP on `internal_test`, `balanced_holdout`, `hard_external`, and the primary clean external cohort `clean_source_disjoint_external_LOO_SFE_114`. The raw all-liquid source-disjoint apparent-SFE result is not used in this main performance figure. On the clean external cohort, PCRNN is approximately tied with Owens-Wendt, weaker than XGBoost in MAE, better than Random Forest, and close to Ordinary MLP.

**Figure 3. Paired statistical tests on the primary clean external cohort.** Delta MAE is defined as `MAE(PCRNN) - MAE(comparator)`, so negative values favor PCRNN and positive values favor the comparator. Points show paired delta MAE and horizontal intervals show 95% bootstrap confidence intervals. On `clean_source_disjoint_external_LOO_SFE_114`, PCRNN is statistically tied with Owens-Wendt, slightly weaker than XGBoost in MAE, significantly better than Random Forest, and not significantly different from Ordinary MLP.

**Figure 4. LOO-SFE leakage sensitivity on matched source-disjoint external samples.** MAE is compared before and after LOO-SFE correction on the same 114 LOO-SFE-feasible samples. The before-correction condition uses all-liquid apparent SFE features, whereas the after-correction condition removes the target liquid from apparent SFE fitting. PCRNN MAE increases from 7.9098 to 19.0381 degrees, demonstrating that apparent SFE leakage can materially alter external-validation conclusions.

**Figure 5. PCRNN forward check after LOO-SFE feature replacement.** The distribution of PCRNN prediction shifts is shown for the 114 LOO-SFE-feasible source-disjoint external samples. After replacing all-liquid apparent SFE features with LOO-SFE features, 107 of 114 predictions changed, with a mean absolute shift of 12.0957 degrees and a maximum absolute shift of 55.7162 degrees.

**Supplementary Figure S1. High-risk source-disjoint external diagnostic only.** MAE is shown for the 34 high-risk or LOO-SFE-infeasible source-disjoint samples under original-only apparent SFE evaluation. This cohort is reported only as a diagnostic appendix and is not used as the primary external validation claim.

Caption guardrails:

- Do not use the original all-148 source-disjoint apparent-SFE MAE as the main external result.
- Do not mix the high-risk 34 cohort into the main clean external result.
- Do not state that PCRNN is superior on the clean external cohort.
- Do not treat apparent or inferred SFE as independent measured SFE.
