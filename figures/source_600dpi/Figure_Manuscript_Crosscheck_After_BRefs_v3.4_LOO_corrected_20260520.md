# Figure Manuscript Crosscheck After BRefs v3.4 LOO-SFE Corrected 20260520

## Public Figure Order Checked

The public figure files and public release result tables were checked for figure order and locked numeric values.

## Figure Order in Current Manuscript

| Manuscript figure | Manuscript role | Final source file prefix | Status |
|---|---|---|---|
| Figure 1 | Leakage-aware modeling and validation workflow | `Figure1_framework` | PASS |
| Figure 2 | Main MAE comparison under corrected evidence hierarchy | `Figure2_main_MAE_comparison` | PASS |
| Figure 3 | Paired statistical tests on the primary clean external cohort | `Figure3_clean_external_stat_tests` | PASS |
| Figure 4 | LOO-SFE leakage sensitivity on matched source-disjoint external samples | `Figure4_LOO_SFE_sensitivity` | PASS |
| Figure 5 | PCRNN forward check after LOO-SFE feature replacement | `Figure5_forward_check` | PASS |
| Supplementary Figure S1 | High-risk source-disjoint external diagnostic only | `FigureS1_high_risk_diagnostic` | PASS |

Figure 3 and Figure 4 are not reversed in the final package. The package follows the current numeric-locked manuscript order, not the earlier working figure order.

## Core Numeric Crosscheck

| Locked value | Manuscript status | Figure/source package status | Status |
|---|---|---|---|
| PCRNN clean external MAE = 19.0381 | present | Figure 2 / Figure 3 / Figure 4 context preserved | PASS |
| XGBoost clean external MAE = 17.0573 | present | Figure 2 / Figure 3 context preserved | PASS |
| LOO-SFE matched PCRNN before/after MAE = 7.9098 / 19.0381 | present | Figure 4 preserved | PASS |
| Forward check = 107/114 predictions changed | present | Figure 5 preserved | PASS |
| Mean absolute prediction shift = 12.0957 | present | Figure 5 preserved | PASS |
| Max absolute prediction shift = 55.7162 | present | Figure 5 preserved | PASS |

## Supplementary Figure S1 Check

The manuscript states that the high-risk 34 cohort is excluded from the primary clean external claim and that Supplementary Figure S1 reports it as diagnostic appendix material only. The final figure source package preserves this separation:

- Main figures: Figure 1, Figure 2, Figure 3, Figure 4, Figure 5.
- Supplementary figure: Figure S1 only.

## Claim Guardrail Crosscheck

- The primary clean external cohort remains `clean_source_disjoint_external_LOO_SFE_114`.
- The raw all-liquid apparent-SFE source-disjoint MAE near 6.17 is retained only as leakage-risk sensitivity and is not the primary external validation figure.
- High-risk 34 diagnostics are not mixed into the main clean external result.
- PCRNN is not presented as universally dominant; the manuscript and captions state that it is tied with Owens-Wendt, weaker than XGBoost in MAE, better than Random Forest, and not significantly different from Ordinary MLP on the clean external cohort.
