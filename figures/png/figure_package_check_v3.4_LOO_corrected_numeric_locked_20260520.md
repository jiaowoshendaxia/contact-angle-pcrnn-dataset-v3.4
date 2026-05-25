# Figure Package Check v3.4 LOO-SFE Corrected Numeric-Locked 20260520

## Public Package Inputs Checked

- Public numeric-locked PNG figures in `figures/png/`.
- Public 600 dpi/vector figure sources in `figures/source_600dpi/`.
- Public result tables in `results/`.

## Numbering Alignment

| Manuscript label | Expected content | Package file | Status |
|---|---|---|---|
| Figure 1 | framework | `Figure1_framework_v3.4_LOO_corrected_numeric_locked.png` | PASS |
| Figure 2 | main MAE comparison | `Figure2_main_MAE_comparison_v3.4_LOO_corrected_numeric_locked.png` | PASS |
| Figure 3 | clean external paired statistical tests | `Figure3_clean_external_stat_tests_v3.4_LOO_corrected_numeric_locked.png` | PASS |
| Figure 4 | LOO-SFE leakage sensitivity | `Figure4_LOO_SFE_sensitivity_v3.4_LOO_corrected_numeric_locked.png` | PASS |
| Figure 5 | forward check | `Figure5_forward_check_v3.4_LOO_corrected_numeric_locked.png` | PASS |
| Figure S1 | high-risk diagnostic appendix | `FigureS1_high_risk_diagnostic_v3.4_LOO_corrected_numeric_locked.png` | PASS |

## Guardrail Check

- The main external validation figure remains based on `clean_source_disjoint_external_LOO_SFE_114`.
- The raw all-liquid apparent-SFE source-disjoint MAE near 6.17 is not used as the main external validation figure.
- The high-risk 34 cohort remains separated as Supplementary Figure S1 diagnostic material only.
- Figure 3 reports paired delta MAE, confidence intervals, and p values for the clean external cohort.
- Figure 4 reports leakage sensitivity on the matched 114 samples before and after LOO-SFE correction.

## Packaging Check

- Figure numbering follows the public numeric-locked order.
- PNG files were copied into this public release package; model outputs and input data were not changed.
- All six PNG files opened successfully during verification.
