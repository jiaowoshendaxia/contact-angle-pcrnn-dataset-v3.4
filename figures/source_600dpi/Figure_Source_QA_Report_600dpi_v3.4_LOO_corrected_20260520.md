# Figure Source QA Report 600dpi v3.4 LOO-SFE Corrected 20260520

## Direct Answers

1. **Were 600 dpi TIFF files generated for Figure 1-5 and Figure S1?** Yes. Six TIFF files were generated and verified at 600.0 dpi in RGB mode.

2. **Were PDF/EPS/SVG vector versions generated?** PDF and EPS vector versions were generated for Figure 1-5 and Figure S1. SVG was not generated because the requested vector formats were PDF or EPS, and PDF/EPS are standard CMC submission-compatible source formats. PDF is recommended as the primary vector source. EPS is provided for compatibility, with the caveat that matplotlib's PostScript backend renders unsupported transparency as opaque.

3. **Is the Figure 3 / Figure 4 order correct?** Yes. Figure 3 is `clean external paired statistical tests`; Figure 4 is `LOO-SFE leakage sensitivity`.

4. **Is Figure S1 only supplementary?** Yes. Figure S1 is packaged separately as `FigureS1_high_risk_diagnostic...` and is checked against the manuscript as Supplementary Figure S1 / Supplementary Material only. It is not included among main Figure 1-5.

5. **Are the core values consistent with the manuscript?** Yes. The checked manuscript and figure package preserve:
   - PCRNN clean external MAE = 19.0381.
   - XGBoost clean external MAE = 17.0573.
   - LOO-SFE matched PCRNN before/after MAE = 7.9098 / 19.0381.
   - Forward check = 107/114 predictions changed, mean absolute shift = 12.0957, max absolute shift = 55.7162.

6. **Is there any figure source that cannot satisfy CMC submission requirements?** No blocking issue was found. The 600 dpi TIFF files and PDF vectors satisfy the expected high-resolution/source-file requirement. EPS files were also generated; use PDF if exact alpha/transparency preservation is required.

## Format Verification

| Figure | TIFF 600 dpi | PDF vector | EPS vector | Status |
|---|---|---|---|---|
| Figure 1 | yes | yes | yes | PASS |
| Figure 2 | yes | yes | yes | PASS |
| Figure 3 | yes | yes | yes | PASS |
| Figure 4 | yes | yes | yes | PASS |
| Figure 5 | yes | yes | yes | PASS |
| Figure S1 | yes | yes | yes | PASS |

## Figure Order QA

| Figure | Expected content | File prefix | Status |
|---|---|---|---|
| Figure 1 | framework | `Figure1_framework` | PASS |
| Figure 2 | main MAE comparison | `Figure2_main_MAE_comparison` | PASS |
| Figure 3 | clean external paired statistical tests | `Figure3_clean_external_stat_tests` | PASS |
| Figure 4 | LOO-SFE leakage sensitivity | `Figure4_LOO_SFE_sensitivity` | PASS |
| Figure 5 | forward check | `Figure5_forward_check` | PASS |
| Figure S1 | high-risk diagnostic appendix | `FigureS1_high_risk_diagnostic` | PASS |

## Guardrail QA

- The main external performance figure remains based on `clean_source_disjoint_external_LOO_SFE_114`.
- The raw all-liquid source-disjoint apparent-SFE MAE near 6.17 is not used as the main external validation claim.
- The high-risk 34 cohort remains supplementary diagnostic material only.
- No figure source was altered to change model ranking, plotted values, labels, or conclusions.

## Export Notes

- TIFF files were re-rendered at 600 dpi from the locked matplotlib figure script rather than upsampled from PNG, then saved as white-background RGB TIFF files.
- PDF and EPS files were exported from the same locked figure script.
- EPS transparency caveat: the PostScript backend does not support partial transparency; affected elements are rendered opaque in EPS. PDF should be preferred when allowed.
