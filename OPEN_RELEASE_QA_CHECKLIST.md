# Open Release QA Checklist

Release package: `release_public/`

QA date: 2026-05-25

## Required Top-Level Structure

| Item | Status | Notes |
|---|---|---|
| `README.md` | PASS | Public package overview is present. |
| `LICENSE` | PASS | License file is present. |
| `CITATION.cff` | PASS | Citation metadata is present. |
| `data/` | PASS | Public dataset, split manifest, and source-reference maps are present. |
| `results/` | PASS | Locked public result tables are present. |
| `code/` | PASS | Public reproduction scripts and figure requirements are present. |
| `figures/` | PASS | PNG figures and 600 dpi/vector figure source files are present. |
| `docs/` | PASS | Public documentation and release-side QA notes are present. |
| `reproducibility/` | PASS | Public reproducibility instructions and check script are present. |

## Exclusion Checks

| Check | Status | Method / Scope |
|---|---|---|
| No manuscript or article PDF included | PASS | PDF files are limited to figure vector source files named `*_vector.pdf`. |
| No Word/PowerPoint manuscript files included | PASS | No `.doc`, `.docx`, `.ppt`, or `.pptx` files found. |
| No build caches or temporary artifacts included | PASS | No `__pycache__`, `node_modules`, `.pyc`, `.bak`, or `.tmp` files found. |
| No local absolute Windows paths included | PASS | Text and XLSX XML scans found no local drive-path or file-URI references. |
| No private contact information included | PASS | Targeted scan found no email-address or mainland-China mobile-phone patterns. |
| No non-release review notes included | PASS | Public package contains release-facing documentation only; work-in-progress and review filenames were removed from copied figure QA notes. |
| No old pre-v3.4 files included | PASS | Filename and content scans found no pre-v3.4 release artifacts. |
| Figure S1 not mixed into main figures | PASS | Figure S1 remains under figure files as a supplementary diagnostic figure only. |

## Scientific Guardrail Checks

| Guardrail | Status |
|---|---|
| The primary external figure uses `clean_source_disjoint_external_LOO_SFE_114`, not the raw apparent-SFE source-disjoint MAE near 6.17. | PASS |
| High-risk 34 samples are retained only as Supplementary Figure S1 diagnostic material. | PASS |
| Figure 3 is clean external paired statistical tests. | PASS |
| Figure 4 is LOO-SFE leakage sensitivity. | PASS |
| Locked values are preserved: PCRNN clean external MAE 19.0381; XGBoost clean external MAE 17.0573; PCRNN matched before/after 7.9098/19.0381; forward check 107/114, mean absolute shift 12.0957, max absolute shift 55.7162. | PASS |

## Reproducibility Checks

| Check | Status | Command |
|---|---|---|
| Public release consistency check | PASS | `powershell -ExecutionPolicy Bypass -File reproducibility/check_release_reproducibility.ps1` |
| Archive creation | PASS | `release_public.zip` was generated from the final `release_public/` directory. |
| Archive extraction | PASS | `release_public.zip` was extracted in a clean check directory and the public reproducibility check passed from the extracted copy. |

## Release Decision

PASS: The public release directory and archive are ready for open release. No paper PDF, private information, local absolute paths, pre-v3.4 artifacts, or non-release manuscript review materials were found in the final package.
