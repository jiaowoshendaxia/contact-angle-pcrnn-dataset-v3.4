# DATASET QA REPORT

Public release: `contact_angle_dataset_v3.4_public`

## Source And Outputs

- Source workbook: `outputs/contact_angle_dataset_v3.4_20260518.xlsx`
- Public workbook: `contact_angle_dataset_v3.4_public.xlsx`
- Public raw CSV: `contact_angle_dataset_v3.4_public.csv`
- Public split manifest: `split_manifest_v3.4.csv`
- Public source reference map: `Dataset_Source_Reference_Map_verified_v3.4.csv`

## Row And Version Checks

- Raw public records: 616
- Processed feature records: 616
- Split manifest records: 616
- Unique record IDs: 616
- Duplicate record IDs: 0
- Dataset version values in public raw table: v3.4
- Textual old-version markers remaining in public raw table: 0
- Source reference map rows: 32
- Manifest records missing from raw table: 0
- Raw records missing from manifest: 0

## Numeric Integrity Checks

- Numeric public raw fields compared against source workbook: 24
- Numeric cell mismatches introduced by public export preparation: 0
- Contact angle range: 0 to 172.5 degrees
- Contact angle out-of-range rows: 0

## Core Field Missingness

| field | missing rows |
|---|---:|
| record_id | 0 |
| dataset_version | 0 |
| source_type | 0 |
| reference_title | 0 |
| reference_year | 0 |
| solid_name | 0 |
| solid_family | 0 |
| liquid_name | 0 |
| liquid_total_surface_tension_mN_m | 0 |
| liquid_dispersion_mN_m | 0 |
| liquid_polar_mN_m | 0 |
| contact_angle_deg | 0 |
| contact_angle_type | 0 |
| quality_grade | 0 |
| include_in_training | 0 |
| split_group | 0 |
| analysis_split | 0 |
| source_group_id | 0 |
| solid_surface_energy_source_type | 0 |

## Split And Source Summaries

### analysis_split

- internal_train: 156
- source_disjoint_external: 148
- excluded_review: 121
- hard_external: 75
- balanced_holdout: 50
- internal_test: 33
- internal_val: 33

### split_group

- internal_pool: 222
- source_disjoint_external: 148
- unassigned: 121
- hard_external: 75
- balanced_holdout: 50

### include_in_training

- no: 384
- yes: 156
- review: 76

### quality_grade

- A_high: 453
- B_medium: 87
- C_low: 76

### source_type

- literature: 526
- database: 76
- literature_review: 14

### solid_surface_energy_source_type

- inferred_from_contact_angles: 429
- assumed_or_estimated: 120
- literature_reported: 67

## Public Scrub And Privacy/IP Checks

- Dropped internal raw columns: curator, notes
- Local file URLs removed from public reference_url: 76
- Project-internal reference titles sanitized: 76
- Project-internal reference authors sanitized: 76
- Project-internal extraction notes sanitized: 102
- Text encoding/prior-version metadata strings sanitized: 54
- Local path / PDF / image / email / phone pattern hits in public outputs: 0
- Long excerpt candidate cells in public outputs: 0
- Source workbook embedded media files: 0
- Public workbook embedded media files: 0
- Maximum public text-cell length: 204 characters

### Remaining Risk Hits

- None detected.

## Release Decision

PASS: Public package is ready for release-side review. No numeric values were changed; non-release notes/local paths/media/long excerpts were removed or excluded.
