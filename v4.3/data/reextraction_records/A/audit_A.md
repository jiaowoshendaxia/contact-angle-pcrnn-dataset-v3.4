# v4.3 Legacy Re-extraction Group A Audit

## Scope and Rules

- Sources: `SRC001`, `SRC004`, `SRC012`, `SRC014`, and `SRC016`.
- Existing measurement values in directory `63_...` were not used as numeric input.
- A record was retained only when the value, surface, liquid, and angle definition were visible in an original article table or supplement.
- A reported `+/-` value was not silently relabeled as a standard deviation. `std_deg` is populated only when the article explicitly labels the column `SD`.
- This directory is independent and does not modify the existing v4.0/v4.1 data or models.

## Outcome

- Total audit rows: **128**
- Candidate retained rows: **122**
- Explicitly excluded numeric rows: **6**
- Source-level exclusions: **0** (all five sources had at least one directly traceable table).

| source | audit rows | retained | excluded | primary location |
|---|---:|---:|---:|---|
| SRC001 | 30 | 30 | 0 | Article Tables 2-3 |
| SRC004 | 45 | 45 | 0 | Supplementary Table S2 |
| SRC012 | 8 | 8 | 0 | Main-text Table 1 |
| SRC014 | 27 | 27 | 0 | Article Tables 2 and 7 |
| SRC016 | 18 | 12 | 6 | Article Table 2 |

## Source Decisions

### SRC001

- **Decision:** retain 30 primary table values.
- **Correction:** these are packed-powder capillary-rise/Washburn contact angles, not sessile-drop static angles.
- **Correction:** the table treatments are `as prepared`, `UV illuminated`, and `E-beam irradiated`; the legacy `100 h/200 h` labels are unsupported and were not reused.
- **Traceability:** diamond is in Table 2 and ZnO is in Table 3. Each has five probe liquids and a reported `+/-` value of unspecified type.
- **Method:** n-hexane calibration of the effective capillary radius, followed by the Washburn equation.
- **License note:** the retrieved XML states copyright to the authors; the PMC OA API reports `license=none`.

### SRC004

- **Decision:** retain all 45 advancing-angle values in Supplementary Table S2.
- **Correction:** every imported value is `theta_adv`; none is labeled static.
- **Correction:** the table formula is `Tb7O12`, not the legacy `Tb4O7`.
- **Exclusion:** the water contact-angle-hysteresis column was not imported because hysteresis is not a contact-angle target.
- **Method:** Rame-Hart M500 goniometer. Sintered pellets were mirror polished, cleaned, and vacuum desiccated.
- **License note:** the supplement states Macmillan copyright and all rights reserved.

### SRC012

- **Decision:** retain the eight static organic-liquid angles from main-text Table 1.
- **Surfaces:** the table directly compares a flat fluorinated ITO surface with a porous fluorinated silica-coated ITO surface made using a PEDOT template charge of 47.7 mC/cm2.
- **Method:** OCA 20 sessile-drop system, 5 uL organic droplets, mean of three positions.
- **Conservative exclusion:** water angles and durability-test angles appearing in prose/figure captions were not merged into Table 1 surfaces because they refer to different template charges, controls, or post-treatment states.
- **License note:** the author version and ESI are accessible, but the files do not state a Creative Commons license.

### SRC014

- **Decision:** retain 27 primary measurements from Tables 2 and 7.
- **Traceability:** each row contains the article's average, explicitly labeled SD, and sample count `n`.
- **Method:** DropMaster 700 at 20 C on dried polymer films formed on glass.
- **Duplicate control:** this re-extraction is the sole canonical version for DOI `10.1039/D2RA08165B`; legacy copies under both `SRC014` and `OPEN_CELLULOSE_ESTER` must not coexist in a split.
- **Exclusion:** captive-bubble Table 12 was not mixed with the sessile-drop task.
- **License note:** the PMC OA API identifies this article as CC BY.

### SRC016

- **Decision:** retain 12 primary values for Latex 1, Latex 2, Kaolin, and PCC coated-paper substrates.
- **Correction:** the article title is **Printed Paper-Based Arrays as Substrates for Biofilm Formation**.
- **Angle definition:** Table 2 reports Wenzel roughness-corrected `theta_r`; the underlying apparent static angles were measured after stabilization with three 1-2 uL droplets.
- **Explicit exclusion:** six PS/PDMS values are present in `records_A.csv` with `record_status=exclude` because Table 2 footnote `b` attributes them to Juvonen et al. (2013), not to primary measurements in SRC016. PDMS is additionally footnoted as not corrected for roughness.
- **License note:** the XML states copyright to the authors/licensee Springer; the PMC OA API reports `license=none`.

## Integration Requirements

1. Never concatenate these rows with the legacy rows for the same source.
2. Remove the duplicate `OPEN_CELLULOSE_ESTER` copy before any source-group split.
3. Preserve `contact_angle_type`; do not mix capillary-rise, advancing, roughness-corrected, and sessile-drop values without an explicit task definition.
4. Rebuild `surface_group_id` from source, material, treatment, and state after all re-extraction groups are complete.
5. Re-run split leakage checks before model fitting.

## Files

- `records_A.csv`: row-level values and decisions.
- `source_manifest_A.csv`: DOI, official source URLs, evidence files, and counts.
- `source_file_hashes_A.csv`: hashes for the retrieved XML/PDF/text evidence cache.
- `raw_sources/`: local evidence cache used during extraction.
