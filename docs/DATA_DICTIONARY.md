# DATA DICTIONARY

Public release: `contact_angle_dataset_v3.4_public`

## Files

- `contact_angle_dataset_v3.4_public.xlsx`: multi-sheet public workbook.
- `contact_angle_dataset_v3.4_public.csv`: public raw data table, one row per material-liquid contact-angle record.
- `split_manifest_v3.4.csv`: fixed split and evaluation permissions for all public records.
- `Dataset_Source_Reference_Map_verified_v3.4.csv`: DOI/source metadata map for DOI-bearing source literature.

## Public Workbook Sheets

- `README_PUBLIC`: release notes and scrub policy.
- `Raw_Data_Public`: curated public contact-angle rows. Experimental and numeric values are preserved from the source workbook; internal `curator` and `notes` fields are not included.
- `Processed_Features`: derived modeling features from the source workbook.
- `Split_Manifest`: fixed split labels and use permissions.
- `Field_Dictionary`: source field definitions after public-column filtering.
- `Controlled_Vocab`: controlled values after public-column filtering.
- `Source_Reference_Map`: verified source metadata for DOI-bearing literature sources.
- `Freeze_Summary`: source-workbook freeze metadata.

## Raw_Data_Public Fields

| field | priority | type | unit | allowed values | definition | example |
|---|---|---|---|---|---|---|
| record_id | Core | text |  |  | Unique row identifier for one solid-liquid contact-angle record. | CA0001 |
| dataset_version | Recommended | text |  |  | Dataset release version. | v3.4 |
| collection_status | Core | category |  | planned\|extracted\|verified\|excluded | Curation status for the row. | verified |
| source_type | Core | category |  | literature\|experiment\|database\|simulation | Source category: literature, database, experiment, simulation, or literature review. | literature |
| reference_title | Core | text |  |  | Public title or source label for the data source. | Surface wettability of polymer films |
| reference_authors | Recommended | text |  |  | Public author string or source attribution. | Smith et al. |
| reference_year | Core | integer | year |  | Publication or source year. | 2021 |
| reference_doi | Recommended | text |  |  | DOI for the source article when available. | 10.xxxx/xxxxx |
| reference_url | Recommended | url |  |  | Public URL for the source article when available. | https://doi.org/... |
| data_extraction_note | Recommended | text |  |  | Brief public note describing where or how the value was extracted. | table value |
| solid_name | Core | text |  |  | Name of the solid material or surface. | PMMA |
| solid_family | Core | category |  | polymer\|oxide\|glass\|metal\|ceramic\|coating\|composite\|carbon_material\|biomaterial\|other\|unknown | Broad solid material family. | polymer |
| solid_substrate | Optional | text |  |  | Underlying substrate, if reported. | glass |
| surface_treatment | Core | category |  | untreated\|plasma\|UV_ozone\|chemical_grafting\|coating\|thermal\|laser\|etching\|mechanical_polishing\|cleaning\|other\|unknown | High-level surface treatment category. | untreated |
| surface_treatment_detail | Recommended | text |  |  | Detailed treatment description, if reported. | oxygen plasma 60 s |
| coating_or_layer | Optional | text |  |  | Coating, layer, or modifier on the surface, if applicable. | fluorinated silane |
| surface_state | Recommended | category |  | smooth\|rough\|porous\|patterned\|heterogeneous\|unknown | Reported or curated surface state such as smooth, rough, porous, textured, or coated. | smooth |
| roughness_Ra_nm | Recommended | number | nm |  | Arithmetic average roughness Ra in nanometers, if available. | 12.5 |
| roughness_Rq_nm | Optional | number | nm |  | Root-mean-square roughness Rq in nanometers, if available. | 16.3 |
| roughness_r_factor | Optional | number | dimensionless |  | Roughness factor r, if available. | 1.05 |
| solid_total_surface_energy_mJ_m2 | Core | number | mJ/m2 |  | Solid total surface energy feature in mJ/m2. | 42.0 |
| solid_dispersion_mJ_m2 | Core | number | mJ/m2 |  | Solid dispersive surface-energy component in mJ/m2. | 21.0 |
| solid_polar_mJ_m2 | Core | number | mJ/m2 |  | Solid polar surface-energy component in mJ/m2. | 21.0 |
| solid_LW_mJ_m2 | Optional | number | mJ/m2 |  | Solid Lifshitz-van der Waals component in mJ/m2, if available. | 21.0 |
| solid_acid_plus_mJ_m2 | Optional | number | mJ/m2 |  | Solid electron-acceptor acid component in mJ/m2, if available. | 0.5 |
| solid_base_minus_mJ_m2 | Optional | number | mJ/m2 |  | Solid electron-donor base component in mJ/m2, if available. | 25.0 |
| solid_surface_energy_source | Recommended | text |  |  | Public description of the solid surface-energy data source or fitting method. | same paper |
| solid_surface_energy_source_type | Core | category |  | independent_measurement\|inferred_from_contact_angles\|literature_reported\|assumed_or_estimated\|unclear | Provenance class for solid surface-energy descriptors. | inferred_from_contact_angles |
| liquid_name | Core | text |  |  | Probe liquid name. | water |
| liquid_family | Core | category |  | polar\|nonpolar\|protic\|aprotic\|oil\|aqueous_solution\|organic_solvent\|ionic_liquid\|other\|unknown | Broad liquid category. | polar |
| liquid_total_surface_tension_mN_m | Core | number | mN/m |  | Liquid total surface tension in mN/m. | 72.8 |
| liquid_dispersion_mN_m | Core | number | mN/m |  | Liquid dispersive surface-tension component in mN/m. | 21.8 |
| liquid_polar_mN_m | Core | number | mN/m |  | Liquid polar surface-tension component in mN/m. | 51.0 |
| liquid_LW_mN_m | Optional | number | mN/m |  | Liquid Lifshitz-van der Waals component in mN/m, if available. | 21.8 |
| liquid_acid_plus_mN_m | Optional | number | mN/m |  | Liquid electron-acceptor acid component in mN/m, if available. | 25.5 |
| liquid_base_minus_mN_m | Optional | number | mN/m |  | Liquid electron-donor base component in mN/m, if available. | 25.5 |
| liquid_viscosity_mPa_s | Optional | number | mPa*s |  | Liquid viscosity in mPa s, if available. | 0.89 |
| liquid_dipole_moment_D | Optional | number | D |  | Liquid dipole moment in Debye, if available. | 1.85 |
| liquid_dielectric_constant | Optional | number | dimensionless |  | Liquid dielectric constant, if available. | 78.4 |
| liquid_property_source | Recommended | text |  |  | Source note for liquid-property descriptors. | standard value |
| temperature_K | Core | number | K |  | Measurement temperature in Kelvin, if reported or curated. | 298.15 |
| humidity_percent | Optional | number | % |  | Relative humidity in percent, if reported. | 50 |
| pressure_atm | Optional | number | atm |  | Measurement pressure in atm, if reported or assumed. | 1 |
| contact_angle_deg | Core | number | degree |  | Experimental contact angle in degrees. | 70.0 |
| contact_angle_type | Core | category |  | static\|advancing\|receding\|equilibrium\|apparent\|young\|unknown | Type of contact angle, such as static, advancing, receding, apparent, or equilibrium. | static |
| measurement_method | Recommended | category |  | sessile_drop\|captive_bubble\|tilting_plate\|Wilhelmy_plate\|image_analysis\|unknown | Measurement method, such as sessile drop or captive bubble. | sessile_drop |
| droplet_volume_uL | Optional | number | uL |  | Droplet volume in microliters, if reported. | 5 |
| replicates_n | Recommended | integer | count |  | Number of replicates or measurements, if reported. | 5 |
| contact_angle_std_deg | Recommended | number | degree |  | Reported contact-angle standard deviation in degrees, if available. | 2.1 |
| contact_angle_min_deg | Optional | number | degree |  | Reported minimum contact angle in degrees, if available. | 67.5 |
| contact_angle_max_deg | Optional | number | degree |  | Reported maximum contact angle in degrees, if available. | 72.6 |
| is_equilibrium_angle | Recommended | category |  | yes\|no\|unknown | Whether the value is reported or curated as an equilibrium angle. | unknown |
| sample_preparation_notes | Recommended | text |  |  | Brief public sample-preparation or condition note. | cleaned with ethanol |
| quality_grade | Core | category |  | A_high\|B_medium\|C_low\|exclude | Curated evidence-quality grade. | A_high |
| include_in_training | Core | category |  | yes\|no\|review | Whether the record is allowed in the training pool. | yes |
| split_group | Recommended | category |  | internal_pool\|balanced_holdout\|hard_external\|source_disjoint_external\|unassigned | Fixed split group for modeling and evaluation. | unassigned |
| duplicate_group_id | Optional | text |  |  | Duplicate or near-duplicate group identifier, if assigned. | DUP001 |
| conflict_flag | Recommended | category |  | none\|possible_duplicate\|inconsistent_units\|unclear_condition\|outlier\|other | Conflict or ambiguity flag. | none |
| curation_date | Recommended | date | yyyy-mm-dd |  | Curation or freeze date. | 2026-05-12 |
| analysis_split | Core | category |  | internal_train\|internal_val\|internal_test\|balanced_holdout\|hard_external\|source_disjoint_external\|excluded_review | Analysis split used in the v3.4 modeling protocol. | internal_train |
| source_group_id | Core | text |  |  | Source group identifier used for source-disjoint split control. | SRC001 |
| duplicate_policy | Core | category |  | unique_or_condition_distinct_keep\|retain_condition_distinct_near_duplicate_same_split_marked\|retain_condition_distinct_near_duplicate_cross_split_marked_not_independent\|retain_exact_value_duplicate_for_review | Policy applied to duplicate or condition-distinct records. | unique_or_condition_distinct_keep |

## Processed_Features Fields

| field | description |
|---|---|
| record_id | Unique row identifier for one solid-liquid contact-angle record. |
| solid_name | Name of the solid material or surface. |
| liquid_name | Probe liquid name. |
| include_in_training | Whether the record is allowed in the training pool. |
| contact_angle_deg | Experimental contact angle in degrees. |
| cos_theta_exp | Cosine of the experimental contact angle. |
| solid_polar_ratio | Solid polar component divided by solid total surface energy. |
| liquid_polar_ratio | Liquid polar component divided by liquid total surface tension. |
| gamma_ratio_s_l | Ratio between solid and liquid total surface-energy/tension descriptors. |
| gamma_difference_abs | Absolute difference between solid total surface energy and liquid total surface tension. |
| polarity_difference_abs | Absolute difference between solid and liquid polarity ratios. |
| owens_wendt_cos_theta | Owens-Wendt physical baseline prediction in cosine space. |
| owens_wendt_theta_deg | Owens-Wendt physical baseline prediction in degrees. |
| owens_wendt_residual_deg | Experimental contact angle minus Owens-Wendt predicted angle. |
| core_missing_count | Number of missing core modeling fields for the row. |
| row_quality_flag | Derived row-quality flag used for modeling audits. |

## Split Manifest Fields

| field | description |
|---|---|
| record_id | Unique row identifier for one solid-liquid contact-angle record. |
| split_group | Fixed split group for modeling and evaluation. |
| analysis_split | Analysis split used in the v3.4 modeling protocol. |
| model_use | Intended modeling or evaluation use for the record. |
| include_in_training | Whether the record is allowed in the training pool. |
| allow_train | Whether the row may be used for model training. |
| allow_validation | Whether the row may be used for validation. |
| allow_tuning | Whether the row may be used for tuning. |
| allow_hyperparameter_search | Whether the row may be used for hyperparameter search. |
| allow_internal_test_eval | Whether the row may be used for internal-test evaluation. |
| allow_balanced_holdout_eval | Whether the row may be used for balanced-holdout evaluation. |
| allow_external_eval | Whether the row may be used for external evaluation. |
| freeze_status | Freeze status for the split assignment. |
| leakage_policy | Leakage-control policy for the row. |
| source_group_id | Source group identifier used for source-disjoint split control. |
| duplicate_policy | Policy applied to duplicate or condition-distinct records. |
| solid_surface_energy_source_type | Provenance class for solid surface-energy descriptors. |
| quality_grade | Curated evidence-quality grade. |
| solid_family | Broad solid material family. |
| solid_name | Name of the solid material or surface. |
| liquid_name | Probe liquid name. |
| contact_angle_deg | Experimental contact angle in degrees. |
| reference_doi | DOI for the source article when available. |
| source_type | Source category: literature, database, experiment, simulation, or literature review. |
| record_sha256 | Stable row-level hash for split manifest integrity. |

## Public Scrub Notes

- Removed internal `curator` and `notes` columns from the public raw table.
- Removed local file URL values from `reference_url`; DOI-bearing public URLs are retained.
- Sanitized project-internal reference titles/authors and extraction notes for legacy rows without row-level DOI.
- No contact-angle values, surface-energy values, liquid-property values, split labels, or model-feature numeric values were modified.
