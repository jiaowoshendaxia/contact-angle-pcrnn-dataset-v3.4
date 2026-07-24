# Legacy Source Audit for Revision v4.3

## Final Decision

The submitted 587-sample analysis is legacy and is not reused as numerical evidence in revision v4.3. A row-level review of the 20 legacy source candidates produced the following locked decisions:

- 9 sources retained after row-level re-extraction: `SRC001`, `SRC004`, `SRC012`, `SRC014`, `SRC016`, `SRC018`, `SRC019`, `SRC020`, and `SRC021`.
- 3 sources retained after direct verification: `SRC013`, `SRC017`, and `SDX007`.
- 2 sources retained after metadata or angle-type correction: `SRC002` and `SDX010`.
- 6 source entries excluded because they were secondary duplicates, duplicate extractions, attached to the wrong source, or could not be reconciled with the cited values.

The machine-readable final decisions, locators, verification notes, and exclusion reasons are recorded in `source_provenance_decisions_v4_3.csv`.

## Duplicate and Metadata Corrections

1. `SRC011` and `OPEN_CROSS_MATERIAL_2026` share one DOI. The secondary legacy compilation `SRC011` is excluded.
2. `SRC014` and `OPEN_CELLULOSE_ESTER` share one DOI. The row-level `SRC014` re-extraction is the canonical representation, and the alias is removed before split construction.
3. `SDX008` and `OPEN_PLA_FILMS_2021` share one DOI. The duplicate legacy extraction `SDX008` is excluded.
4. `SRC002`, `SRC004`, and `SDX010` retain the reported advancing, equilibrium, or apparent angle type rather than being silently relabelled as static.
5. Censored values in `SRC020` and `SRC021` remain in the measurement audit but are ineligible as point-valued prediction targets.
6. Two peanut-oil measurements from `SRC012` are excluded from model construction because the locked liquid-property table lacks defensible polar and dispersive surface-tension components.

## Locked v4.3 Collection

After provenance filtering, canonical source grouping, target masking, and nonnegative OWRK reconstruction, the frozen v4.3 data contain:

- 794 traceable measurements;
- 35 source groups;
- 338 source-conditioned surface instances;
- 11 liquids;
- 617 probe-assisted candidates;
- 490 primary eligible samples from 20 sources and 156 surfaces.

Primary eligibility requires at least two unique non-target probe liquids and an `interior_fit` or `boundary_fit` nonnegative least-squares state. The target liquid is removed before every physical-summary calculation.

## Locking and Reproducibility Rules

1. DOI, title, table or figure locator, angle type, material state, liquid identity, and reuse status must be traceable.
2. Retained values must be reproduced without interpolation or conversion of censored observations to invented point targets.
3. Duplicate publications and alternate extractions are represented by one canonical source group.
4. Publisher files with redistribution restrictions are not included in the public release; DOI, locator, extraction script, hash, and audit decision are retained.
5. Source and surface groups cannot cross active train, validation, internal-test, legacy-external, or fixed-confirmation splits.
6. All v4.3 models, statistics, figures, and manuscript numbers are regenerated from the frozen audited tables.
