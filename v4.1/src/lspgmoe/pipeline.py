"""Command-line entry point for v4 data, audit, training, and evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from .data import write_migration_artifacts


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("Configuration root must be a mapping.")
    return config


def resolve(workspace: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (workspace / path).resolve()


def build_data(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    workspace = config_path.parent.parent
    config = load_config(config_path)
    project = config["project"]
    data_dir = resolve(workspace, project["output_data_dir"])
    existing_sources = data_dir / "sources_v4.csv"
    if existing_sources.exists():
        try:
            import pandas as pd
            source_frame = pd.read_csv(existing_sources, encoding="utf-8-sig")
            if "new_external_source_flag" in source_frame and (source_frame["new_external_source_flag"] == "yes").any():
                raise RuntimeError(
                    "Refusing to overwrite processed data containing staged external sources. "
                    "Use the existing processed tables or archive them explicitly before rebuilding from legacy data."
                )
        except RuntimeError:
            raise
    return write_migration_artifacts(
        legacy_csv=resolve(workspace, project["legacy_dataset"]),
        data_dir=resolve(workspace, project["output_data_dir"]),
        audit_dir=resolve(workspace, project["output_dir"]) / "audit",
        seed=int(project["seed"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LS-PGMoE v4 research pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ["build-data", "run", "smoke", "baselines", "ablations", "interpret", "audit-duplicates", "validate-import", "extract-jats", "extract-cross-jats", "extract-polymer-jats", "extract-textile-jats", "extract-pla-jats", "extract-carbon-jats", "extract-facemask-jats", "extract-chitosan-jats", "merge-staged"]:
        item = subparsers.add_parser(command)
        item.add_argument("--config", required=True, type=Path)
        if command == "validate-import":
            item.add_argument("--input", required=True, type=Path)
            item.add_argument("--source-id", required=True)
        if command == "extract-jats":
            item.add_argument("--input", required=True, type=Path)
            item.add_argument("--source-id", required=True)
            item.add_argument("--output", required=True, type=Path)
        if command == "extract-cross-jats":
            item.add_argument("--input", required=True, type=Path)
            item.add_argument("--source-id", required=True)
            item.add_argument("--output", required=True, type=Path)
        if command == "extract-polymer-jats":
            item.add_argument("--input", required=True, type=Path)
            item.add_argument("--source-id", required=True)
            item.add_argument("--output", required=True, type=Path)
        if command == "extract-textile-jats":
            item.add_argument("--input", required=True, type=Path)
            item.add_argument("--source-id", required=True)
            item.add_argument("--output", required=True, type=Path)
        if command == "extract-pla-jats":
            item.add_argument("--input", required=True, type=Path)
            item.add_argument("--source-id", required=True)
            item.add_argument("--output", required=True, type=Path)
        if command in {"extract-carbon-jats", "extract-facemask-jats", "extract-chitosan-jats"}:
            item.add_argument("--input", required=True, type=Path)
            item.add_argument("--source-id", required=True)
            item.add_argument("--output", required=True, type=Path)
        if command == "merge-staged":
            item.add_argument("--input", required=True, type=Path)
            item.add_argument("--source-id", required=True)
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args()
    if args.command == "build-data":
        result = build_data(args.config)
    elif args.command == "run":
        config = load_config(args.config.resolve())
        if str(config.get("project", {}).get("model_version", "")).endswith("v4.1"):
            from .v41 import run_v41

            result = run_v41(args.config)
        else:
            from .training import run_smoke_experiment

            result = run_smoke_experiment(args.config)
    elif args.command == "smoke":
        from .training import run_smoke_experiment

        result = run_smoke_experiment(args.config)
    elif args.command == "baselines":
        from .baselines import run_baselines

        result = run_baselines(args.config)
    elif args.command == "ablations":
        from .ablations import run_ablations

        result = run_ablations(args.config)
    elif args.command == "interpret":
        from .interpretation import run_interpretation

        result = run_interpretation(args.config)
    elif args.command == "audit-duplicates":
        from .ingest import audit_semantic_duplicates

        config_path = args.config.resolve()
        config = load_config(config_path)
        root = config_path.parent.parent
        result = audit_semantic_duplicates(
            data_dir=root / config["project"]["output_data_dir"],
            output_dir=root / config["project"]["output_dir"] / "audit",
        )
    elif args.command == "validate-import":
        from .ingest import validate_and_stage

        config_path = args.config.resolve()
        config = load_config(config_path)
        root = config_path.parent.parent
        result = validate_and_stage(
            input_path=(Path(args.input) if Path(args.input).is_absolute() else root / args.input),
            source_id=args.source_id,
            registry_path=root / "data" / "OPEN_DATA_REGISTRY_v4.csv",
            output_dir=root / "data" / "staging",
        )
    elif args.command == "extract-jats":
        from .ingest import extract_cellulose_ester_jats

        config_path = args.config.resolve()
        root = config_path.parent.parent
        input_path = Path(args.input) if Path(args.input).is_absolute() else root / args.input
        output_path = Path(args.output) if Path(args.output).is_absolute() else root / args.output
        result = extract_cellulose_ester_jats(input_path, output_path, args.source_id)
    elif args.command == "extract-cross-jats":
        from .ingest import extract_cross_material_jats

        config_path = args.config.resolve()
        root = config_path.parent.parent
        input_path = Path(args.input) if Path(args.input).is_absolute() else root / args.input
        output_path = Path(args.output) if Path(args.output).is_absolute() else root / args.output
        result = extract_cross_material_jats(input_path, output_path, args.source_id)
    elif args.command == "extract-polymer-jats":
        from .ingest import extract_polymer_contact_angle_jats

        config_path = args.config.resolve()
        root = config_path.parent.parent
        input_path = Path(args.input) if Path(args.input).is_absolute() else root / args.input
        output_path = Path(args.output) if Path(args.output).is_absolute() else root / args.output
        result = extract_polymer_contact_angle_jats(input_path, output_path, args.source_id)
    elif args.command == "extract-textile-jats":
        from .ingest import extract_textile_surface_jats

        config_path = args.config.resolve()
        root = config_path.parent.parent
        input_path = Path(args.input) if Path(args.input).is_absolute() else root / args.input
        output_path = Path(args.output) if Path(args.output).is_absolute() else root / args.output
        result = extract_textile_surface_jats(input_path, output_path, args.source_id)
    elif args.command == "extract-pla-jats":
        from .ingest import extract_pla_films_jats

        config_path = args.config.resolve()
        root = config_path.parent.parent
        input_path = Path(args.input) if Path(args.input).is_absolute() else root / args.input
        output_path = Path(args.output) if Path(args.output).is_absolute() else root / args.output
        result = extract_pla_films_jats(input_path, output_path, args.source_id)
    elif args.command == "extract-carbon-jats":
        from .ingest import extract_carbon_surface_jats

        config_path = args.config.resolve()
        root = config_path.parent.parent
        input_path = Path(args.input) if Path(args.input).is_absolute() else root / args.input
        output_path = Path(args.output) if Path(args.output).is_absolute() else root / args.output
        result = extract_carbon_surface_jats(input_path, output_path, args.source_id)
    elif args.command == "extract-facemask-jats":
        from .ingest import extract_facemask_surface_jats

        config_path = args.config.resolve()
        root = config_path.parent.parent
        input_path = Path(args.input) if Path(args.input).is_absolute() else root / args.input
        output_path = Path(args.output) if Path(args.output).is_absolute() else root / args.output
        result = extract_facemask_surface_jats(input_path, output_path, args.source_id)
    elif args.command == "extract-chitosan-jats":
        from .ingest import extract_chitosan_gelatin_jats

        config_path = args.config.resolve()
        root = config_path.parent.parent
        input_path = Path(args.input) if Path(args.input).is_absolute() else root / args.input
        output_path = Path(args.output) if Path(args.output).is_absolute() else root / args.output
        result = extract_chitosan_gelatin_jats(input_path, output_path, args.source_id)
    elif args.command == "merge-staged":
        from .ingest import merge_staged_source

        config_path = args.config.resolve()
        config = load_config(config_path)
        root = config_path.parent.parent
        input_path = Path(args.input) if Path(args.input).is_absolute() else root / args.input
        result = merge_staged_source(
            staged_path=input_path,
            data_dir=root / config["project"]["output_data_dir"],
            registry_path=root / "data" / "OPEN_DATA_REGISTRY_v4.csv",
            audit_dir=root / config["project"]["output_dir"] / "audit",
            source_id=args.source_id,
            seed=int(config["project"]["seed"]),
        )
    else:
        raise ValueError(f"Unsupported command: {args.command}")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
