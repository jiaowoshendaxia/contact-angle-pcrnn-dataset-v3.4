"""Generate final v3.4 LOO-SFE corrected paper figures.

The script follows the A-side claim guardrails:
- primary clean external validation uses clean_source_disjoint_external_LOO_SFE_114;
- raw all-liquid source-disjoint results are leakage-risk sensitivity only;
- high-risk 34 samples are supplementary diagnostics only;
- no figure presents PCRNN as outperforming all comparators across splits.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    import numpy as np
    import pandas as pd
except ModuleNotFoundError as exc:
    req_file = Path(__file__).with_name("requirements_figures_v3.4_LOO_corrected_20260520.txt")
    missing = exc.name or "a required package"
    raise SystemExit(
        f"Missing Python package: {missing}\n"
        f"Install figure dependencies with:\n"
        f"  python -m pip install -r \"{req_file}\"\n"
    ) from exc


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent

MAIN_RESULTS = ROOT / "results" / "main_result_table_v3.4_LOO_corrected_20260519.csv"
STAT_TESTS = ROOT / "results" / "combined_pcrnn_vs_baselines_stat_tests_v3.4_LOO_corrected_20260519.csv"
PLOT_READY = ROOT / "results" / "plot_ready_metrics_v3.4_LOO_corrected_20260519.csv"
FORWARD_CHECK = ROOT / "results" / "pcrnn_strict_v3.4_loo_sfe_forward_check_20260519.csv"

MODEL_ORDER = ["PCRNN", "Owens-Wendt", "XGBoost", "Random Forest", "Ordinary MLP"]
MAIN_SPLITS = [
    "internal_test",
    "balanced_holdout",
    "hard_external",
    "clean_source_disjoint_external_LOO_SFE_114",
]

SPLIT_LABELS = {
    "internal_test": "internal test",
    "balanced_holdout": "balanced holdout",
    "hard_external": "hard external",
    "clean_source_disjoint_external_LOO_SFE_114": "clean external\nLOO-SFE 114",
    "original_source_disjoint_external_all_liquid_SFE_matched_114": "before correction\nall-liquid SFE\nmatched 114",
    "high_risk_source_disjoint_external_34_original_only": "high-risk 34\noriginal only",
}

MODEL_COLORS = {
    "PCRNN": "#147D64",
    "Owens-Wendt": "#4C78A8",
    "XGBoost": "#F58518",
    "Random Forest": "#E45756",
    "Ordinary MLP": "#B279A2",
}

FRAMEWORK_COLORS = {
    "process": "#EAF4F4",
    "audit": "#FFF3BF",
    "model": "#DDEBFF",
    "eval": "#E8F5E9",
    "risk": "#FDE2E2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render v3.4 LOO-SFE corrected paper figures.")
    parser.add_argument("--main-results", type=Path, default=MAIN_RESULTS)
    parser.add_argument("--stat-tests", type=Path, default=STAT_TESTS)
    parser.add_argument("--plot-ready", type=Path, default=PLOT_READY)
    parser.add_argument("--forward-check", type=Path, default=FORWARD_CHECK)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "figures" / "regenerated")
    return parser.parse_args()


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "savefig.dpi": 300,
        }
    )


def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def load_inputs(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    main = pd.read_csv(args.main_results)
    stats = pd.read_csv(args.stat_tests)
    plot_ready = pd.read_csv(args.plot_ready)
    forward = pd.read_csv(args.forward_check)
    require_columns(main, ["analysis_split", "model_name", "n_samples", "MAE", "RMSE", "P90AE"], "main_results")
    require_columns(stats, ["analysis_split", "model_a", "model_b", "delta_MAE_a_minus_b", "ci_low", "ci_high", "p_value"], "stat_tests")
    require_columns(plot_ready, ["analysis_split", "model_name", "metric_name", "metric_value"], "plot_ready")
    require_columns(forward, ["delta_y_pred_loo_minus_original", "prediction_changed"], "forward_check")
    return main, stats, plot_ready, forward


def label_split(split: str) -> str:
    return SPLIT_LABELS.get(split, split.replace("_", " "))


def save(fig: plt.Figure, out_dir: Path, filename: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / filename, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def annotate_bars(ax: plt.Axes, bars, fmt: str = "{:.1f}", dy: float = 0.35, rotation: int = 0) -> None:
    for bar in bars:
        h = bar.get_height()
        if not math.isfinite(h):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + dy,
            fmt.format(h),
            ha="center",
            va="bottom",
            fontsize=7.3,
            rotation=rotation,
        )


def mae_label(value: float, split: str, model: str) -> str:
    if split == "clean_source_disjoint_external_LOO_SFE_114" and model in {"PCRNN", "Owens-Wendt"}:
        return f"{value:.2f}"
    return f"{value:.1f}"


def draw_box(ax: plt.Axes, xy: tuple[float, float], text: str, color: str, width: float = 1.72, height: float = 0.76) -> None:
    x, y = xy
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.045,rounding_size=0.04",
        linewidth=1.2,
        edgecolor="#263238",
        facecolor=color,
    )
    ax.add_patch(box)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=9.3)


def draw_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.2,
            color="#455A64",
            connectionstyle="arc3,rad=0.0",
        )
    )


def figure1_framework(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(12.6, 5.7), constrained_layout=True)
    ax.set_axis_off()
    ax.set_xlim(0, 10.8)
    ax.set_ylim(0, 4.7)

    boxes = [
        ((0.25, 2.9), "Dataset\nconstruction", FRAMEWORK_COLORS["process"]),
        ((2.15, 2.9), "Source-disjoint\nsplit design", FRAMEWORK_COLORS["process"]),
        ((4.05, 2.9), "SFE leakage\naudit", FRAMEWORK_COLORS["audit"]),
        ((5.95, 2.9), "LOO-SFE\ncorrection", FRAMEWORK_COLORS["audit"]),
        ((7.85, 2.9), "PCRNN\nprediction", FRAMEWORK_COLORS["model"]),
        ((9.55, 2.9), "Strict\nevaluation", FRAMEWORK_COLORS["eval"], 1.0),
    ]
    centers = []
    for item in boxes:
        if len(item) == 4:
            xy, text, color, width = item
        else:
            xy, text, color = item
            width = 1.45
        draw_box(ax, xy, text, color, width=width)
        centers.append((xy[0] + width, xy[1] + 0.38))

    starts = [(0.25 + 1.45, 3.28), (2.15 + 1.45, 3.28), (4.05 + 1.45, 3.28), (5.95 + 1.45, 3.28), (7.85 + 1.45, 3.28)]
    ends = [(2.15, 3.28), (4.05, 3.28), (5.95, 3.28), (7.85, 3.28), (9.55, 3.28)]
    for s, e in zip(starts, ends):
        draw_arrow(ax, s, e)

    draw_box(
        ax,
        (5.75, 1.45),
        "Primary clean external\nclean_source_disjoint_external\nLOO-SFE 114",
        "#DCFCE7",
        width=2.35,
        height=0.9,
    )
    draw_arrow(ax, (6.8, 2.9), (6.93, 2.36))

    draw_box(
        ax,
        (3.95, 0.55),
        "Raw all-liquid SFE\nsensitivity only\nnot primary external",
        "#FEF3C7",
        width=2.2,
        height=0.85,
    )
    draw_arrow(ax, (4.85, 2.9), (5.0, 1.42))

    draw_box(
        ax,
        (7.05, 0.55),
        "High-risk 34\ndiagnostic only\nnot mixed into main claim",
        FRAMEWORK_COLORS["risk"],
        width=2.25,
        height=0.85,
    )
    draw_arrow(ax, (7.15, 2.9), (8.05, 1.42))

    ax.text(
        0.25,
        4.35,
        "Leakage-aware physics-constrained contact-angle prediction workflow",
        fontsize=15,
        weight="bold",
        ha="left",
    )
    ax.text(
        0.25,
        0.12,
        "Key rule: the clean external claim uses LOO-SFE 114; all-liquid SFE and high-risk 34 are separated as sensitivity/diagnostic evidence.",
        fontsize=9.0,
        color="#455A64",
        ha="left",
    )
    save(fig, out_dir, "Figure1_framework_v3.4_LOO_corrected.png")


def figure2_main_mae(main: pd.DataFrame, out_dir: Path) -> None:
    df = main[(main["analysis_split"].isin(MAIN_SPLITS)) & (main["model_name"].isin(MODEL_ORDER))].copy()
    df["analysis_split"] = pd.Categorical(df["analysis_split"], MAIN_SPLITS, ordered=True)
    df["model_name"] = pd.Categorical(df["model_name"], MODEL_ORDER, ordered=True)
    df = df.sort_values(["analysis_split", "model_name"])

    x = np.arange(len(MAIN_SPLITS))
    width = 0.15
    offsets = np.linspace(-2, 2, len(MODEL_ORDER)) * width
    fig, ax = plt.subplots(figsize=(12.2, 6.2), constrained_layout=True)
    ymax = df["MAE"].max() * 1.22
    for model, offset in zip(MODEL_ORDER, offsets):
        sub = df[df["model_name"] == model].set_index("analysis_split").reindex(MAIN_SPLITS)
        bars = ax.bar(
            x + offset,
            sub["MAE"].to_numpy(dtype=float),
            width,
            label=model,
            color=MODEL_COLORS[model],
            edgecolor="#263238",
            linewidth=0.35,
        )
        for bar, split, value in zip(bars, MAIN_SPLITS, sub["MAE"].to_numpy(dtype=float)):
            if not math.isfinite(value):
                continue
            extra_dy = ymax * 0.032 if split == "clean_source_disjoint_external_LOO_SFE_114" and model == "Owens-Wendt" else 0.0
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + ymax * 0.011 + extra_dy,
                mae_label(value, split, model),
                ha="center",
                va="bottom",
                fontsize=7.3,
                rotation=90,
            )
    ax.set_xticks(x)
    ax.set_xticklabels([label_split(s) for s in MAIN_SPLITS])
    ax.set_ylabel("MAE (deg)")
    ax.set_ylim(0, ymax)
    ax.set_title("Main MAE comparison using LOO-SFE clean external validation")
    ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    ax.grid(axis="y", alpha=0.22)
    ax.text(
        0.985,
        0.94,
        "Clean external: PCRNN ~= OW; weaker than XGBoost;\nbetter than RF; not significantly different from MLP",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.4,
        color="#374151",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#F8FAFC", "edgecolor": "#CBD5E1"},
    )
    ax.text(
        0.0,
        -0.20,
        "Raw all-liquid source-disjoint apparent-SFE MAE is excluded from the main external panel.",
        transform=ax.transAxes,
        fontsize=8.4,
        color="#455A64",
        va="top",
    )
    save(fig, out_dir, "Figure2_main_MAE_comparison_v3.4_LOO_corrected.png")


def figure3_sensitivity(main: pd.DataFrame, out_dir: Path) -> None:
    before = "original_source_disjoint_external_all_liquid_SFE_matched_114"
    after = "clean_source_disjoint_external_LOO_SFE_114"
    df = main[(main["analysis_split"].isin([before, after])) & (main["model_name"].isin(MODEL_ORDER))].copy()
    pivot = df.pivot(index="model_name", columns="analysis_split", values="MAE").reindex(MODEL_ORDER)

    fig, ax = plt.subplots(figsize=(9.7, 6.1), constrained_layout=True)
    xs = [0, 1]
    before_label_offset = {
        "PCRNN": -0.18,
        "Owens-Wendt": 0.10,
        "XGBoost": 0.36,
        "Random Forest": 0.0,
        "Ordinary MLP": 0.0,
    }
    after_label_offset = {
        "PCRNN": -0.34,
        "Owens-Wendt": 0.24,
        "XGBoost": 0.0,
        "Random Forest": 0.0,
        "Ordinary MLP": 0.0,
    }
    for model in MODEL_ORDER:
        y = [pivot.loc[model, before], pivot.loc[model, after]]
        lw = 3.2 if model == "PCRNN" else 1.8
        alpha = 1.0 if model == "PCRNN" else 0.72
        ax.plot(xs, y, marker="o", markersize=7, linewidth=lw, color=MODEL_COLORS[model], alpha=alpha, label=model)
        ax.text(-0.03, y[0] + before_label_offset.get(model, 0.0), f"{y[0]:.1f}", ha="right", va="center", fontsize=8)
        ax.text(1.03, y[1] + after_label_offset.get(model, 0.0), mae_label(float(y[1]), after, model), ha="left", va="center", fontsize=8)

    pcrnn_before = float(pivot.loc["PCRNN", before])
    pcrnn_after = float(pivot.loc["PCRNN", after])
    ax.annotate(
        f"PCRNN: {pcrnn_before:.4f} -> {pcrnn_after:.4f} deg",
        xy=(1, pcrnn_after),
        xytext=(0.34, pcrnn_after + 5.0),
        arrowprops={"arrowstyle": "->", "color": "#147D64", "lw": 1.4},
        fontsize=10,
        color="#0F5132",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "#E8F5E9", "edgecolor": "#9CCC9C"},
    )
    ax.set_xlim(-0.22, 1.22)
    ax.set_xticks(xs)
    ax.set_xticklabels(["before correction\nall-liquid SFE\nmatched 114", "after correction\nLOO-SFE\nmatched 114"])
    ax.set_ylabel("MAE (deg)")
    ax.set_title("LOO-SFE leakage sensitivity on matched 114 samples")
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.02, 0.98))
    ax.grid(axis="y", alpha=0.24)
    ax.text(
        0.0,
        -0.18,
        "This sensitivity panel explains leakage risk; the after-correction LOO-SFE 114 cohort is the primary clean external result.",
        transform=ax.transAxes,
        fontsize=8.4,
        color="#455A64",
        va="top",
    )
    save(fig, out_dir, "Figure3_LOO_SFE_sensitivity_v3.4_LOO_corrected.png")


def p_label(p: float) -> str:
    if p < 0.001:
        return "p<0.001"
    return f"p={p:.4f}"


def figure4_stat_tests(stats: pd.DataFrame, out_dir: Path) -> None:
    split = "clean_source_disjoint_external_LOO_SFE_114"
    df = stats[(stats["analysis_split"] == split) & (stats["model_a"] == "PCRNN") & (stats["model_b"].isin(MODEL_ORDER[1:]))].copy()
    df["model_b"] = pd.Categorical(df["model_b"], MODEL_ORDER[1:], ordered=True)
    df = df.sort_values("model_b")

    y = np.arange(len(df))
    delta = df["delta_MAE_a_minus_b"].to_numpy(dtype=float)
    low = df["ci_low"].to_numpy(dtype=float)
    high = df["ci_high"].to_numpy(dtype=float)
    xerr = np.vstack([delta - low, high - delta])

    fig, ax = plt.subplots(figsize=(8.6, 5.2), constrained_layout=True)
    ax.axvline(0, color="#263238", linewidth=1.0, linestyle="--", alpha=0.75)
    colors = ["#6B7280", "#F58518", "#147D64", "#6B7280"]
    ax.errorbar(delta, y, xerr=xerr, fmt="none", ecolor="#374151", elinewidth=2.0, capsize=4, zorder=1)
    ax.scatter(delta, y, s=72, color=colors, edgecolor="#263238", zorder=2)
    for yi, row in zip(y, df.itertuples(index=False)):
        ax.text(row.ci_high + 0.22, yi, p_label(float(row.p_value)), va="center", fontsize=8.4)
    ax.set_yticks(y)
    ax.set_yticklabels([f"PCRNN vs {m}" for m in df["model_b"]])
    ax.invert_yaxis()
    ax.set_xlabel("delta MAE = MAE(PCRNN) - MAE(comparator), deg")
    ax.set_title("Clean external paired tests: LOO-SFE 114")
    ax.grid(axis="x", alpha=0.24)
    ax.text(
        0.01,
        -0.18,
        "Negative favors PCRNN; positive favors comparator. CI crossing zero indicates no clear paired MAE difference.",
        transform=ax.transAxes,
        fontsize=8.4,
        color="#455A64",
        va="top",
    )
    save(fig, out_dir, "Figure4_clean_external_stat_tests_v3.4_LOO_corrected.png")


def figure5_forward_check(forward: pd.DataFrame, out_dir: Path) -> tuple[int, int, float, float]:
    delta = forward["delta_y_pred_loo_minus_original"].astype(float)
    abs_delta = delta.abs()
    n = len(forward)
    changed = int((forward["prediction_changed"].astype(str).str.lower() == "yes").sum())
    mean_abs = float(abs_delta.mean())
    max_abs = float(abs_delta.max())

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), constrained_layout=True)
    axes[0].hist(delta, bins=24, color="#7B68A6", edgecolor="white")
    axes[0].axvline(0, color="#263238", linestyle="--", linewidth=1.0)
    axes[0].set_xlabel("LOO-SFE prediction - all-liquid prediction (deg)")
    axes[0].set_ylabel("Number of samples")
    axes[0].set_title("Signed PCRNN prediction shift")

    y = np.zeros(len(abs_delta))
    rng = np.random.default_rng(20260519)
    jitter = rng.normal(0, 0.035, len(abs_delta))
    axes[1].boxplot(abs_delta, vert=False, widths=0.42, patch_artist=True, showfliers=False, boxprops={"facecolor": "#DDEBFF", "edgecolor": "#374151"})
    axes[1].scatter(abs_delta, y + 1 + jitter, s=18, alpha=0.55, color="#147D64", edgecolors="none")
    axes[1].set_xlabel("Absolute prediction shift (deg)")
    axes[1].set_yticks([])
    axes[1].set_title("Absolute shift after feature replacement")
    axes[1].text(
        0.98,
        0.92,
        f"{changed}/{n} predictions changed\nmean abs shift = {mean_abs:.4f} deg\nmax abs shift = {max_abs:.4f} deg",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#F8FAFC", "edgecolor": "#CBD5E1"},
    )
    fig.suptitle("PCRNN LOO-SFE forward check on 114 feasible source-disjoint samples", y=1.03)
    save(fig, out_dir, "Figure5_forward_check_v3.4_LOO_corrected.png")
    return changed, n, mean_abs, max_abs


def figure_s1_high_risk(main: pd.DataFrame, out_dir: Path) -> None:
    split = "high_risk_source_disjoint_external_34_original_only"
    df = main[(main["analysis_split"] == split) & (main["model_name"].isin(MODEL_ORDER))].copy()
    df["model_name"] = pd.Categorical(df["model_name"], MODEL_ORDER, ordered=True)
    df = df.sort_values("model_name")

    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(8.8, 5.1), constrained_layout=True)
    bars = ax.bar(
        x,
        df["MAE"],
        color=[MODEL_COLORS[m] for m in df["model_name"]],
        edgecolor="#263238",
        linewidth=0.45,
    )
    annotate_bars(ax, bars, dy=max(df["MAE"]) * 0.025)
    ax.set_xticks(x)
    ax.set_xticklabels(df["model_name"], rotation=20, ha="right")
    ax.set_ylabel("MAE (deg)")
    ax.set_title("High-risk 34 diagnostic only / not primary external claim")
    ax.grid(axis="y", alpha=0.24)
    ax.text(
        0.01,
        0.94,
        "Original-only apparent SFE; LOO-SFE infeasible/high-risk cohort",
        transform=ax.transAxes,
        va="top",
        fontsize=8.4,
        color="#7F1D1D",
        bbox={"boxstyle": "round,pad=0.32", "facecolor": "#FEE2E2", "edgecolor": "#FCA5A5"},
    )
    save(fig, out_dir, "FigureS1_high_risk_diagnostic_v3.4_LOO_corrected.png")


def clean_external_values(main: pd.DataFrame) -> dict[str, float]:
    split = "clean_source_disjoint_external_LOO_SFE_114"
    sub = main[main["analysis_split"] == split].set_index("model_name")
    return {model: float(sub.loc[model, "MAE"]) for model in MODEL_ORDER if model in sub.index}


def write_generation_note(
    out_dir: Path,
    main: pd.DataFrame,
    stats: pd.DataFrame,
    changed: int,
    n_forward: int,
    mean_abs_shift: float,
    max_abs_shift: float,
) -> None:
    clean = clean_external_values(main)
    pcrnn_sensitivity = main[
        (main["model_name"] == "PCRNN")
        & (
            main["analysis_split"].isin(
                [
                    "original_source_disjoint_external_all_liquid_SFE_matched_114",
                    "clean_source_disjoint_external_LOO_SFE_114",
                ]
            )
        )
    ].set_index("analysis_split")
    before = float(pcrnn_sensitivity.loc["original_source_disjoint_external_all_liquid_SFE_matched_114", "MAE"])
    after = float(pcrnn_sensitivity.loc["clean_source_disjoint_external_LOO_SFE_114", "MAE"])
    clean_stats = stats[stats["analysis_split"] == "clean_source_disjoint_external_LOO_SFE_114"]

    lines = [
        "# Figure Generation Note v3.4 LOO-SFE Corrected 20260519",
        "",
        "## Source Files",
        "",
        "- Figure 1: manuscript/guardrail files plus corrected evidence hierarchy.",
        "- Figure 2: `outputs/main_result_table_v3.4_LOO_corrected_20260519.csv`.",
        "- Figure 3: `outputs/main_result_table_v3.4_LOO_corrected_20260519.csv`, using matched 114 before/after LOO-SFE rows.",
        "- Figure 4: `outputs/combined_pcrnn_vs_baselines_stat_tests_v3.4_LOO_corrected_20260519.csv`.",
        "- Figure 5: `outputs/pcrnn_strict/pcrnn_strict_v3.4_loo_sfe_forward_check_20260519.csv`.",
        "- Supplementary Figure S1: `outputs/main_result_table_v3.4_LOO_corrected_20260519.csv`, high-risk 34 diagnostic rows.",
        "",
        "## Figure Conclusions",
        "",
        "### Figure 1",
        "Shows the workflow: dataset construction, source-disjoint split, SFE leakage audit, LOO-SFE correction, PCRNN prediction, and strict evaluation. It explicitly separates primary clean external validation from leakage-risk sensitivity and high-risk diagnostics.",
        "",
        "### Figure 2",
        f"Uses `clean_source_disjoint_external_LOO_SFE_114` for the main clean external panel. Clean external MAE values: PCRNN {clean.get('PCRNN'):.4f}, Owens-Wendt {clean.get('Owens-Wendt'):.4f}, XGBoost {clean.get('XGBoost'):.4f}, Random Forest {clean.get('Random Forest'):.4f}, Ordinary MLP {clean.get('Ordinary MLP'):.4f}. This supports the corrected conclusion that PCRNN is essentially tied with Owens-Wendt, weaker than XGBoost in MAE, better than Random Forest, and close to Ordinary MLP.",
        "",
        "### Figure 3",
        f"Shows leakage sensitivity on matched 114 samples. PCRNN MAE changes from {before:.4f} before correction to {after:.4f} after LOO-SFE correction, demonstrating that all-liquid apparent SFE can materially alter the external validation conclusion.",
        "",
        "### Figure 4",
        "Shows paired delta MAE and 95% CI for clean external comparisons:",
    ]
    for row in clean_stats.itertuples(index=False):
        lines.append(
            f"- PCRNN vs {row.model_b}: delta MAE {row.delta_MAE_a_minus_b:.4f}, "
            f"95% CI [{row.ci_low:.4f}, {row.ci_high:.4f}], p={row.p_value:.4g}."
        )
    lines.extend(
        [
            "",
            "### Figure 5",
            f"Forward check confirms that {changed}/{n_forward} PCRNN predictions changed after LOO-SFE feature replacement. Mean absolute prediction shift = {mean_abs_shift:.4f} deg; max absolute shift = {max_abs_shift:.4f} deg.",
            "",
            "### Supplementary Figure S1",
            "Shows high-risk 34 performance as diagnostic only. These rows are not mixed into the primary clean external claim.",
            "",
            "## Claim Guardrail Compliance",
            "",
            "- The full raw source-disjoint apparent-SFE sensitivity result is not used in the main external performance figure.",
            "- The main external figure uses `clean_source_disjoint_external_LOO_SFE_114`.",
            "- The high-risk 34 cohort appears only in Supplementary Figure S1 and is labeled diagnostic only.",
            "- No figure presents PCRNN as outperforming all comparators across splits or as superior on the clean external cohort.",
            "- Figure 3 labels all-liquid apparent SFE as leakage-risk sensitivity.",
            "- Figure 4 reports uncertainty and p values rather than only winner labels.",
        ]
    )
    (out_dir / "figure_generation_note_v3.4_LOO_corrected_20260519.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    configure_style()
    main_results, stat_tests, _plot_ready, forward = load_inputs(args)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    figure1_framework(args.out_dir)
    figure2_main_mae(main_results, args.out_dir)
    figure3_sensitivity(main_results, args.out_dir)
    figure4_stat_tests(stat_tests, args.out_dir)
    changed, n_forward, mean_abs_shift, max_abs_shift = figure5_forward_check(forward, args.out_dir)
    figure_s1_high_risk(main_results, args.out_dir)
    write_generation_note(args.out_dir, main_results, stat_tests, changed, n_forward, mean_abs_shift, max_abs_shift)

    print(f"Figures and note written to {args.out_dir}")


if __name__ == "__main__":
    main()
