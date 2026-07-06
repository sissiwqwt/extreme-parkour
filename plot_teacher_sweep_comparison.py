#!/usr/bin/env python3
"""Plot base-vs-TT teacher sweep evaluation curves.

The TT teacher was trained with a phase-1/phase-2 schedule. By default, its
reported checkpoint 0 is treated as global iteration 3500, so every TT
checkpoint is shifted by +3500 on the x-axis.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_METRICS = [
    "success_rate",
    "stuck_rate",
    "fall_rate",
    "mean_normalized_waypoints",
    "mean_mxd",
    "mean_edge_violation",
]


METRIC_LABELS = {
    "success_rate": "Success Rate",
    "stuck_rate": "Stuck Rate",
    "fall_rate": "Fall Rate",
    "mean_normalized_waypoints": "Normalized Waypoints",
    "mean_mxd": "Mean MXD",
    "mean_edge_violation": "Mean Edge Violation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize base teacher and TT teacher sweep evaluation metrics."
    )
    parser.add_argument(
        "--base-csv",
        default="base-teacher/sweep_summary.csv",
        help="Base teacher summary CSV.",
    )
    parser.add_argument(
        "--tt-csv",
        default="tt-teacher/20260615_022654/summary_metrics.csv",
        help="TT teacher summary CSV.",
    )
    parser.add_argument(
        "--tt-offset",
        type=int,
        default=3500,
        help="Add this offset to TT checkpoints on the x-axis.",
    )
    parser.add_argument(
        "--output-dir",
        default="teacher_sweep_plots",
        help="Directory for plots and merged CSV.",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        default=DEFAULT_METRICS,
        help="Metric base names to plot, e.g. success_rate stuck_rate.",
    )
    parser.add_argument(
        "--no-std",
        action="store_true",
        help="Do not draw standard-deviation bands.",
    )
    parser.add_argument(
        "--show-tt-offset-line",
        action="store_true",
        default=True,
        help="Draw a vertical line at --tt-offset.",
    )
    return parser.parse_args()


def load_summary(path: str | Path, label: str, iteration_offset: int) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing CSV: {path}")

    df = pd.read_csv(path)
    if "checkpoint" not in df.columns:
        raise ValueError(f"{path} does not contain a checkpoint column")

    df = df.copy()
    df["teacher"] = label
    df["source_checkpoint"] = df["checkpoint"].astype(int)
    df["iteration"] = df["source_checkpoint"] + int(iteration_offset)
    return df


def metric_columns(df: pd.DataFrame, metric: str) -> tuple[str | None, str | None]:
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    if mean_col not in df.columns:
        return None, None
    if std_col not in df.columns:
        std_col = None
    return mean_col, std_col


def plot_metric(
    ax,
    df: pd.DataFrame,
    metric: str,
    draw_std: bool,
    tt_offset: int | None = None,
) -> bool:
    colors = {"Base": "#2c7fb8", "Task-targeted": "#d95f02"}
    markers = {"Base": "o", "Task-targeted": "s"}
    plotted = False

    for teacher, group in df.groupby("teacher", sort=False):
        mean_col, std_col = metric_columns(group, metric)
        if mean_col is None:
            continue
        group = group.sort_values("iteration")
        x = group["iteration"].to_numpy()
        y = group[mean_col].to_numpy()
        color = colors.get(teacher, None)
        ax.plot(
            x,
            y,
            marker=markers.get(teacher, "o"),
            linewidth=2.2,
            markersize=4.5,
            label=teacher,
            color=color,
        )
        if draw_std and std_col is not None:
            std = group[std_col].fillna(0.0).to_numpy()
            ax.fill_between(x, y - std, y + std, color=color, alpha=0.16, linewidth=0)
        plotted = True

    ax.set_title(METRIC_LABELS.get(metric, metric), fontsize=12, weight="bold")
    ax.set_xlabel("Training Iteration")
    ax.grid(True, alpha=0.25)
    if tt_offset is not None:
        ax.axvline(tt_offset, color="#666666", linestyle="--", linewidth=1.2, alpha=0.55)
    if metric.endswith("_rate") or metric == "mean_normalized_waypoints":
        ax.set_ylim(-0.03, 1.03)
    return plotted


def save_individual_plots(
    df: pd.DataFrame,
    metrics: list[str],
    output_dir: Path,
    draw_std: bool,
    tt_offset: int | None,
) -> None:
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(8.8, 5.0))
        if not plot_metric(ax, df, metric, draw_std, tt_offset):
            plt.close(fig)
            continue
        ax.legend(frameon=True)
        fig.tight_layout()
        fig.savefig(output_dir / f"{metric}.png", dpi=180)
        plt.close(fig)


def save_grid_plot(
    df: pd.DataFrame,
    metrics: list[str],
    output_dir: Path,
    draw_std: bool,
    tt_offset: int | None,
) -> None:
    available = []
    for metric in metrics:
        if any(metric_columns(group, metric)[0] is not None for _, group in df.groupby("teacher")):
            available.append(metric)
    if not available:
        raise ValueError("None of the requested metrics were found in the input CSV files")

    ncols = 2
    nrows = (len(available) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.0, 4.2 * nrows), squeeze=False)
    axes_flat = axes.ravel()
    for ax, metric in zip(axes_flat, available):
        plot_metric(ax, df, metric, draw_std, tt_offset)
    for ax in axes_flat[len(available):]:
        ax.axis("off")

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=True, bbox_to_anchor=(0.5, 0.955))
    fig.suptitle(
        "Teacher Sweep Evaluation: Base vs Task-targeted",
        fontsize=16,
        weight="bold",
        y=0.99,
    )
    if tt_offset is not None:
        fig.text(
            0.5,
            0.935,
            f"Dashed line: TT checkpoint 0 mapped to global iteration {tt_offset}",
            ha="center",
            va="center",
            fontsize=10,
            color="#555555",
        )
    fig.tight_layout(rect=(0, 0, 1, 0.915))
    fig.savefig(output_dir / "teacher_sweep_comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base = load_summary(args.base_csv, "Base", 0)
    tt = load_summary(args.tt_csv, "Task-targeted", args.tt_offset)
    merged = pd.concat([base, tt], ignore_index=True, sort=False)
    merged.to_csv(output_dir / "teacher_sweep_comparison.csv", index=False)

    draw_std = not args.no_std
    tt_offset_line = args.tt_offset if args.show_tt_offset_line else None
    save_grid_plot(merged, args.metrics, output_dir, draw_std, tt_offset_line)
    save_individual_plots(merged, args.metrics, output_dir, draw_std, tt_offset_line)

    print(f"Wrote plots and merged CSV to {output_dir}")
    print(f"TT checkpoint offset: +{args.tt_offset} iterations")


if __name__ == "__main__":
    main()
