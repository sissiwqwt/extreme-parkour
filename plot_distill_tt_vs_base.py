#!/usr/bin/env python3
"""Compare base vs TT distillation sweeps and their teacher-student gaps.

Outputs one focused view for each comparison. By default the plot window is
iteration 2200 to 6500.

Definitions:
  student advantage:
    positive means TT student is better than base student.

  distillation gap:
    positive means the student is worse than its teacher reference.
    For higher-is-better metrics, gap = teacher - student.
    For lower-is-better metrics, gap = student - teacher.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_METRICS = [
    "success_rate",
    "fall_rate",
    "stuck_rate",
    "mean_normalized_waypoints",
    "mean_mxd",
    "mean_edge_violation",
]

HIGHER_IS_BETTER = {
    "success_rate",
    "mean_normalized_waypoints",
    "mean_mxd",
}

METRIC_LABELS = {
    "success_rate": "Success Rate",
    "fall_rate": "Fall Rate",
    "stuck_rate": "Stuck Rate",
    "mean_normalized_waypoints": "Normalized Waypoints",
    "mean_mxd": "Mean MXD",
    "mean_edge_violation": "Mean Edge Violation",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot base-vs-TT distill metrics and distillation gaps."
    )
    parser.add_argument("--base-teacher-csv", default="base-teacher/sweep_summary.csv")
    parser.add_argument("--base-student-csv", default="base-distill/20260612_150259/summary_metrics.csv")
    parser.add_argument("--tt-teacher-csv", default="tt-teacher/20260615_022654/summary_metrics.csv")
    parser.add_argument("--tt-student-csv", default="distill_depth_tt/summary_metrics.csv")
    parser.add_argument("--output-dir", default="distill_tt_vs_base_plots")
    parser.add_argument("--metrics", nargs="*", default=DEFAULT_METRICS)
    parser.add_argument(
        "--teacher-ref",
        choices=("final", "best"),
        default="final",
        help="Teacher reference used for distillation gap.",
    )
    parser.add_argument(
        "--best-metric",
        default="success_rate",
        help="Metric used when --teacher-ref best.",
    )
    parser.add_argument(
        "--no-std",
        action="store_true",
        help="Do not draw student std bands in raw metric plots.",
    )
    parser.add_argument("--start-iter", type=int, default=2200)
    parser.add_argument("--end-iter", type=int, default=6500)
    return parser.parse_args()


def load_summary(path: str | Path, label: str) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing CSV: {path}")
    df = pd.read_csv(path).copy()
    if "checkpoint" not in df.columns:
        raise ValueError(f"{path} does not contain checkpoint")
    df["checkpoint"] = df["checkpoint"].astype(int)
    df["iteration"] = df["checkpoint"]
    df["label"] = label
    return df.sort_values("iteration")


def mean_col(metric: str) -> str:
    return f"{metric}_mean"


def std_col(metric: str) -> str:
    return f"{metric}_std"


def available_metrics(frames: list[pd.DataFrame], metrics: list[str]) -> list[str]:
    out = []
    for metric in metrics:
        col = mean_col(metric)
        if all(col in df.columns for df in frames):
            out.append(metric)
    return out


def teacher_reference(df: pd.DataFrame, metrics: list[str], mode: str, best_metric: str) -> pd.Series:
    if mode == "best":
        col = mean_col(best_metric)
        if col not in df.columns:
            raise ValueError(f"Cannot use --teacher-ref best: missing {col}")
        return df.loc[df[col].idxmax()]
    return df.sort_values("iteration").iloc[-1]


def metric_gap(teacher_value: float, student_value: float, metric: str) -> float:
    if metric in HIGHER_IS_BETTER:
        return teacher_value - student_value
    return student_value - teacher_value


def student_advantage(tt_value: float, base_value: float, metric: str) -> float:
    if metric in HIGHER_IS_BETTER:
        return tt_value - base_value
    return base_value - tt_value


def interp_on_common_x(left: pd.DataFrame, right: pd.DataFrame, metric: str) -> pd.DataFrame:
    col = mean_col(metric)
    left_x = left["iteration"].to_numpy(dtype=float)
    right_x = right["iteration"].to_numpy(dtype=float)
    xmin = max(float(left_x.min()), float(right_x.min()))
    xmax = min(float(left_x.max()), float(right_x.max()))
    if xmin > xmax:
        return pd.DataFrame(columns=["iteration", "left", "right"])

    xs = sorted(
        set(left.loc[(left["iteration"] >= xmin) & (left["iteration"] <= xmax), "iteration"].astype(int))
        | set(right.loc[(right["iteration"] >= xmin) & (right["iteration"] <= xmax), "iteration"].astype(int))
    )
    if not xs:
        return pd.DataFrame(columns=["iteration", "left", "right"])
    xs_np = np.asarray(xs, dtype=float)
    return pd.DataFrame(
        {
            "iteration": xs,
            "left": np.interp(xs_np, left_x, left[col].to_numpy(dtype=float)),
            "right": np.interp(xs_np, right_x, right[col].to_numpy(dtype=float)),
        }
    )


def make_gap_table(
    teacher: pd.Series,
    student: pd.DataFrame,
    label: str,
    metrics: list[str],
) -> pd.DataFrame:
    rows = []
    for _, row in student.iterrows():
        out = {"label": label, "iteration": int(row["iteration"]), "checkpoint": int(row["checkpoint"])}
        for metric in metrics:
            out[f"{metric}_gap"] = metric_gap(float(teacher[mean_col(metric)]), float(row[mean_col(metric)]), metric)
            out[f"{metric}_teacher"] = float(teacher[mean_col(metric)])
            out[f"{metric}_student"] = float(row[mean_col(metric)])
        rows.append(out)
    return pd.DataFrame(rows)


def make_advantage_table(base_student: pd.DataFrame, tt_student: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    tables = []
    for metric in metrics:
        aligned = interp_on_common_x(tt_student, base_student, metric)
        if aligned.empty:
            continue
        metric_rows = pd.DataFrame(
            {
                "iteration": aligned["iteration"],
                f"{metric}_tt_student": aligned["left"],
                f"{metric}_base_student": aligned["right"],
                f"{metric}_tt_advantage": [
                    student_advantage(tt, base, metric)
                    for tt, base in zip(aligned["left"], aligned["right"])
                ],
            }
        )
        tables.append(metric_rows)
    if not tables:
        return pd.DataFrame()
    out = tables[0]
    for table in tables[1:]:
        out = out.merge(table, on="iteration", how="outer")
    return out.sort_values("iteration")


def setup_axes(metrics: list[str], title: str):
    ncols = 2
    nrows = (len(metrics) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.5, 4.0 * nrows), squeeze=False)
    axes_flat = axes.ravel()
    fig.suptitle(title, fontsize=16, weight="bold", y=0.99)
    return fig, axes_flat


def finish_grid(fig, axes_flat, used: int, output_path: Path) -> None:
    for ax in axes_flat[used:]:
        ax.axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_student_metrics(
    base_student: pd.DataFrame,
    tt_student: pd.DataFrame,
    metrics: list[str],
    output_path: Path,
    start_iter: int,
    end_iter: int,
    draw_std: bool,
) -> None:
    fig, axes = setup_axes(metrics, f"Distilled Student Metrics: Iteration {start_iter}-{end_iter}")
    colors = {"Base student": "#2c7fb8", "TT student": "#d95f02"}
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        for label, df in [("Base student", base_student), ("TT student", tt_student)]:
            sub = df[(df["iteration"] >= start_iter) & (df["iteration"] <= end_iter)]
            if sub.empty:
                continue
            x = sub["iteration"].to_numpy()
            y = sub[mean_col(metric)].to_numpy()
            ax.plot(x, y, marker="o", linewidth=2.1, markersize=4.0, label=label, color=colors[label])
            sc = std_col(metric)
            if draw_std and sc in sub.columns:
                std = sub[sc].fillna(0.0).to_numpy()
                ax.fill_between(x, y - std, y + std, color=colors[label], alpha=0.15, linewidth=0)
        ax.set_title(METRIC_LABELS.get(metric, metric), weight="bold")
        ax.set_xlabel("Distillation Iteration")
        ax.set_xlim(start_iter, end_iter)
        ax.grid(True, alpha=0.25)
        if metric.endswith("_rate") or metric == "mean_normalized_waypoints":
            ax.set_ylim(-0.03, 1.03)
        ax.legend(frameon=True)
    finish_grid(fig, axes, len(metrics), output_path)


def plot_student_advantage(
    advantage: pd.DataFrame,
    metrics: list[str],
    output_path: Path,
    start_iter: int,
    end_iter: int,
) -> None:
    fig, axes = setup_axes(metrics, f"TT Student Advantage over Base Student: Iteration {start_iter}-{end_iter}")
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        col = f"{metric}_tt_advantage"
        sub = advantage[(advantage["iteration"] >= start_iter) & (advantage["iteration"] <= end_iter)]
        if col in sub.columns and not sub.empty:
            ax.plot(sub["iteration"], sub[col], marker="o", linewidth=2.1, color="#d95f02")
            ax.axhline(0.0, color="#333333", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_title(METRIC_LABELS.get(metric, metric), weight="bold")
        ax.set_xlabel("Common Distillation Iteration")
        ax.set_xlim(start_iter, end_iter)
        ax.set_ylabel("positive = TT better")
        ax.grid(True, alpha=0.25)
    finish_grid(fig, axes, len(metrics), output_path)


def plot_distillation_gap(
    gaps: pd.DataFrame,
    metrics: list[str],
    output_path: Path,
    start_iter: int,
    end_iter: int,
) -> None:
    fig, axes = setup_axes(metrics, f"Teacher-Student Distillation Gap: Iteration {start_iter}-{end_iter}")
    colors = {"Base gap": "#2c7fb8", "TT gap": "#d95f02"}
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        for label, group in gaps.groupby("label", sort=False):
            sub = group[(group["iteration"] >= start_iter) & (group["iteration"] <= end_iter)]
            col = f"{metric}_gap"
            if col not in sub.columns or sub.empty:
                continue
            ax.plot(sub["iteration"], sub[col], marker="o", linewidth=2.1, markersize=4.0, label=label, color=colors.get(label))
        ax.axhline(0.0, color="#333333", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_title(METRIC_LABELS.get(metric, metric), weight="bold")
        ax.set_xlabel("Student Distillation Iteration")
        ax.set_xlim(start_iter, end_iter)
        ax.set_ylabel("positive = student worse")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=True)
    finish_grid(fig, axes, len(metrics), output_path)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    base_teacher = load_summary(args.base_teacher_csv, "Base teacher")
    base_student = load_summary(args.base_student_csv, "Base student")
    tt_teacher = load_summary(args.tt_teacher_csv, "TT teacher")
    tt_student = load_summary(args.tt_student_csv, "TT student")

    metrics = available_metrics([base_teacher, base_student, tt_teacher, tt_student], args.metrics)
    if not metrics:
        raise ValueError("No requested metrics are available in all four CSVs")

    base_ref = teacher_reference(base_teacher, metrics, args.teacher_ref, args.best_metric)
    tt_ref = teacher_reference(tt_teacher, metrics, args.teacher_ref, args.best_metric)

    gaps = pd.concat(
        [
            make_gap_table(base_ref, base_student, "Base gap", metrics),
            make_gap_table(tt_ref, tt_student, "TT gap", metrics),
        ],
        ignore_index=True,
    )
    advantage = make_advantage_table(base_student, tt_student, metrics)

    base_teacher.to_csv(output_dir / "base_teacher_loaded.csv", index=False)
    base_student.to_csv(output_dir / "base_student_loaded.csv", index=False)
    tt_teacher.to_csv(output_dir / "tt_teacher_loaded.csv", index=False)
    tt_student.to_csv(output_dir / "tt_student_loaded.csv", index=False)
    gaps.to_csv(output_dir / "distillation_gap.csv", index=False)
    advantage.to_csv(output_dir / "tt_student_advantage.csv", index=False)

    draw_std = not args.no_std
    start_iter = args.start_iter
    end_iter = args.end_iter
    if start_iter > end_iter:
        raise ValueError(f"--start-iter must be <= --end-iter, got {start_iter} > {end_iter}")
    window_name = f"{start_iter}_to_{end_iter}"
    plot_student_metrics(
        base_student,
        tt_student,
        metrics,
        output_dir / f"student_metrics_{window_name}.png",
        start_iter,
        end_iter,
        draw_std,
    )
    plot_student_advantage(
        advantage,
        metrics,
        output_dir / f"tt_student_advantage_{window_name}.png",
        start_iter,
        end_iter,
    )
    plot_distillation_gap(
        gaps,
        metrics,
        output_dir / f"distillation_gap_{window_name}.png",
        start_iter,
        end_iter,
    )

    print(f"Wrote distillation comparison plots to {output_dir}")
    print(
        "Teacher references: "
        f"base ckpt {int(base_ref['checkpoint'])}, TT ckpt {int(tt_ref['checkpoint'])} "
        f"(--teacher-ref {args.teacher_ref})"
    )


if __name__ == "__main__":
    main()
