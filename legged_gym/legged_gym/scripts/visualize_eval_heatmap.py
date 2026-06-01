"""Visualize terrain-by-difficulty evaluation metrics as heatmaps.

Example:
    python legged_gym/legged_gym/scripts/visualize_eval_heatmap.py \
        results/new_evaluation/nontt_all/base_all_all-difficulty_12000.json

    python legged_gym/legged_gym/scripts/visualize_eval_heatmap.py \
        results/new_evaluation/nontt_all/base_all_all-difficulty_12000.json \
        --drop-terrains narrow_gap,climbing_wall \
        --metrics success_rate mean_mxd
"""

import argparse
import csv
import json
import math
import os
from collections import OrderedDict, defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


DEFAULT_METRICS = OrderedDict(
    [
        ("episodes", ("count", None)),
        ("success_rate", ("mean", "success")),
        ("fall_rate", ("mean", "fall")),
        ("stuck_rate", ("mean", "stuck")),
        ("mean_mxd", ("mean", "mxd")),
        ("mean_normalized_waypoints", ("mean", "normalized_waypoints")),
        ("mean_episode_length", ("mean", "episode_length")),
        ("mean_edge_violation", ("mean", "edge_violation")),
        ("mean_heading_loss", ("mean", "mean_heading_loss")),
    ]
)

MUTED_ORANGE_RED = LinearSegmentedColormap.from_list(
    "muted_orange_red",
    ["#fff8ef", "#f2c59d", "#dc8b67", "#b84f49"],
)
TERRAIN_ORDER = [
    "parkour",
    "parkour_hurdle",
    "parkour_flat",
    "parkour_step",
    "parkour_gap",
    "alternating_step",
    "bean_gap",
    "beam_gap",
    "asymmetric_gap",
    "parkour_v2",
    "narrow_gap",
    "climbing_wall",
]
TERRAIN_RANK = {name: idx for idx, name in enumerate(TERRAIN_ORDER)}


def parse_name_list(values):
    items = []
    for value in values or []:
        for item in value.split(","):
            item = item.strip()
            if item:
                items.append(item)
    return items


def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f, object_pairs_hook=OrderedDict)


def resolve_csv_path(json_path, data, csv_path_arg):
    candidates = []
    if csv_path_arg:
        candidates.append(csv_path_arg)

    json_dir = os.path.dirname(os.path.abspath(json_path))
    json_stem = os.path.splitext(os.path.basename(json_path))[0]
    candidates.append(os.path.join(json_dir, json_stem + ".csv"))

    json_csv_path = data.get("csv_path")
    if json_csv_path:
        candidates.append(json_csv_path)
        candidates.append(os.path.join(json_dir, os.path.basename(json_csv_path)))

    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate

    raise FileNotFoundError(
        "Could not find the raw evaluation CSV. Tried: "
        + ", ".join(os.path.normpath(path) for path in candidates if path)
    )


def parse_float(value):
    if value is None or value == "":
        return math.nan
    return float(value)


def load_rows(csv_path):
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    required = {"terrain_name", "difficulty"}
    missing = sorted(required - set(fieldnames))
    if missing:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing)}")
    return rows, fieldnames


def sort_difficulties(values):
    return sorted(values, key=lambda item: parse_float(item))


def sort_terrains(values):
    return sorted(values, key=lambda name: (TERRAIN_RANK.get(name, len(TERRAIN_RANK)), name))


def aggregate(rows, metrics):
    buckets = defaultdict(list)
    terrains = []
    difficulties = set()
    seen_terrains = set()

    for row in rows:
        terrain = row["terrain_name"]
        difficulty = row["difficulty"]
        buckets[(terrain, difficulty)].append(row)
        difficulties.add(difficulty)
        if terrain not in seen_terrains:
            terrains.append(terrain)
            seen_terrains.add(terrain)

    terrains = sort_terrains(terrains)
    difficulties = sort_difficulties(difficulties)
    matrices = {
        metric: np.full((len(terrains), len(difficulties)), np.nan, dtype=float)
        for metric in metrics
    }

    for row_idx, terrain in enumerate(terrains):
        for col_idx, difficulty in enumerate(difficulties):
            bucket = buckets.get((terrain, difficulty), [])
            if not bucket:
                continue
            for metric in metrics:
                mode, source_col = DEFAULT_METRICS[metric]
                if mode == "count":
                    matrices[metric][row_idx, col_idx] = len(bucket)
                    continue

                values = [
                    parse_float(row.get(source_col))
                    for row in bucket
                    if row.get(source_col) not in (None, "")
                ]
                finite = [value for value in values if math.isfinite(value)]
                if finite:
                    matrices[metric][row_idx, col_idx] = float(np.mean(finite))

    return terrains, difficulties, matrices


def matrices_from_repaired_json(data, metrics, drop_terrains):
    terrain_difficulty_metrics = data.get("terrain_difficulty_metrics")
    if not terrain_difficulty_metrics:
        return None

    terrains = sort_terrains([name for name in terrain_difficulty_metrics.keys() if name not in drop_terrains])
    difficulty_values = set()
    for terrain in terrains:
        difficulty_values.update(terrain_difficulty_metrics[terrain].keys())
    difficulties = sort_difficulties(difficulty_values)

    matrices = {
        metric: np.full((len(terrains), len(difficulties)), np.nan, dtype=float)
        for metric in metrics
    }
    for row_idx, terrain in enumerate(terrains):
        by_difficulty = terrain_difficulty_metrics[terrain]
        for col_idx, difficulty in enumerate(difficulties):
            values = by_difficulty.get(str(difficulty), by_difficulty.get(difficulty, {}))
            for metric in metrics:
                matrices[metric][row_idx, col_idx] = parse_float(values.get(metric))
    return terrains, difficulties, matrices


def format_value(value, metric):
    if not np.isfinite(value):
        return ""
    if metric == "episodes":
        return f"{value:.0f}"
    if metric.endswith("_rate") or metric == "mean_normalized_waypoints":
        return f"{value:.2f}"
    if abs(value) >= 100:
        return f"{value:.0f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def color_limits(matrix, metric, vmin, vmax):
    if vmin is not None or vmax is not None:
        return vmin, vmax
    if metric.endswith("_rate") or metric == "mean_normalized_waypoints":
        return 0.0, 1.0

    finite = matrix[np.isfinite(matrix)]
    if finite.size == 0:
        return 0.0, 1.0
    low = float(np.nanmin(finite))
    high = float(np.nanmax(finite))
    if math.isclose(low, high):
        pad = abs(low) * 0.05 or 1.0
        return low - pad, high + pad
    return low, high


def safe_metric_name(metric):
    return metric.replace("/", "_").replace("\\", "_").replace(" ", "_")


def plot_heatmap(matrix, terrains, difficulties, metric, title, output_path, dpi, vmin, vmax):
    rows, cols = matrix.shape
    cell_size = 0.72
    left_margin = max(2.2, min(4.8, max(len(t) for t in terrains) * 0.12))
    bottom_margin = 1.0
    fig_width = left_margin + cols * cell_size + 1.0
    fig_height = bottom_margin + rows * cell_size + 0.9

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=False)
    masked = np.ma.masked_invalid(matrix)
    image = ax.imshow(masked, cmap=MUTED_ORANGE_RED, vmin=vmin, vmax=vmax)

    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.set_xticklabels(difficulties)
    ax.set_yticklabels(terrains)
    ax.set_xlabel("difficulty", labelpad=8)
    ax.set_aspect("equal")

    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="#ffffff", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", length=0, labelsize=9)

    threshold = vmin + (vmax - vmin) * 0.62 if vmax > vmin else vmax
    for row_idx in range(rows):
        for col_idx in range(cols):
            value = matrix[row_idx, col_idx]
            color = "#fffaf4" if np.isfinite(value) and value >= threshold else "#372a26"
            ax.text(
                col_idx,
                row_idx,
                format_value(value, metric),
                ha="center",
                va="center",
                fontsize=8,
                color=color,
            )

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(title, pad=14, fontsize=13)
    cbar = fig.colorbar(image, ax=ax, fraction=0.026, pad=0.02)
    cbar.set_label(metric, rotation=90)
    cbar.ax.tick_params(labelsize=8, length=0)
    cbar.outline.set_visible(False)

    fig.subplots_adjust(left=left_margin / fig_width, bottom=bottom_margin / fig_height, right=0.93, top=0.90)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize each evaluation metric as a terrain-by-difficulty heatmap."
    )
    parser.add_argument("json_path", help="Path to an evaluation JSON file.")
    parser.add_argument("--csv-path", help="Raw per-episode CSV path. Defaults to the CSV next to the JSON.")
    parser.add_argument(
        "-o",
        "--output-dir",
        help="Output directory. Defaults to '<json_stem>_heatmaps' next to the JSON.",
    )
    parser.add_argument(
        "--drop-terrains",
        nargs="*",
        default=[],
        help="Terrain names to exclude; repeated and comma-separated values are accepted.",
    )
    parser.add_argument(
        "--metrics",
        nargs="*",
        help="Metrics to plot. Defaults to all known metrics.",
    )
    parser.add_argument("--title-prefix", help="Optional prefix added before each metric in the figure title.")
    parser.add_argument("--dpi", type=int, default=180, help="Output image DPI.")
    parser.add_argument("--vmin", type=float, help="Shared color minimum override.")
    parser.add_argument("--vmax", type=float, help="Shared color maximum override.")
    args = parser.parse_args()

    data = load_json(args.json_path)
    drop_terrains = set(parse_name_list(args.drop_terrains))
    metrics = parse_name_list(args.metrics) if args.metrics else list(DEFAULT_METRICS.keys())
    unknown_metrics = sorted(set(metrics) - set(DEFAULT_METRICS.keys()))
    if unknown_metrics:
        raise ValueError(f"Unknown metrics: {', '.join(unknown_metrics)}")

    source = "repaired JSON terrain_difficulty_metrics"
    repaired = None if args.csv_path else matrices_from_repaired_json(data, metrics, drop_terrains)
    if repaired:
        terrains, difficulties, matrices = repaired
        if not terrains:
            raise ValueError("No terrains left after applying --drop-terrains.")
    else:
        csv_path = resolve_csv_path(args.json_path, data, args.csv_path)
        rows, fieldnames = load_rows(csv_path)
        rows = [row for row in rows if row["terrain_name"] not in drop_terrains]
        if not rows:
            raise ValueError("No rows left after applying --drop-terrains.")

        missing_columns = sorted(
            source_col
            for metric in metrics
            for _, source_col in [DEFAULT_METRICS[metric]]
            if source_col and source_col not in fieldnames
        )
        if missing_columns:
            raise ValueError(f"CSV is missing required metric columns: {', '.join(missing_columns)}")
        terrains, difficulties, matrices = aggregate(rows, metrics)
        source = csv_path
    output_dir = args.output_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.json_path)),
        os.path.splitext(os.path.basename(args.json_path))[0] + "_heatmaps",
    )
    os.makedirs(output_dir, exist_ok=True)

    json_name = os.path.basename(args.json_path)
    saved_paths = []
    for metric in metrics:
        matrix = matrices[metric]
        vmin, vmax = color_limits(matrix, metric, args.vmin, args.vmax)
        title = f"{args.title_prefix + ' - ' if args.title_prefix else ''}{metric} ({json_name})"
        output_path = os.path.join(output_dir, f"{safe_metric_name(metric)}.png")
        plot_heatmap(matrix, terrains, difficulties, metric, title, output_path, args.dpi, vmin, vmax)
        saved_paths.append(output_path)

    print(f"Loaded: {source}")
    print(f"Saved {len(saved_paths)} heatmaps to {output_dir}")
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
