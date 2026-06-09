"""Sweep checkpoint evaluations and plot metric curves.

Default target:
    proj_name=parkour_heading
    exptid in:
        heading_pre1000_latent1_unfreeze
        heading_pre300_latent1_unfreeze
        heading_pre0_latent1_unfreeze

Example:
    python sweep_evaluation.py --repeats 3

Cross-project example:
    python sweep_evaluation.py \
        --experiments "{parkour_heading,heading_pre1000_latent1_unfreeze},{another_project,another_run}"
"""

import argparse
import csv
import json
import re
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


DEFAULT_PROJ_NAME = "parkour_heading"
DEFAULT_EXPTIDS = (
    "heading_pre1000_latent1_unfreeze",
    "heading_pre300_latent1_unfreeze",
    "heading_pre0_latent1_unfreeze",
)
DEFAULT_METRICS = (
    "success_rate",
    "mean_mxd",
    "mean_normalized_waypoints",
    "fall_rate",
    "stuck_rate",
    "mean_edge_violation",
    "mean_heading_loss",
)


@dataclass(frozen=True)
class Experiment:
    proj_name: str
    exptid: str

    @property
    def label(self):
        return self.exptid if self.proj_name == DEFAULT_PROJ_NAME else f"{self.proj_name}/{self.exptid}"


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "value"


def parse_experiment(value, default_proj_name):
    value = value.strip()
    if value.startswith("{") and value.endswith("}"):
        parts = [part.strip() for part in value[1:-1].split(",", 1)]
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise argparse.ArgumentTypeError("Braced experiment must be {PROJ_NAME,EXPTID}.")
        return Experiment(parts[0], parts[1])

    if ":" in value:
        proj_name, exptid = value.split(":", 1)
    elif "/" in value:
        proj_name, exptid = value.split("/", 1)
    else:
        proj_name, exptid = default_proj_name, value
    proj_name = proj_name.strip()
    exptid = exptid.strip()
    if not proj_name or not exptid:
        raise argparse.ArgumentTypeError(
            "Experiment must be EXPTID, PROJ_NAME:EXPTID, or PROJ_NAME/EXPTID."
        )
    return Experiment(proj_name, exptid)


def parse_experiment_list(value, default_proj_name):
    value = value.strip()
    if not value:
        return []

    if "{" not in value:
        return [parse_experiment(part, default_proj_name) for part in value.split(",") if part.strip()]

    experiments = []
    pos = 0
    for match in re.finditer(r"\{([^{}]+)\}", value):
        if value[pos:match.start()].strip(" ,"):
            raise argparse.ArgumentTypeError(
                "Experiment list must use {PROJ_NAME,EXPTID},{PROJ_NAME,EXPTID}."
            )
        experiments.append(parse_experiment(match.group(0), default_proj_name))
        pos = match.end()
    if value[pos:].strip(" ,"):
        raise argparse.ArgumentTypeError(
            "Experiment list must use {PROJ_NAME,EXPTID},{PROJ_NAME,EXPTID}."
        )
    return experiments


def resolve_experiments(args):
    raw_values = []
    raw_values.extend(args.experiment)
    if args.experiments:
        raw_values.append(args.experiments)

    if not raw_values:
        return [Experiment(args.proj_name, exptid) for exptid in DEFAULT_EXPTIDS]

    experiments = []
    seen = set()
    for value in raw_values:
        for exp in parse_experiment_list(value, args.proj_name):
            key = (exp.proj_name, exp.exptid)
            if key not in seen:
                experiments.append(exp)
                seen.add(key)
    return experiments


def script_dir():
    return Path(__file__).resolve().parent


def legged_root():
    return script_dir().parents[1]


def checkpoint_steps(log_dir, start_step):
    if not log_dir.is_dir():
        return []

    steps = []
    for path in log_dir.glob("model_*.pt"):
        match = re.fullmatch(r"model_(\d+)\.pt", path.name)
        if match:
            step = int(match.group(1))
            if step >= start_step:
                steps.append(step)
    return sorted(set(steps))


def latest_json(output_dir):
    json_paths = sorted(output_dir.glob("*.json"), key=lambda path: path.stat().st_mtime)
    return json_paths[-1] if json_paths else None


def run_one_evaluation(args, extra_eval_args, exp, checkpoint, repeat_idx, run_root):
    run_id = f"ckpt{checkpoint}_rep{repeat_idx}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    output_dir = (
        run_root
        / safe_name(exp.proj_name)
        / safe_name(exp.exptid)
        / f"checkpoint_{checkpoint}"
        / f"repeat_{repeat_idx}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        args.python_bin,
        str(script_dir() / "evaluation.py"),
        "--task",
        args.task,
        "--device",
        args.device,
        "--rl_device",
        args.rl_device,
        "--proj_name",
        exp.proj_name,
        "--exptid",
        exp.exptid,
        "--checkpoint",
        str(checkpoint),
        "--policy_id",
        exp.label,
        "--policy_type",
        args.policy_type,
        "--terrain_set",
        args.terrain_set,
        "--difficulty_mode",
        args.difficulty_mode,
        "--eval_episodes",
        str(args.eval_episodes),
        "--num_envs",
        str(args.num_envs),
        "--seed",
        str(args.seed_base + repeat_idx),
        "--run_id",
        run_id,
        "--output_dir",
        str(output_dir),
        "--headless",
    ]
    if args.use_camera:
        cmd.append("--use_camera")
    if args.enable_heading_model:
        cmd.append("--enable_heading_model")
    if args.terrain_names:
        cmd.extend(["--terrain_names", args.terrain_names])
    if args.heading_eval_mode:
        cmd.extend(["--heading_eval_mode", args.heading_eval_mode])
    if args.heading_corruption_std is not None:
        cmd.extend(["--heading_corruption_std", str(args.heading_corruption_std)])
    cmd.extend(extra_eval_args)

    print(f"Running {exp.label} checkpoint={checkpoint} repeat={repeat_idx}")
    subprocess.run(cmd, cwd=str(script_dir()), check=True)

    json_path = latest_json(output_dir)
    if json_path is None:
        raise RuntimeError(f"evaluation.py did not write a JSON file in {output_dir}")
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "proj_name": exp.proj_name,
        "exptid": exp.exptid,
        "label": exp.label,
        "checkpoint": checkpoint,
        "repeat": repeat_idx,
        "json_path": str(json_path),
        "csv_path": data.get("csv_path", ""),
        **data.get("metrics", {}),
    }


def mean_std(values):
    values = [float(value) for value in values]
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean, var ** 0.5


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows, metrics):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["proj_name"], row["exptid"], row["label"], row["checkpoint"])].append(row)

    summary = []
    for (proj_name, exptid, label, checkpoint), group in sorted(
        grouped.items(), key=lambda item: (item[0][2], int(item[0][3]))
    ):
        out = {
            "proj_name": proj_name,
            "exptid": exptid,
            "label": label,
            "checkpoint": checkpoint,
            "repeats": len(group),
        }
        for metric in metrics:
            mean, std = mean_std([row.get(metric, 0.0) for row in group])
            out[f"{metric}_mean"] = mean
            out[f"{metric}_std"] = std
        summary.append(out)
    return summary


def plot_metrics(summary_rows, metrics, output_dir):
    import matplotlib.pyplot as plt

    by_label = defaultdict(list)
    for row in summary_rows:
        by_label[row["label"]].append(row)

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(8.5, 5.0))
        for label, rows in sorted(by_label.items()):
            rows = sorted(rows, key=lambda row: int(row["checkpoint"]))
            xs = [int(row["checkpoint"]) for row in rows]
            ys = [float(row[f"{metric}_mean"]) for row in rows]
            yerr = [float(row[f"{metric}_std"]) for row in rows]
            ax.errorbar(xs, ys, yerr=yerr, marker="o", linewidth=1.8, capsize=3, label=label)
        ax.set_xlabel("checkpoint step")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(plot_dir / f"{safe_name(metric)}.png", dpi=180)
        plt.close(fig)


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Run evaluation.py over multiple checkpoints and experiments, repeat each "
            "evaluation, aggregate metrics, and plot curves."
        )
    )
    parser.add_argument(
        "--experiment",
        action="append",
        default=[],
        help=(
            "Experiment as EXPTID, PROJ_NAME:EXPTID, PROJ_NAME/EXPTID, or "
            "{PROJ_NAME,EXPTID}. Repeatable."
        ),
    )
    parser.add_argument(
        "--experiments",
        default=None,
        help='Comma-separated experiment list, e.g. "{proj1,expt1},{proj2,expt2}".',
    )
    parser.add_argument("--proj-name", default=DEFAULT_PROJ_NAME, help="Default project for bare EXPTID values.")
    parser.add_argument("--start-step", type=int, default=0, help="Evaluate checkpoints with step >= this value.")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=1000)
    parser.add_argument("--python-bin", default=sys.executable or "python")
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--task", default="a1")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rl-device", default="cuda:0")
    parser.add_argument("--policy-type", choices=("auto", "base", "depth"), default="depth")
    parser.add_argument("--terrain-set", default="effective")
    parser.add_argument("--terrain-names", default=None)
    parser.add_argument("--difficulty-mode", default="single")
    parser.add_argument("--eval-episodes", type=int, default=256)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--heading-eval-mode", default="predicted")
    parser.add_argument("--heading-corruption-std", type=float, default=None)
    parser.add_argument("--metrics", nargs="*", default=list(DEFAULT_METRICS))
    parser.add_argument("--use-camera", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-heading-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Only write CSV/JSON summaries; do not import matplotlib or save plots.",
    )
    return parser


def main():
    parser = build_parser()
    args, extra_eval_args = parser.parse_known_args()

    if args.repeats < 1:
        parser.error("--repeats must be >= 1")
    if args.start_step < 0:
        parser.error("--start-step must be >= 0")

    experiments = resolve_experiments(args)
    run_root = Path(args.output_root) if args.output_root else (
        legged_root() / "logs" / "sweep_evaluation" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    run_root.mkdir(parents=True, exist_ok=True)

    rows = []
    skipped = []
    for exp in experiments:
        log_dir = legged_root() / "logs" / exp.proj_name / exp.exptid
        steps = checkpoint_steps(log_dir, args.start_step)
        if not steps:
            skipped.append(
                {
                    "proj_name": exp.proj_name,
                    "exptid": exp.exptid,
                    "reason": f"no model_*.pt checkpoint >= {args.start_step} in {log_dir}",
                }
            )
            continue
        for checkpoint in steps:
            for repeat_idx in range(args.repeats):
                rows.append(run_one_evaluation(args, extra_eval_args, exp, checkpoint, repeat_idx, run_root))

    metrics = list(args.metrics)
    repeat_fields = [
        "proj_name",
        "exptid",
        "label",
        "checkpoint",
        "repeat",
        "json_path",
        "csv_path",
        *metrics,
    ]
    write_csv(run_root / "repeat_metrics.csv", rows, repeat_fields)

    summary_rows = aggregate_rows(rows, metrics)
    summary_fields = ["proj_name", "exptid", "label", "checkpoint", "repeats"]
    for metric in metrics:
        summary_fields.extend([f"{metric}_mean", f"{metric}_std"])
    write_csv(run_root / "summary_metrics.csv", summary_rows, summary_fields)
    write_csv(run_root / "skipped.csv", skipped, ["proj_name", "exptid", "reason"])

    manifest = {
        "output_root": str(run_root),
        "experiments": [exp.__dict__ for exp in experiments],
        "metrics": metrics,
        "repeats": args.repeats,
        "start_step": args.start_step,
        "num_completed_runs": len(rows),
        "num_summary_rows": len(summary_rows),
        "skipped": skipped,
    }
    with (run_root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if summary_rows and not args.no_plot:
        plot_metrics(summary_rows, metrics, run_root)

    print(f"Wrote sweep outputs to {run_root}")
    print(f"Completed evaluation runs: {len(rows)}")
    if skipped:
        print(f"Skipped experiments: {len(skipped)}; see {run_root / 'skipped.csv'}")


if __name__ == "__main__":
    main()
