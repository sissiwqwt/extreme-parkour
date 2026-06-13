#!/usr/bin/env python3
"""Run evaluation.py multiple times and aggregate metrics.

This wrapper keeps the repeat-evaluation workflow from the aux branches while
defaulting all outputs into legged_gym/logs and optionally auto-selecting a GPU.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run evaluation.py multiple times and report the mean value of each "
            "top-level metric."
        )
    )
    parser.add_argument(
        "--num_eval",
        type=int,
        required=True,
        help="Number of evaluation.py runs to execute.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=None,
        help="Directory used to store per-run outputs and aggregate results.",
    )
    parser.add_argument(
        "--aggregate_json",
        type=Path,
        default=None,
        help="Optional path to write the aggregated results as JSON.",
    )
    parser.add_argument(
        "--auto_gpu",
        action="store_true",
        help="Automatically select the least busy visible GPU for all runs.",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=None,
        help="Use a specific GPU id. Overrides --auto_gpu when both are set.",
    )
    args, eval_args = parser.parse_known_args()
    if args.num_eval <= 0:
        parser.error("--num_eval must be a positive integer.")
    forbidden = ("--output_dir", "--device", "--sim_device", "--rl_device", "--graphics_device_id")
    for arg in eval_args:
        if any(arg == name or arg.startswith(f"{name}=") for name in forbidden):
            parser.error(
                "Do not pass --output_dir/--device/--sim_device/--rl_device/"
                "--graphics_device_id to this wrapper. Use --output_root, "
                "--gpu_id, or --auto_gpu instead."
            )
    return args, eval_args


def default_output_root(eval_args):
    script_dir = Path(__file__).resolve().parent
    logs_dir = script_dir.parents[1] / "logs" / "evaluation_repeat"
    exptid = _extract_flag_value(eval_args, "--exptid") or "policy"
    policy_type = _extract_flag_value(eval_args, "--policy_type") or "auto"
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return logs_dir / f"{_safe_name(exptid)}_{_safe_name(policy_type)}_{timestamp}"


def _extract_flag_value(args, flag_name):
    for idx, arg in enumerate(args):
        if arg == flag_name and idx + 1 < len(args):
            return args[idx + 1]
        if arg.startswith(f"{flag_name}="):
            return arg.split("=", 1)[1]
    return None


def _safe_name(value):
    keep = []
    for ch in str(value):
        keep.append(ch if ch.isalnum() or ch in "._-" else "_")
    sanitized = "".join(keep).strip("_")
    return sanitized or "eval"


def find_summary_json(run_dir: Path) -> Path:
    json_files = sorted(path for path in run_dir.glob("*.json") if path.is_file())
    if len(json_files) != 1:
        raise RuntimeError(
            f"Expected exactly one summary JSON in {run_dir}, found {len(json_files)}."
        )
    return json_files[0]


def collect_numeric_metrics(metrics):
    return {
        key: value
        for key, value in metrics.items()
        if isinstance(value, (int, float))
    }


def summarize_metric_values(run_summaries):
    metric_keys = sorted(
        {
            key
            for summary in run_summaries
            for key in collect_numeric_metrics(summary.get("metrics", {})).keys()
        }
    )
    metric_stats = {}
    for key in metric_keys:
        values = [
            float(summary["metrics"][key])
            for summary in run_summaries
            if key in summary.get("metrics", {})
            and isinstance(summary["metrics"][key], (int, float))
        ]
        if not values:
            continue
        metric_stats[key] = {
            "count": len(values),
            "mean": sum(values) / len(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "values": values,
        }
    return metric_stats


def _query_gpu_table():
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []

    rows = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        try:
            rows.append(
                {
                    "index": int(parts[0]),
                    "memory_used_mb": int(parts[1]),
                    "utilization_gpu": int(parts[2]),
                }
            )
        except ValueError:
            continue
    return rows


def pick_gpu_id():
    rows = _query_gpu_table()
    if not rows:
        return None
    rows.sort(key=lambda row: (row["memory_used_mb"], row["utilization_gpu"], row["index"]))
    return rows[0]["index"]


def evaluation_env(gpu_id):
    env = os.environ.copy()
    if gpu_id is None:
        return env, []
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device_args = [
        "--device",
        "cuda:0",
        "--sim_device",
        "cuda:0",
        "--rl_device",
        "cuda:0",
        "--graphics_device_id",
        "0",
    ]
    return env, device_args


def main():
    args, eval_args = parse_args()
    script_dir = Path(__file__).resolve().parent
    evaluation_py = script_dir / "evaluation.py"
    if not evaluation_py.exists():
        raise FileNotFoundError(f"Missing evaluation script: {evaluation_py}")

    output_root = args.output_root or default_output_root(eval_args)
    output_root.mkdir(parents=True, exist_ok=True)

    selected_gpu = args.gpu_id
    if selected_gpu is None and args.auto_gpu:
        selected_gpu = pick_gpu_id()
        if selected_gpu is None:
            print("[INFO] Auto GPU selection unavailable; using current device settings.")
        else:
            print(f"[INFO] Auto-selected GPU {selected_gpu}.")
    elif selected_gpu is not None:
        print(f"[INFO] Using GPU {selected_gpu}.")

    child_env, device_args = evaluation_env(selected_gpu)

    run_summaries = []
    run_metric_paths = []
    for run_idx in range(args.num_eval):
        run_dir = output_root / f"run_{run_idx + 1:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(evaluation_py),
            "--output_dir",
            str(run_dir),
            *device_args,
            *eval_args,
        ]
        print(f"[Run {run_idx + 1}/{args.num_eval}] {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, cwd=script_dir, env=child_env, check=True)

        summary_path = find_summary_json(run_dir)
        with summary_path.open("r") as f:
            summary = json.load(f)
        run_summaries.append(summary)
        run_metric_paths.append(str(summary_path))

    metric_stats = summarize_metric_values(run_summaries)
    mean_metrics = {
        key: stats["mean"]
        for key, stats in metric_stats.items()
    }

    aggregate = {
        "num_eval": args.num_eval,
        "selected_gpu": selected_gpu,
        "output_root": str(output_root),
        "mean_metrics": mean_metrics,
        "metric_stats": metric_stats,
        "run_metric_paths": run_metric_paths,
        "run_summaries": run_summaries,
    }

    print("\nMean metrics:")
    print(json.dumps(mean_metrics, indent=2, sort_keys=True))

    aggregate_json = args.aggregate_json or (output_root / "aggregate.json")
    aggregate_json.parent.mkdir(parents=True, exist_ok=True)
    with aggregate_json.open("w") as f:
        json.dump(aggregate, f, indent=2)
    print(f"Wrote aggregate JSON: {aggregate_json}")


if __name__ == "__main__":
    main()
