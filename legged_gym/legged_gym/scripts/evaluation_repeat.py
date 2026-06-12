#!/usr/bin/env python3
"""Run evaluation.py multiple times and report mean metrics."""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run evaluation.py multiple times and print the mean value of each "
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
        help=(
            "Directory used to store per-run outputs. Defaults to a temporary "
            "directory that is removed after completion."
        ),
    )
    parser.add_argument(
        "--aggregate_json",
        type=Path,
        default=None,
        help="Optional path to write the aggregated results as JSON.",
    )
    parser.add_argument(
        "--keep_run_dirs",
        action="store_true",
        help="Keep per-run output directories when using a temporary output root.",
    )
    args, eval_args = parser.parse_known_args()
    if args.num_eval <= 0:
        parser.error("--num_eval must be a positive integer.")
    if any(arg == "--output_dir" or arg.startswith("--output_dir=") for arg in eval_args):
        parser.error(
            "Do not pass --output_dir to this wrapper. Use --output_root instead."
        )
    return args, eval_args


def find_summary_json(run_dir: Path) -> Path:
    json_files = sorted(run_dir.glob("*.json"))
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


def main():
    args, eval_args = parse_args()
    script_dir = Path(__file__).resolve().parent
    evaluation_py = script_dir / "evaluation.py"
    if not evaluation_py.exists():
        raise FileNotFoundError(f"Missing evaluation script: {evaluation_py}")

    temp_root = None
    output_root = args.output_root
    if output_root is None:
        temp_root = Path(tempfile.mkdtemp(prefix="evaluation_repeat_"))
        output_root = temp_root
    output_root.mkdir(parents=True, exist_ok=True)

    run_summaries = []
    try:
        for run_idx in range(args.num_eval):
            run_dir = output_root / f"run_{run_idx + 1:03d}"
            run_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable,
                str(evaluation_py),
                "--output_dir",
                str(run_dir),
                *eval_args,
            ]
            print(f"[Run {run_idx + 1}/{args.num_eval}] {' '.join(cmd)}", flush=True)
            subprocess.run(cmd, cwd=script_dir, check=True)

            summary_path = find_summary_json(run_dir)
            with summary_path.open("r") as f:
                summary = json.load(f)
            run_summaries.append(summary)

        metric_keys = sorted(
            {
                key
                for summary in run_summaries
                for key in collect_numeric_metrics(summary.get("metrics", {})).keys()
            }
        )
        mean_metrics = {}
        for key in metric_keys:
            values = [
                float(summary["metrics"][key])
                for summary in run_summaries
                if key in summary.get("metrics", {})
                and isinstance(summary["metrics"][key], (int, float))
            ]
            if values:
                mean_metrics[key] = sum(values) / len(values)

        aggregate = {
            "num_eval": args.num_eval,
            "mean_metrics": mean_metrics,
            "run_metric_paths": [
                str(find_summary_json(output_root / f"run_{idx + 1:03d}"))
                for idx in range(args.num_eval)
            ],
        }

        print("\nMean metrics:")
        print(json.dumps(mean_metrics, indent=2, sort_keys=True))

        if args.aggregate_json is not None:
            args.aggregate_json.parent.mkdir(parents=True, exist_ok=True)
            with args.aggregate_json.open("w") as f:
                json.dump(aggregate, f, indent=2)
            print(f"Wrote aggregate JSON: {args.aggregate_json}")
    finally:
        if temp_root is not None and not args.keep_run_dirs:
            shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
