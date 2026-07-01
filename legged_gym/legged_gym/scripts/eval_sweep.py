#!/usr/bin/env python3
"""Sweep checkpoints and aggregate repeated evaluation results."""

import argparse
import csv
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from tqdm import tqdm


CSV_FIELDS = [
    "proj_name",
    "exptid",
    "label",
    "checkpoint",
    "repeats",
    "success_rate_mean",
    "success_rate_std",
    "mean_mxd_mean",
    "mean_mxd_std",
    "mean_normalized_waypoints_mean",
    "mean_normalized_waypoints_std",
    "fall_rate_mean",
    "fall_rate_std",
    "stuck_rate_mean",
    "stuck_rate_std",
    "mean_edge_violation_mean",
    "mean_edge_violation_std",
]

METRIC_NAMES = (
    "success_rate",
    "mean_mxd",
    "mean_normalized_waypoints",
    "fall_rate",
    "stuck_rate",
    "mean_edge_violation",
)


def _local_pythonpath_entries(script_dir: Path):
    repo_root = script_dir.parents[3]
    return [
        str(repo_root / "legged_gym"),
        str(repo_root / "rsl_rl"),
    ]


def _prepend_pythonpath(env, entries):
    child_env = env.copy()
    existing = child_env.get("PYTHONPATH")
    path_parts = [entry for entry in entries if entry]
    if existing:
        path_parts.append(existing)
    child_env["PYTHONPATH"] = os.pathsep.join(path_parts)
    return child_env


def _ensure_python_runtime_lib(env):
    child_env = env.copy()
    lib_dir = Path(sys.executable).resolve().parents[1] / "lib"
    if not lib_dir.exists():
        return child_env
    existing = child_env.get("LD_LIBRARY_PATH")
    if existing:
        parts = existing.split(os.pathsep)
        if str(lib_dir) not in parts:
            child_env["LD_LIBRARY_PATH"] = os.pathsep.join([str(lib_dir), *parts])
    else:
        child_env["LD_LIBRARY_PATH"] = str(lib_dir)
    return child_env


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sweep checkpoints for one or more exptids, repeatedly run evaluation, "
            "and write a single JSON plus a summary CSV."
        )
    )
    parser.add_argument("--proj_name", type=str, required=True)
    parser.add_argument(
        "--exptids",
        nargs="*",
        default=None,
        help="Specific exptids to sweep. Defaults to all subdirectories under logs/<proj_name>.",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Label written into the summary CSV. Defaults to the current git branch name.",
    )
    parser.add_argument("--repeats", type=int, required=True)
    parser.add_argument("--checkpoint_start", type=int, default=0)
    parser.add_argument("--checkpoint_step", type=int, default=500)
    parser.add_argument("--checkpoint_end", type=int, default=None)
    parser.add_argument(
        "--logs_root",
        type=Path,
        default=None,
        help="Optional root directory that contains <proj_name>/<exptid>/model_*.pt.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=None,
        help="Directory used to store the sweep JSON and CSV.",
    )
    parser.add_argument("--json_path", type=Path, default=None)
    parser.add_argument("--csv_path", type=Path, default=None)
    parser.add_argument(
        "--auto_gpu",
        action="store_true",
        help="Auto-select the least busy GPU for all evaluations.",
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=None,
        help="Use a specific GPU id. Overrides --auto_gpu.",
    )
    args, eval_args = parser.parse_known_args()
    if args.repeats <= 0:
        parser.error("--repeats must be a positive integer.")
    if args.checkpoint_start < 0:
        parser.error("--checkpoint_start must be non-negative.")
    if args.checkpoint_step <= 0:
        parser.error("--checkpoint_step must be a positive integer.")
    if args.checkpoint_end is not None and args.checkpoint_end < args.checkpoint_start:
        parser.error("--checkpoint_end must be >= --checkpoint_start.")

    forbidden = (
        "--output_dir",
        "--output_root",
        "--aggregate_json",
        "--num_eval",
        "--checkpoint",
        "--exptid",
        "--proj_name",
        "--device",
        "--sim_device",
        "--rl_device",
        "--graphics_device_id",
        "--gpu_id",
        "--auto_gpu",
    )
    for arg in eval_args:
        if any(arg == name or arg.startswith(f"{name}=") for name in forbidden):
            parser.error(
                "Do not pass sweep-controlled args through to evaluation: "
                + ", ".join(forbidden)
            )
    return args, eval_args


def _safe_name(value):
    keep = []
    for ch in str(value):
        keep.append(ch if ch.isalnum() or ch in "._-" else "_")
    sanitized = "".join(keep).strip("_")
    return sanitized or "eval"


def default_logs_root(script_dir: Path, proj_name: str) -> Path:
    return script_dir.parents[1] / "logs" / proj_name


def current_git_branch(script_dir: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=script_dir.parents[2],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown_branch"
    branch = result.stdout.strip()
    return branch or "unknown_branch"


def default_output_root(script_dir: Path, proj_name: str, label: str) -> Path:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return script_dir.parents[1] / "logs" / "evaluation_sweep" / (
        f"{_safe_name(proj_name)}_{_safe_name(label)}_{timestamp}"
    )


def discover_exptids(logs_root: Path, requested_exptids):
    if requested_exptids:
        return list(requested_exptids)
    if not logs_root.exists():
        raise FileNotFoundError(f"Logs root does not exist: {logs_root}")
    exptids = sorted(path.name for path in logs_root.iterdir() if path.is_dir())
    if not exptids:
        raise RuntimeError(f"No exptid directories found under {logs_root}")
    return exptids


def discover_checkpoints(expt_dir: Path, checkpoint_start: int, checkpoint_step: int, checkpoint_end):
    pattern = re.compile(r"model_(\d+)\.pt$")
    available = []
    for path in expt_dir.iterdir():
        match = pattern.fullmatch(path.name)
        if match:
            available.append(int(match.group(1)))
    available = sorted(set(available))
    if not available:
        raise RuntimeError(f"No checkpoint files matching model_*.pt found in {expt_dir}")

    selected = []
    for checkpoint in available:
        if checkpoint < checkpoint_start:
            continue
        if checkpoint_end is not None and checkpoint > checkpoint_end:
            continue
        if (checkpoint - checkpoint_start) % checkpoint_step != 0:
            continue
        selected.append(checkpoint)

    if not selected:
        raise RuntimeError(
            f"No checkpoints matched start={checkpoint_start}, step={checkpoint_step}, "
            f"end={checkpoint_end} in {expt_dir}"
        )
    return selected


def load_json(path: Path):
    with path.open("r") as f:
        return json.load(f)


def load_existing_csv_rows(path: Path):
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            normalized = {}
            for field in CSV_FIELDS:
                normalized[field] = row.get(field)
            rows.append(normalized)
        return rows


def load_existing_json_records(path: Path):
    if not path.exists():
        return []
    payload = load_json(path)
    if not isinstance(payload, dict):
        return []
    records = payload.get("records", [])
    return records if isinstance(records, list) else []


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
    env = _prepend_pythonpath(env, _local_pythonpath_entries(Path(__file__).resolve().parent))
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


def normalize_repeat_aggregate(aggregate, temp_output_root: Path):
    run_summaries = aggregate.get("run_summaries")
    if run_summaries is None:
        run_summaries = []
        for path_str in aggregate.get("run_metric_paths", []):
            path = Path(path_str)
            if not path.is_absolute():
                path = temp_output_root / path
            run_summaries.append(load_json(path))

    metric_stats = aggregate.get("metric_stats")
    if metric_stats is None:
        metric_stats = summarize_metric_values(run_summaries)

    mean_metrics = aggregate.get("mean_metrics")
    if mean_metrics is None:
        mean_metrics = {key: stats["mean"] for key, stats in metric_stats.items()}

    normalized = dict(aggregate)
    normalized["run_summaries"] = run_summaries
    normalized["metric_stats"] = metric_stats
    normalized["mean_metrics"] = mean_metrics
    return normalized


def run_via_evaluation_repeat(
    script_dir: Path,
    proj_name: str,
    exptid: str,
    logs_root: Path,
    checkpoint: int,
    repeats: int,
    gpu_id,
    use_auto_gpu: bool,
    eval_args,
):
    evaluation_repeat_py = script_dir / "evaluation_repeat.py"
    if not evaluation_repeat_py.exists():
        return None
    child_env = _prepend_pythonpath(os.environ.copy(), _local_pythonpath_entries(script_dir))
    child_env = _ensure_python_runtime_lib(child_env)

    with tempfile.TemporaryDirectory(prefix=f"eval_sweep_{_safe_name(exptid)}_{checkpoint}_") as temp_dir:
        temp_root = Path(temp_dir)
        run_root = temp_root / "runs"
        aggregate_json = temp_root / "aggregate.json"
        cmd = [
            sys.executable,
            str(evaluation_repeat_py),
            "--num_eval",
            str(repeats),
            "--output_root",
            str(run_root),
            "--aggregate_json",
            str(aggregate_json),
        ]
        if gpu_id is not None:
            cmd.extend(["--gpu_id", str(gpu_id)])
        elif use_auto_gpu:
            cmd.append("--auto_gpu")
        cmd.extend(
            [
                "--logs_root",
                str(logs_root),
                "--proj_name",
                proj_name,
                "--exptid",
                exptid,
                "--checkpoint",
                str(checkpoint),
                *eval_args,
            ]
        )
        result = subprocess.run(
            cmd,
            cwd=script_dir,
            env=child_env,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"evaluation_repeat.py failed for exptid={exptid} checkpoint={checkpoint}\n"
                f"Command: {' '.join(cmd)}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
        aggregate = normalize_repeat_aggregate(load_json(aggregate_json), run_root)
        aggregate["command"] = cmd
        aggregate["stdout"] = result.stdout
        aggregate["stderr"] = result.stderr
        aggregate["runner"] = "evaluation_repeat.py"
        return aggregate


def run_via_evaluation_py(
    script_dir: Path,
    proj_name: str,
    exptid: str,
    logs_root: Path,
    checkpoint: int,
    repeats: int,
    gpu_id,
    eval_args,
):
    evaluation_py = script_dir / "evaluation.py"
    if not evaluation_py.exists():
        raise FileNotFoundError(f"Missing evaluation script: {evaluation_py}")

    child_env, device_args = evaluation_env(gpu_id)
    run_summaries = []
    per_run_outputs = []

    with tempfile.TemporaryDirectory(prefix=f"eval_sweep_{_safe_name(exptid)}_{checkpoint}_") as temp_dir:
        temp_root = Path(temp_dir)
        for run_idx in range(repeats):
            run_dir = temp_root / f"run_{run_idx + 1:03d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                sys.executable,
                str(evaluation_py),
                "--output_dir",
                str(run_dir),
                *device_args,
                "--logs_root",
                str(logs_root),
                "--proj_name",
                proj_name,
                "--exptid",
                exptid,
                "--checkpoint",
                str(checkpoint),
                *eval_args,
            ]
            result = subprocess.run(
                cmd,
                cwd=script_dir,
                env=child_env,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"evaluation.py failed for exptid={exptid} checkpoint={checkpoint} run={run_idx + 1}\n"
                    f"Command: {' '.join(cmd)}\n"
                    f"stdout:\n{result.stdout}\n"
                    f"stderr:\n{result.stderr}"
                )
            summary = load_json(find_summary_json(run_dir))
            run_summaries.append(summary)
            per_run_outputs.append(
                {
                    "run_index": run_idx + 1,
                    "command": cmd,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            )

    metric_stats = summarize_metric_values(run_summaries)
    mean_metrics = {key: stats["mean"] for key, stats in metric_stats.items()}
    return {
        "num_eval": repeats,
        "selected_gpu": gpu_id,
        "mean_metrics": mean_metrics,
        "metric_stats": metric_stats,
        "run_summaries": run_summaries,
        "per_run_outputs": per_run_outputs,
        "runner": "evaluation.py_fallback",
    }


def run_repeat_eval(
    script_dir: Path,
    proj_name: str,
    exptid: str,
    logs_root: Path,
    checkpoint: int,
    repeats: int,
    gpu_id,
    use_auto_gpu: bool,
    eval_args,
):
    aggregate = run_via_evaluation_repeat(
        script_dir=script_dir,
        proj_name=proj_name,
        exptid=exptid,
        logs_root=logs_root,
        checkpoint=checkpoint,
        repeats=repeats,
        gpu_id=gpu_id,
        use_auto_gpu=use_auto_gpu,
        eval_args=eval_args,
    )
    if aggregate is not None:
        return aggregate
    return run_via_evaluation_py(
        script_dir=script_dir,
        proj_name=proj_name,
        exptid=exptid,
        logs_root=logs_root,
        checkpoint=checkpoint,
        repeats=repeats,
        gpu_id=gpu_id,
        eval_args=eval_args,
    )


def build_csv_row(proj_name: str, exptid: str, label: str, checkpoint: int, repeats: int, aggregate):
    metric_stats = aggregate.get("metric_stats", {})
    row = {
        "proj_name": proj_name,
        "exptid": exptid,
        "label": label,
        "checkpoint": checkpoint,
        "repeats": repeats,
    }
    for metric_name in METRIC_NAMES:
        stats = metric_stats.get(metric_name)
        row[f"{metric_name}_mean"] = None if stats is None else stats.get("mean")
        row[f"{metric_name}_std"] = None if stats is None else stats.get("std")
    return row


def write_outputs(
    json_path: Path,
    csv_path: Path,
    json_payload,
    csv_rows,
):
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w") as f:
        json.dump(json_payload, f, indent=2)

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(csv_rows)


def main():
    args, eval_args = parse_args()
    script_dir = Path(__file__).resolve().parent

    label = args.label or current_git_branch(script_dir)
    logs_root = args.logs_root or default_logs_root(script_dir, args.proj_name)
    output_root = args.output_root or default_output_root(script_dir, args.proj_name, label)
    output_root.mkdir(parents=True, exist_ok=True)
    json_path = args.json_path or (output_root / "sweep_results.json")
    csv_path = args.csv_path or (output_root / "sweep_summary.csv")

    exptids = discover_exptids(logs_root, args.exptids)
    sweep_plan = []
    for exptid in exptids:
        expt_dir = logs_root / exptid
        if not expt_dir.exists():
            raise FileNotFoundError(f"Missing exptid directory: {expt_dir}")
        checkpoints = discover_checkpoints(
            expt_dir,
            args.checkpoint_start,
            args.checkpoint_step,
            args.checkpoint_end,
        )
        sweep_plan.append(
            {
                "exptid": exptid,
                "log_dir": str(expt_dir),
                "checkpoints": checkpoints,
            }
        )

    selected_gpu = args.gpu_id
    if selected_gpu is None and args.auto_gpu:
        selected_gpu = pick_gpu_id()

    existing_records = load_existing_json_records(json_path)
    existing_csv_rows = load_existing_csv_rows(csv_path)
    records = list(existing_records)
    csv_rows = list(existing_csv_rows)
    json_payload = {
        "proj_name": args.proj_name,
        "label": label,
        "repeats": args.repeats,
        "checkpoint_start": args.checkpoint_start,
        "checkpoint_step": args.checkpoint_step,
        "checkpoint_end": args.checkpoint_end,
        "selected_gpu": selected_gpu,
        "logs_root": str(logs_root),
        "eval_args": eval_args,
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "records": records,
    }

    write_outputs(
        json_path=json_path,
        csv_path=csv_path,
        json_payload=json_payload,
        csv_rows=csv_rows,
    )

    total_jobs = sum(len(item["checkpoints"]) for item in sweep_plan)
    progress = tqdm(total=total_jobs, desc="eval sweep", unit="ckpt")
    try:
        for item in sweep_plan:
            exptid = item["exptid"]
            for checkpoint in item["checkpoints"]:
                progress.set_postfix_str(f"{exptid}@{checkpoint}")
                aggregate = run_repeat_eval(
                    script_dir=script_dir,
                    proj_name=args.proj_name,
                    exptid=exptid,
                    logs_root=logs_root,
                    checkpoint=checkpoint,
                    repeats=args.repeats,
                    gpu_id=selected_gpu,
                    use_auto_gpu=args.auto_gpu and args.gpu_id is None,
                    eval_args=eval_args,
                )
                record = {
                    "proj_name": args.proj_name,
                    "exptid": exptid,
                    "label": label,
                    "checkpoint": checkpoint,
                    "repeats": args.repeats,
                    "aggregate": aggregate,
                }
                records.append(record)
                csv_rows.append(
                    build_csv_row(
                        proj_name=args.proj_name,
                        exptid=exptid,
                        label=label,
                        checkpoint=checkpoint,
                        repeats=args.repeats,
                        aggregate=aggregate,
                    )
                )
                write_outputs(
                    json_path=json_path,
                    csv_path=csv_path,
                    json_payload=json_payload,
                    csv_rows=csv_rows,
                )
                progress.update(1)
    finally:
        progress.close()

    print(f"Wrote sweep JSON: {json_path}")
    print(f"Wrote sweep CSV: {csv_path}")


if __name__ == "__main__":
    main()
