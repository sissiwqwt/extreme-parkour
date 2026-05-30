"""Episode-level policy evaluation for the parkour experiments.

This script follows experiment_design.md and writes one CSV row per finished
episode, one terrain-level summary CSV row per evaluated terrain, and a JSON
summary. It supports the base/teacher policy and the depth-camera student
policy used by the existing play.py scripts.
"""

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict

import faulthandler

import isaacgym  # noqa: F401
import numpy as np
import torch

from legged_gym import LEGGED_GYM_ROOT_DIR
from legged_gym.envs import *  # noqa: F401,F403
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import (
    class_to_dict,
    parse_bool,
    parse_sim_params,
    set_seed,
    update_cfg_from_args,
)


CSV_FIELDS = [
    "policy_id",
    "checkpoint",
    "seed",
    "env_id",
    "terrain_name",
    "terrain_type",
    "terrain_level",
    "difficulty",
    "start_x",
    "final_x",
    "mxd",
    "num_waypoints",
    "normalized_waypoints",
    "episode_length",
    "success",
    "fall",
    "stuck",
    "edge_violation",
    "mean_heading_loss",
    "failure_reason",
]

TERRAIN_CSV_FIELDS = [
    "terrain_name",
    "episodes",
    "success_rate",
    "fall_rate",
    "stuck_rate",
    "mean_mxd",
    "mean_normalized_waypoints",
    "mean_episode_length",
    "mean_edge_violation",
    "mean_heading_loss",
]

DIFFICULTY_CSV_FIELDS = [
    "difficulty",
    "episodes",
    "success_rate",
    "fall_rate",
    "stuck_rate",
    "mean_mxd",
    "mean_normalized_waypoints",
    "mean_episode_length",
    "mean_edge_violation",
    "mean_heading_loss",
]

ALL_DIFFICULTIES = tuple(round(0.1 * i, 1) for i in range(1, 11))

EFFECTIVE_TERRAINS = {
    "parkour": 1.0,
    "parkour_hurdle": 1.0,
    "parkour_flat": 1.0,
    "parkour_step": 1.0,
    "parkour_gap": 1.0,
    "alternating_step": 1.0,
    "beam_gap": 1.0,
    "asymmetric_gap": 1.0,
    "parkour_v2": 1.0,
    "narrow_gap": 1.0,
    "climbing_wall": 1.0,
}

BASELINE_TERRAINS = {
    "parkour": 0.2,
    "parkour_hurdle": 0.2,
    "parkour_flat": 0.2,
    "parkour_step": 0.2,
    "parkour_gap": 0.2,
}

CUSTOM_TERRAIN_NAMES = (
    "alternating_step",
    "beam_gap",
    "asymmetric_gap",
    "parkour_v2",
    "narrow_gap",
    "climbing_wall",
)

DESIGN_NEW_TERRAINS = {
    "alternating_step": 0.2,
    "beam_gap": 0.2,
    "asymmetric_gap": 0.2,
    "climbing_wall": 0.2,
    "parkour_v2": 0.2,
}

IMPLEMENTED_TERRAIN_NAMES = {
    "smooth slope",
    "rough slope up",
    "rough slope down",
    "rough stairs up",
    "rough stairs down",
    "discrete",
    "stepping stones",
    "gaps",
    "smooth flat",
    "pit",
    "wall",
    "platform",
    "large stairs up",
    "large stairs down",
    "parkour",
    "parkour_hurdle",
    "parkour_flat",
    "parkour_step",
    "parkour_gap",
    *CUSTOM_TERRAIN_NAMES,
    "demo",
}

TERRAIN_SETS = {
    "effective": EFFECTIVE_TERRAINS,
    "baseline": BASELINE_TERRAINS,
    "design_new": DESIGN_NEW_TERRAINS,
}

TERRAIN_ALIASES = {
    "slanted_hurdle": "climbing_wall",
    "biased_gap": "asymmetric_gap",
    "bean_gap": "beam_gap",
}


def _pop_eval_argv():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--policy_id", type=str, default=None)
    parser.add_argument("--eval_episodes", type=int, default=256)
    parser.add_argument("--eval_max_steps", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--terrain_set",
        choices=("effective", "baseline", "design_new", "new", "all"),
        default="effective",
        help=(
            "effective uses the default 11 valid terrains; baseline uses original "
            "parkour terrains; design_new maps the design terrains onto the "
            "currently implemented terrain.py names; new uses all current custom "
            "terrain.py terrains with env_cfg weights."
        ),
    )
    parser.add_argument(
        "--terrain_names",
        type=str,
        default=None,
        help=(
            "Comma-separated terrain names; must match "
            "env_cfg.terrain.terrain_dict keys."
        ),
    )
    parser.add_argument("--episode_length_s", type=float, default=60.0)
    parser.add_argument("--success_threshold", type=float, default=1.0)
    parser.add_argument("--stuck_window_s", type=float, default=2.0)
    parser.add_argument("--stuck_threshold_m", type=float, default=0.1)
    parser.add_argument("--push_robots", type=parse_bool, default=False)
    parser.add_argument("--randomize_friction", type=parse_bool, default=True)
    parser.add_argument("--add_noise", type=parse_bool, default=True)
    parser.add_argument("--max_difficulty", type=parse_bool, default=True)
    parser.add_argument(
        "--difficulty_mode",
        choices=("single", "all-difficulty", "all-cifficulty"),
        default="single",
        help=(
            "single keeps the existing --max_difficulty behavior; "
            "all-difficulty evaluates fixed terrain difficulties 0.1 through 1.0 "
            "in one run. all-cifficulty is accepted as a backward-compatible alias."
        ),
    )
    parser.add_argument(
        "--policy_type",
        choices=("auto", "base", "depth"),
        default="auto",
        help="auto uses --use_camera to choose depth vs base inference.",
    )
    ns, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    return ns


def _safe_filename(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "eval"


def _checkpoint_label(args):
    return "latest" if args.checkpoint is None or args.checkpoint == -1 else str(args.checkpoint)


def _zeroed_terrain_dict(env_cfg):
    return {name: 0.0 for name in env_cfg.terrain.terrain_dict.keys()}


def _canonical_terrain_name(name, env_cfg):
    requested = name.strip()
    name = TERRAIN_ALIASES.get(requested, requested)
    if name not in env_cfg.terrain.terrain_dict:
        raise ValueError(
            f"Unknown terrain '{name}'. "
            f"Available: {sorted(env_cfg.terrain.terrain_dict.keys())}"
        )
    if name not in IMPLEMENTED_TERRAIN_NAMES:
        raise ValueError(
            f"Terrain '{name}' is present in env_cfg.terrain.terrain_dict but "
            "is not implemented as a Terrain.make_terrain branch."
        )
    if requested != name:
        print(f"Using terrain alias '{requested}' -> '{name}'.")
    return name


def _normalize_weights(terrain_weights):
    total = sum(float(weight) for weight in terrain_weights.values())
    if total <= 0:
        raise ValueError("Evaluation terrain weights must sum to a positive value.")
    return {name: float(weight) / total for name, weight in terrain_weights.items()}


def _current_custom_terrain_weights(env_cfg):
    weights = {}
    for name in CUSTOM_TERRAIN_NAMES:
        if name in env_cfg.terrain.terrain_dict:
            weights[name] = float(env_cfg.terrain.terrain_dict[name])
    if any(weight > 0.0 for weight in weights.values()):
        return {name: weight for name, weight in weights.items() if weight > 0.0}
    return {name: 1.0 for name in weights}


def _selected_terrain_weights(terrain_set, env_cfg):
    if terrain_set == "new":
        return _current_custom_terrain_weights(env_cfg)
    if terrain_set == "all":
        return dict(EFFECTIVE_TERRAINS)
    return dict(TERRAIN_SETS[terrain_set])


def _all_difficulty_mode(eval_cfg):
    return eval_cfg.difficulty_mode in ("all-difficulty", "all-cifficulty")


def _difficulty_from_level(level, eval_cfg):
    if not _all_difficulty_mode(eval_cfg):
        return ""
    if 0 <= level < len(ALL_DIFFICULTIES):
        return ALL_DIFFICULTIES[level]
    return ""


def _install_all_difficulty_curriculum(env_cfg):
    env_cfg.terrain.eval_all_difficulties = list(ALL_DIFFICULTIES)

    from legged_gym.utils import terrain as terrain_module

    if getattr(terrain_module.Terrain.curiculum, "_eval_all_difficulty_patch", False):
        return

    original_curiculum = terrain_module.Terrain.curiculum

    def eval_curiculum(self, random=False, max_difficulty=False):
        difficulties = getattr(self.cfg, "eval_all_difficulties", None)
        if difficulties is None:
            return original_curiculum(self, random=random, max_difficulty=max_difficulty)

        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                choice = j / self.cfg.num_cols + 0.001
                terrain = self.make_terrain(choice, float(difficulties[i]))
                self.add_terrain_to_map(terrain, i, j)

    eval_curiculum._eval_all_difficulty_patch = True
    terrain_module.Terrain.curiculum = eval_curiculum


def _terrain_name_map(env_cfg):
    proportions = np.asarray(env_cfg.terrain.terrain_proportions, dtype=np.float64)
    if proportions.sum() <= 0:
        return {}

    cumulative = np.cumsum(proportions / proportions.sum())
    result = {}
    keys = list(env_cfg.terrain.terrain_dict.keys())
    for col in range(env_cfg.terrain.num_cols):
        choice = col / env_cfg.terrain.num_cols + 0.001
        terrain_idx = int(np.searchsorted(cumulative, choice, side="right"))
        terrain_idx = min(terrain_idx, len(keys) - 1)
        result[col] = keys[terrain_idx]
    return result


def _configure_eval_env(env_cfg, eval_cfg):
    env_cfg.env.episode_length_s = eval_cfg.episode_length_s
    env_cfg.commands.resampling_time = eval_cfg.episode_length_s
    if _all_difficulty_mode(eval_cfg):
        env_cfg.terrain.num_rows = len(ALL_DIFFICULTIES)
        _install_all_difficulty_curriculum(env_cfg)
    else:
        env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = max(5, env_cfg.terrain.num_cols)
    env_cfg.terrain.height = [0.02, 0.02]
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_difficulty = eval_cfg.max_difficulty
    env_cfg.depth.angle = [0, 1]
    env_cfg.noise.add_noise = eval_cfg.add_noise
    env_cfg.domain_rand.randomize_friction = eval_cfg.randomize_friction
    env_cfg.domain_rand.push_robots = eval_cfg.push_robots
    env_cfg.domain_rand.push_interval_s = 6
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False

    terrain_dict = _zeroed_terrain_dict(env_cfg)
    if eval_cfg.terrain_names:
        names = [n for n in eval_cfg.terrain_names.split(",") if n.strip()]
        if not names:
            raise ValueError("--terrain_names was provided but no terrain names were parsed.")
        weight = 1.0 / len(names)
        for name in names:
            terrain_dict[_canonical_terrain_name(name, env_cfg)] = weight
    else:
        selected = _normalize_weights(
            _selected_terrain_weights(eval_cfg.terrain_set, env_cfg)
        )
        for name, weight in selected.items():
            terrain_dict[_canonical_terrain_name(name, env_cfg)] = weight

    env_cfg.terrain.terrain_dict = terrain_dict
    env_cfg.terrain.terrain_proportions = list(terrain_dict.values())
    active_terrain_count = sum(weight > 0.0 for weight in terrain_dict.values())
    env_cfg.terrain.num_cols = max(env_cfg.terrain.num_cols, active_terrain_count)


def _tensor_to_list(tensor):
    return tensor.detach().cpu().numpy().tolist()


def _failure_reason(success, fall, stuck, timeout):
    if success:
        return "success"
    if fall:
        return "fall"
    if stuck:
        return "stuck"
    if timeout:
        return "timeout"
    return "reset"


def _split_depth_heading(depth_output, depth_encoder_cfg):
    heading_dim = (
        depth_encoder_cfg.get("heading_dim", 4)
        if depth_encoder_cfg.get("enable_heading_model", False)
        else 2
    )
    return depth_output[:, :-heading_dim], depth_output[:, -heading_dim:]


def _heading_label(obs, depth_encoder_cfg):
    if not depth_encoder_cfg.get("enable_heading_model", False):
        return obs[:, 6:8]
    delta_yaw = obs[:, 6]
    delta_next_yaw = obs[:, 7]
    return torch.stack(
        (
            torch.cos(delta_yaw),
            torch.sin(delta_yaw),
            torch.cos(delta_next_yaw),
            torch.sin(delta_next_yaw),
        ),
        dim=-1,
    )


def _heading_to_actor_yaw(heading_pred, depth_encoder_cfg):
    if not depth_encoder_cfg.get("enable_heading_model", False):
        return depth_encoder_cfg.get("heading_output_scale", 1.5) * heading_pred
    delta_yaw = torch.atan2(heading_pred[:, 1], heading_pred[:, 0])
    delta_next_yaw = torch.atan2(heading_pred[:, 3], heading_pred[:, 2])
    return torch.stack((delta_yaw, delta_next_yaw), dim=-1)


def _install_terminal_snapshot(env):
    env._eval_terminal_valid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._eval_terminal_final_x = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    env._eval_terminal_goal_idx = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env._eval_terminal_episode_length = torch.zeros(
        env.num_envs, dtype=torch.long, device=env.device
    )
    env._eval_terminal_level = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env._eval_terminal_col = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env._eval_terminal_class = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    env._eval_terminal_timeout = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    original_reset_idx = env.reset_idx

    def wrapped_reset_idx(env_ids):
        if len(env_ids) > 0:
            ids = env_ids.detach().clone()
            env._eval_terminal_valid[ids] = True
            env._eval_terminal_final_x[ids] = env.root_states[ids, 0]
            env._eval_terminal_goal_idx[ids] = env.cur_goal_idx[ids]
            env._eval_terminal_episode_length[ids] = env.episode_length_buf[ids]
            env._eval_terminal_level[ids] = env.terrain_levels[ids]
            env._eval_terminal_col[ids] = env.terrain_types[ids]
            env._eval_terminal_class[ids] = env.env_class[ids]
            env._eval_terminal_timeout[ids] = env.time_out_buf[ids]
        return original_reset_idx(env_ids)

    env.reset_idx = wrapped_reset_idx


def _make_eval_env(name, args, env_cfg, terrain_dict):
    task_class = task_registry.get_task_class(name)
    env_cfg, _ = update_cfg_from_args(env_cfg, None, args)
    env_cfg.terrain.terrain_dict = dict(terrain_dict)
    env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
    set_seed(env_cfg.seed)
    sim_params = {"sim": class_to_dict(env_cfg.sim)}
    sim_params = parse_sim_params(args, sim_params)
    env = task_class(
        cfg=env_cfg,
        sim_params=sim_params,
        physics_engine=args.physics_engine,
        sim_device=args.sim_device,
        headless=args.headless,
    )
    return env, env_cfg


def _run_policy_step(args, env, obs, infos, ppo_runner, policy, depth_encoder, policy_type):
    depth_latent = None
    heading_loss = None
    if policy_type == "depth":
        if infos.get("depth") is not None:
            obs_student = obs[:, : env.cfg.env.n_proprio].clone()
            obs_student[:, 6:8] = 0
            with torch.no_grad():
                depth_latent_and_heading = depth_encoder(infos["depth"], obs_student)
            depth_encoder_cfg = ppo_runner.alg.depth_encoder_paras
            depth_latent, heading_pred = _split_depth_heading(
                depth_latent_and_heading, depth_encoder_cfg
            )
            heading_target = _heading_label(obs, depth_encoder_cfg)
            heading_loss = (heading_target - heading_pred).norm(p=2, dim=1)
            obs[:, 6:8] = _heading_to_actor_yaw(heading_pred, depth_encoder_cfg)

    with torch.no_grad():
        if policy_type == "depth" and hasattr(ppo_runner.alg, "depth_actor"):
            actions = ppo_runner.alg.depth_actor(
                obs.detach(), hist_encoding=True, scandots_latent=depth_latent
            )
        else:
            actions = policy(obs.detach(), hist_encoding=True, scandots_latent=depth_latent)
    return actions, heading_loss


def evaluate(args, eval_cfg):
    faulthandler.enable()
    args.headless = True

    policy_type = eval_cfg.policy_type
    if policy_type == "auto":
        policy_type = "depth" if args.use_camera else "base"
    if policy_type == "depth":
        args.use_camera = True
    elif policy_type == "base":
        args.use_camera = False

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    if args.num_envs is None:
        args.num_envs = 256 if policy_type == "base" else 128
    env_cfg.env.num_envs = args.num_envs
    _configure_eval_env(env_cfg, eval_cfg)
    terrain_dict = dict(env_cfg.terrain.terrain_dict)
    if policy_type == "depth":
        env_cfg.depth.camera_num_envs = env_cfg.env.num_envs
        env_cfg.depth.camera_terrain_num_rows = env_cfg.terrain.num_rows
        env_cfg.depth.camera_terrain_num_cols = env_cfg.terrain.num_cols
        env_cfg.terrain.max_error_camera = env_cfg.terrain.max_error
        env_cfg.terrain.horizontal_scale_camera = env_cfg.terrain.horizontal_scale

    log_pth = os.path.join(LEGGED_GYM_ROOT_DIR, "logs", args.proj_name, args.exptid)
    env, env_cfg = _make_eval_env(args.task, args, env_cfg, terrain_dict)
    _install_terminal_snapshot(env)
    obs = env.get_observations()

    train_cfg.runner.resume = True
    ppo_runner, train_cfg, resolved_log_pth = task_registry.make_alg_runner(
        log_root=log_pth,
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
        return_log_dir=True,
        init_wandb=False,
    )
    policy = ppo_runner.get_inference_policy(device=env.device)
    depth_encoder = (
        ppo_runner.get_depth_encoder_inference_policy(device=env.device)
        if policy_type == "depth"
        else None
    )

    policy_id = eval_cfg.policy_id or args.exptid or "policy"
    checkpoint = _checkpoint_label(args)
    output_dir = eval_cfg.output_dir or os.path.join(
        LEGGED_GYM_ROOT_DIR, "logs", "evaluation"
    )
    os.makedirs(output_dir, exist_ok=True)
    basename_parts = [policy_type, eval_cfg.terrain_set]
    if _all_difficulty_mode(eval_cfg):
        basename_parts.append("all-difficulty")
    basename_parts.append(checkpoint)
    basename = "_".join(_safe_filename(part) for part in basename_parts)
    csv_path = os.path.join(output_dir, basename + ".csv")
    terrain_csv_path = os.path.join(output_dir, basename + "_by_terrain.csv")
    difficulty_csv_path = (
        os.path.join(output_dir, basename + "_by_difficulty.csv")
        if _all_difficulty_mode(eval_cfg)
        else None
    )
    json_path = os.path.join(output_dir, basename + ".json")

    terrain_lookup = _terrain_name_map(env_cfg)
    num_goals = max(1, int(env_cfg.terrain.num_goals))
    success_threshold = float(eval_cfg.success_threshold)
    stuck_window_steps = max(1, int(round(eval_cfg.stuck_window_s / env.dt)))
    max_steps = eval_cfg.eval_max_steps or int(env.max_episode_length * 20)

    start_x = env.root_states[:, 0].clone()
    last_window_x = env.root_states[:, 0].clone()
    edge_sum = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    step_sum = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    heading_loss_sum = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    heading_loss_count = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)

    rows = []
    infos = {
        "depth": env.depth_buffer.clone().to(ppo_runner.device)[:, -1]
        if policy_type == "depth" and ppo_runner.if_depth
        else None
    }

    print(
        f"Evaluating {policy_id} ({policy_type}) for {eval_cfg.eval_episodes} episodes "
        f"with {env.num_envs} envs -> {csv_path}"
    )

    for step in range(max_steps):
        actions, heading_loss = _run_policy_step(
            args, env, obs, infos, ppo_runner, policy, depth_encoder, policy_type
        )
        if heading_loss is not None:
            heading_loss_sum += heading_loss.detach()
            heading_loss_count += 1.0

        prev_start_x = start_x.clone()
        prev_last_window_x = last_window_x.clone()

        obs, _, _, dones, infos = env.step(actions.detach())

        feet_at_edge = getattr(env, "feet_at_edge", None)
        if feet_at_edge is not None:
            edge_sum += feet_at_edge.sum(dim=1).float()
        step_sum += 1.0

        if (step + 1) % stuck_window_steps == 0:
            last_window_x = env.root_states[:, 0].clone()

        done_ids = (dones > 0).nonzero(as_tuple=False).flatten()
        if done_ids.numel() > 0:
            for env_id in _tensor_to_list(done_ids):
                if not bool(env._eval_terminal_valid[env_id].item()):
                    continue
                final_x = float(env._eval_terminal_final_x[env_id].item())
                sx = float(prev_start_x[env_id].item())
                mxd = final_x - sx
                num_waypoints = int(env._eval_terminal_goal_idx[env_id].item())
                normalized = min(1.0, num_waypoints / float(num_goals))
                success = normalized >= success_threshold
                timeout = bool(env._eval_terminal_timeout[env_id].item())
                fall = bool((dones[env_id] > 0).item() and not timeout and not success)
                recent_progress = final_x - float(prev_last_window_x[env_id].item())
                stuck = (not success) and (not fall) and recent_progress < eval_cfg.stuck_threshold_m
                edge_violation = float(
                    edge_sum[env_id].item() / max(step_sum[env_id].item(), 1.0)
                )
                mean_heading_loss = float(
                    heading_loss_sum[env_id].item()
                    / max(heading_loss_count[env_id].item(), 1.0)
                )
                terrain_col = int(env._eval_terminal_col[env_id].item())
                terrain_level = int(env._eval_terminal_level[env_id].item())
                terrain_name = terrain_lookup.get(terrain_col, f"col_{terrain_col}")

                rows.append(
                    {
                        "policy_id": policy_id,
                        "checkpoint": checkpoint,
                        "seed": env_cfg.seed,
                        "env_id": int(env_id),
                        "terrain_name": terrain_name,
                        "terrain_type": int(env._eval_terminal_class[env_id].item()),
                        "terrain_level": terrain_level,
                        "difficulty": _difficulty_from_level(terrain_level, eval_cfg),
                        "start_x": sx,
                        "final_x": final_x,
                        "mxd": mxd,
                        "num_waypoints": num_waypoints,
                        "normalized_waypoints": normalized,
                        "episode_length": int(
                            env._eval_terminal_episode_length[env_id].item()
                        ),
                        "success": int(success),
                        "fall": int(fall),
                        "stuck": int(stuck),
                        "edge_violation": edge_violation,
                        "mean_heading_loss": mean_heading_loss,
                        "failure_reason": _failure_reason(success, fall, stuck, timeout),
                    }
                )

            start_x[done_ids] = env.root_states[done_ids, 0]
            last_window_x[done_ids] = env.root_states[done_ids, 0]
            edge_sum[done_ids] = 0.0
            step_sum[done_ids] = 0.0
            heading_loss_sum[done_ids] = 0.0
            heading_loss_count[done_ids] = 0.0
            env._eval_terminal_valid[done_ids] = False

        if len(rows) >= eval_cfg.eval_episodes:
            rows = rows[: eval_cfg.eval_episodes]
            break

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    summary = _summarize(
        rows,
        csv_path,
        terrain_csv_path,
        difficulty_csv_path,
        resolved_log_pth,
        env_cfg,
        eval_cfg,
        policy_type,
    )
    with open(terrain_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TERRAIN_CSV_FIELDS)
        writer.writeheader()
        writer.writerows(_terrain_summary_rows(summary["terrain_metrics"]))
    if difficulty_csv_path is not None:
        with open(difficulty_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=DIFFICULTY_CSV_FIELDS)
            writer.writeheader()
            writer.writerows(_difficulty_summary_rows(summary["difficulty_metrics"]))
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary["metrics"], indent=2))
    print(f"Wrote CSV: {csv_path}")
    print(f"Wrote terrain CSV: {terrain_csv_path}")
    if difficulty_csv_path is not None:
        print(f"Wrote difficulty CSV: {difficulty_csv_path}")
    print(f"Wrote JSON: {json_path}")


def _mean(group, key):
    return float(np.mean([row[key] for row in group])) if group else 0.0


def _make_terrain_metrics(group):
    return {
        "episodes": len(group),
        "success_rate": _mean(group, "success"),
        "fall_rate": _mean(group, "fall"),
        "stuck_rate": _mean(group, "stuck"),
        "mean_mxd": _mean(group, "mxd"),
        "mean_normalized_waypoints": _mean(group, "normalized_waypoints"),
        "mean_episode_length": _mean(group, "episode_length"),
        "mean_edge_violation": _mean(group, "edge_violation"),
        "mean_heading_loss": _mean(group, "mean_heading_loss"),
    }


def _terrain_summary_rows(terrain_metrics):
    return [
        {"terrain_name": name, **metrics}
        for name, metrics in sorted(terrain_metrics.items())
    ]


def _difficulty_summary_rows(difficulty_metrics):
    return [
        {"difficulty": difficulty, **metrics}
        for difficulty, metrics in sorted(
            difficulty_metrics.items(), key=lambda item: float(item[0])
        )
    ]


def _summarize(
    rows,
    csv_path,
    terrain_csv_path,
    difficulty_csv_path,
    resolved_log_pth,
    env_cfg,
    eval_cfg,
    policy_type,
):
    metrics = {
        "episodes": len(rows),
        "success_rate": float(np.mean([r["success"] for r in rows])) if rows else 0.0,
        "fall_rate": float(np.mean([r["fall"] for r in rows])) if rows else 0.0,
        "stuck_rate": float(np.mean([r["stuck"] for r in rows])) if rows else 0.0,
        "mean_mxd": float(np.mean([r["mxd"] for r in rows])) if rows else 0.0,
        "mean_normalized_waypoints": float(np.mean([r["normalized_waypoints"] for r in rows]))
        if rows
        else 0.0,
        "mean_edge_violation": float(np.mean([r["edge_violation"] for r in rows]))
        if rows
        else 0.0,
        "mean_heading_loss": float(np.mean([r["mean_heading_loss"] for r in rows]))
        if rows
        else 0.0,
    }

    by_terrain = defaultdict(list)
    for row in rows:
        by_terrain[row["terrain_name"]].append(row)
    terrain_metrics = {}
    active_terrain_names = [
        name for name, weight in env_cfg.terrain.terrain_dict.items() if weight > 0.0
    ]
    for name in active_terrain_names:
        terrain_metrics[name] = _make_terrain_metrics(by_terrain.get(name, []))
    for name, group in by_terrain.items():
        if name not in terrain_metrics:
            terrain_metrics[name] = _make_terrain_metrics(group)

    difficulty_metrics = {}
    if _all_difficulty_mode(eval_cfg):
        by_difficulty = defaultdict(list)
        for row in rows:
            by_difficulty[str(row["difficulty"])].append(row)
        for difficulty in ALL_DIFFICULTIES:
            difficulty_metrics[str(difficulty)] = _make_terrain_metrics(
                by_difficulty.get(str(difficulty), [])
            )

    return {
        "csv_path": csv_path,
        "terrain_csv_path": terrain_csv_path,
        "difficulty_csv_path": difficulty_csv_path,
        "resolved_log_path": resolved_log_pth,
        "policy_type": policy_type,
        "terrain_dict": env_cfg.terrain.terrain_dict,
        "difficulty_mode": eval_cfg.difficulty_mode,
        "all_difficulties": list(ALL_DIFFICULTIES) if _all_difficulty_mode(eval_cfg) else [],
        "success_threshold": eval_cfg.success_threshold,
        "stuck_window_s": eval_cfg.stuck_window_s,
        "stuck_threshold_m": eval_cfg.stuck_threshold_m,
        "metrics": metrics,
        "terrain_metrics": terrain_metrics,
        "difficulty_metrics": difficulty_metrics,
    }


if __name__ == "__main__":
    eval_cli = _pop_eval_argv()
    cli = get_args()
    evaluate(cli, eval_cli)
