"""Headless rollout for the base policy with true waypoint-yaw observations.

This script is intentionally separate from play_copy.py because play_copy.py is
for vision/depth policies and replaces obs[:, 6:8] with yaw predicted from the
depth encoder. Here obs[:, 6:8] is left unchanged, so the base policy receives
the same waypoint-yaw inputs produced by the environment as in base training.
"""

import argparse
import os
import re
import sys

import faulthandler

import isaacgym
from isaacgym import gymapi

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry
from legged_gym.utils.helpers import (
    class_to_dict,
    parse_sim_params,
    set_seed,
    update_cfg_from_args,
)

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R_scipy


DEFAULT_TERRAIN_DIFFICULTY = 1.0


def _pop_script_argv():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--video_out",
        type=str,
        default=None,
        help="Output directory for mp4 files. Default: legged_gym/demos.",
    )
    parser.add_argument("--video_fps", type=float, default=None)
    parser.add_argument(
        "--record_env",
        type=int,
        default=None,
        help="Which parallel env index to record. Default: record all parallel envs.",
    )
    parser.add_argument(
        "--record_camera",
        type=str,
        default="third_person",
        choices=("third_person", "body"),
        help="Base policy playback supports third_person without --use_camera. "
        "body requires --use_camera and is intended for vision policies.",
    )
    parser.add_argument("--third_dist", type=float, default=2.6)
    parser.add_argument("--third_height", type=float, default=0.75)
    parser.add_argument("--third_lookat_z", type=float, default=0.12)
    parser.add_argument("--camera_smooth", type=float, default=0.14)
    parser.add_argument("--record_width", type=int, default=960)
    parser.add_argument("--record_height", type=int, default=540)
    parser.add_argument(
        "--terrain_difficulty",
        type=float,
        default=DEFAULT_TERRAIN_DIFFICULTY,
        help="Fixed normalized terrain difficulty in [0, 1].",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU simulation/RL. Default uses the repo's normal CUDA settings.",
    )
    ns, rest = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    return ns


def get_load_path_jit(root, checkpoint=-1, model_name_include="model"):
    if checkpoint == -1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: "{0:0>15}".format(m))
        return models[-1]
    return "model_{}.pt".format(checkpoint)


def _safe_filename_part(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "terrain"


def _default_video_dir():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(script_dir, "..", "demos"))


def _format_difficulty_value(value):
    return f"{float(value):.3f}".rstrip("0").rstrip(".")


def _terrain_counts(terrain_dict):
    terrain_counts = {}
    for terrain_name, env_count in terrain_dict.items():
        terrain_name = str(terrain_name).strip()
        if not terrain_name:
            raise ValueError("Terrain names in env_cfg.terrain.terrain_dict must not be empty")
        env_count = int(env_count)
        if env_count <= 0:
            continue
        terrain_counts[terrain_name] = env_count
    if not terrain_counts:
        raise ValueError("env_cfg.terrain.terrain_dict must contain at least one positive env count")
    return terrain_counts


def _generation_terrain_dict(terrain_counts):
    return {terrain_name: 1.0 for terrain_name in terrain_counts.keys()}


def _terrain_names_by_env(terrain_counts):
    terrain_names = []
    for terrain_name, env_count in terrain_counts.items():
        terrain_names.extend([terrain_name] * int(env_count))
    return terrain_names


def _apply_fixed_terrain_config(env_cfg, terrain_counts, terrain_difficulty):
    env_cfg.terrain.terrain_dict = _generation_terrain_dict(terrain_counts)
    env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
    env_cfg.terrain.num_cols = len(env_cfg.terrain.terrain_dict)
    env_cfg.terrain.num_rows = 1
    env_cfg.terrain.fixed_difficulty = float(terrain_difficulty)


def _patch_fixed_terrain_difficulty():
    from legged_gym.utils import terrain_ver2 as terrain_module

    Terrain = terrain_module.Terrain
    if getattr(Terrain.curiculum, "_base_fixed_difficulty_patch", False):
        return

    original_curiculum = Terrain.curiculum

    def curiculum_with_fixed_difficulty(self, random=False, max_difficulty=False):
        fixed_difficulty = getattr(self.cfg, "fixed_difficulty", None)
        if fixed_difficulty is None:
            return original_curiculum(self, random=random, max_difficulty=max_difficulty)

        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                choice = j / self.cfg.num_cols + 0.001
                terrain = self.make_terrain(choice, float(fixed_difficulty))
                self.add_terrain_to_map(terrain, i, j)

    curiculum_with_fixed_difficulty._base_fixed_difficulty_patch = True
    Terrain.curiculum = curiculum_with_fixed_difficulty


def _requested_terrain_difficulty(rec_cfg):
    if rec_cfg.terrain_difficulty < 0.0 or rec_cfg.terrain_difficulty > 1.0:
        raise ValueError(
            f"--terrain_difficulty must be in [0, 1], got {rec_cfg.terrain_difficulty}"
        )
    return rec_cfg.terrain_difficulty


def _refresh_env_terrain_state(env):
    env.env_origins[:] = env.terrain_origins[env.terrain_levels, env.terrain_types]
    env.env_class[:] = env.terrain_class[env.terrain_levels, env.terrain_types]
    temp = env.terrain_goals[env.terrain_levels, env.terrain_types]
    last_col = temp[:, -1].unsqueeze(1)
    env.env_goals[:] = torch.cat(
        (temp, last_col.repeat(1, env.cfg.env.num_future_goal_obs, 1)),
        dim=1,
    )
    env.cur_goals = env._gather_cur_goals()
    env.next_goals = env._gather_cur_goals(future=1)


def _assign_env_terrains(env, terrain_counts):
    env.cfg.terrain.curriculum = False
    terrain_names = list(terrain_counts.keys())
    terrain_name_to_col = {terrain_name: col for col, terrain_name in enumerate(terrain_names)}
    assigned_names = _terrain_names_by_env(terrain_counts)
    if len(assigned_names) != env.num_envs:
        raise ValueError(
            f"Requested {len(assigned_names)} envs from terrain counts but env has {env.num_envs}"
        )
    env.terrain_levels[:] = 0
    env.terrain_types[:] = torch.tensor(
        [terrain_name_to_col[name] for name in assigned_names],
        dtype=torch.long,
        device=env.device,
    )
    _refresh_env_terrain_state(env)
    env_ids = torch.arange(env.num_envs, device=env.device)
    env.reset_idx(env_ids)
    return assigned_names


def _record_env_ids(rec_cfg, num_envs):
    if rec_cfg.record_env is None:
        return list(range(num_envs))
    if rec_cfg.record_env < 0 or rec_cfg.record_env >= num_envs:
        raise ValueError(
            f"--record_env {rec_cfg.record_env} out of range [0, {num_envs - 1}]"
        )
    return [rec_cfg.record_env]


def _video_path_for_env(output_dir, terrain_name, terrain_difficulty, env_id):
    terrain = _safe_filename_part(terrain_name)
    difficulty = _safe_filename_part(_format_difficulty_value(terrain_difficulty))
    return os.path.join(output_dir, f"{terrain}_{difficulty}_{env_id}_base.mp4")


def _output_dir(video_out):
    if video_out is None:
        return _default_video_dir()
    if video_out.lower().endswith(".mp4"):
        parent_dir = os.path.dirname(video_out)
        return parent_dir or "."
    return video_out


def _make_env_with_terrain_override(name, args, env_cfg, terrain_counts, terrain_difficulty):
    task_class = task_registry.get_task_class(name)
    env_cfg, _ = update_cfg_from_args(env_cfg, None, args)
    _apply_fixed_terrain_config(env_cfg, terrain_counts, terrain_difficulty)
    env_cfg.depth.camera_terrain_num_rows = env_cfg.terrain.num_rows
    env_cfg.depth.camera_terrain_num_cols = env_cfg.terrain.num_cols
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


def _rgba_to_bgr(rgba):
    rgb = rgba.reshape(rgba.shape[0], -1, 4)[..., :3]
    return rgb[..., ::-1].copy()


def _create_third_person_camera(env, env_id, width, height, horizontal_fov=None):
    props = gymapi.CameraProperties()
    props.width = width
    props.height = height
    if horizontal_fov is not None:
        props.horizontal_fov = float(horizontal_fov)
    return env.gym.create_camera_sensor(env.envs[env_id], props)


class _SmoothedThirdPersonCam:
    def __init__(self, alpha):
        self.alpha = float(np.clip(alpha, 1e-6, 1.0))
        self._cam = None
        self._tgt = None

    def smooth(self, cam_world, target_world):
        if self._cam is None:
            self._cam = cam_world.astype(np.float64).copy()
            self._tgt = target_world.astype(np.float64).copy()
        else:
            self._cam = (1.0 - self.alpha) * self._cam + self.alpha * cam_world
            self._tgt = (1.0 - self.alpha) * self._tgt + self.alpha * target_world
        return self._cam, self._tgt


def _third_person_pose(env, env_id, dist, height, lookat_z):
    root = env.root_states[env_id, :3].detach().cpu().numpy()
    quat = env.root_states[env_id, 3:7].detach().cpu().numpy()
    rot = R_scipy.from_quat(quat)
    cam_world = root + rot.apply(np.array([-float(dist), 0.0, float(height)]))
    target_world = root.copy()
    target_world[2] += float(lookat_z)
    return cam_world, target_world


def _grab_third_person_rgb(env, env_id, cam_handle, smoother, dist, height, lookat_z):
    cam_raw, tgt_raw = _third_person_pose(env, env_id, dist, height, lookat_z)
    cam_s, tgt_s = smoother.smooth(cam_raw, tgt_raw)
    env.gym.set_camera_location(
        cam_handle,
        env.envs[env_id],
        gymapi.Vec3(float(cam_s[0]), float(cam_s[1]), float(cam_s[2])),
        gymapi.Vec3(float(tgt_s[0]), float(tgt_s[1]), float(tgt_s[2])),
    )
    env.gym.step_graphics(env.sim)
    env.gym.render_all_camera_sensors(env.sim)
    rgba = env.gym.get_camera_image(
        env.sim,
        env.envs[env_id],
        cam_handle,
        gymapi.IMAGE_COLOR,
    )
    return _rgba_to_bgr(rgba)


def _grab_body_camera_rgb(env, env_id):
    env.gym.step_graphics(env.sim)
    env.gym.render_all_camera_sensors(env.sim)
    rgba = env.gym.get_camera_image(
        env.sim,
        env.envs[env_id],
        env.cam_handles[env_id],
        gymapi.IMAGE_COLOR,
    )
    return _rgba_to_bgr(rgba)


def _configure_eval_env(env_cfg):
    env_cfg.env.episode_length_s = 60
    env_cfg.commands.resampling_time = 60
    env_cfg.terrain.height = [0.02, 0.02]
    env_cfg.terrain.terrain_dict = {
        "smooth slope": 0,
        "rough slope up": 0,
        "rough slope down": 0,
        "rough stairs up": 0,
        "rough stairs down": 0,
        "discrete": 0,
        "stepping stones": 0,
        "gaps": 0,
        "smooth flat": 0,
        "pit": 0,
        "wall": 0,
        "platform": 0,
        "large stairs up": 0,
        "large stairs down": 0,
        "parkour": 0,
        "parkour_hurdle": 0,
        "parkour_flat": 0,
        "parkour_step": 0,
        "parkour_gap": 0,
        "alternating_step": 2,
        "beam_gap": 2,
        "asymmetric_gap": 2,
        "parkour_v2": 2,
        "narrow_gap": 2,
        "climbing_wall": 2,
        "demo": 0,
    }
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_difficulty = False
    env_cfg.depth.angle = [0, 1]
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 6
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False


def play_base(args, rec_cfg):
    faulthandler.enable()
    args.headless = True
    args.use_camera = False

    if rec_cfg.cpu:
        print("[INFO] Running in CPU mode (GPU disabled)")
        args.sim_device = "cpu"
        args.rl_device = "cpu"
        args.pipeline = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        print("[INFO] Running with default CUDA settings")

    if rec_cfg.record_camera == "body":
        raise RuntimeError(
            "base_play.py does not enable the body-mounted depth camera. "
            "Use --record_camera third_person for base-policy playback."
        )

    log_pth = "../../logs/{}/{}".format(args.proj_name, args.exptid)
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    if args.nodelay:
        env_cfg.domain_rand.action_delay_view = 0
    _configure_eval_env(env_cfg)
    terrain_counts = _terrain_counts(env_cfg.terrain.terrain_dict)
    terrain_difficulty = _requested_terrain_difficulty(rec_cfg)
    args.num_envs = sum(terrain_counts.values())
    env_cfg.env.num_envs = args.num_envs

    # Isaac Gym can create camera sensors in headless mode only when the env
    # keeps a real graphics device. LeggedRobot does that behind depth.use_camera.
    # We use that switch for rendering only; policy inference below still uses
    # the base actor and the true waypoint yaw already present in obs[:, 6:8].
    env_cfg.depth.use_camera = True
    env_cfg.depth.camera_num_envs = env_cfg.env.num_envs
    env_cfg.terrain.max_error_camera = env_cfg.terrain.max_error
    env_cfg.terrain.horizontal_scale_camera = env_cfg.terrain.horizontal_scale
    args.use_camera = True

    _patch_fixed_terrain_difficulty()
    env, env_cfg = _make_env_with_terrain_override(
        name=args.task,
        args=args,
        env_cfg=env_cfg,
        terrain_counts=terrain_counts,
        terrain_difficulty=terrain_difficulty,
    )
    terrain_names_by_env = _assign_env_terrains(env, terrain_counts)
    obs = env.get_observations()
    args.use_camera = False
    train_cfg.depth_encoder.if_depth = False

    record_env_ids = _record_env_ids(rec_cfg, env.num_envs)
    cam_handles = {}
    smoothers = {}
    for env_id in record_env_ids:
        cam_handles[env_id] = _create_third_person_camera(
            env,
            env_id,
            rec_cfg.record_width,
            rec_cfg.record_height,
            horizontal_fov=getattr(env.cfg.depth, "horizontal_fov", None),
        )
        smoothers[env_id] = _SmoothedThirdPersonCam(rec_cfg.camera_smooth)

    train_cfg.runner.resume = True
    ppo_runner, train_cfg, log_pth = task_registry.make_alg_runner(
        log_root=log_pth,
        env=env,
        name=args.task,
        args=args,
        train_cfg=train_cfg,
        return_log_dir=True,
        init_wandb=False,
    )

    if args.use_jit:
        jit_dir = os.path.join(log_pth, "traced")
        model = get_load_path_jit(root=jit_dir, checkpoint=args.checkpoint)
        path = os.path.join(jit_dir, model)
        print("Loading jit for policy:", path)
        policy_jit = torch.jit.load(path, map_location=env.device)
        policy = None
    else:
        policy = ppo_runner.get_inference_policy(device=env.device)
        policy_jit = None

    output_dir = _output_dir(rec_cfg.video_out)
    os.makedirs(output_dir, exist_ok=True)

    fps = rec_cfg.video_fps if rec_cfg.video_fps is not None else float(round(1.0 / env.dt, 3))
    writers = {}
    video_paths = {}
    first_frames = {}
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    def _record_frame(env_id):
        return _grab_third_person_rgb(
            env,
            env_id,
            cam_handles[env_id],
            smoothers[env_id],
            rec_cfg.third_dist,
            rec_cfg.third_height,
            rec_cfg.third_lookat_z,
        )

    for env_id in record_env_ids:
        video_path = _video_path_for_env(
            output_dir,
            terrain_names_by_env[env_id],
            terrain_difficulty,
            env_id,
        )
        first = _record_frame(env_id)
        h, w = first.shape[:2]
        writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
        if not writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter for {video_path}")
        writer.write(first)
        writers[env_id] = writer
        video_paths[env_id] = video_path
        first_frames[env_id] = (w, h)

    print(f"Recording base policy envs {record_env_ids} -> {output_dir} @ {fps} FPS")
    for env_id in record_env_ids:
        w, h = first_frames[env_id]
        print(f"  env {env_id} -> {video_paths[env_id]} ({w}x{h})")
    print(f"Using fixed terrain difficulty {terrain_difficulty:.2f}.")
    print(f"Running {env.num_envs} envs across {len(terrain_counts)} terrain types.")
    for terrain_name, env_count in terrain_counts.items():
        print(f"  {terrain_name}: {env_count} envs")
    print("Using true waypoint yaw from obs[:, 6:8]; no depth encoder is used.")

    try:
        for _ in range(int(env.max_episode_length)):
            if policy_jit is not None:
                obs_jit = torch.cat(
                    (
                        obs.detach()[:, : env_cfg.env.n_proprio + env_cfg.env.n_priv],
                        obs.detach()[:, -env_cfg.env.history_len * env_cfg.env.n_proprio :],
                    ),
                    dim=1,
                )
                actions = policy_jit(obs_jit)
            else:
                actions = policy(
                    obs.detach(),
                    hist_encoding=True,
                    scandots_latent=None,
                )

            obs, _, _, _, _ = env.step(actions.detach())
            for env_id in record_env_ids:
                writers[env_id].write(_record_frame(env_id))
    finally:
        for writer in writers.values():
            writer.release()
        print("Video writers closed.")


if __name__ == "__main__":
    rec = _pop_script_argv()
    cli = get_args()
    play_base(cli, rec)
