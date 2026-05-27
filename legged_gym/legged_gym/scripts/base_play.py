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


def _pop_script_argv():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--video_out",
        type=str,
        default=None,
        help="Output video file path (.mp4 recommended). Default: legged_gym/demos/<terrain>_base_<ckpt>.mp4.",
    )
    parser.add_argument("--video_fps", type=float, default=None)
    parser.add_argument("--record_env", type=int, default=0)
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
    difficulty_group = parser.add_mutually_exclusive_group()
    difficulty_group.add_argument(
        "--terrain_level",
        type=int,
        default=None,
        help="Fixed terrain row to play on. With the default 5 rows, valid values are 0..4.",
    )
    difficulty_group.add_argument(
        "--terrain_difficulty",
        type=float,
        default=None,
        help="Fixed normalized terrain difficulty in [0, 1]. Mapped to the nearest terrain row.",
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
    model, _ = _get_model_and_checkpoint(root, checkpoint, model_name_include)
    return model


def _get_model_and_checkpoint(root, checkpoint=-1, model_name_include="model"):
    if checkpoint == -1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: "{0:0>15}".format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint)
    return model, _checkpoint_from_model_name(model, checkpoint)


def _checkpoint_from_model_name(model, fallback):
    match = re.search(r"model_(\d+)\.pt$", model)
    if match is not None:
        return match.group(1)
    if fallback != -1:
        return str(fallback)
    match = re.search(r"(\d+)", model)
    if match is not None:
        return match.group(1)
    return "latest"


def _active_terrain_name(terrain_dict):
    active = [(name, weight) for name, weight in terrain_dict.items() if weight > 0]
    if not active:
        return "terrain"
    return max(active, key=lambda item: item[1])[0]


def _safe_filename_part(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_") or "terrain"


def _default_video_path(env_cfg, checkpoint):
    terrain = _safe_filename_part(_active_terrain_name(env_cfg.terrain.terrain_dict))
    script_dir = os.path.dirname(os.path.abspath(__file__))
    demos_dir = os.path.abspath(os.path.join(script_dir, "..", "demos"))
    return os.path.join(demos_dir, f"{terrain}_base_{checkpoint}.mp4")


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


def _make_env_with_terrain_override(name, args, env_cfg, terrain_dict):
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


def _requested_terrain_level(rec_cfg, num_rows):
    if rec_cfg.terrain_level is not None:
        if rec_cfg.terrain_level < 0 or rec_cfg.terrain_level >= num_rows:
            raise ValueError(
                f"--terrain_level must be in [0, {num_rows - 1}], got {rec_cfg.terrain_level}"
            )
        return rec_cfg.terrain_level

    if rec_cfg.terrain_difficulty is not None:
        if rec_cfg.terrain_difficulty < 0.0 or rec_cfg.terrain_difficulty > 1.0:
            raise ValueError(
                f"--terrain_difficulty must be in [0, 1], got {rec_cfg.terrain_difficulty}"
            )
        return int(round(rec_cfg.terrain_difficulty * (num_rows - 1)))

    return None


def _apply_fixed_terrain_level(env, terrain_level):
    if terrain_level is None:
        return

    env.cfg.terrain.curriculum = False
    env.terrain_levels[:] = int(terrain_level)
    env._refresh_terrain_state()
    env_ids = torch.arange(env.num_envs, device=env.device)
    env.reset_idx(env_ids)


def _configure_eval_env(env_cfg, rec_cfg):
    env_cfg.env.episode_length_s = 60
    env_cfg.commands.resampling_time = 60
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.height = [0.02, 0.02]
    env_cfg.terrain.terrain_dict = {
        "smooth slope": 0.0,
        "rough slope up": 0.0,
        "rough slope down": 0.0,
        "rough stairs up": 0.0,
        "rough stairs down": 0.0,
        "discrete": 0.0,
        "stepping stones": 0.0,
        "gaps": 0.0,
        "smooth flat": 0.0,
        "pit": 0.0,
        "wall": 0.0,
        "platform": 0.0,
        "large stairs up": 0.0,
        "large stairs down": 0.0,
        "parkour": 0.0,
        "parkour_hurdle": 0.0,
        "parkour_flat": 0.0,
        "parkour_step": 0.0,
        "parkour_gap": 0.0,
        "alternating_step": 0.0,
        "bean_gap": 0.0,
        "asymmetric_gap": 0.0,
        "parkour_v2": 0.0,
        "narrow_gap": 0.0,
        "climbing_wall": 1.0,
        "demo": 0.0,
    }
    terrain_dict = dict(env_cfg.terrain.terrain_dict)
    env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
    terrain_level = _requested_terrain_level(rec_cfg, env_cfg.terrain.num_rows)
    if terrain_level is None:
        env_cfg.terrain.curriculum = False
        env_cfg.terrain.max_difficulty = True
    else:
        env_cfg.terrain.curriculum = True
        env_cfg.terrain.task_targeted_curriculum = False
        env_cfg.terrain.max_init_terrain_level = terrain_level
    env_cfg.depth.angle = [0, 1]
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 6
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    return terrain_dict, terrain_level


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

    if args.num_envs is None:
        args.num_envs = max(1, rec_cfg.record_env + 1)

    log_pth = "../../logs/{}/{}".format(args.proj_name, args.exptid)
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    if args.nodelay:
        env_cfg.domain_rand.action_delay_view = 0
    env_cfg.env.num_envs = args.num_envs
    terrain_dict, terrain_level = _configure_eval_env(env_cfg, rec_cfg)

    # Isaac Gym can create camera sensors in headless mode only when the env
    # keeps a real graphics device. LeggedRobot does that behind depth.use_camera.
    # We use that switch for rendering only; policy inference below still uses
    # the base actor and the true waypoint yaw already present in obs[:, 6:8].
    env_cfg.depth.use_camera = True
    env_cfg.depth.camera_num_envs = env_cfg.env.num_envs
    env_cfg.depth.camera_terrain_num_rows = env_cfg.terrain.num_rows
    env_cfg.depth.camera_terrain_num_cols = env_cfg.terrain.num_cols
    env_cfg.terrain.max_error_camera = env_cfg.terrain.max_error
    env_cfg.terrain.horizontal_scale_camera = env_cfg.terrain.horizontal_scale
    args.use_camera = True

    env, env_cfg = _make_env_with_terrain_override(
        name=args.task,
        args=args,
        env_cfg=env_cfg,
        terrain_dict=terrain_dict,
    )
    _apply_fixed_terrain_level(env, terrain_level)
    obs = env.get_observations()
    args.use_camera = False
    train_cfg.depth_encoder.if_depth = False

    if rec_cfg.record_env < 0 or rec_cfg.record_env >= env.num_envs:
        raise ValueError(
            f"--record_env {rec_cfg.record_env} out of range [0, {env.num_envs - 1}]"
        )

    cam_handle = _create_third_person_camera(
        env,
        rec_cfg.record_env,
        rec_cfg.record_width,
        rec_cfg.record_height,
        horizontal_fov=getattr(env.cfg.depth, "horizontal_fov", None),
    )
    smoother = _SmoothedThirdPersonCam(rec_cfg.camera_smooth)

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
        model, checkpoint = _get_model_and_checkpoint(
            root=jit_dir, checkpoint=args.checkpoint
        )
        path = os.path.join(jit_dir, model)
        print("Loading jit for policy:", path)
        policy_jit = torch.jit.load(path, map_location=env.device)
        policy = None
    else:
        _, checkpoint = _get_model_and_checkpoint(
            root=log_pth, checkpoint=args.checkpoint
        )
        policy = ppo_runner.get_inference_policy(device=env.device)
        policy_jit = None

    video_path = rec_cfg.video_out
    if video_path is None:
        video_path = _default_video_path(env_cfg, checkpoint)
    os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)

    fps = rec_cfg.video_fps if rec_cfg.video_fps is not None else float(round(1.0 / env.dt, 3))
    first = _grab_third_person_rgb(
        env,
        rec_cfg.record_env,
        cam_handle,
        smoother,
        rec_cfg.third_dist,
        rec_cfg.third_height,
        rec_cfg.third_lookat_z,
    )
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {video_path}")
    writer.write(first)
    print(f"Recording base policy env {rec_cfg.record_env} -> {video_path} @ {fps} FPS, {w}x{h}")
    print("Using true waypoint yaw from obs[:, 6:8]; no depth encoder is used.")
    if terrain_level is not None:
        terrain_difficulty = terrain_level / max(env_cfg.terrain.num_rows - 1, 1)
        print(
            f"Using fixed terrain level {terrain_level}/{env_cfg.terrain.num_rows - 1} "
            f"(difficulty {terrain_difficulty:.2f})."
        )

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
            writer.write(
                _grab_third_person_rgb(
                    env,
                    rec_cfg.record_env,
                    cam_handle,
                    smoother,
                    rec_cfg.third_dist,
                    rec_cfg.third_height,
                    rec_cfg.third_lookat_z,
                )
            )
    finally:
        writer.release()
        print("Video writer closed.")


if __name__ == "__main__":
    rec = _pop_script_argv()
    cli = get_args()
    play_base(cli, rec)
