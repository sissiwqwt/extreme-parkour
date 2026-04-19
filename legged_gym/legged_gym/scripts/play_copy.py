# Created by Xie Hongzhao. For Inteligent Robotics Course Project @ Shanghai Jiao Tong University (2026 Spring).

# Headless rollout with policy inference; saves RGB video (no web viewer).
# Recording can use the depth-policy body camera, or a separate third-person camera
# with optional temporal smoothing (see --record_camera).


# Usage:
# ```bash
# cd ${SCRIPT_DIR}
# python play_headless_record.py
# python legged_gym/scripts/play_headless_record.py \
#   --use_camera --headless \
#   --exptid {your_exp_id} --proj_name {your_project_name}

import argparse
import os
import sys
from datetime import datetime

import faulthandler

# `envs` before `legged_gym.utils`: envs/__init__.py imports task_registry while
# utils/__init__.py also pulls task_registry — importing utils first causes a cycle.
import isaacgym
from isaacgym import gymapi

from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R_scipy


def _pop_script_argv():
    """Parse and remove recording-related flags before `get_args()` consumes sys.argv."""
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument(
        "--video_out",
        type=str,
        default=None,
        help="Output video file path (.mp4 recommended). Default: headless_play_<timestamp>.mp4 in CWD.",
    )
    p.add_argument(
        "--video_fps",
        type=float,
        default=None,
        help="Video FPS. Default: 1 / env.dt (policy control rate).",
    )
    p.add_argument(
        "--record_env",
        type=int,
        default=0,
        help="Which parallel env index to read the tracking camera from.",
    )
    p.add_argument(
        "--record_camera",
        type=str,
        default="third_person",
        choices=("third_person", "body"),
        help="third_person: world-space follow cam (set_camera_location), smoothed. "
        "body: RGB from the depth body-mounted sensor (same as policy).",
    )
    p.add_argument(
        "--third_dist",
        type=float,
        default=2.6,
        help="Third-person: base-link offset along -X_body (meters behind nose).",
    )
    p.add_argument(
        "--third_height",
        type=float,
        default=0.75,
        help="Third-person: vertical offset in base link frame (meters).",
    )
    p.add_argument(
        "--third_lookat_z",
        type=float,
        default=0.12,
        help="Third-person: look target = base position + this world Z offset.",
    )
    p.add_argument(
        "--camera_smooth",
        type=float,
        default=0.14,
        help="Third-person EMA factor in [0,1]: larger = snappier, smaller = less shake.",
    )
    p.add_argument(
        "--record_width",
        type=int,
        default=960,
        help="Third-person camera width (body mode uses depth sensor size).",
    )
    p.add_argument(
        "--record_height",
        type=int,
        default=540,
        help="Third-person camera height (body mode uses depth sensor size).",
    )
    ns, rest = p.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    return ns


def get_load_path_jit(root, checkpoint=-1, model_name_include="model"):
    if checkpoint == -1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: "{0:0>15}".format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint)
    return model


def _rgba_to_bgr(rgba):
    rgb = rgba.reshape(rgba.shape[0], -1, 4)[..., :3]
    return rgb[..., ::-1].copy()


def _grab_body_camera_rgb(env, env_id):
    """RGB from the depth pipeline camera (attach_camera_to_body, FOLLOW_TRANSFORM)."""
    env.gym.step_graphics(env.sim)
    env.gym.render_all_camera_sensors(env.sim)
    rgba = env.gym.get_camera_image(
        env.sim,
        env.envs[env_id],
        env.cam_handles[env_id],
        gymapi.IMAGE_COLOR,
    )
    return _rgba_to_bgr(rgba)


def _create_third_person_camera(env, env_id, width, height, horizontal_fov=None):
    props = gymapi.CameraProperties()
    props.width = width
    props.height = height
    if horizontal_fov is not None:
        props.horizontal_fov = float(horizontal_fov)
    return env.gym.create_camera_sensor(env.envs[env_id], props)


class _SmoothedThirdPersonCam:
    """Exponential moving average on camera / look-at world positions (reduces base jitter)."""

    def __init__(self, alpha):
        self.alpha = float(np.clip(alpha, 1e-6, 1.0))
        self._cam = None
        self._tgt = None

    def smooth(self, cam_world, target_world):
        a = self.alpha
        if self._cam is None:
            self._cam = cam_world.astype(np.float64).copy()
            self._tgt = target_world.astype(np.float64).copy()
        else:
            self._cam = (1.0 - a) * self._cam + a * cam_world
            self._tgt = (1.0 - a) * self._tgt + a * target_world
        return self._cam, self._tgt


def _third_person_pose(env, env_id, dist, height, lookat_z):
    """Desired camera position and look-at from current root state (Isaac Gym quat xyzw)."""
    root = env.root_states[env_id, :3].detach().cpu().numpy()
    quat = env.root_states[env_id, 3:7].detach().cpu().numpy()
    rot = R_scipy.from_quat(quat)
    offset_body = np.array([-float(dist), 0.0, float(height)], dtype=np.float64)
    cam_world = root + rot.apply(offset_body)
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
        env.sim, env.envs[env_id], cam_handle, gymapi.IMAGE_COLOR
    )
    return _rgba_to_bgr(rgba)


def play_headless_record(args, rec_cfg):
    faulthandler.enable()

    args.headless = True

    if not args.use_camera:
        raise RuntimeError(
            "Body-mounted tracking camera requires depth camera to be enabled. "
            "Re-run with `--use_camera` (same as training / `play.py` for vision policies)."
        )

    if args.num_envs is None:
        args.num_envs = max(1, rec_cfg.record_env + 1)

    log_pth = "../../logs/{}/".format(args.proj_name) + args.exptid

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    if args.nodelay:
        env_cfg.domain_rand.action_delay_view = 0
    env_cfg.env.num_envs = args.num_envs
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
        "smooth flat": 0,
        "pit": 0.0,
        "wall": 0.0,
        "platform": 0.0,
        "large stairs up": 0.0,
        "large stairs down": 0.0,
        "parkour": 0.2,
        "parkour_hurdle": 0.2,
        "parkour_flat": 0.0,
        "parkour_step": 0.2,
        "parkour_gap": 0.2,
        "demo": 0.2,
    }
    env_cfg.terrain.terrain_proportions = list(env_cfg.terrain.terrain_dict.values())
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_difficulty = True

    env_cfg.depth.angle = [0, 1]
    env_cfg.noise.add_noise = True
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 6
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False

    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    if not env.cfg.depth.use_camera or not env.cam_handles:
        raise RuntimeError(
            "Environment has no camera handles; enable depth camera in config / `--use_camera`."
        )

    if rec_cfg.record_env < 0 or rec_cfg.record_env >= env.num_envs:
        raise ValueError(
            f"--record_env {rec_cfg.record_env} out of range [0, {env.num_envs - 1}]"
        )

    third_person_handle = None
    third_smoothing = None
    if rec_cfg.record_camera == "third_person":
        third_person_handle = _create_third_person_camera(
            env,
            rec_cfg.record_env,
            rec_cfg.record_width,
            rec_cfg.record_height,
            horizontal_fov=getattr(env.cfg.depth, "horizontal_fov", None),
        )
        third_smoothing = _SmoothedThirdPersonCam(rec_cfg.camera_smooth)

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
        path = os.path.join(log_pth, "traced")
        model = get_load_path_jit(root=path, checkpoint=args.checkpoint)
        path = os.path.join(path, model)
        print("Loading jit for policy: ", path)
        policy_jit = torch.jit.load(path, map_location=env.device)
    else:
        policy = ppo_runner.get_inference_policy(device=env.device)
    _estimator = ppo_runner.get_estimator_inference_policy(device=env.device)
    if env.cfg.depth.use_camera:
        depth_encoder = ppo_runner.get_depth_encoder_inference_policy(device=env.device)

    actions = torch.zeros(env.num_envs, 12, device=env.device, requires_grad=False)
    infos = {}
    infos["depth"] = (
        env.depth_buffer.clone().to(ppo_runner.device)[:, -1]
        if ppo_runner.if_depth
        else None
    )
    depth_latent = None

    import cv2

    video_path = rec_cfg.video_out
    if video_path is None:
        video_path = os.path.abspath(
            f"headless_play_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
        )
    os.makedirs(os.path.dirname(video_path) or ".", exist_ok=True)

    fps = (
        rec_cfg.video_fps
        if rec_cfg.video_fps is not None
        else float(round(1.0 / env.dt, 3))
    )

    def _record_frame():
        if rec_cfg.record_camera == "third_person":
            return _grab_third_person_rgb(
                env,
                rec_cfg.record_env,
                third_person_handle,
                third_smoothing,
                rec_cfg.third_dist,
                rec_cfg.third_height,
                rec_cfg.third_lookat_z,
            )
        return _grab_body_camera_rgb(env, rec_cfg.record_env)

    first = _record_frame()
    h, w = first.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open VideoWriter for {video_path}")
    writer.write(first)
    mode_desc = (
        f"third_person (smooth α={rec_cfg.camera_smooth}, "
        f"dist={rec_cfg.third_dist}, h={rec_cfg.third_height})"
        if rec_cfg.record_camera == "third_person"
        else "body-mounted depth camera"
    )
    print(
        f"Recording env {rec_cfg.record_env} ({mode_desc}) → {video_path} @ {fps} FPS, {w}x{h}"
    )

    try:
        for _ in range(int(env.max_episode_length)):
            if args.use_jit:
                if env.cfg.depth.use_camera:
                    if infos["depth"] is not None:
                        depth_latent = torch.ones(
                            (env_cfg.env.num_envs, 32), device=env.device
                        )
                        actions, depth_latent = policy_jit(
                            obs.detach(), True, infos["depth"], depth_latent
                        )
                    else:
                        depth_buffer = torch.ones(
                            (env_cfg.env.num_envs, 58, 87), device=env.device
                        )
                        actions, depth_latent = policy_jit(
                            obs.detach(), False, depth_buffer, depth_latent
                        )
                else:
                    obs_jit = torch.cat(
                        (
                            obs.detach()[
                                :, : env_cfg.env.n_proprio + env_cfg.env.n_priv
                            ],
                            obs.detach()[
                                :, -env_cfg.env.history_len * env_cfg.env.n_proprio :
                            ],
                        ),
                        dim=1,
                    )
                    actions = policy_jit(obs_jit)
            else:
                if env.cfg.depth.use_camera:
                    if infos["depth"] is not None:
                        obs_student = obs[:, : env_cfg.env.n_proprio].clone()
                        obs_student[:, 6:8] = 0
                        depth_latent_and_yaw = depth_encoder(
                            infos["depth"], obs_student
                        )
                        depth_latent = depth_latent_and_yaw[:, :-2]
                        yaw = depth_latent_and_yaw[:, -2:]
                        obs[:, 6:8] = 1.5 * yaw
                else:
                    depth_latent = None

                if hasattr(ppo_runner.alg, "depth_actor"):
                    actions = ppo_runner.alg.depth_actor(
                        obs.detach(), hist_encoding=True, scandots_latent=depth_latent
                    )
                else:
                    actions = policy(
                        obs.detach(), hist_encoding=True, scandots_latent=depth_latent
                    )

            obs, _, _, _, infos = env.step(actions.detach())

            writer.write(_record_frame())

            # print(
            #     "time:",
            #     env.episode_length_buf[env.lookat_id].item() / 50,
            #     "cmd vx",
            #     env.commands[env.lookat_id, 0].item(),
            #     "actual vx",
            #     env.base_lin_vel[env.lookat_id, 0].item(),
            # )
    finally:
        writer.release()
        print("Video writer closed.")


if __name__ == "__main__":
    rec = _pop_script_argv()
    cli = get_args()
    play_headless_record(cli, rec)
