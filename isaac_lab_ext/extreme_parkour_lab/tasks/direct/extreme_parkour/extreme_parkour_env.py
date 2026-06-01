"""First-pass Isaac Lab direct environment for Extreme Parkour.

This file ports the base proprioceptive control path from
`legged_gym.envs.base.legged_robot.LeggedRobot`.  It deliberately starts with a
plane terrain and zeroed height scans so the robot dynamics, observation layout,
and reward wiring can be validated before the custom parkour terrain and depth
camera are added.
"""

from __future__ import annotations

from collections.abc import Sequence
import math

import torch

try:
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation
    from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim import SimulationCfg
    from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
    from isaaclab.utils import configclass
    from isaaclab.utils.math import quat_rotate_inverse, sample_uniform
    from isaaclab_assets.robots.unitree import UNITREE_A1_CFG
except ModuleNotFoundError as exc:  # pragma: no cover - evaluated only in Isaac Lab.
    raise ModuleNotFoundError(
        "ExtremeParkourEnv must be imported from an Isaac Lab Python environment. "
        "Install Isaac Sim/Isaac Lab before launching this task."
    ) from exc


def _wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
    return (angles + math.pi) % (2 * math.pi) - math.pi


def _euler_from_quaternion_xyzw(quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = quat[:, 0]
    y = quat[:, 1]
    z = quat[:, 2]
    w = quat[:, 3]
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_arg = torch.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = torch.asin(pitch_arg)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


@configclass
class ExtremeParkourEnvCfg(DirectRLEnvCfg):
    """Configuration matching the Isaac Gym base-policy dimensions."""

    # env
    episode_length_s = 20.0
    decimation = 4
    action_scale = 0.25
    action_space = 12
    n_scan = 132
    n_priv = 9
    n_priv_latent = 29
    n_proprio = 53
    history_len = 10
    observation_space = n_proprio + n_scan + history_len * n_proprio + n_priv_latent + n_priv
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        render_interval=decimation,
        gravity=(0.0, 0.0, -9.81),
        physx=sim_utils.PhysxCfg(
            solver_type=1,
            min_position_iteration_count=4,
            max_position_iteration_count=4,
            min_velocity_iteration_count=0,
            max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.5,
        ),
    )

    # scene and robot
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=3.0,
        replicate_physics=True,
        clone_in_fabric=True,
    )
    robot = UNITREE_A1_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # original control/reward constants
    lin_vel_cmd_range = (0.3, 0.8)
    heading_range = (-1.6, 1.6)
    resampling_time_s = 6.0
    next_goal_threshold = 0.2
    base_height_cutoff = -0.25
    roll_pitch_cutoff = 1.5
    clip_actions = 1.2
    clip_observations = 100.0
    lin_vel_scale = 2.0
    ang_vel_scale = 0.25
    dof_pos_scale = 1.0
    dof_vel_scale = 0.05

    rew_tracking_goal_vel = 1.5
    rew_tracking_yaw = 0.5
    rew_lin_vel_z = -1.0
    rew_ang_vel_xy = -0.05
    rew_orientation = -1.0
    rew_dof_acc = -2.5e-7
    rew_action_rate = -0.1
    rew_delta_torques = -1.0e-7
    rew_torques = -1.0e-5
    rew_hip_pos = -0.5
    rew_dof_error = -0.04


class ExtremeParkourEnv(DirectRLEnv):
    cfg: ExtremeParkourEnvCfg

    def __init__(self, cfg: ExtremeParkourEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.joint_names = list(self.robot.joint_names)
        self.body_names = list(self.robot.body_names)
        self._joint_order = self._indices(
            [
                "RR_hip_joint",
                "RR_thigh_joint",
                "RR_calf_joint",
                "FL_hip_joint",
                "FL_thigh_joint",
                "FL_calf_joint",
                "RL_hip_joint",
                "RL_thigh_joint",
                "RL_calf_joint",
                "FR_hip_joint",
                "FR_thigh_joint",
                "FR_calf_joint",
            ],
            self.joint_names,
        )
        self._feet_order = self._indices(["FL_foot", "FR_foot", "RL_foot", "RR_foot"], self.body_names)
        self._hip_indices = self._indices(["FR_hip_joint", "FL_hip_joint", "RR_hip_joint", "RL_hip_joint"], self.joint_names)

        self.dt = self.cfg.sim.dt * self.cfg.decimation
        self.actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.torques = torch.zeros_like(self.actions)
        self.last_torques = torch.zeros_like(self.actions)
        self.last_joint_vel = torch.zeros_like(self.robot.data.joint_vel)
        self.obs_history_buf = torch.zeros(self.num_envs, self.cfg.history_len, self.cfg.n_proprio, device=self.device)
        self.contact_buf = torch.zeros(self.num_envs, 100, 4, device=self.device)
        self.commands = torch.zeros(self.num_envs, 4, device=self.device)
        self.target_yaw = torch.zeros(self.num_envs, device=self.device)
        self.delta_yaw = torch.zeros(self.num_envs, device=self.device)
        self.delta_next_yaw = torch.zeros(self.num_envs, device=self.device)
        self.cur_goals = torch.zeros(self.num_envs, 3, device=self.device)
        self.next_goals = torch.zeros(self.num_envs, 3, device=self.device)
        self.env_class = torch.full((self.num_envs,), 17.0, device=self.device)
        self.gravity_vec = torch.tensor((0.0, 0.0, -1.0), device=self.device).repeat(self.num_envs, 1)
        self.default_joint_pos = self.robot.data.default_joint_pos.clone()

        self._refresh_derived_state()
        self._resample_commands(torch.arange(self.num_envs, device=self.device))

    def _indices(self, names: list[str], source: list[str]) -> torch.Tensor:
        missing = [name for name in names if name not in source]
        if missing:
            raise RuntimeError(f"Missing expected names in Isaac Lab asset: {missing}. Available: {source}")
        return torch.tensor([source.index(name) for name in names], device=self.device, dtype=torch.long)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                )
            ),
        )
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        clip_actions = self.cfg.clip_actions / self.cfg.action_scale
        self.actions = torch.clip(actions, -clip_actions, clip_actions)
        ordered_actions = self.actions[:, self._joint_order]
        target_joint_pos = self.default_joint_pos + ordered_actions * self.cfg.action_scale
        self._joint_pos_target = target_joint_pos

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self._joint_pos_target)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        self._refresh_derived_state()
        scan = torch.zeros(self.num_envs, self.cfg.n_scan, device=self.device)
        priv_explicit = torch.cat(
            (
                self.base_lin_vel * self.cfg.lin_vel_scale,
                torch.zeros_like(self.base_lin_vel),
                torch.zeros_like(self.base_lin_vel),
            ),
            dim=-1,
        )
        priv_latent = torch.zeros(self.num_envs, self.cfg.n_priv_latent, device=self.device)
        foot_contacts = self._get_foot_contacts().float() - 0.5
        obs_now = torch.cat(
            (
                self.base_ang_vel * self.cfg.ang_vel_scale,
                torch.stack((self.roll, self.pitch), dim=1),
                torch.zeros(self.num_envs, 1, device=self.device),
                self.delta_yaw[:, None],
                self.delta_next_yaw[:, None],
                torch.zeros(self.num_envs, 2, device=self.device),
                self.commands[:, 0:1],
                torch.zeros(self.num_envs, 1, device=self.device),
                torch.ones(self.num_envs, 1, device=self.device),
                self._reindex((self.joint_pos - self.default_joint_pos) * self.cfg.dof_pos_scale),
                self._reindex(self.joint_vel * self.cfg.dof_vel_scale),
                self._reindex(self.actions),
                self._reindex_feet(foot_contacts),
            ),
            dim=-1,
        )
        obs_now[:, 6:8] = 0.0
        self.obs_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([obs_now] * self.cfg.history_len, dim=1),
            torch.cat((self.obs_history_buf[:, 1:], obs_now.unsqueeze(1)), dim=1),
        )
        obs = torch.cat((obs_now, scan, priv_explicit, priv_latent, self.obs_history_buf.reshape(self.num_envs, -1)), dim=-1)
        return {"policy": torch.clip(obs, -self.cfg.clip_observations, self.cfg.clip_observations)}

    def _get_rewards(self) -> torch.Tensor:
        self._refresh_derived_state()
        norm = torch.norm(self.target_pos_rel, dim=-1, keepdim=True)
        target_vec = self.target_pos_rel / (norm + 1e-5)
        goal_vel = torch.minimum(torch.sum(target_vec * self.root_lin_vel_w[:, :2], dim=-1), self.commands[:, 0])
        rew_tracking_goal_vel = goal_vel / (self.commands[:, 0] + 1e-5)
        rew_tracking_yaw = torch.exp(-torch.abs(self.delta_yaw))
        rew_lin_vel_z = torch.square(self.base_lin_vel[:, 2])
        rew_ang_vel_xy = torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)
        rew_orientation = torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
        rew_dof_acc = torch.sum(torch.square((self.last_joint_vel - self.joint_vel) / self.dt), dim=1)
        rew_action_rate = torch.norm(self.last_actions - self.actions, dim=1)
        rew_delta_torques = torch.sum(torch.square(self.torques - self.last_torques), dim=1)
        rew_torques = torch.sum(torch.square(self.torques), dim=1)
        rew_hip_pos = torch.sum(torch.square(self.joint_pos[:, self._hip_indices] - self.default_joint_pos[:, self._hip_indices]), dim=1)
        rew_dof_error = torch.sum(torch.square(self.joint_pos - self.default_joint_pos), dim=1)
        rewards = (
            self.cfg.rew_tracking_goal_vel * rew_tracking_goal_vel
            + self.cfg.rew_tracking_yaw * rew_tracking_yaw
            + self.cfg.rew_lin_vel_z * rew_lin_vel_z
            + self.cfg.rew_ang_vel_xy * rew_ang_vel_xy
            + self.cfg.rew_orientation * rew_orientation
            + self.cfg.rew_dof_acc * rew_dof_acc
            + self.cfg.rew_action_rate * rew_action_rate
            + self.cfg.rew_delta_torques * rew_delta_torques
            + self.cfg.rew_torques * rew_torques
            + self.cfg.rew_hip_pos * rew_hip_pos
            + self.cfg.rew_dof_error * rew_dof_error
        )
        rewards = torch.clip(rewards * self.dt, min=0.0)
        self.last_actions[:] = self.actions
        self.last_joint_vel[:] = self.joint_vel
        self.last_torques[:] = self.torques
        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._refresh_derived_state()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        roll_cutoff = torch.abs(self.roll) > self.cfg.roll_pitch_cutoff
        pitch_cutoff = torch.abs(self.pitch) > self.cfg.roll_pitch_cutoff
        height_cutoff = self.root_pos_w[:, 2] < self.cfg.base_height_cutoff
        died = roll_cutoff | pitch_cutoff | height_cutoff
        return died, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        root_state = self.robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        root_state[:, :2] += sample_uniform(-0.3, 0.3, (len(env_ids), 2), self.device)
        self.robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self.obs_history_buf[env_ids] = 0.0
        self.contact_buf[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.last_joint_vel[env_ids] = 0.0
        self.last_torques[env_ids] = 0.0
        self._resample_commands(torch.as_tensor(env_ids, device=self.device, dtype=torch.long))

    def _refresh_derived_state(self) -> None:
        self.root_pos_w = self.robot.data.root_pos_w
        self.root_quat_w = self.robot.data.root_quat_w
        self.root_lin_vel_w = self.robot.data.root_lin_vel_w
        self.root_ang_vel_w = self.robot.data.root_ang_vel_w
        self.joint_pos = self.robot.data.joint_pos
        self.joint_vel = self.robot.data.joint_vel
        self.base_lin_vel = quat_rotate_inverse(self.root_quat_w, self.root_lin_vel_w)
        self.base_ang_vel = quat_rotate_inverse(self.root_quat_w, self.root_ang_vel_w)
        self.projected_gravity = quat_rotate_inverse(self.root_quat_w, self.gravity_vec)
        self.roll, self.pitch, self.yaw = _euler_from_quaternion_xyzw(self.root_quat_w)
        self.target_pos_rel = self.cur_goals[:, :2] - self.root_pos_w[:, :2]
        self.next_target_pos_rel = self.next_goals[:, :2] - self.root_pos_w[:, :2]
        self.target_yaw = torch.atan2(self.target_pos_rel[:, 1], self.target_pos_rel[:, 0])
        next_target_yaw = torch.atan2(self.next_target_pos_rel[:, 1], self.next_target_pos_rel[:, 0])
        self.delta_yaw = _wrap_to_pi(self.target_yaw - self.yaw)
        self.delta_next_yaw = _wrap_to_pi(next_target_yaw - self.yaw)
        if hasattr(self.robot.data, "applied_torque"):
            self.torques = self.robot.data.applied_torque

    def _resample_commands(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        self.commands[env_ids, 0] = sample_uniform(
            self.cfg.lin_vel_cmd_range[0],
            self.cfg.lin_vel_cmd_range[1],
            (len(env_ids),),
            self.device,
        )
        self.commands[env_ids, 3] = sample_uniform(
            self.cfg.heading_range[0],
            self.cfg.heading_range[1],
            (len(env_ids),),
            self.device,
        )
        self.cur_goals[env_ids, 0] = self.root_pos_w[env_ids, 0] + self.cfg.resampling_time_s * self.commands[env_ids, 0]
        self.cur_goals[env_ids, 1] = self.root_pos_w[env_ids, 1]
        self.next_goals[env_ids, 0] = self.cur_goals[env_ids, 0] + self.cfg.resampling_time_s * self.commands[env_ids, 0]
        self.next_goals[env_ids, 1] = self.cur_goals[env_ids, 1]

    def _reindex(self, value: torch.Tensor) -> torch.Tensor:
        return value[:, [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]]

    def _reindex_feet(self, value: torch.Tensor) -> torch.Tensor:
        return value[:, [1, 0, 3, 2]]

    def _get_foot_contacts(self) -> torch.Tensor:
        # Contact sensors are enabled on the built-in asset, but the exact field
        # name has changed across Isaac Lab releases. Keep this zeroed until the
        # target Isaac Lab version is available for validation.
        return torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
