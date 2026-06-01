from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import euler_xyz_from_quat

from .a1_parkour_env_cfg import A1ParkourEnvCfg
from .terrain_goals import curriculum_terrain_names, generate_goal_track


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


class A1ParkourEnv(DirectRLEnv):
    """Initial Isaac Lab environment for the extreme-parkour A1 task.

    This first pass intentionally ports only the base locomotion shell:
    A1 asset setup, direct RL control, rough-terrain scene hookup, and a
    simple reward/termination loop. Custom parkour goals, terrain classes,
    curriculum, teacher-student depth flow, and the original observation
    layout remain to be ported.
    """

    cfg: A1ParkourEnvCfg

    def __init__(self, cfg: A1ParkourEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        action_dim = gym.spaces.flatdim(self.single_action_space)
        self._actions = torch.zeros(self.num_envs, action_dim, device=self.device)
        self._previous_actions = torch.zeros_like(self._actions)
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)
        self._goal_reach_timer = torch.zeros(self.num_envs, device=self.device)
        self._cur_goal_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._reached_goal = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._terrain_class = torch.zeros(self.num_envs, 2, device=self.device)
        self._terrain_name_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._env_goals = torch.zeros(
            self.num_envs,
            self.cfg.num_goals + self.cfg.num_future_goal_obs,
            3,
            device=self.device,
        )
        self._cur_goals = torch.zeros(self.num_envs, 3, device=self.device)
        self._next_goals = torch.zeros(self.num_envs, 3, device=self.device)
        self._target_pos_rel = torch.zeros(self.num_envs, 2, device=self.device)
        self._next_target_pos_rel = torch.zeros(self.num_envs, 2, device=self.device)
        self._target_yaw = torch.zeros(self.num_envs, device=self.device)
        self._next_target_yaw = torch.zeros(self.num_envs, device=self.device)
        self._delta_yaw = torch.zeros(self.num_envs, device=self.device)
        self._delta_next_yaw = torch.zeros(self.num_envs, device=self.device)
        self._terrain_names = list(self.cfg.terrain_dict.keys())
        terrain_generator_cfg = self.cfg.terrain.terrain_generator
        if terrain_generator_cfg is None:
            raise ValueError("A1ParkourEnv requires terrain.terrain_generator to be configured.")
        self._terrain_names_by_col = curriculum_terrain_names(terrain_generator_cfg.num_cols, self.cfg.terrain_dict)
        self._rng = np.random.default_rng()

        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in (
                "track_lin_vel_xy_exp",
                "track_ang_vel_z_exp",
                "lin_vel_z_l2",
                "ang_vel_xy_l2",
                "joint_torques_l2",
                "joint_acc_l2",
                "action_rate_l2",
                "feet_air_time",
                "undesired_contacts",
                "flat_orientation_l2",
                "base_height_l2",
                "goal_heading",
                "goal_progress",
            )
        }

        self._base_id, _ = self._contact_sensor.find_bodies("trunk")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*foot")
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(".*thigh|.*calf")

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        self._height_scanner = RayCaster(self.cfg.height_scanner)
        self.scene.sensors["height_scanner"] = self._height_scanner

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        self._processed_actions = self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _generate_goals(self, env_ids: torch.Tensor):
        root_pos = self._robot.data.default_root_state[env_ids, :3] + self._terrain.env_origins[env_ids]

        goals_world = torch.zeros(len(env_ids), self.cfg.num_goals, 3, device=self.device)
        for local_idx, env_id in enumerate(env_ids):
            terrain_col = int(self._terrain.terrain_types[env_id].item())
            terrain_name = self._terrain_names_by_col[terrain_col]
            goal_track = generate_goal_track(
                terrain_name=terrain_name,
                num_goals=self.cfg.num_goals,
                env_length=self.cfg.terrain_length,
                env_width=self.cfg.terrain_width,
                difficulty=float(self._rng.uniform(0.0, 1.0)),
                rng=self._rng,
            )
            goal_track[:, 0] += float(root_pos[local_idx, 0].item() - 1.0)
            goal_track[:, 1] += float(root_pos[local_idx, 1].item() - self.cfg.terrain_width * 0.5)
            goals_world[local_idx] = torch.from_numpy(goal_track).to(self.device)

            terrain_id = self._terrain_names.index(terrain_name)
            self._terrain_name_ids[env_id] = terrain_id
            self._terrain_class[env_id, 0] = 0.0 if terrain_name == "climbing_wall" else 1.0
            self._terrain_class[env_id, 1] = 1.0 if terrain_name == "climbing_wall" else 0.0

        last_goal = goals_world[:, -1:].repeat(1, self.cfg.num_future_goal_obs, 1)
        self._env_goals[env_ids] = torch.cat((goals_world, last_goal), dim=1)

    def _gather_goals(self, future: int = 0) -> torch.Tensor:
        goal_idx = (self._cur_goal_idx + future).clamp(max=self._env_goals.shape[1] - 1)
        return self._env_goals[torch.arange(self.num_envs, device=self.device), goal_idx]

    def _update_goals(self):
        next_flag = self._goal_reach_timer > (self.cfg.reach_goal_delay_s / self.step_dt)
        self._cur_goal_idx[next_flag] += 1
        self._cur_goal_idx.clamp_(max=self.cfg.num_goals)
        self._goal_reach_timer[next_flag] = 0.0

        self._cur_goals = self._gather_goals(future=0)
        self._next_goals = self._gather_goals(future=1)

        root_xy = self._robot.data.root_pos_w[:, :2]
        self._reached_goal = torch.norm(root_xy - self._cur_goals[:, :2], dim=1) < self.cfg.next_goal_threshold
        self._goal_reach_timer[self._reached_goal] += 1.0

        self._target_pos_rel = self._cur_goals[:, :2] - root_xy
        self._next_target_pos_rel = self._next_goals[:, :2] - root_xy

        target_vec = self._target_pos_rel / (torch.norm(self._target_pos_rel, dim=-1, keepdim=True) + 1e-5)
        next_target_vec = self._next_target_pos_rel / (torch.norm(self._next_target_pos_rel, dim=-1, keepdim=True) + 1e-5)
        self._target_yaw = torch.atan2(target_vec[:, 1], target_vec[:, 0])
        self._next_target_yaw = torch.atan2(next_target_vec[:, 1], next_target_vec[:, 0])

        _, _, yaw = euler_xyz_from_quat(self._robot.data.root_quat_w)
        self._delta_yaw = wrap_to_pi(self._target_yaw - yaw)
        self._delta_next_yaw = wrap_to_pi(self._next_target_yaw - yaw)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        self._update_goals()

        height_scan = (
            self._height_scanner.data.pos_w[:, 2].unsqueeze(1) - self._height_scanner.data.ray_hits_w[..., 2] - 0.5
        ).clip(-1.0, 1.0)

        roll, pitch, _ = euler_xyz_from_quat(self._robot.data.root_quat_w)
        imu_obs = torch.stack((roll, pitch), dim=1)
        base_obs = torch.cat(
            (
                self._robot.data.root_ang_vel_b,
                imu_obs,
                torch.zeros_like(self._delta_yaw.unsqueeze(1)),
                self._delta_yaw.unsqueeze(1),
                self._delta_next_yaw.unsqueeze(1),
                torch.zeros(self.num_envs, 2, device=self.device),
                self._commands[:, 0:1],
                self._terrain_class,
                self._robot.data.root_ang_vel_b,
                self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                self._robot.data.joint_vel,
                self._actions,
                self._contact_sensor.data.current_contact_time[:, self._feet_ids] > 0.0,
                height_scan,
            ),
            dim=-1,
        )

        # The Isaac Gym task contains additional goal/terrain-history/depth
        # terms. Keep a fixed-width placeholder so the RL config shape remains
        # stable while those observation terms are ported incrementally.
        if base_obs.shape[1] < self.cfg.observation_space:
            pad = torch.zeros(
                self.num_envs,
                self.cfg.observation_space - base_obs.shape[1],
                device=self.device,
                dtype=base_obs.dtype,
            )
            base_obs = torch.cat((base_obs, pad), dim=-1)

        return {"policy": base_obs}

    def _get_rewards(self) -> torch.Tensor:
        lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        z_vel_error = torch.square(self._robot.data.root_lin_vel_b[:, 2])
        ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        joint_acc = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        base_height_error = torch.square(self._robot.data.root_pos_w[:, 2] - self.cfg.base_height_target)
        goal_progress = self._reached_goal.float()
        goal_heading = torch.exp(-torch.abs(self._delta_yaw))

        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]
        feet_air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1) * (
            torch.norm(self._commands[:, :2], dim=1) > 0.1
        )

        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        undesired_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1)[0] > 1.0
        )
        undesired_contact_count = torch.sum(undesired_contact, dim=1)

        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)

        rewards = {
            "track_lin_vel_xy_exp": torch.exp(-lin_vel_error / 0.25) * self.cfg.lin_vel_reward_scale * self.step_dt,
            "track_ang_vel_z_exp": torch.exp(-yaw_rate_error / 0.25) * self.cfg.yaw_rate_reward_scale * self.step_dt,
            "lin_vel_z_l2": z_vel_error * self.cfg.z_vel_reward_scale * self.step_dt,
            "ang_vel_xy_l2": ang_vel_error * self.cfg.ang_vel_reward_scale * self.step_dt,
            "joint_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "joint_acc_l2": joint_acc * self.cfg.joint_accel_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "feet_air_time": feet_air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "undesired_contacts": undesired_contact_count * self.cfg.undesired_contacts_reward_scale * self.step_dt,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,
            "base_height_l2": base_height_error * self.cfg.base_height_reward_scale * self.step_dt,
            "goal_heading": goal_heading * self.step_dt,
            "goal_progress": goal_progress * self.step_dt,
        }

        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1)
        reached_end = self._cur_goal_idx >= self.cfg.num_goals
        time_out = torch.logical_or(time_out, reached_end)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if len(env_ids) == self.num_envs:
            self.episode_length_buf[:] = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(-1.0, 1.0)
        self._cur_goal_idx[env_ids] = 0
        self._goal_reach_timer[env_ids] = 0.0
        self._reached_goal[env_ids] = False

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
        self._generate_goals(env_ids)
        self._cur_goals[env_ids] = self._gather_goals(future=0)[env_ids]
        self._next_goals[env_ids] = self._gather_goals(future=1)[env_ids]

        self.extras["log"] = {}
        for key, value in self._episode_sums.items():
            episodic_sum_avg = torch.mean(value[env_ids])
            self.extras["log"][f"Episode_Reward/{key}"] = episodic_sum_avg / self.max_episode_length_s
            value[env_ids] = 0.0
        self.extras["log"]["Episode_Termination/base_contact"] = torch.count_nonzero(
            self.reset_terminated[env_ids]
        ).item()
        self.extras["log"]["Episode_Termination/time_out"] = torch.count_nonzero(
            self.reset_time_outs[env_ids]
        ).item()
