from __future__ import annotations

from copy import deepcopy

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.unitree import UNITREE_A1_CFG
from .terrain_cfg import PARKOUR_TERRAINS_CFG


@configclass
class A1ParkourEnvCfg(DirectRLEnvCfg):
    """Initial Isaac Lab port of the A1 parkour task.

    This is a migration scaffold, not a feature-complete port. It keeps the
    A1 robot/control setup close to the old Isaac Gym config and provides a
    stable Isaac Lab target for incremental porting of terrain goals, parkour
    rewards, waypoint observations, and depth sensing.
    """

    episode_length_s = 20.0
    decimation = 4
    action_scale = 0.25
    action_space = 12
    observation_space = 235
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=PARKOUR_TERRAINS_CFG,
        max_init_terrain_level=4,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=4.0, replicate_physics=True)

    robot: ArticulationCfg = deepcopy(UNITREE_A1_CFG).replace(prim_path="/World/envs/env_.*/Robot")
    robot.actuators["base_legs"].stiffness = 40.0
    robot.actuators["base_legs"].damping = 1.0
    robot.spawn.articulation_props.enabled_self_collisions = False

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )

    height_scanner: RayCasterCfg = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/trunk",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # Legacy-parity metadata from the Gym config, not yet wired into task logic.
    num_goals = 8
    num_future_goal_obs = 2
    next_goal_threshold = 0.2
    reach_goal_delay_s = 0.1
    waypoint_delta = 0.7
    terrain_length = 18.0
    terrain_width = 4.0
    terrain_dict = {
        "parkour": 0.15,
        "parkour_hurdle": 0.10,
        "parkour_flat": 0.10,
        "parkour_step": 0.10,
        "parkour_gap": 0.10,
        "alternating_step": 0.10,
        "beam_gap": 0.10,
        "asymmetric_gap": 0.10,
        "parkour_v2": 0.15,
        "narrow_gap": 0.10,
        "climbing_wall": 0.10,
    }

    # Reward scales, seeded from the rough locomotion baseline. These should be
    # replaced with the parkour-specific reward terms from legged_robot.py.
    lin_vel_reward_scale = 1.0
    yaw_rate_reward_scale = 0.5
    z_vel_reward_scale = -2.0
    ang_vel_reward_scale = -0.05
    joint_torque_reward_scale = -2.5e-5
    joint_accel_reward_scale = -2.5e-7
    action_rate_reward_scale = -0.01
    feet_air_time_reward_scale = 0.125
    undesired_contacts_reward_scale = -1.0
    flat_orientation_reward_scale = 0.0
    base_height_target = 0.25
    base_height_reward_scale = 0.0
