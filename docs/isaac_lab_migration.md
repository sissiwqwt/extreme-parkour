# Isaac Lab Migration Notes

This repository was originally built on Isaac Gym Preview through `legged_gym`.
The Isaac Lab port should be developed as a separate task until it reaches
feature parity, so the existing Isaac Gym training path remains usable.

## Target

- Isaac Lab direct workflow (`DirectRLEnv`), because the original environment
  computes observations, rewards, resets, terrain curriculum, and action delay
  directly in one task class.
- Keep the original `rsl_rl` policy architecture and PPO settings where
  possible.
- Start from the base policy path without depth, then add the depth-camera
  distillation path after dynamics and terrain rewards are close.

## Current Local Blocker

The current machine cannot run the simulator:

- Python is 3.13.
- PyTorch is CPU-only.
- `isaacgym`, `isaacsim`, and `isaaclab` are not installed.

Full reproduction needs an Isaac Sim/Isaac Lab environment with an NVIDIA GPU.
This scaffold is therefore checked syntactically where possible, but simulator
launch must be verified on the target machine.

## Porting Map

Original Isaac Gym code:

- `BaseTask.create_sim()` -> Isaac Lab `_setup_scene()`
- `LeggedRobot.step()` action preprocessing -> `_pre_physics_step()`
- Isaac Gym torque write in the decimation loop -> `_apply_action()`
- `compute_observations()` -> `_get_observations()`
- `compute_reward()` and `_reward_*()` -> `_get_rewards()`
- `check_termination()` -> `_get_dones()`
- `reset_idx()` -> `_reset_idx()`
- `gymtorch.wrap_tensor(...)` state buffers -> `Articulation.data.*`
- `isaacgym.terrain_utils` heightfield/trimesh -> Isaac Lab terrain importer or
  generated mesh/USD terrain
- `get_camera_image_gpu_tensor` -> Isaac Lab camera/tiled camera sensors

## Verification Plan

1. Import task package inside Isaac Lab.
2. Launch one environment with plane terrain.
3. Launch many environments with the built-in Unitree A1 asset.
4. Match joint order, default joint angles, torque limits, and contact body names.
5. Port/generated parkour trimesh terrain and height sampler.
6. Re-enable task-targeted curriculum.
7. Re-enable depth camera and the distillation runner.
8. Fine-tune from Isaac Gym checkpoints and compare evaluation CSV metrics.

