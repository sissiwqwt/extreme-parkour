# Extreme Parkour Repository — Instruction Guide

This document gives you a practical “big picture” of the repository, then drills into the most important files so you can read the code with a map in mind.

## 1) Big picture: what this repo is and how it is organized

This repository implements reinforcement learning for quadruped **extreme parkour locomotion** in Isaac Gym.

At a high level, there are two main Python packages:

- `legged_gym/`: simulation environment, robot/task configs, training/play scripts, and utility plumbing.
- `rsl_rl/`: PPO implementation and runner code used by `legged_gym`.

Core idea:

1. A task name (default: `a1`) is mapped to an environment class + config.
2. The environment simulates many robots in parallel on GPU.
3. PPO trains a policy from observations/rewards.
4. You can run trained policies in viewer/web mode and export TorchScript for deployment.

---

## 2) Code map (mental model)

### Entry points (what you run)

- `legged_gym/legged_gym/scripts/train.py`: launches training.
- `legged_gym/legged_gym/scripts/play.py`: loads a checkpoint and runs inference/visualization.
- `legged_gym/legged_gym/scripts/save_jit.py`: exports policy to JIT/TorchScript for deployment.

### Task/environment registration and wiring

- `legged_gym/legged_gym/envs/__init__.py`: registers `a1` and `go1` tasks in the global `task_registry`.
- `legged_gym/legged_gym/utils/task_registry.py`: factory that builds environments and PPO runners.

### Environment behavior/configuration

- `legged_gym/legged_gym/envs/base/legged_robot.py`: main environment dynamics loop, observation/reward/reset workflow, camera/depth handling, goal updates.
- `legged_gym/legged_gym/envs/base/legged_robot_config.py`: base configuration schema (env, terrain, rewards, control, domain randomization, etc.).
- `legged_gym/legged_gym/envs/a1/a1_parkour_config.py`: A1-specific overrides (init pose, PD gains, URDF, reward targets).

### RL algorithm backend

- `rsl_rl/rsl_rl/runners/on_policy_runner.py`: rollout/train loop orchestration.
- `rsl_rl/rsl_rl/algorithms/ppo.py`: PPO losses and updates.
- `rsl_rl/rsl_rl/modules/*`: policy/value/depth-estimator networks.

---

## 3) End-to-end control flow (train)

When you run `python train.py ...`, this is the path:

1. `train.py` parses CLI args and initializes W&B/log directory.
2. It calls `task_registry.make_env(...)` to create Isaac Gym env from registered task config.
3. It calls `task_registry.make_alg_runner(...)` to build PPO runner.
4. Runner `learn(...)` performs rollout collection + PPO updates until max iterations.

Important detail: `task_registry` applies command-line overrides into env/train configs (e.g., `--use_camera`, `--num_envs`, `--delay`, `--resume`). This is centralized in `helpers.update_cfg_from_args(...)`.

---

## 4) Detailed explanation of important files

## A) `legged_gym/legged_gym/utils/task_registry.py`

Think of this file as the **dependency injection container** for tasks:

- `register(name, task_class, env_cfg, train_cfg)`: binds a task name to class/config pair.
- `make_env(...)`: fetches the task class and config, applies CLI overrides, parses sim params, sets seeds, and constructs the Isaac Gym task instance.
- `make_alg_runner(...)`: converts train config to dict, creates `OnPolicyRunner`, and handles checkpoint resume/load path logic.

Why it matters:

- It cleanly separates “what task to run” from training scripts.
- It is the key place to extend repo with a new robot/task.

How to add a new task:

1. Create env/config classes.
2. Register them in `envs/__init__.py` using `task_registry.register(...)`.
3. Pass `--task your_task_name` at runtime.

## B) `legged_gym/legged_gym/envs/base/legged_robot.py`

This is the most central environment runtime.

Main responsibilities:

- **Simulation stepping**: `step(actions)` applies action delay logic, computes torques, advances physics at `decimation` substeps, clips observations.
- **Sensor pipeline**: optional depth camera acquisition and preprocessing (`update_depth_buffer`, cropping/resizing/normalizing).
- **Task progression**: `_update_goals()` updates current/next waypoint logic.
- **RL interface**: `post_physics_step()` refreshes tensors, computes kinematics, checks termination, computes rewards, resets finished envs.

Why it matters:

- If training is unstable, rewards feel wrong, or motion quality is bad, this is one of the first files to inspect.
- Most performance-sensitive task mechanics are here.

## C) `legged_gym/legged_gym/envs/a1/a1_parkour_config.py`

This file specializes the base config for A1 parkour:

- Initial base height and default joint angles.
- PD control gains, action scale, decimation.
- URDF asset path and contact rules (penalty/termination links).
- Reward-related targets like base height.
- PPO-specific overrides like entropy coefficient / experiment naming.

Why it matters:

- It defines your behavior envelope more than network architecture does.
- Most practical tuning for gait style and robustness starts here (plus base config).

## D) `legged_gym/legged_gym/utils/helpers.py`

This is runtime glue:

- `get_args()` defines custom CLI flags used across scripts.
- `update_cfg_from_args(...)` injects runtime changes into env/train cfg.
- `get_load_path(...)` checkpoint discovery (including short run-id matching).
- `parse_sim_params(...)`, seeding utilities, and JIT export helpers.

Why it matters:

- Many “why does this run differently than expected?” questions come from hidden CLI-based config overrides here.

---

## 5) Practical usage guide

## Installation (from README)

1. Create Python env (`python=3.8`).
2. Install specific CUDA-enabled torch versions.
3. Install Isaac Gym Python package.
4. Install local editable packages:
   - `rsl_rl`
   - `legged_gym`
5. Install extra deps (`wandb`, `opencv-python`, etc.).

> Note: this repo expects NVIDIA Isaac Gym and compatible GPU/CUDA setup.

## Training examples

From `legged_gym/legged_gym/scripts`:

- Base policy:
  ```bash
  python train.py --exptid xxx-xx-WHATEVER --device cuda:0
  ```
- Distillation policy:
  ```bash
  python train.py --exptid yyy-yy-WHATEVER --device cuda:0 --resume --resumeid xxx-xx --delay --use_camera
  ```

Common flags to know:

- `--task` (`a1` default)
- `--exptid`, `--proj_name` (log structure)
- `--resume`, `--resumeid`, `--checkpoint`
- `--use_camera` (depth branch)
- `--delay` / `--nodelay`
- `--headless`, `--num_envs`, `--seed`

## Playback / evaluation

- Run saved policy:
  ```bash
  python play.py --exptid xxx-xx
  ```
- Distillation/camera run:
  ```bash
  python play.py --exptid yyy-yy --delay --use_camera
  ```
- Headless web viewer mode:
  ```bash
  python play.py --exptid xxx-xx --web
  ```

`play.py` customizes evaluation terrain/domain-rand defaults for demos and optionally loads JIT policies.

## Export for deployment

```bash
python save_jit.py --exptid xxx-xx
```

Exported artifacts go to a `traced/` directory under the corresponding experiment log folder.

---

## 6) If you are new: suggested reading order

1. `README.md` (workflow + command examples)
2. `legged_gym/legged_gym/scripts/train.py` (minimal training entry)
3. `legged_gym/legged_gym/utils/task_registry.py` (how components are assembled)
4. `legged_gym/legged_gym/envs/__init__.py` (which task names are available)
5. `legged_gym/legged_gym/envs/a1/a1_parkour_config.py` (task-specific behavior)
6. `legged_gym/legged_gym/envs/base/legged_robot.py` (runtime mechanics)
7. `rsl_rl/rsl_rl/runners/on_policy_runner.py` + `algorithms/ppo.py` (actual learning loop)

---

## 7) Quick extension checklist

To add a new robot/task:

1. Add robot URDF/resources and a config class inheriting base config.
2. Implement or reuse `LeggedRobot` environment class logic.
3. Register task in `envs/__init__.py`.
4. Tune rewards/domain randomization/control gains in config.
5. Train with `train.py --task your_task` and validate with `play.py`.

---

If you want, I can also produce a second document that is a **function-by-function walkthrough** of `legged_robot.py` with “what to edit for X behavior” notes.
