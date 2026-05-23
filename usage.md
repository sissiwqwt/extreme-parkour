# Usage Guide

## Big Picture

This repository is a parkour locomotion stack built from three layers:

1. `isaacgym/` provides the simulator runtime, Python bindings, assets, and vendor documentation.
2. `legged_gym/` defines the robot environments, terrain/task configs, training and evaluation scripts, logging layout, and robot assets used by this project.
3. `rsl_rl/` provides the reinforcement learning implementation, including PPO, rollout storage, and the policy modules that are saved into checkpoints.

In practice, almost all project-specific work happens in `legged_gym/`, while `rsl_rl/` is the learning backend and `isaacgym/` is the simulator dependency.

For this fork, the main workflow is:

- train on a headless remote server
- save checkpoints under `legged_gym/logs/`
- copy the checkpoints, or the full run directory, to another machine
- run playback there with the same codebase and assets

The important detail is that a checkpoint stores model weights, not the whole environment. The target machine still needs this repository, Isaac Gym, and the robot assets under `legged_gym/resources/`.

## Top-Level Repo Layout

### `README.md`

The upstream quick-start note. It gives the short installation and training/play workflow, but it does not explain the internal composition of the repo in much detail.

### `install.sh`

A compact install recipe mirroring the root README. It shows the expected package installation order:

- install PyTorch
- install Isaac Gym from `isaacgym/python`
- install `rsl_rl`
- install `legged_gym`
- install extra Python packages such as `wandb`, `opencv-python`, `pyfqmr`, and `flask`

### `images/`

Static media used by the README and paper teaser. This folder is informational only and is not part of training or playback.

### `isaacgym/`

This is the bundled Isaac Gym distribution. It includes:

- `python/`: the Python package that exposes Isaac Gym APIs
- `docs/`: vendor documentation and install notes
- `assets/`: simulator assets shipped with Isaac Gym
- `docker/`: vendor Docker helpers

You usually do not edit this folder unless you are debugging the simulator itself. For normal use, it is a dependency layer.

### `legged_gym/`

This is the main project package. It contains:

- environment definitions
- task registration
- training/evaluation/playback scripts
- utilities for config parsing, terrain generation, logging, and web visualization
- robot URDFs and meshes
- generated training logs and checkpoints

If you want to understand or change project behavior, start here.

### `rsl_rl/`

This is the RL backend. It contains:

- PPO update logic
- policy and encoder network definitions
- rollout storage
- on-policy runner logic

If you need to change how training works internally, how checkpoints are saved, or how the model architecture is assembled, this is the layer to inspect.

## What This Fork Actually Trains

The active task registration lives in `legged_gym/legged_gym/envs/__init__.py`.

Right now the important registrations are:

- `"a1"` -> `LeggedRobot` with `A1ParkourCfg()` and `A1ParkourCfgPPO()`
- `"go1"` -> `LeggedRobot` with `Go1RoughCfg()` and `Go1RoughCfgPPO()`

That means the default task name `a1` is not a generic A1 rough-terrain task in this fork. It is explicitly wired to the parkour configuration in `legged_gym/legged_gym/envs/a1/a1_parkour_config.py`.

## `legged_gym/` In Detail

### `legged_gym/setup.py`

Installs the `legged_gym` Python package and declares its main dependencies:

- `isaacgym`
- `rsl-rl`
- `matplotlib`

### `legged_gym/legged_gym/__init__.py`

Defines:

- `LEGGED_GYM_ROOT_DIR`
- `LEGGED_GYM_ENVS_DIR`

Many scripts and utilities use these paths to locate configs, logs, and assets.

### `legged_gym/legged_gym/envs/`

This folder contains the simulation tasks and config classes.

Important parts:

- `base/`: shared environment logic and shared config schema
- `a1/`: A1-specific configuration
- `go1/`, `anymal_*`, `cassie/`: other robot/task definitions included from upstream or related work
- `__init__.py`: task registration

#### `envs/base/`

This is the core environment framework.

Important files:

- `base_config.py`
  - recursively instantiates nested config classes
  - this is why configs are written as class trees rather than plain dictionaries
- `legged_robot_config.py`
  - the main shared config schema
  - defines environment dimensions, observation structure, terrain settings, command ranges, depth camera settings, domain randomization, reward defaults, and PPO-related config containers
- `legged_robot.py`
  - the main robot environment implementation used by the A1 task in this repo
- `base_task.py`
  - shared task infrastructure beneath `LeggedRobot`

If you want to understand observation size, terrain curriculum, camera usage, action delay, or command sampling, `legged_robot_config.py` is one of the most important files in the whole repo.

#### `envs/a1/`

This folder specializes the generic legged robot environment for A1.

Important files:

- `a1_config.py`
  - upstream-style A1 config baseline
- `a1_parkour_config.py`
  - the parkour-specific A1 config used by the default task in this repo

`a1_parkour_config.py` overrides several important things:

- initial base height and default joint angles
- PD control stiffness and damping
- A1 asset path: `resources/robots/a1/urdf/a1.urdf`
- contact penalties and termination rules
- PPO defaults such as entropy coefficient and experiment name

If you are tuning the project’s main behavior, this is one of the first files to edit.

### `legged_gym/legged_gym/utils/`

This folder contains project glue code and helper infrastructure.

Important files:

- `helpers.py`
  - command-line argument definitions
  - config override logic
  - seed setup
  - checkpoint path resolution
  - Isaac Gym device/simulator argument parsing
- `task_registry.py`
  - central registry that maps task names to environment classes and config objects
  - creates environments and PPO runners
  - handles resume/load behavior
- `terrain.py`
  - terrain construction utilities used by the environment
- `math.py`
  - utility math helpers
- `logger.py`
  - logging helpers
- `storage.py`
  - helper storage logic
- `webviewer.py`
  - browser-based viewer for headless inspection
- `webviewer.html`
  - HTML frontend used by the web viewer

Two files are especially operationally important:

- `helpers.py`, because it defines the CLI surface
- `task_registry.py`, because it decides how task configs and checkpoints are resolved

### `legged_gym/legged_gym/scripts/`

This folder contains the operational entry points.

#### `train.py`

This is the main training entry point.

What it does:

- forces `args.headless = True`
- creates the log directory under `legged_gym/logs/<proj_name>/<exptid>`
- initializes Weights & Biases logging unless disabled
- builds the environment through `task_registry.make_env(...)`
- builds the PPO runner through `task_registry.make_alg_runner(...)`
- starts learning

For remote-server use, this is the main script you care about during training.

#### `play.py`

This is the playback/evaluation entry point. You asked not to include instructions on how to run it, so the main thing to know is what it expects:

- a valid task definition in code
- a run directory under `legged_gym/logs/<proj_name>/<exptid>`
- at least one `model_*.pt` checkpoint in that run directory
- the repository assets and Isaac Gym environment available on the target machine

It can also optionally use:

- `--web` for browser-based visualization
- `--use_jit` to load exported TorchScript artifacts
- `--use_camera` for the vision/distillation path

#### `save_jit.py`

This script exports a trained policy into transferable artifacts under the run’s `traced/` directory.

It does two main things:

- saves a traced TorchScript policy like `<exptid>-<checkpoint>-base_jit.pt`
- saves vision-related weights like `<exptid>-<checkpoint>-vision_weight.pt`

Use this when you want a more deployment-oriented artifact than the raw training checkpoint.

#### `evaluate.py`

Runs many environments and reports aggregate metrics such as:

- mean reward
- mean episode length
- mean number of waypoints reached
- edge violation statistics

This is useful for quantitative comparison of checkpoints, not for the minimum train-transfer-play loop.

#### `visualize.py`

Offline analysis helper for saved numpy feature data. It can generate:

- t-SNE plots
- histograms
- trajectory plots
- state-coverage heatmaps
- scatter plots

This is optional analysis tooling.

#### `fetch.py`

Remote utility script for copying a checkpoint from another machine via `ssh` and `rsync`. It is specific to the original author’s machine naming and paths, so treat it as a custom ops helper rather than general infrastructure.

#### `play_copy.py`

An alternate playback-related script present in the repo. It is not the primary path and does not appear central to the main workflow.

### `legged_gym/resources/`

This folder contains simulator assets required by the environments:

- `robots/`
  - URDFs
  - meshes
  - textures
  - per-robot licenses
- `actuator_nets/`
  - learned actuator network assets used by some robot setups

For the main A1 path, the important part is `resources/robots/a1/`, especially the URDF and meshes. These files must exist on the machine where playback runs.

### `legged_gym/logs/`

This is generated output, not source code. It contains training runs and logging artifacts.

Observed structure in this repo:

- `legged_gym/logs/parkour_new/<run_name>/model_*.pt`
- `legged_gym/logs/parkour_test/<run_name>/model_*.pt`
- `legged_gym/logs/wandb/`

For example:

- `legged_gym/logs/parkour_new/02-train-base/model_10000.pt`

This folder is critical operationally because it contains the checkpoints you will transfer off the headless server.

### `legged_gym/legged_gym/tests/`

Contains `test_env.py`, a minimal environment smoke test that instantiates the selected task and steps it forward with zero actions.

This is useful as a sanity check for environment setup.

### Duplicate config copies under `legged_gym/legged_gym/scripts/legged_gym/...`

There are duplicate-looking config files inside:

- `legged_gym/legged_gym/scripts/legged_gym/envs/base/`
- `legged_gym/legged_gym/scripts/legged_gym/envs/a1/`

These do not appear to be the main source of truth. The actual runtime imports point to `legged_gym/legged_gym/envs/...`. Treat the copies under `scripts/legged_gym/...` as leftovers or auxiliary copies unless you have a specific reason to use them.

## `rsl_rl/` In Detail

### `rsl_rl/setup.py`

Installs the RL package and declares its Python dependencies such as PyTorch and NumPy.

### `rsl_rl/rsl_rl/algorithms/`

Contains the RL update algorithm implementation.

Important file:

- `ppo.py`
  - PPO optimization logic used during training

### `rsl_rl/rsl_rl/runners/`

Contains the training runner orchestration.

Important file:

- `on_policy_runner.py`
  - builds the actor-critic, estimator, and optional depth modules
  - owns the main training loop
  - saves checkpoints
  - loads checkpoints for resume/playback

This file is one of the most important backend files in the repo.

Checkpoint contents written by `OnPolicyRunner.save()` include:

- `model_state_dict`
- `estimator_state_dict`
- `optimizer_state_dict`
- `iter`
- `infos`

If depth/vision training is enabled, checkpoints also include:

- `depth_encoder_state_dict`
- `depth_actor_state_dict`

That is why raw checkpoints are enough to reconstruct a policy on another machine, as long as the code and assets are present there too.

### `rsl_rl/rsl_rl/modules/`

Network building blocks live here.

Important files:

- `actor_critic.py`
  - actor-critic policy definitions
- `actor_critic_recurrent.py`
  - recurrent variants
- `estimator.py`
  - estimator module used alongside the actor-critic
- `depth_backbone.py`
  - depth/vision encoder components

If you want to change policy architecture, encoder shape, or vision model internals, this is the place.

### `rsl_rl/rsl_rl/storage/`

Contains rollout buffer logic.

Important file:

- `rollout_storage.py`

### `rsl_rl/rsl_rl/env/`

Contains the vector-environment interface abstraction.

Important file:

- `vec_env.py`

This is the interface expected by the runner and algorithm layer.

### `rsl_rl/rsl_rl/utils/`

Utility helpers used by the RL backend.

## Headless Remote-Server Workflow

Since you train on a remote headless server and play elsewhere, the repo should be used with a clear separation between training-time artifacts and runtime dependencies.

### What happens on the remote server

The remote server is mainly used for:

- running training
- generating checkpoints
- optionally inspecting runs through the browser-based web viewer
- optionally exporting JIT artifacts

The code already supports headless operation well:

- `train.py` forces headless mode
- `play.py` supports `--web`
- `webviewer.py` serves frames through Flask for remote viewing

### What you should transfer to the other machine

There are two practical choices.

#### Minimum transfer

Copy the specific checkpoint file you want to use:

- `legged_gym/logs/<proj_name>/<run_name>/model_<N>.pt`

This is the smallest unit needed for raw-checkpoint playback.

#### Recommended transfer

Copy the whole run directory:

- `legged_gym/logs/<proj_name>/<run_name>/`

This is simpler because it preserves:

- all saved checkpoints for that run
- later checkpoint selection flexibility
- any `traced/` export folder you create afterward

#### If you export TorchScript

Also copy:

- `legged_gym/logs/<proj_name>/<run_name>/traced/`

This contains the JIT-exported artifacts created by `save_jit.py`.

### What you do not need to transfer

You normally do not need:

- `legged_gym/logs/wandb/`
- unrelated run directories
- README images
- training-only analysis outputs unless you specifically want them

### What must already exist on the playback machine

The destination machine still needs:

- this repository
- the installed `legged_gym` package
- the installed `rsl_rl` package
- Isaac Gym installed from `isaacgym/python`
- robot assets under `legged_gym/resources/`

The reason is simple: the checkpoint contains neural network weights, but playback still reconstructs the model and environment from source code.

## Which Files To Edit For Common Tasks

### Change the parkour task behavior

Edit:

- `legged_gym/legged_gym/envs/a1/a1_parkour_config.py`
- sometimes `legged_gym/legged_gym/envs/base/legged_robot_config.py`

Use this for:

- terrain proportions
- command ranges
- rewards
- control gains
- base state and robot-specific behavior

### Change shared environment logic

Edit:

- `legged_gym/legged_gym/envs/base/legged_robot.py`
- `legged_gym/legged_gym/envs/base/base_task.py`

Use this for:

- observation computation
- reset logic
- stepping behavior
- reward plumbing

### Change command-line arguments or run-time overrides

Edit:

- `legged_gym/legged_gym/utils/helpers.py`

Use this for:

- adding CLI flags
- changing default task names
- changing how config values are overridden from arguments
- changing checkpoint lookup behavior

### Change task registration

Edit:

- `legged_gym/legged_gym/envs/__init__.py`

Use this for:

- remapping task names
- changing which config class is attached to `"a1"`
- registering new tasks

### Change PPO training internals or checkpoint structure

Edit:

- `rsl_rl/rsl_rl/runners/on_policy_runner.py`
- `rsl_rl/rsl_rl/algorithms/ppo.py`
- files under `rsl_rl/rsl_rl/modules/`

Use this for:

- model architecture
- optimizer/update behavior
- save frequency
- resume/load details
- depth/distillation model handling

### Change export packaging

Edit:

- `legged_gym/legged_gym/scripts/save_jit.py`

Use this for:

- naming exported files
- traced model inputs
- what gets included in `traced/`

## Minimal Mental Model

If you only want the shortest operational understanding of this repo, it is this:

- `isaacgym/` is the simulator
- `legged_gym/` is the task/environment/project layer
- `rsl_rl/` is the PPO learning layer
- the default `a1` task is a parkour-configured A1 environment
- training writes checkpoints to `legged_gym/logs/<proj_name>/<run_name>/`
- those checkpoints can be copied to another machine
- the other machine still needs the repo code, Isaac Gym, and robot assets to use the checkpoint

That is the core structure the rest of the repo builds on.
