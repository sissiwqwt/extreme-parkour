# Extreme Parkour Isaac Lab Port

This directory is the migration workspace for running the project on Isaac Lab.
It is intentionally separate from `legged_gym/` until parity is reached.

## Expected Layout

Add this directory to `PYTHONPATH` when launching from Isaac Lab:

```bash
export PYTHONPATH=/path/to/extreme-parkour-lzm/isaac_lab_ext:$PYTHONPATH
```

Then import the task package before training:

```bash
python -c "import extreme_parkour_lab.tasks.direct.extreme_parkour"
```

A typical Isaac Lab RSL-RL launch will look like:

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Extreme-Parkour-A1-Direct-v0 \
  --num_envs 4096
```

The current scaffold is not full parity yet. It establishes the Isaac Lab task
shape and ports the base proprioceptive observation/reward path. The remaining
large items are parkour trimesh import, task-targeted terrain curriculum, and
depth-camera distillation.

