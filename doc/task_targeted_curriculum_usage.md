# Task-targeted Curriculum Usage

本文档说明如何开启、关闭和调参 task-targeted curriculum 训练策略，并给出一些 bash 脚本示例。

## 配置位置

主要配置在：

```text
extreme-parkour/legged_gym/legged_gym/envs/base/legged_robot_config.py
```

相关字段位于 `LeggedRobotCfg.terrain`：

```python
curriculum = True
task_targeted_curriculum = True
task_curriculum_window = 200
task_curriculum_min_samples = 50
task_curriculum_up_threshold = 0.75
task_curriculum_down_threshold = 0.35
```

含义简述：

- `curriculum`: 是否启用 terrain curriculum 总开关。
- `task_targeted_curriculum`: 是否启用按任务类型独立统计、独立调节的 curriculum。
- `task_curriculum_window`: 每个任务滑动窗口中保留的最近 episode 样本数。
- `task_curriculum_min_samples`: 每个任务累计多少个新样本后尝试更新一次难度。
- `task_curriculum_up_threshold`: 窗口完成率高于该阈值时升高该任务难度。
- `task_curriculum_down_threshold`: 窗口完成率低于该阈值时降低该任务难度。

## 开启策略

默认推荐配置：

```python
curriculum = True
task_targeted_curriculum = True
```

训练命令示例：

```bash
cd extreme-parkour/legged_gym/legged_gym/scripts
python train.py --exptid ttc-on-001 --device cuda:0 --headless
```

## 关闭 task-targeted curriculum

如果只想使用原始 per-env terrain curriculum：

```python
curriculum = True
task_targeted_curriculum = False
```

如果想完全关闭 terrain curriculum：

```python
curriculum = False
```

关闭 task-targeted curriculum 后，代码会回退到原来的 `_update_terrain_curriculum()` 逻辑：每个环境在 reset 时根据本 episode 表现即时升降自己的 `terrain_level`。

## 调参建议

更平滑、保守：

```python
task_curriculum_window = 400
task_curriculum_min_samples = 100
task_curriculum_up_threshold = 0.8
task_curriculum_down_threshold = 0.3
```

更快推进：

```python
task_curriculum_window = 100
task_curriculum_min_samples = 25
task_curriculum_up_threshold = 0.7
task_curriculum_down_threshold = 0.4
```

调参时建议只同时改一到两个字段，并通过 wandb / console 观察：

- `terrain_task_level`: active task 的平均难度。
- `terrain_task_success_rate`: active task 的平均窗口完成率。
- `terrain_task_<id>_level`: 单个任务的难度。
- `terrain_task_<id>_success_rate`: 单个任务的窗口完成率。
- `terrain_task_<id>_samples`: 单个任务当前滑动窗口内的样本数。
- `terrain_task_<id>_new_samples`: 单个任务距离下一次难度判断已累计的新样本数。

## Bash demo: 开启训练

保存为 `scripts/train_ttc_on.sh`：

```bash
#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../extreme-parkour/legged_gym/legged_gym/scripts"

python train.py \
  --exptid ttc-on-001 \
  --proj_name parkour_ttc \
  --device cuda:0 \
  --headless
```

## Bash demo: 临时关闭 task-targeted curriculum

该脚本会临时把 `task_targeted_curriculum = False`，训练结束后自动恢复原配置。适合做 ablation。

保存为 `scripts/train_ttc_off_once.sh`：

```bash
#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CFG="$ROOT/extreme-parkour/legged_gym/legged_gym/envs/base/legged_robot_config.py"
BACKUP="$CFG.bak.ttc"

cp "$CFG" "$BACKUP"
restore_config() {
  mv "$BACKUP" "$CFG"
}
trap restore_config EXIT

cd "$ROOT"

python - <<'PY'
from pathlib import Path

cfg = Path("extreme-parkour/legged_gym/legged_gym/envs/base/legged_robot_config.py")
text = cfg.read_text()
text = text.replace("task_targeted_curriculum = True", "task_targeted_curriculum = False")
cfg.write_text(text)
PY

cd "$ROOT/extreme-parkour/legged_gym/legged_gym/scripts"
python train.py \
  --exptid ttc-off-001 \
  --proj_name parkour_ttc_ablation \
  --device cuda:0 \
  --headless
```

## Bash demo: 恢复训练

task-targeted curriculum 的状态会随 checkpoint 保存和恢复。恢复训练时使用原项目的 resume 参数即可：

```bash
cd extreme-parkour/legged_gym/legged_gym/scripts

python train.py \
  --exptid ttc-on-resume-001 \
  --proj_name parkour_ttc \
  --device cuda:0 \
  --headless \
  --resume \
  --resumeid ttc-on-001
```

注意：恢复时建议保持 terrain 相关配置一致，尤其是 `num_rows`、`num_cols`、`terrain_dict` 和 `task_curriculum_window`。如果任务数量或窗口大小不匹配，代码会跳过 curriculum state 的恢复，以避免把旧状态错误套到新的地形布局上。
