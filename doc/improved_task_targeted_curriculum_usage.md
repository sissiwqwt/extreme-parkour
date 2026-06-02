# Improved Task-targeted Terrain Curriculum

本文档说明本次新增的 improved task-targeted curriculum, 仅覆盖 teacher RL 训练阶段的 terrain curriculum, 不涉及 heading 或 vision distillation。

## 修改概况

改动保留了原始 non-TT terrain curriculum 路径；只有在 `curriculum=True` 且 `task_targeted_curriculum=True` 时才进入 TTC 逻辑。新增开关全部关闭时，旧 TTC 的窗口、最小样本数、阈值和静态 terrain column 分配行为保持不变。

主要实现位置：

- `legged_gym/legged_gym/utils/task_targeted_curriculum.py`: 环形 buffer 统计、动态 effective window、暂停状态更新、任务采样权重。
- `legged_gym/legged_gym/envs/base/legged_robot.py`: TTC effective 参数、pause/resume、reset 时按任务权重重新选择 terrain column、日志、checkpoint state。
- `legged_gym/legged_gym/envs/base/legged_robot_config.py`: 默认配置。
- `legged_gym/legged_gym/utils/helpers.py`: CLI 覆盖参数。
- `rsl_rl/rsl_rl/runners/on_policy_runner.py`: teacher RL 每个 iteration 向环境传入当前训练 iteration。

## 新增策略

1. 动态窗口：

早期使用较短窗口，后期线性增大窗口。buffer 容量取 `task_curriculum_window`、`window_start`、`window_end` 的最大值，计算成功率时只读取最近 `effective_window` 个样本，不频繁重建 buffer。

2. 动态最小样本数：

早期降低每个任务触发难度更新所需的新样本数，后期线性增加到更稳定的值。

3. 动态阈值：

早期 `up_threshold` 更低、`down_threshold` 更高，使任务更容易升降难度；后期回到保守阈值，降低震荡。

4. 优先任务采样和 solved-task pause：

每个任务维护窗口成功率。成功率高于 `pause_success_threshold` 时进入 paused 状态；低于 `resume_success_threshold` 时恢复。reset 时根据任务困难度 `1 - success_rate` 计算采样权重，paused 任务默认保留 `min_sampling_weight`，用于避免完全遗忘并继续监控。

5. 滞后任务高难度噪声：

每轮 TTC 更新后，以 active task 中窗口成功率最高的任务作为参照。如果某任务成功率落后至少 `task_curriculum_lag_success_gap=0.20`，则标记为 lagging task。下一次 reset 采到该任务时，有 `task_curriculum_lag_level_noise_prob=0.50` 的概率把本 episode 的 terrain row 临时提高 `task_curriculum_lag_level_noise_levels=1` 级。该提升只影响当前 reset 样本，不修改任务持久 curriculum level。

当前 terrain map 仍然静态生成；优先采样通过 reset 时为 env 重新分配已有 terrain column 实现，不重建 height map。

## 默认配置

配置位于 `LeggedRobotCfg.terrain`：

```python
task_curriculum_dynamic_window = True
task_curriculum_window_start = 80
task_curriculum_window_end = 300
task_curriculum_window_warmup_iters = 3000

task_curriculum_dynamic_min_samples = True
task_curriculum_min_samples_start = 20
task_curriculum_min_samples_end = 60
task_curriculum_min_samples_warmup_iters = 3000

task_curriculum_dynamic_thresholds = True
task_curriculum_up_threshold_start = 0.60
task_curriculum_up_threshold_end = 0.75
task_curriculum_down_threshold_start = 0.45
task_curriculum_down_threshold_end = 0.35
task_curriculum_threshold_warmup_iters = 3000

task_curriculum_prioritized_sampling = True
task_curriculum_pause_solved_tasks = True
task_curriculum_pause_success_threshold = 0.95
task_curriculum_resume_success_threshold = 0.90
task_curriculum_min_sampling_weight = 0.02
task_curriculum_priority_alpha = 1.0

task_curriculum_lagged_level_noise = True
task_curriculum_lag_success_gap = 0.20
task_curriculum_lag_level_noise_prob = 0.50
task_curriculum_lag_level_noise_levels = 1
```

旧基础 TTC 参数仍然保留：

```python
task_curriculum_window = 200
task_curriculum_min_samples = 50
task_curriculum_up_threshold = 0.75
task_curriculum_down_threshold = 0.35
```

## 使用方法

默认启用 improved TTC：

```bash
cd legged_gym/legged_gym/scripts
python train.py --exptid improved-ttc-001 --proj_name parkour_ttc --device cuda:0 --headless
```

关闭所有新增机制，回到旧 TTC：

```bash
python train.py \
  --exptid old-ttc-ablation \
  --device cuda:0 \
  --headless \
  --task_curriculum_dynamic_window False \
  --task_curriculum_dynamic_min_samples False \
  --task_curriculum_dynamic_thresholds False \
  --task_curriculum_prioritized_sampling False \
  --task_curriculum_pause_solved_tasks False
```

关闭 task-targeted curriculum，回到原始 per-env terrain curriculum：

```bash
python train.py --exptid nontt --device cuda:0 --headless --task_targeted_curriculum False
```

## 常用 CLI 覆盖

示例：更快 warmup, 更激进地关注困难任务。

```bash
python train.py \
  --exptid improved-ttc-fast \
  --device cuda:0 \
  --headless \
  --task_curriculum_window_start 60 \
  --task_curriculum_window_end 240 \
  --task_curriculum_window_warmup_iters 2000 \
  --task_curriculum_min_samples_start 16 \
  --task_curriculum_min_samples_end 50 \
  --task_curriculum_min_samples_warmup_iters 2000 \
  --task_curriculum_priority_alpha 1.5 \
  --task_curriculum_min_sampling_weight 0.01
```

示例：关闭滞后任务高难度噪声，或调强该机制。

```bash
python train.py \
  --exptid improved-ttc-no-lag-noise \
  --device cuda:0 \
  --headless \
  --task_curriculum_lagged_level_noise False

python train.py \
  --exptid improved-ttc-strong-lag-noise \
  --device cuda:0 \
  --headless \
  --task_curriculum_lag_success_gap 0.15 \
  --task_curriculum_lag_level_noise_prob 0.75 \
  --task_curriculum_lag_level_noise_levels 2
```

## 日志字段

新增全局日志：

- `terrain_task_effective_window`
- `terrain_task_effective_min_samples`
- `terrain_task_effective_up_threshold`
- `terrain_task_effective_down_threshold`
- `terrain_task_sampling_weight_mean`
- `terrain_task_num_paused`
- `terrain_task_num_lagging`
- `terrain_task_lag_boosted_envs`

新增逐任务日志：

- `terrain_task_<id>_paused`
- `terrain_task_<id>_lagging`
- `terrain_task_<id>_sampling_weight`
- `terrain_task_<id>_level`
- `terrain_task_<id>_success_rate`
- `terrain_task_<id>_samples`
- `terrain_task_<id>_new_samples`

## Checkpoint Resume

TTC state 会随 checkpoint 保存，包括任务 level、成功率 buffer、sample counters、paused flags 和 sampling weights。旧 checkpoint 没有新字段时会默认初始化新字段；如果窗口容量不同，会拷贝可重叠的历史样本并初始化新增 slot。

恢复训练示例：

```bash
python train.py \
  --exptid improved-ttc-resume \
  --device cuda:0 \
  --headless \
  --resume \
  --resumeid improved-ttc-001
```

建议 resume 时保持 `num_rows`、`num_cols`、`terrain_dict` 等 terrain layout 配置一致；如果任务数量不匹配，代码会跳过 terrain curriculum state 加载。
