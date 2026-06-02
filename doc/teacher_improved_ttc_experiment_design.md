# Teacher Base Improved TTC Experiment Design

本文档设计 teacher RL 训练阶段的 terrain curriculum 对比实验。实验只比较：

- `improved_tt`: 当前新增 improved task-targeted curriculum。
- `nontt`: 原始 per-env terrain curriculum, 即 `task_targeted_curriculum=False`。

不再测试旧版 TT，也不涉及 heading / vision distillation。

## 实验目标

验证 improved TT 是否在训练早期、中期、晚期都优于或至少不弱于 nontt：

- 早期：是否更快覆盖任务、降低样本不足导致的 curriculum 停滞。
- 中期：是否更集中训练困难任务，并让不同任务的 level 推进更均衡。
- 晚期：是否减少已解决任务占用样本，提升困难任务最终成功率与高难度通过率。

## 实验分组

推荐每组至少跑 3 个 seed，资源允许时跑 5 个 seed。

| Group | Curriculum | 说明 |
|---|---|---|
| `improved_tt` | `curriculum=True`, `task_targeted_curriculum=True` | 默认 improved TTC, 包含动态窗口、动态样本数、动态阈值、优先采样、solved pause、lagged level noise |
| `nontt` | `curriculum=True`, `task_targeted_curriculum=False` | 原始 per-env terrain curriculum |

建议 seed：

```text
1, 2, 3
```

若训练不稳定或差异较小，再扩展到：

```text
1, 2, 3, 4, 5
```

## 训练长度与阶段划分

teacher base 使用 `LeggedRobotCfgPPO.runner.max_iterations = 50000`。为了覆盖早中晚期，建议在以下 checkpoint 做评估：

| 阶段 | Iteration | 目的 |
|---|---:|---|
| Early-1 | 500 | 观察动态窗口和低 min samples 是否让 TTC 更早开始更新 |
| Early-2 | 1500 | 观察早期任务成功率、level 分化和 sampling weights |
| Mid-1 | 5000 | 观察动态阈值 warmup 后的任务推进 |
| Mid-2 | 10000 | 观察困难任务是否被更高频采样并追上 |
| Late-1 | 25000 | 观察高难度任务稳定性 |
| Late-2 | 50000 | 最终收敛表现 |

如果算力有限，可先跑短版：

```text
1500, 5000, 10000, 25000
```

## 训练命令

Improved TT:

```bash
cd legged_gym/legged_gym/scripts

python train.py \
  --exptid improved-tt-seed1 \
  --proj_name teacher_ttc_base \
  --device cuda:0 \
  --headless \
  --seed 1
```

Non-TT:

```bash
cd legged_gym/legged_gym/scripts

python train.py \
  --exptid nontt-seed1 \
  --proj_name teacher_ttc_base \
  --device cuda:0 \
  --headless \
  --seed 1 \
  --task_targeted_curriculum False
```

批量跑 seed 时只替换 `--exptid` 和 `--seed`。

## 评估方式

每个 checkpoint 都用相同评估设置，固定 evaluation steps，不训练，只采集 episode 成功率、任务 level、terrain class 和 reset 结果。

建议每个 checkpoint 至少评估：

```text
num_eval_episodes_per_task >= 100
```

如果当前 play/eval 脚本按 step 数运行，则建议：

```text
--eval_steps 20000
```

并保证 improved_tt 与 nontt 使用相同：

- checkpoint iteration
- seed 或 evaluation seed
- env 数量
- terrain layout
- 不开启 camera / heading / vision distillation

## 核心指标

### 1. 全局性能

记录：

- mean episode reward
- mean episode length
- overall success rate
- mean terrain level
- high-level success rate, 例如 `terrain_level >= 7`

预期：

- Early: improved_tt 的 mean terrain level 可能略高或波动更大，但 success rate 不应明显低于 nontt。
- Mid: improved_tt 应更快提升 mean terrain level 和 high-level success rate。
- Late: improved_tt 的高难任务成功率应高于 nontt。

### 2. 逐任务性能

按 `terrain_task_<id>` 记录：

- `terrain_task_<id>_success_rate`
- `terrain_task_<id>_level`
- `terrain_task_<id>_samples`
- task-level eval success rate
- task-level high-level success rate

重点看：

- 最差任务成功率 `min_task_success_rate`
- 任务成功率方差 `std_task_success_rate`
- 任务 level 方差 `std_task_level`
- 最差任务 level `min_task_level`

预期：

- Early: improved_tt 的 `terrain_task_<id>_samples/new_samples` 更快达到更新阈值。
- Mid: 最差任务不应长期落后，`std_task_level` 应低于或接近 nontt。
- Late: `min_task_success_rate` 和困难任务 high-level success rate 应高于 nontt。

### 3. Improved TTC 机制诊断

只对 improved_tt 记录：

- `terrain_task_effective_window`
- `terrain_task_effective_min_samples`
- `terrain_task_effective_up_threshold`
- `terrain_task_effective_down_threshold`
- `terrain_task_sampling_weight_mean`
- `terrain_task_num_paused`
- `terrain_task_num_lagging`
- `terrain_task_lag_boosted_envs`
- `terrain_task_<id>_sampling_weight`
- `terrain_task_<id>_paused`
- `terrain_task_<id>_lagging`

判断机制是否正常：

- Early: `effective_window` 接近 80, `effective_min_samples` 接近 20。
- Around 3000 iters: dynamic 参数应接近最终值。
- Mid/Late: 已解决任务出现 paused, 困难任务 sampling weight 更高。
- 当任务成功率落后最高任务超过 0.2 时，`terrain_task_<id>_lagging=1`, 且 `terrain_task_lag_boosted_envs > 0`。

## 阶段性假设

### Early

比较 checkpoint:

```text
500, 1500
```

假设：

- improved_tt 的任务更新次数更多。
- improved_tt 的 active task 样本数更均衡。
- improved_tt 不会因为窗口过大、min_samples 过高而长期不更新。

主要图：

- iteration vs `terrain_task_effective_window`
- iteration vs `terrain_task_effective_min_samples`
- iteration vs per-task `new_samples`
- iteration vs per-task `level`

### Mid

比较 checkpoint:

```text
5000, 10000
```

假设：

- improved_tt 更关注低成功率任务。
- lagging task 的 boosted samples 会推动滞后任务 level 上升。
- per-task success rate gap 小于 nontt。

主要图：

- per-task success rate bar chart
- per-task level bar chart
- per-task sampling weight bar chart
- `terrain_task_num_lagging` and `terrain_task_lag_boosted_envs`

### Late

比较 checkpoint:

```text
25000, 50000
```

假设：

- improved_tt 的困难任务最终成功率更高。
- improved_tt 的 high-level success rate 更高。
- paused task 仍保持基本成功率，没有明显遗忘。

主要图：

- high-level success rate by task
- final min/mean/std task success rate
- final min/mean/std task level
- paused task success rate over time

## 推荐判据

Improved TT 可认为有效，如果满足多数条件：

- Early: 1500 iters 时，至少一半 active tasks 已完成若干次 level 更新。
- Mid: 10000 iters 时，`min_task_success_rate` 高于 nontt 或 `std_task_success_rate` 低于 nontt。
- Late: 50000 iters 时，overall success rate 不低于 nontt，且 high-level success rate 高于 nontt。
- Late: 最差任务成功率高于 nontt。
- Improved TTC 诊断日志显示 priority sampling、pause、lagged noise 至少在中后期实际触发。

## 结果表模板

| Group | Seed | Iter | Overall SR | Mean Level | High-level SR | Min Task SR | Std Task SR | Min Task Level | Std Task Level |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| improved_tt | 1 | 1500 | | | | | | | |
| nontt | 1 | 1500 | | | | | | | |
| improved_tt | 1 | 10000 | | | | | | | |
| nontt | 1 | 10000 | | | | | | | |
| improved_tt | 1 | 50000 | | | | | | | |
| nontt | 1 | 50000 | | | | | | | |

## 注意事项

- 两组必须使用同一份 terrain layout, 不要在实验中途改 `terrain_dict`, `num_rows`, `num_cols`。
- nontt 不会产出 TTC 逐任务诊断日志，因此 nontt 的逐任务指标应从 evaluation episode 的 terrain class / env class 汇总得到。
- improved_tt 的 lagged level noise 是临时难度扰动，不应把 `terrain_level` 的单次均值波动误读为持久 curriculum level 提升。
- 如果训练资源不足，优先保留 1500、10000、50000 三个评估点，分别代表早期、中期、晚期。
