# Heading Model Refactor

本文档记录本次 heading model 架构调整、当前结果和运行方法。

## 目标

本次修改的目标是降低 action distillation 阶段对 heading predictor 的污染：

- 默认训练仍然不启用 heading model；
- 不带 `--enable_heading_model` 时保持原始 depth distillation 路径；
- 启用 heading model 后，将原先单体 `RecurrentDepthBackbone` 拆成共享视觉主干和两个独立 head；
- heading pretrain 阶段训练 heading 相关参数；
- action distillation 阶段冻结 heading 相关参数，避免 action loss 改坏已经预训练好的 heading 表征。

## 架构变化

原结构：

```text
DepthOnlyFCBackbone58x87
  -> combination_mlp
  -> GRU
  -> output_mlp: 512 -> 32 + heading_dim
```

新结构：

```text
depth image + proprioception
        |
        v
VisualStudentBackbone
  - DepthOnlyFCBackbone58x87
  - combination_mlp
  - GRU
        |
        v
shared feature, 512 dim
        |
        +--> DepthLatentHead
        |       -> depth_latent, 32 dim
        |
        +--> HeadingPredictorHead
                -> heading_pred, heading_dim dim
```

外部接口保持兼容：

```python
return torch.cat((depth_latent, heading_pred), dim=-1)
```

因此 runner、play、evaluate 中原有的 split 逻辑仍然可以使用。

## 默认行为

默认配置仍然是：

```python
enable_heading_model = False
heading_pretrain_iters = 0
```

不传 `--enable_heading_model` 时：

- encoder 输出维度仍按 `32 + 2` 处理；
- 后 2 维仍作为 yaw delta prediction；
- action distillation loss 仍为：

```text
L = action_loss_weight * L_action + heading_loss_weight * L_yaw
```

也就是说，默认训练方式仍然是不带 body-frame heading model 的原始兼容训练。

## 启用 Heading Model 后的行为

传入 `--enable_heading_model` 后：

- `heading_dim = 4`；
- heading label 为：

```text
[
  cos(delta_yaw),
  sin(delta_yaw),
  cos(delta_next_yaw),
  sin(delta_next_yaw)
]
```

- actor 接口仍保持不变；
- 进入 actor 前，将 4 维 body-frame vector 转回 `obs[:, 6:8]`：

```python
delta_yaw = atan2(sin_current, cos_current)
delta_next_yaw = atan2(sin_next, cos_next)
```

## 参数隔离策略

新增配置：

```python
freeze_backbone_during_action_distillation = True
```

该配置只在 `enable_heading_model=True` 时生效。

### Heading Pretrain 阶段

条件：

```text
local_iteration < heading_pretrain_iters
```

训练参数：

```text
VisualStudentBackbone + HeadingPredictorHead
```

冻结参数：

```text
DepthLatentHead + depth_actor
```

loss：

```text
L = heading_loss_weight * L_heading
```

### Action Distillation 阶段

训练参数：

```text
DepthLatentHead + depth_actor
```

冻结参数：

```text
VisualStudentBackbone + HeadingPredictorHead
```

loss：

```text
L = action_loss_weight * L_action
```

此阶段仍会计算 heading loss 用于日志，但不会用它反向更新 heading predictor。

## 修改文件

- `rsl_rl/rsl_rl/modules/depth_backbone.py`
  - 新增 `VisualStudentBackbone`；
  - 新增 `DepthLatentHead`；
  - 新增 `HeadingPredictorHead`；
  - `RecurrentDepthBackbone` 改为组合上述三个模块；
  - 保持输出格式为 `depth_latent + heading_pred`；
  - 新增 heading/action 参数分组和 trainable 控制方法；
  - 新增旧版 `output_mlp` state dict 到新双 head 结构的兼容映射。

- `rsl_rl/rsl_rl/algorithms/ppo.py`
  - heading model 开启时使用独立 optimizer 参数组；
  - heading pretrain optimizer 更新 `VisualStudentBackbone + HeadingPredictorHead`；
  - action distillation optimizer 更新 `DepthLatentHead + depth_actor`；
  - 新增 `set_heading_pretrain_mode()` 控制冻结策略；
  - heading model 开启时，action 阶段不再用 heading loss 反传更新参数。

- `rsl_rl/rsl_rl/runners/on_policy_runner.py`
  - 每个 vision iteration 根据当前阶段调用 `set_heading_pretrain_mode()`。

- `legged_gym/legged_gym/envs/base/legged_robot_config.py`
  - 新增 `freeze_backbone_during_action_distillation = True`；
  - 默认仍然 `enable_heading_model = False`。

- `legged_gym/legged_gym/scripts/save_jit.py`
  - 支持从新旧两种 depth encoder state dict 中推断 heading dim。

- `legged_gym/legged_gym/utils/task_registry.py`
  - 在 resume / play / evaluate / evaluation 构造 runner 之前检查 checkpoint；
  - 若 checkpoint 中的 depth encoder 是 `32 + 4` 输出，会自动启用 `enable_heading_model=True` 和 `heading_dim=4`；
  - 若 checkpoint 中的 depth encoder 是 `32 + 2` 输出，且命令行没有显式启用 heading，则保持默认非 heading 路径。

## 验证结果

已完成轻量语法检查：

```bash
python -m py_compile \
  legged_gym/legged_gym/utils/task_registry.py \
  legged_gym/legged_gym/scripts/evaluation.py \
  legged_gym/legged_gym/scripts/play.py \
  legged_gym/legged_gym/scripts/evaluate.py \
  rsl_rl/rsl_rl/modules/depth_backbone.py \
  rsl_rl/rsl_rl/algorithms/ppo.py \
  rsl_rl/rsl_rl/runners/on_policy_runner.py \
  legged_gym/legged_gym/scripts/save_jit.py \
  legged_gym/legged_gym/envs/base/legged_robot_config.py
```

结果：通过。

尚未运行 Isaac Gym 训练或评估，因为完整仿真训练耗时较长。

## Checkpoint 兼容性

当前普通 `.pt` checkpoint 的加载路径已经支持自动识别：

- 旧版 depth checkpoint：
  - old monolithic key：`output_mlp.0.weight`；
  - 输出维度 `34 = 32 + 2`；
  - 不传 `--enable_heading_model` 时按原始 yaw delta 路径构造。

- 新版 heading checkpoint：
  - new split-head key：`heading_predictor_head.output_mlp.0.weight`；
  - 或旧 monolithic `output_mlp.0.weight` 但输出维度为 `36 = 32 + 4`；
  - 即使忘记传 `--enable_heading_model`，`task_registry.make_alg_runner()` 也会在构造 runner 前自动启用 heading model。

因此下面这些脚本加载普通 checkpoint 时兼容开 heading 和不开 heading：

```text
legged_gym/legged_gym/scripts/play.py
legged_gym/legged_gym/scripts/evaluate.py
legged_gym/legged_gym/scripts/evaluation.py
```

注意：`play.py --use_jit` 走的是 traced policy 路径，不是普通 `.pt` checkpoint 加载路径；JIT 部署仍需要部署侧按导出的 `depth_heading_dim` 做对应处理。

## Smoke Test 更新

`legged_gym/legged_gym/scripts/smoke_test_remote_pipeline.sh` 已按新 heading model 兼容路径更新：

- 保留原来的 heading-off student 训练；
- 保留显式 `--enable_heading_model` 的 heading-on student 训练；
- `play` 模式新增 `play_student_heading_on_auto_detect_headless`，加载 heading checkpoint 时故意不传 `--enable_heading_model`，用于验证 checkpoint 自动识别；
- 新增 `eval` 模式，调用 `evaluation.py` 检查：
  - heading-off depth checkpoint；
  - heading-on checkpoint + 显式 `--enable_heading_model`；
  - heading-on checkpoint + 自动识别；
- `all` 模式现在会依次运行 train、play、eval、web、video。

可单独运行 evaluation smoke：

```bash
bash legged_gym/legged_gym/scripts/smoke_test_remote_pipeline.sh eval
```

可调小评估规模：

```bash
EVAL_EPISODES=8 EVAL_MAX_STEPS=200 bash legged_gym/legged_gym/scripts/smoke_test_remote_pipeline.sh eval
```

## 运行方法

### 原始默认训练

不启用 heading model：

```bash
python legged_gym/scripts/train.py \
  --task a1 \
  --use_camera \
  --resume \
  --resumeid TEACHER_OR_BASE_RUN_ID \
  --exptid depth_baseline \
  --proj_name parkour_heading
```

该命令保持原始 depth distillation 行为。

## Heading-only 实验命令

这一组命令用于只比较 heading model 的影响：

```text
curriculum = True
task_targeted_curriculum = False
wandb = enabled
```

注意不要添加 `--no_wandb`，否则 wandb 会被关闭。运行前请确认远端主机已经完成 wandb 登录，或设置了 `WANDB_API_KEY`。

把下面命令中的 `TEACHER_OR_BASE_RUN_ID` 替换成 teacher/base checkpoint 所在 run 名。

### 1. 测速 Baseline：不开 Heading

建议先跑 50 iteration 看常规参数速度：

```bash
python legged_gym/legged_gym/scripts/train.py \
  --task a1 \
  --device cuda:0 \
  --rl_device cuda:0 \
  --proj_name parkour_heading \
  --exptid speed_heading_off_50 \
  --use_camera \
  --resume \
  --resumeid TEACHER_OR_BASE_RUN_ID \
  --checkpoint -1 \
  --curriculum True \
  --task_targeted_curriculum False \
  --max_iterations 50
```

### 2. 测速 Heading：启用新 Heading

```bash
python legged_gym/legged_gym/scripts/train.py \
  --task a1 \
  --device cuda:0 \
  --rl_device cuda:0 \
  --proj_name parkour_heading \
  --exptid speed_heading_on_50_pre10 \
  --use_camera \
  --enable_heading_model \
  --heading_pretrain_iters 10 \
  --heading_loss_weight 1.0 \
  --action_loss_weight 1.0 \
  --resume \
  --resumeid TEACHER_OR_BASE_RUN_ID \
  --checkpoint -1 \
  --curriculum True \
  --task_targeted_curriculum False \
  --max_iterations 50
```

### 3. 正式训练 Baseline：不开 Heading

测速确认正常后，用同样 curriculum 设置拉长训练：

```bash
python legged_gym/legged_gym/scripts/train.py \
  --task a1 \
  --device cuda:0 \
  --rl_device cuda:0 \
  --proj_name parkour_heading \
  --exptid baseline_heading_off_5000 \
  --use_camera \
  --resume \
  --resumeid TEACHER_OR_BASE_RUN_ID \
  --checkpoint -1 \
  --curriculum True \
  --task_targeted_curriculum False \
  --max_iterations 5000
```

### 4. 正式训练 Heading：启用新 Heading

```bash
python legged_gym/legged_gym/scripts/train.py \
  --task a1 \
  --device cuda:0 \
  --rl_device cuda:0 \
  --proj_name parkour_heading \
  --exptid heading_on_5000_pre1000 \
  --use_camera \
  --enable_heading_model \
  --heading_pretrain_iters 1000 \
  --heading_loss_weight 1.0 \
  --action_loss_weight 1.0 \
  --resume \
  --resumeid TEACHER_OR_BASE_RUN_ID \
  --checkpoint -1 \
  --curriculum True \
  --task_targeted_curriculum False \
  --max_iterations 5000
```

### 5. 播放 Heading Checkpoint

```bash
python legged_gym/legged_gym/scripts/play.py \
  --task a1 \
  --device cuda:0 \
  --rl_device cuda:0 \
  --use_camera \
  --proj_name parkour_heading \
  --exptid heading_on_5000_pre1000 \
  --checkpoint -1 \
  --enable_heading_model \
  --headless \
  --play_steps 1000
```

`play.py` 会自动识别 heading checkpoint；这里仍显式传 `--enable_heading_model`，方便命令含义更清楚。

### 6. 评估 Heading Checkpoint

```bash
python legged_gym/legged_gym/scripts/evaluation.py \
  --task a1 \
  --device cuda:0 \
  --rl_device cuda:0 \
  --use_camera \
  --proj_name parkour_heading \
  --exptid heading_on_5000_pre1000 \
  --checkpoint -1 \
  --enable_heading_model \
  --policy_type depth \
  --terrain_set effective \
  --eval_episodes 1000
```

### 7. 评估 Baseline Checkpoint

```bash
python legged_gym/legged_gym/scripts/evaluation.py \
  --task a1 \
  --device cuda:0 \
  --rl_device cuda:0 \
  --use_camera \
  --proj_name parkour_heading \
  --exptid baseline_heading_off_5000 \
  --checkpoint -1 \
  --policy_type depth \
  --terrain_set effective \
  --eval_episodes 1000
```

### 观察指标

训练阶段主要看 wandb：

```text
Perf/total_fps
Perf/collection time
Perf/learning_time
Loss_depth/heading
Loss_depth/depth_actor
Loss_depth/heading_pretrain
Train/mean_reward
Episode_rew/*
```

评估阶段主要看 `evaluation.py` 输出的 CSV / JSON：

```text
success_rate
fall_rate
mean_mxd
mean_normalized_waypoints
mean_episode_length
mean_edge_violation
mean_heading_loss
```

### 启用新 Heading Model

建议启用时一定设置非零 heading pretrain iterations：

```bash
python legged_gym/scripts/train.py \
  --task a1 \
  --use_camera \
  --resume \
  --resumeid TEACHER_OR_BASE_RUN_ID \
  --exptid heading_refactor_1000 \
  --proj_name parkour_heading \
  --enable_heading_model \
  --heading_pretrain_iters 1000 \
  --heading_loss_weight 1.0 \
  --action_loss_weight 1.0
```

### 评估 Heading Model

评估启用了 heading model 的 checkpoint 时，也需要传 `--enable_heading_model`：

```bash
python legged_gym/scripts/evaluate.py \
  --task a1 \
  --use_camera \
  --exptid heading_refactor_1000 \
  --proj_name parkour_heading \
  --enable_heading_model
```

### 播放 Heading Model

```bash
python legged_gym/scripts/play.py \
  --task a1 \
  --use_camera \
  --exptid heading_refactor_1000 \
  --proj_name parkour_heading \
  --enable_heading_model
```

### 导出 JIT

```bash
python legged_gym/scripts/save_jit.py \
  --exptid heading_refactor_1000 \
  --proj_name parkour_heading \
  --checkpoint -1
```

导出的 vision weight 中仍会记录：

```text
depth_encoder_output_dim
depth_heading_dim
depth_heading_mode
```

部署侧仍按 `depth_heading_dim` 判断：

- `2`：旧 yaw delta 输出；
- `4`：body-frame heading vector，需用 `atan2(sin, cos)` 转回 actor yaw 输入。

## 注意事项

- 如果启用 `--enable_heading_model` 但 `--heading_pretrain_iters 0`，heading predictor 没有预训练，action 阶段又会冻结 heading 相关参数，因此不建议这样运行。
- 新结构的 depth encoder state dict key 与旧版不同；当旧 checkpoint 的输出维度与当前 `heading_dim` 匹配时，会自动把旧 `output_mlp` 拆成 `DepthLatentHead` 和 `HeadingPredictorHead`；若维度不匹配，runner 仍会跳过 depth encoder 权重。
- actor 输入语义没有改变，当前仍是兼容版 B-1，不是 actor 显式接收 heading feature 的 B-2。
