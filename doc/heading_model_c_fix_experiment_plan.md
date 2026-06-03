# Heading Model C 修复与实验计划

本文档记录 `heading-model-C` 训练失败的代码级原因、已完成修复，以及后续推荐实验步骤。

## 背景问题

对比 `heading-model-C` 与原 `heading-model` B 路径时，观察到：

- `heading-model-C` 的 loss 下降，但 reward 没有稳定上升；
- `heading-model-C` 的 reward 震荡明显；
- 从 play / distill play 看，蒸馏后的 depth policy 基本没有学会可用动作，甚至走路也不稳定。

核心原因不是 heading vector 的 `atan2` 推理转换写反，而是训练目标和冻结策略让 depth latent 语义断开了。

## 根因

`depth_actor` 的 actor 输入中，`scandots_latent` 应该与 teacher actor 的 scan encoder latent 同语义。推理和训练时 depth policy 会这样调用 actor：

```python
actions = depth_actor(obs, hist_encoding=True, scandots_latent=depth_latent)
```

但之前 `heading-model-C` 中：

- heading pretrain 阶段只优化 heading loss；
- `depth_latent` 没有对齐 teacher 的 `scandots_latent`；
- action distillation 阶段默认冻结 shared visual backbone；
- 因此 action 阶段只能在一个主要为 heading 任务优化过的固定视觉特征上训练 `DepthLatentHead + depth_actor`。

结果是 heading loss 可以下降，但 actor 收到的 `depth_latent` 不是原 actor 期望的 scan latent 分布，reward 和 play 表现自然会崩。

## 已完成修复

### 1. 恢复 latent alignment loss

文件：

- `rsl_rl/rsl_rl/algorithms/ppo.py`
- `rsl_rl/rsl_rl/runners/on_policy_runner.py`

现在 heading pretrain 和 action distillation 都会计算：

```python
latent_loss = || scandots_latent.detach() - depth_latent ||_2
```

并加入优化目标：

```text
heading pretrain:
  L = heading_loss_weight * L_heading
    + latent_loss_weight * L_latent

action distillation:
  L = action_loss_weight * L_action
    + latent_loss_weight * L_latent
```

非 heading model 的原始路径也会加入 latent loss，默认权重见配置。

### 2. pretrain 阶段训练 DepthLatentHead

文件：

- `rsl_rl/rsl_rl/algorithms/ppo.py`

之前 pretrain 阶段即使计算 latent loss，也不会更新 `DepthLatentHead`。现在 heading pretrain optimizer 包含：

```text
VisualStudentBackbone
HeadingPredictorHead
DepthLatentHead
```

pretrain 阶段冻结 `depth_actor`，但不冻结 `DepthLatentHead`。

### 3. action distillation 阶段默认不冻结 shared backbone

文件：

- `legged_gym/legged_gym/envs/base/legged_robot_config.py`
- `rsl_rl/rsl_rl/algorithms/ppo.py`

默认配置改为：

```python
latent_loss_weight = 1.0
freeze_backbone_during_action_distillation = False
```

这样 action 阶段可以继续调整 shared visual backbone，避免 backbone 被固定在只适合 heading 的特征空间。

### 4. 修正 `delta_yaw_ok` mask

文件：

- `legged_gym/legged_gym/envs/base/legged_robot.py`

之前逻辑：

```python
self.extras["delta_yaw_ok"] = self.delta_yaw < 0.6
```

负的大角度会被错误认为 ok。现在改为 wrap 后取绝对值：

```python
delta_yaw_wrapped = atan2(sin(delta_yaw), cos(delta_yaw))
delta_yaw_ok = abs(delta_yaw_wrapped) < 0.6
```

### 5. 新增命令行 ablation 参数

文件：

- `legged_gym/legged_gym/utils/helpers.py`

新增：

```text
--latent_loss_weight
--freeze_backbone_during_action_distillation
```

用于不用改代码直接做消融实验。

## 推荐实验步骤

### 0. 先做语法检查

```bash
python -m py_compile \
  legged_gym/legged_gym/envs/base/legged_robot.py \
  legged_gym/legged_gym/envs/base/legged_robot_config.py \
  legged_gym/legged_gym/utils/helpers.py \
  rsl_rl/rsl_rl/algorithms/ppo.py \
  rsl_rl/rsl_rl/runners/on_policy_runner.py
```

### 1. 快速 sanity run

建议先跑 50-100 iteration，确认 loss 都有日志且没有 shape / optimizer 错误。

```bash
python legged_gym/legged_gym/scripts/train.py \
  --task a1 \
  --device cuda:0 \
  --rl_device cuda:0 \
  --proj_name parkour_heading \
  --exptid heading_c_fix_smoke_pre10 \
  --use_camera \
  --enable_heading_model \
  --heading_pretrain_iters 10 \
  --heading_loss_weight 1.0 \
  --action_loss_weight 1.0 \
  --latent_loss_weight 1.0 \
  --freeze_backbone_during_action_distillation False \
  --resume \
  --resumeid TEACHER_OR_BASE_RUN_ID \
  --checkpoint -1 \
  --curriculum True \
  --task_targeted_curriculum False \
  --max_iterations 100
```

重点看：

```text
Loss_depth/heading
Loss_depth/latent
Loss_depth/depth_actor
Loss_depth/delta_yaw_ok_percent
Train/mean_reward
```

### 2. 主实验：修复后的 C

```bash
python legged_gym/legged_gym/scripts/train.py \
  --task a1 \
  --device cuda:0 \
  --rl_device cuda:0 \
  --proj_name parkour_heading \
  --exptid heading_c_fix_pre2000_latent1_unfreeze \
  --use_camera \
  --enable_heading_model \
  --heading_pretrain_iters 2000 \
  --heading_loss_weight 1.0 \
  --action_loss_weight 1.0 \
  --latent_loss_weight 1.0 \
  --freeze_backbone_during_action_distillation False \
  --resume \
  --resumeid TEACHER_OR_BASE_RUN_ID \
  --checkpoint -1 \
  --curriculum True \
  --task_targeted_curriculum False \
  --max_iterations 9000
```

### 3. 消融 A：只验证 latent loss

保持 action 阶段冻结 backbone，但加入 latent loss：

```bash
python legged_gym/legged_gym/scripts/train.py \
  --task a1 \
  --device cuda:0 \
  --rl_device cuda:0 \
  --proj_name parkour_heading \
  --exptid heading_c_fix_pre2000_latent1_freeze \
  --use_camera \
  --enable_heading_model \
  --heading_pretrain_iters 2000 \
  --heading_loss_weight 1.0 \
  --action_loss_weight 1.0 \
  --latent_loss_weight 1.0 \
  --freeze_backbone_during_action_distillation True \
  --resume \
  --resumeid TEACHER_OR_BASE_RUN_ID \
  --checkpoint -1 \
  --curriculum True \
  --task_targeted_curriculum False \
  --max_iterations 9000
```

若该实验仍明显差于主实验，说明 action 阶段更新 shared backbone 是必要的。

### 4. 消融 B：验证 latent loss 权重

建议对比：

```text
latent_loss_weight = 0.25
latent_loss_weight = 1.0
latent_loss_weight = 2.0
```

如果 `Loss_depth/latent` 下降但 `Loss_depth/depth_actor` 上升或 reward 变差，说明 latent loss 过强，压制了 action imitation。

### 5. 播放检查

```bash
python legged_gym/legged_gym/scripts/play.py \
  --task a1 \
  --device cuda:0 \
  --rl_device cuda:0 \
  --use_camera \
  --proj_name parkour_heading \
  --exptid heading_c_fix_pre2000_latent1_unfreeze \
  --checkpoint -1 \
  --enable_heading_model \
  --headless \
  --play_steps 1000
```

如果 play 仍然完全不会走，优先检查：

- 是否加载到了正确 checkpoint；
- checkpoint 是否是修复后训练生成的；
- `Loss_depth/latent` 是否实际下降；
- `Loss_depth/depth_actor` 是否下降；
- `Train/mean_reward` 是否在 pretrain 结束后恢复上升。

### 6. 固定地形视频检查

如果使用 `distill_play.py`，注意确认测试地形模块和训练地形一致。当前工作区此前出现过 `terrain_ver2` 与 `terrain` 的切换，这会影响固定地形播放结论。

```bash
python legged_gym/legged_gym/scripts/distill_play.py \
  --task a1 \
  --device cuda:0 \
  --rl_device cuda:0 \
  --use_camera \
  --proj_name parkour_heading \
  --exptid heading_c_fix_pre2000_latent1_unfreeze \
  --checkpoint -1 \
  --enable_heading_model \
  --headless
```

## 判断修复是否有效

修复有效时，预期现象是：

- pretrain 阶段 `Loss_depth/heading` 和 `Loss_depth/latent` 都下降；
- action 阶段 `Loss_depth/depth_actor` 下降；
- pretrain 切换到 action 阶段后 reward 可能短暂掉落，但应逐渐恢复上升；
- play 中至少应先恢复基础行走，再观察 parkour 地形成功率。

如果 heading loss 继续下降但 reward 不上升，下一步优先检查 actor 输入的 `depth_latent` 分布：

```text
mean/std/norm(depth_latent)
mean/std/norm(scandots_latent)
cosine_similarity(depth_latent, scandots_latent)
```

这比继续调 heading loss 更直接。
