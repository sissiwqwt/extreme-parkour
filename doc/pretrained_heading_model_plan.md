# Pretrained Heading Model 开发规划

本文档整理 `tech_report.md` 中 **Pretrained Headding Model** 创新的实现思路，目标是在后续开发时快速定位 extreme-parkour 中需要修改的模块，并明确两条可落地路径。

## 1. 创新目标

在 teacher-to-student 视觉蒸馏中显式加入 heading 预测中间步骤：

- teacher 可以利用 goals / waypoint 信息；
- student 当前主要蒸馏 teacher actor 的动作；
- 对 `bean_gap`、`asymmetric_gap`、`parkour_v2` 这类依赖方向精度的地形，显式 heading 表征有助于提升稳定性和可解释性；
- 训练逻辑从“直接学动作”变为“先学朝哪里走，再学怎么输出动作”。

## 2. 当前框架中的相关链路

现有 extreme-parkour 已经存在一个接近的机制：`depth_encoder` 输出 `depth_latent + yaw`，再把 yaw 回填到 student observation 中。

### 2.1 waypoint / yaw label 来源

文件：

- `extreme-parkour/legged_gym/legged_gym/envs/base/legged_robot.py`

关键位置：

- `_update_goals()`
  - 维护当前 waypoint 和下一 waypoint；
  - 计算：
    - `target_pos_rel`
    - `next_target_pos_rel`
    - `target_yaw`
    - `next_target_yaw`
- `compute_observations()`
  - 当前 observation 中包含：
    - `delta_yaw = target_yaw - yaw`
    - `delta_next_yaw = next_target_yaw - yaw`
  - 这两个量进入 `obs[:, 6:8]`。

当前 `obs_buf` 中 heading 相关槽位大致为：

```text
obs[:, 5] = 0 * delta_yaw
obs[:, 6] = delta_yaw
obs[:, 7] = delta_next_yaw
```

### 2.2 depth encoder 当前输出

文件：

- `extreme-parkour/rsl_rl/rsl_rl/modules/depth_backbone.py`

当前结构：

```text
Depth image
   |
DepthOnlyFCBackbone58x87
   |
combination_mlp(depth_feature + proprioception)
   |
GRU
   |
output_mlp -> 32 + 2
```

其中：

- 前 32 维：`depth_latent`
- 后 2 维：当前代码中作为 yaw 预测使用

### 2.3 vision distillation 当前流程

文件：

- `extreme-parkour/rsl_rl/rsl_rl/runners/on_policy_runner.py`

当前 `learn_vision()` 中的核心逻辑：

```python
depth_latent_and_yaw = self.alg.depth_encoder(infos["depth"].clone(), obs_prop_depth)
depth_latent = depth_latent_and_yaw[:, :-2]
yaw = 1.5 * depth_latent_and_yaw[:, -2:]

yaw_buffer_student.append(yaw)
yaw_buffer_teacher.append(obs[:, 6:8])

obs_student = obs.clone()
obs_student[infos["delta_yaw_ok"], 6:8] = yaw.detach()[infos["delta_yaw_ok"]]

actions_student = self.alg.depth_actor(
    obs_student,
    hist_encoding=True,
    scandots_latent=depth_latent,
)
```

也就是说，student actor 的 heading 输入来自 depth encoder 的预测结果。

### 2.4 当前 loss

文件：

- `extreme-parkour/rsl_rl/rsl_rl/algorithms/ppo.py`

当前 `update_depth_actor()` 中：

```python
depth_actor_loss = (actions_teacher_batch.detach() - actions_student_batch).norm(p=2, dim=1).mean()
yaw_loss = (yaw_teacher_batch.detach() - yaw_student_batch).norm(p=2, dim=1).mean()
loss = depth_actor_loss + yaw_loss
```

当前缺点：

- heading / yaw 预测没有独立预训练阶段；
- label 仍是 yaw 标量形式，而不是 body frame 下的方向向量；
- heading head 和 depth latent head 没有被显式拆分；
- action distillation 和 heading supervision 混在同一个 update 中。

## 3. 两条实现路径

建议分两条路径推进：先做兼容版快速验证，再做正式 body-frame heading vector 版。

## 4. 路径 A：兼容版 Heading Pretrain

### 4.1 目标

保持现有 actor 输入维度和语义基本不变，仍使用 `obs[:, 6:8]` 作为 heading 槽位，但将 heading 预测从 action 蒸馏中拆出来，形成明确的预训练阶段。

### 4.2 优点

- 改动小；
- 与现有 teacher / depth actor / checkpoint 更兼容；
- 适合快速验证“先预训练 heading 是否能改善动作蒸馏”；
- 不需要立即修改 `n_proprio`、actor MLP 输入维度和旧策略结构。

### 4.3 训练阶段

阶段 1：Heading 预训练

- 冻结 `depth_actor`；
- 训练 `depth_encoder` 中的视觉主干和 heading 输出；
- label 使用当前 `obs[:, 6:8]`，即 `[delta_yaw, delta_next_yaw]`；
- loss 只使用 `L_heading`。

阶段 2：Action 蒸馏

- 使用预训练好的 heading predictor；
- 将预测 heading 回填到 `obs_student[:, 6:8]`；
- 使用 `depth_latent` 作为 `scandots_latent` 输入 actor；
- 蒸馏 teacher action。

阶段 3：可选联合微调

- 同时优化：
  - `L_heading`
  - `L_action`
  - 可选 `L_latent`

### 4.4 需要修改的文件

#### `depth_backbone.py`

可先保持输出 `32 + 2` 不变，也可以轻量拆分为：

```text
shared feature
   |------ depth_latent_head -> 32
   |------ heading_head      -> 2
```

推荐先拆 head，但返回格式仍保持兼容，例如：

```python
return torch.cat([depth_latent, heading_pred], dim=-1)
```

这样可以少改 `play.py`、`evaluate.py` 和旧训练逻辑。

#### `on_policy_runner.py`

在 `learn_vision()` 中增加阶段控制：

- `heading_pretrain_iters`
- `joint_finetune_iters`
- 当前 iteration 决定使用哪种 update。

需要新增或调整 buffer：

- `heading_pred_buffer`
- `heading_label_buffer`
- `actions_student_buffer`
- `actions_teacher_buffer`

#### `ppo.py`

新增：

```python
update_heading_predictor(...)
update_depth_actor(...)
update_depth_joint(...)
```

或至少将现有 `update_depth_actor()` 内部按权重拆开：

```python
loss = heading_loss_weight * heading_loss + action_loss_weight * depth_actor_loss
```

#### `legged_robot_config.py`

在 `class depth_encoder` 中增加配置：

```python
heading_pretrain_iters = 1000
joint_finetune_iters = 0
heading_loss_weight = 1.0
action_loss_weight = 1.0
latent_loss_weight = 0.0
heading_mode = "yaw_delta"
heading_dim = 2
freeze_actor_during_heading_pretrain = True
```

### 4.5 路径 A 的验证指标

- `Loss_depth/heading`
- `Loss_depth/depth_actor`
- `Loss_depth/delta_yaw_ok_percent`
- `Train/mean_reward`
- 分地形评估：
  - `bean_gap`
  - `asymmetric_gap`
  - `parkour_v2`

## 5. 路径 B：正式 Body-frame Heading Vector

### 5.1 目标

将 label 从 yaw 标量改为 body frame 下的 heading 向量，更符合报告中的创新描述。

### 5.2 label 设计

当前 waypoint heading：

```python
heading_current = [cos(delta_yaw), sin(delta_yaw)]
```

当前 + 下一 waypoint heading：

```python
heading_current_next = [
    cos(delta_yaw),
    sin(delta_yaw),
    cos(delta_next_yaw),
    sin(delta_next_yaw),
]
```

这样相比直接监督角度有两个好处：

- 避免角度 wrap-around 问题；
- heading 表征天然在 body frame 下，方向含义更明确。

### 5.3 两种 actor 接入方式

方式 1：兼容回填

- heading predictor 输出 body-frame vector；
- 再转换回 actor 需要的 yaw delta：

```python
delta_yaw_pred = atan2(sin_pred, cos_pred)
```

- 回填到 `obs[:, 6:8]`。

优点：

- actor 结构不变；
- 可以复用现有 teacher 和 student 结构。

缺点：

- actor 最终看到的仍是 yaw scalar；
- body-frame vector 主要体现在中间监督上。

方式 2：显式 heading feature 输入

- actor 直接接收 heading vector；
- actor 输入从：

```text
proprio + depth_latent + priv_explicit + hist/priv_latent
```

改为：

```text
proprio_without_old_heading + heading_feature + depth_latent + priv_explicit + hist/priv_latent
```

优点：

- 最符合报告中“actor 显式接收 heading feature”的设计；
- heading 表征更清晰。

缺点：

- actor 输入维度和语义改变；
- teacher / student 需要更谨慎地重新训练或迁移；
- `play.py`、`evaluate.py`、`save_jit.py` 都要同步适配。

### 5.4 推荐策略

路径 B 建议先采用方式 1：body-frame vector supervision + yaw scalar 回填。

原因：

- 可以验证 body-frame heading label 是否比 yaw scalar 更稳定；
- 不破坏 actor 输入结构；
- 成功后再升级到方式 2。

## 6. 推荐开发顺序

1. 实现路径 A：兼容版 heading pretrain。
2. 跑 baseline 对比：
   - 原始 depth distillation；
   - heading pretrain + action distillation。
3. 实现路径 B 的方式 1：
   - `heading_mode = "body_vec_current"`；
   - `heading_mode = "body_vec_current_next"`。
4. 对比三种 heading label：
   - `yaw_delta`
   - `body_vec_current`
   - `body_vec_current_next`
5. 最后考虑路径 B 的方式 2：actor 显式输入 heading feature。

## 7. 最小改动版本清单

为了最快启动开发，建议第一轮只改：

- `depth_backbone.py`
  - 拆分 `depth_latent_head` 和 `heading_head`；
  - 保持返回 `torch.cat([depth_latent, heading_pred], dim=-1)`。
- `on_policy_runner.py`
  - 增加 heading pretrain 阶段；
  - 增加 heading label buffer；
  - 根据阶段选择 update 函数。
- `ppo.py`
  - 增加独立 heading loss update；
  - 增加 loss weight。
- `legged_robot_config.py`
  - 增加 heading 相关配置。

暂时不改：

- `Actor` 输入维度；
- `n_proprio`；
- `play.py` / `evaluate.py` 的推理接口；
- `save_jit.py`。

## 8. 风险点

- 如果直接把 `obs[:, 6:8]` 从 yaw scalar 改成 `[cos, sin]`，旧 actor 的输入语义会被破坏；
- 如果 heading 输出从 2 维变成 4 维，需要同步处理 actor 输入维度，否则维度不匹配；
- `delta_yaw_ok` 当前只基于 `delta_yaw < 0.6`，如果换成 body-frame vector，应重新定义 mask 或先沿用当前 mask；
- 当前 `depth_encoder.detach_hidden_states()` 假设 encoder 有 GRU hidden state，重构时要保留这个接口；
- `play.py`、`evaluate.py`、`save_jit.py` 都依赖 `depth_encoder` 输出格式，第一版最好保持兼容输出。

## 9. 建议命名

配置命名：

```python
heading_mode = "yaw_delta"
heading_dim = 2
heading_pretrain_iters = 1000
heading_loss_weight = 1.0
action_loss_weight = 1.0
latent_loss_weight = 0.0
```

模块命名：

```python
VisualStudentBackbone
HeadingPredictorHead
DepthLatentHead
```

loss 命名：

```text
L_heading
L_action
L_latent
```

日志命名：

```text
Loss_depth/heading
Loss_depth/action
Loss_depth/latent
Loss_depth/delta_yaw_ok_percent
```

## 10. 路径 A 当前实现状态

已实现一个默认关闭的路径 A 版本。

### 10.1 默认行为

默认配置：

```python
enable_heading_model = False
heading_pretrain_iters = 0
heading_output_scale = 1.5
heading_loss_weight = 1.0
action_loss_weight = 1.0
```

在默认情况下，原有 vision distillation 行为保持不变：

- `depth_encoder` 仍输出 `32 + 2`；
- 前 32 维仍作为 `depth_latent`；
- 后 2 维仍作为 yaw / heading prediction；
- `obs_student[:, 6:8]` 仍由预测 yaw 回填；
- `update_depth_actor()` 仍同时优化 action distillation loss 和 yaw loss；
- 旧 checkpoint 的模型结构和 state dict key 不变。

### 10.2 启用方式

可通过配置启用：

```python
class depth_encoder:
    enable_heading_model = True
    heading_pretrain_iters = 1000
```

也可以通过命令行启用：

```bash
python legged_gym/scripts/train.py \
  --task a1 \
  --use_camera \
  --resume \
  --resumeid TEACHER_OR_BASE_RUN_ID \
  --exptid heading_a_pretrain \
  --proj_name parkour_heading \
  --enable_heading_model \
  --heading_pretrain_iters 1000
```

如需调整 loss 权重：

```bash
python legged_gym/scripts/train.py \
  --task a1 \
  --use_camera \
  --resume \
  --resumeid TEACHER_OR_BASE_RUN_ID \
  --exptid heading_a_w1 \
  --proj_name parkour_heading \
  --enable_heading_model \
  --heading_pretrain_iters 1000 \
  --heading_loss_weight 1.0 \
  --action_loss_weight 1.0
```

### 10.3 训练阶段行为

当 `enable_heading_model=True` 且当前 vision iteration 小于 `heading_pretrain_iters` 时：

- 使用 teacher action 与环境交互；
- 不调用 `depth_actor` 产生 student action；
- 只训练 `depth_encoder`；
- loss 为：

```text
L = heading_loss_weight * L_heading
```

当预训练阶段结束后：

- 恢复原有 student rollout；
- predicted heading 回填 `obs_student[:, 6:8]`；
- `depth_latent` 作为 `scandots_latent` 输入 `depth_actor`；
- loss 为：

```text
L = action_loss_weight * L_action + heading_loss_weight * L_heading
```

### 10.4 当前实现涉及文件

- `extreme-parkour/rsl_rl/rsl_rl/runners/on_policy_runner.py`
  - 增加 `enable_heading_model`、`heading_pretrain_iters`、`heading_output_scale`；
  - 在 `learn_vision()` 开头插入 heading-only pretrain stage；
  - 日志新增 `Loss_depth/heading` 和 `Loss_depth/heading_pretrain`。

- `extreme-parkour/rsl_rl/rsl_rl/algorithms/ppo.py`
  - 新增 `update_heading_predictor()`；
  - `update_depth_actor()` 支持 `heading_loss_weight` 和 `action_loss_weight`。

- `extreme-parkour/legged_gym/legged_gym/envs/base/legged_robot_config.py`
  - 新增 heading model 配置项。

- `extreme-parkour/legged_gym/legged_gym/utils/helpers.py`
  - 新增命令行参数：
    - `--enable_heading_model`
    - `--heading_pretrain_iters`
    - `--heading_loss_weight`
    - `--action_loss_weight`

## 11. 路径 A 实验建议

建议至少跑三组实验：

### 11.1 原始 baseline

```bash
python legged_gym/scripts/train.py \
  --task a1 \
  --use_camera \
  --resume \
  --resumeid TEACHER_OR_BASE_RUN_ID \
  --exptid depth_baseline \
  --proj_name parkour_heading
```

### 11.2 路径 A：短 heading pretrain

```bash
python legged_gym/scripts/train.py \
  --task a1 \
  --use_camera \
  --resume \
  --resumeid TEACHER_OR_BASE_RUN_ID \
  --exptid heading_a_500 \
  --proj_name parkour_heading \
  --enable_heading_model \
  --heading_pretrain_iters 500
```

### 11.3 路径 A：长 heading pretrain

```bash
python legged_gym/scripts/train.py \
  --task a1 \
  --use_camera \
  --resume \
  --resumeid TEACHER_OR_BASE_RUN_ID \
  --exptid heading_a_1000 \
  --proj_name parkour_heading \
  --enable_heading_model \
  --heading_pretrain_iters 1000
```

重点观察：

- `Loss_depth/heading`
- `Loss_depth/depth_actor`
- `Loss_depth/heading_pretrain`
- `Train/mean_reward`
- `Episode_rew/terrain_task_*`
- `bean_gap`、`asymmetric_gap`、`parkour_v2` 上的成功率或平均 episode length。

## 12. 在路径 A 基础上继续实现路径 B

路径 B 的目标是把 heading label 从 yaw scalar 改为 body-frame heading vector。

### 12.1 新增 heading label 生成函数

建议在 `legged_robot.py` 中新增函数：

```python
def get_heading_label(self, mode="yaw_delta"):
    if mode == "yaw_delta":
        return torch.cat([
            self.delta_yaw[:, None],
            self.delta_next_yaw[:, None],
        ], dim=-1)

    if mode == "body_vec_current":
        return torch.stack([
            torch.cos(self.delta_yaw),
            torch.sin(self.delta_yaw),
        ], dim=-1)

    if mode == "body_vec_current_next":
        return torch.stack([
            torch.cos(self.delta_yaw),
            torch.sin(self.delta_yaw),
            torch.cos(self.delta_next_yaw),
            torch.sin(self.delta_next_yaw),
        ], dim=-1)
```

然后在 `learn_vision()` 中用：

```python
heading_label = self.env.get_heading_label(self.depth_encoder_cfg["heading_mode"])
```

替代当前：

```python
yaw_buffer_teacher.append(obs[:, 6:8])
```

### 12.2 路径 B-1：body vector supervision + yaw 回填

这是建议优先做的路径 B 版本，兼容 actor 输入。

做法：

- `heading_mode = "body_vec_current_next"`；
- `heading_dim = 4`；
- depth encoder 后 4 维输出 body-frame vector；
- loss 监督 4 维 heading vector；
- 回填 actor 时，将 vector 转换回 yaw delta：

```python
delta_yaw = torch.atan2(heading_pred[:, 1], heading_pred[:, 0])
delta_next_yaw = torch.atan2(heading_pred[:, 3], heading_pred[:, 2])
obs_student[:, 6:8] = torch.stack([delta_yaw, delta_next_yaw], dim=-1)
```

需要修改：

- `depth_backbone.py`
  - 输出维度从 `32 + 2` 改为 `32 + heading_dim`；
  - 为了兼容旧 checkpoint，可以新增可选参数，而不是直接写死。
- `on_policy_runner.py`
  - 根据 `heading_mode` 生成 label；
  - 根据 `heading_mode` 决定如何把 prediction 回填到 `obs[:, 6:8]`。
- `ppo.py`
  - `update_heading_predictor()` 不需要关心 label 语义，只需要支持不同 heading dim。
- `play.py` / `evaluate.py`
  - 推理时同样需要把 body vector 转回 `obs[:, 6:8]`。
- `save_jit.py`
  - 如果导出 student policy，需要同步处理新的 heading dim。

### 12.3 路径 B-2：actor 显式接收 heading feature

这是更彻底的版本，也更容易影响旧策略。

做法：

- actor 输入显式改为：

```text
proprio_without_old_heading + heading_feature + depth_latent + priv_explicit + hist/priv_latent
```

需要修改：

- `actor_critic.py`
  - `Actor.forward()` 增加 `heading_feature=None` 参数；
  - actor backbone 输入维度增加 `heading_dim` 或替换原有 `obs[:, 6:8]`。
- `on_policy_runner.py`
  - 调用 `depth_actor(..., heading_feature=heading_pred)`。
- `ActorCriticRMA.act_inference()`
  - 同步透传 `heading_feature`。
- `play.py` / `evaluate.py` / `save_jit.py`
  - 同步新 actor 接口。

风险：

- 旧 teacher actor 的输入语义不再完全匹配；
- 很可能需要重新训练 teacher 或至少重新训练 student actor；
- checkpoint 兼容性会下降。

因此建议等 B-1 确认有效后，再进入 B-2。

### 12.4 路径 B 样例命令

B-1 已实现。当前不再额外提供 A/B 切换参数：

- `enable_heading_model = False`：保持原始 depth distillation，encoder 输出 `32 + 2`，后 2 维为 yaw delta；
- `enable_heading_model = True`：启用路径 B，encoder 输出 `32 + 4`，后 4 维为 body-frame heading vector。

训练命令形态如下：

```bash
python legged_gym/scripts/train.py \
  --task a1 \
  --use_camera \
  --resume \
  --resumeid TEACHER_OR_BASE_RUN_ID \
  --exptid heading_b_vec_next \
  --proj_name parkour_heading \
  --enable_heading_model \
  --heading_pretrain_iters 1000
```

当前不需要传 `--heading_mode`。启用 heading model 后，代码固定使用：

```text
heading_mode = body_vec_current_next
heading_dim = 4
heading_label = [
    cos(delta_yaw),
    sin(delta_yaw),
    cos(delta_next_yaw),
    sin(delta_next_yaw),
]
```

## 13. 路径 B 当前实现状态

### 13.1 训练侧行为

启用 `--enable_heading_model` 后：

- `RecurrentDepthBackbone` 输出维度从 `32 + 2` 变为 `32 + 4`；
- 前 32 维仍为 `depth_latent`；
- 后 4 维为 body-frame heading vector；
- heading label 由 `obs[:, 6:8]` 中的 yaw delta 在线转换；
- heading loss 直接监督这 4 维 vector；
- actor 接口不变，进入 `depth_actor` 前将 vector 转回 yaw delta。

转换逻辑：

```python
delta_yaw = atan2(sin_current, cos_current)
delta_next_yaw = atan2(sin_next, cos_next)
obs_student[:, 6:8] = [delta_yaw, delta_next_yaw]
```

### 13.2 默认关闭时的兼容性

虽然配置中保留：

```python
heading_mode = "body_vec_current_next"
heading_dim = 4
```

但只要 `enable_heading_model=False`，代码会强制按原始路径处理：

- encoder 构造为 `32 + 2` 输出；
- split 输出时按 2 维 yaw 切分；
- 推理时仍使用 `heading_output_scale * yaw_pred`；
- 原始训练和旧 checkpoint 不受影响。

### 13.3 涉及文件

- `extreme-parkour/rsl_rl/rsl_rl/modules/depth_backbone.py`
  - `RecurrentDepthBackbone(..., heading_dim=2)` 支持可变 heading 输出维度。
- `extreme-parkour/rsl_rl/rsl_rl/runners/on_policy_runner.py`
  - `enable_heading_model=True` 时构造 `heading_dim=4` 的 depth encoder；
  - 新增 heading label 构造；
  - 新增 vector-to-yaw 回填逻辑。
- `extreme-parkour/legged_gym/legged_gym/scripts/play.py`
  - 非 JIT 推理路径支持 `32 + 4` 输出。
- `extreme-parkour/legged_gym/legged_gym/scripts/evaluate.py`
  - 与 `play.py` 同步。
- `extreme-parkour/legged_gym/legged_gym/scripts/save_jit.py`
  - 导出 depth encoder 权重时额外保存 `depth_heading_dim` 和 `depth_heading_mode`。

### 13.4 训练样例

原始 baseline，不启用 heading model：

```bash
python legged_gym/scripts/train.py \
  --task a1 \
  --use_camera \
  --resume \
  --resumeid TEACHER_OR_BASE_RUN_ID \
  --exptid depth_baseline \
  --proj_name parkour_heading
```

路径 B，500 iteration heading pretrain：

```bash
python legged_gym/scripts/train.py \
  --task a1 \
  --use_camera \
  --resume \
  --resumeid TEACHER_OR_BASE_RUN_ID \
  --exptid heading_b_vec_500 \
  --proj_name parkour_heading \
  --enable_heading_model \
  --heading_pretrain_iters 500
```

路径 B，1000 iteration heading pretrain：

```bash
python legged_gym/scripts/train.py \
  --task a1 \
  --use_camera \
  --resume \
  --resumeid TEACHER_OR_BASE_RUN_ID \
  --exptid heading_b_vec_1000 \
  --proj_name parkour_heading \
  --enable_heading_model \
  --heading_pretrain_iters 1000 \
  --heading_loss_weight 1.0 \
  --action_loss_weight 1.0
```

### 13.5 评估和播放样例

评估路径 B 模型时，也需要传 `--enable_heading_model`，否则 runner 会按原始 `32 + 2` encoder 构造，加载 `32 + 4` checkpoint 会维度不匹配。

训练恢复时，如果从旧的 `32 + 2` depth encoder checkpoint 开启路径 B，代码会跳过不匹配的 depth encoder 权重并给出 warning；actor、estimator、optimizer 仍按原逻辑加载。建议路径 B 最稳的启动方式是从 base teacher checkpoint 或无 depth encoder 的 checkpoint 开始。

```bash
python legged_gym/scripts/evaluate.py \
  --task a1 \
  --use_camera \
  --exptid heading_b_vec_1000 \
  --proj_name parkour_heading \
  --enable_heading_model
```

一次性评估 0.1-1.0 共 10 个固定难度等级，并额外输出 `_by_difficulty.csv`：

```bash
python legged_gym/scripts/evaluation.py \
  --task a1 \
  --use_camera \
  --exptid heading_b_vec_1000 \
  --proj_name parkour_heading \
  --enable_heading_model \
  --difficulty_mode all-difficulty \
  --eval_episodes 1000
```

兼容别名：

```bash
python legged_gym/scripts/evaluation.py \
  --task a1 \
  --use_camera \
  --exptid heading_b_vec_1000 \
  --proj_name parkour_heading \
  --enable_heading_model \
  --difficulty_mode all-cifficulty \
  --eval_episodes 1000
```

播放路径 B 模型：

```bash
python legged_gym/scripts/play.py \
  --task a1 \
  --use_camera \
  --exptid heading_b_vec_1000 \
  --proj_name parkour_heading \
  --enable_heading_model
```

导出 JIT actor 和 depth encoder 权重：

```bash
python legged_gym/scripts/save_jit.py \
  --exptid heading_b_vec_1000 \
  --checkpoint -1
```

注意：`save_jit.py` 当前只 trace actor，depth encoder 权重单独保存。部署侧需要根据 `depth_heading_dim` 判断：

- `depth_heading_dim == 2`：旧 yaw delta 输出，使用 `heading_output_scale * pred`；
- `depth_heading_dim == 4`：body vector 输出，使用 `atan2(sin, cos)` 转回 actor 的 `obs[:, 6:8]`。

### 13.6 后续可选增强

当前 B 是“body-vector supervision + yaw 回填”的兼容版本。若要继续做更彻底的 B-2，需要：

- 修改 `Actor.forward()`，让 actor 显式接收 `heading_feature`；
- 调整 actor backbone 输入维度；
- 同步 `ActorCriticRMA.act_inference()`、`play.py`、`evaluate.py`、`save_jit.py`；
- 重新训练 student actor，必要时重新训练 teacher。

该版本会破坏更多 checkpoint 兼容性，建议等当前 B-1 在 `bean_gap`、`asymmetric_gap`、`parkour_v2` 上有稳定收益后再推进。
