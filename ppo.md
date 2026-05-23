# /rsl_rl 中 PPO 实现原理说明

`/rsl_rl` 里的 PPO 不是“纯教科书版 PPO”，而是一个以 PPO 为主干、叠加了 privileged information、history adaptation、estimator 和可选 depth 模块的 on-policy 框架。核心代码在 [rsl_rl/rsl_rl/algorithms/ppo.py](rsl_rl/rsl_rl/algorithms/ppo.py)、[rsl_rl/rsl_rl/storage/rollout_storage.py](rsl_rl/rsl_rl/storage/rollout_storage.py)、[rsl_rl/rsl_rl/modules/actor_critic.py](rsl_rl/rsl_rl/modules/actor_critic.py) 和 [rsl_rl/rsl_rl/runners/on_policy_runner.py](rsl_rl/rsl_rl/runners/on_policy_runner.py)。

先给结论：它的“PPO 本体”仍然是标准的 clipped PPO：

\[
L^{CLIP}(\theta)=\mathbb E[\max(-r_tA_t,-\mathrm{clip}(r_t,1-\epsilon,1+\epsilon)A_t)]
\]

再加上 value loss、entropy bonus，以及两个项目特有项：

1. `priv_reg_loss`：让 history encoder 输出的 latent 接近 privileged latent。
2. `estimator_loss`：用 proprio 估计一部分 privileged states。

## 1. 整体训练流程

每轮迭代在 `OnPolicyRunner.learn_RL()` 中完成，见 `rsl_rl/rsl_rl/runners/on_policy_runner.py`。

训练流程如下：

1. 用当前策略在并行环境里采样 `num_steps_per_env` 步。
2. 把 `obs`、`critic_obs`、`action`、`reward`、`done`、`value`、`old_log_prob`、`old_mu`、`old_sigma` 存入 rollout buffer。
3. 用最后一步 critic value 做 bootstrap，计算 returns 和 advantages。
4. 对这批 rollout 做多 epoch、多 mini-batch 的 PPO 更新。
5. 每隔 `dagger_update_freq` 轮，再额外执行一次 `update_dagger()`，训练 history encoder。

这就是典型的 on-policy PPO：先采一批，再重复利用这一批样本做若干轮更新。

## 2. actor 和 critic 的输入不对称

这份实现最关键的工程差异，是 actor 和 critic 使用不同观测。

在 `OnPolicyRunner.__init__()` 中：

- actor 使用环境普通观测 `obs`
- critic 使用 `privileged_obs`
- 如果环境没有 privileged observation，则 critic 退回使用普通 `obs`

这属于 asymmetric actor-critic。直观上：

- actor 负责部署，尽量只依赖部署时可获得的观测
- critic 只在训练时用，可以看更多信息，从而提供更稳定、更准确的价值估计

## 3. actor 的结构

`Actor` 在 `rsl_rl/rsl_rl/modules/actor_critic.py` 中实现。它会把输入拆成几个部分：

- `proprio`
- `scan`
- `priv_explicit`
- `latent`

其中 `latent` 有两种来源：

1. `infer_priv_latent(obs)`：从 privileged latent 部分直接提取
2. `infer_hist_latent(obs)`：从历史 proprio 序列中编码得到

因此 actor 实际上有两条行为路径：

- teacher 路径：使用 privileged latent
- student 路径：使用 history latent

项目的目标之一，就是让 student 路径在没有 privileged 信息时也能逼近 teacher。

## 4. 动作分布

这是一个连续动作空间 PPO。`ActorCriticRMA` 中使用高斯策略：

- 均值 `mean = actor(observations, hist_encoding)`
- 方差由一个全局可学习参数 `self.std` 给出
- 最终策略分布是 `Normal(mean, std)`

采样时调用 `distribution.sample()`，计算 log-prob 时调用 `distribution.log_prob(actions).sum(dim=-1)`。

## 5. rollout 采样阶段做了什么

在 `PPO.act()` 中，每次与环境交互前会完成以下步骤：

1. 如果 `train_with_estimated_states=True`，先用 estimator 从 proprio 预测 privileged states，并写回 actor 输入。
2. 用 actor 采样动作。
3. 用 critic 计算当前状态的 value。
4. 保存本步 transition 的关键信息：
   - `actions`
   - `values`
   - `actions_log_prob`
   - `action_mean`
   - `action_sigma`
   - `observations`
   - `critic_observations`

这一步保存的 `old_log_prob`、`old_mu`、`old_sigma` 会在后续 PPO 更新时作为“旧策略”的统计量使用。

## 6. rollout buffer 存储内容

`RolloutStorage` 中保存了 PPO 所需的完整 on-policy 轨迹：

- `observations`
- `privileged_observations`
- `actions`
- `rewards`
- `dones`
- `values`
- `actions_log_prob`
- `mu`
- `sigma`

这里额外保存 `mu` 和 `sigma`，是为了后续计算 KL 和旧策略比值，不需要再回放旧网络。

## 7. returns 和 advantages 的计算

`RolloutStorage.compute_returns()` 使用的是 GAE，也就是 Generalized Advantage Estimation。

公式为：

\[
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
\]

\[
A_t = \delta_t + \gamma\lambda A_{t+1}
\]

\[
R_t = A_t + V(s_t)
\]

代码实现对应：

- `delta = reward + gamma * next_value - value`
- `advantage = delta + gamma * lam * advantage`
- `returns = advantage + value`

之后再做 advantage 标准化：

\[
\hat A_t = \frac{A_t-\mu_A}{\sigma_A+10^{-8}}
\]

这样可以显著提升 PPO 训练稳定性。

## 8. timeout bootstrap 修正

在 `PPO.process_env_step()` 中，如果 `infos` 里包含 `time_outs`，代码会做如下修正：

\[
r_t \leftarrow r_t + \gamma V(s_{t+1})
\]

但只对时间截断的 episode 生效。

这意味着：

- 真正失败终止仍然按 terminal 处理
- 因时间上限结束的轨迹不会被错误地当成环境自然终止

这样可以减少 time-limit bias。

## 9. PPO 更新主干

`PPO.update()` 是 PPO 的核心。

对于每个 mini-batch，先重新用当前网络计算：

- 当前策略下该 batch 动作的 `log_prob`
- 当前 critic 给出的 `value`
- 当前策略的 `mu` 和 `sigma`
- 当前策略的熵 `entropy`

### 9.1 ratio

PPO 的重要变量是新旧策略比值：

\[
r_t(\theta)=\exp\left(\log \pi_\theta(a_t|s_t)-\log \pi_{\theta_{\mathrm{old}}}(a_t|s_t)\right)
\]

代码中对应：

```python
ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
```

### 9.2 clipped surrogate loss

PPO 的核心目标函数是：

\[
L_t^{clip}=\max(-A_tr_t,\,-A_t\mathrm{clip}(r_t,1-\epsilon,1+\epsilon))
\]

代码中先算两部分：

```python
surrogate = -torch.squeeze(advantages_batch) * ratio
surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
    ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
)
surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()
```

这里之所以是 `max`，是因为代码写的是“最小化 loss”的形式，而论文常写成“最大化目标”的形式，两者只是符号方向不同。

### 9.3 value loss

value function 的损失也支持 clipping。

若启用 `use_clipped_value_loss`，则：

\[
V_t^{clip}=V_{old}(s_t)+\mathrm{clip}(V_\theta(s_t)-V_{old}(s_t),-\epsilon,\epsilon)
\]

然后比较：

- `(V - R)^2`
- `(V_clip - R)^2`

取两者中更大的那个作为 value loss。这是为了防止 critic 一次更新幅度过大。

### 9.4 entropy bonus

总损失中有一项：

\[
-\beta \cdot \mathbb E[\mathcal H(\pi)]
\]

这就是 entropy regularization。它的作用是：

- 保持策略分布不要过快塌缩
- 提高探索性

## 10. 自适应 KL 学习率控制

如果配置为 `schedule == "adaptive"`，代码会额外计算 old/new policy 间的 KL，并用它调节学习率。

逻辑是：

- 如果 KL 大于 `desired_kl * 2`，说明更新太猛，减小学习率
- 如果 KL 小于 `desired_kl / 2`，说明更新太保守，增大学习率

这不是 PPO 必需项，但在工程里很常见。这里 KL 没有直接作为 loss 项，而是作为步长调节信号。

## 11. 项目特有扩展一：estimator

在 `PPO.update()` 中，除 PPO 主损失外，还会更新一个 estimator：

- 输入：`obs_batch[:, :num_prop]`
- 目标：观测中对应的 privileged states 片段
- 损失：均方误差 MSE

公式上就是：

\[
L_{\mathrm{est}} = \| \hat p(s) - p(s) \|_2^2
\]

它的意义是：让系统学会只根据 proprio 去恢复一部分训练时才可见的 privileged state。

注意这里 estimator 使用的是独立优化器，单独执行：

1. `zero_grad()`
2. `backward()`
3. `clip_grad_norm_()`
4. `step()`

也就是说 estimator loss 不直接并入 actor-critic 的 PPO 总损失。

## 12. 项目特有扩展二：privileged latent 和 history latent 对齐

这是该项目最关键的 adaptation 设计之一。

在 `PPO.update()` 中：

- `priv_latent_batch = infer_priv_latent(obs_batch)`
- `hist_latent_batch = infer_hist_latent(obs_batch)`

定义：

\[
L_{\mathrm{priv}} = \| z_{\mathrm{priv}} - z_{\mathrm{hist}} \|_2
\]

它的含义是让 history encoder 学到一个与 privileged latent 对齐的表示。

这里还用了一个逐步增长的 `priv_reg_coef`：

- 前期弱约束甚至不约束
- 后期逐渐增大该正则强度

这样可以避免训练初期 history encoder 还没成形时，过早强行对齐导致优化不稳定。

## 13. 项目特有扩展三：DAgger 风格的 history encoder 更新

`PPO.update_dagger()` 会单独训练 history encoder。

它的做法是：

1. 用 teacher 路径得到 `priv_latent_batch`
2. 用 history encoder 得到 `hist_latent_batch`
3. 最小化两者之间的 L2 距离

即：

\[
L_{\mathrm{hist}} = \| z_{\mathrm{priv}} - z_{\mathrm{hist}} \|_2
\]

但这一步只更新 `history_encoder` 自己的参数，不更新整个 actor-critic。

从思想上说，这是一种蒸馏或 DAgger 风格的 adaptation：

- teacher：可访问 privileged latent
- student：只能访问历史 proprio

student 被单独训练去模仿 teacher 的隐变量表示。

## 14. 总损失的结构

actor-critic 的 PPO 总损失在代码中写为：

\[
L = L_{\mathrm{ppo}}
+ c_v L_{\mathrm{value}}
- c_e H
+ c_{\mathrm{priv}} L_{\mathrm{priv}}
\]

对应代码：

```python
loss = surrogate_loss + \
       self.value_loss_coef * value_loss - \
       self.entropy_coef * entropy_batch.mean() + \
       priv_reg_coef * priv_reg_loss
```

注意：

- `estimator_loss` 不在这个总 loss 里
- `estimator` 用独立优化器单独训练
- `history_encoder` 还会在 `update_dagger()` 中额外单独训练

所以整个系统实际上并行更新三部分：

1. `actor_critic`：用 PPO 主损失
2. `estimator`：用 privileged state 回归损失
3. `history_encoder`：用 latent 对齐蒸馏损失

## 15. 与标准 PPO 的差异

如果只看 PPO 本体，这里仍然是标准的：

- on-policy rollout
- GAE
- clipped policy loss
- clipped value loss
- entropy regularization
- mini-batch 多 epoch 更新

但项目相对于教科书版 PPO，多了以下机制：

- asymmetric critic：critic 训练时看 privileged observation
- actor 输入被结构化拆分为 `prop + scan + priv_explicit + latent`
- latent 有 teacher 和 student 两套来源
- estimator 负责从 proprio 预测 privileged states
- history encoder 负责从历史序列恢复 latent
- 通过 DAgger / distillation 风格训练 history encoder
- 可选 depth encoder 和 depth actor 分支

因此更准确地说，这个项目实现的是：

“一个标准 PPO 优化器，外面包了一层 privileged learning 和 adaptation 框架。”

## 16. 一个很重要的实现观察

有一个值得特别注意的细节：

- rollout 时，`hist_encoding` 在 `OnPolicyRunner.learn_RL()` 中由 `it % dagger_update_freq == 0` 控制
- 这意味着某些迭代采样时 actor 走的是 history latent 分支

但在 `PPO.update()` 中，重新计算当前策略分布时调用的是：

```python
self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
```

这里没有显式传入 `hist_encoding=True`，因此会走默认值 `False`，也就是 privileged latent 分支。

这意味着从严格实现上看：

- PPO 主更新更接近在优化 teacher policy
- history 分支主要靠 `update_dagger()` 去对齐和蒸馏

所以不要把这份代码理解成“单一策略网络上的标准 PPO”。更准确的理解是：

- teacher policy 由 PPO 驱动学习
- student adaptation 分支通过额外蒸馏学习逼近 teacher

## 17. 一句话总结

`/rsl_rl` 中的 PPO 可以概括为：

“以标准 clipped PPO 为优化核心，使用 asymmetric actor-critic 训练 teacher policy，再通过 estimator 和 history encoder 把 privileged 信息蒸馏到 deployment-friendly 的 student 表示中。”
