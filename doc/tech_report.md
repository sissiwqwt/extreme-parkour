# New Obstacle & Terrain

+ **Alternating step**: 沿前进方向连续若干小 step，但高度上下交替。比起原有parkour_step（连续上+连续下）更具有交错性，更强调步态节律。
    ![alt text](figures/alternative_step.png)

+ **Bean gap**: 大部分区域是坑，中央只留一条窄而连续的落脚梁；梁上再插入一两个短 gap。比起原有的parkour_gap对横向偏差更敏感，更强调 heading 精度。
    ![alt text](figures/bean_gap.png)

+ **Biased gap**: 前后相邻的落脚平台左右错开，迫使机器狗选择最优起跳点和落脚点。更强调heading精度。
    ![alt text](figures/asymmetric_gap.png)

+ **Parkour v2**: 由上述地形随机排列形成，更考验策略的可泛化性与统一性。
    ![alt text](figures/parkour_v2.png)


# Pretrained Headding Model

+ **目标**：在 teacher-to-student 蒸馏中，显式加入 heading 预测中间步骤；先学习“朝向哪里走”，再学习“如何输出动作”。

+ **动机**
  + teacher 策略可利用 goals / waypoint 信息；
  + 现有 student 主要蒸馏 actor 动作，heading 信息未被显式建模；
  + 对 **bean gap / biased gap / parkour v2** 等更依赖方向精度的地形，中间 heading 表征有助于提升可解释性与稳定性。

+ **Heading label 设计**
  + label 改为 **body frame 下的 heading 向量**，而非单一 yaw 标量；
  + 监督目标使用相对 waypoint 方向的单位向量表示；
  + 可选两种形式：
    + 当前 waypoint heading 向量；
    + 当前 + 下一 waypoint heading 向量。

+ **验证 idea**
  + 保留现有 actor 主体结构；
  + 先将预测得到的 heading 特征回填到 student 输入中的 heading 槽位；
  + 用于快速验证：
    + heading 中间监督是否有效；
    + 方向预测是否能改善动作蒸馏效果；
    + 在复杂地形上的鲁棒性是否提升。

+ **学生视觉模型重构**
  + 共享视觉主干：`VisualStudentBackbone`
    + 输入：本体感知 + 深度图；
    + 输出：共享视觉特征。
  + 方向预测头：`HeadingPredictorHead`
    + 输入：共享视觉特征；
    + 输出：body frame 下的 heading 向量。
  + 地形潜变量头：`DepthLatentHead`
    + 输入：共享视觉特征；
    + 输出：深度 latent / 地形 latent。
  + actor 显式接收新参数：
    + 输入：本体感知 + heading feature + depth latent；
    + 输出：电机动作。

+ **训练流程**
  + **阶段 1：Heading 预训练**
    + 冻结 actor；
    + 仅训练视觉主干 + heading 预测头；
    + 目标：从本体感知 + 深度图稳定恢复 body-frame heading 向量。
  + **阶段 2：Action 蒸馏**
    + 使用已训练的 heading predictor；
    + 将 predicted heading 与 depth latent 一起输入 actor；
    + 蒸馏 teacher 动作输出。
  + **阶段 3：可选联合微调**
    + 联合优化 backbone / heading head / latent head / actor；
    + 同时兼顾：
      + heading 预测误差；
      + action 蒸馏误差；
      + 可选 latent 对齐误差。

+ **训练目标**
  + `L_heading`：预测 heading 向量与 teacher heading label 的误差；
  + `L_action`：student actor 与 teacher actor 的动作蒸馏误差；
  + `L_latent`（可选）：depth latent 与 teacher terrain latent 的对齐误差；
  + 联合微调阶段总损失：
    + `L = w_h * L_heading + w_a * L_action + w_l * L_latent`

+ **模型结构示意**

```text
          proprio + depth image
                   |
                   v
        +-----------------------+
        | VisualStudentBackbone |
        +-----------------------+
             |             |
             |             |
             v             v
   +----------------+   +----------------+
   | HeadingPredict |   | DepthLatent    |
   |     orHead     |   |      Head      |
   +----------------+   +----------------+
             |             |
             +------+\ /---+
                    \ X
                     v
          proprio + heading + latent
                     |
                     v
                +---------+
                |  Actor  |
                +---------+
                     |
                     v
                  actions
```

+ **预期收益**
  + 显式恢复 teacher 的 waypoint / direction 信息；
  + 将“去哪里”与“怎么走”解耦；
  + 提高 student 策略的可解释性；
  + 在窄梁、错位落脚点、连续复杂障碍上提升 heading 精度与动作稳定性。

# Task-targeted Curriculum

+ **目标**
  + 将 terrain curriculum 从“统一、按单个环境即时升降”扩展为“按任务类型分别统计、分别调节难度”；
  + 针对 `terrain_dict` 中权重非零的任务，分别维护各自的 curriculum 难度；
  + 使不同障碍任务能够按照各自学习进度独立推进，而不是互相牵制。

+ **动机**
  + 现有 curriculum 主要依据单个环境在当前 episode 中的表现即时升降，缺少任务维度上的区分；
  + 不同任务的几何结构、控制难点和学习速度差异较大，统一难度调整容易造成：
    + 简单任务过早“带动”整体升难；
    + 困难任务长期卡住，拖累整体训练节奏；
    + 某些任务训练不足，另一些任务过拟合。
  + 基于固定样本窗口的统计比单次 episode 即时调整更平滑：
    + 能削弱偶然成功 / 失败带来的波动；
    + 能降低难度来回震荡；
    + 更适合作为 curriculum 的长期调度信号。
  + 引入滞回阈值后，可进一步增强防抖能力：
    + 高于上阈值再升难；
    + 低于下阈值再降难；
    + 中间区间保持不变。

+ **方案**
  + **基本思想**
    + 以任务类型为单位维护 curriculum；
    + 对 `terrain_dict` 中权重非零的任务分别统计完成情况；
    + “完成情况”沿用原始 curriculum 定义，不改变原有成功 / 失败判据。
  + **统计方式**
    + 采用“固定样本数窗口”统计各任务最近一段时间的完成表现；
    + 以任务为单位计算窗口内完成率 / 失败率；
    + 不根据单个 episode 立即调整难度，而是等待窗口统计结果。
  + **升降规则**
    + 使用“固定样本数 + 滞回阈值”；
    + 当某任务窗口内表现稳定高于上阈值时，提升该任务难度；
    + 当某任务窗口内表现稳定低于下阈值时，降低该任务难度；
    + 当表现落在中间区间时，保持当前难度不变。
  + **样本不足处理**
    + 若某一统计窗口内，单个任务收集到的 episode 样本不足；
    + 则该任务本轮暂不升降难度；
    + 优先等待足够样本后再更新，避免由稀疏数据驱动错误调整。
  + **整体效果**
    + 各任务难度随各自学习进度独立演化；
    + 形成“task-targeted”而非“global-shared”的 curriculum 调度机制。

+ **预期收益**
  + 让不同任务按照各自收敛速度推进，提升 curriculum 与任务难度的匹配性；
  + 减少简单任务与困难任务之间的相互干扰；
  + 平滑难度调整过程，降低由单次 episode 波动引起的抖动；
  + 提高复杂障碍任务的持续训练机会，缓解“强者恒强、弱者长期滞后”的现象；
  + 有助于提升整体训练稳定性、任务覆盖均衡性以及最终策略的泛化表现。
