# Terrain-Targeted Curriculum 方法介绍

大家好，我这一部分主要介绍我们现在使用的 terrain-targeted curriculum 方法，以及它是如何从前几轮失败实验中逐步调整出来的。

我们的问题是：机器人需要在多种 parkour 地形上都保持较高成功率，但不同地形的难度结构并不一样。例如 gap 类地形主要失败形式是摔落，而 step、hurdle 和 climbing wall 这类地形更常见的问题是卡在障碍物前，也就是 stuck。如果只看 all-terrain 的平均 success rate，很容易掩盖这种结构化失败。我们之前观察到，一些地形已经接近满分，但 `parkour_hurdle`、`parkour_step`、`alternating_step` 和 `climbing_wall` 长期是 stuck-dominated failure。

最早的 task-targeted curriculum 是按 terrain column 来维护难度的。这样做的问题是，同一类地形可能被分散到多个 column，各自维护自己的 success buffer 和 terrain level，样本效率比较低。后来我们改成按 terrain name 聚合，也就是说所有 `climbing_wall` column 共享同一个 curriculum state，所有 `parkour_step` column 也共享同一个 curriculum state。这样 curriculum 真正变成了按地形类别自适应，而不是按地图列自适应。

第二个重要修改是难度采样方式。之前每个 terrain task 维护的是一个中心 level，然后在中心附近加 jitter。但是对于 climbing wall 这种存在明显临界难度的地形，中心 level 一旦被推高，很多样本会同时变成 stuck，训练信号会变差。所以我们现在改成：每个 terrain task 维护一个 `max_level`，reset 时实际 level 从 `[0, max_level]` 之间随机采样。这样保留了 task-targeted 的特点，也就是每种地形有自己的难度进度；同时也保留了 non-TT curriculum 的优点，也就是同一类地形内部仍然有低、中、高不同难度样本。

第三个修改是 curriculum 的 success signal。我们试过只看最终完成所有 goals，这个信号太稀疏；也试过只看移动距离，但它和 parkour 的 waypoint completion 不完全一致。现在采用 waypoint-based success：如果机器人完成了 75% 以上的 goals，就给这个 terrain 一个 success signal；如果完成不到 25%，就给 failure signal；中间区间不更新 curriculum。这个信号比最终成功更密集，又比单纯距离更贴合任务目标。

除了 curriculum，我们还加入了 reward shaping。主要包括 `goal_progress`，奖励机器人靠近当前 waypoint；`goal_reached`，奖励成功切换 waypoint；以及 `stuck` penalty，当机器人在一段时间内前进速度很低、并且到目标点的距离几乎没有减少时给予惩罚。对于最容易 stuck 的四类地形：`parkour_hurdle`、`parkour_step`、`alternating_step` 和 `climbing_wall`，我们进一步增强了 stuck penalty。

训练上我们采用两阶段策略。Phase 1 主要集中训练 stuck-prone terrains，让 policy 先学会跨台阶、过 hurdle 和 climbing wall 这些之前缺失的动作技能。Phase 2 再把 all terrains 混回来，同时保留对这些困难地形的较高采样权重，目标是恢复整体泛化能力，并避免 phase 1 造成其它地形遗忘。

总结来说，现在的方法包括五个核心点：按 terrain name 聚合的 curriculum、每类地形维护 max level、从 `[0, max_level]` 随机采样实际难度、基于 waypoint progress 的 curriculum update，以及针对 stuck failure 的 reward shaping。它的目标是在保留 task-targeted 自适应能力的同时，避免早期 TT 方法中难度样本过窄、容易震荡和 stuck 地形学不起来的问题。
