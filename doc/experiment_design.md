# 实验设计：自适应四足机器人 Parkour

## 1. 项目目标

本项目围绕课程主题 "Quadruped robot parkour"，并基于 Extreme Parkour baseline 开展。目标是在 Isaac Gym 中训练并评估一个能够利用 proprioception 和 vision 跨越复杂障碍的四足机器人。

本项目包含两个层次的目标：

1. 在仿真中复现 Extreme Parkour baseline。
2. 验证项目 proposal 中提出的改进：
   - 新的 parkour terrains：beam gap、biased gap、alternating step、slanted hurdle 和 parkour v2。
   - Task-targeted curriculum learning。
   - Heading-aware student distillation。

## 2. 实验平台

### Main Platform

所有正式的 training、distillation、quantitative evaluation 和 demo recording 都应在带显示输出的 GPU server 上进行。

优先使用 headed server，原因如下：

- Student distillation 需要 camera rendering 和 Isaac Gym GPU camera tensors。
- 大规模 evaluation 可以使用更多 parallel environments，例如 128 或 256 envs。
- Demo recording 和基于 camera 的 playback 更稳定。
- 所有报告结果来自相同的 hardware 和 software environment。

### Local Machine

本地 RTX 4060 机器仅用于：

- 快速 checkpoint sanity checks。
- 单机器人 visualization。
- 调试 playback commands。
- 备份 demo inspection。

除非 server-side evaluation 不可用，否则本地 playback 结果不应作为主要 quantitative results。

## 3. 需要训练和评估的 Policies

### P0: Teacher / Base Policy

这是 phase-1 Extreme Parkour policy。

Inputs:

- Proprioception。
- Privileged terrain scandots。
- Oracle heading / waypoint direction。

当前 checkpoint:

```text
logs/parkour_new/01-trainn-base/model_49500.pt
```

该 policy 可用于验证基础 parkour 能力，也可作为 distillation 的 teacher。

### P1: Baseline Student / Deployable Policy

这是原始 Extreme Parkour pipeline 中 phase-2 distilled policy。

Inputs:

- Proprioception。
- Front-facing depth image。

Training command template:

```bash
python train.py \
  --exptid 02-distill-baseline \
  --task a1 \
  --device cuda:0 \
  --rl_device cuda:0 \
  --resume \
  --resumeid 01-tra \
  --checkpoint 49500 \
  --delay \
  --use_camera
```

完整 baseline reproduction 必须包含该 policy，因为原论文中的 deployable policy 使用 depth images。

### P2: Fine-Tuned Policy on New Terrains

该 policy 在我们新设计的 terrains 上 fine-tune 或 retrain。

至少需要训练一个 fine-tuned teacher/base policy。如果时间允许，再从该 policy distill 出对应的 student policy。

### P3: Proposed Policy with Task-Targeted Curriculum

该 policy 使用 proposal 中提出的 task-targeted curriculum，而不是原始 global curriculum。

它用于测试 independent per-terrain curriculum states 是否能够提升训练稳定性和最终性能。

### P4: Heading-Aware Student Policy

该 policy 使用 proposal 中提出的 heading-aware distillation pipeline。

它会在 action distillation 之前或过程中显式预训练 heading prediction module。

这是一个价值较高但风险也更高的实验。

## 4. Metrics

所有 quantitative experiments 都应报告以下 metrics。

### Success Rate

机器人成功到达目标赛道终点，或到达足够多 waypoints 的 episodes 百分比。

Recommended implementation:

```text
success = cur_goal_idx >= terrain.num_goals
```

或者，如果 full-course completion 过于严格：

```text
success = normalized_waypoints >= threshold
```

报告中必须清楚说明 threshold。

### Mean X-Displacement (MXD)

失败或 timeout 之前的平均前向位移。

```text
MXD = final_root_x - start_root_x
```

数值越高越好。

### Fall Rate

由于跌倒而不是 timeout 或成功完成导致 episode 终止的百分比。

Falling 可以通过以下 reset reasons 检测：

- Excessive roll。
- Excessive pitch。
- Base height below threshold。

### Stuck Rate

机器人仍然存活但前向进展不足的 episodes 百分比。

Recommended definition:

```text
stuck = forward displacement over the last 2 seconds < 0.1 m
```

具体 threshold 可以调整，但必须在 evaluation 前固定。

### Collision / Failure Cases

当前 environment 中 collision 可能无法作为干净的 label 直接获取。可以通过以下方式分析：

- Visual inspection。
- 与 obstacle 反复接触后没有继续前进。
- 在 obstacle 附近速度突然下降。
- 从 qualitative videos 中标注 failure reason。

### Mean Edge Violation (MEV)

该指标沿用 Extreme Parkour paper。

Recommended proxy:

```text
MEV = average feet_at_edge contact count per timestep
```

数值越低越好。

### Heading Error

仅用于 heading-aware distillation experiments。

```text
heading_error = arccos(dot(pred_heading, teacher_heading))
```

数值越低越好。

### Qualitative Video

对于每种主要 terrain，记录：

- 一个成功 rollout。
- 一个有代表性的失败 rollout。

Failure videos 应分类为：

- Falling。
- Collision。
- Stuck。
- Heading drift。
- Edge stepping。
- Bad takeoff or landing。

## 5. Required Logging

当前 `play.py` 主要是 visualization script，不保存 per-env results。在运行正式实验前，应向 `play.py` 添加 evaluation logger，或创建专门的 evaluation script。

Logger 应为每个 episode 保存一行：

```text
policy_id
checkpoint
seed
env_id
terrain_name
terrain_type
terrain_level
start_x
final_x
mxd
num_waypoints
normalized_waypoints
episode_length
success
fall
stuck
edge_violation
failure_reason
```

Recommended output format:

```text
CSV for tables
JSON for detailed rollout metadata
MP4 for qualitative examples
```

如果没有 per-env logging，就无法可靠报告 success rate、MXD、fall rate 和 stuck rate。

## 6. Must-Do Experiments

以下实验是必做项，因为它们直接满足课程 2.4 要求。

### E1. Full Baseline Reproduction

Course requirement:

```text
Reproduce a baseline parkour policy in Isaac Gym.
```

Goal:

复现 Extreme Parkour 中的 teacher/base policy 和 student/deployable policy。

Required conditions:

- Isaac Gym environment 能在 headed GPU server 上运行。
- Teacher/base checkpoint 可用。
- `--use_camera` training 可用于 student distillation。
- Depth camera tensors 能正确 render。

Steps:

1. 验证 teacher/base checkpoint:

```bash
python play.py \
  --exptid 01-tra \
  --task a1 \
  --device cuda:0 \
  --rl_device cuda:0 \
  --checkpoint 49500 \
  --num_envs 1
```

2. 训练 baseline student:

```bash
python train.py \
  --exptid 02-distill-baseline \
  --task a1 \
  --device cuda:0 \
  --rl_device cuda:0 \
  --resume \
  --resumeid 01-tra \
  --checkpoint 49500 \
  --delay \
  --use_camera
```

3. 在相同 standard terrains 上评估 teacher 和 student。

Expected result:

- Teacher 应在 original parkour terrains 上表现较强。
- Student 应接近 teacher performance，但允许存在一定性能下降。

### E2. Standard Obstacle Evaluation

Course requirement:

```text
Validate the robot's performance on standard obstacle tracks.
```

Goal:

在原始 Extreme Parkour terrain types 上评估 teacher 和 student policies。

Terrains:

- `parkour_hurdle`
- `parkour_step`
- `parkour_gap`
- `parkour`
- mixed original parkour course

Evaluation setting:

```text
num_envs: 128 or 256 on server
episode length: 20s or 30s
levels: 0, 1, 2, 3, 4
seeds: at least 3 if time permits
```

Metrics:

- Success Rate。
- MXD。
- Fall Rate。
- Stuck Rate。
- MEV。
- Qualitative video。

Output table:

| Policy | Terrain | Level | Episodes | Success Rate | MXD | Fall Rate | Stuck Rate | MEV |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Teacher | Hurdle | 0-4 | | | | | | |
| Student | Hurdle | 0-4 | | | | | | |
| Teacher | Gap | 0-4 | | | | | | |
| Student | Gap | 0-4 | | | | | | |

### E3. New Terrain Zero-Shot Evaluation

Course requirement:

```text
Design a new terrain or obstacle type.
```

Goal:

展示新设计 terrains 的挑战性，并揭示 baseline 的局限。

New terrains:

- Beam gap。
- Biased gap。
- Alternating step。
- Slanted hurdle。
- Parkour v2。

Note:

最终报告中使用统一名称。Proposal 中同时出现了 "beam gap" 和 "bean gap"；除非代码中使用了另一个固定名称，否则使用 "beam gap"。

Evaluation:

- 不进行 fine-tuning。
- 直接测试 P0 teacher 和 P1 student。
- 使用与 E2 相同的 evaluation metrics。

Expected observations:

- Beam gap：lateral deviation 导致跌落。
- Biased gap：错误 heading 导致 bad takeoff 或 landing。
- Alternating step：gait rhythm 不稳定或 stuck behavior。
- Slanted hurdle：对方向变化适应较差。
- Parkour v2：混合 terrain 上出现 generalization failures。

Output table:

| Policy | New Terrain | Level | Success Rate | MXD | Fall Rate | Main Failure |
|---|---|---:|---:|---:|---:|---|
| Teacher | Beam gap | | | | | |
| Student | Beam gap | | | | | |
| Teacher | Biased gap | | | | | |
| Student | Biased gap | | | | | |

### E4. Fine-Tuning / Retraining on New Terrains

Course requirement:

```text
Fine-tune or retrain the policy to adapt to the newly designed environment.
```

Goal:

提升 policy 在 new terrains 上的性能。

Minimum required version:

- 在 new terrains 上 fine-tune teacher/base policy。

Preferred version:

- 在 new terrains 上 fine-tune teacher/base policy。
- 从 fine-tuned teacher distill 一个新的 student policy。

Policy comparison:

```text
P0: Original teacher
P1: Original student
P2: Fine-tuned teacher
P2-student: Distilled student from fine-tuned teacher
```

Evaluation:

- 同时在 original terrains 和 new terrains 上测试。
- 用于检查提升效果以及是否发生 regression。

Expected result:

- Fine-tuning 后 new terrain success rate 提高。
- Original terrain performance 不应崩溃。

Output table:

| Policy | Original Terrain SR | New Terrain SR | MXD | Fall Rate | Notes |
|---|---:|---:|---:|---:|---|
| Original Teacher | | | | | |
| Fine-Tuned Teacher | | | | | |
| Original Student | | | | | |
| Fine-Tuned Student | | | | | |

### E5. Failure Case Analysis and Qualitative Video

Course requirement:

```text
Evaluate the success rate and analyze the failure cases.
```

Goal:

不仅解释 policy 是否失败，还要解释失败原因。

For each main terrain, collect:

- 一个成功 video。
- 一个失败 video。
- Failure type。
- 简短说明。

Failure categories:

- Fall。
- Collision。
- Stuck。
- Heading drift。
- Edge stepping。
- Bad takeoff。
- Bad landing。

Example table:

| Terrain | Policy | Case | Failure Type | Description |
|---|---|---|---|---|
| Beam gap | Student | Failure | Heading drift | Robot approaches the narrow beam at an angle and falls laterally. |
| Biased gap | Teacher | Failure | Bad landing | Robot jumps from the wrong side and lands partially outside the platform. |

## 7. Strongly Recommended Innovation Experiment

### E6. Task-Targeted Curriculum Ablation

这是 proposal 中最实际可行的 innovation experiment。

Goal:

比较原始 global curriculum 和提出的 task-targeted curriculum。

Methods:

```text
C0: Original global curriculum
C1: Task-targeted curriculum
```

Training conditions:

- 使用相同 initial checkpoint，或相同 from-scratch setup。
- 相同 total training iterations。
- 相同 number of environments。
- 相同 terrain distribution。
- 如果可行，使用相同 random seeds。

Logged training curves:

- Per-task success rate。
- Per-task terrain level。
- Per-task episode length。
- Overall reward。
- Curriculum level oscillation。

Evaluation:

- 在 original 和 new terrains 上测试 C0 与 C1。
- 报告 per-terrain success rate 和 MXD。

Expected result:

Task-targeted curriculum 应通过避免 easy tasks 主导共享 curriculum schedule，提升 beam gap 和 biased gap 等 hard terrains 的性能。

Output table:

| Method | Hurdle SR | Gap SR | Step SR | Beam Gap SR | Biased Gap SR | Avg SR |
|---|---:|---:|---:|---:|---:|---:|
| Global curriculum | | | | | | |
| Task-targeted curriculum | | | | | | |

## 8. Optional High-Value Experiment

### E7. Heading-Aware Distillation Ablation

该实验直接测试 proposal 中的第二项创新。

Goal:

测试显式 heading prediction 是否能提升 student 在 direction-sensitive terrains 上的性能。

Full comparison:

```text
H0: Original student distillation
H1: Heading-pretrained student
H2: Oracle heading upper bound
H3: Masked heading / no heading
```

Minimum comparison:

```text
H0: Original student distillation
H1: Heading-pretrained student
```

Required conditions:

- Headed GPU server。
- Camera rendering 可用。
- Depth image input 可用。
- Teacher heading labels 可提取。

Main terrains:

- Beam gap。
- Biased gap。
- Staggered or alternating obstacle sequence。
- Parkour v2。

Metrics:

- Success Rate。
- MXD。
- Heading angular error。
- Lateral deviation。
- Fall Rate。
- MEV。

Expected result:

Heading-aware distillation 应减少 heading drift，并提升 narrow 或 misaligned obstacles 上的 success rate。

Backup plan:

如果完整 heading-aware student training 不稳定，则只训练 heading predictor 并评估 heading prediction error。这仍可支持显式 heading supervision 有意义的论点。

## 9. Recommended Execution Order

1. 添加 per-env evaluation logging。
2. 在 server 上验证 teacher/base checkpoint。
3. 使用 `--use_camera` 训练 baseline student。
4. 在 original standard terrains 上评估 teacher 和 student。
5. 实现或验证 new terrain generation。
6. 在 new terrains 上运行 zero-shot evaluation。
7. 在 new terrains 上 fine-tune teacher/base policy。
8. 如果时间允许，distill fine-tuned student。
9. 运行 task-targeted curriculum ablation。
10. 如果 distillation pipeline 稳定，运行 heading-aware distillation ablation。
11. 记录成功和失败案例的 qualitative videos。
12. 生成最终 tables 和 plots。

## 10. Minimum Acceptable Final Deliverable

如果时间有限，final report 至少必须包含：

1. 完整或明确限定范围的 baseline reproduction。
2. Standard obstacle evaluation。
3. 至少两个 new terrains。
4. Baseline 在 new terrains 上的 zero-shot performance。
5. New terrains 上的 fine-tuning 或 retraining result。
6. Success rate、MXD、fall/stuck analysis。
7. Qualitative success and failure videos。

Recommended minimum new terrains:

```text
Beam gap
Biased gap
```

Recommended minimum policies:

```text
Original teacher/base
Baseline student
Fine-tuned teacher/base
```

如果 student distillation 变得不稳定，应在报告中清楚说明，并将 teacher/base experiments 作为主要 simulation results。不过，在 headed server 可用的情况下，优先目标仍是包含 student policy。

## 11. Final Report Structure

Recommended experiment section:

```text
4. Experiments

4.1 Experimental Setup
Hardware, Isaac Gym setup, policy checkpoints, terrain settings.

4.2 Baseline Reproduction
Teacher/base and student/deployable policy reproduction.

4.3 Standard Terrain Evaluation
Hurdle, gap, step, ramp, and mixed parkour.

4.4 New Terrain Zero-Shot Evaluation
Beam gap, biased gap, alternating step, slanted hurdle, parkour v2.

4.5 Fine-Tuning on New Terrains
Before/after comparison.

4.6 Task-Targeted Curriculum Ablation
Global curriculum vs proposed curriculum.

4.7 Heading-Aware Distillation
Original student vs heading-pretrained student.

4.8 Failure Case Analysis
Falling, collision, stuck, heading drift, edge stepping.
```

## 12. Key Claims to Support

实验应支持以下 claims：

1. Extreme Parkour baseline 可以在 Isaac Gym 中复现。
2. Baseline 在 standard parkour terrains 上表现良好。
3. 提出的 new terrains 暴露了 generalization failures，尤其是 heading-sensitive failures。
4. Fine-tuning 或 retraining 能提升 new terrains 上的性能。
5. Task-targeted curriculum 能改善不同 obstacle types 之间的 performance balance。
6. Heading-aware distillation 能改善 direction-sensitive student policy behavior。
