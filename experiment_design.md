# Experiment Design: Adaptive Quadruped Robot Parkour

## 1. Project Goal

This project follows the course topic "Quadruped robot parkour" and builds on the Extreme Parkour baseline. The goal is to train and evaluate a quadruped robot in Isaac Gym that can overcome complex obstacles using proprioception and vision.

The project has two levels of goals:

1. Reproduce the Extreme Parkour baseline in simulation.
2. Verify the proposed improvements from our project proposal:
   - New parkour terrains: beam gap, biased gap, alternating step, slanted hurdle, and parkour v2.
   - Task-targeted curriculum learning.
   - Heading-aware student distillation.

## 2. Experimental Platform

### Main Platform

All official training, distillation, quantitative evaluation, and demo recording should be conducted on the headed GPU server.

The headed server is preferred because:

- Student distillation requires camera rendering and Isaac Gym GPU camera tensors.
- Large-scale evaluation can use more parallel environments, such as 128 or 256 envs.
- Demo recording and camera-based playback are more stable.
- All reported results come from the same hardware and software environment.

### Local Machine

The local RTX 4060 machine is used only for:

- Quick checkpoint sanity checks.
- Single-robot visualization.
- Debugging playback commands.
- Backup demo inspection.

Local playback results should not be used as the main quantitative results unless server-side evaluation is unavailable.

## 3. Policies to Train and Evaluate

### P0: Teacher / Base Policy

This is the phase-1 Extreme Parkour policy.

Inputs:

- Proprioception.
- Privileged terrain scandots.
- Oracle heading / waypoint direction.

Current checkpoint:

```text
logs/parkour_new/01-trainn-base/model_49500.pt
```

This policy is useful for validating basic parkour ability and for serving as the teacher for distillation.

### P1: Baseline Student / Deployable Policy

This is the phase-2 distilled policy from the original Extreme Parkour pipeline.

Inputs:

- Proprioception.
- Front-facing depth image.

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

This policy is required for a full baseline reproduction because the original paper's deployable policy uses depth images.

### P2: Fine-Tuned Policy on New Terrains

This policy is fine-tuned or retrained on our newly designed terrains.

At minimum, train a fine-tuned teacher/base policy. If time permits, distill a corresponding student policy from it.

### P3: Proposed Policy with Task-Targeted Curriculum

This policy uses the proposed task-targeted curriculum instead of the original global curriculum.

It is used to test whether independent per-terrain curriculum states improve training stability and final performance.

### P4: Heading-Aware Student Policy

This policy uses the proposed heading-aware distillation pipeline.

It explicitly pretrains a heading prediction module before or during action distillation.

This is a high-value but higher-risk experiment.

## 4. Metrics

All quantitative experiments should report the following metrics.

### Success Rate

The percentage of episodes where the robot successfully reaches the target course endpoint or reaches enough waypoints.

Recommended implementation:

```text
success = cur_goal_idx >= terrain.num_goals
```

or, if full-course completion is too strict:

```text
success = normalized_waypoints >= threshold
```

The threshold must be stated clearly in the report.

### Mean X-Displacement (MXD)

Average forward displacement before failure or timeout.

```text
MXD = final_root_x - start_root_x
```

Higher is better.

### Fall Rate

Percentage of episodes terminated by falling rather than timeout or successful completion.

Falling can be detected by reset reasons such as:

- Excessive roll.
- Excessive pitch.
- Base height below threshold.

### Stuck Rate

Percentage of episodes where the robot remains alive but makes insufficient forward progress.

Recommended definition:

```text
stuck = forward displacement over the last 2 seconds < 0.1 m
```

The exact threshold can be adjusted, but it must be fixed before evaluation.

### Collision / Failure Cases

Collision may not be directly available as a clean label in the current environment. It can be analyzed using:

- Visual inspection.
- Repeated contact with obstacle followed by no progress.
- Sudden velocity loss near an obstacle.
- Failure reason annotation from qualitative videos.

### Mean Edge Violation (MEV)

This follows the Extreme Parkour paper.

Recommended proxy:

```text
MEV = average feet_at_edge contact count per timestep
```

Lower is better.

### Heading Error

Used only for heading-aware distillation experiments.

```text
heading_error = arccos(dot(pred_heading, teacher_heading))
```

Lower is better.

### Qualitative Video

For each main terrain, record:

- One successful rollout.
- One representative failure rollout.

Failure videos should be categorized as:

- Falling.
- Collision.
- Stuck.
- Heading drift.
- Edge stepping.
- Bad takeoff or landing.

## 5. Required Logging

The current `play.py` is mainly a visualization script and does not save per-env results. Before running official experiments, add an evaluation logger to `play.py` or create a dedicated evaluation script.

The logger should save one row per episode:

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

Without per-env logging, success rate, MXD, fall rate, and stuck rate cannot be reported reliably.

## 6. Must-Do Experiments

The following experiments are required because they directly satisfy the course 2.4 requirements.

### E1. Full Baseline Reproduction

Course requirement:

```text
Reproduce a baseline parkour policy in Isaac Gym.
```

Goal:

Reproduce both the teacher/base policy and the student/deployable policy from Extreme Parkour.

Required conditions:

- Isaac Gym environment works on the headed GPU server.
- Teacher/base checkpoint is available.
- `--use_camera` training works for student distillation.
- Depth camera tensors render correctly.

Steps:

1. Verify the teacher/base checkpoint:

```bash
python play.py \
  --exptid 01-tra \
  --task a1 \
  --device cuda:0 \
  --rl_device cuda:0 \
  --checkpoint 49500 \
  --num_envs 1
```

2. Train baseline student:

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

3. Evaluate teacher and student on the same standard terrains.

Expected result:

- Teacher should show strong performance on original parkour terrains.
- Student should approach teacher performance, though some performance drop is acceptable.

### E2. Standard Obstacle Evaluation

Course requirement:

```text
Validate the robot's performance on standard obstacle tracks.
```

Goal:

Evaluate teacher and student policies on the original Extreme Parkour terrain types.

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

- Success Rate.
- MXD.
- Fall Rate.
- Stuck Rate.
- MEV.
- Qualitative video.

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

Show that the newly designed terrains are challenging and reveal limitations of the baseline.

New terrains:

- Beam gap.
- Biased gap.
- Alternating step.
- Slanted hurdle.
- Parkour v2.

Note:

Use one consistent name in the final report: "beam gap".

Evaluation:

- No fine-tuning.
- Test P0 teacher and P1 student directly.
- Use the same evaluation metrics as E2.

Expected observations:

- Beam gap: lateral deviation causes falls.
- Biased gap: wrong heading causes bad takeoff or landing.
- Alternating step: unstable gait rhythm or stuck behavior.
- Slanted hurdle: poor adaptation to direction changes.
- Parkour v2: mixed terrain generalization failures.

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

Improve performance on the new terrains.

Minimum required version:

- Fine-tune the teacher/base policy on the new terrains.

Preferred version:

- Fine-tune teacher/base policy on new terrains.
- Distill a new student policy from the fine-tuned teacher.

Policy comparison:

```text
P0: Original teacher
P1: Original student
P2: Fine-tuned teacher
P2-student: Distilled student from fine-tuned teacher
```

Evaluation:

- Test on both original terrains and new terrains.
- This checks improvement and possible regression.

Expected result:

- New terrain success rate increases after fine-tuning.
- Original terrain performance should not collapse.

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

Explain not only whether the policy fails, but why it fails.

For each main terrain, collect:

- One successful video.
- One failure video.
- Failure type.
- Short explanation.

Failure categories:

- Fall.
- Collision.
- Stuck.
- Heading drift.
- Edge stepping.
- Bad takeoff.
- Bad landing.

Example table:

| Terrain | Policy | Case | Failure Type | Description |
|---|---|---|---|---|
| Beam gap | Student | Failure | Heading drift | Robot approaches the narrow beam at an angle and falls laterally. |
| Biased gap | Teacher | Failure | Bad landing | Robot jumps from the wrong side and lands partially outside the platform. |

## 7. Strongly Recommended Innovation Experiment

### E6. Task-Targeted Curriculum Ablation

This is the most practical innovation experiment from the project proposal.

Goal:

Compare the original global curriculum with the proposed task-targeted curriculum.

Methods:

```text
C0: Original global curriculum
C1: Task-targeted curriculum
```

Training conditions:

- Same initial checkpoint or same from-scratch setup.
- Same total training iterations.
- Same number of environments.
- Same terrain distribution.
- Same random seeds if possible.

Logged training curves:

- Per-task success rate.
- Per-task terrain level.
- Per-task episode length.
- Overall reward.
- Curriculum level oscillation.

Evaluation:

- Test both C0 and C1 on original and new terrains.
- Report per-terrain success rate and MXD.

Expected result:

Task-targeted curriculum should improve hard terrains such as beam gap and biased gap by preventing easy tasks from dominating the shared curriculum schedule.

Output table:

| Method | Hurdle SR | Gap SR | Step SR | Beam Gap SR | Biased Gap SR | Avg SR |
|---|---:|---:|---:|---:|---:|---:|
| Global curriculum | | | | | | |
| Task-targeted curriculum | | | | | | |

## 8. Optional High-Value Experiment

### E7. Heading-Aware Distillation Ablation

This experiment directly tests the second innovation in the proposal.

Goal:

Test whether explicit heading prediction improves student performance on direction-sensitive terrains.

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

- Headed GPU server.
- Camera rendering works.
- Depth image input works.
- Teacher heading labels can be extracted.

Main terrains:

- Beam gap.
- Biased gap.
- Staggered or alternating obstacle sequence.
- Parkour v2.

Metrics:

- Success Rate.
- MXD.
- Heading angular error.
- Lateral deviation.
- Fall Rate.
- MEV.

Expected result:

Heading-aware distillation should reduce heading drift and improve success rate on narrow or misaligned obstacles.

Backup plan:

If full heading-aware student training is not stable, train only the heading predictor and evaluate heading prediction error. This still supports the claim that explicit heading supervision is meaningful.

## 9. Recommended Execution Order

1. Add per-env evaluation logging.
2. Verify teacher/base checkpoint on the server.
3. Train baseline student with `--use_camera`.
4. Evaluate teacher and student on original standard terrains.
5. Implement or verify new terrain generation.
6. Run zero-shot evaluation on new terrains.
7. Fine-tune teacher/base policy on new terrains.
8. Distill fine-tuned student if time permits.
9. Run task-targeted curriculum ablation.
10. Run heading-aware distillation ablation if the distillation pipeline is stable.
11. Record qualitative videos for success and failure cases.
12. Generate final tables and plots.

## 10. Minimum Acceptable Final Deliverable

If time is limited, the final report must include at least:

1. Full or clearly scoped baseline reproduction.
2. Standard obstacle evaluation.
3. At least two new terrains.
4. Zero-shot baseline performance on the new terrains.
5. Fine-tuning or retraining result on the new terrains.
6. Success rate, MXD, fall/stuck analysis.
7. Qualitative success and failure videos.

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

If student distillation becomes unstable, state this clearly and report the teacher/base experiments as the main simulation results. However, with the headed server available, the preferred target is to include the student policy.

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

The experiments should support the following claims:

1. The Extreme Parkour baseline can be reproduced in Isaac Gym.
2. The baseline performs well on standard parkour terrains.
3. The proposed new terrains expose generalization failures, especially heading-sensitive failures.
4. Fine-tuning or retraining improves performance on new terrains.
5. Task-targeted curriculum improves performance balance across obstacle types.
6. Heading-aware distillation improves direction-sensitive student policy behavior.
