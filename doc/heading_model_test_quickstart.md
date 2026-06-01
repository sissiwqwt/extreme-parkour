# Heading Model 训练服务器运行指南

目标：在训练服务器上系统比较 depth distillation baseline、heading-model、task-targeted curriculum，并完成训练、play、evaluate、JIT 推理 smoke test。

本文默认当前仓库结构保持为：

```text
project-extreme-parkour/
  extreme-parkour/
    legged_gym/
    rsl_rl/
    checkpoints/base_train/model_49500.pt
```

当前重点分支：

- `heading-model-B`：body-frame current/next heading vector，`heading_dim=4`。
- `heading-model`：兼容版 heading pretrain，可作为 A/B 对照。
- baseline：不传 `--enable_heading_model`。

## 0. 环境和变量

在训练服务器上：

```bash
cd /path/to/project-extreme-parkour
conda activate parkour

export TASK=a1
export DEVICE=cuda:0
export RL_DEVICE=cuda:0
export PROJ=parkour_heading_server
export TEACHER_RUN=base_train
export TEACHER_CKPT=49500
```

如果 teacher checkpoint 不在 `legged_gym/logs/<proj>/<run>/model_<iter>.pt` 结构下，建议创建 symlink，而不是改训练代码：

```bash
mkdir -p extreme-parkour/legged_gym/logs/$PROJ
ln -sfn "$(pwd)/extreme-parkour/checkpoints/base_train" \
  "extreme-parkour/legged_gym/logs/$PROJ/$TEACHER_RUN"
```

这样加载路径会解析到：

```text
extreme-parkour/legged_gym/logs/parkour_heading_server/base_train/model_49500.pt
```

## 1. 本地已通过的 Smoke Test

训练 smoke：

```bash
./test/smoke_test_headingModel.sh local
```

play/evaluate/JIT smoke：

```bash
./test/smoke_test_headingModel_play_inference.sh all
```

训练服务器先跑较大的 server smoke：

```bash
./test/smoke_test_headingModel.sh server
```

通过标准：

- 能加载 teacher checkpoint。
- 能完成 heading pretrain 和 action distillation 阶段切换。
- log 中出现 `Heading pretrain: True/False`、`Loss_depth/heading`、`Loss_depth/depth_actor`。
- 无 `shape mismatch`、`size mismatch`、`CUDA device-side assert`、OOM。
- play/evaluate 能加载生成的 student checkpoint。
- JIT forward 输出 action shape 为 `(1, 12)`，heading dim 为 `4`。

## 2. 训练矩阵

建议至少跑 4 组：

| Run | Heading model | Terrain curriculum | Task-targeted curriculum | 用途 |
|---|---:|---:|---:|---|
| `baseline_depth_ttc_on_2000` | no | on | on | baseline + TTC |
| `baseline_depth_curr_off_2000` | no | off | off | baseline without curriculum |
| `heading_B_ttc_on_2000` | yes | on | on | proposed main run |
| `heading_B_curr_off_2000` | yes | off | off | heading without curriculum |

如果要单独比较“terrain curriculum on，但 task-targeted curriculum off”，需要临时把配置里的：

```python
task_targeted_curriculum = True
```

改成：

```python
task_targeted_curriculum = False
```

配置文件：

```text
extreme-parkour/legged_gym/legged_gym/envs/base/legged_robot_config.py
```

`--curriculum False` 会关闭 terrain curriculum 总开关，因此 task-targeted curriculum 也不会生效。

## 3. 训练命令

训练命令建议从 `extreme-parkour/` 目录运行：

```bash
cd /path/to/project-extreme-parkour/extreme-parkour
```

### 3.1 Baseline，不加 heading-model，TTC on

```bash
python legged_gym/legged_gym/scripts/train.py \
  --task $TASK \
  --device $DEVICE \
  --rl_device $RL_DEVICE \
  --use_camera \
  --resume \
  --resumeid $TEACHER_RUN \
  --checkpoint $TEACHER_CKPT \
  --proj_name $PROJ \
  --exptid baseline_depth_ttc_on_2000 \
  --max_iterations 2000
```

### 3.2 Baseline，不加 heading-model，curriculum off

```bash
python legged_gym/legged_gym/scripts/train.py \
  --task $TASK \
  --device $DEVICE \
  --rl_device $RL_DEVICE \
  --use_camera \
  --resume \
  --resumeid $TEACHER_RUN \
  --checkpoint $TEACHER_CKPT \
  --proj_name $PROJ \
  --exptid baseline_depth_curr_off_2000 \
  --max_iterations 2000 \
  --curriculum False
```

### 3.3 Heading-model-B，TTC on

```bash
git checkout heading-model-B

python legged_gym/legged_gym/scripts/train.py \
  --task $TASK \
  --device $DEVICE \
  --rl_device $RL_DEVICE \
  --use_camera \
  --resume \
  --resumeid $TEACHER_RUN \
  --checkpoint $TEACHER_CKPT \
  --proj_name $PROJ \
  --exptid heading_B_ttc_on_2000 \
  --enable_heading_model \
  --heading_pretrain_iters 500 \
  --max_iterations 2000
```

### 3.4 Heading-model-B，curriculum off

```bash
python legged_gym/legged_gym/scripts/train.py \
  --task $TASK \
  --device $DEVICE \
  --rl_device $RL_DEVICE \
  --use_camera \
  --resume \
  --resumeid $TEACHER_RUN \
  --checkpoint $TEACHER_CKPT \
  --proj_name $PROJ \
  --exptid heading_B_curr_off_2000 \
  --enable_heading_model \
  --heading_pretrain_iters 500 \
  --max_iterations 2000 \
  --curriculum False
```

### 3.5 Heading-model A 对照

```bash
git checkout heading-model

python legged_gym/legged_gym/scripts/train.py \
  --task $TASK \
  --device $DEVICE \
  --rl_device $RL_DEVICE \
  --use_camera \
  --resume \
  --resumeid $TEACHER_RUN \
  --checkpoint $TEACHER_CKPT \
  --proj_name $PROJ \
  --exptid heading_A_ttc_on_2000 \
  --enable_heading_model \
  --heading_pretrain_iters 500 \
  --max_iterations 2000
```

## 4. Resume 长训

如果 2000 iter 初步可行，继续到 5000 或更长：

```bash
cd /path/to/project-extreme-parkour/extreme-parkour

python legged_gym/legged_gym/scripts/train.py \
  --task $TASK \
  --device $DEVICE \
  --rl_device $RL_DEVICE \
  --use_camera \
  --resume \
  --resumeid heading_B_ttc_on_2000 \
  --checkpoint -1 \
  --proj_name $PROJ \
  --exptid heading_B_ttc_on_5000 \
  --enable_heading_model \
  --heading_pretrain_iters 0 \
  --max_iterations 3000
```

说明：

- `--checkpoint -1` 加载该 run 下最后一个 `model_*.pt`。
- resume 后如果不想再次进入 heading-only pretrain，设 `--heading_pretrain_iters 0`。
- 当前 runner 会从 checkpoint iteration 后继续计数，并保存 final checkpoint。

## 5. Play Smoke / 可视化运行

`play.py` 的 log 路径是相对 `legged_gym/legged_gym/scripts` 写的，建议从 scripts 目录运行：

```bash
cd /path/to/project-extreme-parkour/extreme-parkour/legged_gym/legged_gym/scripts
```

### 5.1 Heading-model-B play

```bash
python play.py \
  --task $TASK \
  --device $DEVICE \
  --rl_device $RL_DEVICE \
  --proj_name $PROJ \
  --exptid heading_B_ttc_on_2000 \
  --checkpoint -1 \
  --use_camera \
  --enable_heading_model \
  --headless \
  --play_steps 200
```

### 5.2 Baseline play

```bash
python play.py \
  --task $TASK \
  --device $DEVICE \
  --rl_device $RL_DEVICE \
  --proj_name $PROJ \
  --exptid baseline_depth_ttc_on_2000 \
  --checkpoint -1 \
  --use_camera \
  --headless \
  --play_steps 200
```

如果需要窗口显示，去掉 `--headless`。如果需要 web viewer，加 `--web`，但这不建议作为 smoke 的第一步。

## 6. Evaluate 验证

同样从 scripts 目录运行：

```bash
cd /path/to/project-extreme-parkour/extreme-parkour/legged_gym/legged_gym/scripts
```

### 6.1 Heading-model-B evaluate

```bash
python evaluate.py \
  --task $TASK \
  --device $DEVICE \
  --rl_device $RL_DEVICE \
  --proj_name $PROJ \
  --exptid heading_B_ttc_on_2000 \
  --checkpoint -1 \
  --use_camera \
  --enable_heading_model
```

### 6.2 Baseline evaluate

```bash
python evaluate.py \
  --task $TASK \
  --device $DEVICE \
  --rl_device $RL_DEVICE \
  --proj_name $PROJ \
  --exptid baseline_depth_ttc_on_2000 \
  --checkpoint -1 \
  --use_camera
```

短测可加：

```bash
--eval_steps 100
```

正式记录建议使用默认 `1500` steps，记录：

- `Mean reward`
- `Mean episode length`
- `Mean number of waypoints`
- `Mean edge violation`

## 7. JIT / 推理导出

当前 `save_jit.py` 导出两类文件：

- actor JIT：
  - `<exptid>-<checkpoint>-base_jit.pt`
- depth encoder 权重和 heading metadata：
  - `<exptid>-<checkpoint>-vision_weight.pt`

从 scripts 目录运行：

```bash
cd /path/to/project-extreme-parkour/extreme-parkour/legged_gym/legged_gym/scripts
```

### 7.1 Heading-model-B JIT 导出

```bash
python save_jit.py \
  --proj_name $PROJ \
  --exptid heading_B_ttc_on_2000 \
  --checkpoint -1
```

导出目录：

```text
extreme-parkour/legged_gym/logs/<PROJ>/heading_B_ttc_on_2000/traced/
```

推理 smoke 检查：

```bash
cd /path/to/project-extreme-parkour

PROJ_NAME=$PROJ \
EXPTID=heading_B_ttc_on_2000 \
CHECKPOINT=-1 \
./test/smoke_test_headingModel_play_inference.sh jit_forward
```

通过标准：

- JIT actor 能加载。
- CPU forward 输出 `Actions shape: (1, 12)`。
- `Depth heading dim: 4`。

### 7.2 Baseline JIT 导出

```bash
python save_jit.py \
  --proj_name $PROJ \
  --exptid baseline_depth_ttc_on_2000 \
  --checkpoint -1
```

baseline 的 depth heading dim 通常为 `2`，heading-model-B 应为 `4`。

## 8. 推荐结果表

| Run | Branch | Heading | Curriculum | TTC | Pretrain iters | Mean reward | Waypoint ratio | Episode length | Edge violation |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_depth_ttc_on_2000 | heading-model-B | no | on | on | 0 | | | | |
| baseline_depth_curr_off_2000 | heading-model-B | no | off | off | 0 | | | | |
| heading_B_ttc_on_2000 | heading-model-B | yes | on | on | 500 | | | | |
| heading_B_curr_off_2000 | heading-model-B | yes | off | off | 500 | | | | |
| heading_A_ttc_on_2000 | heading-model | yes | on | on | 500 | | | | |

## 9. 判断标准

可行性：

- 训练能完成目标 iterations。
- play/evaluate 能加载 checkpoint。
- JIT 导出和 CPU forward 正常。
- heading-model-B 的 `vision_weight.pt` 中 `depth_heading_dim == 4`。

初步有效性：

- `Mean number of waypoints` 高于 baseline。
- 或 `Mean episode length` 高于 baseline。
- 或 `Mean edge violation` 低于 baseline。
- 训练中 `Loss_depth/heading` 在 pretrain 阶段下降，action distillation 阶段不出现 NaN/Inf。

继续长训建议：

- B 明显优于 baseline：继续 `heading_B_ttc_on_2000` resume 到 `5000+`。
- B 与 baseline 接近：尝试 `heading_pretrain_iters=100/200/500` ablation。
- B 不如 baseline：先降低 heading pretrain 时长，再检查 heading loss weight/action loss weight。
- TTC on 明显优于 curriculum off：后续正式结果保留 TTC。
