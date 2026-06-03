#!/usr/bin/env bash
set -Eeuo pipefail

# Runs the repaired heading-model-C experiment batch, evaluates each finished
# policy, records fixed-terrain distill_play videos, and uploads artifacts to
# Google Drive via rclone.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEGGED_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${LEGGED_ROOT}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
TASK="${TASK:-a1}"
DEVICE="${DEVICE:-cuda:0}"
RL_DEVICE="${RL_DEVICE:-${DEVICE}}"
PROJ_NAME="${PROJ_NAME:-parkour_heading}"
TEACHER_CHECKPOINT_PATH="${TEACHER_CHECKPOINT_PATH:-${1:-}}"

HEADING_PRETRAIN_ITERS="${HEADING_PRETRAIN_ITERS:-1000}"
MAX_ITERATIONS="${MAX_ITERATIONS:-7000}"
EVAL_EPISODES="${EVAL_EPISODES:-256}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-128}"
DISTILL_ENVS_PER_TERRAIN="${DISTILL_ENVS_PER_TERRAIN:-1}"
DISTILL_USE_GPU="${DISTILL_USE_GPU:-1}"
DISTILL_RECORD_CAMERA="${DISTILL_RECORD_CAMERA:-third_person}"

GDRIVE_REMOTE="${GDRIVE_REMOTE:-gdrive}"
GDRIVE_ROOT="${GDRIVE_ROOT:-extreme-parkour/heading_c_fix}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${LEGGED_ROOT}/logs/heading_c_fix_artifacts}"

CONFIG_PATH="${LEGGED_ROOT}/legged_gym/envs/base/legged_robot_config.py"
TRAIN_PY="${SCRIPT_DIR}/train.py"
EVAL_PY="${SCRIPT_DIR}/evaluation.py"
DISTILL_PLAY_PY="${SCRIPT_DIR}/distill_play.py"

usage() {
  cat <<EOF
Usage:
  TEACHER_CHECKPOINT_PATH=/path/to/model_XXXXX.pt bash $0

Optional environment variables:
  TASK=${TASK}
  DEVICE=${DEVICE}
  PROJ_NAME=${PROJ_NAME}
  GDRIVE_REMOTE=${GDRIVE_REMOTE}
  GDRIVE_ROOT=${GDRIVE_ROOT}
  EVAL_EPISODES=${EVAL_EPISODES}
  DISTILL_ENVS_PER_TERRAIN=${DISTILL_ENVS_PER_TERRAIN}
EOF
}

require_file() {
  local path="$1"
  local label="$2"
  if [[ -z "${path}" || ! -f "${path}" ]]; then
    echo "Missing ${label}: ${path}" >&2
    usage >&2
    exit 2
  fi
}

require_command() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "Required command not found: ${cmd}" >&2
    exit 2
  fi
}

gdrive_target() {
  local exptid="$1"
  echo "${GDRIVE_REMOTE}:${GDRIVE_ROOT}/${exptid}"
}

login_google_drive() {
  require_command rclone
  if ! rclone config show "${GDRIVE_REMOTE}" >/dev/null 2>&1; then
    echo "rclone remote '${GDRIVE_REMOTE}' is not configured. Opening rclone config..."
    rclone config
  fi
  echo "Checking Google Drive login for '${GDRIVE_REMOTE}:'."
  rclone lsd "${GDRIVE_REMOTE}:" >/dev/null
}

active_terrains_csv() {
  "${PYTHON_BIN}" - "${CONFIG_PATH}" <<'PY'
import ast
import re
import sys
from pathlib import Path

text = Path(sys.argv[1]).read_text()
match = re.search(r"terrain_dict\s*=\s*(\{.*?\})", text, flags=re.S)
if match is None:
    raise SystemExit("terrain_dict not found")
terrain_dict = ast.literal_eval(match.group(1))
names = [name for name, weight in terrain_dict.items() if float(weight) > 0.0]
print(",".join(names))
PY
}

make_distill_play_copy() {
  local terrain_csv="$1"
  local output_py="$2"
  "${PYTHON_BIN}" - "${DISTILL_PLAY_PY}" "${output_py}" "${terrain_csv}" "${DISTILL_ENVS_PER_TERRAIN}" <<'PY'
import pprint
import re
import sys
from pathlib import Path

src, dst, terrain_csv, envs_per_terrain = sys.argv[1:5]
terrain_names = [name for name in terrain_csv.split(",") if name]
envs_per_terrain = int(envs_per_terrain)
if envs_per_terrain <= 0:
    raise SystemExit("DISTILL_ENVS_PER_TERRAIN must be positive")
terrain_counts = {name: envs_per_terrain for name in terrain_names}

text = Path(src).read_text()
replacement = "DEFAULT_DISTILL_TERRAIN_ENVS = " + pprint.pformat(
    terrain_counts, sort_dicts=False
)
updated = re.sub(
    r"DEFAULT_DISTILL_TERRAIN_ENVS\s*=\s*\{.*?\n\}",
    replacement,
    text,
    count=1,
    flags=re.S,
)
if updated == text:
    raise SystemExit("failed to replace DEFAULT_DISTILL_TERRAIN_ENVS")
Path(dst).write_text(updated)
PY
}

latest_checkpoint() {
  local log_dir="$1"
  find "${log_dir}" -maxdepth 1 -type f -name "model_*.pt" | sort -V | tail -n 1
}

upload_dir() {
  local local_dir="$1"
  local remote_dir="$2"
  if [[ -d "${local_dir}" ]]; then
    rclone copy "${local_dir}" "${remote_dir}"
  fi
}

upload_file() {
  local local_file="$1"
  local remote_dir="$2"
  if [[ -f "${local_file}" ]]; then
    rclone copy "${local_file}" "${remote_dir}"
  fi
}

run_train() {
  local exptid="$1"
  local latent_weight="$2"
  local freeze_backbone="$3"

  echo "Training ${exptid}"
  (
    cd "${SCRIPT_DIR}"
    "${PYTHON_BIN}" "${TRAIN_PY}" \
      --task "${TASK}" \
      --device "${DEVICE}" \
      --rl_device "${RL_DEVICE}" \
      --proj_name "${PROJ_NAME}" \
      --exptid "${exptid}" \
      --use_camera \
      --enable_heading_model \
      --heading_pretrain_iters "${HEADING_PRETRAIN_ITERS}" \
      --heading_loss_weight 1.0 \
      --action_loss_weight 1.0 \
      --latent_loss_weight "${latent_weight}" \
      --freeze_backbone_during_action_distillation "${freeze_backbone}" \
      --teacher_checkpoint_path "${TEACHER_CHECKPOINT_PATH}" \
      --curriculum True \
      --task_targeted_curriculum False \
      --max_iterations "${MAX_ITERATIONS}"
  )
}

run_evaluation() {
  local exptid="$1"
  local terrain_csv="$2"
  local output_dir="${ARTIFACT_ROOT}/${exptid}/evaluation"

  mkdir -p "${output_dir}"
  echo "Evaluating ${exptid}"
  (
    cd "${SCRIPT_DIR}"
    "${PYTHON_BIN}" "${EVAL_PY}" \
      --task "${TASK}" \
      --device "${DEVICE}" \
      --rl_device "${RL_DEVICE}" \
      --use_camera \
      --enable_heading_model \
      --proj_name "${PROJ_NAME}" \
      --exptid "${exptid}" \
      --checkpoint -1 \
      --policy_id "${exptid}" \
      --policy_type depth \
      --terrain_names "${terrain_csv}" \
      --difficulty_mode all-difficulty \
      --eval_episodes "${EVAL_EPISODES}" \
      --num_envs "${EVAL_NUM_ENVS}" \
      --output_dir "${output_dir}" \
      --headless
  )
}

run_distill_videos() {
  local exptid="$1"
  local terrain_csv="$2"
  local temp_distill="${SCRIPT_DIR}/.distill_play_${exptid}_all_active.py"
  local distill_gpu_args=()

  mkdir -p "${ARTIFACT_ROOT}/${exptid}"
  make_distill_play_copy "${terrain_csv}" "${temp_distill}"

  if [[ "${DISTILL_USE_GPU}" == "1" || "${DISTILL_USE_GPU}" == "true" || "${DISTILL_USE_GPU}" == "True" ]]; then
    distill_gpu_args+=(--use_gpu)
  fi

  for difficulty in 0.7 1.0; do
    local video_dir="${ARTIFACT_ROOT}/${exptid}/videos/difficulty_${difficulty}"
    mkdir -p "${video_dir}"
    echo "Recording ${exptid} distill_play videos at difficulty ${difficulty}"
    (
      cd "${SCRIPT_DIR}"
      "${PYTHON_BIN}" "${temp_distill}" \
        --task "${TASK}" \
        --device "${DEVICE}" \
        --rl_device "${RL_DEVICE}" \
        --use_camera \
        --enable_heading_model \
        --proj_name "${PROJ_NAME}" \
        --exptid "${exptid}" \
        --checkpoint -1 \
        --video_out "${video_dir}" \
        --record_camera "${DISTILL_RECORD_CAMERA}" \
        --terrain_difficulty "${difficulty}" \
        --headless \
        "${distill_gpu_args[@]}"
    )
  done
}

upload_artifacts() {
  local exptid="$1"
  local log_dir="${LEGGED_ROOT}/logs/${PROJ_NAME}/${exptid}"
  local ckpt
  ckpt="$(latest_checkpoint "${log_dir}")"
  if [[ -z "${ckpt}" ]]; then
    echo "No checkpoint found in ${log_dir}" >&2
    exit 1
  fi

  local remote
  remote="$(gdrive_target "${exptid}")"
  echo "Uploading artifacts for ${exptid} to ${remote}"
  upload_file "${ckpt}" "${remote}/checkpoint"
  upload_dir "${ARTIFACT_ROOT}/${exptid}/evaluation" "${remote}/evaluation"
  upload_dir "${ARTIFACT_ROOT}/${exptid}/videos" "${remote}/videos"
}

main() {
  require_file "${TEACHER_CHECKPOINT_PATH}" "TEACHER_CHECKPOINT_PATH"
  require_file "${CONFIG_PATH}" "terrain config"
  require_file "${TRAIN_PY}" "train.py"
  require_file "${EVAL_PY}" "evaluation.py"
  require_file "${DISTILL_PLAY_PY}" "distill_play.py"

  mkdir -p "${ARTIFACT_ROOT}"
  login_google_drive

  local terrains
  terrains="$(active_terrains_csv)"
  if [[ -z "${terrains}" ]]; then
    echo "No active terrains found in ${CONFIG_PATH}" >&2
    exit 1
  fi
  echo "Active terrains: ${terrains}"

  local experiments=(
    "heading_c_fix_pre1000_latent1_unfreeze|1.0|False"
    "heading_c_fix_pre1000_latent1_freeze|1.0|True"
    "heading_c_fix_pre1000_latent025_unfreeze|0.25|False"
    "heading_c_fix_pre1000_latent2_unfreeze|2.0|False"
  )

  for spec in "${experiments[@]}"; do
    IFS="|" read -r exptid latent_weight freeze_backbone <<<"${spec}"
    run_train "${exptid}" "${latent_weight}" "${freeze_backbone}"
    run_evaluation "${exptid}" "${terrains}"
    run_distill_videos "${exptid}" "${terrains}"
    upload_artifacts "${exptid}"
  done
}

main "$@"
