#!/usr/bin/env bash
set -Eeuo pipefail

# Runs the repaired heading-model-C experiment batch, evaluates each finished
# policy, records fixed-terrain distill_play videos, and uploads artifacts to
# the selected cloud drive via rclone.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEGGED_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
REPO_ROOT="$(cd "${LEGGED_ROOT}/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
TASK="${TASK:-a1}"
DEVICE="${DEVICE:-cuda:0}"
RL_DEVICE="${RL_DEVICE:-${DEVICE}}"
PROJ_NAME="${PROJ_NAME:-parkour_heading}"
TEACHER_CHECKPOINT_PATH="${TEACHER_CHECKPOINT_PATH:-}"
UPLOAD_DRIVE="${UPLOAD_DRIVE:-quark}"

HEADING_PRETRAIN_ITERS="${HEADING_PRETRAIN_ITERS:-1000}"
MAX_ITERATIONS="${MAX_ITERATIONS:-5000}"
EVAL_EPISODES="${EVAL_EPISODES:-256}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-128}"
DISTILL_ENVS_PER_TERRAIN="${DISTILL_ENVS_PER_TERRAIN:-1}"
DISTILL_USE_GPU="${DISTILL_USE_GPU:-1}"
DISTILL_RECORD_CAMERA="${DISTILL_RECORD_CAMERA:-third_person}"

QUARK_REMOTE="${QUARK_REMOTE:-quark}"
QUARK_ROOT="${QUARK_ROOT:-extreme-parkour/heading_c_fix}"
BAIDU_REMOTE="${BAIDU_REMOTE:-baidu}"
BAIDU_ROOT="${BAIDU_ROOT:-extreme-parkour/heading_c_fix}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${LEGGED_ROOT}/logs/heading_c_fix_artifacts}"

# If the selected rclone remote is unavailable or unreachable, skip uploads.
UPLOAD_AVAILABLE=0
UPLOAD_REMOTE=""
UPLOAD_ROOT=""
UPLOAD_NAME=""

CONFIG_PATH="${LEGGED_ROOT}/legged_gym/envs/base/legged_robot_config.py"
TRAIN_PY="${SCRIPT_DIR}/train.py"
EVAL_PY="${SCRIPT_DIR}/evaluation.py"
DISTILL_PLAY_PY="${SCRIPT_DIR}/distill_play.py"

usage() {
  cat <<EOF
Usage:
  TEACHER_CHECKPOINT_PATH=/path/to/model_XXXXX.pt bash $0 --drive quark
  bash $0 --drive baidu /path/to/model_XXXXX.pt

Optional environment variables:
  TASK=${TASK}
  DEVICE=${DEVICE}
  PROJ_NAME=${PROJ_NAME}
  UPLOAD_DRIVE=${UPLOAD_DRIVE}
  QUARK_REMOTE=${QUARK_REMOTE}
  QUARK_ROOT=${QUARK_ROOT}
  BAIDU_REMOTE=${BAIDU_REMOTE}
  BAIDU_ROOT=${BAIDU_ROOT}
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

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --drive|--upload-drive)
        if [[ $# -lt 2 ]]; then
          echo "Missing value for $1" >&2
          usage >&2
          exit 2
        fi
        UPLOAD_DRIVE="$2"
        shift 2
        ;;
      --drive=*|--upload-drive=*)
        UPLOAD_DRIVE="${1#*=}"
        shift
        ;;
      --teacher-checkpoint-path)
        if [[ $# -lt 2 ]]; then
          echo "Missing value for $1" >&2
          usage >&2
          exit 2
        fi
        TEACHER_CHECKPOINT_PATH="$2"
        shift 2
        ;;
      --teacher-checkpoint-path=*)
        TEACHER_CHECKPOINT_PATH="${1#*=}"
        shift
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      --)
        shift
        break
        ;;
      -*)
        echo "Unknown option: $1" >&2
        usage >&2
        exit 2
        ;;
      *)
        if [[ -z "${TEACHER_CHECKPOINT_PATH}" ]]; then
          TEACHER_CHECKPOINT_PATH="$1"
        else
          echo "Unexpected extra argument: $1" >&2
          usage >&2
          exit 2
        fi
        shift
        ;;
    esac
  done

  while [[ $# -gt 0 ]]; do
    if [[ -z "${TEACHER_CHECKPOINT_PATH}" ]]; then
      TEACHER_CHECKPOINT_PATH="$1"
    else
      echo "Unexpected extra argument: $1" >&2
      usage >&2
      exit 2
    fi
    shift
  done
}

configure_upload_drive() {
  case "${UPLOAD_DRIVE}" in
    quark)
      UPLOAD_REMOTE="${QUARK_REMOTE}"
      UPLOAD_ROOT="${QUARK_ROOT}"
      UPLOAD_NAME="Quark Cloud Drive"
      ;;
    baidu)
      UPLOAD_REMOTE="${BAIDU_REMOTE}"
      UPLOAD_ROOT="${BAIDU_ROOT}"
      UPLOAD_NAME="Baidu Netdisk"
      ;;
    none|skip|off)
      UPLOAD_REMOTE=""
      UPLOAD_ROOT=""
      UPLOAD_NAME="upload"
      UPLOAD_AVAILABLE=0
      ;;
    *)
      echo "Unsupported upload drive '${UPLOAD_DRIVE}'. Use one of: quark, baidu, none." >&2
      usage >&2
      exit 2
      ;;
  esac
}

upload_target() {
  local exptid="$1"
  echo "${UPLOAD_REMOTE}:${UPLOAD_ROOT}/${exptid}"
}

login_upload_drive() {
  configure_upload_drive

  if [[ "${UPLOAD_DRIVE}" == "none" || "${UPLOAD_DRIVE}" == "skip" || "${UPLOAD_DRIVE}" == "off" ]]; then
    echo "Upload disabled by --drive ${UPLOAD_DRIVE}." >&2
    return 0
  fi

  # If rclone is not installed or remote is not configured/unreachable,
  # mark UPLOAD_AVAILABLE=0 and continue; uploads will be skipped.
  if ! command -v rclone >/dev/null 2>&1; then
    echo "rclone not found; skipping ${UPLOAD_NAME} uploads." >&2
    UPLOAD_AVAILABLE=0
    return 0
  fi

  if ! rclone config show "${UPLOAD_REMOTE}" >/dev/null 2>&1; then
    echo "rclone remote '${UPLOAD_REMOTE}' is not configured; skipping uploads." >&2
    UPLOAD_AVAILABLE=0
    return 0
  fi

  echo "Checking ${UPLOAD_NAME} login for '${UPLOAD_REMOTE}:'..."
  if ! rclone lsd "${UPLOAD_REMOTE}:" >/dev/null 2>&1; then
    echo "Unable to reach ${UPLOAD_NAME} remote '${UPLOAD_REMOTE}:'. uploads will be skipped." >&2
    UPLOAD_AVAILABLE=0
    return 0
  fi

  UPLOAD_AVAILABLE=1
  echo "${UPLOAD_NAME} remote '${UPLOAD_REMOTE}' is available."
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
    if [[ "${UPLOAD_AVAILABLE}" != "1" ]]; then
      echo "${UPLOAD_NAME} unavailable, skipping upload of directory ${local_dir}" >&2
      return 0
    fi
    if ! rclone copy "${local_dir}" "${remote_dir}"; then
      echo "Warning: failed to upload directory ${local_dir} to ${remote_dir}" >&2
    fi
  fi
}

upload_file() {
  local local_file="$1"
  local remote_dir="$2"
  if [[ -f "${local_file}" ]]; then
    if [[ "${UPLOAD_AVAILABLE}" != "1" ]]; then
      echo "${UPLOAD_NAME} unavailable, skipping upload of file ${local_file}" >&2
      return 0
    fi
    if ! rclone copy "${local_file}" "${remote_dir}"; then
      echo "Warning: failed to upload file ${local_file} to ${remote_dir}" >&2
    fi
  fi
}

run_train() {
  local exptid="$1"
  local latent_weight="$2"
  local freeze_backbone="$3"

  echo "Training ${exptid}"
  local log_dir="${LEGGED_ROOT}/logs/${PROJ_NAME}/${exptid}"
  mkdir -p "${log_dir}"
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
    echo "No checkpoint found in ${log_dir}, skipping upload for ${exptid}" >&2
    return 0
  fi

  if [[ "${UPLOAD_AVAILABLE}" != "1" ]]; then
    echo "Skipping upload for ${exptid} because ${UPLOAD_NAME} is unavailable." >&2
    return 0
  fi

  local remote
  remote="$(upload_target "${exptid}")"
  echo "Uploading artifacts for ${exptid} to ${remote}"
  upload_file "${ckpt}" "${remote}/checkpoint"
  upload_dir "${ARTIFACT_ROOT}/${exptid}/evaluation" "${remote}/evaluation"
  upload_dir "${ARTIFACT_ROOT}/${exptid}/videos" "${remote}/videos"
}

main() {
  parse_args "$@"

  require_file "${TEACHER_CHECKPOINT_PATH}" "TEACHER_CHECKPOINT_PATH"
  require_file "${CONFIG_PATH}" "terrain config"
  require_file "${TRAIN_PY}" "train.py"
  require_file "${EVAL_PY}" "evaluation.py"
  require_file "${DISTILL_PLAY_PY}" "distill_play.py"

  mkdir -p "${ARTIFACT_ROOT}"
  login_upload_drive

  local terrains
  terrains="$(active_terrains_csv)"
  if [[ -z "${terrains}" ]]; then
    echo "No active terrains found in ${CONFIG_PATH}" >&2
    exit 1
  fi
  echo "Active terrains: ${terrains}"

  local experiments=(
    "heading_pre1000_latent1_unfreeze|1.0|False"
    "heading_pre1000_latent1_freeze|1.0|True"
    "heading_pre1000_latent025_unfreeze|0.25|False"
    "heading_pre1000_latent2_unfreeze|2.0|False"
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
