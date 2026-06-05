#!/usr/bin/env bash
set -Eeuo pipefail

# Minimal heading-model-C ablation batch. By default this does not retrain the
# existing 1000-pretrain C-main run; it only evaluates it as a reference, then
# trains/evaluates the no-pretrain and short-pretrain ablations.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEGGED_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
TASK="${TASK:-a1}"
DEVICE="${DEVICE:-cuda:0}"
RL_DEVICE="${RL_DEVICE:-${DEVICE}}"
PROJ_NAME="${PROJ_NAME:-parkour_heading}"
TEACHER_CHECKPOINT_PATH="${TEACHER_CHECKPOINT_PATH:-}"
UPLOAD_DRIVE="${UPLOAD_DRIVE:-quark}"
RUN_MODE="${RUN_MODE:-full}"

MAX_ITERATIONS="${MAX_ITERATIONS:-5000}"
EVAL_EPISODES="${EVAL_EPISODES:-256}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-128}"
DISTILL_ENVS_PER_TERRAIN="${DISTILL_ENVS_PER_TERRAIN:-1}"
DISTILL_USE_GPU="${DISTILL_USE_GPU:-1}"
DISTILL_RECORD_CAMERA="${DISTILL_RECORD_CAMERA:-third_person}"

REFERENCE_EXPTID="${REFERENCE_EXPTID:-heading_pre1000_latent1_unfreeze}"
INCLUDE_REFERENCE="${INCLUDE_REFERENCE:-1}"
EXPERIMENT_SPECS="${EXPERIMENT_SPECS:-heading_pre0_latent1_unfreeze|0|1.0|False;heading_pre500_latent1_unfreeze|500|1.0|False}"

QUARK_REMOTE="${QUARK_REMOTE:-quark}"
QUARK_ROOT="${QUARK_ROOT:-extreme-parkour/heading_minimal_ablation}"
BAIDU_REMOTE="${BAIDU_REMOTE:-baidu}"
BAIDU_ROOT="${BAIDU_ROOT:-extreme-parkour/heading_minimal_ablation}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${LEGGED_ROOT}/logs/heading_minimal_ablation_artifacts}"

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
  TEACHER_CHECKPOINT_PATH=/path/to/model_XXXXX.pt bash $0 --mode full --drive quark
  bash $0 --mode train-only --drive none /path/to/model_XXXXX.pt

Modes:
  train-only         Train only.
  train-upload       Train, then upload checkpoints.
  train-eval-upload  Train, upload checkpoints, evaluate, then upload evaluation.
  full               Train, record videos, evaluate, then upload checkpoints/evaluation/videos.

Default ablations:
  ${EXPERIMENT_SPECS}

Optional environment variables:
  REFERENCE_EXPTID=${REFERENCE_EXPTID}
  INCLUDE_REFERENCE=${INCLUDE_REFERENCE}
  EXPERIMENT_SPECS='exptid|pretrain_iters|latent_weight|freeze_backbone;...'
  RUN_MODE=${RUN_MODE}
  MAX_ITERATIONS=${MAX_ITERATIONS}
  EVAL_EPISODES=${EVAL_EPISODES}
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

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --drive|--upload-drive)
        [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
        UPLOAD_DRIVE="$2"
        shift 2
        ;;
      --drive=*|--upload-drive=*)
        UPLOAD_DRIVE="${1#*=}"
        shift
        ;;
      --teacher-checkpoint-path)
        [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
        TEACHER_CHECKPOINT_PATH="$2"
        shift 2
        ;;
      --teacher-checkpoint-path=*)
        TEACHER_CHECKPOINT_PATH="${1#*=}"
        shift
        ;;
      --mode|--run-mode)
        [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
        RUN_MODE="$2"
        shift 2
        ;;
      --mode=*|--run-mode=*)
        RUN_MODE="${1#*=}"
        shift
        ;;
      -h|--help)
        usage
        exit 0
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
}

validate_run_mode() {
  case "${RUN_MODE}" in
    train-only|train-upload|train-eval-upload|full) ;;
    *)
      echo "Unsupported mode '${RUN_MODE}'. Use one of: train-only, train-upload, train-eval-upload, full." >&2
      exit 2
      ;;
  esac
}

should_upload_checkpoint() {
  [[ "${RUN_MODE}" == "train-upload" || "${RUN_MODE}" == "train-eval-upload" || "${RUN_MODE}" == "full" ]]
}

should_evaluate() {
  [[ "${RUN_MODE}" == "train-eval-upload" || "${RUN_MODE}" == "full" ]]
}

should_record() {
  [[ "${RUN_MODE}" == "full" ]]
}

configure_upload_drive() {
  case "${UPLOAD_DRIVE}" in
    quark) UPLOAD_REMOTE="${QUARK_REMOTE}"; UPLOAD_ROOT="${QUARK_ROOT}"; UPLOAD_NAME="Quark Cloud Drive" ;;
    baidu) UPLOAD_REMOTE="${BAIDU_REMOTE}"; UPLOAD_ROOT="${BAIDU_ROOT}"; UPLOAD_NAME="Baidu Netdisk" ;;
    none|skip|off) UPLOAD_NAME="upload"; UPLOAD_AVAILABLE=0 ;;
    *) echo "Unsupported upload drive '${UPLOAD_DRIVE}'. Use one of: quark, baidu, none." >&2; exit 2 ;;
  esac
}

login_upload_drive() {
  configure_upload_drive
  if [[ "${UPLOAD_DRIVE}" == "none" || "${UPLOAD_DRIVE}" == "skip" || "${UPLOAD_DRIVE}" == "off" ]]; then
    echo "Upload disabled."
    return 0
  fi
  if ! command -v rclone >/dev/null 2>&1; then
    echo "rclone not found; uploads will be skipped." >&2
    return 0
  fi
  if ! rclone config show "${UPLOAD_REMOTE}" >/dev/null 2>&1; then
    echo "rclone remote '${UPLOAD_REMOTE}' is not configured; uploads will be skipped." >&2
    return 0
  fi
  if ! rclone lsd "${UPLOAD_REMOTE}:" >/dev/null 2>&1; then
    echo "Unable to reach '${UPLOAD_REMOTE}:'; uploads will be skipped." >&2
    return 0
  fi
  UPLOAD_AVAILABLE=1
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
print(",".join(name for name, weight in terrain_dict.items() if float(weight) > 0.0))
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
terrain_counts = {name: int(envs_per_terrain) for name in terrain_names}
text = Path(src).read_text()
replacement = "DEFAULT_DISTILL_TERRAIN_ENVS = " + pprint.pformat(terrain_counts, sort_dicts=False)
updated = re.sub(r"DEFAULT_DISTILL_TERRAIN_ENVS\s*=\s*\{.*?\n\}", replacement, text, count=1, flags=re.S)
if updated == text:
    raise SystemExit("failed to replace DEFAULT_DISTILL_TERRAIN_ENVS")
Path(dst).write_text(updated)
PY
}

latest_checkpoint() {
  local log_dir="$1"
  find "${log_dir}" -maxdepth 1 -type f -name "model_*.pt" | sort -V | tail -n 1
}

run_train() {
  local exptid="$1"
  local pretrain_iters="$2"
  local latent_weight="$3"
  local freeze_backbone="$4"

  echo "Training ${exptid}"
  mkdir -p "${LEGGED_ROOT}/logs/${PROJ_NAME}/${exptid}"
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
      --heading_pretrain_iters "${pretrain_iters}" \
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
  local temp_distill="${SCRIPT_DIR}/.distill_play_${exptid}_minimal.py"
  local distill_gpu_args=()
  make_distill_play_copy "${terrain_csv}" "${temp_distill}"
  if [[ "${DISTILL_USE_GPU}" == "1" || "${DISTILL_USE_GPU}" == "true" || "${DISTILL_USE_GPU}" == "True" ]]; then
    distill_gpu_args+=(--use_gpu)
  fi
  for difficulty in 0.7 1.0; do
    local video_dir="${ARTIFACT_ROOT}/${exptid}/videos/difficulty_${difficulty}"
    mkdir -p "${video_dir}"
    echo "Recording ${exptid} videos at difficulty ${difficulty}"
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

upload_path() {
  local exptid="$1"
  echo "${UPLOAD_REMOTE}:${UPLOAD_ROOT}/${exptid}"
}

upload_checkpoint() {
  local exptid="$1"
  local log_dir="${LEGGED_ROOT}/logs/${PROJ_NAME}/${exptid}"
  local ckpt
  ckpt="$(latest_checkpoint "${log_dir}")"
  if [[ -z "${ckpt}" ]]; then
    echo "No checkpoint found for ${exptid}; skipping upload." >&2
    return 0
  fi
  if [[ "${UPLOAD_AVAILABLE}" != "1" ]]; then
    echo "Upload unavailable; skipping ${exptid}."
    return 0
  fi
  local remote
  remote="$(upload_path "${exptid}")"
  rclone copy "${ckpt}" "${remote}/checkpoint" || true
}

upload_evaluation() {
  local exptid="$1"
  if [[ "${UPLOAD_AVAILABLE}" != "1" ]]; then
    echo "Upload unavailable; skipping evaluation upload for ${exptid}."
    return 0
  fi
  local remote
  remote="$(upload_path "${exptid}")"
  [[ -d "${ARTIFACT_ROOT}/${exptid}/evaluation" ]] && rclone copy "${ARTIFACT_ROOT}/${exptid}/evaluation" "${remote}/evaluation" || true
}

upload_videos() {
  local exptid="$1"
  if [[ "${UPLOAD_AVAILABLE}" != "1" ]]; then
    echo "Upload unavailable; skipping video upload for ${exptid}."
    return 0
  fi
  local remote
  remote="$(upload_path "${exptid}")"
  [[ -d "${ARTIFACT_ROOT}/${exptid}/videos" ]] && rclone copy "${ARTIFACT_ROOT}/${exptid}/videos" "${remote}/videos" || true
}

run_reference_if_requested() {
  local terrains="$1"
  if ! should_evaluate && ! should_record; then
    return 0
  fi
  if [[ "${INCLUDE_REFERENCE}" != "1" && "${INCLUDE_REFERENCE}" != "true" && "${INCLUDE_REFERENCE}" != "True" ]]; then
    return 0
  fi
  local log_dir="${LEGGED_ROOT}/logs/${PROJ_NAME}/${REFERENCE_EXPTID}"
  if [[ -z "$(latest_checkpoint "${log_dir}")" ]]; then
    echo "Reference checkpoint not found for ${REFERENCE_EXPTID}; skipping reference evaluation." >&2
    return 0
  fi
  if should_record; then
    run_distill_videos "${REFERENCE_EXPTID}" "${terrains}"
  fi
  if should_evaluate; then
    run_evaluation "${REFERENCE_EXPTID}" "${terrains}"
  fi
  if should_upload_checkpoint; then
    upload_checkpoint "${REFERENCE_EXPTID}"
  fi
  if should_evaluate; then
    upload_evaluation "${REFERENCE_EXPTID}"
  fi
  if should_record; then
    upload_videos "${REFERENCE_EXPTID}"
  fi
}

main() {
  parse_args "$@"
  validate_run_mode
  require_file "${TEACHER_CHECKPOINT_PATH}" "TEACHER_CHECKPOINT_PATH"
  require_file "${CONFIG_PATH}" "terrain config"
  require_file "${TRAIN_PY}" "train.py"
  require_file "${EVAL_PY}" "evaluation.py"
  require_file "${DISTILL_PLAY_PY}" "distill_play.py"
  mkdir -p "${ARTIFACT_ROOT}"
  login_upload_drive
  echo "Run mode: ${RUN_MODE}"

  local terrains
  terrains="$(active_terrains_csv)"
  [[ -n "${terrains}" ]] || { echo "No active terrains found." >&2; exit 1; }
  echo "Active terrains: ${terrains}"

  run_reference_if_requested "${terrains}"

  IFS=";" read -r -a specs <<<"${EXPERIMENT_SPECS}"
  for spec in "${specs[@]}"; do
    [[ -n "${spec}" ]] || continue
    IFS="|" read -r exptid pretrain_iters latent_weight freeze_backbone <<<"${spec}"
    run_train "${exptid}" "${pretrain_iters}" "${latent_weight}" "${freeze_backbone}"
    if [[ "${RUN_MODE}" == "train-upload" || "${RUN_MODE}" == "train-eval-upload" ]]; then
      upload_checkpoint "${exptid}"
    fi
    if should_record; then
      run_distill_videos "${exptid}" "${terrains}"
    fi
    if should_evaluate; then
      run_evaluation "${exptid}" "${terrains}"
    fi
    if [[ "${RUN_MODE}" == "full" ]]; then
      upload_checkpoint "${exptid}"
    fi
    if should_evaluate; then
      upload_evaluation "${exptid}"
    fi
    if should_record; then
      upload_videos "${exptid}"
    fi
  done
}

main "$@"
