#!/usr/bin/env bash
set -Eeuo pipefail

# Evaluates one trained heading-model-C checkpoint under predicted, oracle, and
# corrupted heading inputs. This is evaluation-only: no policy weights are
# trained or modified.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LEGGED_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
TASK="${TASK:-a1}"
DEVICE="${DEVICE:-cuda:0}"
RL_DEVICE="${RL_DEVICE:-${DEVICE}}"
PROJ_NAME="${PROJ_NAME:-parkour_heading}"
TARGET_EXPTID="${TARGET_EXPTID:-heading_pre1000_latent1_unfreeze}"
CHECKPOINT="${CHECKPOINT:--1}"
UPLOAD_DRIVE="${UPLOAD_DRIVE:-quark}"
RUN_MODE="${RUN_MODE:-full}"

EVAL_EPISODES="${EVAL_EPISODES:-256}"
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-128}"
HEADING_CORRUPTION_STD="${HEADING_CORRUPTION_STD:-0.5}"
HEADING_EVAL_MODES="${HEADING_EVAL_MODES:-predicted oracle corrupted}"
SENSITIVE_TERRAINS="${SENSITIVE_TERRAINS:-beam_gap,asymmetric_gap,climbing_wall,alternating_step,parkour_v2}"
DISTILL_ENVS_PER_TERRAIN="${DISTILL_ENVS_PER_TERRAIN:-1}"
DISTILL_USE_GPU="${DISTILL_USE_GPU:-1}"
DISTILL_RECORD_CAMERA="${DISTILL_RECORD_CAMERA:-third_person}"

QUARK_REMOTE="${QUARK_REMOTE:-quark}"
QUARK_ROOT="${QUARK_ROOT:-extreme-parkour/heading_oracle_corrupted}"
BAIDU_REMOTE="${BAIDU_REMOTE:-baidu}"
BAIDU_ROOT="${BAIDU_ROOT:-extreme-parkour/heading_oracle_corrupted}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${LEGGED_ROOT}/logs/heading_oracle_corrupted_artifacts}"

UPLOAD_AVAILABLE=0
UPLOAD_REMOTE=""
UPLOAD_ROOT=""
UPLOAD_NAME=""

CONFIG_PATH="${LEGGED_ROOT}/legged_gym/envs/base/legged_robot_config.py"
EVAL_PY="${SCRIPT_DIR}/evaluation.py"
DISTILL_PLAY_PY="${SCRIPT_DIR}/distill_play.py"

usage() {
  cat <<EOF
Usage:
  TARGET_EXPTID=heading_pre1000_latent1_unfreeze bash $0 --mode full --drive quark
  bash $0 --mode train-eval-upload --drive none heading_pre1000_latent1_unfreeze

Modes:
  train-only         No-op training placeholder; verifies the target checkpoint.
  train-upload       Upload the target checkpoint only.
  train-eval-upload  Upload the target checkpoint, evaluate, then upload evaluation.
  full               Record videos, evaluate, then upload checkpoint/evaluation/videos.

Note:
  This script is evaluation-only; it never trains or modifies policy weights.

Optional environment variables:
  TARGET_EXPTID=${TARGET_EXPTID}
  CHECKPOINT=${CHECKPOINT}
  SENSITIVE_TERRAINS=${SENSITIVE_TERRAINS}
  HEADING_EVAL_MODES='${HEADING_EVAL_MODES}'
  HEADING_CORRUPTION_STD=${HEADING_CORRUPTION_STD}
  RUN_MODE=${RUN_MODE}
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
        TARGET_EXPTID="$1"
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

mode_artifact_dir() {
  local mode="$1"
  if [[ "${mode}" == "corrupted" ]]; then
    echo "${ARTIFACT_ROOT}/${TARGET_EXPTID}/${mode}_std${HEADING_CORRUPTION_STD}"
  else
    echo "${ARTIFACT_ROOT}/${TARGET_EXPTID}/${mode}"
  fi
}

run_evaluation_mode() {
  local mode="$1"
  local output_dir
  output_dir="$(mode_artifact_dir "${mode}")/evaluation"
  mkdir -p "${output_dir}"
  echo "Evaluating ${TARGET_EXPTID} with heading_eval_mode=${mode}"
  (
    cd "${SCRIPT_DIR}"
    "${PYTHON_BIN}" "${EVAL_PY}" \
      --task "${TASK}" \
      --device "${DEVICE}" \
      --rl_device "${RL_DEVICE}" \
      --use_camera \
      --enable_heading_model \
      --proj_name "${PROJ_NAME}" \
      --exptid "${TARGET_EXPTID}" \
      --checkpoint "${CHECKPOINT}" \
      --policy_id "${TARGET_EXPTID}_${mode}" \
      --policy_type depth \
      --terrain_names "${SENSITIVE_TERRAINS}" \
      --difficulty_mode all-difficulty \
      --eval_episodes "${EVAL_EPISODES}" \
      --num_envs "${EVAL_NUM_ENVS}" \
      --heading_eval_mode "${mode}" \
      --heading_corruption_std "${HEADING_CORRUPTION_STD}" \
      --output_dir "${output_dir}" \
      --headless
  )
}

run_videos_mode() {
  local mode="$1"
  local temp_distill="${SCRIPT_DIR}/.distill_play_${TARGET_EXPTID}_${mode}.py"
  local distill_gpu_args=()
  make_distill_play_copy "${SENSITIVE_TERRAINS}" "${temp_distill}"
  if [[ "${DISTILL_USE_GPU}" == "1" || "${DISTILL_USE_GPU}" == "true" || "${DISTILL_USE_GPU}" == "True" ]]; then
    distill_gpu_args+=(--use_gpu)
  fi
  for difficulty in 0.7 1.0; do
    local video_dir
    video_dir="$(mode_artifact_dir "${mode}")/videos/difficulty_${difficulty}"
    mkdir -p "${video_dir}"
    echo "Recording ${TARGET_EXPTID} ${mode} videos at difficulty ${difficulty}"
    (
      cd "${SCRIPT_DIR}"
      "${PYTHON_BIN}" "${temp_distill}" \
        --task "${TASK}" \
        --device "${DEVICE}" \
        --rl_device "${RL_DEVICE}" \
        --use_camera \
        --enable_heading_model \
        --proj_name "${PROJ_NAME}" \
        --exptid "${TARGET_EXPTID}" \
        --checkpoint "${CHECKPOINT}" \
        --video_out "${video_dir}" \
        --record_camera "${DISTILL_RECORD_CAMERA}" \
        --terrain_difficulty "${difficulty}" \
        --heading_eval_mode "${mode}" \
        --heading_corruption_std "${HEADING_CORRUPTION_STD}" \
        --headless \
        "${distill_gpu_args[@]}"
    )
  done
}

upload_mode_artifacts() {
  local mode="$1"
  if [[ "${UPLOAD_AVAILABLE}" != "1" ]]; then
    echo "Upload unavailable; skipping ${mode}."
    return 0
  fi
  local suffix="${mode}"
  [[ "${mode}" == "corrupted" ]] && suffix="corrupted_std${HEADING_CORRUPTION_STD}"
  local remote="${UPLOAD_REMOTE}:${UPLOAD_ROOT}/${TARGET_EXPTID}/${suffix}"
  local local_dir
  local_dir="$(mode_artifact_dir "${mode}")"
  [[ -d "${local_dir}/evaluation" ]] && rclone copy "${local_dir}/evaluation" "${remote}/evaluation" || true
  [[ -d "${local_dir}/videos" ]] && rclone copy "${local_dir}/videos" "${remote}/videos" || true
}

upload_checkpoint() {
  if [[ "${UPLOAD_AVAILABLE}" != "1" ]]; then
    echo "Upload unavailable; skipping checkpoint upload for ${TARGET_EXPTID}."
    return 0
  fi
  local log_dir="${LEGGED_ROOT}/logs/${PROJ_NAME}/${TARGET_EXPTID}"
  local ckpt
  if [[ "${CHECKPOINT}" == "-1" ]]; then
    ckpt="$(latest_checkpoint "${log_dir}")"
  else
    ckpt="${log_dir}/model_${CHECKPOINT}.pt"
  fi
  if [[ -z "${ckpt}" || ! -f "${ckpt}" ]]; then
    echo "No checkpoint found for upload: ${ckpt}" >&2
    return 0
  fi
  rclone copy "${ckpt}" "${UPLOAD_REMOTE}:${UPLOAD_ROOT}/${TARGET_EXPTID}/checkpoint" || true
}

upload_mode_evaluation() {
  local mode="$1"
  if [[ "${UPLOAD_AVAILABLE}" != "1" ]]; then
    echo "Upload unavailable; skipping ${mode} evaluation upload."
    return 0
  fi
  local suffix="${mode}"
  [[ "${mode}" == "corrupted" ]] && suffix="corrupted_std${HEADING_CORRUPTION_STD}"
  local remote="${UPLOAD_REMOTE}:${UPLOAD_ROOT}/${TARGET_EXPTID}/${suffix}"
  local local_dir
  local_dir="$(mode_artifact_dir "${mode}")"
  [[ -d "${local_dir}/evaluation" ]] && rclone copy "${local_dir}/evaluation" "${remote}/evaluation" || true
}

upload_mode_videos() {
  local mode="$1"
  if [[ "${UPLOAD_AVAILABLE}" != "1" ]]; then
    echo "Upload unavailable; skipping ${mode} video upload."
    return 0
  fi
  local suffix="${mode}"
  [[ "${mode}" == "corrupted" ]] && suffix="corrupted_std${HEADING_CORRUPTION_STD}"
  local remote="${UPLOAD_REMOTE}:${UPLOAD_ROOT}/${TARGET_EXPTID}/${suffix}"
  local local_dir
  local_dir="$(mode_artifact_dir "${mode}")"
  [[ -d "${local_dir}/videos" ]] && rclone copy "${local_dir}/videos" "${remote}/videos" || true
}

main() {
  parse_args "$@"
  validate_run_mode
  require_file "${CONFIG_PATH}" "terrain config"
  require_file "${EVAL_PY}" "evaluation.py"
  require_file "${DISTILL_PLAY_PY}" "distill_play.py"

  local log_dir="${LEGGED_ROOT}/logs/${PROJ_NAME}/${TARGET_EXPTID}"
  if [[ -z "$(latest_checkpoint "${log_dir}")" ]]; then
    echo "No checkpoint found for ${TARGET_EXPTID} in ${log_dir}" >&2
    exit 2
  fi

  mkdir -p "${ARTIFACT_ROOT}"
  login_upload_drive
  echo "Run mode: ${RUN_MODE}"
  echo "Sensitive terrains: ${SENSITIVE_TERRAINS}"

  if [[ "${RUN_MODE}" == "train-only" ]]; then
    echo "Evaluation-only script: target checkpoint verified; no training to run."
    return 0
  fi

  if [[ "${RUN_MODE}" == "train-upload" || "${RUN_MODE}" == "train-eval-upload" ]]; then
    upload_checkpoint
  fi

  for mode in ${HEADING_EVAL_MODES}; do
    if should_record; then
      run_videos_mode "${mode}"
    fi
    if should_evaluate; then
      run_evaluation_mode "${mode}"
    fi
    if [[ "${RUN_MODE}" == "full" ]]; then
      upload_checkpoint
      upload_mode_artifacts "${mode}"
    elif should_evaluate; then
      upload_mode_evaluation "${mode}"
    fi
  done
}

main "$@"
