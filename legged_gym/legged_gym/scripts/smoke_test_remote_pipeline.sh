#!/usr/bin/env bash
set -u

# End-to-end smoke tests for remote/headless training and play.
#
# Usage:
#   bash smoke_test_remote_pipeline.sh all
#   bash smoke_test_remote_pipeline.sh train
#   bash smoke_test_remote_pipeline.sh play
#   bash smoke_test_remote_pipeline.sh web
#   bash smoke_test_remote_pipeline.sh video
#
# Common overrides:
#   PROJ=parkour_chain_smoke DEVICE=cuda:1 RL_DEVICE=cuda:1 bash smoke_test_remote_pipeline.sh all
#   TEACHER_ITERS=100 STUDENT_ITERS=100 HEADING_PRETRAIN_ITERS=10 bash smoke_test_remote_pipeline.sh train

MODE="${1:-all}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if REPO_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then
  :
else
  REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
fi
LEGGED_ROOT="$REPO_ROOT/legged_gym"
LOG_ROOT="$LEGGED_ROOT/logs"

TASK="${TASK:-a1}"
DEVICE="${DEVICE:-cuda:0}"
RL_DEVICE="${RL_DEVICE:-cuda:0}"
PROJ="${PROJ:-parkour_chain_smoke}"

TEACHER_ITERS="${TEACHER_ITERS:-20}"
STUDENT_ITERS="${STUDENT_ITERS:-20}"
HEADING_PRETRAIN_ITERS="${HEADING_PRETRAIN_ITERS:-5}"
PLAY_STEPS="${PLAY_STEPS:-100}"
WEB_PLAY_STEPS="${WEB_PLAY_STEPS:-600}"
VIDEO_SECONDS="${VIDEO_SECONDS:-8}"
WEB_PORT="${WEB_PORT:-5000}"

TEACHER_TTC_OFF_RUN="${TEACHER_TTC_OFF_RUN:-teacher_base_ttc_off_smoke}"
TEACHER_TTC_ON_RUN="${TEACHER_TTC_ON_RUN:-teacher_base_ttc_on_smoke}"
STUDENT_HEADING_OFF_TTC_OFF_RUN="${STUDENT_HEADING_OFF_TTC_OFF_RUN:-student_heading_off_ttc_off_smoke}"
STUDENT_HEADING_OFF_TTC_ON_RUN="${STUDENT_HEADING_OFF_TTC_ON_RUN:-student_heading_off_ttc_on_smoke}"
STUDENT_HEADING_ON_TTC_OFF_RUN="${STUDENT_HEADING_ON_TTC_OFF_RUN:-student_heading_on_ttc_off_smoke}"
STUDENT_HEADING_ON_TTC_ON_RUN="${STUDENT_HEADING_ON_TTC_ON_RUN:-student_heading_on_ttc_on_smoke}"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
RESULT_DIR="$LOG_ROOT/$PROJ/smoke_results/$RUN_TS"
SUMMARY="$RESULT_DIR/summary.tsv"
mkdir -p "$RESULT_DIR"

TOTAL=0
PASSED=0
FAILED=0
SKIPPED=0

log_info() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

write_summary_header() {
  printf 'status\tname\texit_code\tduration_s\tlog_file\tcommand\n' > "$SUMMARY"
}

append_summary() {
  local status="$1"
  local name="$2"
  local exit_code="$3"
  local duration="$4"
  local log_file="$5"
  local command="$6"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$status" "$name" "$exit_code" "$duration" "$log_file" "$command" >> "$SUMMARY"
}

quote_cmd() {
  local quoted=""
  local arg
  for arg in "$@"; do
    printf -v quoted '%s %q' "$quoted" "$arg"
  done
  printf '%s' "${quoted# }"
}

run_test() {
  local name="$1"
  shift

  local log_file="$RESULT_DIR/${name}.log"
  local command
  command="$(quote_cmd "$@")"
  local start
  start="$(date +%s)"

  TOTAL=$((TOTAL + 1))
  log_info "START $name"
  {
    echo "name: $name"
    echo "cwd: $(pwd)"
    echo "command: $command"
    echo "started_at: $(date '+%F %T')"
    echo
  } > "$log_file"

  "$@" >> "$log_file" 2>&1
  local exit_code=$?
  local end
  end="$(date +%s)"
  local duration=$((end - start))

  if [[ "$exit_code" -eq 0 ]]; then
    PASSED=$((PASSED + 1))
    append_summary "PASS" "$name" "$exit_code" "$duration" "$log_file" "$command"
    log_info "PASS  $name (${duration}s)"
  else
    FAILED=$((FAILED + 1))
    append_summary "FAIL" "$name" "$exit_code" "$duration" "$log_file" "$command"
    log_info "FAIL  $name (${duration}s, exit=$exit_code)"
    log_info "Failed command: $command"
    log_info "Log file: $log_file"
    log_info "Last 60 log lines:"
    tail -n 60 "$log_file" || true
  fi

  return "$exit_code"
}

skip_test() {
  local name="$1"
  local reason="$2"
  TOTAL=$((TOTAL + 1))
  SKIPPED=$((SKIPPED + 1))
  append_summary "SKIP" "$name" "-" "0" "-" "$reason"
  log_info "SKIP  $name: $reason"
}

require_checkpoint() {
  local run_name="$1"
  local checkpoint="$2"
  local path="$LOG_ROOT/$PROJ/$run_name/model_${checkpoint}.pt"
  [[ -f "$path" ]]
}

train_teacher() {
  cd "$REPO_ROOT" || return 1

  run_test "train_teacher_ttc_off" \
    python legged_gym/legged_gym/scripts/train.py \
      --task "$TASK" \
      --device "$DEVICE" \
      --rl_device "$RL_DEVICE" \
      --proj_name "$PROJ" \
      --exptid "$TEACHER_TTC_OFF_RUN" \
      --curriculum True \
      --task_targeted_curriculum False \
      --max_iterations "$TEACHER_ITERS" \
      --no_wandb

  run_test "train_teacher_ttc_on" \
    python legged_gym/legged_gym/scripts/train.py \
      --task "$TASK" \
      --device "$DEVICE" \
      --rl_device "$RL_DEVICE" \
      --proj_name "$PROJ" \
      --exptid "$TEACHER_TTC_ON_RUN" \
      --curriculum True \
      --task_targeted_curriculum True \
      --max_iterations "$TEACHER_ITERS" \
      --no_wandb
}

train_student() {
  cd "$REPO_ROOT" || return 1

  if require_checkpoint "$TEACHER_TTC_OFF_RUN" "$TEACHER_ITERS"; then
    run_test "train_student_heading_off_ttc_off" \
      python legged_gym/legged_gym/scripts/train.py \
        --task "$TASK" \
        --device "$DEVICE" \
        --rl_device "$RL_DEVICE" \
        --proj_name "$PROJ" \
        --exptid "$STUDENT_HEADING_OFF_TTC_OFF_RUN" \
        --use_camera \
        --resume \
        --resumeid "$TEACHER_TTC_OFF_RUN" \
        --checkpoint "$TEACHER_ITERS" \
        --curriculum True \
        --task_targeted_curriculum False \
        --max_iterations "$STUDENT_ITERS" \
        --no_wandb

    run_test "train_student_heading_on_ttc_off" \
      python legged_gym/legged_gym/scripts/train.py \
        --task "$TASK" \
        --device "$DEVICE" \
        --rl_device "$RL_DEVICE" \
        --proj_name "$PROJ" \
        --exptid "$STUDENT_HEADING_ON_TTC_OFF_RUN" \
        --use_camera \
        --enable_heading_model \
        --heading_pretrain_iters "$HEADING_PRETRAIN_ITERS" \
        --resume \
        --resumeid "$TEACHER_TTC_OFF_RUN" \
        --checkpoint "$TEACHER_ITERS" \
        --curriculum True \
        --task_targeted_curriculum False \
        --max_iterations "$STUDENT_ITERS" \
        --no_wandb
  else
    skip_test "train_student_heading_off_ttc_off" "missing $TEACHER_TTC_OFF_RUN/model_${TEACHER_ITERS}.pt"
    skip_test "train_student_heading_on_ttc_off" "missing $TEACHER_TTC_OFF_RUN/model_${TEACHER_ITERS}.pt"
  fi

  if require_checkpoint "$TEACHER_TTC_ON_RUN" "$TEACHER_ITERS"; then
    run_test "train_student_heading_off_ttc_on" \
      python legged_gym/legged_gym/scripts/train.py \
        --task "$TASK" \
        --device "$DEVICE" \
        --rl_device "$RL_DEVICE" \
        --proj_name "$PROJ" \
        --exptid "$STUDENT_HEADING_OFF_TTC_ON_RUN" \
        --use_camera \
        --resume \
        --resumeid "$TEACHER_TTC_ON_RUN" \
        --checkpoint "$TEACHER_ITERS" \
        --curriculum True \
        --task_targeted_curriculum True \
        --max_iterations "$STUDENT_ITERS" \
        --no_wandb

    run_test "train_student_heading_on_ttc_on" \
      python legged_gym/legged_gym/scripts/train.py \
        --task "$TASK" \
        --device "$DEVICE" \
        --rl_device "$RL_DEVICE" \
        --proj_name "$PROJ" \
        --exptid "$STUDENT_HEADING_ON_TTC_ON_RUN" \
        --use_camera \
        --enable_heading_model \
        --heading_pretrain_iters "$HEADING_PRETRAIN_ITERS" \
        --resume \
        --resumeid "$TEACHER_TTC_ON_RUN" \
        --checkpoint "$TEACHER_ITERS" \
        --curriculum True \
        --task_targeted_curriculum True \
        --max_iterations "$STUDENT_ITERS" \
        --no_wandb
  else
    skip_test "train_student_heading_off_ttc_on" "missing $TEACHER_TTC_ON_RUN/model_${TEACHER_ITERS}.pt"
    skip_test "train_student_heading_on_ttc_on" "missing $TEACHER_TTC_ON_RUN/model_${TEACHER_ITERS}.pt"
  fi
}

play_smoke() {
  cd "$SCRIPT_DIR" || return 1

  run_test "play_teacher_headless" \
    python play.py \
      --task "$TASK" \
      --device "$DEVICE" \
      --rl_device "$RL_DEVICE" \
      --proj_name "$PROJ" \
      --exptid "$TEACHER_TTC_ON_RUN" \
      --checkpoint -1 \
      --headless \
      --play_steps "$PLAY_STEPS"

  run_test "play_student_heading_off_headless" \
    python play.py \
      --task "$TASK" \
      --device "$DEVICE" \
      --rl_device "$RL_DEVICE" \
      --proj_name "$PROJ" \
      --exptid "$STUDENT_HEADING_OFF_TTC_ON_RUN" \
      --checkpoint -1 \
      --use_camera \
      --headless \
      --play_steps "$PLAY_STEPS"

  run_test "play_student_heading_on_headless" \
    python play.py \
      --task "$TASK" \
      --device "$DEVICE" \
      --rl_device "$RL_DEVICE" \
      --proj_name "$PROJ" \
      --exptid "$STUDENT_HEADING_ON_TTC_ON_RUN" \
      --checkpoint -1 \
      --use_camera \
      --enable_heading_model \
      --headless \
      --play_steps "$PLAY_STEPS"
}

wait_for_http() {
  local url="$1"
  local seconds="$2"
  local i
  for ((i = 0; i < seconds; i++)); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

capture_stream_bytes() {
  local url="$1"
  local output="$2"
  local seconds="$3"
  rm -f "$output"
  timeout "$seconds" curl -fsS "$url" -o "$output" >/dev/null 2>&1 || true
  [[ -s "$output" ]]
}

web_smoke() {
  cd "$SCRIPT_DIR" || return 1

  local name="play_web_student_heading_on"
  local log_file="$RESULT_DIR/${name}.log"
  local command
  command="$(quote_cmd python play.py \
    --task "$TASK" \
    --device "$DEVICE" \
    --rl_device "$RL_DEVICE" \
    --proj_name "$PROJ" \
    --exptid "$STUDENT_HEADING_ON_TTC_ON_RUN" \
    --checkpoint -1 \
    --use_camera \
    --enable_heading_model \
    --headless \
    --web \
    --play_steps "$WEB_PLAY_STEPS")"

  TOTAL=$((TOTAL + 1))
  log_info "START $name"
  {
    echo "name: $name"
    echo "cwd: $(pwd)"
    echo "command: $command"
    echo "started_at: $(date '+%F %T')"
    echo
  } > "$log_file"

  python play.py \
    --task "$TASK" \
    --device "$DEVICE" \
    --rl_device "$RL_DEVICE" \
    --proj_name "$PROJ" \
    --exptid "$STUDENT_HEADING_ON_TTC_ON_RUN" \
    --checkpoint -1 \
    --use_camera \
    --enable_heading_model \
    --headless \
    --web \
    --play_steps "$WEB_PLAY_STEPS" >> "$log_file" 2>&1 &

  local play_pid=$!
  local start
  start="$(date +%s)"
  local exit_code=0

  if ! wait_for_http "http://127.0.0.1:${WEB_PORT}/" 60; then
    exit_code=10
    echo "ERROR: web viewer did not respond on port ${WEB_PORT}" >> "$log_file"
  else
    if ! capture_stream_bytes "http://127.0.0.1:${WEB_PORT}/_route_stream" "$RESULT_DIR/web_color_stream.bin" 10; then
      exit_code=11
      echo "ERROR: color stream endpoint did not produce data" >> "$log_file"
    fi
    if ! capture_stream_bytes "http://127.0.0.1:${WEB_PORT}/_route_stream_depth" "$RESULT_DIR/web_depth_stream.bin" 10; then
      exit_code=12
      echo "ERROR: depth stream endpoint did not produce data" >> "$log_file"
    fi
  fi

  wait "$play_pid"
  local play_exit=$?
  if [[ "$exit_code" -eq 0 && "$play_exit" -ne 0 ]]; then
    exit_code="$play_exit"
  fi

  local end
  end="$(date +%s)"
  local duration=$((end - start))
  if [[ "$exit_code" -eq 0 ]]; then
    PASSED=$((PASSED + 1))
    append_summary "PASS" "$name" "$exit_code" "$duration" "$log_file" "$command"
    log_info "PASS  $name (${duration}s)"
  else
    FAILED=$((FAILED + 1))
    append_summary "FAIL" "$name" "$exit_code" "$duration" "$log_file" "$command"
    log_info "FAIL  $name (${duration}s, exit=$exit_code)"
    log_info "Log file: $log_file"
    tail -n 80 "$log_file" || true
  fi
}

video_smoke() {
  cd "$SCRIPT_DIR" || return 1

  if ! command -v ffmpeg >/dev/null 2>&1; then
    skip_test "video_capture_student_heading_on" "ffmpeg not found"
    return 0
  fi

  local name="video_capture_student_heading_on"
  local log_file="$RESULT_DIR/${name}.log"
  local video_path="$RESULT_DIR/${STUDENT_HEADING_ON_TTC_ON_RUN}_web_smoke.mp4"
  local command="python play.py --web ... + ffmpeg capture to $video_path"

  TOTAL=$((TOTAL + 1))
  log_info "START $name"
  {
    echo "name: $name"
    echo "cwd: $(pwd)"
    echo "command: $command"
    echo "video_path: $video_path"
    echo "started_at: $(date '+%F %T')"
    echo
  } > "$log_file"

  python play.py \
    --task "$TASK" \
    --device "$DEVICE" \
    --rl_device "$RL_DEVICE" \
    --proj_name "$PROJ" \
    --exptid "$STUDENT_HEADING_ON_TTC_ON_RUN" \
    --checkpoint -1 \
    --use_camera \
    --enable_heading_model \
    --headless \
    --web \
    --play_steps "$WEB_PLAY_STEPS" >> "$log_file" 2>&1 &

  local play_pid=$!
  local start
  start="$(date +%s)"
  local exit_code=0

  if ! wait_for_http "http://127.0.0.1:${WEB_PORT}/" 60; then
    exit_code=20
    echo "ERROR: web viewer did not respond on port ${WEB_PORT}" >> "$log_file"
  else
    ffmpeg -y \
      -t "$VIDEO_SECONDS" \
      -i "http://127.0.0.1:${WEB_PORT}/_route_stream" \
      -pix_fmt yuv420p \
      "$video_path" >> "$log_file" 2>&1
    local ffmpeg_exit=$?
    if [[ "$ffmpeg_exit" -ne 0 ]]; then
      exit_code="$ffmpeg_exit"
    elif [[ ! -s "$video_path" ]]; then
      exit_code=21
      echo "ERROR: video file was not created or is empty: $video_path" >> "$log_file"
    fi
  fi

  wait "$play_pid"
  local play_exit=$?
  if [[ "$exit_code" -eq 0 && "$play_exit" -ne 0 ]]; then
    exit_code="$play_exit"
  fi

  local end
  end="$(date +%s)"
  local duration=$((end - start))
  if [[ "$exit_code" -eq 0 ]]; then
    PASSED=$((PASSED + 1))
    append_summary "PASS" "$name" "$exit_code" "$duration" "$log_file" "$command"
    log_info "PASS  $name (${duration}s)"
    log_info "Video: $video_path"
  else
    FAILED=$((FAILED + 1))
    append_summary "FAIL" "$name" "$exit_code" "$duration" "$log_file" "$command"
    log_info "FAIL  $name (${duration}s, exit=$exit_code)"
    log_info "Log file: $log_file"
    tail -n 80 "$log_file" || true
  fi
}

print_config() {
  log_info "repo root: $REPO_ROOT"
  log_info "result dir: $RESULT_DIR"
  log_info "mode: $MODE"
  log_info "task=$TASK device=$DEVICE rl_device=$RL_DEVICE proj=$PROJ"
  log_info "teacher_iters=$TEACHER_ITERS student_iters=$STUDENT_ITERS heading_pretrain_iters=$HEADING_PRETRAIN_ITERS"
}

print_final_summary() {
  log_info "Summary file: $SUMMARY"
  log_info "Totals: total=$TOTAL pass=$PASSED fail=$FAILED skip=$SKIPPED"
  if [[ "$FAILED" -gt 0 ]]; then
    log_info "Failed tests:"
    awk -F '\t' 'NR > 1 && $1 == "FAIL" {print "  - " $2 " exit=" $3 " log=" $5}' "$SUMMARY" || true
    return 1
  fi
  return 0
}

main() {
  write_summary_header
  print_config

  case "$MODE" in
    train)
      train_teacher
      train_student
      ;;
    play)
      play_smoke
      ;;
    web)
      web_smoke
      ;;
    video)
      video_smoke
      ;;
    all)
      train_teacher
      train_student
      play_smoke
      web_smoke
      video_smoke
      ;;
    *)
      echo "Unknown mode: $MODE" >&2
      echo "Expected one of: all, train, play, web, video" >&2
      return 2
      ;;
  esac

  print_final_summary
}

main "$@"
