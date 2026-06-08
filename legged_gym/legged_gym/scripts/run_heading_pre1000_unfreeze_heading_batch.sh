#!/usr/bin/env bash
set -Eeuo pipefail

# Base experiment for the pre1000 C-main setting with one intentional change:
# the heading predictor head remains trainable during the second-stage action
# distillation phase.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export REFERENCE_EXPTID="${REFERENCE_EXPTID:-heading_pre1000_latent1_unfreeze}"
export INCLUDE_REFERENCE="${INCLUDE_REFERENCE:-1}"
export EXPERIMENT_SPECS="${EXPERIMENT_SPECS:-heading_pre1000_latent1_unfreeze_heading|1000|1.0|False|True}"

exec bash "${SCRIPT_DIR}/run_heading_minimal_ablation_batch.sh" "$@"
