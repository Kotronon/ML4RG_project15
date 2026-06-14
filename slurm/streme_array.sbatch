#!/usr/bin/env bash
#SBATCH --job-name=streme-pipeline
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/streme-%A_%a.out
#SBATCH --error=slurm/logs/streme-%A_%a.err

set -euo pipefail

PROJECT="${ML4RG_PROJECT:-/s/project/ml4rg_students/2026/project15}"
STAGE="${STAGE:-all}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PIPELINE_SCRIPT="${PIPELINE_SCRIPT:-${SLURM_SUBMIT_DIR}/streme_pipeline.py}"

"${PYTHON_BIN}" "${PIPELINE_SCRIPT}" \
    --project "${PROJECT}" \
    --array-index "${SLURM_ARRAY_TASK_ID}" \
    --stage "${STAGE}"
