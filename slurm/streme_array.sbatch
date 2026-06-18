#!/usr/bin/env bash
#SBATCH --job-name=streme-pipeline
#SBATCH --cpus-per-task=1
#SBATCH --mem=32GB
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/streme-%A_%a.out
#SBATCH --error=slurm/logs/streme-%A_%a.err

set -euo pipefail

PROJECT="${ML4RG_PROJECT:-/s/project/ml4rg_students/2026/project15}"
STAGE="${STAGE:-all}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python || true)}"
PIPELINE_SCRIPT="${PIPELINE_SCRIPT:-${SLURM_SUBMIT_DIR}/streme_pipeline.py}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found or not executable: ${PYTHON_BIN}" >&2
    exit 127
fi
if [[ ! -f "${PIPELINE_SCRIPT}" ]]; then
    echo "Pipeline script not found: ${PIPELINE_SCRIPT}" >&2
    exit 1
fi
if ! "${PYTHON_BIN}" -c "import pandas" 2>/dev/null; then
    echo "Python environment does not provide pandas: ${PYTHON_BIN}" >&2
    exit 1
fi

"${PYTHON_BIN}" "${PIPELINE_SCRIPT}" \
    --project "${PROJECT}" \
    --array-index "${SLURM_ARRAY_TASK_ID}" \
    --stage "${STAGE}"
