#!/usr/bin/env bash

set -euo pipefail

PROJECT="${ML4RG_PROJECT:-/s/project/ml4rg_students/2026/project15}"
FASTA_DIR="${PROJECT}/working/sequence_datasets_fastas"
MAX_CONCURRENT="${MAX_CONCURRENT:-20}"
SLURM_PARTITION="${SLURM_PARTITION:-}"
SLURM_ACCOUNT="${SLURM_ACCOUNT:-}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python || true)}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

REQUIRED_FILES=(
    "${REPO_DIR}/streme_pipeline.py"
    "${SCRIPT_DIR}/streme_array.sbatch"
    "${SCRIPT_DIR}/streme_summary.sbatch"
)
for required_file in "${REQUIRED_FILES[@]}"; do
    if [[ ! -f "${required_file}" ]]; then
        echo "Required file not found: ${required_file}" >&2
        exit 1
    fi
done
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found or not executable: ${PYTHON_BIN}" >&2
    exit 127
fi
if ! "${PYTHON_BIN}" -c "import pandas" 2>/dev/null; then
    echo "Python environment does not provide pandas: ${PYTHON_BIN}" >&2
    echo "Activate the project environment or set PYTHON_BIN explicitly." >&2
    exit 1
fi

mkdir -p "${REPO_DIR}/slurm/logs"

FASTA_COUNT="$(
    find "${FASTA_DIR}" -maxdepth 1 -type f \
        \( -name '*.fa' -o -name '*.fasta' \) |
        wc -l
)"
FASTA_COUNT="${FASTA_COUNT//[[:space:]]/}"

if [[ "${FASTA_COUNT}" -eq 0 ]]; then
    echo "No FASTA files found in ${FASTA_DIR}" >&2
    exit 1
fi

ARRAY_RANGE="0-$((FASTA_COUNT - 1))%${MAX_CONCURRENT}"
SCHEDULER_ARGS=()
if [[ -n "${SLURM_PARTITION}" ]]; then
    SCHEDULER_ARGS+=(--partition="${SLURM_PARTITION}")
fi
if [[ -n "${SLURM_ACCOUNT}" ]]; then
    SCHEDULER_ARGS+=(--account="${SLURM_ACCOUNT}")
fi

COMMON_ARGS=(
    --parsable
    --array="${ARRAY_RANGE}"
    --chdir="${REPO_DIR}"
    "${SCHEDULER_ARGS[@]}"
)
EXPORT_BASE="ALL,ML4RG_PROJECT=${PROJECT},PIPELINE_SCRIPT=${REPO_DIR}/streme_pipeline.py,PYTHON_BIN=${PYTHON_BIN}"

STREME_JOB="$(
    sbatch "${COMMON_ARGS[@]}" \
        --job-name="streme-streme" \
        --export="${EXPORT_BASE},STAGE=streme" \
        "${SCRIPT_DIR}/streme_array.sbatch"
)"
STREME_JOB="${STREME_JOB%%;*}"

FIMO_JASPAR_JOB="$(
    sbatch "${COMMON_ARGS[@]}" \
        --job-name="streme-fimo-jaspar" \
        --export="${EXPORT_BASE},STAGE=fimo_jaspar" \
        "${SCRIPT_DIR}/streme_array.sbatch"
)"
FIMO_JASPAR_JOB="${FIMO_JASPAR_JOB%%;*}"

TOMTOM_JOB="$(
    sbatch "${COMMON_ARGS[@]}" \
        --job-name="streme-tomtom" \
        --dependency="aftercorr:${STREME_JOB}" \
        --export="${EXPORT_BASE},STAGE=tomtom" \
        "${SCRIPT_DIR}/streme_array.sbatch"
)"
TOMTOM_JOB="${TOMTOM_JOB%%;*}"

FIMO_STREME_JOB="$(
    sbatch "${COMMON_ARGS[@]}" \
        --job-name="streme-fimo-streme" \
        --dependency="aftercorr:${STREME_JOB}" \
        --export="${EXPORT_BASE},STAGE=fimo_streme" \
        "${SCRIPT_DIR}/streme_array.sbatch"
)"
FIMO_STREME_JOB="${FIMO_STREME_JOB%%;*}"

SUMMARY_JOB="$(
    sbatch --parsable \
        --job-name="streme-summary" \
        --chdir="${REPO_DIR}" \
        "${SCHEDULER_ARGS[@]}" \
        --dependency="afterany:${STREME_JOB}:${FIMO_JASPAR_JOB}:${TOMTOM_JOB}:${FIMO_STREME_JOB}" \
        --export="${EXPORT_BASE}" \
        "${SCRIPT_DIR}/streme_summary.sbatch"
)"
SUMMARY_JOB="${SUMMARY_JOB%%;*}"

printf 'Submitted %s FASTAs (maximum %s concurrent tasks per stage).\n' \
    "${FASTA_COUNT}" "${MAX_CONCURRENT}"
printf 'STREME:       %s\n' "${STREME_JOB}"
printf 'FIMO-JASPAR:  %s\n' "${FIMO_JASPAR_JOB}"
printf 'TOMTOM:       %s (after corresponding STREME task)\n' "${TOMTOM_JOB}"
printf 'FIMO-STREME:  %s (after corresponding STREME task)\n' "${FIMO_STREME_JOB}"
printf 'SUMMARY:      %s (after all pipeline arrays finish)\n' "${SUMMARY_JOB}"
