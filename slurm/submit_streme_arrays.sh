#!/usr/bin/env bash

set -euo pipefail

PROJECT="${ML4RG_PROJECT:-/s/project/ml4rg_students/2026/project15}"
FASTA_DIR="${PROJECT}/working/sequence_datasets_fastas"
MAX_CONCURRENT="${MAX_CONCURRENT:-20}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

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
COMMON_ARGS=(
    --parsable
    --array="${ARRAY_RANGE}"
    --chdir="${REPO_DIR}"
)
EXPORT_BASE="ALL,ML4RG_PROJECT=${PROJECT},PIPELINE_SCRIPT=${REPO_DIR}/streme_pipeline.py"

STREME_JOB="$(
    sbatch "${COMMON_ARGS[@]}" \
        --export="${EXPORT_BASE},STAGE=streme" \
        "${SCRIPT_DIR}/streme_array.sbatch"
)"
STREME_JOB="${STREME_JOB%%;*}"

FIMO_JASPAR_JOB="$(
    sbatch "${COMMON_ARGS[@]}" \
        --export="${EXPORT_BASE},STAGE=fimo_jaspar" \
        "${SCRIPT_DIR}/streme_array.sbatch"
)"
FIMO_JASPAR_JOB="${FIMO_JASPAR_JOB%%;*}"

TOMTOM_JOB="$(
    sbatch "${COMMON_ARGS[@]}" \
        --dependency="aftercorr:${STREME_JOB}" \
        --export="${EXPORT_BASE},STAGE=tomtom" \
        "${SCRIPT_DIR}/streme_array.sbatch"
)"
TOMTOM_JOB="${TOMTOM_JOB%%;*}"

FIMO_STREME_JOB="$(
    sbatch "${COMMON_ARGS[@]}" \
        --dependency="aftercorr:${STREME_JOB}" \
        --export="${EXPORT_BASE},STAGE=fimo_streme" \
        "${SCRIPT_DIR}/streme_array.sbatch"
)"
FIMO_STREME_JOB="${FIMO_STREME_JOB%%;*}"

printf 'Submitted %s FASTAs (maximum %s concurrent tasks per stage).\n' \
    "${FASTA_COUNT}" "${MAX_CONCURRENT}"
printf 'STREME:       %s\n' "${STREME_JOB}"
printf 'FIMO-JASPAR:  %s\n' "${FIMO_JASPAR_JOB}"
printf 'TOMTOM:       %s (after corresponding STREME task)\n' "${TOMTOM_JOB}"
printf 'FIMO-STREME:  %s (after corresponding STREME task)\n' "${FIMO_STREME_JOB}"
