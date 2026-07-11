#!/usr/bin/env bash
#SBATCH --job-name=streme-n100
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/streme-n100-%j.out
#SBATCH --error=slurm/logs/streme-n100-%j.err

set -euo pipefail

PROJECT="${ML4RG_PROJECT:-/s/project/ml4rg_students/2026/project15}"
DATASET="${DATASET:-_saccharomyces_cerevisiae_sequence_mapper}"
RESULT_TAG="${RESULT_TAG:-nmotifs_100}"
NMOTIFS="${NMOTIFS:-100}"
STREME_TIME="${STREME_TIME:-14400}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python || true)}"
PIPELINE_SCRIPT="${PIPELINE_SCRIPT:-${SLURM_SUBMIT_DIR}/streme_pipeline.py}"
FASTA="${PROJECT}/working/sequence_datasets_fastas/${DATASET}.fasta"

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 127
fi
if [[ ! -f "${PIPELINE_SCRIPT}" ]]; then
    echo "Pipeline script not found: ${PIPELINE_SCRIPT}" >&2
    exit 1
fi
if [[ ! -s "${FASTA}" ]]; then
    echo "FASTA not found or empty: ${FASTA}" >&2
    exit 1
fi

COMMON_ARGS=(
    --project "${PROJECT}"
    --result-tag "${RESULT_TAG}"
    --fasta "${FASTA}"
    --streme-time "${STREME_TIME}"
    --minw 6
    --maxw 20
    --nmotifs "${NMOTIFS}"
    --fimo-thresh 1e-4
    --fimo-max-stored-scores 1000000
    --strict-cache
)

echo "Running STREME: dataset=${DATASET}, nmotifs=${NMOTIFS}, result_tag=${RESULT_TAG}"
"${PYTHON_BIN}" "${PIPELINE_SCRIPT}" "${COMMON_ARGS[@]}" --stage streme --force

echo "Running FIMO with the new STREME motifs"
"${PYTHON_BIN}" "${PIPELINE_SCRIPT}" "${COMMON_ARGS[@]}" --stage fimo_streme --force

echo "Completed results: ${PROJECT}/working/streme_results/${RESULT_TAG}"
