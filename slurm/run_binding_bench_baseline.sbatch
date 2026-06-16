#!/usr/bin/env bash
#SBATCH --job-name=bb-fimo-streme
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/binding-bench-%j.out
#SBATCH --error=slurm/logs/binding-bench-%j.err

set -euo pipefail

PROJECT="${ML4RG_PROJECT:-/s/project/ml4rg_students/2026/project15}"
DATASET="${DATASET:-_saccharomyces_cerevisiae_sequence_mapper}"
SOURCE="${SOURCE:-fimo_streme}"
TARGET_DATASET="${TARGET_DATASET:-DNA_rossi_chipexo}"
PYTHON_BIN="${PYTHON_BIN:-python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
CONVERTER_SCRIPT="${CONVERTER_SCRIPT:-${REPO_DIR}/fimo_to_binding_bench.py}"
BINDING_BENCH_REPO="${BINDING_BENCH_REPO:-}"
BINDING_BENCH_BIN="${BINDING_BENCH_BIN:-binding_bench}"

INPUT_DIR="${PROJECT}/working/binding_bench_inputs"
RUN_DIR="${PROJECT}/working/binding_bench_runs/${SOURCE}_${DATASET}"
PREDICTIONS="${INPUT_DIR}/${SOURCE}_${DATASET}_predictions.parquet"
FEATURE_RANKS="${INPUT_DIR}/${SOURCE}_${DATASET}_feature_ranks.parquet"

mkdir -p "${INPUT_DIR}" "${RUN_DIR}" "${REPO_DIR}/slurm/logs"

echo "Project:          ${PROJECT}"
echo "Repo dir:         ${REPO_DIR}"
echo "Source:           ${SOURCE}"
echo "Dataset:          ${DATASET}"
echo "Target dataset:   ${TARGET_DATASET}"
echo "Python:           $(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"
echo "Converter script: ${CONVERTER_SCRIPT}"

"${PYTHON_BIN}" "${CONVERTER_SCRIPT}" \
    --project "${PROJECT}" \
    --source "${SOURCE}" \
    --dataset "${DATASET}" \
    --predictions-out "${PREDICTIONS}" \
    --feature-ranks-out "${FEATURE_RANKS}"

echo "Predictions:      ${PREDICTIONS}"
echo "Feature ranks:    ${FEATURE_RANKS}"
echo "Run dir:          ${RUN_DIR}"

if [[ -n "${BINDING_BENCH_REPO}" ]]; then
    export PYTHONPATH="${BINDING_BENCH_REPO}/src:${PYTHONPATH:-}"
    "${PYTHON_BIN}" -m binding_bench.cli discrete \
        --input "${PREDICTIONS}" \
        --dataset "${TARGET_DATASET}" \
        --output_dir "${RUN_DIR}" \
        --metric jaccard precision_lb recall_lb \
        --best_assignment \
        --ba_by_feature_rank \
        --feature_rank_path "${FEATURE_RANKS}" \
        --feature_rank_name motif_score \
        --display_name "STREME-FIMO S. cerevisiae" \
        --overwrite
else
    "${BINDING_BENCH_BIN}" discrete \
        --input "${PREDICTIONS}" \
        --dataset "${TARGET_DATASET}" \
        --output_dir "${RUN_DIR}" \
        --metric jaccard precision_lb recall_lb \
        --best_assignment \
        --ba_by_feature_rank \
        --feature_rank_path "${FEATURE_RANKS}" \
        --feature_rank_name motif_score \
        --display_name "STREME-FIMO S. cerevisiae" \
        --overwrite
fi

echo "Binding Bench baseline finished."
