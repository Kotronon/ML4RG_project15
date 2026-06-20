#!/usr/bin/env bash
#SBATCH --job-name=bb-fimo-streme
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/binding-bench-%j.out
#SBATCH --error=slurm/logs/binding-bench-%j.err

set -euo pipefail

PROJECT="${ML4RG_PROJECT:-/s/project/ml4rg_students/2026/project15}"
DATASET="${DATASET:-_saccharomyces_cerevisiae_sequence_mapper}"
SOURCE="${SOURCE:-fimo_streme}"
RESULT_TAG="${RESULT_TAG:-}"
FILTER_LOW_COMPLEXITY="${FILTER_LOW_COMPLEXITY:-0}"
TARGET_DATASET="${TARGET_DATASET:-DNA_rossi_chipexo}"
DISPLAY_NAME="${DISPLAY_NAME:-STREME-FIMO S. cerevisiae}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${REPO_DIR:-}" ]]; then
    if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/fimo_to_binding_bench.py" ]]; then
        REPO_DIR="${SLURM_SUBMIT_DIR}"
    else
        REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
    fi
fi
CONVERTER_SCRIPT="${CONVERTER_SCRIPT:-${REPO_DIR}/fimo_to_binding_bench.py}"
BINDING_BENCH_REPO="${BINDING_BENCH_REPO:-}"
BINDING_BENCH_BIN="${BINDING_BENCH_BIN:-binding_bench}"

INPUT_DIR="${PROJECT}/working/binding_bench_inputs"
RUN_NAME="${SOURCE}_${DATASET}"
FIMO_INPUT="${PROJECT}/working/streme_results/${SOURCE}/${DATASET}/fimo.tsv"
if [[ -n "${RESULT_TAG}" ]]; then
    RUN_NAME="${RUN_NAME}_${RESULT_TAG}"
    FIMO_INPUT="${PROJECT}/working/streme_results/${RESULT_TAG}/${SOURCE}/${DATASET}/fimo.tsv"
fi
CONVERTER_EXTRA_ARGS=()
case "${FILTER_LOW_COMPLEXITY}" in
    0) ;;
    1)
        RUN_NAME="${RUN_NAME}_filtered"
        CONVERTER_EXTRA_ARGS+=(--filter-low-complexity)
        ;;
    *)
        echo "FILTER_LOW_COMPLEXITY must be 0 or 1, got: ${FILTER_LOW_COMPLEXITY}" >&2
        exit 2
        ;;
esac
RUN_DIR="${PROJECT}/working/binding_bench_runs/${RUN_NAME}"
PREDICTIONS="${INPUT_DIR}/${RUN_NAME}_predictions.parquet"
FEATURE_RANKS="${INPUT_DIR}/${RUN_NAME}_feature_ranks.parquet"

mkdir -p "${INPUT_DIR}" "${RUN_DIR}" "${REPO_DIR}/slurm/logs"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    echo "Set PYTHON_BIN to the full Python path from test.ipynb, e.g. sys.executable." >&2
    exit 127
fi
if [[ ! -s "${FIMO_INPUT}" ]]; then
    echo "FIMO result not found or empty: ${FIMO_INPUT}" >&2
    exit 1
fi

echo "Project:          ${PROJECT}"
echo "Repo dir:         ${REPO_DIR}"
echo "Source:           ${SOURCE}"
echo "Result tag:       ${RESULT_TAG:-default}"
echo "Low complexity:   ${FILTER_LOW_COMPLEXITY}"
echo "Dataset:          ${DATASET}"
echo "Target dataset:   ${TARGET_DATASET}"
echo "Python:           $(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"
echo "Converter script: ${CONVERTER_SCRIPT}"
echo "FIMO input:       ${FIMO_INPUT}"

"${PYTHON_BIN}" "${CONVERTER_SCRIPT}" \
    --project "${PROJECT}" \
    --source "${SOURCE}" \
    --dataset "${DATASET}" \
    --fimo "${FIMO_INPUT}" \
    "${CONVERTER_EXTRA_ARGS[@]}" \
    --predictions-out "${PREDICTIONS}" \
    --feature-ranks-out "${FEATURE_RANKS}"

echo "Predictions:      ${PREDICTIONS}"
echo "Feature ranks:    ${FEATURE_RANKS}"
echo "Run dir:          ${RUN_DIR}"

if [[ -n "${BINDING_BENCH_REPO}" ]]; then
    export PYTHONPATH="${BINDING_BENCH_REPO}/src:${PYTHONPATH:-}"
    "${PYTHON_BIN}" -c 'from binding_bench.cli import main; main()' discrete \
        --input "${PREDICTIONS}" \
        --dataset "${TARGET_DATASET}" \
        --output_dir "${RUN_DIR}" \
        --metric jaccard precision_lb recall_lb \
        --best_assignment \
        --ba_by_feature_rank \
        --feature_rank_path "${FEATURE_RANKS}" \
        --feature_rank_name motif_score \
        --display_name "${DISPLAY_NAME}" \
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
        --display_name "${DISPLAY_NAME}" \
        --overwrite
fi

echo "Binding Bench baseline finished."
