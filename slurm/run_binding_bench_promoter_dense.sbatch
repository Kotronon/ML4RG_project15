#!/usr/bin/env bash
#SBATCH --job-name=bb-prom-dense
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --partition=student_project
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/logs/binding-bench-promoter-dense-%j.out
#SBATCH --error=slurm/logs/binding-bench-promoter-dense-%j.err

set -euo pipefail

PROJECT="${ML4RG_PROJECT:-/s/project/ml4rg_students/2026/project15}"
PYTHON_BIN="${PYTHON_BIN:-/opt/modules/i12g/anaconda/envs/ml4rg_project15/bin/python}"
TARGET_DATASET="${TARGET_DATASET:-DNA_rossi_chipexo}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${REPO_DIR:-}" ]]; then
    if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/supervised_baseline/export_promoter_dense_predictions.py" ]]; then
        REPO_DIR="${SLURM_SUBMIT_DIR}"
    else
        REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
    fi
fi

EXPORT_SCRIPT="${EXPORT_SCRIPT:-${REPO_DIR}/supervised_baseline/export_promoter_dense_predictions.py}"
BINDING_BENCH_REPO="${BINDING_BENCH_REPO:-${HOME}/binding_bench}"
BINDING_BENCH_BIN="${BINDING_BENCH_BIN:-binding_bench}"

INPUT_MODE="${INPUT_MODE:-raw}"
MODEL="${MODEL:-dense_small_cnn}"
MODEL_DIR="${MODEL_DIR:-${PROJECT}/working/supervised_baseline/models/promoter_${INPUT_MODE}_${MODEL}}"
CHECKPOINT="${CHECKPOINT:-${MODEL_DIR}/best.pt}"

NMS_RADIUS_BP="${NMS_RADIUS_BP:-0}"
PRE_NMS_FACTOR="${PRE_NMS_FACTOR:-20}"
SCORE_MODE="${SCORE_MODE:-logit}"
FEATURE_RANK_METHOD="${FEATURE_RANK_METHOD:-max-score}"
FEATURE_RANK_NAME="${FEATURE_RANK_NAME:-model_score}"
METHOD_NAME="${METHOD_NAME:-promoter_${INPUT_MODE}_${MODEL}_${SCORE_MODE}_nms${NMS_RADIUS_BP}}"
DISPLAY_NAME="${DISPLAY_NAME:-Promoter ${INPUT_MODE} ${MODEL} (${SCORE_MODE}, nms=${NMS_RADIUS_BP})}"
INPUT_DIR="${PROJECT}/working/binding_bench_inputs"
RUN_DIR="${RUN_DIR:-${PROJECT}/working/binding_bench_reports/${METHOD_NAME}}"
PREDICTIONS="${PREDICTIONS:-${INPUT_DIR}/${METHOD_NAME}_predictions.parquet}"
FEATURE_RANKS="${FEATURE_RANKS:-${INPUT_DIR}/${METHOD_NAME}_feature_ranks.parquet}"

TOP_K_PER_TF="${TOP_K_PER_TF:-5000}"
EXPORT_BATCH_SIZE="${EXPORT_BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-0}"
DEVICE="${DEVICE:-auto}"
SITES_PATH="${SITES_PATH:-}"
REGIONS_PATH="${REGIONS_PATH:-}"
EMBEDDINGS_PATH="${EMBEDDINGS_PATH:-}"
EMBEDDING_COLUMN="${EMBEDDING_COLUMN:-}"
EMBEDDING_KEY_COLUMN="${EMBEDDING_KEY_COLUMN:-}"
SEQUENCE_ORIENTATION="${SEQUENCE_ORIENTATION:-}"
INCLUDE_TERMINAL_ATG="${INCLUDE_TERMINAL_ATG:-false}"
MAX_REGIONS="${MAX_REGIONS:-}"
EXTRA_EXPORT_ARGS="${EXTRA_EXPORT_ARGS:-}"

mkdir -p "${INPUT_DIR}" "${RUN_DIR}" "${REPO_DIR}/slurm/logs"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 127
fi

CONDA_PREFIX="$(${PYTHON_BIN} -c 'import sys; print(sys.prefix)')"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

echo "Project:              ${PROJECT}"
echo "Repo dir:             ${REPO_DIR}"
echo "Export script:        ${EXPORT_SCRIPT}"
echo "Python:               $(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"
echo "Conda prefix:         ${CONDA_PREFIX}"
echo "Input mode:           ${INPUT_MODE}"
echo "Model:                ${MODEL}"
echo "Checkpoint:           ${CHECKPOINT}"
echo "Predictions:          ${PREDICTIONS}"
echo "Feature ranks:        ${FEATURE_RANKS}"
echo "Run dir:              ${RUN_DIR}"
echo "Target dataset:       ${TARGET_DATASET}"
echo "Top K per TF:         ${TOP_K_PER_TF}"
echo "NMS radius bp:        ${NMS_RADIUS_BP}"
echo "Pre-NMS factor:       ${PRE_NMS_FACTOR}"
echo "Score mode:           ${SCORE_MODE}"
echo "Feature rank method:  ${FEATURE_RANK_METHOD}"
echo "Device:               ${DEVICE}"

OPTIONAL_ARGS=()
if [[ -n "${SITES_PATH}" ]]; then
    OPTIONAL_ARGS+=(--sites-path "${SITES_PATH}")
fi
if [[ -n "${REGIONS_PATH}" ]]; then
    OPTIONAL_ARGS+=(--regions-path "${REGIONS_PATH}")
fi
if [[ -n "${EMBEDDINGS_PATH}" ]]; then
    OPTIONAL_ARGS+=(--embeddings-path "${EMBEDDINGS_PATH}")
fi
if [[ -n "${EMBEDDING_COLUMN}" ]]; then
    OPTIONAL_ARGS+=(--embedding-column "${EMBEDDING_COLUMN}")
fi
if [[ -n "${EMBEDDING_KEY_COLUMN}" ]]; then
    OPTIONAL_ARGS+=(--embedding-key-column "${EMBEDDING_KEY_COLUMN}")
fi
if [[ -n "${SEQUENCE_ORIENTATION}" ]]; then
    OPTIONAL_ARGS+=(--sequence-orientation "${SEQUENCE_ORIENTATION}")
fi
if [[ "${INCLUDE_TERMINAL_ATG}" == "true" ]]; then
    OPTIONAL_ARGS+=(--include-terminal-atg)
fi
if [[ -n "${MAX_REGIONS}" ]]; then
    OPTIONAL_ARGS+=(--max-regions "${MAX_REGIONS}")
fi

"${PYTHON_BIN}" "${EXPORT_SCRIPT}" \
    --checkpoint "${CHECKPOINT}" \
    --input-mode "${INPUT_MODE}" \
    --predictions-out "${PREDICTIONS}" \
    --feature-ranks-out "${FEATURE_RANKS}" \
    --top-k-per-tf "${TOP_K_PER_TF}" \
    --nms-radius-bp "${NMS_RADIUS_BP}" \
    --pre-nms-factor "${PRE_NMS_FACTOR}" \
    --batch-size "${EXPORT_BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --device "${DEVICE}" \
    --score-mode "${SCORE_MODE}" \
    --feature-rank-method "${FEATURE_RANK_METHOD}" \
    "${OPTIONAL_ARGS[@]}" \
    ${EXTRA_EXPORT_ARGS}

if [[ -n "${BINDING_BENCH_REPO}" && -d "${BINDING_BENCH_REPO}/src" ]]; then
    export PYTHONPATH="${BINDING_BENCH_REPO}/src:${PYTHONPATH:-}"
    "${PYTHON_BIN}" -c 'from binding_bench.cli import main; main()' discrete \
        --input "${PREDICTIONS}" \
        --dataset "${TARGET_DATASET}" \
        --output_dir "${RUN_DIR}" \
        --metric jaccard precision_lb recall_lb \
        --best_assignment \
        --ba_by_feature_rank \
        --feature_rank_path "${FEATURE_RANKS}" \
        --feature_rank_name "${FEATURE_RANK_NAME}" \
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
        --feature_rank_name "${FEATURE_RANK_NAME}" \
        --display_name "${DISPLAY_NAME}" \
        --overwrite
fi

echo "Promoter dense Binding Bench run finished."
