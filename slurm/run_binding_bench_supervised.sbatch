#!/usr/bin/env bash
#SBATCH --job-name=bb-supervised
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH --partition=student_project
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/logs/binding-bench-supervised-%j.out
#SBATCH --error=slurm/logs/binding-bench-supervised-%j.err

set -euo pipefail

PROJECT="${ML4RG_PROJECT:-/s/project/ml4rg_students/2026/project15}"
PYTHON_BIN="${PYTHON_BIN:-/opt/modules/i12g/anaconda/envs/ml4rg_project15/bin/python}"
TARGET_DATASET="${TARGET_DATASET:-DNA_rossi_chipexo}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${REPO_DIR:-}" ]]; then
    if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/supervised_baseline/export_binding_bench_predictions.py" ]]; then
        REPO_DIR="${SLURM_SUBMIT_DIR}"
    else
        REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
    fi
fi

EXPORT_SCRIPT="${EXPORT_SCRIPT:-${REPO_DIR}/supervised_baseline/export_binding_bench_predictions.py}"
BINDING_BENCH_REPO="${BINDING_BENCH_REPO:-${HOME}/binding_bench}"
BINDING_BENCH_BIN="${BINDING_BENCH_BIN:-binding_bench}"

MODEL="${MODEL:-small_cnn}"
MODEL_DIR="${MODEL_DIR:-${PROJECT}/working/supervised_baseline/models/${MODEL}_overfit}"
CHECKPOINT="${CHECKPOINT:-${MODEL_DIR}/best.pt}"
REGIONS_PATH="${REGIONS_PATH:-/s/project/multispecies/fungi_code/tf_sae/processed/sequence_datasets/fungi_upstream_ATG_1000/_saccharomyces_cerevisiae_sequence_mapper.parquet}"

NMS_RADIUS_BP="${NMS_RADIUS_BP:-50}"
PEAK_CALLER="${PEAK_CALLER:-nms}"
PRE_NMS_FACTOR="${PRE_NMS_FACTOR:-20}"
SCORE_MODE="${SCORE_MODE:-logit}"
FEATURE_RANK_METHOD="${FEATURE_RANK_METHOD:-max-score}"
FEATURE_RANK_NAME="${FEATURE_RANK_NAME:-model_score}"
METHOD_NAME="${METHOD_NAME:-supervised_${MODEL}_${SCORE_MODE}_nms${NMS_RADIUS_BP}}"
DISPLAY_NAME="${DISPLAY_NAME:-Supervised ${MODEL} (${SCORE_MODE})}"
INPUT_DIR="${PROJECT}/working/binding_bench_inputs"
RUN_DIR="${PROJECT}/working/binding_bench_runs/${METHOD_NAME}"
PREDICTIONS="${PREDICTIONS:-${INPUT_DIR}/${METHOD_NAME}_predictions.parquet}"
FEATURE_RANKS="${FEATURE_RANKS:-${INPUT_DIR}/${METHOD_NAME}_feature_ranks.parquet}"

WINDOW_SIZE="${WINDOW_SIZE:-}"
SEQUENCE_ORIENTATION="${SEQUENCE_ORIENTATION:-}"
SCAN_STRIDE="${SCAN_STRIDE:-1}"
TOP_K_PER_TF="${TOP_K_PER_TF:-5000}"
EXPORT_BATCH_SIZE="${EXPORT_BATCH_SIZE:-8192}"
DEVICE="${DEVICE:-auto}"
INFERENCE_TF_NAMES="${INFERENCE_TF_NAMES:-}"
INFERENCE_TF_NAMES_PATH="${INFERENCE_TF_NAMES_PATH:-}"
INFERENCE_TF_EMBEDDINGS_PATH="${INFERENCE_TF_EMBEDDINGS_PATH:-}"
EXTRA_EXPORT_ARGS="${EXTRA_EXPORT_ARGS:-}"

mkdir -p "${INPUT_DIR}" "${RUN_DIR}" "${REPO_DIR}/slurm/logs"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    echo "Set PYTHON_BIN to the full Python path from the ml4rg_project15 environment." >&2
    exit 127
fi

CONDA_PREFIX="$(${PYTHON_BIN} -c 'import sys; print(sys.prefix)')"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export PYTHONUNBUFFERED=1

echo "Project:              ${PROJECT}"
echo "Repo dir:             ${REPO_DIR}"
echo "Export script:        ${EXPORT_SCRIPT}"
echo "Python:               $(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"
echo "Conda prefix:         ${CONDA_PREFIX}"
echo "Model:                ${MODEL}"
echo "Checkpoint:           ${CHECKPOINT}"
echo "Regions:              ${REGIONS_PATH}"
echo "Predictions:          ${PREDICTIONS}"
echo "Feature ranks:        ${FEATURE_RANKS}"
echo "Run dir:              ${RUN_DIR}"
echo "Target dataset:       ${TARGET_DATASET}"
echo "Scan stride:          ${SCAN_STRIDE}"
echo "Top K per TF:         ${TOP_K_PER_TF}"
echo "NMS radius bp:        ${NMS_RADIUS_BP}"
echo "Peak caller:          ${PEAK_CALLER}"
echo "Pre-NMS factor:       ${PRE_NMS_FACTOR}"
echo "Score mode:           ${SCORE_MODE}"
echo "Feature rank method:  ${FEATURE_RANK_METHOD}"
echo "Device:               ${DEVICE}"
echo "Inference TF names:   ${INFERENCE_TF_NAMES:-<checkpoint>}"
echo "Inference TF file:    ${INFERENCE_TF_NAMES_PATH:-<none>}"
echo "Inference TF emb:     ${INFERENCE_TF_EMBEDDINGS_PATH:-<checkpoint>}"

WINDOW_ARG=()
if [[ -n "${WINDOW_SIZE}" ]]; then
    WINDOW_ARG=(--window-size "${WINDOW_SIZE}")
fi

ORIENTATION_ARG=()
if [[ -n "${SEQUENCE_ORIENTATION}" ]]; then
    ORIENTATION_ARG=(--sequence-orientation "${SEQUENCE_ORIENTATION}")
fi

INFERENCE_ARG=()
if [[ -n "${INFERENCE_TF_NAMES}" ]]; then
    INFERENCE_ARG+=(--inference-tf-names "${INFERENCE_TF_NAMES}")
fi
if [[ -n "${INFERENCE_TF_NAMES_PATH}" ]]; then
    INFERENCE_ARG+=(--inference-tf-names-path "${INFERENCE_TF_NAMES_PATH}")
fi
if [[ -n "${INFERENCE_TF_EMBEDDINGS_PATH}" ]]; then
    INFERENCE_ARG+=(--inference-tf-embeddings-path "${INFERENCE_TF_EMBEDDINGS_PATH}")
fi

"${PYTHON_BIN}" "${EXPORT_SCRIPT}" \
    --checkpoint "${CHECKPOINT}" \
    --regions-path "${REGIONS_PATH}" \
    --predictions-out "${PREDICTIONS}" \
    --feature-ranks-out "${FEATURE_RANKS}" \
    --scan-stride "${SCAN_STRIDE}" \
    --top-k-per-tf "${TOP_K_PER_TF}" \
    --nms-radius-bp "${NMS_RADIUS_BP}" \
    --peak-caller "${PEAK_CALLER}" \
    --pre-nms-factor "${PRE_NMS_FACTOR}" \
    --batch-size "${EXPORT_BATCH_SIZE}" \
    --device "${DEVICE}" \
    --score-mode "${SCORE_MODE}" \
    --feature-rank-method "${FEATURE_RANK_METHOD}" \
    "${WINDOW_ARG[@]}" \
    "${ORIENTATION_ARG[@]}" \
    "${INFERENCE_ARG[@]}" \
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

echo "Supervised Binding Bench run finished."
