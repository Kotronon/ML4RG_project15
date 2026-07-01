#!/usr/bin/env bash
#SBATCH --job-name=prom-dense
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --partition=student_project
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/logs/promoter-dense-%j.out
#SBATCH --error=slurm/logs/promoter-dense-%j.err

set -euo pipefail

PROJECT="${ML4RG_PROJECT:-/s/project/ml4rg_students/2026/project15}"
PYTHON_BIN="${PYTHON_BIN:-/opt/modules/i12g/anaconda/envs/ml4rg_project15/bin/python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${REPO_DIR:-}" ]]; then
    if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/supervised_baseline/train_promoter_dense.py" ]]; then
        REPO_DIR="${SLURM_SUBMIT_DIR}"
    else
        REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
    fi
fi

TRAIN_SCRIPT="${TRAIN_SCRIPT:-${REPO_DIR}/supervised_baseline/train_promoter_dense.py}"
SITES_PATH="${SITES_PATH:-/s/project/multispecies/fungi_code/tf_sae/binding_bench_datasets/val/dna/_saccharomyces_cerevisiae/DNA_rossi_chipexo_sites.parquet}"
REGIONS_PATH="${REGIONS_PATH:-/s/project/multispecies/fungi_code/tf_sae/processed/sequence_datasets/fungi_upstream_ATG_1000/_saccharomyces_cerevisiae_sequence_mapper.parquet}"

INPUT_MODE="${INPUT_MODE:-raw}"
MODEL="${MODEL:-dense_small_cnn}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT}/working/supervised_baseline/models/promoter_${INPUT_MODE}_${MODEL}}"
EMBEDDINGS_PATH="${EMBEDDINGS_PATH:-}"
EMBEDDING_COLUMN="${EMBEDDING_COLUMN:-emb}"
EMBEDDING_KEY_COLUMN="${EMBEDDING_KEY_COLUMN:-}"
TF_EMBEDDINGS_PATH="${TF_EMBEDDINGS_PATH:-}"
TF_EMBEDDING_KEY_COLUMN="${TF_EMBEDDING_KEY_COLUMN:-}"
TF_EMBEDDING_COLUMN="${TF_EMBEDDING_COLUMN:-emb}"
TF_NAME_MAP="${TF_NAME_MAP:-}"
DROP_MISSING_TF_EMBEDDINGS="${DROP_MISSING_TF_EMBEDDINGS:-false}"
TF_NAME_FILTER_FROM_EMBEDDINGS="${TF_NAME_FILTER_FROM_EMBEDDINGS:-false}"

MIN_SITES_PER_TF="${MIN_SITES_PER_TF:-15}"
MAX_REGIONS="${MAX_REGIONS:-}"
SEQUENCE_ORIENTATION="${SEQUENCE_ORIENTATION:-strand-aware}"
INCLUDE_TERMINAL_ATG="${INCLUDE_TERMINAL_ATG:-false}"
HIDDEN_CHANNELS="${HIDDEN_CHANNELS:-128}"
KERNEL_SIZE="${KERNEL_SIZE:-7}"
DILATIONS="${DILATIONS:-1,2,4,8,16}"
DROPOUT="${DROPOUT:-0.1}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LR="${LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
MIN_LR="${MIN_LR:-0.00001}"
PLATEAU_PATIENCE="${PLATEAU_PATIENCE:-10}"
PLATEAU_FACTOR="${PLATEAU_FACTOR:-0.5}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-auto}"
PROMOTER_SPLIT_MODE="${PROMOTER_SPLIT_MODE:-none}"
PROMOTER_SPLIT_PATH="${PROMOTER_SPLIT_PATH:-}"
PROMOTER_SPLIT_OUT="${PROMOTER_SPLIT_OUT:-}"
TRAIN_FRACTION="${TRAIN_FRACTION:-0.8}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
VAL_CHROMS="${VAL_CHROMS:-}"
TEST_CHROMS="${TEST_CHROMS:-}"
EVAL_EVERY="${EVAL_EVERY:-1}"
TF_SPLIT_MODE="${TF_SPLIT_MODE:-none}"
TF_SPLIT_PATH="${TF_SPLIT_PATH:-}"
TF_SPLIT_OUT="${TF_SPLIT_OUT:-}"
TF_TRAIN_FRACTION="${TF_TRAIN_FRACTION:-0.7}"
TF_VAL_FRACTION="${TF_VAL_FRACTION:-0.15}"
NO_POS_WEIGHT="${NO_POS_WEIGHT:-false}"
SAVE_EVERY="${SAVE_EVERY:-10}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "${REPO_DIR}/slurm/logs" "${OUTPUT_DIR}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 127
fi
if [[ "${INPUT_MODE}" == "embedding" && -z "${EMBEDDINGS_PATH}" ]]; then
    echo "INPUT_MODE=embedding requires EMBEDDINGS_PATH=/path/to/promoter_embeddings.{npy,parquet}" >&2
    exit 1
fi

CONDA_PREFIX="$(${PYTHON_BIN} -c 'import sys; print(sys.prefix)')"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

echo "Project:              ${PROJECT}"
echo "Repo dir:             ${REPO_DIR}"
echo "Train script:         ${TRAIN_SCRIPT}"
echo "Python:               $(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"
echo "Conda prefix:         ${CONDA_PREFIX}"
echo "Sites:                ${SITES_PATH}"
echo "Regions:              ${REGIONS_PATH}"
echo "Output dir:           ${OUTPUT_DIR}"
echo "Input mode:           ${INPUT_MODE}"
echo "Embeddings:           ${EMBEDDINGS_PATH:-<none>}"
echo "Embedding column:     ${EMBEDDING_COLUMN}"
echo "Embedding key column: ${EMBEDDING_KEY_COLUMN:-<row-aligned>}"
echo "TF embeddings:        ${TF_EMBEDDINGS_PATH:-<none>}"
echo "TF embedding key:     ${TF_EMBEDDING_KEY_COLUMN:-<auto>}"
echo "TF embedding column:  ${TF_EMBEDDING_COLUMN}"
echo "TF name map:          ${TF_NAME_MAP:-<none>}"
echo "Drop missing TF emb:  ${DROP_MISSING_TF_EMBEDDINGS}"
echo "Filter TFs to emb:    ${TF_NAME_FILTER_FROM_EMBEDDINGS}"
echo "Model:                ${MODEL}"
echo "Hidden channels:      ${HIDDEN_CHANNELS}"
echo "Kernel size:          ${KERNEL_SIZE}"
echo "Dilations:            ${DILATIONS}"
echo "Dropout:              ${DROPOUT}"
echo "Epochs:               ${EPOCHS}"
echo "Batch size:           ${BATCH_SIZE}"
echo "LR:                   ${LR}"
echo "Weight decay:         ${WEIGHT_DECAY}"
echo "LR scheduler:         ${LR_SCHEDULER}"
echo "Device:               ${DEVICE}"
echo "Promoter split mode:  ${PROMOTER_SPLIT_MODE}"
echo "Promoter split path:  ${PROMOTER_SPLIT_PATH:-<none>}"
echo "Promoter split out:   ${PROMOTER_SPLIT_OUT:-<output-dir>/promoter_split.json}"
echo "Train fraction:       ${TRAIN_FRACTION}"
echo "Val fraction:         ${VAL_FRACTION}"
echo "Val chroms:           ${VAL_CHROMS:-<none>}"
echo "Test chroms:          ${TEST_CHROMS:-<none>}"
echo "Eval every:           ${EVAL_EVERY}"
echo "TF split mode:        ${TF_SPLIT_MODE}"
echo "TF split path:        ${TF_SPLIT_PATH:-<none>}"
echo "TF split out:         ${TF_SPLIT_OUT:-<output-dir>/tf_split.json}"
echo "TF train fraction:    ${TF_TRAIN_FRACTION}"
echo "TF val fraction:      ${TF_VAL_FRACTION}"
echo "Sequence orientation: ${SEQUENCE_ORIENTATION}"
echo "Include terminal ATG: ${INCLUDE_TERMINAL_ATG}"
echo "No pos weight:        ${NO_POS_WEIGHT}"
echo "Extra args:           ${EXTRA_ARGS}"

OPTIONAL_ARGS=()
if [[ "${INPUT_MODE}" == "embedding" ]]; then
    OPTIONAL_ARGS+=(--embeddings-path "${EMBEDDINGS_PATH}")
    OPTIONAL_ARGS+=(--embedding-column "${EMBEDDING_COLUMN}")
    if [[ -n "${EMBEDDING_KEY_COLUMN}" ]]; then
        OPTIONAL_ARGS+=(--embedding-key-column "${EMBEDDING_KEY_COLUMN}")
    fi
fi
if [[ -n "${TF_EMBEDDINGS_PATH}" ]]; then
    OPTIONAL_ARGS+=(--tf-embeddings-path "${TF_EMBEDDINGS_PATH}")
fi
if [[ -n "${TF_EMBEDDING_KEY_COLUMN}" ]]; then
    OPTIONAL_ARGS+=(--tf-embedding-key-column "${TF_EMBEDDING_KEY_COLUMN}")
fi
if [[ -n "${TF_EMBEDDING_COLUMN}" ]]; then
    OPTIONAL_ARGS+=(--tf-embedding-column "${TF_EMBEDDING_COLUMN}")
fi
if [[ -n "${TF_NAME_MAP}" ]]; then
    OPTIONAL_ARGS+=(--tf-name-map "${TF_NAME_MAP}")
fi
if [[ "${DROP_MISSING_TF_EMBEDDINGS}" == "true" ]]; then
    OPTIONAL_ARGS+=(--drop-missing-tf-embeddings)
fi
if [[ "${TF_NAME_FILTER_FROM_EMBEDDINGS}" == "true" ]]; then
    OPTIONAL_ARGS+=(--tf-name-filter-from-embeddings)
fi
if [[ -n "${MAX_REGIONS}" ]]; then
    OPTIONAL_ARGS+=(--max-regions "${MAX_REGIONS}")
fi
if [[ -n "${PROMOTER_SPLIT_PATH}" ]]; then
    OPTIONAL_ARGS+=(--promoter-split-path "${PROMOTER_SPLIT_PATH}")
fi
if [[ -n "${PROMOTER_SPLIT_OUT}" ]]; then
    OPTIONAL_ARGS+=(--promoter-split-out "${PROMOTER_SPLIT_OUT}")
fi
if [[ -n "${TF_SPLIT_PATH}" ]]; then
    OPTIONAL_ARGS+=(--tf-split-path "${TF_SPLIT_PATH}")
fi
if [[ -n "${TF_SPLIT_OUT}" ]]; then
    OPTIONAL_ARGS+=(--tf-split-out "${TF_SPLIT_OUT}")
fi
if [[ "${NO_POS_WEIGHT}" == "true" ]]; then
    OPTIONAL_ARGS+=(--no-pos-weight)
fi
if [[ "${INCLUDE_TERMINAL_ATG}" == "true" ]]; then
    OPTIONAL_ARGS+=(--include-terminal-atg)
fi

"${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
    --sites-path "${SITES_PATH}" \
    --regions-path "${REGIONS_PATH}" \
    --input-mode "${INPUT_MODE}" \
    --output-dir "${OUTPUT_DIR}" \
    --min-sites-per-tf "${MIN_SITES_PER_TF}" \
    --sequence-orientation "${SEQUENCE_ORIENTATION}" \
    --model "${MODEL}" \
    --hidden-channels "${HIDDEN_CHANNELS}" \
    --kernel-size "${KERNEL_SIZE}" \
    --dilations "${DILATIONS}" \
    --dropout "${DROPOUT}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --lr-scheduler "${LR_SCHEDULER}" \
    --min-lr "${MIN_LR}" \
    --plateau-patience "${PLATEAU_PATIENCE}" \
    --plateau-factor "${PLATEAU_FACTOR}" \
    --num-workers "${NUM_WORKERS}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --promoter-split-mode "${PROMOTER_SPLIT_MODE}" \
    --train-fraction "${TRAIN_FRACTION}" \
    --val-fraction "${VAL_FRACTION}" \
    --val-chroms "${VAL_CHROMS}" \
    --test-chroms "${TEST_CHROMS}" \
    --eval-every "${EVAL_EVERY}" \
    --tf-split-mode "${TF_SPLIT_MODE}" \
    --tf-train-fraction "${TF_TRAIN_FRACTION}" \
    --tf-val-fraction "${TF_VAL_FRACTION}" \
    --save-every "${SAVE_EVERY}" \
    "${OPTIONAL_ARGS[@]}" \
    ${EXTRA_ARGS}

echo "Promoter dense training finished."
