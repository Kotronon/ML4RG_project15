#!/usr/bin/env bash
#SBATCH --job-name=sup-cnn
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --partition=student_project
#SBATCH --output=slurm/logs/supervised-%j.out
#SBATCH --error=slurm/logs/supervised-%j.err
#SBATCH --gres=gpu:1

set -euo pipefail

PROJECT="${ML4RG_PROJECT:-/s/project/ml4rg_students/2026/project15}"
PYTHON_BIN="${PYTHON_BIN:-/opt/modules/i12g/anaconda/envs/ml4rg_project15/bin/python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${REPO_DIR:-}" ]]; then
    if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/supervised_baseline/train.py" ]]; then
        REPO_DIR="${SLURM_SUBMIT_DIR}"
    else
        REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
    fi
fi

TRAIN_SCRIPT="${TRAIN_SCRIPT:-${REPO_DIR}/supervised_baseline/train.py}"
SITES_PATH="${SITES_PATH:-/s/project/multispecies/fungi_code/tf_sae/binding_bench_datasets/val/dna/_saccharomyces_cerevisiae/DNA_rossi_chipexo_sites.parquet}"
REGIONS_PATH="${REGIONS_PATH:-/s/project/multispecies/fungi_code/tf_sae/processed/sequence_datasets/fungi_upstream_ATG_1000/_saccharomyces_cerevisiae_sequence_mapper.parquet}"

MODEL="${MODEL:-small_cnn}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT}/working/supervised_baseline/models/${MODEL}_overfit}"
WINDOW_SIZE="${WINDOW_SIZE:-101}"
NEGATIVE_RATIO="${NEGATIVE_RATIO:-1.0}"
MIN_SITES_PER_TF="${MIN_SITES_PER_TF:-15}"
HIDDEN_CHANNELS="${HIDDEN_CHANNELS:-128}"
KERNEL_SIZE="${KERNEL_SIZE:-7}"
DILATIONS="${DILATIONS:-1,2,4,8,16}"
TRANSBIND_DILATIONS="${TRANSBIND_DILATIONS:-1,2,4}"
DROPOUT="${DROPOUT:-0.1}"
NUM_HEADS="${NUM_HEADS:-8}"
TF_EMBEDDINGS_PATH="${TF_EMBEDDINGS_PATH:-}"
TF_EMBEDDING_KEY_COLUMN="${TF_EMBEDDING_KEY_COLUMN:-}"
TF_EMBEDDING_COLUMN="${TF_EMBEDDING_COLUMN:-emb}"
TF_NAME_MAP="${TF_NAME_MAP:-}"
TF_NAME_FILTER_FROM_EMBEDDINGS="${TF_NAME_FILTER_FROM_EMBEDDINGS:-false}"
EXCLUDE_TF_NAMES="${EXCLUDE_TF_NAMES:-}"
EXCLUDE_TF_NAMES_PATH="${EXCLUDE_TF_NAMES_PATH:-}"
TRANSBIND_NO_MAX_POOL="${TRANSBIND_NO_MAX_POOL:-false}"
TRANSBIND_TF_BIAS="${TRANSBIND_TF_BIAS:-false}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-512}"
LR="${LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
LR_SCHEDULER="${LR_SCHEDULER:-none}"
MIN_LR="${MIN_LR:-0.00001}"
PLATEAU_PATIENCE="${PLATEAU_PATIENCE:-10}"
PLATEAU_FACTOR="${PLATEAU_FACTOR:-0.5}"
NUM_WORKERS="${NUM_WORKERS:-0}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-auto}"
SEQUENCE_ORIENTATION="${SEQUENCE_ORIENTATION:-strand-aware}"
SAVE_EVERY="${SAVE_EVERY:-10}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "${REPO_DIR}/slurm/logs" "${OUTPUT_DIR}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    echo "Set PYTHON_BIN to the full Python path from the ml4rg_project15 environment." >&2
    exit 127
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
echo "Model:                ${MODEL}"
echo "Window size:          ${WINDOW_SIZE}"
echo "Negative ratio:       ${NEGATIVE_RATIO}"
echo "Min sites per TF:     ${MIN_SITES_PER_TF}"
echo "Hidden channels:      ${HIDDEN_CHANNELS}"
echo "Kernel size:          ${KERNEL_SIZE}"
echo "Dilations:            ${DILATIONS}"
echo "TransBind dilations:  ${TRANSBIND_DILATIONS}"
echo "Dropout:              ${DROPOUT}"
echo "Num heads:            ${NUM_HEADS}"
echo "TF embeddings:        ${TF_EMBEDDINGS_PATH:-<none>}"
echo "TF embedding key:     ${TF_EMBEDDING_KEY_COLUMN:-<auto>}"
echo "TF embedding column:  ${TF_EMBEDDING_COLUMN}"
echo "TF name map:          ${TF_NAME_MAP:-<none>}"
echo "Filter TFs to emb:    ${TF_NAME_FILTER_FROM_EMBEDDINGS}"
echo "Exclude TF names:     ${EXCLUDE_TF_NAMES:-<none>}"
echo "Exclude TF file:      ${EXCLUDE_TF_NAMES_PATH:-<none>}"
echo "TransBind no pool:    ${TRANSBIND_NO_MAX_POOL}"
echo "TransBind TF bias:    ${TRANSBIND_TF_BIAS}"
echo "Epochs:               ${EPOCHS}"
echo "Batch size:           ${BATCH_SIZE}"
echo "LR:                   ${LR}"
echo "Weight decay:         ${WEIGHT_DECAY}"
echo "LR scheduler:         ${LR_SCHEDULER}"
echo "Min LR:               ${MIN_LR}"
echo "Device:               ${DEVICE}"
echo "Sequence orientation: ${SEQUENCE_ORIENTATION}"
echo "Extra args:           ${EXTRA_ARGS}"

OPTIONAL_ARGS=()
if [[ "${TF_NAME_FILTER_FROM_EMBEDDINGS}" == "true" ]]; then
    OPTIONAL_ARGS+=(--tf-name-filter-from-embeddings)
fi
if [[ -n "${EXCLUDE_TF_NAMES}" ]]; then
    OPTIONAL_ARGS+=(--exclude-tf-names "${EXCLUDE_TF_NAMES}")
fi
if [[ -n "${EXCLUDE_TF_NAMES_PATH}" ]]; then
    OPTIONAL_ARGS+=(--exclude-tf-names-path "${EXCLUDE_TF_NAMES_PATH}")
fi
if [[ "${TRANSBIND_NO_MAX_POOL}" == "true" ]]; then
    OPTIONAL_ARGS+=(--transbind-no-max-pool)
fi
if [[ "${TRANSBIND_TF_BIAS}" == "true" ]]; then
    OPTIONAL_ARGS+=(--transbind-tf-bias)
fi

"${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
    --sites-path "${SITES_PATH}" \
    --regions-path "${REGIONS_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --model "${MODEL}" \
    --window-size "${WINDOW_SIZE}" \
    --negative-ratio "${NEGATIVE_RATIO}" \
    --min-sites-per-tf "${MIN_SITES_PER_TF}" \
    --hidden-channels "${HIDDEN_CHANNELS}" \
    --kernel-size "${KERNEL_SIZE}" \
    --dilations "${DILATIONS}" \
    --transbind-dilations "${TRANSBIND_DILATIONS}" \
    --dropout "${DROPOUT}" \
    --num-heads "${NUM_HEADS}" \
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
    --sequence-orientation "${SEQUENCE_ORIENTATION}" \
    --save-every "${SAVE_EVERY}" \
    ${TF_EMBEDDINGS_PATH:+--tf-embeddings-path "${TF_EMBEDDINGS_PATH}"} \
    ${TF_EMBEDDING_KEY_COLUMN:+--tf-embedding-key-column "${TF_EMBEDDING_KEY_COLUMN}"} \
    --tf-embedding-column "${TF_EMBEDDING_COLUMN}" \
    ${TF_NAME_MAP:+--tf-name-map "${TF_NAME_MAP}"} \
    "${OPTIONAL_ARGS[@]}" \
    ${EXTRA_ARGS}

echo "Supervised training finished."
