#!/usr/bin/env bash
#SBATCH --job-name=sae-train
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --partition=student_project
#SBATCH --output=slurm/logs/sae-train-%j.out
#SBATCH --error=slurm/logs/sae-train-%j.err

set -euo pipefail

PROJECT="${ML4RG_PROJECT:-/s/project/ml4rg_students/2026/project15}"
PYTHON_BIN="${PYTHON_BIN:-/opt/modules/i12g/anaconda/envs/ml4rg_project15/bin/python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

EMBEDDING_PATH="${EMBEDDING_PATH:-/s/project/ml4rg_students/2026/project15/raw/embeddings/other_models/specieslm_v1/_saccharomyces_cerevisiae/embs_ds_fungi_upstream_ATG_1000_l10.npy}"
SEQ_OFFSET="${SEQ_OFFSET:-1}"
SEQ_LEN="${SEQ_LEN:-1000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT}/working/sae/models}"

# SAE hyperparameters
VARIANT="${VARIANT:-topk}"          # vanilla | topk | batchtopk
EXPANSION_FACTOR="${EXPANSION_FACTOR:-8}"
K="${K:-64}"                         # active features per sample (TopK / BatchTopK)
K_FRACTION="${K_FRACTION:-}"         # if set, overrides K as fraction of hidden dim
L1_COEFF="${L1_COEFF:-0.001}"        # VanillaSAE only

# Training
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-512}"
LR="${LR:-2e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0}"
VAL_FRACTION="${VAL_FRACTION:-0.05}"
NUM_WORKERS="${NUM_WORKERS:-4}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-auto}"
SAVE_EVERY="${SAVE_EVERY:-10}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

CONDA_PREFIX="$(${PYTHON_BIN} -c 'import sys; print(sys.prefix)')"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

mkdir -p "${REPO_DIR}/slurm/logs"

# Build optional k-fraction argument
K_FRAC_ARG=""
if [[ -n "${K_FRACTION}" ]]; then
    K_FRAC_ARG="--k-fraction ${K_FRACTION}"
fi

echo "Embedding:        ${EMBEDDING_PATH}"
echo "Variant:          ${VARIANT}"
echo "Expansion factor: ${EXPANSION_FACTOR}"
echo "K:                ${K}"
echo "Epochs:           ${EPOCHS}"

"${PYTHON_BIN}" -m sae.train \
    --embedding-path "${EMBEDDING_PATH}" \
    --seq-offset "${SEQ_OFFSET}" \
    --seq-len "${SEQ_LEN}" \
    --output-root "${OUTPUT_ROOT}" \
    --variant "${VARIANT}" \
    --expansion-factor "${EXPANSION_FACTOR}" \
    --k "${K}" \
    ${K_FRAC_ARG} \
    --l1-coeff "${L1_COEFF}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --val-fraction "${VAL_FRACTION}" \
    --num-workers "${NUM_WORKERS}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --save-every "${SAVE_EVERY}" \
    ${EXTRA_ARGS}

echo "SAE training finished."
