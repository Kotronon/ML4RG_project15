#!/usr/bin/env bash
#SBATCH --job-name=sup-cnn
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/supervised-%j.out
#SBATCH --error=slurm/logs/supervised-%j.err

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
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT}/working/supervised_baseline/models/small_cnn_overfit}"

WINDOW_SIZE="${WINDOW_SIZE:-101}"
NEGATIVE_RATIO="${NEGATIVE_RATIO:-1.0}"
MIN_SITES_PER_TF="${MIN_SITES_PER_TF:-15}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-512}"
LR="${LR:-0.001}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
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

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

echo "Project:              ${PROJECT}"
echo "Repo dir:             ${REPO_DIR}"
echo "Train script:         ${TRAIN_SCRIPT}"
echo "Python:               $(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"
echo "Sites:                ${SITES_PATH}"
echo "Regions:              ${REGIONS_PATH}"
echo "Output dir:           ${OUTPUT_DIR}"
echo "Window size:          ${WINDOW_SIZE}"
echo "Negative ratio:       ${NEGATIVE_RATIO}"
echo "Min sites per TF:     ${MIN_SITES_PER_TF}"
echo "Epochs:               ${EPOCHS}"
echo "Batch size:           ${BATCH_SIZE}"
echo "Device:               ${DEVICE}"
echo "Sequence orientation: ${SEQUENCE_ORIENTATION}"
echo "Extra args:           ${EXTRA_ARGS}"

"${PYTHON_BIN}" "${TRAIN_SCRIPT}" \
    --sites-path "${SITES_PATH}" \
    --regions-path "${REGIONS_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --window-size "${WINDOW_SIZE}" \
    --negative-ratio "${NEGATIVE_RATIO}" \
    --min-sites-per-tf "${MIN_SITES_PER_TF}" \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --lr "${LR}" \
    --weight-decay "${WEIGHT_DECAY}" \
    --num-workers "${NUM_WORKERS}" \
    --seed "${SEED}" \
    --device "${DEVICE}" \
    --sequence-orientation "${SEQUENCE_ORIENTATION}" \
    --save-every "${SAVE_EVERY}" \
    ${EXTRA_ARGS}

echo "Supervised training finished."
