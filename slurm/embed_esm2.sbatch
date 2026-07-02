#!/usr/bin/env bash
#SBATCH --job-name=embed-esm2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --partition=student_project
#SBATCH --gres=gpu:1
#SBATCH --output=slurm/logs/embed-esm2-%j.out
#SBATCH --error=slurm/logs/embed-esm2-%j.err

set -euo pipefail

PROJECT="${ML4RG_PROJECT:-/s/project/ml4rg_students/2026/project15}"
PYTHON_BIN="${PYTHON_BIN:-/opt/modules/i12g/anaconda/envs/ml4rg_project15/bin/python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${REPO_DIR:-}" ]]; then
    if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/supervised_baseline/embed_proteome_esm2.py" ]]; then
        REPO_DIR="${SLURM_SUBMIT_DIR}"
    else
        REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
    fi
fi

EMBED_SCRIPT="${EMBED_SCRIPT:-${REPO_DIR}/supervised_baseline/embed_proteome_esm2.py}"
OUTPUT="${OUTPUT:-${PROJECT}/working/protein_embeddings/scer_esm2_emb.parquet}"
KEEP_NAMES="${KEEP_NAMES:-}"
MODEL="${ESM_MODEL:-facebook/esm2_t33_650M_UR50D}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
DEVICE="${DEVICE:-cuda}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

mkdir -p "${REPO_DIR}/slurm/logs" "$(dirname "${OUTPUT}")"

resolve_fasta() {
    if [[ -n "${FASTA:-}" ]]; then
        printf '%s\n' "${FASTA}"
        return 0
    fi

    local candidates=(
        "${PROJECT}/raw/protein_sequences/orf_trans_all.fasta"
        "${PROJECT}/raw/protein_sequences/orf_trans_all.fasta.gz"
        "${PROJECT}/raw/proteomes/orf_trans_all.fasta"
        "${PROJECT}/raw/proteomes/orf_trans_all.fasta.gz"
        "${PROJECT}/raw/sgd/orf_trans_all.fasta"
        "${PROJECT}/raw/sgd/orf_trans_all.fasta.gz"
        "${PROJECT}/working/SGD/orf_trans_all.fasta"
        "${PROJECT}/working/SGD/orf_trans_all.fasta.gz"
        "${PROJECT}/working/protein_embeddings/orf_trans_all.fasta"
        "${PROJECT}/working/protein_embeddings/orf_trans_all.fasta.gz"
        "${REPO_DIR}/orf_trans_all.fasta"
        "${REPO_DIR}/orf_trans_all.fasta.gz"
        "${HOME}/project15/raw/protein_sequences/orf_trans_all.fasta"
        "${HOME}/project15/raw/proteomes/orf_trans_all.fasta"
        "${HOME}/project15/raw/sgd/orf_trans_all.fasta"
        "${HOME}/project15/working/SGD/orf_trans_all.fasta"
    )

    local candidate
    for candidate in "${candidates[@]}"; do
        if [[ -f "${candidate}" ]]; then
            printf '%s\n' "${candidate}"
            return 0
        fi
    done

    local found
    found="$(
        find "${PROJECT}" "${HOME}" -maxdepth 7 \
            \( -name 'orf_trans_all.fasta' -o -name 'orf_trans_all.fasta.gz' \) \
            -type f -print -quit 2>/dev/null || true
    )"
    if [[ -n "${found}" ]]; then
        printf '%s\n' "${found}"
        return 0
    fi

    return 1
}

if ! FASTA="$(resolve_fasta)"; then
    echo "Could not find orf_trans_all.fasta automatically." >&2
    echo "Set FASTA=/path/to/orf_trans_all.fasta or FASTA=/path/to/orf_trans_all.fasta.gz." >&2
    exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 127
fi
if [[ ! -f "${FASTA}" ]]; then
    echo "FASTA not found: ${FASTA}" >&2
    echo "Set FASTA=/path/to/orf_trans_all.fasta or place it in the repo root." >&2
    exit 1
fi

CONDA_PREFIX="$(${PYTHON_BIN} -c 'import sys; print(sys.prefix)')"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

echo "Repo dir:     ${REPO_DIR}"
echo "Python:       $(${PYTHON_BIN} -c 'import sys; print(sys.executable)')"
echo "Conda prefix: ${CONDA_PREFIX}"
echo "FASTA:        ${FASTA}"
echo "Output:       ${OUTPUT}"
echo "Keep names:   ${KEEP_NAMES:-<all proteins>}"
echo "ESM model:    ${MODEL}"
echo "Batch size:   ${BATCH_SIZE}"
echo "Max length:   ${MAX_LENGTH}"
echo "Device:       ${DEVICE}"
echo "Extra args:   ${EXTRA_ARGS}"

"${PYTHON_BIN}" "${EMBED_SCRIPT}" \
    "${FASTA}" \
    --output "${OUTPUT}" \
    --model "${MODEL}" \
    --batch-size "${BATCH_SIZE}" \
    --max-length "${MAX_LENGTH}" \
    --device "${DEVICE}" \
    ${KEEP_NAMES:+--keep-names "${KEEP_NAMES}"} \
    ${EXTRA_ARGS}

echo "ESM-2 embedding finished: ${OUTPUT}"
