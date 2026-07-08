#!/usr/bin/env bash
#SBATCH --job-name=sae-eval
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --partition=student_project
#SBATCH --output=slurm/logs/sae-eval-%j.out
#SBATCH --error=slurm/logs/sae-eval-%j.err

set -euo pipefail

PROJECT="${ML4RG_PROJECT:-/s/project/ml4rg_students/2026/project15}"
PYTHON_BIN="${PYTHON_BIN:-/opt/modules/i12g/anaconda/envs/ml4rg_project15/bin/python}"
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"

CHECKPOINT="${CHECKPOINT:-${PROJECT}/working/sae/models/topk_ef8_k32/best.pt}"
EMBEDDING_PATH="${EMBEDDING_PATH:-/s/project/ml4rg_students/2026/project15/raw/embeddings/other_models/specieslm_v1/_saccharomyces_cerevisiae/embs_ds_fungi_upstream_ATG_1000_l10.npy}"
HALF_WINDOW="${HALF_WINDOW:-5}"
SEQ_OFFSET="${SEQ_OFFSET:-1}"
SEQ_LEN="${SEQ_LEN:-1000}"
SITES_PATH="${SITES_PATH:-/s/project/multispecies/fungi_code/tf_sae/binding_bench_datasets/val/dna/_saccharomyces_cerevisiae/DNA_rossi_chipexo_sites.parquet}"
REGIONS_PATH="${REGIONS_PATH:-/s/project/multispecies/fungi_code/tf_sae/processed/sequence_datasets/fungi_upstream_ATG_1000/_saccharomyces_cerevisiae_sequence_mapper.parquet}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT}/working/sae/evals}"

MIN_SITES_PER_TF="${MIN_SITES_PER_TF:-15}"
BATCH_SIZE="${BATCH_SIZE:-512}"
NUM_WORKERS="${NUM_WORKERS:-0}"
DEVICE="${DEVICE:-auto}"
TOP_N_FEATURES="${TOP_N_FEATURES:-10}"
MAX_EVAL_POSITIONS="${MAX_EVAL_POSITIONS:-500000}"
GENE_HELD_OUT="${GENE_HELD_OUT:-false}"
TEST_FRACTION="${TEST_FRACTION:-0.2}"
BEST_ASSIGNMENT_PATH="${BEST_ASSIGNMENT_PATH:-}"
ASSIGNMENT_METRIC="${ASSIGNMENT_METRIC:-jaccard}"
ASSIGNMENT_DATASET="${ASSIGNMENT_DATASET:-DNA_rossi_chipexo}"
ALL_FEATURES="${ALL_FEATURES:-false}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

if [[ "${GENE_HELD_OUT}" == "true" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} --gene-held-out"
fi
if [[ -n "${BEST_ASSIGNMENT_PATH}" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} --best-assignment-path ${BEST_ASSIGNMENT_PATH} --assignment-metric ${ASSIGNMENT_METRIC} --assignment-dataset ${ASSIGNMENT_DATASET}"
    if [[ "${ALL_FEATURES}" == "true" ]]; then
        EXTRA_ARGS="${EXTRA_ARGS} --all-features"
    fi
fi

CONDA_PREFIX="$(${PYTHON_BIN} -c 'import sys; print(sys.prefix)')"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

mkdir -p "${REPO_DIR}/slurm/logs"

echo "Checkpoint: ${CHECKPOINT}"
echo "Embeddings: ${EMBEDDING_PATH}"
if [[ -n "${BEST_ASSIGNMENT_PATH}" ]]; then
    echo "Best assignment: ${BEST_ASSIGNMENT_PATH}"
fi

"${PYTHON_BIN}" -m sae.evaluate \
    --checkpoint "${CHECKPOINT}" \
    --npy-path "${EMBEDDING_PATH}" \
    --sites-path "${SITES_PATH}" \
    --regions-path "${REGIONS_PATH}" \
    --output-root "${OUTPUT_ROOT}" \
    --half-window "${HALF_WINDOW}" \
    --min-sites-per-tf "${MIN_SITES_PER_TF}" \
    --seq-offset "${SEQ_OFFSET}" \
    --seq-len "${SEQ_LEN}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --device "${DEVICE}" \
    --top-n-features "${TOP_N_FEATURES}" \
    --max-eval-positions "${MAX_EVAL_POSITIONS}" \
    --test-fraction "${TEST_FRACTION}" \
    ${EXTRA_ARGS}

echo "SAE evaluation finished."
