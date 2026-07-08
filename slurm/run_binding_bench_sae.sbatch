#!/usr/bin/env bash
#SBATCH --job-name=bb-sae
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --partition=student_project
#SBATCH --output=slurm/logs/binding-bench-sae-%j.out
#SBATCH --error=slurm/logs/binding-bench-sae-%j.err

set -euo pipefail

PROJECT="${ML4RG_PROJECT:-/s/project/ml4rg_students/2026/project15}"
PYTHON_BIN="${PYTHON_BIN:-/opt/modules/i12g/anaconda/envs/ml4rg_project15/bin/python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${REPO_DIR:-}" ]]; then
    if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/sae/export_binding_bench_predictions.py" ]]; then
        REPO_DIR="${SLURM_SUBMIT_DIR}"
    else
        REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
    fi
fi

BINDING_BENCH_REPO="${BINDING_BENCH_REPO:-${HOME}/binding_bench}"
BINDING_BENCH_BIN="${BINDING_BENCH_BIN:-binding_bench}"
TARGET_DATASET="${TARGET_DATASET:-DNA_rossi_chipexo}"

CHECKPOINT="${CHECKPOINT:-${PROJECT}/working/sae/models/topk_ef8_k32/best.pt}"
EMBEDDING_PATH="${EMBEDDING_PATH:-/s/project/ml4rg_students/2026/project15/raw/embeddings/other_models/specieslm_v1/_saccharomyces_cerevisiae/embs_ds_fungi_upstream_ATG_1000_l10.npy}"
REGIONS_PATH="${REGIONS_PATH:-/s/project/multispecies/fungi_code/tf_sae/processed/sequence_datasets/fungi_upstream_ATG_1000/_saccharomyces_cerevisiae_sequence_mapper.parquet}"
SEQ_OFFSET="${SEQ_OFFSET:-1}"
SEQ_LEN="${SEQ_LEN:-1000}"

NMS_RADIUS_BP="${NMS_RADIUS_BP:-50}"
PEAK_CALLER="${PEAK_CALLER:-nms}"
PRE_NMS_FACTOR="${PRE_NMS_FACTOR:-5}"
TOP_K_PER_FEATURE="${TOP_K_PER_FEATURE:-5000}"
FEATURE_RANK_METHOD="${FEATURE_RANK_METHOD:-max-score}"
FEATURE_RANK_NAME="${FEATURE_RANK_NAME:-sae_activation}"
BATCH_SIZE="${BATCH_SIZE:-1000}"
DEVICE="${DEVICE:-auto}"
MAX_REGIONS="${MAX_REGIONS:-}"
EXTRA_EXPORT_ARGS="${EXTRA_EXPORT_ARGS:-}"

METHOD_NAME="${METHOD_NAME:-sae_topk_nms${NMS_RADIUS_BP}}"
DISPLAY_NAME="${DISPLAY_NAME:-SAE TopK (nms=${NMS_RADIUS_BP})}"
INPUT_DIR="${PROJECT}/working/binding_bench_inputs"
RUN_DIR="${RUN_DIR:-${PROJECT}/working/binding_bench_runs/${METHOD_NAME}}"
PREDICTIONS="${PREDICTIONS:-${INPUT_DIR}/${METHOD_NAME}_predictions.parquet}"
FEATURE_RANKS="${FEATURE_RANKS:-${INPUT_DIR}/${METHOD_NAME}_feature_ranks.parquet}"

mkdir -p "${INPUT_DIR}" "${RUN_DIR}" "${REPO_DIR}/slurm/logs"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 127
fi

CONDA_PREFIX="$("${PYTHON_BIN}" -c 'import sys; print(sys.prefix)')"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

echo "Project:              ${PROJECT}"
echo "Repo dir:             ${REPO_DIR}"
echo "Checkpoint:           ${CHECKPOINT}"
echo "Embeddings:           ${EMBEDDING_PATH}"
echo "Predictions:          ${PREDICTIONS}"
echo "Feature ranks:        ${FEATURE_RANKS}"
echo "Run dir:              ${RUN_DIR}"
echo "Target dataset:       ${TARGET_DATASET}"
echo "Top K per feature:    ${TOP_K_PER_FEATURE}"
echo "NMS radius bp:        ${NMS_RADIUS_BP}"
echo "Peak caller:          ${PEAK_CALLER}"

OPTIONAL_ARGS=()
if [[ -n "${MAX_REGIONS}" ]]; then
    OPTIONAL_ARGS+=(--max-regions "${MAX_REGIONS}")
fi

"${PYTHON_BIN}" -m sae.export_binding_bench_predictions \
    --checkpoint "${CHECKPOINT}" \
    --npy-path "${EMBEDDING_PATH}" \
    --regions-path "${REGIONS_PATH}" \
    --predictions-out "${PREDICTIONS}" \
    --feature-ranks-out "${FEATURE_RANKS}" \
    --seq-offset "${SEQ_OFFSET}" \
    --seq-len "${SEQ_LEN}" \
    --top-k-per-feature "${TOP_K_PER_FEATURE}" \
    --nms-radius-bp "${NMS_RADIUS_BP}" \
    --peak-caller "${PEAK_CALLER}" \
    --pre-nms-factor "${PRE_NMS_FACTOR}" \
    --feature-rank-method "${FEATURE_RANK_METHOD}" \
    --batch-size "${BATCH_SIZE}" \
    --device "${DEVICE}" \
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

echo "SAE Binding Bench run finished."
echo "Best-assignment tables: ${RUN_DIR}/binding_bench/discrete/${TARGET_DATASET}/full_best_assnt/"
echo "Run matched AUROC with:"
echo "  BEST_ASSIGNMENT_PATH=${RUN_DIR} sbatch slurm/eval_sae.sbatch"
