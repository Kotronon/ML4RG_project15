#!/usr/bin/env bash
set -euo pipefail

PROJECT="${ML4RG_PROJECT:-/s/project/ml4rg_students/2026/project15}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_SBATCH="${REPO_DIR}/slurm/train_promoter_dense.sbatch"
EVAL_SBATCH="${REPO_DIR}/slurm/run_binding_bench_promoter_dense.sbatch"

PROMOTER_EMBEDDINGS_PATH="${EMBEDDINGS_PATH:-${SPECIESLM_EMBEDDINGS_PATH:-}}"
EMBEDDING_COLUMN="${EMBEDDING_COLUMN:-emb}"
EMBEDDING_KEY_COLUMN="${EMBEDDING_KEY_COLUMN:-}"

COMMON_EPOCHS="${EPOCHS:-100}"
COMMON_LR="${LR:-0.001}"
COMMON_WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
COMMON_LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
COMMON_MIN_LR="${MIN_LR:-0.00001}"
COMMON_SAVE_EVERY="${SAVE_EVERY:-10}"
RAW_BATCH_SIZE="${RAW_BATCH_SIZE:-8}"
EMBEDDING_BATCH_SIZE="${EMBEDDING_BATCH_SIZE:-4}"
TOP_K_PER_TF="${TOP_K_PER_TF:-5000}"
NMS_RADIUS_BP="${NMS_RADIUS_BP:-0}"
SCORE_MODE="${SCORE_MODE:-logit}"

if [[ -z "${PROMOTER_EMBEDDINGS_PATH}" ]]; then
    cat >&2 <<'EOF'
Set EMBEDDINGS_PATH=/path/to/promoter_position_embeddings.{npy,parquet}
before launching the embedding baselines.

The raw baselines can still be submitted directly with:
  INPUT_MODE=raw MODEL=dense_small_cnn sbatch slurm/train_promoter_dense.sbatch
EOF
    exit 1
fi

submit_one() {
    local input_mode="$1"
    local model="$2"
    local name="$3"
    local label="$4"
    local hidden_channels="$5"
    local batch_size="$6"

    export INPUT_MODE="${input_mode}"
    export MODEL="${model}"
    export OUTPUT_DIR="${PROJECT}/working/supervised_baseline/models/${name}"
    export HIDDEN_CHANNELS="${hidden_channels}"
    export EPOCHS="${COMMON_EPOCHS}"
    export LR="${COMMON_LR}"
    export WEIGHT_DECAY="${COMMON_WEIGHT_DECAY}"
    export LR_SCHEDULER="${COMMON_LR_SCHEDULER}"
    export MIN_LR="${COMMON_MIN_LR}"
    export BATCH_SIZE="${batch_size}"
    export SAVE_EVERY="${COMMON_SAVE_EVERY}"
    export DILATIONS="${DILATIONS:-1,2,4,8,16}"
    export EMBEDDINGS_PATH=""
    export EMBEDDING_COLUMN="${EMBEDDING_COLUMN}"
    export EMBEDDING_KEY_COLUMN="${EMBEDDING_KEY_COLUMN}"

    if [[ "${input_mode}" == "embedding" ]]; then
        export EMBEDDINGS_PATH="${PROMOTER_EMBEDDINGS_PATH}"
    fi

    local train_job
    train_job="$(sbatch --parsable --job-name="tr-${name}" "${TRAIN_SBATCH}")"

    export MODEL_DIR="${OUTPUT_DIR}"
    export METHOD_NAME="promoter_${name}_${SCORE_MODE}_nms${NMS_RADIUS_BP}"
    export DISPLAY_NAME="${label}"
    export TOP_K_PER_TF="${TOP_K_PER_TF}"
    export NMS_RADIUS_BP="${NMS_RADIUS_BP}"
    export SCORE_MODE="${SCORE_MODE}"
    export EXPORT_BATCH_SIZE="${batch_size}"

    local eval_job
    eval_job="$(sbatch --parsable --dependency="afterok:${train_job}" --job-name="bb-${name}" "${EVAL_SBATCH}")"

    echo "${name}: train=${train_job} eval=${eval_job}"
}

submit_one "raw" "dense_small_cnn" "promoter_raw_small_cnn" \
    "Promoter raw small CNN" "128" "${RAW_BATCH_SIZE}"
submit_one "raw" "dense_res_dilated_cnn" "promoter_raw_res_dilated_cnn" \
    "Promoter raw residual dilated CNN" "256" "${RAW_BATCH_SIZE}"
submit_one "embedding" "dense_small_cnn" "promoter_embedding_small_cnn" \
    "Promoter embedding small CNN" "128" "${EMBEDDING_BATCH_SIZE}"
submit_one "embedding" "dense_res_dilated_cnn" "promoter_embedding_res_dilated_cnn" \
    "Promoter embedding residual dilated CNN" "256" "${EMBEDDING_BATCH_SIZE}"
