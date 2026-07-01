#!/usr/bin/env bash
set -euo pipefail

PROJECT="${ML4RG_PROJECT:-/s/project/ml4rg_students/2026/project15}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EVAL_SBATCH="${REPO_DIR}/slurm/run_binding_bench_promoter_dense.sbatch"

SPECIESLM_L10="${SPECIESLM_L10:-${PROJECT}/raw/embeddings/other_models/specieslm_v1/_saccharomyces_cerevisiae/embs_ds_fungi_upstream_ATG_1000_l10.npy}"
SCORE_MODE="${SCORE_MODE:-logit}"
NMS_RADIUS_BP="${NMS_RADIUS_BP:-0}"
TOP_K_PER_TF="${TOP_K_PER_TF:-5000}"
PRE_NMS_FACTOR="${PRE_NMS_FACTOR:-20}"

submit_one() {
    local input_mode="$1"
    local model="$2"
    local name="$3"
    local label="$4"
    local export_batch_size="$5"
    local embeddings_path="${6:-}"

    local model_dir="${PROJECT}/working/supervised_baseline/models/${name}"
    local checkpoint="${model_dir}/best.pt"
    if [[ ! -f "${checkpoint}" ]]; then
        echo "Missing checkpoint: ${checkpoint}" >&2
        exit 1
    fi

    export INPUT_MODE="${input_mode}"
    export MODEL="${model}"
    export MODEL_DIR="${model_dir}"
    export METHOD_NAME="${name}_${SCORE_MODE}_nms${NMS_RADIUS_BP}"
    export DISPLAY_NAME="${label}"
    export SCORE_MODE="${SCORE_MODE}"
    export NMS_RADIUS_BP="${NMS_RADIUS_BP}"
    export TOP_K_PER_TF="${TOP_K_PER_TF}"
    export PRE_NMS_FACTOR="${PRE_NMS_FACTOR}"
    export EXPORT_BATCH_SIZE="${export_batch_size}"
    export RUN_DIR="${PROJECT}/working/binding_bench_reports/${METHOD_NAME}"
    export EMBEDDINGS_PATH="${embeddings_path}"

    local job_id
    job_id="$(sbatch --parsable --job-name="bb-${name}" "${EVAL_SBATCH}")"
    echo "${name}: eval=${job_id} report=${RUN_DIR}"
}

submit_one "raw" "dense_small_cnn" "promoter_raw_small_cnn" \
    "Promoter raw small CNN dense" "16"

submit_one "raw" "dense_res_dilated_cnn" "promoter_raw_res_dilated_cnn" \
    "Promoter raw residual dilated CNN dense" "16"

submit_one "embedding" "dense_small_cnn" "promoter_specieslm_l10_small_cnn" \
    "Promoter SpeciesLM l10 small CNN dense" "4" "${SPECIESLM_L10}"

submit_one "embedding" "dense_res_dilated_cnn" "promoter_specieslm_l10_res_dilated_cnn" \
    "Promoter SpeciesLM l10 residual dilated CNN dense" "2" "${SPECIESLM_L10}"
