#!/usr/bin/env bash
set -euo pipefail

PROJECT="${ML4RG_PROJECT:-/s/project/ml4rg_students/2026/project15}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_SBATCH="${REPO_DIR}/slurm/train_supervised.sbatch"
EVAL_SBATCH="${REPO_DIR}/slurm/run_binding_bench_supervised.sbatch"

TF_EMBEDDINGS="${TF_EMBEDDINGS_PATH:-${PROJECT}/working/protein_embeddings/scer_esm2_tf_emb.parquet}"
TF_KEY_COLUMN="${TF_EMBEDDING_KEY_COLUMN:-gene}"

COMMON_EPOCHS="${EPOCHS:-200}"
COMMON_LR="${LR:-0.0005}"
COMMON_WEIGHT_DECAY="${WEIGHT_DECAY:-0.00001}"
COMMON_LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
COMMON_MIN_LR="${MIN_LR:-0.00001}"
COMMON_BATCH_SIZE="${BATCH_SIZE:-512}"
COMMON_SAVE_EVERY="${SAVE_EVERY:-20}"

submit_one() {
    local name="$1"
    local label="$2"
    local hidden_channels="$3"
    local transbind_dilations="$4"
    local no_max_pool="$5"
    local tf_bias="$6"

    export MODEL="transbind_lite"
    export OUTPUT_DIR="${PROJECT}/working/supervised_baseline/models/${name}"
    export TF_EMBEDDINGS_PATH="${TF_EMBEDDINGS}"
    export TF_EMBEDDING_KEY_COLUMN="${TF_KEY_COLUMN}"
    export HIDDEN_CHANNELS="${hidden_channels}"
    export TRANSBIND_DILATIONS="${transbind_dilations}"
    export TRANSBIND_NO_MAX_POOL="${no_max_pool}"
    export TRANSBIND_TF_BIAS="${tf_bias}"
    export EPOCHS="${COMMON_EPOCHS}"
    export LR="${COMMON_LR}"
    export WEIGHT_DECAY="${COMMON_WEIGHT_DECAY}"
    export LR_SCHEDULER="${COMMON_LR_SCHEDULER}"
    export MIN_LR="${COMMON_MIN_LR}"
    export BATCH_SIZE="${COMMON_BATCH_SIZE}"
    export SAVE_EVERY="${COMMON_SAVE_EVERY}"
    export EXTRA_ARGS="--drop-missing-tf-embeddings"

    local train_job
    train_job="$(sbatch --parsable --job-name="tr-${name}" "${TRAIN_SBATCH}")"

    export MODEL="transbind_lite"
    export MODEL_DIR="${OUTPUT_DIR}"
    export METHOD_NAME="supervised_${name}_logit_nms50"
    export DISPLAY_NAME="${label}"
    export SCORE_MODE="logit"
    export NMS_RADIUS_BP="50"
    export SCAN_STRIDE="${SCAN_STRIDE:-1}"
    export TOP_K_PER_TF="${TOP_K_PER_TF:-5000}"
    export PRE_NMS_FACTOR="${PRE_NMS_FACTOR:-20}"
    export EXPORT_BATCH_SIZE="${EXPORT_BATCH_SIZE:-8192}"

    local eval_job
    eval_job="$(sbatch --parsable --dependency="afterok:${train_job}" --job-name="bb-${name}" "${EVAL_SBATCH}")"

    echo "${name}: train=${train_job} eval=${eval_job}"
}

submit_one "transbind_ab_base200" "TransBind ablation: base200" "128" "1,2,4" "false" "false"
submit_one "transbind_ab_hidden256" "TransBind ablation: hidden256" "256" "1,2,4" "false" "false"
submit_one "transbind_ab_long_dil" "TransBind ablation: long dilations" "128" "1,2,4,8,16" "false" "false"
submit_one "transbind_ab_no_pool" "TransBind ablation: no maxpool" "128" "1,2,4" "true" "false"
submit_one "transbind_ab_tf_bias" "TransBind ablation: TF bias" "128" "1,2,4" "false" "true"
