#!/usr/bin/env bash
set -euo pipefail

PROJECT="${ML4RG_PROJECT:-/s/project/ml4rg_students/2026/project15}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DENSE_SBATCH="${REPO_DIR}/slurm/train_promoter_dense.sbatch"
WINDOW_SBATCH="${REPO_DIR}/slurm/train_supervised.sbatch"

ESMDBP="${TF_EMBEDDINGS_PATH:-${PROJECT}/working/protein_embeddings/scer_esmdbp_tf_emb.parquet}"
TF_KEY_COLUMN="${TF_EMBEDDING_KEY_COLUMN:-gene}"

# Use an existing split when provided. Otherwise make a named-similarity split
# around supervisor-suggested easy/recoverable TFs that are present in the
# current S. cerevisiae BindingBench validation table. RAP1/REB1 are not always
# present in this table after filtering, so keep them opt-in via TF_VAL_NAMES.
COMMON_TF_SPLIT_PATH="${TF_SPLIT_PATH:-}"
COMMON_TF_SPLIT_MODE="${TF_SPLIT_MODE:-named_similarity}"
COMMON_TF_VAL_NAMES="${TF_VAL_NAMES:-}"
COMMON_TF_TEST_NAMES="${TF_TEST_NAMES:-abf1,cbf1}"
COMMON_TF_SIMILARITY_THRESHOLD="${TF_SIMILARITY_THRESHOLD:-0.90}"

COMMON_EPOCHS="${EPOCHS:-100}"
COMMON_LR="${LR:-0.0005}"
COMMON_WEIGHT_DECAY="${WEIGHT_DECAY:-0.0001}"
COMMON_BATCH_SIZE="${BATCH_SIZE:-4}"
COMMON_SAVE_EVERY="${SAVE_EVERY:-20}"

submit_dense_motif() {
    export INPUT_MODE="raw"
    export MODEL="dense_protein_motif_cnn"
    export OUTPUT_DIR="${PROJECT}/working/supervised_baseline/models/dense_motifcnn_esmdbp_transfer"
    export TF_EMBEDDINGS_PATH="${ESMDBP}"
    export TF_EMBEDDING_KEY_COLUMN="${TF_KEY_COLUMN}"
    export DROP_MISSING_TF_EMBEDDINGS="true"
    export PROMOTER_SPLIT_MODE="${PROMOTER_SPLIT_MODE:-chromosome}"
    export VAL_CHROMS="${VAL_CHROMS:-VII}"
    export TEST_CHROMS="${TEST_CHROMS:-VIII,IX}"
    export TF_SPLIT_PATH="${COMMON_TF_SPLIT_PATH}"
    export TF_SPLIT_MODE="${COMMON_TF_SPLIT_MODE}"
    export TF_VAL_NAMES="${COMMON_TF_VAL_NAMES}"
    export TF_TEST_NAMES="${COMMON_TF_TEST_NAMES}"
    export TF_SIMILARITY_THRESHOLD="${COMMON_TF_SIMILARITY_THRESHOLD}"
    export HIDDEN_CHANNELS="${HIDDEN_CHANNELS:-256}"
    export MOTIF_KERNEL_SIZES="${MOTIF_KERNEL_SIZES:-7,11,15,21}"
    export DILATIONS="${DILATIONS:-1,2,4,8,16}"
    export DROPOUT="${DROPOUT:-0.15}"
    export TF_EMBEDDING_DROPOUT="${TF_EMBEDDING_DROPOUT:-0.05}"
    export PROTEIN_NOISE_STD="${PROTEIN_NOISE_STD:-0.01}"
    export PROTEIN_L2_NORMALIZE="${PROTEIN_L2_NORMALIZE:-true}"
    export SCORER="${SCORER:-multihead_bilinear}"
    export SCORER_HEADS="${SCORER_HEADS:-8}"
    export SCORER_PAIR_DIM="${SCORER_PAIR_DIM:-32}"
    export SCORER_BIAS_MODE="${SCORER_BIAS_MODE:-tf}"
    export EPOCHS="${COMMON_EPOCHS}"
    export BATCH_SIZE="${COMMON_BATCH_SIZE}"
    export LR="${COMMON_LR}"
    export WEIGHT_DECAY="${COMMON_WEIGHT_DECAY}"
    export LR_SCHEDULER="${LR_SCHEDULER:-cosine}"
    export SELECTION_METRIC="${SELECTION_METRIC:-val_average_precision}"
    export SAVE_EVERY="${COMMON_SAVE_EVERY}"

    sbatch --parsable --job-name="dense-motif-esmdbp" "${DENSE_SBATCH}"
}

submit_window_transbind() {
    export MODEL="transbind_lite"
    export OUTPUT_DIR="${PROJECT}/working/supervised_baseline/models/window_transbindlite_esmdbp_transfer"
    export TF_EMBEDDINGS_PATH="${ESMDBP}"
    export TF_EMBEDDING_KEY_COLUMN="${TF_KEY_COLUMN}"
    export DROP_MISSING_TF_EMBEDDINGS="true"
    export TF_SPLIT_PATH="${COMMON_TF_SPLIT_PATH}"
    export TF_SPLIT_MODE="${COMMON_TF_SPLIT_MODE}"
    export TF_VAL_NAMES="${COMMON_TF_VAL_NAMES}"
    export TF_TEST_NAMES="${COMMON_TF_TEST_NAMES}"
    export TF_SIMILARITY_THRESHOLD="${COMMON_TF_SIMILARITY_THRESHOLD}"
    export WINDOW_SIZE="${WINDOW_SIZE:-1001}"
    export NEGATIVE_RATIO="${NEGATIVE_RATIO:-1.0}"
    export HIDDEN_CHANNELS="${WINDOW_HIDDEN_CHANNELS:-256}"
    export TRANSBIND_DILATIONS="${TRANSBIND_DILATIONS:-1,2,4,8}"
    export DROPOUT="${WINDOW_DROPOUT:-0.15}"
    export NUM_HEADS="${NUM_HEADS:-8}"
    export EPOCHS="${WINDOW_EPOCHS:-${COMMON_EPOCHS}}"
    export BATCH_SIZE="${WINDOW_BATCH_SIZE:-128}"
    export LR="${WINDOW_LR:-${COMMON_LR}}"
    export WEIGHT_DECAY="${WINDOW_WEIGHT_DECAY:-${COMMON_WEIGHT_DECAY}}"
    export LR_SCHEDULER="${WINDOW_LR_SCHEDULER:-cosine}"
    export SELECTION_METRIC="${WINDOW_SELECTION_METRIC:-val_average_precision}"
    export SAVE_EVERY="${WINDOW_SAVE_EVERY:-${COMMON_SAVE_EVERY}}"

    sbatch --parsable --job-name="win-tb-esmdbp" "${WINDOW_SBATCH}"
}

dense_job="$(submit_dense_motif)"
window_job="$(submit_window_transbind)"

echo "dense_motifcnn_esmdbp_transfer=${dense_job}"
echo "window_transbindlite_esmdbp_transfer=${window_job}"
