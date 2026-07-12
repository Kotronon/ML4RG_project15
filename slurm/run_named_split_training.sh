#!/usr/bin/env bash
set -euo pipefail

ML4RG_ENV_BIN="${ML4RG_ENV_BIN:-/Users/kathi/miniforge3/envs/ml4rg/bin}"
if [[ -d "$ML4RG_ENV_BIN" ]]; then
  export PATH="$ML4RG_ENV_BIN:$PATH"
  hash -r 2>/dev/null || true
fi

PYTHON_BIN="$(command -v python3)"
BINDING_BENCH_BIN="$(command -v binding_bench)"
if [[ "$PYTHON_BIN" != "$ML4RG_ENV_BIN/"* || "$BINDING_BENCH_BIN" != "$ML4RG_ENV_BIN/"* ]]; then
  echo "Expected ml4rg tools from $ML4RG_ENV_BIN, but got:" >&2
  echo "  python3: $PYTHON_BIN" >&2
  echo "  binding_bench: $BINDING_BENCH_BIN" >&2
  echo "Run with ML4RG_ENV_BIN=/path/to/ml4rg/bin if your env lives elsewhere." >&2
  exit 1
fi
if [[ "${1:-}" == "--check" ]]; then
  echo "python3: $PYTHON_BIN"
  echo "binding_bench: $BINDING_BENCH_BIN"
  exit 0
fi

COMMON_ARGS=(
  --sites-path project15/working/binding_datasets/scer_pugh_rossi_dedup/scer_pugh_rossi_dedup.sites.parquet
  --regions-path project15/working/sequence_datasets/fungi_upstream_ATG_1000/_saccharomyces_cerevisiae_sequence_mapper.parquet
  --promoter-split-path project15/working/binding_datasets/scer_pugh_rossi_dedup/scer_pugh_rossi_dedup.promoter_split_random_80_10_10.json
  --tf-split-path project15/working/binding_datasets/scer_pugh_rossi_dedup/scer_pugh_rossi_dedup.tf_split_named_similarity_80_10_10.json
  --label-smoothing-mode hard-dilate
  --label-smoothing-radius-bp 10
  --epochs 50
  --eval-every 1
  --save-every 10
  --final-eval-scope test_only
)

run() {
  name="$1"
  shift
  log_path="project15/working/supervised_baseline/${name}.log"
  mkdir -p "$(dirname "$log_path")"
  : > "$log_path"

  echo "===== START $name $(date '+%Y-%m-%d %H:%M:%S') ====="
  "$@" > >(tee -a "$log_path") 2>&1 &
  command_pid=$!
  last_reported_epoch=""

  while kill -0 "$command_pid" 2>/dev/null; do
    sleep 60
    latest_epoch="$(
      awk '/^epoch=/{line=$0} END{if (line!="") {sub(/^epoch=0*/, "", line); sub(/ .*/, "", line); print line}}' "$log_path" 2>/dev/null || true
    )"
    if [[ -n "$latest_epoch" && "$latest_epoch" != "$last_reported_epoch" ]]; then
      echo "===== $name completed epoch $latest_epoch $(date '+%Y-%m-%d %H:%M:%S') ====="
      last_reported_epoch="$latest_epoch"
    fi
  done

  set +e
  wait "$command_pid"
  status=$?
  set -e

  latest_epoch="$(
    awk '/^epoch=/{line=$0} END{if (line!="") {sub(/^epoch=0*/, "", line); sub(/ .*/, "", line); print line}}' "$log_path" 2>/dev/null || true
  )"
  if [[ "$status" -ne 0 ]]; then
    echo "===== FAILED $name status=$status latest_epoch=${latest_epoch:-none} $(date '+%Y-%m-%d %H:%M:%S') ====="
    return "$status"
  fi
  echo "===== DONE $name latest_epoch=${latest_epoch:-none} $(date '+%Y-%m-%d %H:%M:%S') ====="
}

run dna_only_singlehead_bce \
  python3 -u supervised_baseline/train_promoter_dense.py \
  "${COMMON_ARGS[@]}" \
  --input-mode raw \
  --model dense_motif_dilated_attention_cnn \
  --label-mode merged_train_tfs \
  --loss bce \
  --output-dir project15/working/supervised_baseline/models/dna_only_raw_singlehead_named_hard_dilate21_bce

run dna_only_singlehead_focal \
  python3 -u supervised_baseline/train_promoter_dense.py \
  "${COMMON_ARGS[@]}" \
  --input-mode raw \
  --model dense_motif_dilated_attention_cnn \
  --label-mode merged_train_tfs \
  --loss focal \
  --output-dir project15/working/supervised_baseline/models/dna_only_raw_singlehead_named_hard_dilate21_focal

run dna_only_multitask_bce \
  python3 -u supervised_baseline/train_promoter_dense.py \
  "${COMMON_ARGS[@]}" \
  --input-mode raw \
  --model dense_motif_dilated_attention_multitask_cnn \
  --label-mode tf_and_merged_train_tfs \
  --loss bce \
  --output-dir project15/working/supervised_baseline/models/dna_only_raw_multitask_named_hard_dilate21_bce

run dna_only_multitask_focal \
  python3 -u supervised_baseline/train_promoter_dense.py \
  "${COMMON_ARGS[@]}" \
  --input-mode raw \
  --model dense_motif_dilated_attention_multitask_cnn \
  --label-mode tf_and_merged_train_tfs \
  --loss focal \
  --output-dir project15/working/supervised_baseline/models/dna_only_raw_multitask_named_hard_dilate21_focal





PROJECT=project15
DATASET=_saccharomyces_cerevisiae_sequence_mapper

SITES=$PROJECT/working/binding_datasets/scer_pugh_rossi_dedup/scer_pugh_rossi_dedup.sites.parquet
REGIONS=$PROJECT/working/sequence_datasets/fungi_upstream_ATG_1000/_saccharomyces_cerevisiae_sequence_mapper.parquet
FASTA=$PROJECT/working/sequence_datasets_fastas/_saccharomyces_cerevisiae_sequence_mapper.fasta

PROM_SPLIT=$PROJECT/working/binding_datasets/scer_pugh_rossi_dedup/scer_pugh_rossi_dedup.promoter_split_random_80_10_10.json
TF_SPLIT=$PROJECT/working/binding_datasets/scer_pugh_rossi_dedup/scer_pugh_rossi_dedup.tf_split_named_similarity_80_10_10.json

BB_IN=$PROJECT/working/binding_bench_inputs
TAG=scer_pugh_rossi_dedup_nmotifs150

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export MEME_BIN="$CONDA_PREFIX/bin"
fi


BB_IN=project15/working/binding_bench_inputs
BB_RUN=project15/working/binding_bench_runs
EVAL=project15/working/binding_bench_inputs/dna_only_heldout_eval

run_bb() {
  name="$1"
  pred="$2"
  ranks="$3"

  echo "===== Binding Bench train: $name ====="
  binding_bench discrete \
    -i "$pred" \
    -t train_merged_train_tfs \
    --sites-path "$EVAL/merged_train_tfs_sites.parquet" \
    --regions-path "$EVAL/train_promoter_regions.parquet" \
    -o "$BB_RUN/train_compare/$name" \
    --feature_rank_path "$ranks" \
    --metric precision recall f1 jaccard precision_lb recall_lb \
    --best_assignment \
    --ba_by_feature_rank \
    --overwrite

  echo "===== Binding Bench test: $name ====="
  binding_bench discrete \
    -i "$pred" \
    -t test_merged_train_tfs \
    --sites-path "$EVAL/merged_train_tfs_sites.parquet" \
    --regions-path "$EVAL/test_promoter_regions.parquet" \
    -o "$BB_RUN/test_compare/$name" \
    --feature_rank_path "$ranks" \
    --metric precision recall f1 jaccard precision_lb recall_lb \
    --best_assignment \
    --ba_by_feature_rank \
    --overwrite
}

python3 supervised_baseline/export_promoter_dense_predictions.py \
  --checkpoint project15/working/supervised_baseline/models/dna_only_raw_multitask_named_hard_dilate21_bce/best.pt \
  --predictions-out "$BB_IN/dna_only_raw_multitask_named_hard_dilate21_bce_predictions.parquet" \
  --feature-ranks-out "$BB_IN/dna_only_raw_multitask_named_hard_dilate21_bce_feature_ranks.parquet" \
  --score-mode logit \
  --top-k-per-tf 5000 \
  --nms-radius-bp 50

run_bb dna_only_raw_multitask_named_hard_dilate21_bce \
  "$BB_IN/dna_only_raw_multitask_named_hard_dilate21_bce_predictions.parquet" \
  "$BB_IN/dna_only_raw_multitask_named_hard_dilate21_bce_feature_ranks.parquet"

python3 supervised_baseline/export_promoter_dense_predictions.py \
  --checkpoint project15/working/supervised_baseline/models/dna_only_raw_multitask_named_hard_dilate21_focal/best.pt \
  --predictions-out "$BB_IN/dna_only_raw_multitask_named_hard_dilate21_focal_predictions.parquet" \
  --feature-ranks-out "$BB_IN/dna_only_raw_multitask_named_hard_dilate21_focal_feature_ranks.parquet" \
  --score-mode logit \
  --top-k-per-tf 5000 \
  --nms-radius-bp 50

run_bb dna_only_raw_multitask_named_hard_dilate21_focal \
  "$BB_IN/dna_only_raw_multitask_named_hard_dilate21_focal_predictions.parquet" \
  "$BB_IN/dna_only_raw_multitask_named_hard_dilate21_focal_feature_ranks.parquet"


python3 supervised_baseline/export_promoter_dense_predictions.py \
  --checkpoint project15/working/supervised_baseline/models/dna_only_raw_singlehead_named_hard_dilate21_bce/best.pt \
  --predictions-out "$BB_IN/dna_only_raw_singlehead_named_hard_dilate21_bce_predictions.parquet" \
  --feature-ranks-out "$BB_IN/dna_only_raw_singlehead_named_hard_dilate21_bce_feature_ranks.parquet" \
  --score-mode logit \
  --top-k-per-tf 5000 \
  --nms-radius-bp 50

run_bb dna_only_raw_singlehead_named_hard_dilate21_bce \
  "$BB_IN/dna_only_raw_singlehead_named_hard_dilate21_bce_predictions.parquet" \
  "$BB_IN/dna_only_raw_singlehead_named_hard_dilate21_bce_feature_ranks.parquet"

python3 supervised_baseline/export_promoter_dense_predictions.py \
  --checkpoint project15/working/supervised_baseline/models/dna_only_raw_singlehead_named_hard_dilate21_focal/best.pt \
  --predictions-out "$BB_IN/dna_only_raw_singlehead_named_hard_dilate21_focal_predictions.parquet" \
  --feature-ranks-out "$BB_IN/dna_only_raw_singlehead_named_hard_dilate21_focal_feature_ranks.parquet" \
  --score-mode logit \
  --top-k-per-tf 5000 \
  --nms-radius-bp 50


COMMON_TRAIN_ARGS=(
  --sites-path "$SITES"
  --regions-path "$REGIONS"
  --input-mode raw
  --label-mode merged_train_tfs
  --loss bce
  --label-smoothing-mode hard-dilate
  --label-smoothing-radius-bp 10
  --promoter-split-path "$PROM_SPLIT"
  --tf-split-path "$TF_SPLIT"
  --epochs 50
  --eval-every 1
  --save-every 10
  --final-eval-scope test_only
)

run small_cnn_named_hard_dilate21_bce_train \
  python3 -u supervised_baseline/train_promoter_dense.py \
  "${COMMON_TRAIN_ARGS[@]}" \
  --model dense_small_cnn \
  --hidden-channels 128 \
  --output-dir "$PROJECT/working/supervised_baseline/models/small_cnn_named_hard_dilate21_bce"

run res_dilated_cnn_named_hard_dilate21_bce_train \
  python3 -u supervised_baseline/train_promoter_dense.py \
  "${COMMON_TRAIN_ARGS[@]}" \
  --model dense_res_dilated_cnn \
  --hidden-channels 256 \
  --dilations 1,2,4,8,16 \
  --output-dir "$PROJECT/working/supervised_baseline/models/res_dilated_cnn_named_hard_dilate21_bce"

python3 supervised_baseline/export_promoter_dense_predictions.py \
  --checkpoint "$PROJECT/working/supervised_baseline/models/small_cnn_named_hard_dilate21_bce/best.pt" \
  --predictions-out "$BB_IN/small_cnn_named_hard_dilate21_bce_predictions.parquet" \
  --feature-ranks-out "$BB_IN/small_cnn_named_hard_dilate21_bce_feature_ranks.parquet" \
  --score-mode logit \
  --top-k-per-tf 5000 \
  --nms-radius-bp 50

python3 supervised_baseline/export_promoter_dense_predictions.py \
  --checkpoint "$PROJECT/working/supervised_baseline/models/res_dilated_cnn_named_hard_dilate21_bce/best.pt" \
  --predictions-out "$BB_IN/res_dilated_cnn_named_hard_dilate21_bce_predictions.parquet" \
  --feature-ranks-out "$BB_IN/res_dilated_cnn_named_hard_dilate21_bce_feature_ranks.parquet" \
  --score-mode logit \
  --top-k-per-tf 5000 \
  --nms-radius-bp 50

run_bb small_cnn_named_hard_dilate21_bce \
  "$BB_IN/small_cnn_named_hard_dilate21_bce_predictions.parquet" \
  "$BB_IN/small_cnn_named_hard_dilate21_bce_feature_ranks.parquet"

run_bb res_dilated_cnn_named_hard_dilate21_bce \
  "$BB_IN/res_dilated_cnn_named_hard_dilate21_bce_predictions.parquet" \
  "$BB_IN/res_dilated_cnn_named_hard_dilate21_bce_feature_ranks.parquet"

run_bb dna_only_raw_singlehead_named_hard_dilate21_focal \
  "$BB_IN/dna_only_raw_singlehead_named_hard_dilate21_focal_predictions.parquet" \
  "$BB_IN/dna_only_raw_singlehead_named_hard_dilate21_focal_feature_ranks.parquet"


# STREME-only, ohne FIMO
run_bb streme_sites_nmotifs_150 \
  "$BB_IN/streme_sites__saccharomyces_cerevisiae_sequence_mapper_nmotifs_150_predictions.parquet" \
  "$BB_IN/streme_sites__saccharomyces_cerevisiae_sequence_mapper_nmotifs_150_feature_ranks.parquet"

# JASPAR-FIMO
run_bb jaspar_fimo \
  "$BB_IN/fimo_jaspar__saccharomyces_cerevisiae_sequence_mapper_predictions.parquet" \
  "$BB_IN/fimo_jaspar__saccharomyces_cerevisiae_sequence_mapper_feature_ranks.parquet"

# Old full-data references. Keep these disabled for the fair named-split comparison.
# run_bb promoter_raw_small_cnn_old_reference \
#   "$BB_IN/promoter_raw_small_cnn_logit_nms50_merged_train_tfs_predictions.parquet" \
#   "$BB_IN/promoter_raw_small_cnn_logit_nms50_merged_train_tfs_feature_ranks.parquet"
#
# run_bb promoter_raw_res_dilated_cnn_old_reference \
#   "$BB_IN/promoter_raw_res_dilated_cnn_logit_nms50_merged_train_tfs_predictions.parquet" \
#   "$BB_IN/promoter_raw_res_dilated_cnn_logit_nms50_merged_train_tfs_feature_ranks.parquet"
