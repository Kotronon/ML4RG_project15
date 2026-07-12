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

FAILED_STEPS=()
SKIPPED_STEPS=()

record_failure() {
  FAILED_STEPS+=("$1")
  echo "===== RECORDED FAILURE: $1 ====="
}

record_skip() {
  SKIPPED_STEPS+=("$1")
  echo "===== SKIP: $1 ====="
}

require_files() {
  label="$1"
  shift
  missing=()
  for path in "$@"; do
    if [[ ! -s "$path" ]]; then
      missing+=("$path")
    fi
  done
  if (( ${#missing[@]} > 0 )); then
    record_skip "$label missing: ${missing[*]}"
    return 1
  fi
  return 0
}

bb_success_exists() {
  conf_path="$1"
  if [[ ! -s "$conf_path" ]]; then
    return 1
  fi
  python3 - "$conf_path" <<'PY'
import sys
import yaml

try:
    with open(sys.argv[1]) as handle:
        payload = yaml.safe_load(handle) or {}
except Exception:
    raise SystemExit(1)
raise SystemExit(0 if payload.get("status") == "SUCCESS" else 1)
PY
}

print_summary() {
  echo "===== QUEUE SUMMARY $(date '+%Y-%m-%d %H:%M:%S') ====="
  if (( ${#FAILED_STEPS[@]} == 0 )); then
    echo "Failures: none"
  else
    echo "Failures:"
    printf '  - %s\n' "${FAILED_STEPS[@]}"
  fi
  if (( ${#SKIPPED_STEPS[@]} == 0 )); then
    echo "Skipped: none"
  else
    echo "Skipped:"
    printf '  - %s\n' "${SKIPPED_STEPS[@]}"
  fi
}

trap print_summary EXIT

COMMON_ARGS=(
  --sites-path project15/working/binding_datasets/scer_pugh_rossi_dedup/scer_pugh_rossi_dedup.sites.parquet
  --regions-path project15/working/sequence_datasets/fungi_upstream_ATG_1000/_saccharomyces_cerevisiae_sequence_mapper.parquet
  --promoter-split-path project15/working/binding_datasets/scer_pugh_rossi_dedup/scer_pugh_rossi_dedup.promoter_split_random_80_10_10.json
  --tf-split-path project15/working/binding_datasets/scer_pugh_rossi_dedup/scer_pugh_rossi_dedup.tf_split_named_similarity_80_10_10.json
  --label-smoothing-mode hard-dilate
  --label-smoothing-radius-bp 10
  --epochs 50
  --eval-every 1
  --selection-metric val_dilated_average_precision
  --save-every 10
  --final-eval-scope test_only
)

run() {
  name="$1"
  shift
  command_args=("$@")
  log_path="project15/working/supervised_baseline/${name}.log"
  mkdir -p "$(dirname "$log_path")"

  output_dir=""
  expected_epochs=""
  for ((i = 0; i < ${#command_args[@]}; i++)); do
    if [[ "${command_args[$i]}" == "--output-dir" && $((i + 1)) -lt ${#command_args[@]} ]]; then
      output_dir="${command_args[$((i + 1))]}"
    fi
    if [[ "${command_args[$i]}" == "--epochs" && $((i + 1)) -lt ${#command_args[@]} ]]; then
      expected_epochs="${command_args[$((i + 1))]}"
    fi
  done
  if [[ "${command_args[*]}" == *"train_promoter_dense.py"* && -n "$output_dir" ]]; then
    expected_epochs="${expected_epochs:-50}"
    if [[ -f "$output_dir/best.pt" && -f "$output_dir/history.json" ]]; then
      if python3 - "$output_dir/history.json" "$expected_epochs" <<'PY'
import json
import sys
from pathlib import Path

history = json.loads(Path(sys.argv[1]).read_text())
expected = int(sys.argv[2])
last_epoch = int(history[-1].get("epoch", 0)) if history else 0
raise SystemExit(0 if last_epoch >= expected else 1)
PY
      then
        echo "===== SKIP $name existing completed training in $output_dir ====="
        return 0
      fi
    fi
  fi

  : > "$log_path"

  echo "===== START $name $(date '+%Y-%m-%d %H:%M:%S') ====="
  "${command_args[@]}" > >(tee -a "$log_path") 2>&1 &
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
    record_failure "$name status=$status latest_epoch=${latest_epoch:-none}"
    return 0
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

if ! python3 baseline_evaluations/prepare_split_binding_bench_inputs.py \
  --sites-path "$SITES" \
  --regions-path "$REGIONS" \
  --promoter-split-path "$PROM_SPLIT" \
  --tf-split-path "$TF_SPLIT" \
  --output-dir "$EVAL"; then
  record_failure "prepare_split_binding_bench_inputs"
fi

run_export() {
  name="$1"
  checkpoint="$2"
  pred="$3"
  ranks="$4"

  require_files "$name export" "$checkpoint" || return 0
  if [[ -s "$pred" && -s "$ranks" && "$pred" -nt "$checkpoint" && "$ranks" -nt "$checkpoint" ]]; then
    record_skip "$name export already up to date"
    return 0
  fi
  echo "===== Export predictions: $name ====="
  if ! python3 supervised_baseline/export_promoter_dense_predictions.py \
    --checkpoint "$checkpoint" \
    --predictions-out "$pred" \
    --feature-ranks-out "$ranks" \
    --score-mode logit \
    --top-k-per-tf 5000 \
    --nms-radius-bp 50; then
    record_failure "$name export"
  fi
}

run_bb() {
  name="$1"
  pred="$2"
  ranks="$3"
  feature_filter="${4:-}"

  echo "===== Binding Bench train: $name ====="
  train_conf="$BB_RUN/train_compare/$name/binding_bench/discrete/train_merged_train_tfs/benchmark_conf.yaml"
  if bb_success_exists "$train_conf"; then
    record_skip "$name binding_bench train already SUCCESS"
  else
  if require_files "$name binding_bench train" \
    "$pred" \
    "$ranks" \
    "$EVAL/merged_train_tfs_sites.parquet" \
    "$EVAL/train_promoter_regions.parquet" \
    ${feature_filter:+"$feature_filter"}; then
  if ! binding_bench discrete \
    -i "$pred" \
    -t train_merged_train_tfs \
    --sites-path "$EVAL/merged_train_tfs_sites.parquet" \
    --regions-path "$EVAL/train_promoter_regions.parquet" \
    ${feature_filter:+--feature-name-filter-path "$feature_filter"} \
    -o "$BB_RUN/train_compare/$name" \
    --feature_rank_path "$ranks" \
    --metric precision recall f1 jaccard precision_lb recall_lb \
    --best_assignment \
    --ba_by_feature_rank \
    --overwrite; then
    record_failure "$name binding_bench train"
  fi
  fi
  fi

  echo "===== Binding Bench test: $name ====="
  test_conf="$BB_RUN/test_compare/$name/binding_bench/discrete/test_merged_train_tfs/benchmark_conf.yaml"
  if bb_success_exists "$test_conf"; then
    record_skip "$name binding_bench test already SUCCESS"
  else
  if require_files "$name binding_bench test" \
    "$pred" \
    "$ranks" \
    "$EVAL/merged_train_tfs_sites.parquet" \
    "$EVAL/test_promoter_regions.parquet" \
    ${feature_filter:+"$feature_filter"}; then
  if ! binding_bench discrete \
    -i "$pred" \
    -t test_merged_train_tfs \
    --sites-path "$EVAL/merged_train_tfs_sites.parquet" \
    --regions-path "$EVAL/test_promoter_regions.parquet" \
    ${feature_filter:+--feature-name-filter-path "$feature_filter"} \
    -o "$BB_RUN/test_compare/$name" \
    --feature_rank_path "$ranks" \
    --metric precision recall f1 jaccard precision_lb recall_lb \
    --best_assignment \
    --ba_by_feature_rank \
    --overwrite; then
    record_failure "$name binding_bench test"
  fi
  fi
  fi
}

run_bb_test_tfs() {
  name="$1"
  pred="$2"
  ranks="$3"

  echo "===== Binding Bench test promoters + test TFs: $name ====="
  conf="$BB_RUN/test_compare_test_tfs/$name/binding_bench/discrete/test_promoters_test_tfs/benchmark_conf.yaml"
  if bb_success_exists "$conf"; then
    record_skip "$name binding_bench test_tfs already SUCCESS"
    return 0
  fi
  if ! require_files "$name binding_bench test_tfs" \
    "$pred" \
    "$ranks" \
    "$EVAL/test_tfs_sites.parquet" \
    "$EVAL/test_promoter_regions.parquet" \
    "$EVAL/test_tfs_feature_filter.txt"; then
    return 0
  fi
  if ! binding_bench discrete \
    -i "$pred" \
    -t test_promoters_test_tfs \
    --sites-path "$EVAL/test_tfs_sites.parquet" \
    --regions-path "$EVAL/test_promoter_regions.parquet" \
    --feature-name-filter-path "$EVAL/test_tfs_feature_filter.txt" \
    --target-name-filter-path "$EVAL/test_tfs_feature_filter.txt" \
    -o "$BB_RUN/test_compare_test_tfs/$name" \
    --feature_rank_path "$ranks" \
    --metric precision recall f1 jaccard precision_lb recall_lb \
    --best_assignment \
    --ba_by_feature_rank \
    --overwrite; then
    record_failure "$name binding_bench test_tfs"
  fi
}

run_bb_merged_test_tfs() {
  name="$1"
  pred="$2"
  ranks="$3"

  echo "===== Binding Bench test promoters + merged test TFs: $name ====="
  conf="$BB_RUN/test_compare_merged_test_tfs/$name/binding_bench/discrete/test_merged_test_tfs/benchmark_conf.yaml"
  if bb_success_exists "$conf"; then
    record_skip "$name binding_bench merged_test_tfs already SUCCESS"
    return 0
  fi
  if ! require_files "$name binding_bench merged_test_tfs" \
    "$pred" \
    "$ranks" \
    "$EVAL/merged_test_tfs_sites.parquet" \
    "$EVAL/test_promoter_regions.parquet"; then
    return 0
  fi
  if ! binding_bench discrete \
    -i "$pred" \
    -t test_merged_test_tfs \
    --sites-path "$EVAL/merged_test_tfs_sites.parquet" \
    --regions-path "$EVAL/test_promoter_regions.parquet" \
    -o "$BB_RUN/test_compare_merged_test_tfs/$name" \
    --feature_rank_path "$ranks" \
    --metric precision recall f1 jaccard precision_lb recall_lb \
    --best_assignment \
    --ba_by_feature_rank \
    --overwrite; then
    record_failure "$name binding_bench merged_test_tfs"
  fi
}

run_export dna_only_raw_multitask_named_hard_dilate21_bce \
  project15/working/supervised_baseline/models/dna_only_raw_multitask_named_hard_dilate21_bce/best.pt \
  "$BB_IN/dna_only_raw_multitask_named_hard_dilate21_bce_predictions.parquet" \
  "$BB_IN/dna_only_raw_multitask_named_hard_dilate21_bce_feature_ranks.parquet"

run_bb dna_only_raw_multitask_named_hard_dilate21_bce \
  "$BB_IN/dna_only_raw_multitask_named_hard_dilate21_bce_predictions.parquet" \
  "$BB_IN/dna_only_raw_multitask_named_hard_dilate21_bce_feature_ranks.parquet" \
  "$EVAL/merged_train_tfs_feature_filter.txt"

run_bb_test_tfs dna_only_raw_multitask_named_hard_dilate21_bce \
  "$BB_IN/dna_only_raw_multitask_named_hard_dilate21_bce_predictions.parquet" \
  "$BB_IN/dna_only_raw_multitask_named_hard_dilate21_bce_feature_ranks.parquet"

run_bb_merged_test_tfs dna_only_raw_multitask_named_hard_dilate21_bce \
  "$BB_IN/dna_only_raw_multitask_named_hard_dilate21_bce_predictions.parquet" \
  "$BB_IN/dna_only_raw_multitask_named_hard_dilate21_bce_feature_ranks.parquet"

run_export dna_only_raw_multitask_named_hard_dilate21_focal \
  project15/working/supervised_baseline/models/dna_only_raw_multitask_named_hard_dilate21_focal/best.pt \
  "$BB_IN/dna_only_raw_multitask_named_hard_dilate21_focal_predictions.parquet" \
  "$BB_IN/dna_only_raw_multitask_named_hard_dilate21_focal_feature_ranks.parquet"

run_bb dna_only_raw_multitask_named_hard_dilate21_focal \
  "$BB_IN/dna_only_raw_multitask_named_hard_dilate21_focal_predictions.parquet" \
  "$BB_IN/dna_only_raw_multitask_named_hard_dilate21_focal_feature_ranks.parquet" \
  "$EVAL/merged_train_tfs_feature_filter.txt"

run_bb_test_tfs dna_only_raw_multitask_named_hard_dilate21_focal \
  "$BB_IN/dna_only_raw_multitask_named_hard_dilate21_focal_predictions.parquet" \
  "$BB_IN/dna_only_raw_multitask_named_hard_dilate21_focal_feature_ranks.parquet"

run_bb_merged_test_tfs dna_only_raw_multitask_named_hard_dilate21_focal \
  "$BB_IN/dna_only_raw_multitask_named_hard_dilate21_focal_predictions.parquet" \
  "$BB_IN/dna_only_raw_multitask_named_hard_dilate21_focal_feature_ranks.parquet"


run_export dna_only_raw_singlehead_named_hard_dilate21_bce \
  project15/working/supervised_baseline/models/dna_only_raw_singlehead_named_hard_dilate21_bce/best.pt \
  "$BB_IN/dna_only_raw_singlehead_named_hard_dilate21_bce_predictions.parquet" \
  "$BB_IN/dna_only_raw_singlehead_named_hard_dilate21_bce_feature_ranks.parquet"

run_bb dna_only_raw_singlehead_named_hard_dilate21_bce \
  "$BB_IN/dna_only_raw_singlehead_named_hard_dilate21_bce_predictions.parquet" \
  "$BB_IN/dna_only_raw_singlehead_named_hard_dilate21_bce_feature_ranks.parquet" \
  "$EVAL/merged_train_tfs_feature_filter.txt"

run_bb_merged_test_tfs dna_only_raw_singlehead_named_hard_dilate21_bce \
  "$BB_IN/dna_only_raw_singlehead_named_hard_dilate21_bce_predictions.parquet" \
  "$BB_IN/dna_only_raw_singlehead_named_hard_dilate21_bce_feature_ranks.parquet"

run_export dna_only_raw_singlehead_named_hard_dilate21_focal \
  project15/working/supervised_baseline/models/dna_only_raw_singlehead_named_hard_dilate21_focal/best.pt \
  "$BB_IN/dna_only_raw_singlehead_named_hard_dilate21_focal_predictions.parquet" \
  "$BB_IN/dna_only_raw_singlehead_named_hard_dilate21_focal_feature_ranks.parquet"


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

run_export small_cnn_named_hard_dilate21_bce \
  "$PROJECT/working/supervised_baseline/models/small_cnn_named_hard_dilate21_bce/best.pt" \
  "$BB_IN/small_cnn_named_hard_dilate21_bce_predictions.parquet" \
  "$BB_IN/small_cnn_named_hard_dilate21_bce_feature_ranks.parquet"

run_export res_dilated_cnn_named_hard_dilate21_bce \
  "$PROJECT/working/supervised_baseline/models/res_dilated_cnn_named_hard_dilate21_bce/best.pt" \
  "$BB_IN/res_dilated_cnn_named_hard_dilate21_bce_predictions.parquet" \
  "$BB_IN/res_dilated_cnn_named_hard_dilate21_bce_feature_ranks.parquet"

run_bb small_cnn_named_hard_dilate21_bce \
  "$BB_IN/small_cnn_named_hard_dilate21_bce_predictions.parquet" \
  "$BB_IN/small_cnn_named_hard_dilate21_bce_feature_ranks.parquet" \
  "$EVAL/merged_train_tfs_feature_filter.txt"

run_bb_merged_test_tfs small_cnn_named_hard_dilate21_bce \
  "$BB_IN/small_cnn_named_hard_dilate21_bce_predictions.parquet" \
  "$BB_IN/small_cnn_named_hard_dilate21_bce_feature_ranks.parquet"

run_bb res_dilated_cnn_named_hard_dilate21_bce \
  "$BB_IN/res_dilated_cnn_named_hard_dilate21_bce_predictions.parquet" \
  "$BB_IN/res_dilated_cnn_named_hard_dilate21_bce_feature_ranks.parquet" \
  "$EVAL/merged_train_tfs_feature_filter.txt"

run_bb_merged_test_tfs res_dilated_cnn_named_hard_dilate21_bce \
  "$BB_IN/res_dilated_cnn_named_hard_dilate21_bce_predictions.parquet" \
  "$BB_IN/res_dilated_cnn_named_hard_dilate21_bce_feature_ranks.parquet"

run_bb dna_only_raw_singlehead_named_hard_dilate21_focal \
  "$BB_IN/dna_only_raw_singlehead_named_hard_dilate21_focal_predictions.parquet" \
  "$BB_IN/dna_only_raw_singlehead_named_hard_dilate21_focal_feature_ranks.parquet" \
  "$EVAL/merged_train_tfs_feature_filter.txt"

run_bb_merged_test_tfs dna_only_raw_singlehead_named_hard_dilate21_focal \
  "$BB_IN/dna_only_raw_singlehead_named_hard_dilate21_focal_predictions.parquet" \
  "$BB_IN/dna_only_raw_singlehead_named_hard_dilate21_focal_feature_ranks.parquet"


# STREME-only, ohne FIMO
run_bb streme_sites_nmotifs_150 \
  "$BB_IN/streme_sites__saccharomyces_cerevisiae_sequence_mapper_nmotifs_150_predictions.parquet" \
  "$BB_IN/streme_sites__saccharomyces_cerevisiae_sequence_mapper_nmotifs_150_feature_ranks.parquet"

run_bb_merged_test_tfs streme_sites_nmotifs_150 \
  "$BB_IN/streme_sites__saccharomyces_cerevisiae_sequence_mapper_nmotifs_150_predictions.parquet" \
  "$BB_IN/streme_sites__saccharomyces_cerevisiae_sequence_mapper_nmotifs_150_feature_ranks.parquet"

# JASPAR-FIMO
run_bb jaspar_fimo \
  "$BB_IN/fimo_jaspar__saccharomyces_cerevisiae_sequence_mapper_predictions.parquet" \
  "$BB_IN/fimo_jaspar__saccharomyces_cerevisiae_sequence_mapper_feature_ranks.parquet"

run_bb_merged_test_tfs jaspar_fimo \
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
