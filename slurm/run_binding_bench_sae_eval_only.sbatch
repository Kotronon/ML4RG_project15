#!/usr/bin/env bash
#SBATCH --job-name=bb-sae-eval
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --partition=student_project
#SBATCH --output=slurm/logs/bb-sae-eval-%j.out
#SBATCH --error=slurm/logs/bb-sae-eval-%j.err

set -euo pipefail

PROJECT="/s/project/ml4rg_students/2026/project15"
PYTHON_BIN="/opt/modules/i12g/anaconda/envs/ml4rg_project15/bin/python"
BINDING_BENCH_REPO="${PROJECT}/binding_bench"

CONDA_PREFIX="$("${PYTHON_BIN}" -c 'import sys; print(sys.prefix)')"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export PYTHONPATH="${BINDING_BENCH_REPO}/src:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

"${PYTHON_BIN}" -c 'from binding_bench.cli import main; main()' discrete \
    --input "${PROJECT}/working/binding_bench_inputs/sae_topk_nms50_predictions.parquet" \
    --dataset DNA_rossi_chipexo \
    --output_dir "${PROJECT}/working/binding_bench_runs/sae_topk_nms50" \
    --metric jaccard precision_lb recall_lb \
    --best_assignment \
    --ba_by_feature_rank \
    --feature_rank_path "${PROJECT}/working/binding_bench_inputs/sae_topk_nms50_feature_ranks.parquet" \
    --feature_rank_name sae_activation \
    --display_name "SAE TopK (nms=50)" \
    --overwrite

echo "Binding Bench eval finished."
