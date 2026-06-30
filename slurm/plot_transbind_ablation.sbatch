#!/usr/bin/env bash
#SBATCH --job-name=ab-plots
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --partition=student_project
#SBATCH --output=slurm/logs/ablation-plots-%j.out
#SBATCH --error=slurm/logs/ablation-plots-%j.err

set -euo pipefail

PROJECT="${ML4RG_PROJECT:-/s/project/ml4rg_students/2026/project15}"
PYTHON_BIN="${PYTHON_BIN:-/opt/modules/i12g/anaconda/envs/ml4rg_project15/bin/python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${REPO_DIR:-}" ]]; then
    if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/baseline_evaluations/compare_binding_bench_runs.py" ]]; then
        REPO_DIR="${SLURM_SUBMIT_DIR}"
    else
        REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
    fi
fi
COMPARE_SCRIPT="${REPO_DIR}/baseline_evaluations/compare_binding_bench_runs.py"
BINDING_BENCH_REPO="${BINDING_BENCH_REPO:-${REPO_DIR}/binding_bench}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_DIR}/baseline_evaluations/plots}"

CONDA_PREFIX="$(${PYTHON_BIN} -c 'import sys; print(sys.prefix)')"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

"${PYTHON_BIN}" "${COMPARE_SCRIPT}" \
    --binding-bench-repo "${BINDING_BENCH_REPO}" \
    --run "Base200=${PROJECT}/working/binding_bench_runs/supervised_transbind_ab_base200_logit_nms50" \
    --run "Hidden256=${PROJECT}/working/binding_bench_runs/supervised_transbind_ab_hidden256_logit_nms50" \
    --run "Long dilations=${PROJECT}/working/binding_bench_runs/supervised_transbind_ab_long_dil_logit_nms50" \
    --run "No maxpool=${PROJECT}/working/binding_bench_runs/supervised_transbind_ab_no_pool_logit_nms50" \
    --run "TF bias=${PROJECT}/working/binding_bench_runs/supervised_transbind_ab_tf_bias_logit_nms50" \
    --run "Full v2=${PROJECT}/working/binding_bench_runs/supervised_transbind_lite_v2_logit_nms50" \
    --score jaccard precision_lb recall_lb \
    --output-dir "${OUTPUT_DIR}" \
    --prefix transbind_ablation

echo "Wrote TransBind ablation plots to: ${OUTPUT_DIR}"
