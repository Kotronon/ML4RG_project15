#!/usr/bin/env bash
#SBATCH --job-name=ab-summary
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --partition=student_project
#SBATCH --output=slurm/logs/ablation-summary-%j.out
#SBATCH --error=slurm/logs/ablation-summary-%j.err

set -euo pipefail

PROJECT="${ML4RG_PROJECT:-/s/project/ml4rg_students/2026/project15}"
PYTHON_BIN="${PYTHON_BIN:-/opt/modules/i12g/anaconda/envs/ml4rg_project15/bin/python}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "${REPO_DIR:-}" ]]; then
    if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/baseline_evaluations/append_binding_bench_summary.py" ]]; then
        REPO_DIR="${SLURM_SUBMIT_DIR}"
    else
        REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
    fi
fi
APPEND_SCRIPT="${REPO_DIR}/baseline_evaluations/append_binding_bench_summary.py"
SUMMARY_TSV="${SUMMARY_TSV:-${REPO_DIR}/baseline_evaluations/all_baselines_comparison.tsv}"

CONDA_PREFIX="$(${PYTHON_BIN} -c 'import sys; print(sys.prefix)')"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

append_one() {
    local method_name="$1"
    local display_name="$2"
    "${PYTHON_BIN}" "${APPEND_SCRIPT}" \
        --summary-tsv "${SUMMARY_TSV}" \
        --run-dir "${PROJECT}/working/binding_bench_runs/supervised_${method_name}_logit_nms50" \
        --method "${display_name}" \
        --replace-existing
}

append_one "transbind_ab_base200" "TransBind ablation: base200"
append_one "transbind_ab_hidden256" "TransBind ablation: hidden256"
append_one "transbind_ab_long_dil" "TransBind ablation: long dilations"
append_one "transbind_ab_no_pool" "TransBind ablation: no maxpool"
append_one "transbind_ab_tf_bias" "TransBind ablation: TF bias"

echo "Appended TransBind ablation summaries to: ${SUMMARY_TSV}"
