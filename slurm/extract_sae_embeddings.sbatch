#!/usr/bin/env bash
#SBATCH --job-name=sae-embed
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --partition=student_project
#SBATCH --output=slurm/logs/sae-embed-%j.out
#SBATCH --error=slurm/logs/sae-embed-%j.err

set -euo pipefail

PROJECT="${ML4RG_PROJECT:-/s/project/ml4rg_students/2026/project15}"
PYTHON_BIN="${PYTHON_BIN:-/opt/modules/i12g/anaconda/envs/ml4rg_project15/bin/python}"
REPO_DIR="${REPO_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"

DATA_ROOT="${DATA_ROOT:-/s/project/multispecies/fungi_code/tf_sae/processed/sequence_datasets/fungi_upstream_ATG_1000}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT}/working/sae/embeddings}"

# Which model to extract with:  evo2_7b | evo2_40b | specieslm
MODEL_NAME="${MODEL_NAME:-evo2_7b}"

# Which species to embed (space-separated list of parquet stems, without _sequence_mapper.parquet)
SPECIES="${SPECIES:-_saccharomyces_cerevisiae _schizosaccharomyces_pombe}"

BATCH_SIZE="${BATCH_SIZE:-4}"
LAYER="${LAYER:--1}"
DEVICE="${DEVICE:-auto}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

CONDA_PREFIX="$(${PYTHON_BIN} -c 'import sys; print(sys.prefix)')"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

mkdir -p "${OUTPUT_ROOT}" "${REPO_DIR}/slurm/logs"

echo "Model:       ${MODEL_NAME}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Species:     ${SPECIES}"

for SPECIES_STEM in ${SPECIES}; do
    REGIONS_PATH="${DATA_ROOT}/${SPECIES_STEM}_sequence_mapper.parquet"
    OUTPUT_PATH="${OUTPUT_ROOT}/${SPECIES_STEM}_${MODEL_NAME}.parquet"

    if [[ ! -f "${REGIONS_PATH}" ]]; then
        echo "WARNING: regions not found at ${REGIONS_PATH}, skipping." >&2
        continue
    fi

    echo "--- Embedding ${SPECIES_STEM} with ${MODEL_NAME} ---"
    "${PYTHON_BIN}" -m sae.extract_embeddings \
        --regions-path "${REGIONS_PATH}" \
        --model-name "${MODEL_NAME}" \
        --output-path "${OUTPUT_PATH}" \
        --batch-size "${BATCH_SIZE}" \
        --layer "${LAYER}" \
        --device "${DEVICE}" \
        ${EXTRA_ARGS}
done

echo "Embedding extraction finished."
