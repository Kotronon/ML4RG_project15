# ML4RG_project15

This Repository is there to share code and store all imortant results.

## Protein-conditioned supervised baseline

Generate yeast protein embeddings from SGD `orf_trans_all.fasta` with ESM-2:

Requires `transformers` and `biopython` in the Python environment.

```bash
/opt/modules/i12g/anaconda/envs/ml4rg_project15/bin/python \
  supervised_baseline/embed_proteome_esm2.py \
  /path/to/orf_trans_all.fasta \
  --output /s/project/ml4rg_students/2026/project15/working/protein_embeddings/scer_esm2_emb.parquet \
  --device cuda
```

Train the protein-conditioned model:

```bash
MODEL=transbind_lite \
TF_EMBEDDINGS_PATH=/s/project/ml4rg_students/2026/project15/working/protein_embeddings/scer_esm2_emb.parquet \
TF_EMBEDDING_KEY_COLUMN=gene \
OUTPUT_DIR=/s/project/ml4rg_students/2026/project15/working/supervised_baseline/models/transbind_lite_esm2 \
sbatch slurm/train_supervised.sbatch
```

If the Binding Bench TF labels do not match the embedding key column exactly,
pass `TF_NAME_MAP=/path/to/tf_name_map.json`, where the JSON maps dataset labels
to embedding-table keys.

## Promoter-level dense baselines

These baselines train directly on 1000 bp promoters and predict a dense
TF-by-position score map. The raw baseline uses one-hot DNA sequence; the
embedding baseline expects precomputed per-position promoter embeddings in
row-aligned `.npy` `[N, L, D]` format or a parquet column containing `[L, D]`
arrays.

Submit all four baseline runs:

```bash
EMBEDDINGS_PATH=/path/to/promoter_position_embeddings.npy \
bash slurm/submit_promoter_dense_baselines.sh
```

Or submit individual training jobs:

```bash
INPUT_MODE=raw \
MODEL=dense_small_cnn \
sbatch slurm/train_promoter_dense.sbatch

INPUT_MODE=embedding \
EMBEDDINGS_PATH=/path/to/promoter_position_embeddings.npy \
MODEL=dense_res_dilated_cnn \
sbatch slurm/train_promoter_dense.sbatch
```

Evaluate a trained dense promoter checkpoint with Binding Bench:

```bash
INPUT_MODE=raw \
MODEL=dense_small_cnn \
MODEL_DIR=/s/project/ml4rg_students/2026/project15/working/supervised_baseline/models/promoter_raw_dense_small_cnn \
sbatch slurm/run_binding_bench_promoter_dense.sbatch
```

By default the dense export uses `NMS_RADIUS_BP=0`, so the first report is a
clean promoter-position baseline. Set `NMS_RADIUS_BP=50` to compare against the
older peak-collapsed export style.
