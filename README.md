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
TF-by-position score map. The default sequence mapper is
`fungi_upstream_ATG_1000`, whose `seq` values are 1003 bp because they include
the terminal ATG. Dense promoter training trims that terminal ATG by default.
The raw baseline uses one-hot DNA sequence; the embedding baseline expects
precomputed per-position promoter embeddings in row-aligned `.npy` `[N, L, D]`
format or a parquet column containing `[L, D]` arrays.

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

For held-out promoter evaluation, train with a promoter split. Random splits are
useful for smoke testing; chromosome splits are the cleaner baseline for
unseen genomic regions. Checkpoints are selected by validation loss when a
validation split is present, and `final_metrics.json` reports train/val/test
dense metrics.

```bash
INPUT_MODE=embedding \
EMBEDDINGS_PATH=/path/to/promoter_position_embeddings.npy \
MODEL=dense_res_dilated_cnn \
PROMOTER_SPLIT_MODE=random \
TRAIN_FRACTION=0.8 \
VAL_FRACTION=0.1 \
OUTPUT_DIR=/s/project/ml4rg_students/2026/project15/working/supervised_baseline/models/promoter_specieslm_l11_res_dilated_cnn_random_promoter_split \
sbatch slurm/train_promoter_dense.sbatch

INPUT_MODE=raw \
MODEL=dense_res_dilated_cnn \
PROMOTER_SPLIT_MODE=chromosome \
VAL_CHROMS=VII \
TEST_CHROMS=VIII,IX \
OUTPUT_DIR=/s/project/ml4rg_students/2026/project15/working/supervised_baseline/models/promoter_raw_res_dilated_cnn_chr_promoter_split \
sbatch slurm/train_promoter_dense.sbatch
```

Train a first protein-conditioned dense model. This uses the same promoter
holdout infrastructure and additionally creates a random TF train/val/test
split. Training loss is computed only on train TFs; validation is computed on
validation promoters crossed with validation TFs.

```bash
INPUT_MODE=embedding \
EMBEDDINGS_PATH=/s/project/ml4rg_students/2026/project15/raw/embeddings/other_models/specieslm_v1/_saccharomyces_cerevisiae/embs_ds_fungi_upstream_ATG_1000_l11.npy \
TF_EMBEDDINGS_PATH=/s/project/ml4rg_students/2026/project15/working/protein_embeddings/scer_esm2_tf_emb.parquet \
TF_EMBEDDING_KEY_COLUMN=gene \
DROP_MISSING_TF_EMBEDDINGS=true \
MODEL=dense_protein_res_dilated_cnn \
PROMOTER_SPLIT_MODE=chromosome \
VAL_CHROMS=VII \
TEST_CHROMS=VIII,IX \
TF_SPLIT_MODE=random \
TF_TRAIN_FRACTION=0.7 \
TF_VAL_FRACTION=0.15 \
OUTPUT_DIR=/s/project/ml4rg_students/2026/project15/working/supervised_baseline/models/promoter_specieslm_l11_dense_protein_res_dilated_cnn_chr_tf_random \
sbatch slurm/train_promoter_dense.sbatch
```

For a TransBind-style variant that keeps the local ResDilated peak scorer as
the main path and adds protein-query cross-attention as a residual context
branch, use:

```bash
INPUT_MODE=embedding \
EMBEDDINGS_PATH=/s/project/ml4rg_students/2026/project15/raw/embeddings/other_models/specieslm_v1/_saccharomyces_cerevisiae/embs_ds_fungi_upstream_ATG_1000_l11.npy \
TF_EMBEDDINGS_PATH=/s/project/ml4rg_students/2026/project15/working/protein_embeddings/scer_esm2_tf_emb.parquet \
TF_EMBEDDING_KEY_COLUMN=gene \
DROP_MISSING_TF_EMBEDDINGS=true \
MODEL=dense_protein_res_dilated_crossattention \
PROMOTER_SPLIT_MODE=chromosome \
VAL_CHROMS=VII \
TEST_CHROMS=VIII,IX \
TF_SPLIT_MODE=similarity \
TF_SIMILARITY_THRESHOLD=0.9 \
SELECTION_METRIC=val_average_precision \
OUTPUT_DIR=/s/project/ml4rg_students/2026/project15/working/supervised_baseline/models/promoter_embedding_resdilated_crossattention_similarity_ap \
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
