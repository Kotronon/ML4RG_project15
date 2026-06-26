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
