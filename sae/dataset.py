"""Datasets for SAE training and evaluation.

NpyPositionDataset:
    Loads position-wise embeddings from a .npy file of shape [N_genes, L, d].
    Each __getitem__ returns one (gene, position) embedding vector.
    Gene ordering must match the corresponding sequence_mapper parquet.

NpyPositionWithLabelsDataset:
    Same, but also resolves each (gene_idx, pos_idx) to a genomic coordinate
    and checks whether it overlaps any TF binding site. Used in evaluate.py.

EmbeddingDataset / EmbeddingWithLabelsDataset (legacy):
    Gene-level parquet-based datasets from the original parquet extraction path.
    Kept for Evo2 / other mean-pooled embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

# ── default paths ────────────────────────────────────────────────────────────

DEFAULT_NPY_PATH = Path(
    "/s/project/ml4rg_students/2026/project15/raw/embeddings/"
    "other_models/specieslm_v1/_saccharomyces_cerevisiae/"
    "embs_ds_fungi_upstream_ATG_1000_l10.npy"
)
DEFAULT_SITES_PATH = Path(
    "/s/project/multispecies/fungi_code/tf_sae/"
    "binding_bench_datasets/val/dna/_saccharomyces_cerevisiae/"
    "DNA_rossi_chipexo_sites.parquet"
)
DEFAULT_REGIONS_PATH = Path(
    "/s/project/multispecies/fungi_code/tf_sae/processed/"
    "sequence_datasets/fungi_upstream_ATG_1000/"
    "_saccharomyces_cerevisiae_sequence_mapper.parquet"
)

# SpeciesLM tokenises 1000bp sequences as:
#   [CLS] tok_0 tok_1 ... tok_999 [SEP] [PAD?]  → 1003 positions
# Positions 1..1000 correspond to sequence bases 0..999.
SPECIESLM_SEQ_OFFSET = 1   # index of first real-sequence token in the L dim
SPECIESLM_SEQ_LEN    = 1000


# ── position index helpers ────────────────────────────────────────────────────

class PosIndex(NamedTuple):
    gene_idx: int
    pos_idx: int   # index into the L dimension of the npy array


# ── SAE training dataset (unsupervised) ──────────────────────────────────────

class NpyPositionDataset(Dataset):
    """One embedding per (gene, position) pair.

    Loads a [N, L, d] npy array and exposes each position as a flat vector.
    Only positions within the real sequence window (seq_offset … seq_offset+seq_len)
    are included; special tokens (CLS/SEP) are excluded.

    Args:
        npy_path:   Path to the .npy file, shape [N_genes, L, d].
        normalize:  Subtract the per-dimension dataset mean (recommended).
        seq_offset: Index of the first real-sequence token in L (default 1 for SpeciesLM).
        seq_len:    Number of real-sequence positions to keep (default 1000).
        dtype:      Torch dtype for returned tensors.
    """

    def __init__(
        self,
        npy_path: str | Path = DEFAULT_NPY_PATH,
        *,
        normalize: bool = True,
        seq_offset: int = SPECIESLM_SEQ_OFFSET,
        seq_len: int = SPECIESLM_SEQ_LEN,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.npy_path = Path(npy_path)
        self.seq_offset = seq_offset
        self.seq_len = seq_len
        self.dtype = dtype

        raw = np.load(self.npy_path).astype(np.float32)  # [N, L, d]
        N, L, d = raw.shape
        end = seq_offset + seq_len
        assert end <= L, f"seq_offset+seq_len={end} exceeds L={L}"

        # Slice to real sequence positions → [N, seq_len, d]
        seq_embs = raw[:, seq_offset:end, :]

        if normalize:
            # Subtract mean over all (gene, position) pairs
            self._mean = seq_embs.reshape(-1, d).mean(axis=0, keepdims=True)  # [1, d]
            seq_embs = seq_embs - self._mean[np.newaxis, :, :]
        else:
            self._mean = np.zeros((1, d), dtype=np.float32)

        # Store as float16 to halve RAM (~10 GB instead of ~20 GB for 6.57M×768)
        # Cast to float32 in __getitem__ just before returning to the model
        self._embs = torch.tensor(seq_embs.reshape(-1, d), dtype=torch.float16)

        # Build index: flat_idx → (gene_idx, pos_idx_in_seq)
        self._index: list[PosIndex] = [
            PosIndex(gene_idx=g, pos_idx=p)
            for g in range(N)
            for p in range(seq_len)
        ]

        self.n_genes = N
        self.seq_len_actual = seq_len
        self.input_dim = d

    def __len__(self) -> int:
        return len(self._embs)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._embs[idx].float()  # cast float16 → float32 for the model

    def gene_pos(self, idx: int) -> PosIndex:
        return self._index[idx]

    def summary(self) -> dict[str, object]:
        return {
            "n_genes": self.n_genes,
            "seq_len": self.seq_len_actual,
            "n_positions": len(self),
            "input_dim": self.input_dim,
            "path": str(self.npy_path),
        }


# ── evaluation dataset (with binding labels) ─────────────────────────────────

@dataclass(frozen=True)
class LabeledPosition:
    gene_idx: int
    pos_idx: int          # 0-based within the sequence (not the L dim)
    chrom: str
    genomic_pos: int      # absolute genomic coordinate of this base
    strand: str
    gene_id: str
    labels: np.ndarray     # shape [n_tfs], 1.0 if a binding site centre falls here ±half_window
    # embedding NOT stored here — loaded on demand from self._seq_embs to save ~20 GB RAM


class NpyPositionWithLabelsDataset(Dataset):
    """Position-level dataset with TF binding labels.

    For each (gene, position), resolves the genomic coordinate and checks
    whether any TF binding site centre falls within ±half_window bases.
    Labels are multi-hot over all TFs with ≥ min_sites_per_tf sites.

    This mirrors the window-based approach in the CNN baseline but operates
    directly on embeddings, enabling apples-to-apples AUROC comparison.
    """

    def __init__(
        self,
        npy_path: str | Path = DEFAULT_NPY_PATH,
        sites_path: str | Path = DEFAULT_SITES_PATH,
        regions_path: str | Path = DEFAULT_REGIONS_PATH,
        *,
        half_window: int = 50,
        min_sites_per_tf: int = 15,
        normalize: bool = True,
        seq_offset: int = SPECIESLM_SEQ_OFFSET,
        seq_len: int = SPECIESLM_SEQ_LEN,
    ) -> None:
        self.half_window = half_window

        # Load embeddings as float16 to save RAM (~10 GB instead of ~20 GB)
        raw = np.load(Path(npy_path))  # float16 [N, L, d]
        N, L, d = raw.shape
        end = seq_offset + seq_len
        self._seq_embs = raw[:, seq_offset:end, :].astype(np.float16)  # [N, seq_len, d]
        if normalize:
            mean = self._seq_embs.reshape(-1, d).mean(axis=0)
            self._seq_embs = (self._seq_embs - mean).astype(np.float16)
        self.input_dim = d

        # Load regions (gene order must match npy row order)
        regions = pl.read_parquet(regions_path)
        assert len(regions) == N, (
            f"npy has {N} genes but regions parquet has {len(regions)} rows. "
            "Gene ordering mismatch — check that the parquet and npy come from the same pipeline."
        )
        region_rows = regions.to_dicts()

        # Load binding sites
        sites = pl.read_parquet(sites_path)
        if min_sites_per_tf > 1:
            sites = sites.filter(pl.len().over("name") >= min_sites_per_tf)
        self.tf_names: list[str] = sites["name"].unique().sort().to_list()
        tf_to_idx = {n: i for i, n in enumerate(self.tf_names)}
        self.n_tfs = len(self.tf_names)

        # Build chrom → sorted list of (centre, tf_idx)
        sites_by_chrom: dict[str, list[tuple[int, int]]] = {}
        for row in sites.iter_rows(named=True):
            centre = (int(row["start"]) + int(row["end"])) // 2
            tf_idx = tf_to_idx[str(row["name"])]
            sites_by_chrom.setdefault(str(row["chrom"]), []).append((centre, tf_idx))
        for lst in sites_by_chrom.values():
            lst.sort()

        # Build per-position records
        self.records: list[LabeledPosition] = []
        for gene_idx, region in enumerate(region_rows):
            chrom = str(region["chrom"])
            reg_start = int(region["start"])
            reg_end = int(region["end"])
            strand = str(region["strand"])
            gene_id = str(region["gene_id"])
            chrom_sites = sites_by_chrom.get(chrom, [])

            for pos_idx in range(seq_len):
                # Map sequence position → genomic coordinate
                if strand == "+":
                    genomic_pos = reg_start + pos_idx
                else:
                    genomic_pos = reg_end - 1 - pos_idx

                labels = self._labels_at(genomic_pos, chrom_sites)
                self.records.append(LabeledPosition(
                    gene_idx=gene_idx,
                    pos_idx=pos_idx,
                    chrom=chrom,
                    genomic_pos=genomic_pos,
                    strand=strand,
                    gene_id=gene_id,
                    labels=labels,
                ))

        self.n_positions = len(self.records)

    def _labels_at(
        self,
        genomic_pos: int,
        chrom_sites: list[tuple[int, int]],
    ) -> np.ndarray:
        labels = np.zeros(self.n_tfs, dtype=np.float32)
        lo, hi = genomic_pos - self.half_window, genomic_pos + self.half_window
        for centre, tf_idx in chrom_sites:
            if centre < lo:
                continue
            if centre > hi:
                break
            labels[tf_idx] = 1.0
        return labels

    def __len__(self) -> int:
        return self.n_positions

    def __getitem__(self, idx: int) -> dict[str, object]:
        rec = self.records[idx]
        # Load embedding on demand from the shared float16 array → cast to float32 for model
        emb = torch.tensor(
            self._seq_embs[rec.gene_idx, rec.pos_idx].astype(np.float32),
            dtype=torch.float32,
        )
        return {
            "embedding": emb,
            "labels": torch.tensor(rec.labels, dtype=torch.float32),
            "gene_idx": rec.gene_idx,
            "pos_idx": rec.pos_idx,
            "genomic_pos": rec.genomic_pos,
            "chrom": rec.chrom,
            "gene_id": rec.gene_id,
        }

    def summary(self) -> dict[str, object]:
        n_pos_with_site = sum(rec.labels.any() for rec in self.records)
        return {
            "n_genes": len(set(r.gene_idx for r in self.records)),
            "n_positions": self.n_positions,
            "n_positions_with_binding_site": n_pos_with_site,
            "n_tfs": self.n_tfs,
            "input_dim": self.input_dim,
        }


# ── Legacy gene-level parquet datasets ───────────────────────────────────────

DEFAULT_EMBEDDING_PATH = Path(
    "/s/project/ml4rg_students/2026/project15/working/sae/embeddings/"
    "_saccharomyces_cerevisiae_evo2_7b.parquet"
)


class EmbeddingDataset(Dataset):
    """Gene-level parquet embeddings (Evo2 / other mean-pooled models)."""

    def __init__(
        self,
        embedding_path: str | Path = DEFAULT_EMBEDDING_PATH,
        *,
        normalize: bool = True,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        df = pl.read_parquet(Path(embedding_path))
        if "embedding" not in df.columns:
            raise ValueError(f"'embedding' column not found in {embedding_path}")
        raw = np.array(df["embedding"].to_list(), dtype=np.float32)
        if normalize:
            raw -= raw.mean(axis=0, keepdims=True)
        self.embeddings = torch.tensor(raw, dtype=dtype)
        self.gene_ids: list[str] = df["gene_id"].to_list() if "gene_id" in df.columns else []
        self.input_dim = raw.shape[1]

    def __len__(self) -> int:
        return len(self.embeddings)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.embeddings[idx]

    def summary(self) -> dict[str, object]:
        return {"n_genes": len(self), "input_dim": self.input_dim}
