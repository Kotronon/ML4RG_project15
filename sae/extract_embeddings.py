"""Extract DNA LM embeddings for upstream regions and save to parquet.

Only intended for anchor species (S. cerevisiae, S. pombe) where saving
embeddings to disk is feasible. For the hundreds of other fungal species,
generate embeddings on-the-fly in the SAE dataset to avoid disk explosion.

Usage:
    python -m sae.extract_embeddings \
        --regions-path /path/to/_saccharomyces_cerevisiae_sequence_mapper.parquet \
        --model-name evo2_7b \
        --output-path /path/to/embeddings/_saccharomyces_cerevisiae_evo2_7b.parquet

Supported --model-name values:
    evo2_7b         Arc Institute Evo 2 7B (hidden dim 4096)
    evo2_40b        Arc Institute Evo 2 40B (hidden dim 8192)
    specieslm       Karollus et al. SpeciesLM (hidden dim 512)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
import torch
from tqdm import tqdm

ModelName = Literal["evo2_7b", "evo2_40b", "specieslm"]

KNOWN_HIDDEN_DIMS: dict[str, int] = {
    "evo2_7b": 4096,
    "evo2_40b": 8192,
    "specieslm": 512,
}

# Batch sizes tuned for 40 GB A100; reduce if OOM.
DEFAULT_BATCH_SIZES: dict[str, int] = {
    "evo2_7b": 4,
    "evo2_40b": 1,
    "specieslm": 32,
}

DEFAULT_REGIONS_PATH = Path(
    "/s/project/multispecies/fungi_code/tf_sae/processed/"
    "sequence_datasets/fungi_upstream_ATG_1000/"
    "_saccharomyces_cerevisiae_sequence_mapper.parquet"
)

DEFAULT_OUTPUT_ROOT = Path(
    "/s/project/ml4rg_students/2026/project15/working/sae/embeddings"
)


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def load_evo2(model_name: str, device: torch.device):
    """Load Evo 2 model. Requires the `evo` package from Arc Institute."""
    from evo2 import Evo  # type: ignore[import]
    model_tag = "evo2_7b" if model_name == "evo2_7b" else "evo2_40b"
    model, tokenizer = Evo(model_tag)
    model = model.to(device).eval()
    return model, tokenizer, "evo2"


def load_specieslm(device: torch.device):
    """Load SpeciesLM (Karollus et al.) from HuggingFace."""
    from transformers import AutoTokenizer, AutoModel  # type: ignore[import]
    model_id = "katarinayuan/SpeciesLM"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    model = model.to(device).eval()
    return model, tokenizer, "specieslm"


def load_model(model_name: str, device: torch.device):
    if model_name in ("evo2_7b", "evo2_40b"):
        return load_evo2(model_name, device)
    if model_name == "specieslm":
        return load_specieslm(device)
    raise ValueError(f"Unknown model: {model_name!r}")


# ---------------------------------------------------------------------------
# Embedding extraction
# ---------------------------------------------------------------------------

def _rc(seq: str) -> str:
    comp = str.maketrans("ACGTacgt", "TGCAtgca")
    return seq.translate(comp)[::-1]


def extract_batch_evo2(
    model,
    tokenizer,
    sequences: list[str],
    device: torch.device,
    layer: int = -1,
) -> np.ndarray:
    """Returns mean-pooled embeddings, shape [B, d]."""
    tokens = torch.tensor(
        [tokenizer.tokenize(s) for s in sequences],
        dtype=torch.long,
        device=device,
    )
    with torch.no_grad():
        outputs, _ = model(tokens, return_embeddings=True, layer_idx=layer)
    # outputs: [B, L, d]  — mean pool across sequence length
    return outputs.float().mean(dim=1).cpu().numpy()


def extract_batch_specieslm(
    model,
    tokenizer,
    sequences: list[str],
    device: torch.device,
) -> np.ndarray:
    """Returns mean-pooled last-hidden-state embeddings, shape [B, d]."""
    enc = tokenizer(
        sequences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc)
    hidden = out.last_hidden_state  # [B, L, d]
    mask = enc["attention_mask"].unsqueeze(-1).float()
    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
    return pooled.float().cpu().numpy()


def extract_embeddings(
    sequences: list[str],
    model,
    tokenizer,
    model_type: str,
    device: torch.device,
    batch_size: int,
    strand: list[str] | None = None,
) -> np.ndarray:
    """Extract embeddings for all sequences, batched. Returns [N, d]."""
    all_embs: list[np.ndarray] = []
    for i in tqdm(range(0, len(sequences), batch_size), desc="Extracting"):
        batch_seqs = sequences[i : i + batch_size]
        if strand is not None:
            # present sequences in 5'→3' direction
            batch_seqs = [
                _rc(s) if strand[i + j] == "-" else s
                for j, s in enumerate(batch_seqs)
            ]
        if model_type == "evo2":
            emb = extract_batch_evo2(model, tokenizer, batch_seqs, device)
        elif model_type == "specieslm":
            emb = extract_batch_specieslm(model, tokenizer, batch_seqs, device)
        else:
            raise ValueError(model_type)
        all_embs.append(emb)
    return np.concatenate(all_embs, axis=0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--regions-path", type=Path, default=DEFAULT_REGIONS_PATH)
    p.add_argument(
        "--model-name",
        choices=list(KNOWN_HIDDEN_DIMS),
        default="evo2_7b",
    )
    p.add_argument(
        "--output-path",
        type=Path,
        help="Where to write the output parquet. Defaults to OUTPUT_ROOT/species_model.parquet",
    )
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--layer", type=int, default=-1, help="Which hidden layer to extract (Evo2 only).")
    p.add_argument("--device", default="auto")
    p.add_argument("--strand-aware", action="store_true", default=True,
                   help="Reverse-complement minus-strand sequences before embedding (default: True).")
    p.add_argument("--no-strand-aware", dest="strand_aware", action="store_false")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    print(f"Device: {device}")

    regions = pl.read_parquet(args.regions_path)
    required = {"gene_id", "chrom", "start", "end", "strand", "seq"}
    missing = required - set(regions.columns)
    if missing:
        raise ValueError(f"Regions parquet missing columns: {sorted(missing)}")

    sequences = regions["seq"].to_list()
    strands = regions["strand"].to_list() if args.strand_aware else None

    batch_size = args.batch_size or DEFAULT_BATCH_SIZES[args.model_name]
    model, tokenizer, model_type = load_model(args.model_name, device)

    print(f"Extracting embeddings for {len(sequences)} sequences with {args.model_name}...")
    embs = extract_embeddings(
        sequences, model, tokenizer, model_type, device, batch_size, strand=strands
    )
    print(f"Embeddings shape: {embs.shape}")

    if args.output_path is None:
        species = args.regions_path.stem.replace("_sequence_mapper", "")
        args.output_root.mkdir(parents=True, exist_ok=True)
        args.output_path = args.output_root / f"{species}_{args.model_name}.parquet"

    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    meta_cols = [c for c in ["gene_id", "chrom", "start", "end", "strand"] if c in regions.columns]
    out = regions.select(meta_cols).with_columns(
        pl.Series("embedding", embs.tolist()),
        pl.lit(args.model_name).alias("model"),
        pl.lit(args.layer).alias("layer"),
    )
    out.write_parquet(args.output_path)
    print(f"Saved → {args.output_path}  ({len(out)} rows, dim={embs.shape[1]})")


if __name__ == "__main__":
    main()
