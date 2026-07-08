"""Export per-position SAE activations as Binding Bench peak predictions.

For each SAE latent feature, scans all promoter positions, keeps the top
scoring sites genome-wide, applies peak calling (NMS or scipy find_peaks),
and writes Binding Bench-compatible ``predictions.parquet`` and
``feature_ranks.parquet``.

Example:
    python -m sae.export_binding_bench_predictions \\
        --checkpoint /path/to/sae/models/topk_ef8_k32/best.pt \\
        --npy-path /path/to/embs_ds_fungi_upstream_ATG_1000_l10.npy \\
        --predictions-out working/binding_bench_inputs/sae_topk_predictions.parquet \\
        --feature-ranks-out working/binding_bench_inputs/sae_topk_feature_ranks.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
try:
    import polars as pl
except ModuleNotFoundError:
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "polars"])
    import polars as pl
import torch
from tqdm import tqdm

try:
    from .binding_bench_utils import genomic_pos_for_offset
    from .dataset import DEFAULT_NPY_PATH, DEFAULT_REGIONS_PATH, SPECIESLM_SEQ_LEN, SPECIESLM_SEQ_OFFSET
    from .model import SAEConfig, SparseAutoencoder, build_sae
except ImportError:
    from binding_bench_utils import genomic_pos_for_offset
    from dataset import DEFAULT_NPY_PATH, DEFAULT_REGIONS_PATH, SPECIESLM_SEQ_LEN, SPECIESLM_SEQ_OFFSET
    from model import SAEConfig, SparseAutoencoder, build_sae

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from supervised_baseline.export_binding_bench_predictions import (  # noqa: E402
    TopKBuffer,
    call_peaks_indices,
    nms_indices,
)

DEFAULT_PROJECT = Path("/s/project/ml4rg_students/2026/project15")
DEFAULT_INPUT_DIR = DEFAULT_PROJECT / "working/binding_bench_inputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--npy-path", type=Path, default=DEFAULT_NPY_PATH)
    parser.add_argument("--regions-path", type=Path, default=DEFAULT_REGIONS_PATH)
    parser.add_argument("--predictions-out", type=Path, required=True)
    parser.add_argument("--feature-ranks-out", type=Path, required=True)
    parser.add_argument("--seq-offset", type=int, default=SPECIESLM_SEQ_OFFSET)
    parser.add_argument("--seq-len", type=int, default=SPECIESLM_SEQ_LEN)
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--top-k-per-feature", type=int, default=5000)
    parser.add_argument("--nms-radius-bp", type=int, default=50)
    parser.add_argument(
        "--peak-caller",
        choices=("nms", "scipy-find-peaks"),
        default="nms",
    )
    parser.add_argument("--pre-nms-factor", type=int, default=20)
    parser.add_argument(
        "--feature-rank-method",
        choices=("max-score", "mean-top-score", "hit-count"),
        default="max-score",
    )
    parser.add_argument("--batch-size", type=int, default=1000, help="Positions per SAE forward pass.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-regions", type=int, help="Limit promoters for debugging.")
    return parser.parse_args()


def get_device(request: str) -> torch.device:
    if request == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(request)


def load_model(path: Path, device: torch.device) -> SparseAutoencoder:
    checkpoint = torch.load(path, map_location=device)
    model = build_sae(SAEConfig(**checkpoint["sae_config"])).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def load_embeddings(
    npy_path: Path,
    *,
    seq_offset: int,
    seq_len: int,
    normalize: bool,
) -> tuple[np.ndarray, np.ndarray | None]:
    raw = np.load(npy_path, mmap_mode="r")
    end = seq_offset + seq_len
    if end > raw.shape[1]:
        raise ValueError(f"seq_offset+seq_len={end} exceeds embedding length {raw.shape[1]}")

    mean: np.ndarray | None = None
    if normalize:
        seq_embs = np.array(raw[:, seq_offset:end, :], dtype=np.float32)
        mean = seq_embs.reshape(-1, seq_embs.shape[-1]).mean(axis=0, keepdims=True)
    return raw, mean


def write_outputs(
    *,
    topk: TopKBuffer,
    chrom_names: list[str],
    regions: list[dict[str, object]],
    predictions_out: Path,
    feature_ranks_out: Path,
    nms_radius_bp: int,
    top_k_per_feature: int,
    peak_caller: str,
    feature_rank_method: str,
    hidden_dim: int,
) -> None:
    prediction_frames: list[pl.DataFrame] = []
    feature_scores: list[tuple[str, float]] = []

    for feature_idx in range(hidden_dim):
        scores = topk.scores[:, feature_idx]
        valid = (
            np.isfinite(scores)
            & (topk.chrom_codes[:, feature_idx] >= 0)
            & (topk.region_codes[:, feature_idx] >= 0)
        )
        if not valid.any():
            continue

        starts_raw = topk.starts[valid, feature_idx]
        chrom_codes_raw = topk.chrom_codes[valid, feature_idx]
        region_codes_raw = topk.region_codes[valid, feature_idx]
        scores_raw = scores[valid]

        if peak_caller == "nms":
            keep = nms_indices(
                chrom_codes=chrom_codes_raw,
                starts=starts_raw,
                scores=scores_raw,
                radius_bp=nms_radius_bp,
                max_keep=top_k_per_feature,
            )
        elif peak_caller == "scipy-find-peaks":
            keep = call_peaks_indices(
                chrom_codes=chrom_codes_raw,
                starts=starts_raw,
                scores=scores_raw,
                radius_bp=nms_radius_bp,
                max_keep=top_k_per_feature,
            )
        else:
            raise ValueError(f"Unknown peak caller: {peak_caller}")

        starts = starts_raw[keep]
        chrom_codes = chrom_codes_raw[keep]
        region_codes = region_codes_raw[keep]
        sorted_scores = scores_raw[keep]
        feature_name = str(feature_idx)

        if feature_rank_method == "max-score":
            feature_score = float(sorted_scores[0])
        elif feature_rank_method == "mean-top-score":
            feature_score = float(np.mean(sorted_scores))
        elif feature_rank_method == "hit-count":
            feature_score = float(len(sorted_scores))
        else:
            raise ValueError(f"Unknown feature-rank method: {feature_rank_method}")
        feature_scores.append((feature_name, feature_score))

        region_rows = [regions[int(code)] for code in region_codes]
        prediction_frames.append(
            pl.DataFrame(
                {
                    "chrom": [chrom_names[int(code)] for code in chrom_codes],
                    "start": starts,
                    "end": starts + 1,
                    "feature_idx": [feature_name] * len(starts),
                    "score": sorted_scores,
                    "strand": ["."] * len(starts),
                    "gene_id": [str(region["gene_id"]) for region in region_rows],
                    "sequence_name": [
                        f"{region['chrom']}:{region['start']}-{region['end']}:{region['strand']}"
                        for region in region_rows
                    ],
                    "region_start": [int(region["start"]) for region in region_rows],
                    "region_end": [int(region["end"]) for region in region_rows],
                    "region_strand": [str(region["strand"]) for region in region_rows],
                }
            )
        )

    if not prediction_frames:
        raise RuntimeError("No SAE feature predictions were produced")

    full_predictions = (
        pl.concat(prediction_frames, how="vertical")
        .sort(["feature_idx", "score"], descending=[False, True])
        .unique(
            subset=["chrom", "start", "end", "feature_idx"],
            keep="first",
            maintain_order=True,
        )
        .sort(
            ["feature_idx", "score", "chrom", "start"],
            descending=[False, True, False, False],
        )
    )

    predictions = full_predictions.select(
        "chrom",
        "start",
        "end",
        "feature_idx",
        "score",
        "strand",
    )
    predictions_out.parent.mkdir(parents=True, exist_ok=True)
    predictions.write_parquet(predictions_out)

    audit_out = predictions_out.with_suffix(".audit.parquet")
    full_predictions.write_parquet(audit_out)

    ranks = (
        pl.DataFrame(
            feature_scores,
            schema=["feature_idx", "feature_score"],
            orient="row",
        )
        .sort(["feature_score", "feature_idx"], descending=[True, False])
        .with_row_index("feature_rank", offset=1)
        .select("feature_idx", "feature_rank", "feature_score")
    )
    feature_ranks_out.parent.mkdir(parents=True, exist_ok=True)
    ranks.write_parquet(feature_ranks_out)

    print(f"Wrote predictions: {predictions_out} ({predictions.height:,} rows)")
    print(f"Wrote audit table: {audit_out} ({full_predictions.height:,} rows)")
    print(f"Wrote feature ranks: {feature_ranks_out} ({ranks.height:,} features)")
    print("Preview:")
    print(predictions.head(5))


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if args.top_k_per_feature <= 0:
        raise ValueError("--top-k-per-feature must be positive")
    if args.nms_radius_bp < 0:
        raise ValueError("--nms-radius-bp must be non-negative")
    if args.pre_nms_factor <= 0:
        raise ValueError("--pre-nms-factor must be positive")

    device = get_device(args.device)
    model = load_model(args.checkpoint, device)
    hidden_dim = model.cfg.hidden_dim
    print(
        f"SAE: variant={model.cfg.variant}  input={model.cfg.input_dim}  "
        f"hidden={hidden_dim}  k={model.cfg.k}"
    )

    regions_df = pl.read_parquet(args.regions_path)
    if args.max_regions is not None:
        regions_df = regions_df.head(args.max_regions)
    regions = regions_df.to_dicts()

    raw, mean = load_embeddings(
        args.npy_path,
        seq_offset=args.seq_offset,
        seq_len=args.seq_len,
        normalize=args.normalize,
    )
    if len(regions) != raw.shape[0]:
        raise ValueError(
            f"Regions parquet has {len(regions)} rows but npy has {raw.shape[0]} genes"
        )

    candidate_k = (
        args.top_k_per_feature
        if args.nms_radius_bp == 0
        else args.top_k_per_feature * args.pre_nms_factor
    )
    topk = TopKBuffer(n_features=hidden_dim, k=candidate_k)
    chrom_to_code: dict[str, int] = {}
    chrom_names: list[str] = []

    for gene_idx, region in enumerate(tqdm(regions, desc="Scanning promoters")):
        chrom = str(region["chrom"])
        chrom_code = chrom_to_code.setdefault(chrom, len(chrom_names))
        if chrom_code == len(chrom_names):
            chrom_names.append(chrom)

        gene_embs = raw[gene_idx, args.seq_offset : args.seq_offset + args.seq_len, :]
        if mean is not None:
            gene_embs = np.array(gene_embs, dtype=np.float32) - mean
        else:
            gene_embs = np.array(gene_embs, dtype=np.float32)

        for start in range(0, args.seq_len, args.batch_size):
            end = min(start + args.batch_size, args.seq_len)
            batch = torch.tensor(gene_embs[start:end], dtype=torch.float32, device=device)
            acts = model(batch)["activations"].detach().cpu().numpy().astype(np.float32)

            pos_offsets = np.arange(start, end, dtype=np.int64)
            genomic_starts = np.array(
                [genomic_pos_for_offset(region, int(pos)) for pos in pos_offsets],
                dtype=np.int64,
            )

            topk.update(
                acts,
                np.full(len(pos_offsets), chrom_code, dtype=np.int32),
                np.full(len(pos_offsets), gene_idx, dtype=np.int32),
                genomic_starts,
            )

    write_outputs(
        topk=topk,
        chrom_names=chrom_names,
        regions=regions,
        predictions_out=args.predictions_out,
        feature_ranks_out=args.feature_ranks_out,
        nms_radius_bp=args.nms_radius_bp,
        top_k_per_feature=args.top_k_per_feature,
        peak_caller=args.peak_caller,
        feature_rank_method=args.feature_rank_method,
        hidden_dim=hidden_dim,
    )


if __name__ == "__main__":
    main()
