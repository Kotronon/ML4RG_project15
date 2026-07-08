from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

try:
    from .dataset import (
        BindingBenchPromoterEmbeddingDataset,
        BindingBenchPromoterSequenceDataset,
        DEFAULT_REGIONS_PATH,
        DEFAULT_SITES_PATH,
        PromoterRecord,
    )
    from .export_binding_bench_predictions import nms_indices
    from .model import build_dense_model, parse_dilations, parse_int_tuple
    from .promoter_splits import load_promoter_split, normalize_promoter_split, split_indices
    from .train_promoter_dense import collate_promoter_batch, prepare_x
except ImportError:
    from dataset import (
        BindingBenchPromoterEmbeddingDataset,
        BindingBenchPromoterSequenceDataset,
        DEFAULT_REGIONS_PATH,
        DEFAULT_SITES_PATH,
        PromoterRecord,
    )
    from export_binding_bench_predictions import nms_indices
    from model import build_dense_model, parse_dilations, parse_int_tuple
    from promoter_splits import load_promoter_split, normalize_promoter_split, split_indices
    from train_promoter_dense import collate_promoter_batch, prepare_x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export DNA-only high-scoring candidate loci for protein reranking."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sites-path", type=Path)
    parser.add_argument("--regions-path", type=Path)
    parser.add_argument("--input-mode", choices=("raw", "embedding"))
    parser.add_argument("--embeddings-path", type=Path)
    parser.add_argument("--embedding-column")
    parser.add_argument("--embedding-key-column")
    parser.add_argument("--min-sites-per-tf", type=int)
    parser.add_argument("--sequence-orientation", choices=("strand-aware", "genomic"))
    parser.add_argument("--include-terminal-atg", action="store_true")
    parser.add_argument("--max-regions", type=int)
    parser.add_argument("--promoter-split-path", type=Path)
    parser.add_argument(
        "--split",
        choices=("all", "train", "val", "test"),
        default="all",
        help="Promoter split to score. Use train for training candidate files.",
    )
    parser.add_argument("--top-k", type=int, default=50_000)
    parser.add_argument("--pre-nms-factor", type=int, default=5)
    parser.add_argument("--nms-radius-bp", type=int, default=50)
    parser.add_argument(
        "--score-mode",
        choices=("logit", "sigmoid"),
        default="logit",
    )
    parser.add_argument(
        "--output-aggregation",
        choices=("max", "mean", "first"),
        default="max",
        help="How to collapse multi-output DNA-only heads into one candidate score.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device) -> dict[str, object]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def get_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def checkpoint_dict(checkpoint: dict[str, object], key: str) -> dict[str, object]:
    value = checkpoint.get(key, {})
    return dict(value) if isinstance(value, dict) else {}


def path_from(value: object, fallback: Path) -> Path:
    if value is None:
        return fallback
    return Path(str(value))


def stringify_int_sequence(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return ",".join(str(int(item)) for item in value)
    return str(value)


def build_config(args: argparse.Namespace, checkpoint: dict[str, object]) -> dict[str, object]:
    saved_args = checkpoint_dict(checkpoint, "args")
    model_config = checkpoint_dict(checkpoint, "model_config")

    input_mode = str(
        args.input_mode
        or model_config.get("input_mode")
        or saved_args.get("input_mode", "raw")
    )
    embeddings_path = args.embeddings_path
    if embeddings_path is None:
        saved_embeddings = model_config.get(
            "embeddings_path", saved_args.get("embeddings_path")
        )
        embeddings_path = Path(str(saved_embeddings)) if saved_embeddings else None
    if input_mode == "embedding" and embeddings_path is None:
        raise ValueError("Embedding input mode requires --embeddings-path")

    if args.include_terminal_atg:
        trim_terminal_atg = False
    else:
        trim_terminal_atg = bool(
            model_config.get(
                "trim_terminal_atg",
                not bool(saved_args.get("include_terminal_atg", False)),
            )
        )

    return {
        "model_name": str(
            checkpoint.get("model_name")
            or model_config.get("model_name")
            or saved_args.get("model", "dense_motif_dilated_attention_cnn")
        ),
        "input_mode": input_mode,
        "input_channels": int(model_config.get("input_channels", 4)),
        "output_channels": int(model_config.get("output_channels", 1)),
        "hidden_channels": int(
            model_config.get("hidden_channels", saved_args.get("hidden_channels", 128))
        ),
        "kernel_size": int(
            model_config.get("kernel_size", saved_args.get("kernel_size", 7))
        ),
        "dropout": float(model_config.get("dropout", saved_args.get("dropout", 0.1))),
        "dilations": parse_dilations(
            stringify_int_sequence(
                model_config.get("dilations", saved_args.get("dilations", "1,2,4,8,16"))
            )
        ),
        "dna_attention_window_bp": int(
            model_config.get(
                "dna_attention_window_bp",
                saved_args.get("dna_attention_window_bp", 50),
            )
        ),
        "dna_attention_layers": int(
            model_config.get(
                "dna_attention_layers",
                saved_args.get("dna_attention_layers", 2),
            )
        ),
        "dna_attention_heads": int(
            model_config.get("dna_attention_heads", saved_args.get("dna_attention_heads", 8))
        ),
        "dna_attention_ffn_multiplier": float(
            model_config.get(
                "dna_attention_ffn_multiplier",
                saved_args.get("dna_attention_ffn_multiplier", 4.0),
            )
        ),
        "motif_kernel_sizes": parse_int_tuple(
            stringify_int_sequence(
                model_config.get(
                    "motif_kernel_sizes",
                    saved_args.get("motif_kernel_sizes", "7,11,15"),
                )
            )
        ),
        "sites_path": args.sites_path
        or path_from(saved_args.get("sites_path"), DEFAULT_SITES_PATH),
        "regions_path": args.regions_path
        or path_from(saved_args.get("regions_path"), DEFAULT_REGIONS_PATH),
        "embeddings_path": embeddings_path,
        "embedding_column": str(
            args.embedding_column
            or model_config.get("embedding_column", saved_args.get("embedding_column", "emb"))
        ),
        "embedding_key_column": (
            args.embedding_key_column
            if args.embedding_key_column is not None
            else model_config.get("embedding_key_column", saved_args.get("embedding_key_column"))
        ),
        "min_sites_per_tf": int(
            args.min_sites_per_tf
            if args.min_sites_per_tf is not None
            else saved_args.get("min_sites_per_tf", 15)
        ),
        "sequence_orientation": str(
            args.sequence_orientation
            or saved_args.get("sequence_orientation", "strand-aware")
        ),
        "trim_terminal_atg": trim_terminal_atg,
    }


def build_dataset(config: dict[str, object], max_regions: int | None):
    common = {
        "sites_path": config["sites_path"],
        "regions_path": config["regions_path"],
        "min_sites_per_tf": int(config["min_sites_per_tf"]),
        "sequence_orientation": str(config["sequence_orientation"]),
        "max_regions": max_regions,
        "trim_terminal_atg": bool(config["trim_terminal_atg"]),
    }
    if config["input_mode"] == "raw":
        return BindingBenchPromoterSequenceDataset(**common)
    return BindingBenchPromoterEmbeddingDataset(
        config["embeddings_path"],
        embedding_column=str(config["embedding_column"]),
        key_column=config["embedding_key_column"],
        **common,
    )


def genomic_positions_for_record(
    record: PromoterRecord,
    sequence_orientation: str,
) -> np.ndarray:
    if sequence_orientation == "strand-aware" and record.strand == "-":
        return np.arange(record.model_end - 1, record.model_start - 1, -1, dtype=np.int64)
    return np.arange(record.model_start, record.model_end, dtype=np.int64)


def aggregate_outputs(logits: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "first":
        return logits[:, 0, :]
    if mode == "mean":
        return logits.mean(dim=1)
    if mode == "max":
        return logits.max(dim=1).values
    raise ValueError(f"Unknown output aggregation: {mode}")


def prune_topk(
    scores: np.ndarray,
    record_indices: np.ndarray,
    offsets: np.ndarray,
    genomic_positions: np.ndarray,
    chrom_codes: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(scores) <= k:
        order = np.argsort(-scores, kind="mergesort")
    else:
        keep = np.argpartition(-scores, k - 1)[:k]
        order = keep[np.argsort(-scores[keep], kind="mergesort")]
    return (
        scores[order],
        record_indices[order],
        offsets[order],
        genomic_positions[order],
        chrom_codes[order],
    )


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.pre_nms_factor <= 0:
        raise ValueError("--pre-nms-factor must be positive")
    if args.nms_radius_bp < 0:
        raise ValueError("--nms-radius-bp must be non-negative")

    device = get_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    config = build_config(args, checkpoint)
    dataset = build_dataset(config, args.max_regions)

    if args.promoter_split_path is not None and args.split != "all":
        split = normalize_promoter_split(load_promoter_split(args.promoter_split_path), dataset.records)
        indices = split_indices(split, args.split)
    else:
        indices = list(range(len(dataset)))
    if not indices:
        raise ValueError(f"No promoters selected for split {args.split!r}")

    model = build_dense_model(
        str(config["model_name"]),
        n_tfs=int(config["output_channels"]),
        input_channels=int(config["input_channels"]),
        hidden_channels=int(config["hidden_channels"]),
        kernel_size=int(config["kernel_size"]),
        dropout=float(config["dropout"]),
        dilations=config["dilations"],
        dna_attention_window_bp=int(config["dna_attention_window_bp"]),
        dna_attention_layers=int(config["dna_attention_layers"]),
        dna_attention_heads=int(config["dna_attention_heads"]),
        dna_attention_ffn_multiplier=float(config["dna_attention_ffn_multiplier"]),
        motif_kernel_sizes=config["motif_kernel_sizes"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=partial(collate_promoter_batch, input_mode=str(config["input_mode"])),
    )

    candidate_k = args.top_k if args.nms_radius_bp == 0 else args.top_k * args.pre_nms_factor
    best_scores = np.empty((0,), dtype=np.float32)
    best_record_indices = np.empty((0,), dtype=np.int32)
    best_offsets = np.empty((0,), dtype=np.int32)
    best_genomic_positions = np.empty((0,), dtype=np.int64)
    best_chrom_codes = np.empty((0,), dtype=np.int32)
    chrom_to_code: dict[str, int] = {}
    chrom_names: list[str] = []

    print(f"Checkpoint:        {args.checkpoint}")
    print(f"Model:             {config['model_name']}")
    print(f"Output channels:   {config['output_channels']}")
    print(f"Input mode:        {config['input_mode']}")
    print(f"Promoters scored:  {len(indices)}")
    print(f"Candidate top-k:   {args.top_k}")
    print(f"Pre-NMS kept:      {candidate_k}")
    print(f"NMS radius bp:     {args.nms_radius_bp}")
    print(f"Output:            {args.output}")

    for batch in tqdm(loader, desc="Scoring DNA candidates"):
        x = prepare_x(batch["x"].to(device, non_blocking=True), str(config["input_mode"]))
        mask = batch["mask"].detach().cpu().numpy().astype(bool, copy=False)
        with torch.no_grad():
            outputs = model(x)
            logits = outputs["logits"] if isinstance(outputs, dict) else outputs
            scores = aggregate_outputs(logits, args.output_aggregation)
            if args.score_mode == "sigmoid":
                scores = scores.sigmoid()
        scores_np = scores.detach().cpu().numpy().astype(np.float32, copy=False)

        batch_scores = []
        batch_record_indices = []
        batch_offsets = []
        batch_genomic_positions = []
        batch_chrom_codes = []
        for local_idx, meta in enumerate(batch["meta"]):
            record_idx = int(meta["record_idx"])
            record = dataset.records[record_idx]
            chrom_code = chrom_to_code.setdefault(record.chrom, len(chrom_names))
            if chrom_code == len(chrom_names):
                chrom_names.append(record.chrom)
            valid_offsets = np.flatnonzero(mask[local_idx])
            if len(valid_offsets) == 0:
                continue
            positions = genomic_positions_for_record(
                record,
                str(config["sequence_orientation"]),
            )[valid_offsets]
            batch_scores.append(scores_np[local_idx, valid_offsets])
            batch_record_indices.append(
                np.full(len(valid_offsets), record_idx, dtype=np.int32)
            )
            batch_offsets.append(valid_offsets.astype(np.int32, copy=False))
            batch_genomic_positions.append(positions)
            batch_chrom_codes.append(
                np.full(len(valid_offsets), chrom_code, dtype=np.int32)
            )

        if not batch_scores:
            continue
        merged_scores = np.concatenate([best_scores, *batch_scores])
        merged_record_indices = np.concatenate([best_record_indices, *batch_record_indices])
        merged_offsets = np.concatenate([best_offsets, *batch_offsets])
        merged_genomic_positions = np.concatenate(
            [best_genomic_positions, *batch_genomic_positions]
        )
        merged_chrom_codes = np.concatenate([best_chrom_codes, *batch_chrom_codes])
        (
            best_scores,
            best_record_indices,
            best_offsets,
            best_genomic_positions,
            best_chrom_codes,
        ) = prune_topk(
            merged_scores,
            merged_record_indices,
            merged_offsets,
            merged_genomic_positions,
            merged_chrom_codes,
            candidate_k,
        )

    if len(best_scores) == 0:
        raise RuntimeError("No candidate scores were produced")

    if args.nms_radius_bp > 0:
        keep = nms_indices(
            chrom_codes=best_chrom_codes,
            starts=best_genomic_positions,
            scores=best_scores,
            radius_bp=args.nms_radius_bp,
            max_keep=args.top_k,
        )
    else:
        keep = np.arange(min(args.top_k, len(best_scores)))

    records = [dataset.records[int(idx)] for idx in best_record_indices[keep]]
    out = pl.DataFrame(
        {
            "rank": np.arange(1, len(keep) + 1, dtype=np.int32),
            "record_idx": best_record_indices[keep],
            "gene_id": [record.gene_id for record in records],
            "chrom": [record.chrom for record in records],
            "genomic_pos": best_genomic_positions[keep],
            "start": best_genomic_positions[keep],
            "end": best_genomic_positions[keep] + 1,
            "offset": best_offsets[keep],
            "score": best_scores[keep],
            "region_start": [record.start for record in records],
            "region_end": [record.end for record in records],
            "region_strand": [record.strand for record in records],
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(args.output)
    metadata = {
        "checkpoint": str(args.checkpoint),
        "sites_path": str(config["sites_path"]),
        "regions_path": str(config["regions_path"]),
        "input_mode": config["input_mode"],
        "embeddings_path": (
            str(config["embeddings_path"]) if config["embeddings_path"] is not None else None
        ),
        "promoter_split_path": (
            str(args.promoter_split_path) if args.promoter_split_path is not None else None
        ),
        "split": args.split,
        "top_k": args.top_k,
        "pre_nms_factor": args.pre_nms_factor,
        "candidate_k": candidate_k,
        "nms_radius_bp": args.nms_radius_bp,
        "score_mode": args.score_mode,
        "output_aggregation": args.output_aggregation,
        "n_candidates": out.height,
    }
    args.output.with_suffix(".metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote candidates: {args.output} ({out.height:,} rows)")
    print(out.head())


if __name__ == "__main__":
    main()
