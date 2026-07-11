from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from .dataset import (
        BindingBenchPromoterEmbeddingDataset,
        BindingBenchPromoterSequenceDataset,
        DEFAULT_REGIONS_PATH,
        DEFAULT_SITES_PATH,
        PromoterRecord,
    )
    from .export_binding_bench_predictions import (
        TopKBuffer,
        call_peaks_indices,
        nms_indices,
        sequence_name_for_region,
    )
    from .model import DENSE_MODEL_NAMES, build_dense_model, parse_dilations, parse_int_tuple
    from .tf_embeddings import load_tf_embeddings
    from .train_promoter_dense import collate_promoter_batch, prepare_x
except ImportError:
    from dataset import (
        BindingBenchPromoterEmbeddingDataset,
        BindingBenchPromoterSequenceDataset,
        DEFAULT_REGIONS_PATH,
        DEFAULT_SITES_PATH,
        PromoterRecord,
    )
    from export_binding_bench_predictions import (
        TopKBuffer,
        call_peaks_indices,
        nms_indices,
        sequence_name_for_region,
    )
    from model import DENSE_MODEL_NAMES, build_dense_model, parse_dilations, parse_int_tuple
    from tf_embeddings import load_tf_embeddings
    from train_promoter_dense import collate_promoter_batch, prepare_x


DEFAULT_PROJECT = Path("/s/project/ml4rg_students/2026/project15")
DEFAULT_INPUT_DIR = DEFAULT_PROJECT / "working/binding_bench_inputs"
PROTEIN_DENSE_MODEL_NAMES = (
    "dense_protein_res_dilated_cnn",
    "dense_protein_local_attention",
    "dense_protein_motif_cnn",
    "dense_protein_residual_bilinear_cnn",
    "dense_protein_direct_scorer_cnn",
    "dense_protein_window_localization_cnn",
    "dense_protein_res_dilated_crossattention",
    "dense_transbind_cnn_lstm_attention",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export promoter-level dense CNN predictions in Binding Bench "
            "discrete format."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sites-path", type=Path, help="Defaults to the saved training path.")
    parser.add_argument("--regions-path", type=Path, help="Defaults to the saved training path.")
    parser.add_argument(
        "--input-mode",
        choices=("raw", "embedding"),
        help="Override checkpoint input mode.",
    )
    parser.add_argument(
        "--embeddings-path",
        type=Path,
        help="Override checkpoint embedding path for --input-mode embedding.",
    )
    parser.add_argument("--embedding-column", help="Override checkpoint embedding column.")
    parser.add_argument("--embedding-key-column", help="Override checkpoint embedding key column.")
    parser.add_argument(
        "--tf-embeddings-path",
        type=Path,
        help="Override checkpoint protein embeddings path for dense protein models.",
    )
    parser.add_argument("--tf-embedding-key-column")
    parser.add_argument("--tf-embedding-column")
    parser.add_argument("--tf-name-map", type=Path)
    parser.add_argument("--predictions-out", type=Path, required=True)
    parser.add_argument("--feature-ranks-out", type=Path, required=True)
    parser.add_argument(
        "--model",
        choices=DENSE_MODEL_NAMES,
        help="Override checkpoint dense model name.",
    )
    parser.add_argument("--hidden-channels", type=int)
    parser.add_argument("--kernel-size", type=int)
    parser.add_argument("--dilations")
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--score-mode", choices=("sigmoid", "logit"), default="logit")
    parser.add_argument("--top-k-per-tf", type=int, default=5000)
    parser.add_argument(
        "--nms-radius-bp",
        type=int,
        default=0,
        help="Use 0 for the clean dense-position baseline; set e.g. 50 for peak collapse.",
    )
    parser.add_argument(
        "--peak-caller",
        choices=("nms", "scipy-find-peaks"),
        default="nms",
        help=(
            "Choose dense positions with greedy NMS or scipy.signal.find_peaks "
            "before top-K export."
        ),
    )
    parser.add_argument(
        "--pre-nms-factor",
        type=int,
        default=20,
        help="Keep top_k_per_tf * pre_nms_factor candidates before NMS.",
    )
    parser.add_argument(
        "--feature-rank-method",
        choices=("max-score", "mean-top-score", "hit-count", "tf-name"),
        default="max-score",
    )
    parser.add_argument("--min-sites-per-tf", type=int)
    parser.add_argument(
        "--sequence-orientation",
        choices=("strand-aware", "genomic"),
        help="Defaults to the value saved in the training checkpoint.",
    )
    parser.add_argument(
        "--include-terminal-atg",
        action="store_true",
        help=(
            "Force export with the terminal ATG retained. By default, export "
            "uses the checkpoint setting and trims ATG for promoter-only models."
        ),
    )
    parser.add_argument("--max-regions", type=int)
    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device) -> dict[str, object]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def get_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def checkpoint_args(checkpoint: dict[str, object]) -> dict[str, object]:
    args = checkpoint.get("args", {})
    return dict(args) if isinstance(args, dict) else {}


def checkpoint_model_config(checkpoint: dict[str, object]) -> dict[str, object]:
    config = checkpoint.get("model_config", {})
    return dict(config) if isinstance(config, dict) else {}


def arg_path(value: object, fallback: Path) -> Path:
    if value is None:
        return fallback
    return Path(str(value))


def resolve_export_config(
    args: argparse.Namespace,
    checkpoint: dict[str, object],
) -> dict[str, object]:
    saved_args = checkpoint_args(checkpoint)
    saved_config = checkpoint_model_config(checkpoint)

    input_mode = str(args.input_mode or saved_config.get("input_mode", saved_args.get("input_mode", "raw")))
    label_mode = str(saved_config.get("label_mode", saved_args.get("label_mode", "tf")))
    model_name = str(
        args.model
        or checkpoint.get("model_name")
        or saved_config.get("model_name")
        or saved_args.get("model", "dense_small_cnn")
    )
    hidden_channels = int(
        args.hidden_channels
        or saved_config.get("hidden_channels", saved_args.get("hidden_channels", 128))
    )
    kernel_size = int(
        args.kernel_size
        or saved_config.get("kernel_size", saved_args.get("kernel_size", 7))
    )
    dropout = args.dropout
    if dropout is None:
        dropout = float(saved_config.get("dropout", saved_args.get("dropout", 0.1)))
    dilations_raw = (
        args.dilations
        if args.dilations is not None
        else saved_config.get("dilations", saved_args.get("dilations", "1,2,4,8,16"))
    )
    tf_embedding_dropout = float(
        saved_config.get("tf_embedding_dropout", saved_args.get("tf_embedding_dropout", 0.0))
    )
    cross_attention_gate_logit_init = float(
        saved_config.get(
            "cross_attention_gate_logit_init",
            saved_args.get("cross_attention_gate_logit_init", -3.0),
        )
    )
    cross_attention_context_pool_sizes_raw = saved_config.get(
        "cross_attention_context_pool_sizes",
        saved_args.get("cross_attention_context_pool_sizes", "4"),
    )
    dna_attention_window_bp = int(
        saved_config.get(
            "dna_attention_window_bp",
            saved_args.get("dna_attention_window_bp", 50),
        )
    )
    dna_attention_layers = int(
        saved_config.get(
            "dna_attention_layers",
            saved_args.get("dna_attention_layers", 2),
        )
    )
    dna_attention_heads = int(
        saved_config.get(
            "dna_attention_heads",
            saved_args.get("dna_attention_heads", 8),
        )
    )
    dna_attention_ffn_multiplier = float(
        saved_config.get(
            "dna_attention_ffn_multiplier",
            saved_args.get("dna_attention_ffn_multiplier", 4.0),
        )
    )
    motif_kernel_sizes_raw = saved_config.get(
        "motif_kernel_sizes",
        saved_args.get("motif_kernel_sizes", "7,11,15"),
    )
    protein_noise_std = float(
        saved_config.get("protein_noise_std", saved_args.get("protein_noise_std", 0.0))
    )
    protein_l2_normalize = bool(
        saved_config.get(
            "protein_l2_normalize",
            saved_args.get("protein_l2_normalize", False),
        )
    )
    protein_delta_gate_logit_init = float(
        saved_config.get(
            "protein_delta_gate_logit_init",
            saved_args.get("protein_delta_gate_logit_init", -3.0),
        )
    )
    scorer = str(saved_config.get("scorer", saved_args.get("scorer", "multihead_bilinear")))
    scorer_heads = int(saved_config.get("scorer_heads", saved_args.get("scorer_heads", 8)))
    scorer_pair_dim = int(
        saved_config.get("scorer_pair_dim", saved_args.get("scorer_pair_dim", 32))
    )
    scorer_hidden_dim = int(
        saved_config.get("scorer_hidden_dim", saved_args.get("scorer_hidden_dim", 128))
    )
    scorer_bias_mode = str(
        saved_config.get("scorer_bias_mode", saved_args.get("scorer_bias_mode", "tf"))
    )
    window_pooling = str(
        saved_config.get(
            "window_pooling",
            saved_args.get("window_pooling", "topk_logmeanexp"),
        )
    )
    window_pooling_top_k = int(
        saved_config.get(
            "window_pooling_top_k",
            saved_args.get("window_pooling_top_k", 10),
        )
    )

    sites_path = args.sites_path or arg_path(saved_args.get("sites_path"), DEFAULT_SITES_PATH)
    regions_path = args.regions_path or arg_path(saved_args.get("regions_path"), DEFAULT_REGIONS_PATH)
    embeddings_path = args.embeddings_path
    if embeddings_path is None:
        saved_embedding = saved_config.get("embeddings_path", saved_args.get("embeddings_path"))
        embeddings_path = Path(str(saved_embedding)) if saved_embedding else None
    embedding_column = str(
        args.embedding_column
        or saved_config.get("embedding_column", saved_args.get("embedding_column", "emb"))
    )
    embedding_key_column = (
        args.embedding_key_column
        if args.embedding_key_column is not None
        else saved_config.get("embedding_key_column", saved_args.get("embedding_key_column"))
    )
    tf_embeddings_path = args.tf_embeddings_path
    if tf_embeddings_path is None:
        saved_tf_embedding = saved_config.get(
            "tf_embeddings_path", saved_args.get("tf_embeddings_path")
        )
        tf_embeddings_path = Path(str(saved_tf_embedding)) if saved_tf_embedding else None
    tf_embedding_key_column = (
        args.tf_embedding_key_column
        if args.tf_embedding_key_column is not None
        else saved_config.get(
            "tf_embedding_key_column", saved_args.get("tf_embedding_key_column")
        )
    )
    tf_embedding_column = str(
        args.tf_embedding_column
        or saved_config.get("tf_embedding_column", saved_args.get("tf_embedding_column", "emb"))
    )
    tf_name_map = args.tf_name_map
    if tf_name_map is None:
        saved_tf_name_map = saved_config.get("tf_name_map", saved_args.get("tf_name_map"))
        tf_name_map = Path(str(saved_tf_name_map)) if saved_tf_name_map else None
    min_sites_per_tf = int(
        args.min_sites_per_tf
        if args.min_sites_per_tf is not None
        else saved_args.get("min_sites_per_tf", 15)
    )
    sequence_orientation = str(
        args.sequence_orientation or saved_args.get("sequence_orientation", "strand-aware")
    )
    if args.include_terminal_atg:
        trim_terminal_atg = False
    else:
        trim_terminal_atg = bool(
            saved_config.get(
                "trim_terminal_atg",
                not bool(saved_args.get("include_terminal_atg", False)),
            )
        )

    if input_mode == "embedding" and embeddings_path is None:
        raise ValueError(
            "Checkpoint/input mode is embedding but no embeddings path was saved. "
            "Pass --embeddings-path."
        )
    if model_name in PROTEIN_DENSE_MODEL_NAMES and tf_embeddings_path is None:
        raise ValueError(
            f"Checkpoint/model is {model_name} but no protein embedding path "
            "was saved. Pass --tf-embeddings-path."
        )

    return {
        "input_mode": input_mode,
        "label_mode": label_mode,
        "model_name": model_name,
        "output_channels": saved_config.get("output_channels"),
        "hidden_channels": hidden_channels,
        "kernel_size": kernel_size,
        "dropout": float(dropout),
        "dilations": parse_dilations(dilations_raw),
        "tf_embedding_dropout": tf_embedding_dropout,
        "cross_attention_gate_logit_init": cross_attention_gate_logit_init,
        "cross_attention_context_pool_sizes": parse_dilations(
            cross_attention_context_pool_sizes_raw
        ),
        "dna_attention_window_bp": dna_attention_window_bp,
        "dna_attention_layers": dna_attention_layers,
        "dna_attention_heads": dna_attention_heads,
        "dna_attention_ffn_multiplier": dna_attention_ffn_multiplier,
        "motif_kernel_sizes": parse_int_tuple(motif_kernel_sizes_raw),
        "protein_noise_std": protein_noise_std,
        "protein_l2_normalize": protein_l2_normalize,
        "protein_delta_gate_logit_init": protein_delta_gate_logit_init,
        "scorer": scorer,
        "scorer_heads": scorer_heads,
        "scorer_pair_dim": scorer_pair_dim,
        "scorer_hidden_dim": scorer_hidden_dim,
        "scorer_bias_mode": scorer_bias_mode,
        "window_pooling": window_pooling,
        "window_pooling_top_k": window_pooling_top_k,
        "sites_path": sites_path,
        "regions_path": regions_path,
        "embeddings_path": embeddings_path,
        "embedding_column": embedding_column,
        "embedding_key_column": embedding_key_column,
        "tf_embeddings_path": tf_embeddings_path,
        "tf_embedding_key_column": tf_embedding_key_column,
        "tf_embedding_column": tf_embedding_column,
        "tf_name_map": tf_name_map,
        "min_sites_per_tf": min_sites_per_tf,
        "sequence_orientation": sequence_orientation,
        "trim_terminal_atg": trim_terminal_atg,
    }


def build_dataset(
    config: dict[str, object],
    max_regions: int | None,
    tf_name_filter: set[str] | None = None,
):
    common = {
        "sites_path": config["sites_path"],
        "regions_path": config["regions_path"],
        "min_sites_per_tf": int(config["min_sites_per_tf"]),
        "sequence_orientation": str(config["sequence_orientation"]),
        "tf_name_filter": tf_name_filter,
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


def infer_input_channels(dataset, input_mode: str) -> int:
    sample = dataset[0]["x"]
    if input_mode == "embedding":
        return int(sample.shape[-1])
    return int(sample.shape[0])


def resolve_output_feature_names(
    *,
    config: dict[str, object],
    checkpoint_tf_names: list[object],
) -> list[str]:
    output_channels_raw = config.get("output_channels")
    output_channels = (
        int(output_channels_raw)
        if output_channels_raw is not None
        else len(checkpoint_tf_names)
    )
    if output_channels <= 0:
        raise ValueError(f"Invalid output channel count: {output_channels}")

    label_mode = str(config.get("label_mode", "tf"))
    if label_mode == "merged_train_tfs" and output_channels == 1:
        return ["merged_train_tfs"]
    if (
        label_mode == "tf_and_merged_train_tfs"
        and output_channels == len(checkpoint_tf_names) + 1
    ):
        return [str(name) for name in checkpoint_tf_names] + ["merged_train_tfs"]
    if output_channels == len(checkpoint_tf_names):
        return [str(name) for name in checkpoint_tf_names]
    if output_channels == 1:
        return ["feature_0"]
    return [f"feature_{idx}" for idx in range(output_channels)]


def record_to_region(record: PromoterRecord) -> dict[str, object]:
    return {
        "chrom": record.chrom,
        "start": record.start,
        "end": record.end,
        "strand": record.strand,
        "gene_id": record.gene_id,
    }


def genomic_positions_for_record(record: PromoterRecord, sequence_orientation: str) -> np.ndarray:
    if sequence_orientation == "strand-aware" and record.strand == "-":
        return np.arange(record.model_end - 1, record.model_start - 1, -1, dtype=np.int64)
    return np.arange(record.model_start, record.model_end, dtype=np.int64)


def score_batch(
    *,
    model: torch.nn.Module,
    batch: dict[str, object],
    device: torch.device,
    input_mode: str,
    score_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    x = prepare_x(batch["x"].to(device, non_blocking=True), input_mode)
    mask = batch["mask"].detach().cpu().numpy().astype(bool, copy=False)
    with torch.no_grad():
        outputs = model(x)
        logits = outputs["logits"] if isinstance(outputs, dict) else outputs
        scores = logits.sigmoid() if score_mode == "sigmoid" else logits
    scores_np = scores.detach().cpu().numpy().astype(np.float32, copy=False)
    return np.transpose(scores_np, (0, 2, 1)), mask


def update_topk_for_batch(
    *,
    topk: TopKBuffer,
    scores: np.ndarray,
    mask: np.ndarray,
    records: list[PromoterRecord],
    record_offset: int,
    chrom_to_code: dict[str, int],
    chrom_names: list[str],
    sequence_orientation: str,
) -> int:
    candidate_scores = []
    candidate_chrom_codes = []
    candidate_region_codes = []
    candidate_starts = []

    for local_idx, record in enumerate(records):
        chrom_code = chrom_to_code.setdefault(record.chrom, len(chrom_names))
        if chrom_code == len(chrom_names):
            chrom_names.append(record.chrom)

        valid_offsets = np.flatnonzero(mask[local_idx])
        if len(valid_offsets) == 0:
            continue
        starts = genomic_positions_for_record(record, sequence_orientation)[valid_offsets]

        candidate_scores.append(scores[local_idx, valid_offsets, :])
        candidate_chrom_codes.append(
            np.full(len(valid_offsets), chrom_code, dtype=np.int32)
        )
        candidate_region_codes.append(
            np.full(len(valid_offsets), record_offset + local_idx, dtype=np.int32)
        )
        candidate_starts.append(starts)

    if not candidate_scores:
        return 0

    flat_scores = np.concatenate(candidate_scores, axis=0)
    flat_chrom_codes = np.concatenate(candidate_chrom_codes, axis=0)
    flat_region_codes = np.concatenate(candidate_region_codes, axis=0)
    flat_starts = np.concatenate(candidate_starts, axis=0)
    topk.update(flat_scores, flat_chrom_codes, flat_region_codes, flat_starts)
    return int(flat_scores.shape[0])


def write_outputs(
    *,
    predictions_out: Path,
    feature_ranks_out: Path,
    topk: TopKBuffer,
    tf_names: list[str],
    chrom_names: list[str],
    records: list[PromoterRecord],
    nms_radius_bp: int,
    top_k_per_tf: int,
    feature_rank_method: str,
    peak_caller: str,
) -> None:
    prediction_frames = []
    feature_scores = []

    for feature_idx, tf_name in enumerate(tf_names):
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
                max_keep=top_k_per_tf,
            )
        elif peak_caller == "scipy-find-peaks":
            keep = call_peaks_indices(
                chrom_codes=chrom_codes_raw,
                starts=starts_raw,
                scores=scores_raw,
                radius_bp=nms_radius_bp,
                max_keep=top_k_per_tf,
            )
        else:
            raise ValueError(f"Unknown peak caller: {peak_caller}")
        starts = starts_raw[keep]
        chrom_codes = chrom_codes_raw[keep]
        region_codes = region_codes_raw[keep]
        sorted_scores = scores_raw[keep]

        if feature_rank_method == "max-score":
            feature_score = float(sorted_scores[0])
        elif feature_rank_method == "mean-top-score":
            feature_score = float(np.mean(sorted_scores))
        elif feature_rank_method == "hit-count":
            feature_score = float(len(sorted_scores))
        elif feature_rank_method == "tf-name":
            feature_score = 0.0
        else:
            raise ValueError(f"Unknown feature-rank method: {feature_rank_method}")
        feature_scores.append((tf_name, feature_score))

        region_rows = [record_to_region(records[int(code)]) for code in region_codes]
        prediction_frames.append(
            pl.DataFrame(
                {
                    "chrom": [chrom_names[int(code)] for code in chrom_codes],
                    "start": starts,
                    "end": starts + 1,
                    "feature_idx": [tf_name] * len(starts),
                    "score": sorted_scores,
                    "strand": ["."] * len(starts),
                    "gene_id": [str(region["gene_id"]) for region in region_rows],
                    "sequence_name": [
                        sequence_name_for_region(region) for region in region_rows
                    ],
                    "region_start": [int(region["start"]) for region in region_rows],
                    "region_end": [int(region["end"]) for region in region_rows],
                    "region_strand": [str(region["strand"]) for region in region_rows],
                }
            )
        )

    if not prediction_frames:
        raise RuntimeError("No predictions were produced")

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
        .sort(
            ["feature_idx"]
            if feature_rank_method == "tf-name"
            else ["feature_score", "feature_idx"],
            descending=[False] if feature_rank_method == "tf-name" else [True, False],
        )
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


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.top_k_per_tf <= 0:
        raise ValueError("--top-k-per-tf must be positive")
    if args.nms_radius_bp < 0:
        raise ValueError("--nms-radius-bp must be non-negative")
    if args.pre_nms_factor <= 0:
        raise ValueError("--pre-nms-factor must be positive")

    device = get_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    tf_names = checkpoint.get("tf_names")
    if not isinstance(tf_names, list) or not tf_names:
        raise ValueError(f"Checkpoint has no tf_names list: {args.checkpoint}")

    config = resolve_export_config(args, checkpoint)
    checkpoint_tf_filter = {str(name).upper() for name in tf_names}
    dataset = build_dataset(config, args.max_regions, tf_name_filter=checkpoint_tf_filter)
    if [str(name) for name in dataset.tf_names] != [str(name) for name in tf_names]:
        raise ValueError(
            "Dataset TF order does not match checkpoint TF order. "
            "Export with the same sites/min-sites settings used for training."
        )
    feature_names = resolve_output_feature_names(
        config=config,
        checkpoint_tf_names=tf_names,
    )

    input_channels = int(
        checkpoint_model_config(checkpoint).get(
            "input_channels",
            infer_input_channels(dataset, str(config["input_mode"])),
        )
    )
    tf_embeddings = None
    if config["model_name"] in PROTEIN_DENSE_MODEL_NAMES:
        tf_embeddings, _ = load_tf_embeddings(
            config["tf_embeddings_path"],
            tf_names,
            key_column=config["tf_embedding_key_column"],
            embedding_column=str(config["tf_embedding_column"]),
            name_mapping_path=config["tf_name_map"],
        )
    model = build_dense_model(
        str(config["model_name"]),
        n_tfs=len(feature_names),
        input_channels=input_channels,
        tf_embeddings=tf_embeddings,
        hidden_channels=int(config["hidden_channels"]),
        kernel_size=int(config["kernel_size"]),
        dropout=float(config["dropout"]),
        dilations=config["dilations"],
        tf_embedding_dropout=float(config["tf_embedding_dropout"]),
        cross_attention_gate_logit_init=float(
            config["cross_attention_gate_logit_init"]
        ),
        cross_attention_context_pool_sizes=config[
            "cross_attention_context_pool_sizes"
        ],
        dna_attention_window_bp=int(config["dna_attention_window_bp"]),
        dna_attention_layers=int(config["dna_attention_layers"]),
        dna_attention_heads=int(config["dna_attention_heads"]),
        dna_attention_ffn_multiplier=float(config["dna_attention_ffn_multiplier"]),
        motif_kernel_sizes=config["motif_kernel_sizes"],
        protein_noise_std=float(config["protein_noise_std"]),
        protein_l2_normalize=bool(config["protein_l2_normalize"]),
        protein_delta_gate_logit_init=float(config["protein_delta_gate_logit_init"]),
        scorer=str(config["scorer"]),
        scorer_heads=int(config["scorer_heads"]),
        scorer_pair_dim=int(config["scorer_pair_dim"]),
        scorer_hidden_dim=int(config["scorer_hidden_dim"]),
        scorer_bias_mode=str(config["scorer_bias_mode"]),
        window_pooling=str(config["window_pooling"]),
        window_pooling_top_k=int(config["window_pooling_top_k"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    candidate_k = (
        args.top_k_per_tf
        if args.nms_radius_bp == 0
        else args.top_k_per_tf * args.pre_nms_factor
    )
    topk = TopKBuffer(n_features=len(feature_names), k=candidate_k)
    chrom_to_code: dict[str, int] = {}
    chrom_names: list[str] = []

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=partial(
            collate_promoter_batch,
            input_mode=str(config["input_mode"]),
        ),
    )
    n_scored_positions = 0

    print(f"Checkpoint:           {args.checkpoint}")
    print(f"Sites:                {config['sites_path']}")
    print(f"Regions:              {config['regions_path']}")
    print(f"Device:               {device}")
    print(f"Model:                {config['model_name']}")
    print(f"Input mode:           {config['input_mode']}")
    if config["model_name"] in PROTEIN_DENSE_MODEL_NAMES:
        print(f"TF embeddings:        {config['tf_embeddings_path']}")
    print(f"Dataset TF labels:    {len(tf_names)}")
    print(f"Model output features:{len(feature_names)}")
    print(f"Promoters:            {len(dataset)}")
    print(f"Top K per TF:         {args.top_k_per_tf}")
    print(f"Pre-NMS candidates:   {candidate_k}")
    print(f"NMS radius bp:        {args.nms_radius_bp}")
    print(f"Peak caller:          {args.peak_caller}")
    print(f"Score mode:           {args.score_mode}")
    print(f"Feature rank method:  {args.feature_rank_method}")

    for batch_idx, batch in enumerate(tqdm(loader, desc="Scoring promoters")):
        record_offset = batch_idx * args.batch_size
        records = dataset.records[record_offset : record_offset + len(batch["x"])]
        scores, mask = score_batch(
            model=model,
            batch=batch,
            device=device,
            input_mode=str(config["input_mode"]),
            score_mode=args.score_mode,
        )
        n_scored_positions += update_topk_for_batch(
            topk=topk,
            scores=scores,
            mask=mask,
            records=records,
            record_offset=record_offset,
            chrom_to_code=chrom_to_code,
            chrom_names=chrom_names,
            sequence_orientation=str(config["sequence_orientation"]),
        )

    print(f"Scored positions:     {n_scored_positions:,}")
    write_outputs(
        predictions_out=args.predictions_out,
        feature_ranks_out=args.feature_ranks_out,
        topk=topk,
        tf_names=feature_names,
        chrom_names=chrom_names,
        records=dataset.records,
        nms_radius_bp=args.nms_radius_bp,
        top_k_per_tf=args.top_k_per_tf,
        feature_rank_method=args.feature_rank_method,
        peak_caller=args.peak_caller,
    )

    metadata = {
        "checkpoint": str(args.checkpoint),
        "sites_path": str(config["sites_path"]),
        "regions_path": str(config["regions_path"]),
        "input_mode": config["input_mode"],
        "embeddings_path": str(config["embeddings_path"]) if config["embeddings_path"] else None,
        "embedding_column": config["embedding_column"],
        "embedding_key_column": config["embedding_key_column"],
        "sequence_orientation": config["sequence_orientation"],
        "trim_terminal_atg": config["trim_terminal_atg"],
        "top_k_per_tf": args.top_k_per_tf,
        "pre_nms_factor": args.pre_nms_factor,
        "candidate_k_per_tf": candidate_k,
        "nms_radius_bp": args.nms_radius_bp,
        "peak_caller": args.peak_caller,
        "score_mode": args.score_mode,
        "feature_rank_method": args.feature_rank_method,
        "n_scored_positions": n_scored_positions,
        "n_tfs": len(tf_names),
        "n_output_features": len(feature_names),
        "feature_names": feature_names,
        "model_config": {
            **config,
            "sites_path": str(config["sites_path"]),
            "regions_path": str(config["regions_path"]),
            "embeddings_path": str(config["embeddings_path"]) if config["embeddings_path"] else None,
            "tf_embeddings_path": (
                str(config["tf_embeddings_path"])
                if config["tf_embeddings_path"]
                else None
            ),
            "tf_name_map": str(config["tf_name_map"]) if config["tf_name_map"] else None,
            "dilations": list(config["dilations"]),
            "cross_attention_context_pool_sizes": list(
                config["cross_attention_context_pool_sizes"]
            ),
        },
    }
    metadata_path = args.predictions_out.with_suffix(".metadata.json")
    with metadata_path.open("w") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Wrote metadata: {metadata_path}")


if __name__ == "__main__":
    main()
