from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

try:
    from .dataset import (
        BindingBenchPromoterEmbeddingDataset,
        BindingBenchPromoterSequenceDataset,
        DEFAULT_REGIONS_PATH,
        DEFAULT_SITES_PATH,
    )
    from .model import DENSE_MODEL_NAMES, build_dense_model, parse_dilations, parse_int_tuple
    from .promoter_splits import (
        load_promoter_split,
        make_all_train_split,
        make_chromosome_promoter_split,
        make_random_promoter_split,
        normalize_promoter_split,
        save_promoter_split,
        split_indices,
    )
    from .tf_embeddings import available_embedding_keys, load_tf_embeddings
    from .tf_splits import (
        load_tf_split,
        make_all_train_tf_split,
        make_named_holdout_tf_split,
        make_named_similarity_holdout_tf_split,
        make_random_tf_split,
        make_similarity_holdout_tf_split,
        normalize_tf_split,
        save_tf_split,
        tf_split_indices,
    )
except ImportError:
    from dataset import (
        BindingBenchPromoterEmbeddingDataset,
        BindingBenchPromoterSequenceDataset,
        DEFAULT_REGIONS_PATH,
        DEFAULT_SITES_PATH,
    )
    from model import DENSE_MODEL_NAMES, build_dense_model, parse_dilations, parse_int_tuple
    from promoter_splits import (
        load_promoter_split,
        make_all_train_split,
        make_chromosome_promoter_split,
        make_random_promoter_split,
        normalize_promoter_split,
        save_promoter_split,
        split_indices,
    )
    from tf_embeddings import available_embedding_keys, load_tf_embeddings
    from tf_splits import (
        load_tf_split,
        make_all_train_tf_split,
        make_named_holdout_tf_split,
        make_named_similarity_holdout_tf_split,
        make_random_tf_split,
        make_similarity_holdout_tf_split,
        normalize_tf_split,
        save_tf_split,
        tf_split_indices,
    )


DEFAULT_MODEL_ROOT = Path(
    "/s/project/ml4rg_students/2026/project15/working/"
    "supervised_baseline/models"
)
PROTEIN_DENSE_MODEL_NAMES = (
    "dense_protein_res_dilated_cnn",
    "dense_protein_local_attention",
    "dense_protein_motif_cnn",
    "dense_protein_res_dilated_crossattention",
    "dense_transbind_cnn_lstm_attention",
)


@dataclass
class DenseEpochStats:
    epoch: int
    train_loss: float
    micro_precision: float
    micro_recall: float
    micro_f1: float
    predicted_positive_rate: float
    lr: float
    val_loss: float | None = None
    val_micro_precision: float | None = None
    val_micro_recall: float | None = None
    val_micro_f1: float | None = None
    val_predicted_positive_rate: float | None = None
    val_average_precision: float | None = None
    val_roc_auc: float | None = None


@dataclass
class DenseSplitStats:
    loss: float
    micro_precision: float
    micro_recall: float
    micro_f1: float
    predicted_positive_rate: float
    n_examples: int
    average_precision: float | None = None
    roc_auc: float | None = None
    per_tf: list[dict[str, object]] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train promoter-level dense TF binding baselines."
    )
    parser.add_argument("--sites-path", type=Path, default=DEFAULT_SITES_PATH)
    parser.add_argument("--regions-path", type=Path, default=DEFAULT_REGIONS_PATH)
    parser.add_argument(
        "--input-mode",
        choices=("raw", "embedding"),
        default="raw",
        help="Use one-hot promoter sequence or precomputed position embeddings.",
    )
    parser.add_argument(
        "--embeddings-path",
        type=Path,
        help="Required for --input-mode embedding. Supports .npy [N,L,D] or parquet.",
    )
    parser.add_argument("--embedding-column", default="emb")
    parser.add_argument(
        "--embedding-key-column",
        help="Optional parquet key column. If omitted, embeddings are row-aligned.",
    )
    parser.add_argument(
        "--tf-embeddings-path",
        type=Path,
        help="Parquet file with one protein embedding row per TF for dense protein models.",
    )
    parser.add_argument(
        "--tf-embedding-key-column",
        help="Column used to match dataset TF labels to protein embedding rows. Defaults to auto.",
    )
    parser.add_argument(
        "--tf-embedding-column",
        default="emb",
        help="Column containing list-valued TF protein embeddings.",
    )
    parser.add_argument(
        "--tf-name-map",
        type=Path,
        help="Optional JSON mapping from dataset TF label to embedding-table key.",
    )
    parser.add_argument(
        "--drop-missing-tf-embeddings",
        action="store_true",
        help=(
            "For dense protein models, filter the dataset to TF labels that have "
            "protein embeddings instead of failing on missing labels."
        ),
    )
    parser.add_argument(
        "--tf-name-filter-from-embeddings",
        action="store_true",
        help=(
            "Filter any dense model to TF labels present in the protein embedding "
            "table. Useful for fair DNA-only comparisons against protein models."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for checkpoints and metadata.",
    )
    parser.add_argument("--min-sites-per-tf", type=int, default=15)
    parser.add_argument("--max-regions", type=int)
    parser.add_argument(
        "--sequence-orientation",
        choices=("strand-aware", "genomic"),
        default="strand-aware",
    )
    parser.add_argument(
        "--include-terminal-atg",
        action="store_true",
        help=(
            "Keep the terminal ATG from upstream_ATG_1000 sequence mappers. "
            "By default, dense promoter baselines trim it and train on 1000 bp promoters."
        ),
    )
    parser.add_argument(
        "--label-smoothing-mode",
        choices=("hard", "hard-dilate", "linear", "gaussian"),
        default="hard",
        help=(
            "Training target used for dense labels. 'hard' keeps exact BindingBench "
            "intervals; the other modes soften labels within --label-smoothing-radius-bp. "
            "Evaluation metrics still use the original hard intervals."
        ),
    )
    parser.add_argument(
        "--label-smoothing-radius-bp",
        type=int,
        default=0,
        help="Radius in bp around each true binding interval for softened training labels.",
    )
    parser.add_argument(
        "--label-smoothing-sigma-bp",
        type=float,
        help=(
            "Gaussian sigma in bp for --label-smoothing-mode gaussian. "
            "Defaults to half the radius."
        ),
    )
    parser.add_argument(
        "--model",
        choices=DENSE_MODEL_NAMES,
        default="dense_small_cnn",
    )
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--kernel-size", type=int, default=7)
    parser.add_argument("--dilations", default="1,2,4,8,16")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--tf-embedding-dropout",
        type=float,
        default=0.0,
        help=(
            "Drop probability applied to TF protein embeddings during training "
            "for dense_protein_res_dilated_crossattention."
        ),
    )
    parser.add_argument(
        "--cross-attention-gate-logit-init",
        type=float,
        default=-3.0,
        help=(
            "Initial logit for the residual CrossAttention score gate. "
            "The actual initial scale is sigmoid(value)."
        ),
    )
    parser.add_argument(
        "--cross-attention-context-pool-sizes",
        default="4",
        help=(
            "Comma-separated MaxPool sizes used to build CrossAttention DNA "
            "context tokens, e.g. '2,4,8'."
        ),
    )
    parser.add_argument(
        "--dna-attention-window-bp",
        type=int,
        default=50,
        help=(
            "Local self-attention radius in bp for dense_protein_local_attention. "
            "Use 0 for full promoter self-attention."
        ),
    )
    parser.add_argument(
        "--dna-attention-layers",
        type=int,
        default=2,
        help="Number of local DNA self-attention blocks.",
    )
    parser.add_argument(
        "--dna-attention-heads",
        type=int,
        default=8,
        help="Number of heads in each local DNA self-attention block.",
    )
    parser.add_argument(
        "--dna-attention-ffn-multiplier",
        type=float,
        default=4.0,
        help="Feed-forward width multiplier inside local DNA self-attention blocks.",
    )
    parser.add_argument(
        "--motif-kernel-sizes",
        default="7,11,15",
        help=(
            "Comma-separated odd convolution widths for dense_protein_motif_cnn's "
            "raw motif stem."
        ),
    )
    parser.add_argument(
        "--protein-noise-std",
        type=float,
        default=0.0,
        help="Gaussian noise std added to TF embeddings during training.",
    )
    parser.add_argument(
        "--protein-l2-normalize",
        action="store_true",
        help="L2-normalize adapted TF vectors before scoring.",
    )
    parser.add_argument(
        "--scorer",
        choices=("multihead_bilinear", "mlp"),
        default="multihead_bilinear",
        help="TF-DNA interaction scorer for dense_protein_local_attention.",
    )
    parser.add_argument(
        "--scorer-heads",
        type=int,
        default=8,
        help="Interaction heads for the multi-head bilinear scorer.",
    )
    parser.add_argument(
        "--scorer-pair-dim",
        type=int,
        default=32,
        help="Per-head dimension for bilinear scorer or pair embedding dim for MLP scorer.",
    )
    parser.add_argument(
        "--scorer-hidden-dim",
        type=int,
        default=128,
        help="Hidden dimension for the MLP scorer.",
    )
    parser.add_argument(
        "--scorer-bias-mode",
        choices=("none", "tf", "dna", "both"),
        default="tf",
        help="Optional additive bias terms in the protein-conditioned scorer.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--lr-scheduler",
        choices=("none", "cosine", "plateau"),
        default="cosine",
    )
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--plateau-patience", type=int, default=10)
    parser.add_argument("--plateau-factor", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--promoter-split-mode",
        choices=("none", "random", "chromosome"),
        default="none",
        help=(
            "Split promoters into train/val/test subsets. Use none for the old "
            "all-promoters training behavior."
        ),
    )
    parser.add_argument(
        "--promoter-split-path",
        type=Path,
        help="Load an existing promoter split JSON. Takes precedence over split mode.",
    )
    parser.add_argument(
        "--promoter-split-out",
        type=Path,
        help="Where to save the resolved promoter split JSON. Defaults to output_dir/promoter_split.json.",
    )
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument(
        "--val-chroms",
        default="",
        help="Comma-separated validation chromosomes for --promoter-split-mode chromosome.",
    )
    parser.add_argument(
        "--test-chroms",
        default="",
        help="Comma-separated test chromosomes for --promoter-split-mode chromosome.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="Evaluate validation split every N epochs. Use 0 to disable epoch validation.",
    )
    parser.add_argument(
        "--tf-split-mode",
        choices=("none", "random", "named", "similarity", "named_similarity"),
        default="none",
        help=(
            "Split TF labels into train/val/test subsets. Meaningful only for "
            "protein-conditioned dense models."
        ),
    )
    parser.add_argument(
        "--tf-split-path",
        type=Path,
        help="Load an existing TF split JSON. Takes precedence over --tf-split-mode.",
    )
    parser.add_argument(
        "--tf-split-out",
        type=Path,
        help="Where to save the resolved TF split JSON. Defaults to output_dir/tf_split.json.",
    )
    parser.add_argument("--tf-train-fraction", type=float, default=0.7)
    parser.add_argument("--tf-val-fraction", type=float, default=0.15)
    parser.add_argument(
        "--tf-val-names",
        default="",
        help="Comma-separated validation TF names for --tf-split-mode named.",
    )
    parser.add_argument(
        "--tf-test-names",
        default="",
        help="Comma-separated test TF names for --tf-split-mode named.",
    )
    parser.add_argument(
        "--tf-similarity-threshold",
        type=float,
        default=0.9,
        help=(
            "Cosine-similarity threshold for --tf-split-mode similarity. TFs with "
            "protein embedding cosine similarity at or above this value are kept "
            "in the same split."
        ),
    )
    parser.add_argument(
        "--no-pos-weight",
        action="store_true",
        help="Disable positive-class weighting in dense BCE.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=(
            "val_loss",
            "val_average_precision",
            "val_roc_auc",
            "val_micro_f1",
            "train_loss",
        ),
        default="val_loss",
        help="Metric used to choose best.pt. Loss is minimized; AP/AUROC/F1 are maximized.",
    )
    parser.add_argument("--save-every", type=int, default=10)
    args = parser.parse_args()
    if args.input_mode == "embedding" and args.embeddings_path is None:
        raise ValueError("--embeddings-path is required for --input-mode embedding")
    if args.model in PROTEIN_DENSE_MODEL_NAMES and args.tf_embeddings_path is None:
        raise ValueError(
            f"--tf-embeddings-path is required for --model {args.model}"
        )
    if (
        args.drop_missing_tf_embeddings or args.tf_name_filter_from_embeddings
    ) and args.tf_embeddings_path is None:
        raise ValueError(
            "--tf-embeddings-path is required when filtering TFs to available embeddings"
        )
    if args.tf_split_mode != "none" and args.model not in PROTEIN_DENSE_MODEL_NAMES:
        raise ValueError("TF holdout splits are only meaningful for dense protein models")
    if args.eval_every < 0:
        raise ValueError("--eval-every must be non-negative")
    if not 0.0 <= args.tf_embedding_dropout < 1.0:
        raise ValueError("--tf-embedding-dropout must be in [0, 1)")
    if args.dna_attention_window_bp < 0:
        raise ValueError("--dna-attention-window-bp must be non-negative")
    if args.dna_attention_layers <= 0:
        raise ValueError("--dna-attention-layers must be positive")
    if args.dna_attention_heads <= 0:
        raise ValueError("--dna-attention-heads must be positive")
    if args.dna_attention_ffn_multiplier <= 0:
        raise ValueError("--dna-attention-ffn-multiplier must be positive")
    if args.protein_noise_std < 0:
        raise ValueError("--protein-noise-std must be non-negative")
    if args.scorer_heads <= 0:
        raise ValueError("--scorer-heads must be positive")
    if args.scorer_pair_dim <= 0:
        raise ValueError("--scorer-pair-dim must be positive")
    if args.scorer_hidden_dim <= 0:
        raise ValueError("--scorer-hidden-dim must be positive")
    if args.label_smoothing_radius_bp < 0:
        raise ValueError("--label-smoothing-radius-bp must be non-negative")
    if args.label_smoothing_sigma_bp is not None and args.label_smoothing_sigma_bp <= 0:
        raise ValueError("--label-smoothing-sigma-bp must be positive")
    if args.label_smoothing_mode != "hard" and args.label_smoothing_radius_bp <= 0:
        raise ValueError(
            "--label-smoothing-radius-bp must be positive when label smoothing is enabled"
        )
    parse_dilations(args.cross_attention_context_pool_sizes)
    parse_int_tuple(args.motif_kernel_sizes)
    if args.output_dir is None:
        args.output_dir = DEFAULT_MODEL_ROOT / f"promoter_{args.input_mode}_{args.model}"
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def serializable_args(args: argparse.Namespace) -> dict[str, object]:
    out = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def build_dataset(
    args: argparse.Namespace,
    tf_name_filter: set[str] | None = None,
):
    common = {
        "sites_path": args.sites_path,
        "regions_path": args.regions_path,
        "min_sites_per_tf": args.min_sites_per_tf,
        "sequence_orientation": args.sequence_orientation,
        "tf_name_filter": tf_name_filter,
        "max_regions": args.max_regions,
        "trim_terminal_atg": not args.include_terminal_atg,
        "label_smoothing_mode": args.label_smoothing_mode,
        "label_smoothing_radius_bp": args.label_smoothing_radius_bp,
        "label_smoothing_sigma_bp": args.label_smoothing_sigma_bp,
    }
    if args.input_mode == "raw":
        return BindingBenchPromoterSequenceDataset(**common)
    return BindingBenchPromoterEmbeddingDataset(
        args.embeddings_path,
        embedding_column=args.embedding_column,
        key_column=args.embedding_key_column,
        **common,
    )


def build_promoter_split(args: argparse.Namespace, dataset) -> dict[str, object]:
    if args.promoter_split_path is not None:
        raw_split = load_promoter_split(args.promoter_split_path)
    elif args.promoter_split_mode == "none":
        raw_split = make_all_train_split(dataset.records)
    elif args.promoter_split_mode == "random":
        raw_split = make_random_promoter_split(
            dataset.records,
            seed=args.seed,
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
        )
    elif args.promoter_split_mode == "chromosome":
        raw_split = make_chromosome_promoter_split(
            dataset.records,
            val_chroms=args.val_chroms,
            test_chroms=args.test_chroms,
        )
    else:
        raise ValueError(f"Unknown promoter split mode: {args.promoter_split_mode}")
    return normalize_promoter_split(raw_split, dataset.records)


def build_tf_split(
    args: argparse.Namespace,
    dataset,
    tf_embeddings: torch.Tensor | None = None,
) -> dict[str, object]:
    if args.tf_split_path is not None:
        raw_split = load_tf_split(args.tf_split_path)
    elif args.tf_split_mode == "none":
        raw_split = make_all_train_tf_split(dataset.tf_names)
    elif args.tf_split_mode == "random":
        raw_split = make_random_tf_split(
            dataset.tf_names,
            seed=args.seed,
            train_fraction=args.tf_train_fraction,
            val_fraction=args.tf_val_fraction,
        )
    elif args.tf_split_mode == "named":
        raw_split = make_named_holdout_tf_split(
            dataset.tf_names,
            val_tfs=args.tf_val_names,
            test_tfs=args.tf_test_names,
        )
    elif args.tf_split_mode == "similarity":
        if tf_embeddings is None:
            raise ValueError("--tf-split-mode similarity requires TF protein embeddings")
        raw_split = make_similarity_holdout_tf_split(
            dataset.tf_names,
            tf_embeddings.detach().cpu().numpy(),
            seed=args.seed,
            train_fraction=args.tf_train_fraction,
            val_fraction=args.tf_val_fraction,
            similarity_threshold=args.tf_similarity_threshold,
        )
    elif args.tf_split_mode == "named_similarity":
        if tf_embeddings is None:
            raise ValueError(
                "--tf-split-mode named_similarity requires TF protein embeddings"
            )
        raw_split = make_named_similarity_holdout_tf_split(
            dataset.tf_names,
            tf_embeddings.detach().cpu().numpy(),
            seed=args.seed,
            val_tfs=args.tf_val_names,
            test_tfs=args.tf_test_names,
            train_fraction=args.tf_train_fraction,
            val_fraction=args.tf_val_fraction,
            similarity_threshold=args.tf_similarity_threshold,
        )
    else:
        raise ValueError(f"Unknown TF split mode: {args.tf_split_mode}")
    return normalize_tf_split(raw_split, dataset.tf_names)


def make_split_loader(
    dataset,
    indices: list[int],
    *,
    input_mode: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    shuffle: bool,
) -> DataLoader | None:
    if not indices:
        return None
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=partial(collate_promoter_batch, input_mode=input_mode),
    )


def tensor_indices(indices: list[int], device: torch.device) -> torch.Tensor | None:
    if not indices:
        return None
    return torch.tensor(indices, dtype=torch.long, device=device)


def infer_input_channels(dataset, input_mode: str) -> int:
    sample = dataset[0]["x"]
    if not isinstance(sample, torch.Tensor):
        raise TypeError("Dataset x must be a torch.Tensor")
    if input_mode == "raw":
        return int(sample.shape[0])
    return int(sample.shape[-1])


def prepare_x(x: torch.Tensor, input_mode: str) -> torch.Tensor:
    if input_mode == "embedding":
        return x.transpose(1, 2).contiguous()
    return x


def forward_dense_model(
    model: torch.nn.Module,
    x: torch.Tensor,
    tf_indices: torch.Tensor | None,
) -> torch.Tensor:
    if getattr(model, "supports_tf_indices", False):
        return model(x, tf_indices=tf_indices)
    logits = model(x)
    if tf_indices is not None:
        logits = logits.index_select(1, tf_indices)
    return logits


def select_tf_axis(
    tensor: torch.Tensor,
    tf_indices: torch.Tensor | None,
) -> torch.Tensor:
    if tf_indices is None:
        return tensor
    return tensor.index_select(1, tf_indices)


def collate_promoter_batch(
    items: list[dict[str, object]],
    input_mode: str,
) -> dict[str, object]:
    if not items:
        raise ValueError("Cannot collate an empty batch")

    y0 = items[0]["y"]
    if not isinstance(y0, torch.Tensor):
        raise TypeError("Dataset y must be a torch.Tensor")
    hard_y0 = items[0].get("hard_y", y0)
    if not isinstance(hard_y0, torch.Tensor):
        raise TypeError("Dataset hard_y must be a torch.Tensor")
    max_len = max(int(item["y"].shape[-1]) for item in items)
    n_tfs = int(y0.shape[0])

    y_batch = torch.zeros((len(items), n_tfs, max_len), dtype=torch.float32)
    hard_y_batch = torch.zeros((len(items), n_tfs, max_len), dtype=torch.float32)
    mask_batch = torch.zeros((len(items), max_len), dtype=torch.bool)

    if input_mode == "raw":
        x0 = items[0]["x"]
        if not isinstance(x0, torch.Tensor):
            raise TypeError("Dataset x must be a torch.Tensor")
        channels = int(x0.shape[0])
        x_batch = torch.zeros((len(items), channels, max_len), dtype=torch.float32)
        for idx, item in enumerate(items):
            x = item["x"]
            y = item["y"]
            hard_y = item.get("hard_y", y)
            mask = item["mask"]
            length = int(y.shape[-1])
            x_batch[idx, :, :length] = x
            y_batch[idx, :, :length] = y
            hard_y_batch[idx, :, :length] = hard_y
            mask_batch[idx, :length] = mask
    elif input_mode == "embedding":
        x0 = items[0]["x"]
        if not isinstance(x0, torch.Tensor):
            raise TypeError("Dataset x must be a torch.Tensor")
        embedding_dim = int(x0.shape[-1])
        x_batch = torch.zeros((len(items), max_len, embedding_dim), dtype=torch.float32)
        for idx, item in enumerate(items):
            x = item["x"]
            y = item["y"]
            hard_y = item.get("hard_y", y)
            mask = item["mask"]
            length = int(y.shape[-1])
            x_batch[idx, :length, :] = x
            y_batch[idx, :, :length] = y
            hard_y_batch[idx, :, :length] = hard_y
            mask_batch[idx, :length] = mask
    else:
        raise ValueError(f"Unsupported input mode: {input_mode}")

    return {
        "x": x_batch,
        "y": y_batch,
        "hard_y": hard_y_batch,
        "mask": mask_batch,
        "meta": [item["meta"] for item in items],
    }


def compute_dense_pos_weight(
    dataset,
    device: torch.device,
    indices: list[int] | None = None,
) -> torch.Tensor:
    positives = np.zeros(len(dataset.tf_names), dtype=np.float64)
    valid_positions = 0.0
    records = dataset.records if indices is None else [dataset.records[idx] for idx in indices]
    for record in records:
        mask = dataset._position_mask(record)
        valid_positions += float(mask.sum())
        labels = dataset._dense_labels(record)
        labels[:, ~mask] = 0.0
        positives += labels.sum(axis=1)

    negatives = valid_positions - positives
    weights = negatives / np.maximum(positives, 1.0)
    weights = np.clip(weights, 1.0, 100.0)
    return torch.tensor(weights, dtype=torch.float32, device=device).view(1, -1, 1)


def dense_bce_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: torch.Tensor | None,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=pos_weight,
        reduction="none",
    )
    valid = mask.unsqueeze(1).expand_as(loss)
    return loss[valid].mean()


def binary_average_precision(scores: np.ndarray, targets: np.ndarray) -> float | None:
    targets_bool = targets.astype(bool, copy=False)
    n_pos = int(targets_bool.sum())
    if n_pos == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    sorted_targets = targets_bool[order]
    tp_cumsum = np.cumsum(sorted_targets)
    ranks = np.arange(1, len(sorted_targets) + 1, dtype=np.float64)
    precision_at_pos = tp_cumsum[sorted_targets] / ranks[sorted_targets]
    return float(precision_at_pos.sum() / n_pos)


def binary_roc_auc(scores: np.ndarray, targets: np.ndarray) -> float | None:
    targets_bool = targets.astype(bool, copy=False)
    n_pos = int(targets_bool.sum())
    n_neg = int(len(targets_bool) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None

    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + 1 + end)
        start = end

    pos_rank_sum = ranks[targets_bool].sum()
    auc = (pos_rank_sum - (n_pos * (n_pos + 1) / 2.0)) / (n_pos * n_neg)
    return float(auc)


def score_metrics(
    score_chunks: list[np.ndarray],
    target_chunks: list[np.ndarray],
) -> tuple[float | None, float | None]:
    if not score_chunks:
        return None, None
    scores = np.concatenate(score_chunks).astype(np.float64, copy=False)
    targets = np.concatenate(target_chunks).astype(bool, copy=False)
    return binary_average_precision(scores, targets), binary_roc_auc(scores, targets)


def build_lr_scheduler(args: argparse.Namespace, optimizer: torch.optim.Optimizer):
    if args.lr_scheduler == "none":
        return None
    if args.lr_scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs,
            eta_min=args.min_lr,
        )
    if args.lr_scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=args.plateau_factor,
            patience=args.plateau_patience,
            min_lr=args.min_lr,
        )
    raise ValueError(f"Unknown LR scheduler: {args.lr_scheduler}")


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def model_config_from_args(args: argparse.Namespace, input_channels: int) -> dict[str, object]:
    return {
        "model_name": args.model,
        "input_mode": args.input_mode,
        "input_channels": input_channels,
        "hidden_channels": args.hidden_channels,
        "kernel_size": args.kernel_size,
        "dropout": args.dropout,
        "dilations": list(parse_dilations(args.dilations)),
        "tf_embedding_dropout": args.tf_embedding_dropout,
        "cross_attention_gate_logit_init": args.cross_attention_gate_logit_init,
        "cross_attention_context_pool_sizes": list(
            parse_dilations(args.cross_attention_context_pool_sizes)
        ),
        "dna_attention_window_bp": args.dna_attention_window_bp,
        "dna_attention_layers": args.dna_attention_layers,
        "dna_attention_heads": args.dna_attention_heads,
        "dna_attention_ffn_multiplier": args.dna_attention_ffn_multiplier,
        "motif_kernel_sizes": list(parse_int_tuple(args.motif_kernel_sizes)),
        "protein_noise_std": args.protein_noise_std,
        "protein_l2_normalize": args.protein_l2_normalize,
        "scorer": args.scorer,
        "scorer_heads": args.scorer_heads,
        "scorer_pair_dim": args.scorer_pair_dim,
        "scorer_hidden_dim": args.scorer_hidden_dim,
        "scorer_bias_mode": args.scorer_bias_mode,
        "embeddings_path": str(args.embeddings_path) if args.embeddings_path else None,
        "embedding_column": args.embedding_column,
        "embedding_key_column": args.embedding_key_column,
        "tf_embeddings_path": str(args.tf_embeddings_path) if args.tf_embeddings_path else None,
        "tf_embedding_key_column": args.tf_embedding_key_column,
        "tf_embedding_column": args.tf_embedding_column,
        "tf_name_map": str(args.tf_name_map) if args.tf_name_map else None,
        "drop_missing_tf_embeddings": args.drop_missing_tf_embeddings,
        "tf_name_filter_from_embeddings": args.tf_name_filter_from_embeddings,
        "trim_terminal_atg": not args.include_terminal_atg,
        "label_smoothing_mode": args.label_smoothing_mode,
        "label_smoothing_radius_bp": args.label_smoothing_radius_bp,
        "label_smoothing_sigma_bp": args.label_smoothing_sigma_bp,
    }


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    dataset,
    input_channels: int,
    stats: DenseEpochStats,
    promoter_split: dict[str, object] | None = None,
    tf_split: dict[str, object] | None = None,
    tf_embedding_metadata: dict[str, object] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_name": args.model,
            "model_config": model_config_from_args(args, input_channels),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": serializable_args(args),
            "tf_names": dataset.tf_names,
            "dataset_summary": dataset.summary(),
            "promoter_split": promoter_split,
            "tf_split": tf_split,
            "tf_embedding_metadata": tf_embedding_metadata,
            "stats": asdict(stats),
        },
        path,
    )


def train_one_epoch(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    input_mode: str,
    pos_weight: torch.Tensor | None,
    tf_indices: torch.Tensor | None,
    epoch: int,
) -> DenseEpochStats:
    model.train()
    total_loss = 0.0
    total_examples = 0
    tp = fp = fn = 0.0
    predicted_positive = 0.0
    total_labels = 0.0

    for batch in loader:
        x = prepare_x(batch["x"].to(device, non_blocking=True), input_mode)
        y = batch["y"].to(device, non_blocking=True)
        hard_y = batch["hard_y"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = forward_dense_model(model, x, tf_indices)
        y = select_tf_axis(y, tf_indices)
        hard_y = select_tf_axis(hard_y, tf_indices)
        batch_pos_weight = select_tf_axis(pos_weight, tf_indices) if pos_weight is not None else None
        if logits.shape != y.shape:
            raise ValueError(f"Logit/label shape mismatch: {logits.shape} vs {y.shape}")
        if logits.shape != hard_y.shape:
            raise ValueError(
                f"Logit/hard-label shape mismatch: {logits.shape} vs {hard_y.shape}"
            )
        loss = dense_bce_loss(logits, y, mask, batch_pos_weight)
        loss.backward()
        optimizer.step()

        batch_size = x.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_examples += batch_size

        with torch.no_grad():
            valid = mask.unsqueeze(1).expand_as(logits)
            pred = logits.sigmoid() >= 0.5
            target = hard_y >= 0.5
            tp += (pred & target & valid).sum().item()
            fp += (pred & ~target & valid).sum().item()
            fn += (~pred & target & valid).sum().item()
            predicted_positive += (pred & valid).sum().item()
            total_labels += valid.sum().item()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return DenseEpochStats(
        epoch=epoch,
        train_loss=total_loss / max(total_examples, 1),
        micro_precision=precision,
        micro_recall=recall,
        micro_f1=f1,
        predicted_positive_rate=predicted_positive / max(total_labels, 1),
        lr=current_lr(optimizer),
    )


def evaluate_dense(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    input_mode: str,
    pos_weight: torch.Tensor | None,
    tf_indices: torch.Tensor | None,
    tf_names: list[str] | None = None,
) -> DenseSplitStats:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    tp = fp = fn = 0.0
    predicted_positive = 0.0
    total_labels = 0.0
    micro_score_chunks: list[np.ndarray] = []
    micro_target_chunks: list[np.ndarray] = []
    per_tf_score_chunks: list[list[np.ndarray]] | None = None
    per_tf_target_chunks: list[list[np.ndarray]] | None = None
    selected_tf_names: list[str] | None = None

    if tf_names is not None:
        if tf_indices is None:
            selected_tf_names = [str(name) for name in tf_names]
        else:
            selected_tf_names = [
                str(tf_names[int(idx)])
                for idx in tf_indices.detach().cpu().numpy().tolist()
            ]
        per_tf_score_chunks = [[] for _ in selected_tf_names]
        per_tf_target_chunks = [[] for _ in selected_tf_names]

    with torch.no_grad():
        for batch in loader:
            x = prepare_x(batch["x"].to(device, non_blocking=True), input_mode)
            y = batch["y"].to(device, non_blocking=True)
            hard_y = batch["hard_y"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            logits = forward_dense_model(model, x, tf_indices)
            y = select_tf_axis(y, tf_indices)
            hard_y = select_tf_axis(hard_y, tf_indices)
            batch_pos_weight = (
                select_tf_axis(pos_weight, tf_indices)
                if pos_weight is not None
                else None
            )
            if logits.shape != y.shape:
                raise ValueError(f"Logit/label shape mismatch: {logits.shape} vs {y.shape}")
            if logits.shape != hard_y.shape:
                raise ValueError(
                    f"Logit/hard-label shape mismatch: {logits.shape} vs {hard_y.shape}"
                )
            loss = dense_bce_loss(logits, y, mask, batch_pos_weight)

            batch_size = x.shape[0]
            total_loss += float(loss.detach().cpu()) * batch_size
            total_examples += batch_size

            valid = mask.unsqueeze(1).expand_as(logits)
            pred = logits.sigmoid() >= 0.5
            target = hard_y >= 0.5
            tp += (pred & target & valid).sum().item()
            fp += (pred & ~target & valid).sum().item()
            fn += (~pred & target & valid).sum().item()
            predicted_positive += (pred & valid).sum().item()
            total_labels += valid.sum().item()

            logits_cpu = logits.detach().cpu().numpy()
            target_cpu = target.detach().cpu().numpy()
            valid_cpu = valid.detach().cpu().numpy()
            micro_score_chunks.append(logits_cpu[valid_cpu])
            micro_target_chunks.append(target_cpu[valid_cpu])
            if (
                per_tf_score_chunks is not None
                and per_tf_target_chunks is not None
            ):
                for local_tf_idx in range(logits_cpu.shape[1]):
                    tf_valid = valid_cpu[:, local_tf_idx, :]
                    per_tf_score_chunks[local_tf_idx].append(
                        logits_cpu[:, local_tf_idx, :][tf_valid]
                    )
                    per_tf_target_chunks[local_tf_idx].append(
                        target_cpu[:, local_tf_idx, :][tf_valid]
                    )

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    average_precision, roc_auc = score_metrics(micro_score_chunks, micro_target_chunks)
    per_tf_metrics = None
    if (
        selected_tf_names is not None
        and per_tf_score_chunks is not None
        and per_tf_target_chunks is not None
    ):
        per_tf_metrics = []
        for name, score_chunk, target_chunk in zip(
            selected_tf_names,
            per_tf_score_chunks,
            per_tf_target_chunks,
        ):
            tf_ap, tf_auc = score_metrics(score_chunk, target_chunk)
            n_pos = int(sum(chunk.astype(bool, copy=False).sum() for chunk in target_chunk))
            n_total = int(sum(len(chunk) for chunk in target_chunk))
            per_tf_metrics.append(
                {
                    "tf_name": name,
                    "average_precision": tf_ap,
                    "roc_auc": tf_auc,
                    "positives": n_pos,
                    "valid_positions": n_total,
                }
            )
    return DenseSplitStats(
        loss=total_loss / max(total_examples, 1),
        micro_precision=precision,
        micro_recall=recall,
        micro_f1=f1,
        predicted_positive_rate=predicted_positive / max(total_labels, 1),
        n_examples=total_examples,
        average_precision=average_precision,
        roc_auc=roc_auc,
        per_tf=per_tf_metrics,
    )


def attach_val_stats(stats: DenseEpochStats, val_stats: DenseSplitStats) -> DenseEpochStats:
    stats.val_loss = val_stats.loss
    stats.val_micro_precision = val_stats.micro_precision
    stats.val_micro_recall = val_stats.micro_recall
    stats.val_micro_f1 = val_stats.micro_f1
    stats.val_predicted_positive_rate = val_stats.predicted_positive_rate
    stats.val_average_precision = val_stats.average_precision
    stats.val_roc_auc = val_stats.roc_auc
    return stats


def selection_value(stats: DenseEpochStats, metric: str) -> float | None:
    value = getattr(stats, metric)
    return float(value) if value is not None else None


def selection_is_better(value: float, best_value: float | None, metric: str) -> bool:
    if best_value is None:
        return True
    if metric.endswith("_loss"):
        return value < best_value
    return value > best_value


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    dilations = parse_dilations(args.dilations)

    tf_name_filter = None
    if args.drop_missing_tf_embeddings or args.tf_name_filter_from_embeddings:
        tf_name_filter = available_embedding_keys(
            args.tf_embeddings_path,
            key_column=args.tf_embedding_key_column,
            name_mapping_path=args.tf_name_map,
        )
        print(f"Filtering TF labels to {len(tf_name_filter)} embedding keys")

    dataset = build_dataset(args, tf_name_filter=tf_name_filter)
    promoter_split = build_promoter_split(args, dataset)
    tf_embeddings = None
    tf_embedding_metadata = None
    if args.model in PROTEIN_DENSE_MODEL_NAMES:
        tf_embeddings, tf_embedding_metadata = load_tf_embeddings(
            args.tf_embeddings_path,
            dataset.tf_names,
            key_column=args.tf_embedding_key_column,
            embedding_column=args.tf_embedding_column,
            name_mapping_path=args.tf_name_map,
        )
        print(
            "TF embeddings:",
            {
                "path": str(args.tf_embeddings_path),
                "n_tfs": int(tf_embeddings.shape[0]),
                "embedding_dim": int(tf_embeddings.shape[1]),
            },
        )

    tf_split = build_tf_split(args, dataset, tf_embeddings=tf_embeddings)
    train_indices = split_indices(promoter_split, "train")
    val_indices = split_indices(promoter_split, "val")
    test_indices = split_indices(promoter_split, "test")
    train_tf_indices = tf_split_indices(tf_split, "train")
    val_tf_indices = tf_split_indices(tf_split, "val")
    test_tf_indices = tf_split_indices(tf_split, "test")
    use_tf_holdout = bool(val_tf_indices or test_tf_indices)
    train_tf_tensor = tensor_indices(train_tf_indices, device) if use_tf_holdout else None
    val_tf_tensor = tensor_indices(val_tf_indices, device) if val_tf_indices else None
    test_tf_tensor = tensor_indices(test_tf_indices, device) if test_tf_indices else None
    input_channels = infer_input_channels(dataset, args.input_mode)
    print("Dataset:", dataset.summary())
    print("Promoter split:", promoter_split["counts"])
    print("TF split:", tf_split["counts"])
    print("Input channels:", input_channels)
    print("Device:", device)
    print("Model:", args.model)

    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=partial(collate_promoter_batch, input_mode=args.input_mode),
    )
    train_eval_loader = make_split_loader(
        dataset,
        train_indices,
        input_mode=args.input_mode,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        shuffle=False,
    )
    val_loader = make_split_loader(
        dataset,
        val_indices,
        input_mode=args.input_mode,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        shuffle=False,
    )
    test_loader = make_split_loader(
        dataset,
        test_indices,
        input_mode=args.input_mode,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
        shuffle=False,
    )

    model = build_dense_model(
        args.model,
        n_tfs=len(dataset.tf_names),
        input_channels=input_channels,
        tf_embeddings=tf_embeddings,
        hidden_channels=args.hidden_channels,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        dilations=dilations,
        tf_embedding_dropout=args.tf_embedding_dropout,
        cross_attention_gate_logit_init=args.cross_attention_gate_logit_init,
        cross_attention_context_pool_sizes=parse_dilations(
            args.cross_attention_context_pool_sizes
        ),
        dna_attention_window_bp=args.dna_attention_window_bp,
        dna_attention_layers=args.dna_attention_layers,
        dna_attention_heads=args.dna_attention_heads,
        dna_attention_ffn_multiplier=args.dna_attention_ffn_multiplier,
        motif_kernel_sizes=parse_int_tuple(args.motif_kernel_sizes),
        protein_noise_std=args.protein_noise_std,
        protein_l2_normalize=args.protein_l2_normalize,
        scorer=args.scorer,
        scorer_heads=args.scorer_heads,
        scorer_pair_dim=args.scorer_pair_dim,
        scorer_hidden_dim=args.scorer_hidden_dim,
        scorer_bias_mode=args.scorer_bias_mode,
    ).to(device)
    print("Trainable parameters:", count_parameters(model))

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = build_lr_scheduler(args, optimizer)
    pos_weight = (
        None
        if args.no_pos_weight
        else compute_dense_pos_weight(dataset, device, indices=train_indices)
    )
    if pos_weight is not None:
        flat_weight = pos_weight.flatten()
        print(
            "Using dense pos_weight:",
            {
                "min": float(flat_weight.min().detach().cpu()),
                "median": float(flat_weight.median().detach().cpu()),
                "max": float(flat_weight.max().detach().cpu()),
            },
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir / "tf_names.json", dataset.tf_names)
    save_json(args.output_dir / "dataset_summary.json", dataset.summary())
    save_json(args.output_dir / "args.json", serializable_args(args))
    save_json(args.output_dir / "model_config.json", model_config_from_args(args, input_channels))
    split_out = args.promoter_split_out or (args.output_dir / "promoter_split.json")
    save_promoter_split(split_out, promoter_split)
    print(f"Saved promoter split: {split_out}")
    tf_split_out = args.tf_split_out or (args.output_dir / "tf_split.json")
    save_tf_split(tf_split_out, tf_split)
    print(f"Saved TF split: {tf_split_out}")
    if tf_embedding_metadata is not None:
        save_json(args.output_dir / "tf_embedding_metadata.json", tf_embedding_metadata)

    use_val_for_selection = val_loader is not None and args.eval_every > 0
    best_metric_name = args.selection_metric
    if best_metric_name.startswith("val_") and not use_val_for_selection:
        print(
            f"Selection metric {best_metric_name!r} needs validation; "
            "falling back to 'train_loss'."
        )
        best_metric_name = "train_loss"
    best_metric_value: float | None = None
    history: list[dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            input_mode=args.input_mode,
            pos_weight=pos_weight,
            tf_indices=train_tf_tensor,
            epoch=epoch,
        )
        if use_val_for_selection and (
            epoch % args.eval_every == 0 or epoch == args.epochs
        ):
            stats = attach_val_stats(
                stats,
                evaluate_dense(
                    model=model,
                    loader=val_loader,
                    device=device,
                    input_mode=args.input_mode,
                    pos_weight=None if use_tf_holdout else pos_weight,
                    tf_indices=val_tf_tensor if use_tf_holdout else None,
                    tf_names=dataset.tf_names,
                ),
            )

        current_selection_value = selection_value(stats, best_metric_name)
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler_value = stats.val_loss if stats.val_loss is not None else stats.train_loss
            scheduler.step(scheduler_value)
        elif scheduler is not None:
            scheduler.step()
            stats.lr = current_lr(optimizer)

        history.append(asdict(stats))
        message = (
            f"epoch={stats.epoch:03d} "
            f"loss={stats.train_loss:.5f} "
            f"micro_p={stats.micro_precision:.4f} "
            f"micro_r={stats.micro_recall:.4f} "
            f"micro_f1={stats.micro_f1:.4f} "
            f"pred_pos={stats.predicted_positive_rate:.5f} "
            f"lr={stats.lr:.6g}"
        )
        if stats.val_loss is not None:
            message += (
                f" val_loss={stats.val_loss:.5f} "
                f"val_micro_f1={stats.val_micro_f1:.4f} "
                f"val_pred_pos={stats.val_predicted_positive_rate:.5f}"
            )
            if stats.val_average_precision is not None:
                message += f" val_ap={stats.val_average_precision:.4f}"
            if stats.val_roc_auc is not None:
                message += f" val_roc_auc={stats.val_roc_auc:.4f}"
        print(message)

        save_json(args.output_dir / "history.json", history)
        save_checkpoint(
            args.output_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            args=args,
            dataset=dataset,
            input_channels=input_channels,
            stats=stats,
            promoter_split=promoter_split,
            tf_split=tf_split,
            tf_embedding_metadata=tf_embedding_metadata,
        )
        if current_selection_value is not None and selection_is_better(
            current_selection_value,
            best_metric_value,
            best_metric_name,
        ):
            best_metric_value = current_selection_value
            save_checkpoint(
                args.output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                args=args,
                dataset=dataset,
                input_channels=input_channels,
                stats=stats,
                promoter_split=promoter_split,
                tf_split=tf_split,
                tf_embedding_metadata=tf_embedding_metadata,
            )
        if args.save_every and epoch % args.save_every == 0:
            save_checkpoint(
                args.output_dir / f"epoch_{epoch:03d}.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                args=args,
                dataset=dataset,
                input_channels=input_channels,
                stats=stats,
                promoter_split=promoter_split,
                tf_split=tf_split,
                tf_embedding_metadata=tf_embedding_metadata,
            )

    best_path = args.output_dir / "best.pt"
    if best_path.exists():
        try:
            best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
        except TypeError:
            best_checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(best_checkpoint["model_state_dict"])

    final_metrics: dict[str, object] = {
        "selection_metric": best_metric_name,
        "best_metric_value": best_metric_value,
        "promoter_split_counts": promoter_split["counts"],
        "tf_split_counts": tf_split["counts"],
    }
    if use_tf_holdout:
        final_jobs = {
            "train_promoters_train_tfs": (train_eval_loader, train_tf_tensor),
            "val_promoters_val_tfs": (val_loader, val_tf_tensor),
            "test_promoters_test_tfs": (test_loader, test_tf_tensor),
            "test_promoters_train_tfs": (test_loader, train_tf_tensor),
            "train_promoters_test_tfs": (train_eval_loader, test_tf_tensor),
        }
        eval_pos_weight = None
    else:
        final_jobs = {
            "train": (train_eval_loader, None),
            "val": (val_loader, None),
            "test": (test_loader, None),
        }
        eval_pos_weight = pos_weight
    for split_name, (split_loader, split_tf_tensor) in final_jobs.items():
        if split_loader is None:
            continue
        if use_tf_holdout and split_tf_tensor is None:
            continue
        split_stats = evaluate_dense(
            model=model,
            loader=split_loader,
            device=device,
            input_mode=args.input_mode,
            pos_weight=eval_pos_weight,
            tf_indices=split_tf_tensor,
            tf_names=dataset.tf_names,
        )
        final_metrics[split_name] = asdict(split_stats)
    save_json(args.output_dir / "final_metrics.json", final_metrics)
    print("Final metrics:", final_metrics)
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
