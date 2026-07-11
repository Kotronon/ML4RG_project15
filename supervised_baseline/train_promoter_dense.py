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
        BindingBenchSampledPromoterWindowDataset,
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
        BindingBenchSampledPromoterWindowDataset,
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
    "dense_protein_residual_bilinear_cnn",
    "dense_protein_direct_scorer_cnn",
    "dense_protein_film_motif_cnn",
    "dense_protein_window_localization_cnn",
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
    val_dilated_micro_precision: float | None = None
    val_dilated_micro_recall: float | None = None
    val_dilated_micro_f1: float | None = None
    val_dilated_average_precision: float | None = None
    val_dilated_roc_auc: float | None = None
    window_loss: float | None = None
    window_micro_precision: float | None = None
    window_micro_recall: float | None = None
    window_micro_f1: float | None = None
    window_predicted_positive_rate: float | None = None
    val_window_loss: float | None = None
    val_window_micro_precision: float | None = None
    val_window_micro_recall: float | None = None
    val_window_micro_f1: float | None = None
    val_window_predicted_positive_rate: float | None = None
    val_window_average_precision: float | None = None
    val_window_roc_auc: float | None = None
    candidate_center_loss: float | None = None


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
    dilated_micro_precision: float | None = None
    dilated_micro_recall: float | None = None
    dilated_micro_f1: float | None = None
    dilated_average_precision: float | None = None
    dilated_roc_auc: float | None = None
    per_tf: list[dict[str, object]] | None = None
    window_loss: float | None = None
    window_micro_precision: float | None = None
    window_micro_recall: float | None = None
    window_micro_f1: float | None = None
    window_predicted_positive_rate: float | None = None
    window_average_precision: float | None = None
    window_roc_auc: float | None = None
    promoter_pair_average_precision: float | None = None
    promoter_pair_roc_auc: float | None = None
    promoter_pair_per_tf: list[dict[str, object]] | None = None


MULTITASK_LABEL_MODE = "tf_and_merged_train_tfs"
LABEL_MODES = ("tf", "merged_train_tfs", MULTITASK_LABEL_MODE)
LOSS_NAMES = ("bce", "focal", "rank")
TRAINING_WINDOW_MODES = ("full_promoter", "sampled_windows")
FINAL_EVAL_SCOPES = ("all", "test_only")
DNA_FINETUNE_MODES = ("none", "upper", "all")


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
        "--label-mode",
        choices=LABEL_MODES,
        default="tf",
        help=(
            "Use per-TF dense labels, or collapse labels into one DNA-only "
            "channel using training TFs only."
        ),
    )
    parser.add_argument(
        "--loss",
        choices=LOSS_NAMES,
        default="bce",
        help="Dense training loss.",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Focusing exponent for --loss focal.",
    )
    parser.add_argument(
        "--focal-alpha",
        type=float,
        help=(
            "Optional focal alpha for positives. Leave unset to use only "
            "--pos-weight with focal modulation."
        ),
    )
    parser.add_argument(
        "--rank-temperature",
        type=float,
        default=1.0,
        help=(
            "Softmax temperature for --loss rank. Lower values make the loss "
            "focus harder on the highest-scoring candidate bases."
        ),
    )
    parser.add_argument(
        "--rank-negative-weight",
        type=float,
        default=0.1,
        help=(
            "Weight for the negative-only sampled-window penalty used by "
            "--loss rank. Set to 0 to train only on windows containing positives."
        ),
    )
    parser.add_argument(
        "--rank-negative-top-k",
        type=int,
        default=10,
        help=(
            "Number of highest-scoring bases used for the negative-only "
            "penalty in --loss rank."
        ),
    )
    parser.add_argument(
        "--window-loss-weight",
        type=float,
        default=1.0,
        help=(
            "Loss weight for models that emit a window-level binding logit. "
            "The dense localization loss keeps weight 1.0."
        ),
    )
    parser.add_argument(
        "--window-pooling",
        choices=("max", "logsumexp", "topk_logmeanexp"),
        default="topk_logmeanexp",
        help=(
            "How models with coupled window logits pool base-wise localization "
            "logits into a window-level binding logit."
        ),
    )
    parser.add_argument(
        "--window-pooling-top-k",
        type=int,
        default=10,
        help="Number of highest base logits used by --window-pooling topk_logmeanexp.",
    )
    parser.add_argument(
        "--training-window-mode",
        choices=TRAINING_WINDOW_MODES,
        default="full_promoter",
        help=(
            "Use full promoters for training, or sample TF-conditioned windows "
            "while keeping full-promoter validation/test evaluation."
        ),
    )
    parser.add_argument(
        "--sampled-window-size",
        type=int,
        default=200,
        help="Window length for --training-window-mode sampled_windows.",
    )
    parser.add_argument(
        "--sampled-window-samples-per-epoch",
        type=int,
        default=50_000,
        help="Number of sampled TF-window pairs per training epoch.",
    )
    parser.add_argument(
        "--sampled-window-positive-fraction",
        type=float,
        default=0.5,
        help="Fraction of sampled windows anchored on the target TF's sites.",
    )
    parser.add_argument(
        "--sampled-window-hard-negative-fraction",
        type=float,
        default=0.25,
        help=(
            "Fraction of sampled windows containing another sampled TF's site "
            "but no target-TF site."
        ),
    )
    parser.add_argument(
        "--candidate-windows-path",
        type=Path,
        help=(
            "Optional parquet of DNA-only candidate loci. Candidate sampling "
            "uses these loci as binding-site-like windows for protein reranking."
        ),
    )
    parser.add_argument(
        "--sampled-window-candidate-fraction",
        type=float,
        default=0.0,
        help=(
            "Fraction of sampled windows anchored on DNA-only candidate loci. "
            "Requires --candidate-windows-path."
        ),
    )
    parser.add_argument(
        "--candidate-tf-positive-fraction",
        type=float,
        default=0.5,
        help=(
            "Within candidate-anchored samples that overlap known train-TF "
            "sites, probability of pairing the candidate with an overlapping "
            "true TF instead of a wrong/non-overlapping TF."
        ),
    )
    parser.add_argument(
        "--candidate-center-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Additional BCE/focal loss weight at the DNA-only proposed base "
            "for candidate-anchored samples. This trains a TF-specific "
            "reranker while preserving the dense localization objective."
        ),
    )
    parser.add_argument(
        "--sampled-window-margin-bp",
        type=int,
        default=30,
        help="Minimum preferred distance between positive anchor and window edge.",
    )
    parser.add_argument(
        "--sampled-window-negative-exclusion-bp",
        type=int,
        default=10,
        help="Buffer around target-TF sites rejected for negative windows.",
    )
    parser.add_argument(
        "--sampled-window-max-attempts",
        type=int,
        default=100,
        help="Maximum rejection-sampling attempts per sampled window.",
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
        "--protein-delta-gate-logit-init",
        type=float,
        default=-3.0,
        help=(
            "Initial logit for residual protein-conditioned score scale in "
            "dense_protein_residual_bilinear_cnn."
        ),
    )
    parser.add_argument(
        "--pretrained-dna-checkpoint",
        type=Path,
        help=(
            "Optional DNA-only dense checkpoint used to initialize the DNA "
            "branch of compatible protein-conditioned models."
        ),
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        help=(
            "Load a complete model checkpoint and continue training from its "
            "weights with a freshly initialized optimizer. This is intended for "
            "second-stage fine-tuning of a protein-conditioned model."
        ),
    )
    parser.add_argument(
        "--freeze-dna-branch",
        action="store_true",
        help=(
            "Freeze the initialized DNA branch and keep it in eval mode during "
            "training. Intended for the first residual protein-conditioning run."
        ),
    )
    parser.add_argument(
        "--dna-finetune-mode",
        choices=DNA_FINETUNE_MODES,
        default="none",
        help=(
            "DNA parameters to unfreeze after --resume-checkpoint. 'upper' "
            "unfreezes the final dilated context block and final local-attention "
            "block; 'all' unfreezes the full DNA encoder."
        ),
    )
    parser.add_argument(
        "--dna-finetune-lr",
        type=float,
        help=(
            "Learning rate for DNA parameters selected by --dna-finetune-mode. "
            "The remaining trainable parameters use --lr."
        ),
    )
    parser.add_argument(
        "--shuffle-tf-embeddings",
        action="store_true",
        help="Shuffle TF protein embeddings after building the TF split as a control.",
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
    parser.add_argument(
        "--film-eval-tf-chunk-size",
        type=int,
        default=1,
        help=(
            "Number of TFs jointly evaluated by dense_protein_film_motif_cnn. "
            "Smaller values use less GPU memory when every base is scored."
        ),
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
            "val_dilated_average_precision",
            "val_dilated_roc_auc",
            "val_dilated_micro_f1",
            "val_window_loss",
            "val_window_average_precision",
            "val_window_roc_auc",
            "val_window_micro_f1",
            "train_loss",
        ),
        default="val_loss",
        help=(
            "Metric used to choose best.pt. Loss is minimized; AP/AUROC/F1 are "
            "maximized. The val_dilated_* metrics compare predictions with the "
            "label-smoothed evaluation target."
        ),
    )
    parser.add_argument(
        "--final-eval-scope",
        choices=FINAL_EVAL_SCOPES,
        default="all",
        help=(
            "Final metric slices to compute. 'test_only' evaluates only "
            "test promoters against held-out test TFs (or the ordinary test "
            "split), avoiding very large all-pair train evaluations."
        ),
    )
    parser.add_argument(
        "--evaluate-only",
        action="store_true",
        help=(
            "Load output-dir/best.pt and run final evaluation without taking "
            "an optimization step. Pass the same dataset and model arguments "
            "used to train that checkpoint."
        ),
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
    if (
        args.tf_split_mode != "none"
        and args.model not in PROTEIN_DENSE_MODEL_NAMES
        and args.label_mode == "tf"
    ):
        raise ValueError("TF holdout splits are only meaningful for dense protein models")
    if args.eval_every < 0:
        raise ValueError("--eval-every must be non-negative")
    if args.focal_gamma < 0:
        raise ValueError("--focal-gamma must be non-negative")
    if args.focal_alpha is not None and not 0.0 <= args.focal_alpha <= 1.0:
        raise ValueError("--focal-alpha must be between 0 and 1")
    if args.rank_temperature <= 0:
        raise ValueError("--rank-temperature must be positive")
    if args.rank_negative_weight < 0:
        raise ValueError("--rank-negative-weight must be non-negative")
    if args.rank_negative_top_k <= 0:
        raise ValueError("--rank-negative-top-k must be positive")
    if args.window_loss_weight < 0:
        raise ValueError("--window-loss-weight must be non-negative")
    if args.window_pooling_top_k <= 0:
        raise ValueError("--window-pooling-top-k must be positive")
    if args.training_window_mode == "sampled_windows" and args.label_mode != "tf":
        raise ValueError("--training-window-mode sampled_windows requires --label-mode tf")
    if args.sampled_window_size <= 0:
        raise ValueError("--sampled-window-size must be positive")
    if args.sampled_window_samples_per_epoch <= 0:
        raise ValueError("--sampled-window-samples-per-epoch must be positive")
    if not 0.0 <= args.sampled_window_positive_fraction <= 1.0:
        raise ValueError("--sampled-window-positive-fraction must be in [0, 1]")
    if not 0.0 <= args.sampled_window_hard_negative_fraction <= 1.0:
        raise ValueError("--sampled-window-hard-negative-fraction must be in [0, 1]")
    if not 0.0 <= args.sampled_window_candidate_fraction <= 1.0:
        raise ValueError("--sampled-window-candidate-fraction must be in [0, 1]")
    if not 0.0 <= args.candidate_tf_positive_fraction <= 1.0:
        raise ValueError("--candidate-tf-positive-fraction must be in [0, 1]")
    if args.candidate_center_loss_weight < 0:
        raise ValueError("--candidate-center-loss-weight must be non-negative")
    if (
        args.sampled_window_positive_fraction
        + args.sampled_window_hard_negative_fraction
        + args.sampled_window_candidate_fraction
        > 1.0
    ):
        raise ValueError(
            "sampled positive, hard-negative, and candidate fractions must sum to <= 1"
        )
    if args.sampled_window_candidate_fraction > 0 and args.candidate_windows_path is None:
        raise ValueError(
            "--candidate-windows-path is required when "
            "--sampled-window-candidate-fraction > 0"
        )
    if (
        args.candidate_center_loss_weight > 0
        and args.sampled_window_candidate_fraction <= 0
    ):
        raise ValueError(
            "--candidate-center-loss-weight requires "
            "--sampled-window-candidate-fraction > 0"
        )
    if args.candidate_center_loss_weight > 0 and args.loss == "rank":
        raise ValueError(
            "--candidate-center-loss-weight is supported with BCE or focal loss, "
            "not the position-ranking objective"
        )
    if args.sampled_window_margin_bp < 0:
        raise ValueError("--sampled-window-margin-bp must be non-negative")
    if args.sampled_window_margin_bp * 2 >= args.sampled_window_size:
        raise ValueError("--sampled-window-margin-bp must leave room inside the window")
    if args.sampled_window_negative_exclusion_bp < 0:
        raise ValueError("--sampled-window-negative-exclusion-bp must be non-negative")
    if args.sampled_window_max_attempts <= 0:
        raise ValueError("--sampled-window-max-attempts must be positive")
    if args.label_mode != "tf" and args.model in PROTEIN_DENSE_MODEL_NAMES:
        raise ValueError("--label-mode merged_train_tfs is intended for DNA-only models")
    if args.freeze_dna_branch and args.pretrained_dna_checkpoint is None:
        raise ValueError("--freeze-dna-branch requires --pretrained-dna-checkpoint")
    if args.resume_checkpoint is not None:
        if not args.resume_checkpoint.is_file():
            raise ValueError(
                f"--resume-checkpoint does not exist: {args.resume_checkpoint}"
            )
        if args.pretrained_dna_checkpoint is not None:
            raise ValueError(
                "--resume-checkpoint and --pretrained-dna-checkpoint are mutually exclusive"
            )
        if args.freeze_dna_branch:
            raise ValueError(
                "--resume-checkpoint cannot be combined with --freeze-dna-branch; "
                "use --dna-finetune-mode instead"
            )
        if args.evaluate_only:
            raise ValueError(
                "--resume-checkpoint cannot be combined with --evaluate-only"
            )
    elif args.dna_finetune_mode != "none":
        raise ValueError("--dna-finetune-mode requires --resume-checkpoint")
    if args.dna_finetune_mode != "none":
        if args.dna_finetune_lr is None or args.dna_finetune_lr <= 0:
            raise ValueError(
                "--dna-finetune-lr must be positive when --dna-finetune-mode is enabled"
            )
    elif args.dna_finetune_lr is not None:
        raise ValueError("--dna-finetune-lr requires --dna-finetune-mode")
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
    if args.film_eval_tf_chunk_size <= 0:
        raise ValueError("--film-eval-tf-chunk-size must be positive")
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
    if args.evaluate_only and not (args.output_dir / "best.pt").is_file():
        raise ValueError(
            "--evaluate-only requires an existing best.pt in --output-dir: "
            f"{args.output_dir / 'best.pt'}"
        )
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
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
    *,
    pairwise_tf_indices: bool = False,
) -> torch.Tensor:
    if pairwise_tf_indices:
        if tf_indices is None:
            raise ValueError("pairwise_tf_indices=True requires tf_indices")
        if getattr(model, "supports_pairwise_tf_indices", False):
            return model(
                x,
                tf_indices=tf_indices,
                pairwise_tf_indices=True,
            )
        logits = model(x)
        gather_idx = tf_indices.view(-1, 1, 1).expand(-1, 1, logits.shape[-1])
        return logits.gather(1, gather_idx)

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


def select_pos_weight_per_example(
    pos_weight: torch.Tensor | None,
    tf_indices: torch.Tensor | None,
) -> torch.Tensor | None:
    if pos_weight is None or tf_indices is None:
        return pos_weight
    flat = pos_weight.view(-1)
    return flat.index_select(0, tf_indices).view(-1, 1, 1)


def is_multitask_label_mode(label_mode: str) -> bool:
    return label_mode == MULTITASK_LABEL_MODE


def select_multitask_axis(
    tensor: torch.Tensor,
    tf_indices: torch.Tensor | None,
) -> torch.Tensor:
    if tf_indices is None:
        return tensor
    tf_part = tensor[:, :-1, ...].index_select(1, tf_indices)
    merged_part = tensor[:, -1:, ...]
    return torch.cat([tf_part, merged_part], dim=1)


def make_dense_targets(
    y: torch.Tensor,
    *,
    label_mode: str,
    model_tf_indices: torch.Tensor | None,
    merge_tf_indices: torch.Tensor | None,
) -> torch.Tensor:
    if label_mode == "tf":
        return select_tf_axis(y, model_tf_indices)
    if label_mode == "merged_train_tfs":
        selected = select_tf_axis(y, merge_tf_indices)
        return selected.amax(dim=1, keepdim=True)
    if is_multitask_label_mode(label_mode):
        tf_targets = select_tf_axis(y, model_tf_indices)
        merged_source = select_tf_axis(y, merge_tf_indices)
        merged_target = merged_source.amax(dim=1, keepdim=True)
        return torch.cat([tf_targets, merged_target], dim=1)
    raise ValueError(f"Unknown label mode: {label_mode}")


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

    batch = {
        "x": x_batch,
        "y": y_batch,
        "hard_y": hard_y_batch,
        "mask": mask_batch,
        "meta": [item["meta"] for item in items],
        "candidate_center_offset": torch.tensor(
            [int(item.get("candidate_center_offset", -1)) for item in items],
            dtype=torch.long,
        ),
    }
    if "tf_idx" in items[0]:
        batch["tf_idx"] = torch.tensor(
            [int(item["tf_idx"]) for item in items],
            dtype=torch.long,
        )
    return batch


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


def compute_merged_pos_weight(
    dataset,
    device: torch.device,
    *,
    indices: list[int] | None = None,
    tf_indices: list[int] | None = None,
) -> torch.Tensor:
    selected_tfs = None if tf_indices is None else {int(idx) for idx in tf_indices}
    positives = 0.0
    valid_positions = 0.0
    records = dataset.records if indices is None else [dataset.records[idx] for idx in indices]
    for record in records:
        mask = dataset._position_mask(record)
        valid_positions += float(mask.sum())
        labels = np.zeros(len(record.sequence), dtype=bool)
        for tf_idx, lo, hi in record.label_intervals:
            if selected_tfs is None or int(tf_idx) in selected_tfs:
                labels[lo:hi] = True
        labels[~mask] = False
        positives += float(labels.sum())

    negatives = valid_positions - positives
    weight = negatives / max(positives, 1.0)
    weight = float(np.clip(weight, 1.0, 100.0))
    return torch.tensor([[[weight]]], dtype=torch.float32, device=device)


def compute_multitask_pos_weight(
    dataset,
    device: torch.device,
    *,
    indices: list[int] | None = None,
    tf_indices: list[int] | None = None,
) -> torch.Tensor:
    dense_weight = compute_dense_pos_weight(dataset, device, indices=indices)
    merged_weight = compute_merged_pos_weight(
        dataset,
        device,
        indices=indices,
        tf_indices=tf_indices,
    )
    return torch.cat([dense_weight, merged_weight], dim=1)


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


def dense_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: torch.Tensor | None,
    *,
    gamma: float,
    alpha: float | None,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=pos_weight,
        reduction="none",
    )
    probs = logits.sigmoid()
    p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
    loss = bce * (1.0 - p_t).clamp_min(1e-6).pow(gamma)
    if alpha is not None:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = loss * alpha_t
    valid = mask.unsqueeze(1).expand_as(loss)
    return loss[valid].mean()


def dense_position_rank_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    *,
    temperature: float,
    negative_weight: float,
    negative_top_k: int,
) -> torch.Tensor:
    valid = mask.unsqueeze(1).expand_as(logits)
    positives = (targets >= 0.5) & valid
    has_valid = valid.any(dim=-1)
    has_positive = positives.any(dim=-1)
    scaled_logits = logits / temperature

    positive_losses = []
    if has_positive.any():
        pos_logits = scaled_logits.masked_fill(~positives, -torch.inf)
        all_logits = scaled_logits.masked_fill(~valid, -torch.inf)
        pos_lse = torch.logsumexp(pos_logits, dim=-1)
        all_lse = torch.logsumexp(all_logits, dim=-1)
        positive_losses.append(-(pos_lse[has_positive] - all_lse[has_positive]))

    losses = []
    if positive_losses:
        losses.append(torch.cat(positive_losses).mean())

    has_negative_only = has_valid & ~has_positive
    if negative_weight > 0 and has_negative_only.any():
        negative_logits = logits.masked_fill(~valid, -torch.inf)[has_negative_only]
        k = min(int(negative_top_k), int(negative_logits.shape[-1]))
        top_logits = torch.topk(negative_logits, k=k, dim=-1).values
        finite_top = torch.isfinite(top_logits)
        top_logits = top_logits.masked_fill(~finite_top, -torch.inf)
        top_counts = finite_top.sum(dim=-1).clamp_min(1).float()
        pooled_logits = torch.logsumexp(top_logits, dim=-1) - top_counts.log()
        losses.append(negative_weight * F.softplus(pooled_logits).mean())

    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).sum()


def dense_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
    pos_weight: torch.Tensor | None,
    *,
    loss_name: str,
    focal_gamma: float,
    focal_alpha: float | None,
    rank_temperature: float,
    rank_negative_weight: float,
    rank_negative_top_k: int,
) -> torch.Tensor:
    if loss_name == "bce":
        return dense_bce_loss(logits, targets, mask, pos_weight)
    if loss_name == "focal":
        return dense_focal_loss(
            logits,
            targets,
            mask,
            pos_weight,
            gamma=focal_gamma,
            alpha=focal_alpha,
        )
    if loss_name == "rank":
        return dense_position_rank_loss(
            logits,
            targets,
            mask,
            temperature=rank_temperature,
            negative_weight=rank_negative_weight,
            negative_top_k=rank_negative_top_k,
        )
    raise ValueError(f"Unknown loss: {loss_name}")


def split_dense_outputs(
    outputs: torch.Tensor | dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(outputs, dict):
        if "logits" not in outputs:
            raise ValueError("Model output dict must contain 'logits'")
        return outputs["logits"], outputs.get("window_logits")
    return outputs, None


def window_targets_from_dense(
    hard_targets: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    valid = mask.unsqueeze(1).expand_as(hard_targets)
    return ((hard_targets >= 0.5) & valid).any(dim=-1).float()


def window_bce_loss(
    window_logits: torch.Tensor,
    window_targets: torch.Tensor,
) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(
        window_logits,
        window_targets,
        reduction="mean",
    )


def candidate_center_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    loss_name: str,
    focal_gamma: float,
    focal_alpha: float | None,
) -> torch.Tensor:
    """Binary TF-specific loss at a proposed DNA candidate position."""
    if loss_name == "bce":
        return F.binary_cross_entropy_with_logits(logits, targets)
    if loss_name == "focal":
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probabilities = logits.sigmoid()
        p_t = probabilities * targets + (1.0 - probabilities) * (1.0 - targets)
        loss = bce * (1.0 - p_t).clamp_min(1e-6).pow(focal_gamma)
        if focal_alpha is not None:
            alpha_t = focal_alpha * targets + (1.0 - focal_alpha) * (1.0 - targets)
            loss = loss * alpha_t
        return loss.mean()
    raise ValueError("Candidate-center loss supports only BCE or focal loss")


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


def model_config_from_args(
    args: argparse.Namespace,
    input_channels: int,
    output_channels: int,
) -> dict[str, object]:
    return {
        "model_name": args.model,
        "input_mode": args.input_mode,
        "input_channels": input_channels,
        "output_channels": output_channels,
        "label_mode": args.label_mode,
        "loss": args.loss,
        "focal_gamma": args.focal_gamma,
        "focal_alpha": args.focal_alpha,
        "rank_temperature": args.rank_temperature,
        "rank_negative_weight": args.rank_negative_weight,
        "rank_negative_top_k": args.rank_negative_top_k,
        "window_loss_weight": args.window_loss_weight,
        "window_pooling": args.window_pooling,
        "window_pooling_top_k": args.window_pooling_top_k,
        "training_window_mode": args.training_window_mode,
        "sampled_window_size": args.sampled_window_size,
        "sampled_window_samples_per_epoch": args.sampled_window_samples_per_epoch,
        "sampled_window_positive_fraction": args.sampled_window_positive_fraction,
        "sampled_window_hard_negative_fraction": (
            args.sampled_window_hard_negative_fraction
        ),
        "candidate_center_loss_weight": args.candidate_center_loss_weight,
        "final_eval_scope": args.final_eval_scope,
        "candidate_windows_path": (
            str(args.candidate_windows_path)
            if args.candidate_windows_path is not None
            else None
        ),
        "sampled_window_candidate_fraction": args.sampled_window_candidate_fraction,
        "candidate_tf_positive_fraction": args.candidate_tf_positive_fraction,
        "sampled_window_margin_bp": args.sampled_window_margin_bp,
        "sampled_window_negative_exclusion_bp": (
            args.sampled_window_negative_exclusion_bp
        ),
        "sampled_window_max_attempts": args.sampled_window_max_attempts,
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
        "protein_delta_gate_logit_init": args.protein_delta_gate_logit_init,
        "pretrained_dna_checkpoint": (
            str(args.pretrained_dna_checkpoint)
            if args.pretrained_dna_checkpoint
            else None
        ),
        "resume_checkpoint": (
            str(args.resume_checkpoint) if args.resume_checkpoint else None
        ),
        "freeze_dna_branch": args.freeze_dna_branch,
        "dna_finetune_mode": args.dna_finetune_mode,
        "dna_finetune_lr": args.dna_finetune_lr,
        "shuffle_tf_embeddings": args.shuffle_tf_embeddings,
        "scorer": args.scorer,
        "scorer_heads": args.scorer_heads,
        "scorer_pair_dim": args.scorer_pair_dim,
        "scorer_hidden_dim": args.scorer_hidden_dim,
        "scorer_bias_mode": args.scorer_bias_mode,
        "film_eval_tf_chunk_size": args.film_eval_tf_chunk_size,
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


def load_checkpoint_cpu(checkpoint_path: Path) -> dict[str, object]:
    try:
        return torch.load(
            checkpoint_path,
            map_location=torch.device("cpu"),
            weights_only=False,
        )
    except TypeError:
        return torch.load(checkpoint_path, map_location=torch.device("cpu"))


def _checkpoint_dict(checkpoint: dict[str, object], key: str) -> dict[str, object]:
    value = checkpoint.get(key, {})
    return dict(value) if isinstance(value, dict) else {}


def _stringify_int_sequence(value: object) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        return ",".join(str(int(item)) for item in value)
    return str(value)


def align_dna_branch_args_from_checkpoint(
    args: argparse.Namespace,
    *,
    input_channels: int,
) -> None:
    dna_branch_models = {
        "dense_protein_residual_bilinear_cnn",
        "dense_protein_direct_scorer_cnn",
        "dense_protein_film_motif_cnn",
        "dense_protein_window_localization_cnn",
    }
    if args.model not in dna_branch_models or args.pretrained_dna_checkpoint is None:
        return

    checkpoint = load_checkpoint_cpu(args.pretrained_dna_checkpoint)
    config = _checkpoint_dict(checkpoint, "model_config")
    saved_args = _checkpoint_dict(checkpoint, "args")
    checkpoint_model = str(
        checkpoint.get("model_name")
        or config.get("model_name")
        or saved_args.get("model", "")
    )
    if checkpoint_model not in {
        "dense_motif_dilated_attention_cnn",
        "dense_protein_residual_bilinear_cnn",
        "dense_protein_direct_scorer_cnn",
        "dense_protein_film_motif_cnn",
        "dense_protein_window_localization_cnn",
    }:
        raise ValueError(
            f"--pretrained-dna-checkpoint for {args.model} must come from "
            "dense_motif_dilated_attention_cnn or another compatible "
            f"dna-branch protein model checkpoint, got {checkpoint_model!r}"
        )

    checkpoint_input_channels = config.get("input_channels", saved_args.get("input_channels"))
    if checkpoint_input_channels is not None and int(checkpoint_input_channels) != input_channels:
        raise ValueError(
            "Pretrained DNA checkpoint input channels do not match this run: "
            f"checkpoint has {checkpoint_input_channels}, dataset has {input_channels}. "
            "Use the same DNA input mode/embedding layer as the DNA-only checkpoint."
        )

    scalar_keys = (
        ("hidden_channels", int),
        ("kernel_size", int),
        ("dropout", float),
        ("dna_attention_window_bp", int),
        ("dna_attention_layers", int),
        ("dna_attention_heads", int),
        ("dna_attention_ffn_multiplier", float),
    )
    sequence_keys = ("dilations", "motif_kernel_sizes")
    applied: dict[str, object] = {}

    for key, caster in scalar_keys:
        value = config.get(key, saved_args.get(key))
        if value is None:
            continue
        cast_value = caster(value)
        setattr(args, key, cast_value)
        applied[key] = cast_value

    for key in sequence_keys:
        value = config.get(key, saved_args.get(key))
        if value is None:
            continue
        string_value = _stringify_int_sequence(value)
        setattr(args, key, string_value)
        applied[key] = string_value

    if applied:
        print(
            "Aligned DNA branch architecture from pretrained checkpoint:",
            {
                "checkpoint": str(args.pretrained_dna_checkpoint),
                "checkpoint_model": checkpoint_model,
                **applied,
            },
        )


def load_pretrained_dna_branch(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> None:
    dna_model = getattr(model, "dna_model", None)
    if dna_model is None:
        raise ValueError(
            f"Model {type(model).__name__} has no dna_model branch for "
            "--pretrained-dna-checkpoint"
        )
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state, dict):
        raise ValueError(f"Checkpoint has no state dict: {checkpoint_path}")

    if any(str(key).startswith("dna_model.") for key in state):
        state = {
            str(key).removeprefix("dna_model."): value
            for key, value in state.items()
            if str(key).startswith("dna_model.")
        }

    incompatible = dna_model.load_state_dict(state, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    print(
        "Loaded pretrained DNA branch:",
        {
            "checkpoint": str(checkpoint_path),
            "missing_keys": missing,
            "unexpected_keys": unexpected,
        },
    )


def freeze_dna_branch(model: torch.nn.Module) -> None:
    freeze_method = getattr(model, "freeze_dna_branch", None)
    if not callable(freeze_method):
        raise ValueError(f"Model {type(model).__name__} does not support DNA freezing")
    freeze_method()


def load_resume_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    *,
    expected_model_name: str,
    expected_tf_names: list[str],
) -> dict[str, object]:
    """Restore a complete model state while keeping a fresh optimizer."""
    checkpoint = load_checkpoint_cpu(checkpoint_path)
    config = _checkpoint_dict(checkpoint, "model_config")
    saved_args = _checkpoint_dict(checkpoint, "args")
    checkpoint_model = str(
        checkpoint.get("model_name")
        or config.get("model_name")
        or saved_args.get("model", "")
    )
    if checkpoint_model != expected_model_name:
        raise ValueError(
            "Resume checkpoint model does not match this run: "
            f"checkpoint has {checkpoint_model!r}, requested {expected_model_name!r}"
        )

    checkpoint_tf_names = checkpoint.get("tf_names")
    if checkpoint_tf_names is not None and list(checkpoint_tf_names) != expected_tf_names:
        raise ValueError(
            "Resume checkpoint TF order does not match the current dataset. "
            "Use the identical sites file, TF embedding table, and TF filtering options."
        )

    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"Resume checkpoint has no model_state_dict: {checkpoint_path}")
    model.load_state_dict(state, strict=True)
    print(
        "Resumed complete model checkpoint with a fresh optimizer:",
        {"checkpoint": str(checkpoint_path), "epoch": checkpoint.get("epoch")},
    )
    return checkpoint


def configure_dna_finetuning(
    model: torch.nn.Module,
    mode: str,
) -> int:
    """Freeze a resumed DNA branch except for the requested upper layers."""
    dna_model = getattr(model, "dna_model", None)
    if dna_model is None:
        raise ValueError(f"Model {type(model).__name__} has no DNA branch to fine-tune")
    if mode not in DNA_FINETUNE_MODES:
        raise ValueError(f"Unknown DNA fine-tune mode: {mode!r}")

    for parameter in dna_model.parameters():
        parameter.requires_grad = False

    unfrozen_modules: list[torch.nn.Module] = []
    if mode == "upper":
        context_blocks = getattr(dna_model, "context_blocks", ())
        attention_blocks = getattr(dna_model, "attention_blocks", ())
        if context_blocks:
            unfrozen_modules.append(context_blocks[-1])
        if attention_blocks:
            unfrozen_modules.append(attention_blocks[-1])
        if not unfrozen_modules:
            raise ValueError("DNA encoder has no upper context blocks to fine-tune")
    elif mode == "all":
        unfrozen_modules.append(dna_model)

    for module in unfrozen_modules:
        for parameter in module.parameters():
            parameter.requires_grad = True

    # DenseProteinDirectScorerCNN keeps the entire DNA branch in eval mode only
    # for first-stage frozen runs. Partial fine-tuning needs model.train() to
    # reach the selected upper blocks, while lower frozen blocks stay in eval.
    if hasattr(model, "dna_frozen"):
        model.dna_frozen = mode == "none"
    if mode == "none":
        dna_model.eval()

    if mode == "none":
        frozen_modules = [dna_model]
    elif mode == "upper":
        frozen_modules = [
            *getattr(dna_model, "motif_stem", ()),
            *(
                module
                for module in (
                    getattr(dna_model, "stem_norm", None),
                    getattr(dna_model, "stem_activation", None),
                    getattr(dna_model, "stem_dropout", None),
                    getattr(dna_model, "head", None),
                )
                if module is not None
            ),
            *list(getattr(dna_model, "context_blocks", ())[:-1]),
            *list(getattr(dna_model, "attention_blocks", ())[:-1]),
        ]
    else:
        frozen_modules = []
    setattr(model, "_frozen_dna_modules", frozen_modules)

    n_trainable = sum(
        parameter.numel() for parameter in dna_model.parameters() if parameter.requires_grad
    )
    print(
        "DNA fine-tuning:",
        {
            "mode": mode,
            "trainable_dna_parameters": n_trainable,
            "unfrozen_modules": [type(module).__name__ for module in unfrozen_modules],
        },
    )
    return n_trainable


def set_frozen_dna_modules_eval(model: torch.nn.Module) -> None:
    """Keep dropout and batch norm fixed in lower frozen DNA layers."""
    for module in getattr(model, "_frozen_dna_modules", ()):
        module.eval()


def build_optimizer(
    model: torch.nn.Module,
    args: argparse.Namespace,
) -> torch.optim.Optimizer:
    dna_model = getattr(model, "dna_model", None)
    dna_parameters = (
        [parameter for parameter in dna_model.parameters() if parameter.requires_grad]
        if dna_model is not None
        else []
    )
    dna_parameter_ids = {id(parameter) for parameter in dna_parameters}
    other_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad and id(parameter) not in dna_parameter_ids
    ]
    if not other_parameters and not dna_parameters:
        raise ValueError("Model has no trainable parameters")

    parameter_groups: list[dict[str, object]] = []
    if other_parameters:
        parameter_groups.append({"params": other_parameters, "lr": args.lr})
    if dna_parameters:
        dna_lr = args.dna_finetune_lr if args.dna_finetune_mode != "none" else args.lr
        parameter_groups.append({"params": dna_parameters, "lr": dna_lr})
    print(
        "Optimizer parameter groups:",
        [
            {
                "lr": group["lr"],
                "parameters": sum(parameter.numel() for parameter in group["params"]),
            }
            for group in parameter_groups
        ],
    )
    return torch.optim.AdamW(parameter_groups, weight_decay=args.weight_decay)


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    dataset,
    input_channels: int,
    output_channels: int,
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
            "model_config": model_config_from_args(args, input_channels, output_channels),
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
    model_tf_indices: torch.Tensor | None,
    merge_tf_indices: torch.Tensor | None,
    label_mode: str,
    loss_name: str,
    focal_gamma: float,
    focal_alpha: float | None,
    rank_temperature: float,
    rank_negative_weight: float,
    rank_negative_top_k: int,
    window_loss_weight: float,
    candidate_center_loss_weight: float,
    epoch: int,
) -> DenseEpochStats:
    model.train()
    set_frozen_dna_modules_eval(model)
    total_loss = 0.0
    total_examples = 0
    tp = fp = fn = 0.0
    predicted_positive = 0.0
    total_labels = 0.0
    window_loss_total = 0.0
    window_batches = 0
    window_tp = window_fp = window_fn = 0.0
    window_predicted_positive = 0.0
    window_total_labels = 0.0
    candidate_loss_total = 0.0
    candidate_examples = 0

    for batch in loader:
        x = prepare_x(batch["x"].to(device, non_blocking=True), input_mode)
        y = batch["y"].to(device, non_blocking=True)
        hard_y = batch["hard_y"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        batch_tf_indices = batch.get("tf_idx")
        if batch_tf_indices is not None:
            batch_tf_indices = batch_tf_indices.to(device, non_blocking=True)
        candidate_center_offsets = batch["candidate_center_offset"].to(
            device,
            non_blocking=True,
        )

        optimizer.zero_grad(set_to_none=True)
        if batch_tf_indices is not None:
            outputs = forward_dense_model(
                model,
                x,
                batch_tf_indices,
                pairwise_tf_indices=True,
            )
            batch_pos_weight = select_pos_weight_per_example(
                pos_weight,
                batch_tf_indices,
            )
        else:
            outputs = forward_dense_model(
                model,
                x,
                None if is_multitask_label_mode(label_mode) else model_tf_indices,
            )
            y = make_dense_targets(
                y,
                label_mode=label_mode,
                model_tf_indices=model_tf_indices,
                merge_tf_indices=merge_tf_indices,
            )
            hard_y = make_dense_targets(
                hard_y,
                label_mode=label_mode,
                model_tf_indices=model_tf_indices,
                merge_tf_indices=merge_tf_indices,
            )
            batch_pos_weight = (
                select_tf_axis(pos_weight, model_tf_indices)
                if pos_weight is not None and label_mode == "tf"
                else pos_weight
            )
            if pos_weight is not None and is_multitask_label_mode(label_mode):
                batch_pos_weight = select_multitask_axis(pos_weight, model_tf_indices)
        logits, window_logits = split_dense_outputs(outputs)
        if batch_tf_indices is None and is_multitask_label_mode(label_mode):
            logits = select_multitask_axis(logits, model_tf_indices)
            if window_logits is not None:
                window_logits = select_multitask_axis(window_logits, model_tf_indices)
        if logits.shape != y.shape:
            raise ValueError(f"Logit/label shape mismatch: {logits.shape} vs {y.shape}")
        if logits.shape != hard_y.shape:
            raise ValueError(
                f"Logit/hard-label shape mismatch: {logits.shape} vs {hard_y.shape}"
            )
        loss_targets = hard_y if loss_name == "rank" else y
        localization_loss = dense_loss(
            logits,
            loss_targets,
            mask,
            batch_pos_weight,
            loss_name=loss_name,
            focal_gamma=focal_gamma,
            focal_alpha=focal_alpha,
            rank_temperature=rank_temperature,
            rank_negative_weight=rank_negative_weight,
            rank_negative_top_k=rank_negative_top_k,
        )
        loss = localization_loss
        candidate_loss_value = None
        if candidate_center_loss_weight > 0:
            candidate_rows = candidate_center_offsets >= 0
            if candidate_rows.any():
                candidate_offsets = candidate_center_offsets[candidate_rows]
                candidate_logits = logits[candidate_rows, 0, candidate_offsets]
                candidate_targets = y[candidate_rows, 0, candidate_offsets]
                candidate_loss_value = candidate_center_loss(
                    candidate_logits,
                    candidate_targets,
                    loss_name=loss_name,
                    focal_gamma=focal_gamma,
                    focal_alpha=focal_alpha,
                )
                loss = loss + candidate_center_loss_weight * candidate_loss_value
        window_targets = None
        window_loss_value = None
        if window_logits is not None:
            window_targets = window_targets_from_dense(hard_y, mask)
            if window_logits.shape != window_targets.shape:
                raise ValueError(
                    "Window logit/label shape mismatch: "
                    f"{window_logits.shape} vs {window_targets.shape}"
                )
            window_loss_value = window_bce_loss(window_logits, window_targets)
            loss = loss + window_loss_weight * window_loss_value
        loss.backward()
        optimizer.step()

        batch_size = x.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_examples += batch_size
        if window_loss_value is not None:
            window_loss_total += float(window_loss_value.detach().cpu()) * batch_size
            window_batches += batch_size
        if candidate_loss_value is not None:
            n_candidates = int((candidate_center_offsets >= 0).sum().item())
            candidate_loss_total += float(candidate_loss_value.detach().cpu()) * n_candidates
            candidate_examples += n_candidates

        with torch.no_grad():
            valid = mask.unsqueeze(1).expand_as(logits)
            pred = logits.sigmoid() >= 0.5
            target = hard_y >= 0.5
            tp += (pred & target & valid).sum().item()
            fp += (pred & ~target & valid).sum().item()
            fn += (~pred & target & valid).sum().item()
            predicted_positive += (pred & valid).sum().item()
            total_labels += valid.sum().item()
            if window_logits is not None and window_targets is not None:
                window_pred = window_logits.sigmoid() >= 0.5
                window_target = window_targets >= 0.5
                window_tp += (window_pred & window_target).sum().item()
                window_fp += (window_pred & ~window_target).sum().item()
                window_fn += (~window_pred & window_target).sum().item()
                window_predicted_positive += window_pred.sum().item()
                window_total_labels += window_target.numel()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    window_precision = (
        window_tp / (window_tp + window_fp)
        if (window_tp + window_fp)
        else None
    )
    window_recall = (
        window_tp / (window_tp + window_fn)
        if (window_tp + window_fn)
        else None
    )
    window_f1 = (
        2 * window_precision * window_recall / (window_precision + window_recall)
        if (
            window_precision is not None
            and window_recall is not None
            and (window_precision + window_recall)
        )
        else None
    )
    return DenseEpochStats(
        epoch=epoch,
        train_loss=total_loss / max(total_examples, 1),
        micro_precision=precision,
        micro_recall=recall,
        micro_f1=f1,
        predicted_positive_rate=predicted_positive / max(total_labels, 1),
        lr=current_lr(optimizer),
        window_loss=(
            window_loss_total / max(window_batches, 1)
            if window_batches
            else None
        ),
        window_micro_precision=window_precision,
        window_micro_recall=window_recall,
        window_micro_f1=window_f1,
        window_predicted_positive_rate=(
            window_predicted_positive / max(window_total_labels, 1)
            if window_total_labels
            else None
        ),
        candidate_center_loss=(
            candidate_loss_total / candidate_examples if candidate_examples else None
        ),
    )


def evaluate_dense(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    input_mode: str,
    pos_weight: torch.Tensor | None,
    model_tf_indices: torch.Tensor | None,
    merge_tf_indices: torch.Tensor | None,
    label_mode: str,
    loss_name: str,
    focal_gamma: float,
    focal_alpha: float | None,
    rank_temperature: float,
    rank_negative_weight: float,
    rank_negative_top_k: int,
    window_loss_weight: float,
    tf_names: list[str] | None = None,
    merged_label_name: str = "merged_train_tfs",
) -> DenseSplitStats:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    tp = fp = fn = 0.0
    dilated_tp = dilated_fp = dilated_fn = 0.0
    predicted_positive = 0.0
    total_labels = 0.0
    window_loss_total = 0.0
    window_batches = 0
    window_tp = window_fp = window_fn = 0.0
    window_predicted_positive = 0.0
    window_total_labels = 0.0
    micro_score_chunks: list[np.ndarray] = []
    micro_target_chunks: list[np.ndarray] = []
    dilated_target_chunks: list[np.ndarray] = []
    window_score_chunks: list[np.ndarray] = []
    window_target_chunks: list[np.ndarray] = []
    promoter_pair_score_chunks: list[np.ndarray] = []
    promoter_pair_target_chunks: list[np.ndarray] = []
    per_tf_score_chunks: list[list[np.ndarray]] | None = None
    per_tf_target_chunks: list[list[np.ndarray]] | None = None
    per_tf_dilated_target_chunks: list[list[np.ndarray]] | None = None
    per_tf_promoter_pair_score_chunks: list[list[np.ndarray]] | None = None
    per_tf_promoter_pair_target_chunks: list[list[np.ndarray]] | None = None
    selected_tf_names: list[str] | None = None

    if tf_names is not None:
        if label_mode == "merged_train_tfs":
            selected_tf_names = [merged_label_name]
        elif is_multitask_label_mode(label_mode):
            if model_tf_indices is None:
                selected_tf_names = [str(name) for name in tf_names]
            else:
                selected_tf_names = [
                    str(tf_names[int(idx)])
                    for idx in model_tf_indices.detach().cpu().numpy().tolist()
                ]
            selected_tf_names.append(merged_label_name)
        elif model_tf_indices is None:
            selected_tf_names = [str(name) for name in tf_names]
        else:
            selected_tf_names = [
                str(tf_names[int(idx)])
                for idx in model_tf_indices.detach().cpu().numpy().tolist()
            ]
        per_tf_score_chunks = [[] for _ in selected_tf_names]
        per_tf_target_chunks = [[] for _ in selected_tf_names]
        per_tf_dilated_target_chunks = [[] for _ in selected_tf_names]
        per_tf_promoter_pair_score_chunks = [[] for _ in selected_tf_names]
        per_tf_promoter_pair_target_chunks = [[] for _ in selected_tf_names]

    with torch.no_grad():
        for batch in loader:
            x = prepare_x(batch["x"].to(device, non_blocking=True), input_mode)
            y = batch["y"].to(device, non_blocking=True)
            hard_y = batch["hard_y"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            batch_tf_indices = batch.get("tf_idx")
            if batch_tf_indices is not None:
                batch_tf_indices = batch_tf_indices.to(device, non_blocking=True)

            if batch_tf_indices is not None:
                outputs = forward_dense_model(
                    model,
                    x,
                    batch_tf_indices,
                    pairwise_tf_indices=True,
                )
                batch_pos_weight = select_pos_weight_per_example(
                    pos_weight,
                    batch_tf_indices,
                )
                selected_tf_names = None
                per_tf_score_chunks = None
                per_tf_target_chunks = None
                per_tf_dilated_target_chunks = None
                per_tf_promoter_pair_score_chunks = None
                per_tf_promoter_pair_target_chunks = None
            else:
                outputs = forward_dense_model(
                    model,
                    x,
                    None if is_multitask_label_mode(label_mode) else model_tf_indices,
                )
                y = make_dense_targets(
                    y,
                    label_mode=label_mode,
                    model_tf_indices=model_tf_indices,
                    merge_tf_indices=merge_tf_indices,
                )
                hard_y = make_dense_targets(
                    hard_y,
                    label_mode=label_mode,
                    model_tf_indices=model_tf_indices,
                    merge_tf_indices=merge_tf_indices,
                )
                batch_pos_weight = (
                    select_tf_axis(pos_weight, model_tf_indices)
                    if pos_weight is not None and label_mode == "tf"
                    else None
                )
                if label_mode != "tf":
                    batch_pos_weight = pos_weight
                if pos_weight is not None and is_multitask_label_mode(label_mode):
                    batch_pos_weight = select_multitask_axis(pos_weight, model_tf_indices)
            logits, window_logits = split_dense_outputs(outputs)
            if batch_tf_indices is None and is_multitask_label_mode(label_mode):
                logits = select_multitask_axis(logits, model_tf_indices)
                if window_logits is not None:
                    window_logits = select_multitask_axis(window_logits, model_tf_indices)
            if logits.shape != y.shape:
                raise ValueError(f"Logit/label shape mismatch: {logits.shape} vs {y.shape}")
            if logits.shape != hard_y.shape:
                raise ValueError(
                    f"Logit/hard-label shape mismatch: {logits.shape} vs {hard_y.shape}"
                )
            loss_targets = hard_y if loss_name == "rank" else y
            localization_loss = dense_loss(
                logits,
                loss_targets,
                mask,
                batch_pos_weight,
                loss_name=loss_name,
                focal_gamma=focal_gamma,
                focal_alpha=focal_alpha,
                rank_temperature=rank_temperature,
                rank_negative_weight=rank_negative_weight,
                rank_negative_top_k=rank_negative_top_k,
            )
            loss = localization_loss
            window_targets = None
            window_loss_value = None
            if window_logits is not None:
                window_targets = window_targets_from_dense(hard_y, mask)
                if window_logits.shape != window_targets.shape:
                    raise ValueError(
                        "Window logit/label shape mismatch: "
                        f"{window_logits.shape} vs {window_targets.shape}"
                    )
                window_loss_value = window_bce_loss(window_logits, window_targets)
                loss = loss + window_loss_weight * window_loss_value

            batch_size = x.shape[0]
            total_loss += float(loss.detach().cpu()) * batch_size
            total_examples += batch_size
            if window_loss_value is not None:
                window_loss_total += float(window_loss_value.detach().cpu()) * batch_size
                window_batches += batch_size

            valid = mask.unsqueeze(1).expand_as(logits)
            pred = logits.sigmoid() >= 0.5
            target = hard_y >= 0.5
            dilated_target = y >= 0.5
            tp += (pred & target & valid).sum().item()
            fp += (pred & ~target & valid).sum().item()
            fn += (~pred & target & valid).sum().item()
            dilated_tp += (pred & dilated_target & valid).sum().item()
            dilated_fp += (pred & ~dilated_target & valid).sum().item()
            dilated_fn += (~pred & dilated_target & valid).sum().item()
            predicted_positive += (pred & valid).sum().item()
            total_labels += valid.sum().item()

            logits_cpu = logits.detach().cpu().numpy()
            target_cpu = target.detach().cpu().numpy()
            dilated_target_cpu = dilated_target.detach().cpu().numpy()
            valid_cpu = valid.detach().cpu().numpy()
            micro_score_chunks.append(logits_cpu[valid_cpu])
            micro_target_chunks.append(target_cpu[valid_cpu])
            dilated_target_chunks.append(dilated_target_cpu[valid_cpu])

            # This is a TransBind-like diagnostic only: a promoter-TF pair is
            # positive when the TF has any annotated site in that promoter.
            # Max pooling asks whether the dense model can surface at least one
            # high-scoring locus, without changing its base-wise objective.
            promoter_scores = logits.masked_fill(~valid, -torch.inf).amax(dim=-1)
            promoter_targets = (target & valid).any(dim=-1)
            promoter_scores_cpu = promoter_scores.detach().cpu().numpy()
            promoter_targets_cpu = promoter_targets.detach().cpu().numpy()
            promoter_pair_score_chunks.append(promoter_scores_cpu.reshape(-1))
            promoter_pair_target_chunks.append(promoter_targets_cpu.reshape(-1))
            if window_logits is not None and window_targets is not None:
                window_pred = window_logits.sigmoid() >= 0.5
                window_target = window_targets >= 0.5
                window_tp += (window_pred & window_target).sum().item()
                window_fp += (window_pred & ~window_target).sum().item()
                window_fn += (~window_pred & window_target).sum().item()
                window_predicted_positive += window_pred.sum().item()
                window_total_labels += window_target.numel()
                window_score_chunks.append(
                    window_logits.detach().cpu().numpy().reshape(-1)
                )
                window_target_chunks.append(
                    window_target.detach().cpu().numpy().reshape(-1)
                )
            if (
                per_tf_score_chunks is not None
                and per_tf_target_chunks is not None
                and per_tf_dilated_target_chunks is not None
                and per_tf_promoter_pair_score_chunks is not None
                and per_tf_promoter_pair_target_chunks is not None
            ):
                for local_tf_idx in range(logits_cpu.shape[1]):
                    tf_valid = valid_cpu[:, local_tf_idx, :]
                    per_tf_score_chunks[local_tf_idx].append(
                        logits_cpu[:, local_tf_idx, :][tf_valid]
                    )
                    per_tf_target_chunks[local_tf_idx].append(
                        target_cpu[:, local_tf_idx, :][tf_valid]
                    )
                    per_tf_dilated_target_chunks[local_tf_idx].append(
                        dilated_target_cpu[:, local_tf_idx, :][tf_valid]
                    )
                    per_tf_promoter_pair_score_chunks[local_tf_idx].append(
                        promoter_scores_cpu[:, local_tf_idx]
                    )
                    per_tf_promoter_pair_target_chunks[local_tf_idx].append(
                        promoter_targets_cpu[:, local_tf_idx]
                    )

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    average_precision, roc_auc = score_metrics(micro_score_chunks, micro_target_chunks)
    dilated_precision = (
        dilated_tp / (dilated_tp + dilated_fp)
        if (dilated_tp + dilated_fp)
        else 0.0
    )
    dilated_recall = (
        dilated_tp / (dilated_tp + dilated_fn)
        if (dilated_tp + dilated_fn)
        else 0.0
    )
    dilated_f1 = (
        2 * dilated_precision * dilated_recall / (dilated_precision + dilated_recall)
        if (dilated_precision + dilated_recall)
        else 0.0
    )
    dilated_average_precision, dilated_roc_auc = score_metrics(
        micro_score_chunks,
        dilated_target_chunks,
    )
    window_precision = (
        window_tp / (window_tp + window_fp)
        if (window_tp + window_fp)
        else None
    )
    window_recall = (
        window_tp / (window_tp + window_fn)
        if (window_tp + window_fn)
        else None
    )
    window_f1 = (
        2 * window_precision * window_recall / (window_precision + window_recall)
        if (
            window_precision is not None
            and window_recall is not None
            and (window_precision + window_recall)
        )
        else None
    )
    window_average_precision, window_roc_auc = score_metrics(
        window_score_chunks,
        window_target_chunks,
    )
    promoter_pair_average_precision, promoter_pair_roc_auc = score_metrics(
        promoter_pair_score_chunks,
        promoter_pair_target_chunks,
    )
    per_tf_metrics = None
    promoter_pair_per_tf = None
    if (
        selected_tf_names is not None
        and per_tf_score_chunks is not None
        and per_tf_target_chunks is not None
        and per_tf_dilated_target_chunks is not None
        and per_tf_promoter_pair_score_chunks is not None
        and per_tf_promoter_pair_target_chunks is not None
    ):
        per_tf_metrics = []
        promoter_pair_per_tf = []
        for name, score_chunk, target_chunk, dilated_target_chunk in zip(
            selected_tf_names,
            per_tf_score_chunks,
            per_tf_target_chunks,
            per_tf_dilated_target_chunks,
        ):
            tf_ap, tf_auc = score_metrics(score_chunk, target_chunk)
            tf_dilated_ap, tf_dilated_auc = score_metrics(
                score_chunk,
                dilated_target_chunk,
            )
            n_pos = int(sum(chunk.astype(bool, copy=False).sum() for chunk in target_chunk))
            n_dilated_pos = int(
                sum(
                    chunk.astype(bool, copy=False).sum()
                    for chunk in dilated_target_chunk
                )
            )
            n_total = int(sum(len(chunk) for chunk in target_chunk))
            per_tf_metrics.append(
                {
                    "tf_name": name,
                    "average_precision": tf_ap,
                    "roc_auc": tf_auc,
                    "positives": n_pos,
                    "dilated_average_precision": tf_dilated_ap,
                    "dilated_roc_auc": tf_dilated_auc,
                    "dilated_positives": n_dilated_pos,
                    "valid_positions": n_total,
                }
            )
        for name, score_chunk, target_chunk in zip(
            selected_tf_names,
            per_tf_promoter_pair_score_chunks,
            per_tf_promoter_pair_target_chunks,
        ):
            tf_ap, tf_auc = score_metrics(score_chunk, target_chunk)
            n_positive = int(
                sum(chunk.astype(bool, copy=False).sum() for chunk in target_chunk)
            )
            n_pairs = int(sum(len(chunk) for chunk in target_chunk))
            promoter_pair_per_tf.append(
                {
                    "tf_name": name,
                    "average_precision": tf_ap,
                    "roc_auc": tf_auc,
                    "positive_promoters": n_positive,
                    "promoter_tf_pairs": n_pairs,
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
        dilated_micro_precision=dilated_precision,
        dilated_micro_recall=dilated_recall,
        dilated_micro_f1=dilated_f1,
        dilated_average_precision=dilated_average_precision,
        dilated_roc_auc=dilated_roc_auc,
        per_tf=per_tf_metrics,
        window_loss=(
            window_loss_total / max(window_batches, 1)
            if window_batches
            else None
        ),
        window_micro_precision=window_precision,
        window_micro_recall=window_recall,
        window_micro_f1=window_f1,
        window_predicted_positive_rate=(
            window_predicted_positive / max(window_total_labels, 1)
            if window_total_labels
            else None
        ),
        window_average_precision=window_average_precision,
        window_roc_auc=window_roc_auc,
        promoter_pair_average_precision=promoter_pair_average_precision,
        promoter_pair_roc_auc=promoter_pair_roc_auc,
        promoter_pair_per_tf=promoter_pair_per_tf,
    )


def attach_val_stats(stats: DenseEpochStats, val_stats: DenseSplitStats) -> DenseEpochStats:
    stats.val_loss = val_stats.loss
    stats.val_micro_precision = val_stats.micro_precision
    stats.val_micro_recall = val_stats.micro_recall
    stats.val_micro_f1 = val_stats.micro_f1
    stats.val_predicted_positive_rate = val_stats.predicted_positive_rate
    stats.val_average_precision = val_stats.average_precision
    stats.val_roc_auc = val_stats.roc_auc
    stats.val_dilated_micro_precision = val_stats.dilated_micro_precision
    stats.val_dilated_micro_recall = val_stats.dilated_micro_recall
    stats.val_dilated_micro_f1 = val_stats.dilated_micro_f1
    stats.val_dilated_average_precision = val_stats.dilated_average_precision
    stats.val_dilated_roc_auc = val_stats.dilated_roc_auc
    stats.val_window_loss = val_stats.window_loss
    stats.val_window_micro_precision = val_stats.window_micro_precision
    stats.val_window_micro_recall = val_stats.window_micro_recall
    stats.val_window_micro_f1 = val_stats.window_micro_f1
    stats.val_window_predicted_positive_rate = (
        val_stats.window_predicted_positive_rate
    )
    stats.val_window_average_precision = val_stats.window_average_precision
    stats.val_window_roc_auc = val_stats.window_roc_auc
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
    if args.shuffle_tf_embeddings:
        if tf_embeddings is None:
            raise ValueError("--shuffle-tf-embeddings requires TF embeddings")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(args.seed)
        permutation = torch.randperm(tf_embeddings.shape[0], generator=generator)
        tf_embeddings = tf_embeddings.index_select(0, permutation)
        if tf_embedding_metadata is None:
            tf_embedding_metadata = {}
        tf_embedding_metadata["shuffled"] = True
        tf_embedding_metadata["shuffle_seed"] = args.seed
        tf_embedding_metadata["shuffle_permutation"] = permutation.tolist()
        print(
            "Shuffled TF embeddings:",
            {"seed": args.seed, "n_tfs": int(tf_embeddings.shape[0])},
        )
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
    align_dna_branch_args_from_checkpoint(args, input_channels=input_channels)
    dilations = parse_dilations(args.dilations)
    if args.label_mode == "merged_train_tfs":
        output_channels = 1
    elif is_multitask_label_mode(args.label_mode):
        output_channels = len(dataset.tf_names) + 1
    else:
        output_channels = len(dataset.tf_names)
    print("Dataset:", dataset.summary())
    print("Promoter split:", promoter_split["counts"])
    print("TF split:", tf_split["counts"])
    print("Input channels:", input_channels)
    print("Output channels:", output_channels)
    print("Label mode:", args.label_mode)
    print("Loss:", args.loss)
    print("Device:", device)
    print("Model:", args.model)

    if args.training_window_mode == "sampled_windows":
        train_dataset = BindingBenchSampledPromoterWindowDataset(
            dataset,
            promoter_indices=train_indices,
            tf_indices=train_tf_indices,
            window_size=args.sampled_window_size,
            samples_per_epoch=args.sampled_window_samples_per_epoch,
            positive_fraction=args.sampled_window_positive_fraction,
            hard_negative_fraction=args.sampled_window_hard_negative_fraction,
            candidate_fraction=args.sampled_window_candidate_fraction,
            candidate_windows_path=args.candidate_windows_path,
            candidate_tf_positive_fraction=args.candidate_tf_positive_fraction,
            margin_bp=args.sampled_window_margin_bp,
            negative_exclusion_bp=args.sampled_window_negative_exclusion_bp,
            seed=args.seed,
            max_sampling_attempts=args.sampled_window_max_attempts,
        )
        train_shuffle = False
        print("Training window sampler:", train_dataset.summary())
    else:
        train_dataset = Subset(dataset, train_indices)
        train_shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_shuffle,
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
        n_tfs=output_channels,
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
        protein_delta_gate_logit_init=args.protein_delta_gate_logit_init,
        scorer=args.scorer,
        scorer_heads=args.scorer_heads,
        scorer_pair_dim=args.scorer_pair_dim,
        scorer_hidden_dim=args.scorer_hidden_dim,
        scorer_bias_mode=args.scorer_bias_mode,
        film_eval_tf_chunk_size=args.film_eval_tf_chunk_size,
        window_pooling=args.window_pooling,
        window_pooling_top_k=args.window_pooling_top_k,
    ).to(device)

    evaluation_checkpoint: dict[str, object] | None = None
    if args.evaluate_only:
        evaluation_checkpoint = load_checkpoint_cpu(args.output_dir / "best.pt")
        model.load_state_dict(evaluation_checkpoint["model_state_dict"])
        print(f"Loaded existing best checkpoint for evaluation: {args.output_dir / 'best.pt'}")
    elif args.resume_checkpoint is not None:
        load_resume_checkpoint(
            model,
            args.resume_checkpoint,
            expected_model_name=args.model,
            expected_tf_names=dataset.tf_names,
        )
    elif args.pretrained_dna_checkpoint is not None:
        load_pretrained_dna_branch(model, args.pretrained_dna_checkpoint, device)
    if args.resume_checkpoint is not None:
        configure_dna_finetuning(model, args.dna_finetune_mode)
    elif args.freeze_dna_branch and not args.evaluate_only:
        freeze_dna_branch(model)
        print("Frozen DNA branch.")
    print("Trainable parameters:", count_parameters(model))

    optimizer = build_optimizer(model, args)
    scheduler = build_lr_scheduler(args, optimizer)
    if args.no_pos_weight:
        pos_weight = None
    elif args.label_mode == "merged_train_tfs":
        pos_weight = compute_merged_pos_weight(
            dataset,
            device,
            indices=train_indices,
            tf_indices=train_tf_indices if use_tf_holdout else None,
        )
    elif is_multitask_label_mode(args.label_mode):
        pos_weight = compute_multitask_pos_weight(
            dataset,
            device,
            indices=train_indices,
            tf_indices=train_tf_indices if use_tf_holdout else None,
        )
    else:
        pos_weight = compute_dense_pos_weight(dataset, device, indices=train_indices)
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

    if not args.evaluate_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        save_json(args.output_dir / "tf_names.json", dataset.tf_names)
        save_json(args.output_dir / "dataset_summary.json", dataset.summary())
        if hasattr(train_dataset, "summary"):
            save_json(args.output_dir / "train_dataset_summary.json", train_dataset.summary())
        save_json(args.output_dir / "args.json", serializable_args(args))
        save_json(
            args.output_dir / "model_config.json",
            model_config_from_args(args, input_channels, output_channels),
        )
        split_out = args.promoter_split_out or (args.output_dir / "promoter_split.json")
        save_promoter_split(split_out, promoter_split)
        print(f"Saved promoter split: {split_out}")
        tf_split_out = args.tf_split_out or (args.output_dir / "tf_split.json")
        save_tf_split(tf_split_out, tf_split)
        print(f"Saved TF split: {tf_split_out}")
        if tf_embedding_metadata is not None:
            save_json(args.output_dir / "tf_embedding_metadata.json", tf_embedding_metadata)
    else:
        print("Evaluation-only mode: preserving existing run metadata.")

    validation_loader = val_loader
    validation_model_tf_tensor = val_tf_tensor if use_tf_holdout and args.label_mode == "tf" else None
    validation_merge_tf_tensor = train_tf_tensor if args.label_mode == "merged_train_tfs" else None
    validation_merged_label_name = "merged_train_tfs"
    validation_description = "validation promoters"
    if is_multitask_label_mode(args.label_mode):
        validation_model_tf_tensor = train_tf_tensor if use_tf_holdout else None
        validation_merge_tf_tensor = train_tf_tensor if use_tf_holdout else None
        validation_description = (
            "validation promoters with train TF heads and merged train TF union"
        )
    if use_tf_holdout and args.label_mode == "tf":
        if validation_model_tf_tensor is not None and validation_loader is None:
            validation_loader = train_eval_loader
            validation_description = "train promoters with held-out validation TFs"
        elif validation_model_tf_tensor is not None:
            validation_description = "validation promoters with held-out validation TFs"
    elif use_tf_holdout and args.label_mode == "merged_train_tfs":
        if val_tf_tensor is not None:
            validation_merge_tf_tensor = val_tf_tensor
            validation_merged_label_name = "merged_val_tfs"
            if validation_loader is None:
                validation_loader = train_eval_loader
                validation_description = (
                    "train promoters with held-out validation TF union"
                )
            else:
                validation_description = (
                    "validation promoters with held-out validation TF union"
                )

    use_val_for_selection = (
        validation_loader is not None
        and args.eval_every > 0
        and (args.label_mode != "tf" or not use_tf_holdout or validation_model_tf_tensor is not None)
    )
    best_metric_name = args.selection_metric
    if best_metric_name.startswith("val_") and not use_val_for_selection:
        print(
            f"Selection metric {best_metric_name!r} needs validation; "
            "falling back to 'train_loss'."
        )
        best_metric_name = "train_loss"
    elif best_metric_name.startswith("val_"):
        print(f"Selection metric {best_metric_name!r} uses {validation_description}.")
    best_metric_value: float | None = None
    if evaluation_checkpoint is not None:
        checkpoint_stats = _checkpoint_dict(evaluation_checkpoint, "stats")
        checkpoint_value = checkpoint_stats.get(best_metric_name)
        if checkpoint_value is not None:
            best_metric_value = float(checkpoint_value)
    history: list[dict[str, object]] = []
    epoch_iterable = () if args.evaluate_only else range(1, args.epochs + 1)
    for epoch in epoch_iterable:
        stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            input_mode=args.input_mode,
            pos_weight=pos_weight,
            model_tf_indices=(
                train_tf_tensor
                if args.label_mode == "tf" or is_multitask_label_mode(args.label_mode)
                else None
            ),
            merge_tf_indices=(
                train_tf_tensor
                if args.label_mode == "merged_train_tfs"
                or is_multitask_label_mode(args.label_mode)
                else None
            ),
            label_mode=args.label_mode,
            loss_name=args.loss,
            focal_gamma=args.focal_gamma,
            focal_alpha=args.focal_alpha,
            rank_temperature=args.rank_temperature,
            rank_negative_weight=args.rank_negative_weight,
            rank_negative_top_k=args.rank_negative_top_k,
            window_loss_weight=args.window_loss_weight,
            candidate_center_loss_weight=args.candidate_center_loss_weight,
            epoch=epoch,
        )
        if use_val_for_selection and (
            epoch % args.eval_every == 0 or epoch == args.epochs
        ):
            stats = attach_val_stats(
                stats,
                evaluate_dense(
                    model=model,
                    loader=validation_loader,
                    device=device,
                    input_mode=args.input_mode,
                    pos_weight=(
                        None if use_tf_holdout and args.label_mode == "tf" else pos_weight
                    ),
                    model_tf_indices=validation_model_tf_tensor,
                    merge_tf_indices=validation_merge_tf_tensor,
                    label_mode=args.label_mode,
                    loss_name=args.loss,
                    focal_gamma=args.focal_gamma,
                    focal_alpha=args.focal_alpha,
                    rank_temperature=args.rank_temperature,
                    rank_negative_weight=args.rank_negative_weight,
                    rank_negative_top_k=args.rank_negative_top_k,
                    window_loss_weight=args.window_loss_weight,
                    tf_names=dataset.tf_names,
                    merged_label_name=validation_merged_label_name,
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
        if stats.window_loss is not None:
            message += (
                f" window_loss={stats.window_loss:.5f} "
                f"window_f1={stats.window_micro_f1 or 0.0:.4f} "
                f"window_pred_pos={stats.window_predicted_positive_rate or 0.0:.5f}"
            )
        if stats.candidate_center_loss is not None:
            message += f" candidate_center_loss={stats.candidate_center_loss:.5f}"
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
            if stats.val_dilated_micro_f1 is not None:
                message += f" val_dilated_f1={stats.val_dilated_micro_f1:.4f}"
            if stats.val_dilated_average_precision is not None:
                message += f" val_dilated_ap={stats.val_dilated_average_precision:.4f}"
            if stats.val_dilated_roc_auc is not None:
                message += f" val_dilated_roc_auc={stats.val_dilated_roc_auc:.4f}"
            if stats.val_window_loss is not None:
                message += (
                    f" val_window_loss={stats.val_window_loss:.5f} "
                    f"val_window_f1={stats.val_window_micro_f1 or 0.0:.4f} "
                    f"val_window_pred_pos={stats.val_window_predicted_positive_rate or 0.0:.5f}"
                )
                if stats.val_window_average_precision is not None:
                    message += f" val_window_ap={stats.val_window_average_precision:.4f}"
                if stats.val_window_roc_auc is not None:
                    message += f" val_window_roc_auc={stats.val_window_roc_auc:.4f}"
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
            output_channels=output_channels,
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
                output_channels=output_channels,
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
                output_channels=output_channels,
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
    if use_tf_holdout and args.label_mode == "tf":
        final_jobs = {
            "train_promoters_train_tfs": (
                train_eval_loader,
                train_tf_tensor,
                None,
                "train_tfs",
            ),
            "test_promoters_train_tfs": (
                test_loader,
                train_tf_tensor,
                None,
                "train_tfs",
            ),
        }
        if val_tf_tensor is not None:
            final_jobs["train_promoters_val_tfs"] = (
                train_eval_loader,
                val_tf_tensor,
                None,
                "val_tfs",
            )
            final_jobs["val_promoters_val_tfs"] = (
                val_loader,
                val_tf_tensor,
                None,
                "val_tfs",
            )
        if test_tf_tensor is not None:
            final_jobs["train_promoters_test_tfs"] = (
                train_eval_loader,
                test_tf_tensor,
                None,
                "test_tfs",
            )
            final_jobs["test_promoters_test_tfs"] = (
                test_loader,
                test_tf_tensor,
                None,
                "test_tfs",
            )
        eval_pos_weight = None
    elif use_tf_holdout and args.label_mode == "merged_train_tfs":
        final_jobs = {
            "train_promoters_train_tfs": (
                train_eval_loader,
                None,
                train_tf_tensor,
                "merged_train_tfs",
            ),
            "val_promoters_train_tfs": (
                val_loader,
                None,
                train_tf_tensor,
                "merged_train_tfs",
            ),
            "test_promoters_train_tfs": (
                test_loader,
                None,
                train_tf_tensor,
                "merged_train_tfs",
            ),
        }
        if val_tf_tensor is not None:
            final_jobs["train_promoters_val_tfs"] = (
                train_eval_loader,
                None,
                val_tf_tensor,
                "merged_val_tfs",
            )
            final_jobs["val_promoters_val_tfs"] = (
                val_loader,
                None,
                val_tf_tensor,
                "merged_val_tfs",
            )
            final_jobs["test_promoters_val_tfs"] = (
                test_loader,
                None,
                val_tf_tensor,
                "merged_val_tfs",
            )
        if test_tf_tensor is not None:
            final_jobs["train_promoters_test_tfs"] = (
                train_eval_loader,
                None,
                test_tf_tensor,
                "merged_test_tfs",
            )
            final_jobs["val_promoters_test_tfs"] = (
                val_loader,
                None,
                test_tf_tensor,
                "merged_test_tfs",
            )
            final_jobs["test_promoters_test_tfs"] = (
                test_loader,
                None,
                test_tf_tensor,
                "merged_test_tfs",
            )
        eval_pos_weight = pos_weight
    elif use_tf_holdout and is_multitask_label_mode(args.label_mode):
        final_jobs = {
            "train_promoters_train_tfs_plus_merged_train_tfs": (
                train_eval_loader,
                train_tf_tensor,
                train_tf_tensor,
                "merged_train_tfs",
            ),
            "val_promoters_train_tfs_plus_merged_train_tfs": (
                val_loader,
                train_tf_tensor,
                train_tf_tensor,
                "merged_train_tfs",
            ),
            "test_promoters_train_tfs_plus_merged_train_tfs": (
                test_loader,
                train_tf_tensor,
                train_tf_tensor,
                "merged_train_tfs",
            ),
        }
        eval_pos_weight = pos_weight
    else:
        final_jobs = {
            "train": (train_eval_loader, None, None, "merged_train_tfs"),
            "val": (val_loader, None, None, "merged_train_tfs"),
            "test": (test_loader, None, None, "merged_train_tfs"),
        }
        eval_pos_weight = pos_weight

    if args.final_eval_scope == "test_only":
        preferred_test_job_names = ("test_promoters_test_tfs", "test")
        final_jobs = {
            name: final_jobs[name]
            for name in preferred_test_job_names
            if name in final_jobs
        }
        if not final_jobs:
            raise ValueError(
                "--final-eval-scope test_only requires a test evaluation slice"
            )
    print("Final evaluation slices:", list(final_jobs))

    for split_name, (
        split_loader,
        split_tf_tensor,
        split_merge_tf_tensor,
        merged_label_name,
    ) in final_jobs.items():
        if split_loader is None:
            continue
        if use_tf_holdout and args.label_mode == "tf" and split_tf_tensor is None:
            continue
        if (
            use_tf_holdout
            and (
                args.label_mode == "merged_train_tfs"
                or is_multitask_label_mode(args.label_mode)
            )
            and split_merge_tf_tensor is None
        ):
            continue
        split_stats = evaluate_dense(
            model=model,
            loader=split_loader,
            device=device,
            input_mode=args.input_mode,
            pos_weight=eval_pos_weight,
            model_tf_indices=(
                split_tf_tensor
                if args.label_mode == "tf" or is_multitask_label_mode(args.label_mode)
                else None
            ),
            merge_tf_indices=(
                split_merge_tf_tensor
                if args.label_mode == "merged_train_tfs"
                or is_multitask_label_mode(args.label_mode)
                else None
            ),
            label_mode=args.label_mode,
            loss_name=args.loss,
            focal_gamma=args.focal_gamma,
            focal_alpha=args.focal_alpha,
            rank_temperature=args.rank_temperature,
            rank_negative_weight=args.rank_negative_weight,
            rank_negative_top_k=args.rank_negative_top_k,
            window_loss_weight=args.window_loss_weight,
            tf_names=dataset.tf_names,
            merged_label_name=merged_label_name,
        )
        final_metrics[split_name] = asdict(split_stats)
    save_json(args.output_dir / "final_metrics.json", final_metrics)
    print("Final metrics:", final_metrics)
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
