from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import polars as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from .dataset import BindingBenchWindowDataset, DEFAULT_REGIONS_PATH, DEFAULT_SITES_PATH
    from .model import MODEL_NAMES, build_model, parse_dilations
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
    from dataset import BindingBenchWindowDataset, DEFAULT_REGIONS_PATH, DEFAULT_SITES_PATH
    from model import MODEL_NAMES, build_model, parse_dilations
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


@dataclass
class EpochStats:
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
class WindowSplitStats:
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
        description="Train a supervised CNN on Binding Bench windows."
    )
    parser.add_argument("--sites-path", type=Path, default=DEFAULT_SITES_PATH)
    parser.add_argument("--regions-path", type=Path, default=DEFAULT_REGIONS_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Directory for checkpoints and metadata. Defaults to a model-specific "
            "directory under the project working area."
        ),
    )
    parser.add_argument("--window-size", type=int, default=101)
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--negative-exclusion-bp", type=int)
    parser.add_argument("--min-sites-per-tf", type=int, default=15)
    parser.add_argument("--max-positive-windows", type=int)
    parser.add_argument(
        "--sequence-orientation",
        choices=("strand-aware", "genomic"),
        default="strand-aware",
    )
    parser.add_argument(
        "--model",
        choices=MODEL_NAMES,
        default="small_cnn",
        help="Model architecture to train.",
    )
    parser.add_argument(
        "--hidden-channels",
        type=int,
        default=128,
        help="Hidden channel count for res_dilated_cnn.",
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=7,
        help="Residual block kernel size for res_dilated_cnn. Must be odd.",
    )
    parser.add_argument(
        "--dilations",
        default="1,2,4,8,16",
        help="Comma-separated residual block dilations for res_dilated_cnn.",
    )
    parser.add_argument(
        "--transbind-dilations",
        default="1,2,4",
        help="Comma-separated residual block dilations for transbind_lite.",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout probability for res_dilated_cnn/transbind_lite.",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
        help="Cross-attention head count for transbind_lite.",
    )
    parser.add_argument(
        "--tf-embeddings-path",
        type=Path,
        help="Parquet file with one protein embedding row per TF for transbind_lite.",
    )
    parser.add_argument(
        "--tf-embedding-key-column",
        help="Column used to match dataset TF labels to embedding rows. Defaults to auto.",
    )
    parser.add_argument(
        "--tf-embedding-column",
        default="emb",
        help="Column containing list-valued protein embeddings.",
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
            "For transbind_lite, filter the training dataset to TF labels that have "
            "protein embeddings instead of failing on missing labels."
        ),
    )
    parser.add_argument(
        "--tf-name-filter-from-embeddings",
        action="store_true",
        help=(
            "Filter any supervised model to TF labels present in the embedding table. "
            "Useful for fair comparison against transbind_lite."
        ),
    )
    parser.add_argument(
        "--exclude-tf-names",
        help=(
            "Comma-separated TF names to hold out from supervised training. "
            "Use with transbind_lite to test protein-embedding generalization."
        ),
    )
    parser.add_argument(
        "--exclude-tf-names-path",
        type=Path,
        help="File with TF names to hold out, either JSON list or one name per line.",
    )
    parser.add_argument(
        "--tf-split-mode",
        choices=("none", "random", "named", "similarity", "named_similarity"),
        default="none",
        help=(
            "Split TF labels into train/val/test subsets. For transbind_lite, "
            "the model keeps all TF embeddings but the loss is computed only on "
            "train TFs."
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
        help="Cosine-similarity threshold for --tf-split-mode similarity.",
    )
    parser.add_argument(
        "--transbind-no-max-pool",
        action="store_true",
        help="Disable the early MaxPool1d layer in transbind_lite.",
    )
    parser.add_argument(
        "--transbind-tf-bias",
        action="store_true",
        help="Add a learned per-TF bias term to transbind_lite logits.",
    )
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--lr-scheduler",
        choices=("none", "cosine", "plateau"),
        default="none",
        help="Optional learning-rate schedule.",
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=1e-5,
        help="Minimum learning rate for cosine/plateau schedules.",
    )
    parser.add_argument(
        "--plateau-patience",
        type=int,
        default=10,
        help="Epochs without loss improvement before plateau LR decay.",
    )
    parser.add_argument(
        "--plateau-factor",
        type=float,
        default=0.5,
        help="Multiplicative LR decay factor for plateau scheduling.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="Evaluate validation TF split every N epochs. Use 0 to disable.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=(
            "train_loss",
            "val_loss",
            "val_average_precision",
            "val_roc_auc",
            "val_micro_f1",
        ),
        default="train_loss",
        help="Metric used to choose best.pt.",
    )
    parser.add_argument(
        "--no-pos-weight",
        action="store_true",
        help="Disable BCE positive-class weighting.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="Save an epoch checkpoint every N epochs; 0 disables periodic checkpoints.",
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = DEFAULT_MODEL_ROOT / f"{args.model}_overfit"
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_tf_names_arg(value: str) -> list[str]:
    names = [part.strip() for part in value.split(",") if part.strip()]
    if not names:
        raise ValueError("--exclude-tf-names did not contain any TF names")
    return names


def read_tf_names(path: Path) -> list[str]:
    text = path.read_text().strip()
    if not text:
        raise ValueError(f"No TF names found in {path}")
    if text[0] in "[{":
        payload = json.loads(text)
        if isinstance(payload, dict):
            if "tf_names" in payload:
                payload = payload["tf_names"]
            elif "names" in payload:
                payload = payload["names"]
            elif "test_tf_names" in payload or "val_tf_names" in payload:
                payload = list(payload.get("val_tf_names", [])) + list(
                    payload.get("test_tf_names", [])
                )
        if not isinstance(payload, list):
            raise ValueError(
                "Expected a JSON list, object with tf_names/names, or TF split "
                f"object with val_tf_names/test_tf_names, in {path}"
            )
        names = [str(name).strip() for name in payload if str(name).strip()]
    else:
        names = [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    if not names:
        raise ValueError(f"No TF names found in {path}")
    return names


def resolve_excluded_tf_names(args: argparse.Namespace) -> set[str]:
    if args.exclude_tf_names and args.exclude_tf_names_path:
        raise ValueError(
            "Use either --exclude-tf-names or --exclude-tf-names-path, not both"
        )
    if args.exclude_tf_names:
        names = parse_tf_names_arg(args.exclude_tf_names)
    elif args.exclude_tf_names_path:
        names = read_tf_names(args.exclude_tf_names_path)
    else:
        names = []
    return {name.upper() for name in names}


def site_tf_names(path: Path) -> set[str]:
    table = pl.read_parquet(path, columns=["name"])
    return {str(name).upper() for name in table.get_column("name").unique().to_list()}


def get_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def compute_pos_weight(
    dataset: BindingBenchWindowDataset,
    device: torch.device,
    tf_indices: list[int] | None = None,
) -> torch.Tensor:
    labels = np.stack([record.labels for record in dataset.records])
    if tf_indices is not None:
        labels = labels[:, tf_indices]
    positives = labels.sum(axis=0)
    negatives = labels.shape[0] - positives
    weights = negatives / np.maximum(positives, 1.0)
    weights = np.clip(weights, 1.0, 100.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def tensor_indices(indices: list[int], device: torch.device) -> torch.Tensor | None:
    if not indices:
        return None
    return torch.tensor(indices, dtype=torch.long, device=device)


def build_tf_split(
    args: argparse.Namespace,
    dataset: BindingBenchWindowDataset,
    *,
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
            raise ValueError("--tf-split-mode similarity requires TF embeddings")
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
            raise ValueError("--tf-split-mode named_similarity requires TF embeddings")
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


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def serializable_args(args: argparse.Namespace) -> dict[str, object]:
    out = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def model_config_from_args(args: argparse.Namespace) -> dict[str, object]:
    return {
        "model_name": args.model,
        "hidden_channels": args.hidden_channels,
        "kernel_size": args.kernel_size,
        "dropout": args.dropout,
        "dilations": list(parse_dilations(args.dilations)),
        "transbind_dilations": list(parse_dilations(args.transbind_dilations)),
        "transbind_use_max_pool": not args.transbind_no_max_pool,
        "transbind_tf_bias": args.transbind_tf_bias,
        "num_heads": args.num_heads,
        "tf_embeddings_path": str(args.tf_embeddings_path) if args.tf_embeddings_path else None,
        "tf_embedding_key_column": args.tf_embedding_key_column,
        "tf_embedding_column": args.tf_embedding_column,
        "tf_name_map": str(args.tf_name_map) if args.tf_name_map else None,
        "tf_split_mode": args.tf_split_mode,
        "tf_split_path": str(args.tf_split_path) if args.tf_split_path else None,
        "tf_train_fraction": args.tf_train_fraction,
        "tf_val_fraction": args.tf_val_fraction,
        "tf_val_names": args.tf_val_names,
        "tf_test_names": args.tf_test_names,
        "tf_similarity_threshold": args.tf_similarity_threshold,
    }


def count_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def build_lr_scheduler(
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer,
) -> torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None:
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


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    dataset: BindingBenchWindowDataset,
    stats: EpochStats,
    tf_split: dict[str, object] | None = None,
    tf_embedding_metadata: dict[str, object] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_name": args.model,
            "model_config": model_config_from_args(args),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "args": serializable_args(args),
            "tf_names": dataset.tf_names,
            "dataset_summary": dataset.summary(),
            "tf_split": tf_split,
            "tf_embedding_metadata": tf_embedding_metadata,
            "stats": asdict(stats),
        },
        path,
    )


def forward_window_model(
    model: torch.nn.Module,
    x: torch.Tensor,
    tf_indices: torch.Tensor | None,
) -> torch.Tensor:
    if tf_indices is not None and getattr(model, "supports_tf_indices", False):
        return model(x, tf_indices=tf_indices)
    logits = model(x)
    if tf_indices is not None:
        logits = logits.index_select(1, tf_indices)
    return logits


def select_tf_axis(tensor: torch.Tensor, tf_indices: torch.Tensor | None) -> torch.Tensor:
    if tf_indices is None:
        return tensor
    return tensor.index_select(1, tf_indices)


def binary_average_precision(scores: np.ndarray, targets: np.ndarray) -> float | None:
    targets_bool = targets.astype(bool, copy=False)
    n_pos = int(targets_bool.sum())
    if n_pos == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    sorted_targets = targets_bool[order]
    tp_cumsum = np.cumsum(sorted_targets, dtype=np.float64)
    ranks = np.arange(1, len(sorted_targets) + 1, dtype=np.float64)
    precision_at_k = tp_cumsum / ranks
    return float(precision_at_k[sorted_targets].sum() / n_pos)


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


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pos_weight: torch.Tensor | None,
    tf_indices: torch.Tensor | None,
    epoch: int,
) -> EpochStats:
    model.train()
    total_loss = 0.0
    total_examples = 0
    tp = fp = fn = 0.0
    predicted_positive = 0.0
    total_labels = 0.0

    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = forward_window_model(model, x, tf_indices)
        y = select_tf_axis(y, tf_indices)
        loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)
        loss.backward()
        optimizer.step()

        batch_size = x.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_examples += batch_size

        with torch.no_grad():
            pred = logits.sigmoid() >= 0.5
            target = y >= 0.5
            tp += (pred & target).sum().item()
            fp += (pred & ~target).sum().item()
            fn += (~pred & target).sum().item()
            predicted_positive += pred.sum().item()
            total_labels += pred.numel()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return EpochStats(
        epoch=epoch,
        train_loss=total_loss / max(total_examples, 1),
        micro_precision=precision,
        micro_recall=recall,
        micro_f1=f1,
        predicted_positive_rate=predicted_positive / max(total_labels, 1),
        lr=current_lr(optimizer),
    )


def evaluate_window(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    pos_weight: torch.Tensor | None,
    tf_indices: torch.Tensor | None,
    tf_names: list[str] | None = None,
) -> WindowSplitStats:
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
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            logits = forward_window_model(model, x, tf_indices)
            y = select_tf_axis(y, tf_indices)
            loss = F.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)

            batch_size = x.shape[0]
            total_loss += float(loss.detach().cpu()) * batch_size
            total_examples += batch_size

            pred = logits.sigmoid() >= 0.5
            target = y >= 0.5
            tp += (pred & target).sum().item()
            fp += (pred & ~target).sum().item()
            fn += (~pred & target).sum().item()
            predicted_positive += pred.sum().item()
            total_labels += pred.numel()

            logits_cpu = logits.detach().cpu().numpy()
            target_cpu = target.detach().cpu().numpy()
            micro_score_chunks.append(logits_cpu.reshape(-1))
            micro_target_chunks.append(target_cpu.reshape(-1))
            if (
                per_tf_score_chunks is not None
                and per_tf_target_chunks is not None
            ):
                for local_tf_idx in range(logits_cpu.shape[1]):
                    per_tf_score_chunks[local_tf_idx].append(logits_cpu[:, local_tf_idx])
                    per_tf_target_chunks[local_tf_idx].append(target_cpu[:, local_tf_idx])

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
                    "valid_windows": n_total,
                }
            )
    return WindowSplitStats(
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


def attach_val_stats(stats: EpochStats, val_stats: WindowSplitStats) -> EpochStats:
    stats.val_loss = val_stats.loss
    stats.val_micro_precision = val_stats.micro_precision
    stats.val_micro_recall = val_stats.micro_recall
    stats.val_micro_f1 = val_stats.micro_f1
    stats.val_predicted_positive_rate = val_stats.predicted_positive_rate
    stats.val_average_precision = val_stats.average_precision
    stats.val_roc_auc = val_stats.roc_auc
    return stats


def selection_value(stats: EpochStats, metric: str) -> float | None:
    value = getattr(stats, metric)
    if value is None:
        return None
    return float(value)


def selection_is_better(
    value: float,
    best_value: float | None,
    metric: str,
) -> bool:
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
        if args.tf_embeddings_path is None:
            raise ValueError(
                "--tf-embeddings-path is required when filtering TFs to embeddings"
            )
        tf_name_filter = available_embedding_keys(
            args.tf_embeddings_path,
            key_column=args.tf_embedding_key_column,
            name_mapping_path=args.tf_name_map,
        )
    excluded_tf_names = resolve_excluded_tf_names(args)
    if excluded_tf_names:
        if tf_name_filter is None:
            tf_name_filter = site_tf_names(args.sites_path)
        before = len(tf_name_filter)
        tf_name_filter = {
            str(name).upper()
            for name in tf_name_filter
            if str(name).upper() not in excluded_tf_names
        }
        if not tf_name_filter:
            raise ValueError("No TF labels remain after --exclude-tf-names filtering")
        print(
            "Excluded TFs:",
            {
                "requested": sorted(excluded_tf_names),
                "n_training_tfs_before": before,
                "n_training_tfs_after": len(tf_name_filter),
            },
        )

    dataset = BindingBenchWindowDataset(
        sites_path=args.sites_path,
        regions_path=args.regions_path,
        window_size=args.window_size,
        negative_ratio=args.negative_ratio,
        min_sites_per_tf=args.min_sites_per_tf,
        seed=args.seed,
        sequence_orientation=args.sequence_orientation,
        negative_exclusion_bp=args.negative_exclusion_bp,
        max_positive_windows=args.max_positive_windows,
        tf_name_filter=tf_name_filter,
    )
    print("Dataset:", dataset.summary())
    print("Device:", device)
    print("Model:", args.model)

    tf_embeddings = None
    tf_embedding_metadata = None
    if args.model == "transbind_lite":
        if args.tf_embeddings_path is None:
            raise ValueError("--tf-embeddings-path is required for --model transbind_lite")
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
    train_tf_indices = tf_split_indices(tf_split, "train")
    val_tf_indices = tf_split_indices(tf_split, "val")
    test_tf_indices = tf_split_indices(tf_split, "test")
    use_tf_holdout = bool(val_tf_indices or test_tf_indices)
    train_tf_tensor = tensor_indices(train_tf_indices, device) if use_tf_holdout else None
    val_tf_tensor = tensor_indices(val_tf_indices, device) if val_tf_indices else None
    test_tf_tensor = tensor_indices(test_tf_indices, device) if test_tf_indices else None
    print("TF split:", tf_split["counts"])

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    eval_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_model(
        args.model,
        n_tfs=len(dataset.tf_names),
        tf_embeddings=tf_embeddings,
        hidden_channels=args.hidden_channels,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        dilations=dilations,
        num_heads=args.num_heads,
        transbind_dilations=parse_dilations(args.transbind_dilations),
        transbind_use_max_pool=not args.transbind_no_max_pool,
        transbind_tf_bias=args.transbind_tf_bias,
    ).to(device)
    print("Trainable parameters:", count_parameters(model))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = build_lr_scheduler(args, optimizer)
    print("LR scheduler:", args.lr_scheduler)
    pos_weight = (
        None
        if args.no_pos_weight
        else compute_pos_weight(
            dataset,
            device,
            tf_indices=train_tf_indices if use_tf_holdout else None,
        )
    )
    if pos_weight is not None:
        print(
            "Using pos_weight:",
            {
                "min": float(pos_weight.min().detach().cpu()),
                "median": float(pos_weight.median().detach().cpu()),
                "max": float(pos_weight.max().detach().cpu()),
            },
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir / "tf_names.json", dataset.tf_names)
    save_json(args.output_dir / "dataset_summary.json", dataset.summary())
    save_json(args.output_dir / "args.json", serializable_args(args))
    save_json(args.output_dir / "model_config.json", model_config_from_args(args))
    tf_split_out = args.tf_split_out or (args.output_dir / "tf_split.json")
    save_tf_split(tf_split_out, tf_split)
    print(f"Saved TF split: {tf_split_out}")
    if tf_embedding_metadata is not None:
        save_json(args.output_dir / "tf_embedding_metadata.json", tf_embedding_metadata)

    use_val_for_selection = val_tf_tensor is not None and args.eval_every > 0
    best_metric_name = args.selection_metric
    if best_metric_name.startswith("val_") and not use_val_for_selection:
        print(
            f"Selection metric {best_metric_name!r} needs validation TFs; "
            "falling back to 'train_loss'."
        )
        best_metric_name = "train_loss"
    best_metric_value: float | None = None
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        stats = train_one_epoch(
            model,
            loader,
            optimizer,
            device,
            pos_weight,
            train_tf_tensor,
            epoch,
        )
        if use_val_for_selection and (
            epoch % args.eval_every == 0 or epoch == args.epochs
        ):
            stats = attach_val_stats(
                stats,
                evaluate_window(
                    model=model,
                    loader=eval_loader,
                    device=device,
                    pos_weight=None,
                    tf_indices=val_tf_tensor,
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

        stats_dict = asdict(stats)
        history.append(stats_dict)
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
            stats=stats,
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
                stats=stats,
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
                stats=stats,
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
        "tf_split_counts": tf_split["counts"],
    }
    if use_tf_holdout:
        final_jobs = {
            "windows_train_tfs": train_tf_tensor,
            "windows_val_tfs": val_tf_tensor,
            "windows_test_tfs": test_tf_tensor,
        }
    else:
        final_jobs = {"windows_all_tfs": None}
    for split_name, split_tf_tensor in final_jobs.items():
        if use_tf_holdout and split_tf_tensor is None:
            continue
        final_metrics[split_name] = asdict(
            evaluate_window(
                model=model,
                loader=eval_loader,
                device=device,
                pos_weight=None,
                tf_indices=split_tf_tensor,
                tf_names=dataset.tf_names,
            )
        )
    save_json(args.output_dir / "final_metrics.json", final_metrics)
    print("Final metrics:", final_metrics)
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
