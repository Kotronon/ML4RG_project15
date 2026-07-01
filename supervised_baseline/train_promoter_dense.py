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
    from .model import DENSE_MODEL_NAMES, build_dense_model, parse_dilations
    from .promoter_splits import (
        load_promoter_split,
        make_all_train_split,
        make_chromosome_promoter_split,
        make_random_promoter_split,
        normalize_promoter_split,
        save_promoter_split,
        split_indices,
    )
except ImportError:
    from dataset import (
        BindingBenchPromoterEmbeddingDataset,
        BindingBenchPromoterSequenceDataset,
        DEFAULT_REGIONS_PATH,
        DEFAULT_SITES_PATH,
    )
    from model import DENSE_MODEL_NAMES, build_dense_model, parse_dilations
    from promoter_splits import (
        load_promoter_split,
        make_all_train_split,
        make_chromosome_promoter_split,
        make_random_promoter_split,
        normalize_promoter_split,
        save_promoter_split,
        split_indices,
    )


DEFAULT_MODEL_ROOT = Path(
    "/s/project/ml4rg_students/2026/project15/working/"
    "supervised_baseline/models"
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


@dataclass
class DenseSplitStats:
    loss: float
    micro_precision: float
    micro_recall: float
    micro_f1: float
    predicted_positive_rate: float
    n_examples: int


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
        "--model",
        choices=DENSE_MODEL_NAMES,
        default="dense_small_cnn",
    )
    parser.add_argument("--hidden-channels", type=int, default=128)
    parser.add_argument("--kernel-size", type=int, default=7)
    parser.add_argument("--dilations", default="1,2,4,8,16")
    parser.add_argument("--dropout", type=float, default=0.1)
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
        "--no-pos-weight",
        action="store_true",
        help="Disable positive-class weighting in dense BCE.",
    )
    parser.add_argument("--save-every", type=int, default=10)
    args = parser.parse_args()
    if args.input_mode == "embedding" and args.embeddings_path is None:
        raise ValueError("--embeddings-path is required for --input-mode embedding")
    if args.eval_every < 0:
        raise ValueError("--eval-every must be non-negative")
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


def build_dataset(args: argparse.Namespace):
    common = {
        "sites_path": args.sites_path,
        "regions_path": args.regions_path,
        "min_sites_per_tf": args.min_sites_per_tf,
        "sequence_orientation": args.sequence_orientation,
        "max_regions": args.max_regions,
        "trim_terminal_atg": not args.include_terminal_atg,
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


def collate_promoter_batch(
    items: list[dict[str, object]],
    input_mode: str,
) -> dict[str, object]:
    if not items:
        raise ValueError("Cannot collate an empty batch")

    y0 = items[0]["y"]
    if not isinstance(y0, torch.Tensor):
        raise TypeError("Dataset y must be a torch.Tensor")
    max_len = max(int(item["y"].shape[-1]) for item in items)
    n_tfs = int(y0.shape[0])

    y_batch = torch.zeros((len(items), n_tfs, max_len), dtype=torch.float32)
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
            mask = item["mask"]
            length = int(y.shape[-1])
            x_batch[idx, :, :length] = x
            y_batch[idx, :, :length] = y
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
            mask = item["mask"]
            length = int(y.shape[-1])
            x_batch[idx, :length, :] = x
            y_batch[idx, :, :length] = y
            mask_batch[idx, :length] = mask
    else:
        raise ValueError(f"Unsupported input mode: {input_mode}")

    return {
        "x": x_batch,
        "y": y_batch,
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
        labels = np.zeros((len(dataset.tf_names), len(record.sequence)), dtype=bool)
        for tf_idx, lo, hi in record.label_intervals:
            labels[tf_idx, lo:hi] = True
        labels[:, ~mask] = False
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
        "embeddings_path": str(args.embeddings_path) if args.embeddings_path else None,
        "embedding_column": args.embedding_column,
        "embedding_key_column": args.embedding_key_column,
        "trim_terminal_atg": not args.include_terminal_atg,
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
        mask = batch["mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        if logits.shape != y.shape:
            raise ValueError(f"Logit/label shape mismatch: {logits.shape} vs {y.shape}")
        loss = dense_bce_loss(logits, y, mask, pos_weight)
        loss.backward()
        optimizer.step()

        batch_size = x.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_examples += batch_size

        with torch.no_grad():
            valid = mask.unsqueeze(1).expand_as(logits)
            pred = logits.sigmoid() >= 0.5
            target = y >= 0.5
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
) -> DenseSplitStats:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    tp = fp = fn = 0.0
    predicted_positive = 0.0
    total_labels = 0.0

    with torch.no_grad():
        for batch in loader:
            x = prepare_x(batch["x"].to(device, non_blocking=True), input_mode)
            y = batch["y"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)

            logits = model(x)
            if logits.shape != y.shape:
                raise ValueError(f"Logit/label shape mismatch: {logits.shape} vs {y.shape}")
            loss = dense_bce_loss(logits, y, mask, pos_weight)

            batch_size = x.shape[0]
            total_loss += float(loss.detach().cpu()) * batch_size
            total_examples += batch_size

            valid = mask.unsqueeze(1).expand_as(logits)
            pred = logits.sigmoid() >= 0.5
            target = y >= 0.5
            tp += (pred & target & valid).sum().item()
            fp += (pred & ~target & valid).sum().item()
            fn += (~pred & target & valid).sum().item()
            predicted_positive += (pred & valid).sum().item()
            total_labels += valid.sum().item()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return DenseSplitStats(
        loss=total_loss / max(total_examples, 1),
        micro_precision=precision,
        micro_recall=recall,
        micro_f1=f1,
        predicted_positive_rate=predicted_positive / max(total_labels, 1),
        n_examples=total_examples,
    )


def attach_val_stats(stats: DenseEpochStats, val_stats: DenseSplitStats) -> DenseEpochStats:
    stats.val_loss = val_stats.loss
    stats.val_micro_precision = val_stats.micro_precision
    stats.val_micro_recall = val_stats.micro_recall
    stats.val_micro_f1 = val_stats.micro_f1
    stats.val_predicted_positive_rate = val_stats.predicted_positive_rate
    return stats


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    dilations = parse_dilations(args.dilations)

    dataset = build_dataset(args)
    promoter_split = build_promoter_split(args, dataset)
    train_indices = split_indices(promoter_split, "train")
    val_indices = split_indices(promoter_split, "val")
    test_indices = split_indices(promoter_split, "test")
    input_channels = infer_input_channels(dataset, args.input_mode)
    print("Dataset:", dataset.summary())
    print("Promoter split:", promoter_split["counts"])
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
        hidden_channels=args.hidden_channels,
        kernel_size=args.kernel_size,
        dropout=args.dropout,
        dilations=dilations,
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

    best_loss = float("inf")
    use_val_for_selection = val_loader is not None and args.eval_every > 0
    best_metric_name = "val_loss" if use_val_for_selection else "train_loss"
    history: list[dict[str, object]] = []
    for epoch in range(1, args.epochs + 1):
        stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            input_mode=args.input_mode,
            pos_weight=pos_weight,
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
                    pos_weight=pos_weight,
                ),
            )

        selection_loss = (
            stats.val_loss
            if stats.val_loss is not None
            else stats.train_loss
            if not use_val_for_selection
            else None
        )
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(selection_loss if selection_loss is not None else stats.train_loss)
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
        )
        if selection_loss is not None and selection_loss < best_loss:
            best_loss = selection_loss
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
        "best_loss": best_loss,
        "promoter_split_counts": promoter_split["counts"],
    }
    final_loaders = {
        "train": train_eval_loader,
        "val": val_loader,
        "test": test_loader,
    }
    for split_name, split_loader in final_loaders.items():
        if split_loader is None:
            continue
        split_stats = evaluate_dense(
            model=model,
            loader=split_loader,
            device=device,
            input_mode=args.input_mode,
            pos_weight=pos_weight,
        )
        final_metrics[split_name] = asdict(split_stats)
    save_json(args.output_dir / "final_metrics.json", final_metrics)
    print("Final metrics:", final_metrics)
    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
