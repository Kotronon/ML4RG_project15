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
from torch.utils.data import DataLoader

try:
    from .dataset import (
        BindingBenchPromoterEmbeddingDataset,
        BindingBenchPromoterSequenceDataset,
        DEFAULT_REGIONS_PATH,
        DEFAULT_SITES_PATH,
    )
    from .model import DENSE_MODEL_NAMES, build_dense_model, parse_dilations
except ImportError:
    from dataset import (
        BindingBenchPromoterEmbeddingDataset,
        BindingBenchPromoterSequenceDataset,
        DEFAULT_REGIONS_PATH,
        DEFAULT_SITES_PATH,
    )
    from model import DENSE_MODEL_NAMES, build_dense_model, parse_dilations


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
        "--no-pos-weight",
        action="store_true",
        help="Disable positive-class weighting in dense BCE.",
    )
    parser.add_argument("--save-every", type=int, default=10)
    args = parser.parse_args()
    if args.input_mode == "embedding" and args.embeddings_path is None:
        raise ValueError("--embeddings-path is required for --input-mode embedding")
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


def compute_dense_pos_weight(dataset, device: torch.device) -> torch.Tensor:
    positives = np.zeros(len(dataset.tf_names), dtype=np.float64)
    valid_positions = 0.0
    for record in dataset.records:
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


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)
    dilations = parse_dilations(args.dilations)

    dataset = build_dataset(args)
    input_channels = infer_input_channels(dataset, args.input_mode)
    print("Dataset:", dataset.summary())
    print("Input channels:", input_channels)
    print("Device:", device)
    print("Model:", args.model)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=partial(collate_promoter_batch, input_mode=args.input_mode),
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
    pos_weight = None if args.no_pos_weight else compute_dense_pos_weight(dataset, device)
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

    best_loss = float("inf")
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        stats = train_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            device=device,
            input_mode=args.input_mode,
            pos_weight=pos_weight,
            epoch=epoch,
        )
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(stats.train_loss)
        elif scheduler is not None:
            scheduler.step()
            stats.lr = current_lr(optimizer)

        history.append(asdict(stats))
        print(
            f"epoch={stats.epoch:03d} "
            f"loss={stats.train_loss:.5f} "
            f"micro_p={stats.micro_precision:.4f} "
            f"micro_r={stats.micro_recall:.4f} "
            f"micro_f1={stats.micro_f1:.4f} "
            f"pred_pos={stats.predicted_positive_rate:.5f} "
            f"lr={stats.lr:.6g}"
        )

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
        )
        if stats.train_loss < best_loss:
            best_loss = stats.train_loss
            save_checkpoint(
                args.output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                args=args,
                dataset=dataset,
                input_channels=input_channels,
                stats=stats,
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
            )

    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
