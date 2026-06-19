from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from .dataset import BindingBenchWindowDataset, DEFAULT_REGIONS_PATH, DEFAULT_SITES_PATH
    from .model import smallCNN
except ImportError:
    from dataset import BindingBenchWindowDataset, DEFAULT_REGIONS_PATH, DEFAULT_SITES_PATH
    from model import smallCNN


DEFAULT_OUTPUT_DIR = Path(
    "/s/project/ml4rg_students/2026/project15/working/"
    "supervised_baseline/models/small_cnn_overfit"
)


@dataclass
class EpochStats:
    epoch: int
    train_loss: float
    micro_precision: float
    micro_recall: float
    micro_f1: float
    predicted_positive_rate: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a small supervised CNN on Binding Bench windows."
    )
    parser.add_argument("--sites-path", type=Path, default=DEFAULT_SITES_PATH)
    parser.add_argument("--regions-path", type=Path, default=DEFAULT_REGIONS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
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
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
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
    return parser.parse_args()


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


def compute_pos_weight(dataset: BindingBenchWindowDataset, device: torch.device) -> torch.Tensor:
    labels = np.stack([record.labels for record in dataset.records])
    positives = labels.sum(axis=0)
    negatives = labels.shape[0] - positives
    weights = negatives / np.maximum(positives, 1.0)
    weights = np.clip(weights, 1.0, 100.0)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def serializable_args(args: argparse.Namespace) -> dict[str, object]:
    out = {}
    for key, value in vars(args).items():
        out[key] = str(value) if isinstance(value, Path) else value
    return out


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    args: argparse.Namespace,
    dataset: BindingBenchWindowDataset,
    stats: EpochStats,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
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
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pos_weight: torch.Tensor | None,
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
        logits = model(x)
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
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

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
    )
    print("Dataset:", dataset.summary())
    print("Device:", device)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = smallCNN(n_tfs=len(dataset.tf_names)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    pos_weight = None if args.no_pos_weight else compute_pos_weight(dataset, device)
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

    best_loss = float("inf")
    history: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        stats = train_one_epoch(model, loader, optimizer, device, pos_weight, epoch)
        history.append(asdict(stats))
        print(
            f"epoch={stats.epoch:03d} "
            f"loss={stats.train_loss:.5f} "
            f"micro_p={stats.micro_precision:.4f} "
            f"micro_r={stats.micro_recall:.4f} "
            f"micro_f1={stats.micro_f1:.4f} "
            f"pred_pos={stats.predicted_positive_rate:.5f}"
        )

        save_json(args.output_dir / "history.json", history)
        save_checkpoint(
            args.output_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            args=args,
            dataset=dataset,
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
                stats=stats,
            )

    print(f"Saved outputs to: {args.output_dir}")


if __name__ == "__main__":
    main()
