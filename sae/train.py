"""Train a Sparse Autoencoder on DNA LM embeddings.

Example:
    python -m sae.train \
        --embedding-path /path/to/_saccharomyces_cerevisiae_evo2_7b.parquet \
        --variant topk \
        --k 64 \
        --expansion-factor 8 \
        --epochs 50 \
        --output-dir /path/to/sae/models/evo2_7b_topk_k64
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

try:
    from .dataset import DEFAULT_NPY_PATH, EmbeddingDataset, NpyPositionDataset
    from .model import SAE_VARIANTS, SAEConfig, SparseAutoencoder, build_sae
except ImportError:
    from dataset import DEFAULT_NPY_PATH, EmbeddingDataset, NpyPositionDataset
    from model import SAE_VARIANTS, SAEConfig, SparseAutoencoder, build_sae

DEFAULT_OUTPUT_ROOT = Path(
    "/s/project/ml4rg_students/2026/project15/working/sae/models"
)


@dataclass
class EpochStats:
    epoch: int
    loss: float
    recon_loss: float
    sparsity_loss: float
    l0: float               # mean active features per sample
    dead_fraction: float    # fraction of features never active in this epoch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embedding-path", type=Path, default=DEFAULT_NPY_PATH,
                   help="Path to embeddings: .npy file [N,L,d] (SpeciesLM) or .parquet (Evo2).")
    p.add_argument("--seq-offset", type=int, default=1,
                   help="Index of first real-sequence token in L dim (SpeciesLM: 1 for CLS skip).")
    p.add_argument("--seq-len", type=int, default=1000,
                   help="Number of sequence positions to use per gene.")
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    # Model
    p.add_argument("--variant", choices=SAE_VARIANTS, default="topk")
    p.add_argument("--expansion-factor", type=int, default=8)
    p.add_argument("--k", type=int, default=64, help="Active features per sample (TopK/BatchTopK).")
    p.add_argument("--k-fraction", type=float, help="k = round(k_fraction * hidden_dim). Overrides --k.")
    p.add_argument("--l1-coeff", type=float, default=1e-3, help="L1 penalty (VanillaSAE only).")
    p.add_argument("--no-normalize-decoder", dest="normalize_decoder", action="store_false", default=True)
    # Training
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--val-fraction", type=float, default=0.05)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--no-normalize-embeddings", dest="normalize_embeddings", action="store_false", default=True)
    return p.parse_args()


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


def save_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def save_checkpoint(
    path: Path,
    model: SparseAutoencoder,
    optimizer: optim.Optimizer,
    epoch: int,
    cfg: SAEConfig,
    stats: EpochStats,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "sae_config": asdict(cfg),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "stats": asdict(stats),
        },
        path,
    )


def train_one_epoch(
    model: SparseAutoencoder,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> EpochStats:
    model.train()
    totals: dict[str, float] = {
        "loss": 0, "recon_loss": 0, "sparsity_loss": 0, "l0": 0
    }
    n = 0
    ever_active = torch.zeros(model.cfg.hidden_dim, device=device)

    for x in loader:
        x = x.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        model.clamp_decoder_norm()

        bs = x.shape[0]
        n += bs
        totals["loss"] += float(out["loss"].detach()) * bs
        totals["recon_loss"] += float(out.get("recon_loss", out["loss"]).detach()) * bs
        totals["sparsity_loss"] += float(out.get("sparsity_loss", torch.zeros(1)).detach()) * bs
        totals["l0"] += float(out["l0"].detach()) * bs
        ever_active += (out["activations"].detach() > 0).float().sum(dim=0)

    dead_fraction = float((ever_active == 0).float().mean())
    return EpochStats(
        epoch=epoch,
        loss=totals["loss"] / n,
        recon_loss=totals["recon_loss"] / n,
        sparsity_loss=totals["sparsity_loss"] / n,
        l0=totals["l0"] / n,
        dead_fraction=dead_fraction,
    )


@torch.no_grad()
def eval_epoch(
    model: SparseAutoencoder,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_recon = 0.0
    total_l0 = 0.0
    n = 0
    for x in loader:
        x = x.to(device, non_blocking=True)
        out = model(x)
        bs = x.shape[0]
        total_recon += float(out.get("recon_loss", out["loss"]).detach()) * bs
        total_l0 += float(out["l0"].detach()) * bs
        n += bs
    return {"val_recon_loss": total_recon / n, "val_l0": total_l0 / n}


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device(args.device)

    suffix = args.embedding_path.suffix.lower()
    if suffix == ".npy":
        dataset = NpyPositionDataset(
            args.embedding_path,
            normalize=args.normalize_embeddings,
            seq_offset=args.seq_offset,
            seq_len=args.seq_len,
        )
    else:
        dataset = EmbeddingDataset(args.embedding_path, normalize=args.normalize_embeddings)
    print("Dataset:", dataset.summary())

    n_val = max(1, int(len(dataset) * args.val_fraction))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda"
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda"
    )

    cfg = SAEConfig(
        input_dim=dataset.input_dim,
        expansion_factor=args.expansion_factor,
        variant=args.variant,
        k=args.k,
        k_fraction=args.k_fraction,
        l1_coeff=args.l1_coeff,
        normalize_decoder=args.normalize_decoder,
    )
    print(f"SAE config: {cfg}")

    if args.output_dir is None:
        tag = f"{args.variant}_ef{args.expansion_factor}_k{cfg.k}"
        args.output_dir = args.output_root / tag
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = build_sae(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"SAE parameters: {n_params:,}  (input={cfg.input_dim}, hidden={cfg.hidden_dim})")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    save_json(args.output_dir / "sae_config.json", asdict(cfg))
    save_json(
        args.output_dir / "args.json",
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    )

    history: list[dict] = []
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        stats = train_one_epoch(model, train_loader, optimizer, device, epoch)
        val_metrics = eval_epoch(model, val_loader, device)
        scheduler.step()

        row = {**asdict(stats), **val_metrics}
        history.append(row)

        print(
            f"epoch={epoch:03d}  "
            f"loss={stats.loss:.5f}  "
            f"recon={stats.recon_loss:.5f}  "
            f"val_recon={val_metrics['val_recon_loss']:.5f}  "
            f"l0={stats.l0:.1f}/{cfg.hidden_dim}  "
            f"dead={stats.dead_fraction:.3f}"
        )
        save_json(args.output_dir / "history.json", history)

        save_checkpoint(args.output_dir / "last.pt", model, optimizer, epoch, cfg, stats)
        if val_metrics["val_recon_loss"] < best_val:
            best_val = val_metrics["val_recon_loss"]
            save_checkpoint(args.output_dir / "best.pt", model, optimizer, epoch, cfg, stats)
        if args.save_every and epoch % args.save_every == 0:
            save_checkpoint(
                args.output_dir / f"epoch_{epoch:03d}.pt", model, optimizer, epoch, cfg, stats
            )

    print(f"Training done. Best val recon: {best_val:.5f}")
    print(f"Outputs → {args.output_dir}")


if __name__ == "__main__":
    main()
