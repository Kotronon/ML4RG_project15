"""Evaluate SAE features against TF binding site labels at position resolution.

For each SAE feature, computes per-TF AUROC using position-level activation
scores vs. position-level binding labels (a position is positive if a known
TF binding site centre falls within ±half_window bases).

This is directly comparable to FIMO / STREME baselines, which also produce
per-position scores.

Example:
    python -m sae.evaluate \
        --checkpoint /path/to/sae/models/topk_ef8_k32/best.pt \
        --npy-path /path/to/specieslm_v1/_saccharomyces_cerevisiae/embs_ds_fungi_upstream_ATG_1000_l10.npy \
        --sites-path /path/to/DNA_rossi_chipexo_sites.parquet \
        --regions-path /path/to/_saccharomyces_cerevisiae_sequence_mapper.parquet \
        --output-dir /path/to/sae/evals/topk_ef8_k32
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score  # type: ignore[import]
from torch.utils.data import DataLoader

try:
    from .dataset import (
        DEFAULT_NPY_PATH,
        DEFAULT_REGIONS_PATH,
        DEFAULT_SITES_PATH,
        NpyPositionWithLabelsDataset,
    )
    from .model import SAEConfig, SparseAutoencoder, build_sae
except ImportError:
    from dataset import (
        DEFAULT_NPY_PATH,
        DEFAULT_REGIONS_PATH,
        DEFAULT_SITES_PATH,
        NpyPositionWithLabelsDataset,
    )
    from model import SAEConfig, SparseAutoencoder, build_sae

DEFAULT_OUTPUT_ROOT = Path(
    "/s/project/ml4rg_students/2026/project15/working/sae/evals"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--npy-path", type=Path, default=DEFAULT_NPY_PATH)
    p.add_argument("--sites-path", type=Path, default=DEFAULT_SITES_PATH)
    p.add_argument("--regions-path", type=Path, default=DEFAULT_REGIONS_PATH)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--half-window", type=int, default=50,
                   help="A position is a positive example if a binding site centre is within this many bp.")
    p.add_argument("--min-sites-per-tf", type=int, default=15)
    p.add_argument("--seq-offset", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--top-n-features", type=int, default=10,
                   help="Report the N features with highest AUROC per TF.")
    return p.parse_args()


def get_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def load_model(checkpoint_path: Path, device: torch.device) -> SparseAutoencoder:
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = SAEConfig(**ckpt["sae_config"])
    model = build_sae(cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


@torch.no_grad()
def collect_activations(
    model: SparseAutoencoder,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (activations [N_pos, h], labels [N_pos, n_tfs])."""
    all_acts: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    for batch in loader:
        emb = batch["embedding"].to(device, non_blocking=True)
        out = model(emb)
        all_acts.append(out["activations"].float().cpu().numpy())
        all_labels.append(batch["labels"].numpy())

    return np.concatenate(all_acts, axis=0), np.concatenate(all_labels, axis=0)


def per_feature_scores(
    activations: np.ndarray,  # [N_pos, h]
    labels: np.ndarray,       # [N_pos, n_tfs]
    tf_names: list[str],
    top_n: int,
) -> dict[str, object]:
    """Per-TF: find which SAE features best separate binding vs. non-binding positions."""
    n_features = activations.shape[1]
    results: dict[str, object] = {}

    for tf_idx, tf_name in enumerate(tf_names):
        y = labels[:, tf_idx]
        n_pos = int(y.sum())
        if n_pos < 5 or (len(y) - n_pos) < 5:
            continue

        aurocs = np.full(n_features, 0.5)
        aps = np.full(n_features, float(n_pos) / len(y))  # baseline = prevalence

        for feat_idx in range(n_features):
            scores = activations[:, feat_idx]
            if scores.max() == 0:
                continue
            try:
                aurocs[feat_idx] = roc_auc_score(y, scores)
                aps[feat_idx] = average_precision_score(y, scores)
            except Exception:
                pass

        top_by_auroc = np.argsort(aurocs)[::-1][:top_n]
        results[tf_name] = {
            "best_features_auroc": [(int(i), float(aurocs[i])) for i in top_by_auroc],
            "best_features_ap":    [(int(i), float(aps[i])) for i in np.argsort(aps)[::-1][:top_n]],
            "max_auroc": float(aurocs.max()),
            "max_ap": float(aps.max()),
            "mean_auroc": float(aurocs.mean()),
            "n_positive_positions": n_pos,
            "prevalence": float(n_pos) / len(y),
        }

    return results


def reconstruction_variance_explained(
    model: SparseAutoencoder,
    activations: np.ndarray,
    embeddings: np.ndarray,
    device: torch.device,
    chunk: int = 8192,
) -> float:
    ss_res = 0.0
    with torch.no_grad():
        for i in range(0, len(activations), chunk):
            z = torch.tensor(activations[i:i+chunk], dtype=torch.float32, device=device)
            x_hat = model.decode(z).cpu().numpy()
            x = embeddings[i:i+chunk]
            ss_res += float(((x - x_hat) ** 2).sum())
    ss_tot = float(((embeddings - embeddings.mean(axis=0)) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def main() -> None:
    args = parse_args()
    device = get_device(args.device)

    model = load_model(args.checkpoint, device)
    print(f"SAE: variant={model.cfg.variant}  input={model.cfg.input_dim}  hidden={model.cfg.hidden_dim}  k={model.cfg.k}")

    print("Building position-level dataset (this resolves genomic coordinates — may take a minute)...")
    dataset = NpyPositionWithLabelsDataset(
        npy_path=args.npy_path,
        sites_path=args.sites_path,
        regions_path=args.regions_path,
        half_window=args.half_window,
        min_sites_per_tf=args.min_sites_per_tf,
        seq_offset=args.seq_offset,
        seq_len=args.seq_len,
    )
    print("Eval dataset:", dataset.summary())

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    print("Collecting SAE activations...")
    activations, labels = collect_activations(model, loader, device)
    print(f"Activations: {activations.shape}  Labels: {labels.shape}")

    mean_l0 = float((activations > 0).sum(axis=1).mean())
    dead_frac = float((activations.max(axis=0) == 0).mean())
    print(f"Mean L0: {mean_l0:.1f}/{model.cfg.hidden_dim}  Dead features: {dead_frac*100:.1f}%")

    # Variance explained — sample 50k positions for speed
    sample_idx = np.random.choice(len(activations), min(50_000, len(activations)), replace=False)
    emb_sample = np.stack([dataset.records[i].embedding for i in sample_idx])
    var_exp = reconstruction_variance_explained(model, activations[sample_idx], emb_sample, device)
    print(f"Variance explained (sample): {var_exp:.4f}")

    print(f"Computing per-feature AUROC/AP for {dataset.n_tfs} TFs across {len(activations):,} positions...")
    tf_results = per_feature_scores(activations, labels, dataset.tf_names, args.top_n_features)

    # Print top results
    print("\nTop TF–feature matches:")
    print(f"{'TF':<30}  {'best_feat':>9}  {'AUROC':>6}  {'AP':>6}  {'n_pos':>6}")
    print("-" * 65)
    sorted_tfs = sorted(tf_results.items(), key=lambda kv: kv[1]["max_auroc"], reverse=True)
    for tf_name, info in sorted_tfs[:25]:
        feat, auc = info["best_features_auroc"][0]
        _, ap = info["best_features_ap"][0]
        print(f"{tf_name:<30}  {feat:>9d}  {auc:>6.4f}  {ap:>6.4f}  {info['n_positive_positions']:>6d}")

    # Save outputs
    if args.output_dir is None:
        ckpt_tag = args.checkpoint.parent.name
        args.output_dir = args.output_root / ckpt_tag
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "checkpoint": str(args.checkpoint),
        "sae_config": asdict(model.cfg),
        "n_positions": len(dataset),
        "n_tfs": dataset.n_tfs,
        "half_window": args.half_window,
        "mean_l0": mean_l0,
        "dead_fraction": dead_frac,
        "variance_explained": var_exp,
    }
    with (args.output_dir / "summary.json").open("w") as f:
        json.dump(summary, f, indent=2)
    with (args.output_dir / "tf_feature_scores.json").open("w") as f:
        json.dump(tf_results, f, indent=2)

    np.savez_compressed(
        args.output_dir / "activations.npz",
        activations=activations,
        labels=labels,
        tf_names=np.array(dataset.tf_names),
        gene_ids=np.array([r.gene_id for r in dataset.records]),
        genomic_pos=np.array([r.genomic_pos for r in dataset.records]),
        chrom=np.array([r.chrom for r in dataset.records]),
    )
    print(f"\nResults saved → {args.output_dir}")


if __name__ == "__main__":
    main()
