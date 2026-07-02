"""Evaluate SAE features against TF binding site labels at position resolution.

Memory-efficient design:
- Activations collected as float16 → ~80 GB for full dataset, so we subsample
  (--max-eval-positions, default 500k) keeping ALL positive positions + random negatives.
- Dataset stores only metadata + labels per record; embeddings loaded on demand.

Example:
    python -m sae.evaluate \
        --checkpoint /path/to/sae/models/topk_ef8_k32/best.pt \
        --npy-path /path/to/embs_ds_fungi_upstream_ATG_1000_l10.npy \
        --sites-path /path/to/DNA_rossi_chipexo_sites.parquet \
        --regions-path /path/to/_saccharomyces_cerevisiae_sequence_mapper.parquet
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except ModuleNotFoundError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "scikit-learn"])
    from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Subset

try:
    from .dataset import DEFAULT_NPY_PATH, DEFAULT_REGIONS_PATH, DEFAULT_SITES_PATH, NpyPositionWithLabelsDataset
    from .model import SAEConfig, SparseAutoencoder, build_sae
except ImportError:
    from dataset import DEFAULT_NPY_PATH, DEFAULT_REGIONS_PATH, DEFAULT_SITES_PATH, NpyPositionWithLabelsDataset
    from model import SAEConfig, SparseAutoencoder, build_sae

DEFAULT_OUTPUT_ROOT = Path("/s/project/ml4rg_students/2026/project15/working/sae/evals")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--npy-path", type=Path, default=DEFAULT_NPY_PATH)
    p.add_argument("--sites-path", type=Path, default=DEFAULT_SITES_PATH)
    p.add_argument("--regions-path", type=Path, default=DEFAULT_REGIONS_PATH)
    p.add_argument("--output-dir", type=Path)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--half-window", type=int, default=50)
    p.add_argument("--min-sites-per-tf", type=int, default=15)
    p.add_argument("--seq-offset", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--top-n-features", type=int, default=10)
    p.add_argument("--max-eval-positions", type=int, default=500_000,
                   help="Subsample this many positions for AUROC (keep all positives + random negatives).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def get_device(r: str) -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu") if r == "auto" else torch.device(r)


def load_model(path: Path, device: torch.device) -> SparseAutoencoder:
    ckpt = torch.load(path, map_location=device)
    model = build_sae(SAEConfig(**ckpt["sae_config"])).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def subsample_indices(
    dataset: NpyPositionWithLabelsDataset,
    max_positions: int,
    seed: int,
) -> np.ndarray:
    """Keep all positive positions (any TF) + random negatives up to max_positions."""
    labels = dataset.get_labels_array()   # [N, n_tfs] — compact numpy array
    is_positive = labels.any(axis=1)
    pos_idx = np.where(is_positive)[0]
    neg_idx = np.where(~is_positive)[0]

    n_neg = max(0, max_positions - len(pos_idx))
    rng = np.random.default_rng(seed)
    if len(neg_idx) > n_neg:
        neg_idx = rng.choice(neg_idx, n_neg, replace=False)

    idx = np.concatenate([pos_idx, neg_idx])
    rng.shuffle(idx)
    print(f"Subsampled {len(idx):,} positions ({len(pos_idx):,} positive + {len(neg_idx):,} negative)")
    return idx


@torch.no_grad()
def collect_activations(
    model: SparseAutoencoder,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns activations [N, h] as float16 and labels [N, n_tfs] as float32."""
    all_acts: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    for batch in loader:
        emb = batch["embedding"].to(device, non_blocking=True)
        out = model(emb)
        all_acts.append(out["activations"].half().cpu().numpy())   # float16 → ~2x less RAM
        all_labels.append(batch["labels"].numpy())
    return np.concatenate(all_acts, axis=0), np.concatenate(all_labels, axis=0)


def per_feature_scores(
    activations: np.ndarray,   # [N, h] float16
    labels: np.ndarray,        # [N, n_tfs]
    tf_names: list[str],
    top_n: int,
) -> dict[str, object]:
    acts_f32 = activations.astype(np.float32)  # cast once for sklearn
    n_features = acts_f32.shape[1]
    results: dict[str, object] = {}

    for tf_idx, tf_name in enumerate(tf_names):
        y = labels[:, tf_idx]
        n_pos = int(y.sum())
        if n_pos < 5 or (len(y) - n_pos) < 5:
            continue

        aurocs = np.full(n_features, 0.5)
        aps    = np.full(n_features, float(n_pos) / len(y))

        for feat_idx in range(n_features):
            scores = acts_f32[:, feat_idx]
            if scores.max() == 0:
                continue
            try:
                aurocs[feat_idx] = roc_auc_score(y, scores)
                aps[feat_idx]    = average_precision_score(y, scores)
            except Exception:
                pass

        top_auroc = np.argsort(aurocs)[::-1][:top_n]
        top_ap    = np.argsort(aps)[::-1][:top_n]
        results[tf_name] = {
            "best_features_auroc": [(int(i), float(aurocs[i])) for i in top_auroc],
            "best_features_ap":    [(int(i), float(aps[i]))    for i in top_ap],
            "max_auroc": float(aurocs.max()),
            "max_ap":    float(aps.max()),
            "mean_auroc": float(aurocs.mean()),
            "n_positive_positions": n_pos,
            "prevalence": float(n_pos) / len(y),
        }
    return results


def main() -> None:
    args = parse_args()
    device = get_device(args.device)

    model = load_model(args.checkpoint, device)
    print(f"SAE: variant={model.cfg.variant}  input={model.cfg.input_dim}  hidden={model.cfg.hidden_dim}  k={model.cfg.k}")

    print("Building position-level dataset...")
    dataset = NpyPositionWithLabelsDataset(
        npy_path=args.npy_path,
        sites_path=args.sites_path,
        regions_path=args.regions_path,
        half_window=args.half_window,
        min_sites_per_tf=args.min_sites_per_tf,
        seq_offset=args.seq_offset,
        seq_len=args.seq_len,
    )
    print("Full dataset:", dataset.summary())

    # Subsample to keep memory feasible
    sub_idx = subsample_indices(dataset, args.max_eval_positions, args.seed)
    subset  = Subset(dataset, sub_idx.tolist())

    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    print("Collecting SAE activations (float16)...")
    activations, labels = collect_activations(model, loader, device)
    print(f"Activations: {activations.shape} {activations.dtype}  ~{activations.nbytes/1e9:.1f} GB")

    mean_l0  = float((activations > 0).sum(axis=1).mean())
    dead_frac = float((activations.max(axis=0) == 0).mean())
    print(f"Mean L0: {mean_l0:.1f}/{model.cfg.hidden_dim}  Dead features: {dead_frac*100:.1f}%")

    print(f"Computing per-feature AUROC/AP for {dataset.n_tfs} TFs...")
    tf_results = per_feature_scores(activations, labels, dataset.tf_names, args.top_n_features)

    print(f"\n{'TF':<30}  {'best_feat':>9}  {'AUROC':>6}  {'AP':>6}  {'n_pos':>6}")
    print("-" * 65)
    for tf_name, info in sorted(tf_results.items(), key=lambda kv: kv[1]["max_auroc"], reverse=True)[:25]:
        feat, auc = info["best_features_auroc"][0]
        _,    ap  = info["best_features_ap"][0]
        print(f"{tf_name:<30}  {feat:>9d}  {auc:>6.4f}  {ap:>6.4f}  {info['n_positive_positions']:>6d}")

    if args.output_dir is None:
        args.output_dir = args.output_root / args.checkpoint.parent.name
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "checkpoint": str(args.checkpoint),
        "sae_config": asdict(model.cfg),
        "n_positions_evaluated": len(sub_idx),
        "n_tfs": dataset.n_tfs,
        "half_window": args.half_window,
        "mean_l0": mean_l0,
        "dead_fraction": dead_frac,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.output_dir / "tf_feature_scores.json").write_text(json.dumps(tf_results, indent=2))

    np.savez_compressed(
        args.output_dir / "activations.npz",
        activations=activations,
        labels=labels,
        tf_names=np.array(dataset.tf_names),
        flat_indices=sub_idx,
        genomic_pos=dataset._genomic_pos[sub_idx],
        chroms=np.array(dataset._chroms)[sub_idx],
        gene_ids=np.array(dataset._gene_ids)[sub_idx],
    )
    print(f"\nResults saved → {args.output_dir}")


if __name__ == "__main__":
    main()
