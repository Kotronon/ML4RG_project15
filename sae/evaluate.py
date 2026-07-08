"""Evaluate SAE features against TF binding site labels at position resolution.

Memory-efficient design:
- Activations collected as float16 → ~80 GB for full dataset, so we subsample
  (--max-eval-positions, default 500k) keeping ALL positive positions + random negatives.
- Dataset stores only metadata + labels per record; embeddings loaded on demand.

By default, scores every SAE feature against every TF. When
``--best-assignment-path`` is provided (from a prior Binding Bench run),
only the assigned TF→feature pairs are scored — the recommended workflow
after exporting SAE peaks with ``sae.export_binding_bench_predictions``.

Example:
    python -m sae.evaluate \
        --checkpoint /path/to/sae/models/topk_ef8_k32/best.pt \
        --npy-path /path/to/embs_ds_fungi_upstream_ATG_1000_l10.npy \
        --sites-path /path/to/DNA_rossi_chipexo_sites.parquet \
        --regions-path /path/to/_saccharomyces_cerevisiae_sequence_mapper.parquet

    python -m sae.evaluate \
        --checkpoint /path/to/sae/models/topk_ef8_k32/best.pt \
        --best-assignment-path working/binding_bench_runs/sae_topk_nms50/binding_bench/discrete/DNA_rossi_chipexo
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
    from .binding_bench_utils import load_tf_feature_assignments, resolve_best_assignment_path
    from .dataset import DEFAULT_NPY_PATH, DEFAULT_REGIONS_PATH, DEFAULT_SITES_PATH, NpyPositionWithLabelsDataset
    from .model import SAEConfig, SparseAutoencoder, build_sae
except ImportError:
    from binding_bench_utils import load_tf_feature_assignments, resolve_best_assignment_path
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
    p.add_argument(
        "--best-assignment-path",
        type=Path,
        help=(
            "Binding Bench discrete run dir or best_assnt_metrics parquet. "
            "When set, AUROC is computed only for the assigned TF→feature pairs."
        ),
    )
    p.add_argument(
        "--assignment-metric",
        default="jaccard",
        help="Metric used to resolve best-assignment parquet from a run directory.",
    )
    p.add_argument(
        "--assignment-dataset",
        default="DNA_rossi_chipexo",
        help="Binding Bench dataset name used to resolve best-assignment parquet.",
    )
    p.add_argument(
        "--all-features",
        action="store_true",
        help="Also compute exhaustive per-feature AUROC when --best-assignment-path is set.",
    )
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


def _auroc_all_features(
    acts: np.ndarray,
    y: np.ndarray,
    max_per_class: int = 5000,
    seed: int = 0,
) -> np.ndarray:
    """Vectorised rank-based AUROC for all features simultaneously.

    Subsamples to max_per_class positives and negatives before ranking so the
    argsort is on a [2*max_per_class, h] matrix — fast and memory-bounded.
    AUROC on 5k+5k is statistically equivalent to using all samples (SE ~0.0002).
    """
    h = acts.shape[1]
    pos_idx = np.where(y)[0]
    neg_idx = np.where(~y)[0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return np.full(h, 0.5)

    rng = np.random.default_rng(seed)
    n_take = min(max_per_class, len(pos_idx), len(neg_idx))
    p_idx = rng.choice(pos_idx, n_take, replace=False)
    n_idx = rng.choice(neg_idx, n_take, replace=False)
    idx = np.concatenate([p_idx, n_idx])
    sub = acts[idx]                                       # [2*n_take, h]
    sub_y = np.concatenate([np.ones(n_take, bool), np.zeros(n_take, bool)])

    n = len(idx)
    order = np.argsort(sub, axis=0)                       # [n, h]
    inv = np.empty_like(order)
    inv[order, np.arange(h)[None, :]] = np.arange(n, dtype=order.dtype)[:, None]
    rank_sum = (inv[sub_y] + 1).sum(axis=0).astype(np.float64)
    denom = float(n_take) * float(n_take)
    return (rank_sum - n_take * (n_take + 1) / 2) / denom


def _ap_all_features(acts: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Average precision per feature (sklearn, only for active features)."""
    n_pos = int(y.sum())
    baseline = float(n_pos) / len(y)
    aps = np.full(acts.shape[1], baseline)
    for j in np.where(acts.max(axis=0) > 0)[0]:
        try:
            aps[j] = average_precision_score(y, acts[:, j])
        except Exception:
            pass
    return aps


def per_feature_scores(
    activations: np.ndarray,   # [N, h] float16
    labels: np.ndarray,        # [N, n_tfs]
    tf_names: list[str],
    top_n: int,
) -> dict[str, object]:
    acts_f32 = activations.astype(np.float32)
    results: dict[str, object] = {}

    for tf_idx, tf_name in enumerate(tf_names):
        y = labels[:, tf_idx].astype(bool)
        n_pos = int(y.sum())
        if n_pos < 5 or (len(y) - n_pos) < 5:
            continue

        aurocs = _auroc_all_features(acts_f32, y)

        top_auroc = np.argsort(aurocs)[::-1][:top_n]
        results[tf_name] = {
            "best_features_auroc": [(int(i), float(aurocs[i])) for i in top_auroc],
            "max_auroc": float(aurocs.max()),
            "mean_auroc": float(aurocs.mean()),
            "n_positive_positions": n_pos,
            "prevalence": float(n_pos) / len(y),
        }
        if (tf_idx + 1) % 10 == 0:
            print(f"  [{tf_idx+1}/{len(tf_names)}] {tf_name}  best_auroc={aurocs.max():.4f}")
    return results


def matched_feature_scores(
    activations: np.ndarray,
    labels: np.ndarray,
    tf_names: list[str],
    assignments: dict[str, int],
) -> dict[str, object]:
    """Score only the Binding Bench-assigned SAE feature for each TF."""
    acts_f32 = activations.astype(np.float32)
    tf_to_idx = {name: idx for idx, name in enumerate(tf_names)}
    results: dict[str, object] = {}

    for tf_name, feat_idx in sorted(assignments.items()):
        if tf_name not in tf_to_idx:
            continue
        if not (0 <= feat_idx < acts_f32.shape[1]):
            continue

        tf_idx = tf_to_idx[tf_name]
        y = labels[:, tf_idx]
        n_pos = int(y.sum())
        if n_pos < 5 or (len(y) - n_pos) < 5:
            continue

        scores = acts_f32[:, feat_idx]
        auroc = 0.5
        ap = float(n_pos) / len(y)
        if scores.max() > 0:
            try:
                auroc = float(roc_auc_score(y, scores))
                ap = float(average_precision_score(y, scores))
            except Exception:
                pass

        results[tf_name] = {
            "assigned_feature": int(feat_idx),
            "auroc": auroc,
            "ap": ap,
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

    assignments: dict[str, int] | None = None
    if args.best_assignment_path is not None:
        assignment_path = resolve_best_assignment_path(
            args.best_assignment_path,
            metric=args.assignment_metric,
            dataset=args.assignment_dataset,
        )
        assignments = load_tf_feature_assignments(assignment_path)
        print(f"Loaded {len(assignments)} TF→feature assignments from {assignment_path}")

    matched_results: dict[str, object] | None = None
    if assignments is not None:
        print(f"Computing matched-feature AUROC/AP for assigned pairs...")
        matched_results = matched_feature_scores(
            activations, labels, dataset.tf_names, assignments
        )
        print(f"\n{'TF':<30}  {'feature':>9}  {'AUROC':>6}  {'AP':>6}  {'n_pos':>6}")
        print("-" * 65)
        for tf_name, info in sorted(
            matched_results.items(), key=lambda kv: kv[1]["auroc"], reverse=True
        )[:25]:
            print(
                f"{tf_name:<30}  {info['assigned_feature']:>9d}  "
                f"{info['auroc']:>6.4f}  {info['ap']:>6.4f}  "
                f"{info['n_positive_positions']:>6d}"
            )

    tf_results: dict[str, object] | None = None
    if assignments is None or args.all_features:
        print(f"Computing per-feature AUROC/AP for {dataset.n_tfs} TFs...")
        tf_results = per_feature_scores(
            activations, labels, dataset.tf_names, args.top_n_features
        )
        print(f"\n{'TF':<30}  {'best_feat':>9}  {'AUROC':>6}  {'n_pos':>6}")
        print("-" * 55)
        for tf_name, info in sorted(
            tf_results.items(), key=lambda kv: kv[1]["max_auroc"], reverse=True
        )[:25]:
            feat, auc = info["best_features_auroc"][0]
            print(f"{tf_name:<30}  {feat:>9d}  {auc:>6.4f}  {info['n_positive_positions']:>6d}")

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
        "best_assignment_path": str(args.best_assignment_path) if args.best_assignment_path else None,
        "matched_features_only": assignments is not None and not args.all_features,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    if matched_results is not None:
        (args.output_dir / "matched_feature_scores.json").write_text(
            json.dumps(matched_results, indent=2)
        )
    if tf_results is not None:
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
