from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

try:
    from .export_promoter_dense_predictions import (
        build_dataset,
        get_device,
        load_checkpoint,
        resolve_export_config,
    )
    from .postprocess_dense_peaks import build_model, export_config_args, split_output
    from .promoter_splits import load_promoter_split, normalize_promoter_split, split_indices
    from .tf_splits import load_tf_split, normalize_tf_split, tf_split_indices
    from .train_promoter_dense import (
        binary_average_precision,
        binary_roc_auc,
        collate_promoter_batch,
        dense_loss,
        forward_dense_model,
        prepare_x,
    )
except ImportError:
    from export_promoter_dense_predictions import (
        build_dataset,
        get_device,
        load_checkpoint,
        resolve_export_config,
    )
    from postprocess_dense_peaks import build_model, export_config_args, split_output
    from promoter_splits import load_promoter_split, normalize_promoter_split, split_indices
    from tf_splits import load_tf_split, normalize_tf_split, tf_split_indices
    from train_promoter_dense import (
        binary_average_precision,
        binary_roc_auc,
        collate_promoter_batch,
        dense_loss,
        forward_dense_model,
        prepare_x,
    )


SPLIT_PAIRS = (
    ("train_promoters_train_tfs", "train", "train"),
    ("val_promoters_val_tfs", "val", "val"),
    ("test_promoters_test_tfs", "test", "test"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a dense protein-conditioned checkpoint on the paired "
            "train/train, val/val, and test/test promoter and TF splits."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sites-path", type=Path)
    parser.add_argument("--regions-path", type=Path)
    parser.add_argument("--input-mode", choices=("raw", "embedding"))
    parser.add_argument("--embeddings-path", type=Path)
    parser.add_argument("--embedding-column")
    parser.add_argument("--embedding-key-column")
    parser.add_argument("--tf-embeddings-path", type=Path)
    parser.add_argument("--tf-embedding-key-column")
    parser.add_argument("--tf-embedding-column")
    parser.add_argument("--tf-name-map", type=Path)
    parser.add_argument("--max-regions", type=int)
    parser.add_argument("--promoter-split-path", type=Path, required=True)
    parser.add_argument("--tf-split-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--histogram-bins",
        type=int,
        default=65_536,
        help="Number of fixed logit histogram bins for streaming basewise AP/AUROC.",
    )
    parser.add_argument("--histogram-logit-min", type=float, default=-20.0)
    parser.add_argument("--histogram-logit-max", type=float, default=20.0)
    return parser.parse_args()


class LogitHistogram:
    """Memory-bounded approximation to ranking metrics over dense score tracks."""

    def __init__(self, bins: int, minimum: float, maximum: float) -> None:
        if bins <= 1:
            raise ValueError("Histogram bins must exceed one")
        if maximum <= minimum:
            raise ValueError("Histogram maximum must exceed minimum")
        self.bins = int(bins)
        self.minimum = float(minimum)
        self.maximum = float(maximum)
        self.counts = np.zeros(self.bins, dtype=np.int64)
        self.positive_counts = np.zeros(self.bins, dtype=np.int64)
        self.clipped_low = 0
        self.clipped_high = 0

    def update(self, scores: np.ndarray, targets: np.ndarray) -> None:
        if scores.shape != targets.shape:
            raise ValueError("Scores and targets must have matching shapes")
        if len(scores) == 0:
            return
        values = scores.astype(np.float64, copy=False)
        self.clipped_low += int(np.count_nonzero(values < self.minimum))
        self.clipped_high += int(np.count_nonzero(values > self.maximum))
        clipped = np.clip(values, self.minimum, self.maximum)
        scaled = (clipped - self.minimum) / (self.maximum - self.minimum)
        indices = np.minimum((scaled * self.bins).astype(np.int64), self.bins - 1)
        self.counts += np.bincount(indices, minlength=self.bins)
        self.positive_counts += np.bincount(
            indices[targets.astype(bool, copy=False)], minlength=self.bins
        )

    def metrics(self) -> dict[str, float | int | None]:
        total = int(self.counts.sum())
        positives = int(self.positive_counts.sum())
        negatives = total - positives
        if positives == 0 or negatives == 0:
            average_precision = None
            roc_auc = None
        else:
            counts_desc = self.counts[::-1]
            positives_desc = self.positive_counts[::-1]
            cumulative_total = np.cumsum(counts_desc)
            cumulative_positive = np.cumsum(positives_desc)
            precision = cumulative_positive / np.maximum(cumulative_total, 1)
            average_precision = float(np.sum(precision * positives_desc) / positives)

            rank_offset = 0.0
            positive_rank_sum = 0.0
            for count, positive_count in zip(self.counts, self.positive_counts):
                if positive_count:
                    average_rank = rank_offset + (count + 1.0) / 2.0
                    positive_rank_sum += positive_count * average_rank
                rank_offset += count
            roc_auc = float(
                (positive_rank_sum - positives * (positives + 1.0) / 2.0)
                / (positives * negatives)
            )
        return {
            "average_precision_histogram": average_precision,
            "roc_auc_histogram": roc_auc,
            "histogram_bins": self.bins,
            "histogram_logit_min": self.minimum,
            "histogram_logit_max": self.maximum,
            "histogram_clipped_low": self.clipped_low,
            "histogram_clipped_high": self.clipped_high,
        }


def metric_counts(prediction: np.ndarray, target: np.ndarray, valid: np.ndarray) -> dict[str, float | int]:
    tp = int(np.count_nonzero(prediction & target & valid))
    fp = int(np.count_nonzero(prediction & ~target & valid))
    fn = int(np.count_nonzero(~prediction & target & valid))
    calls = int(np.count_nonzero(prediction & valid))
    total = int(np.count_nonzero(valid))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "predicted_positive_rate": calls / max(total, 1),
    }


def saved_option(checkpoint: dict[str, object], key: str, default: object) -> object:
    model_config = checkpoint.get("model_config")
    if isinstance(model_config, dict) and key in model_config:
        return model_config[key]
    saved_args = checkpoint.get("args")
    if isinstance(saved_args, dict) and key in saved_args:
        return saved_args[key]
    return default


def evaluate_split(
    *,
    name: str,
    model: torch.nn.Module,
    dataset,
    promoter_indices: list[int],
    tf_indices: list[int],
    input_mode: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    checkpoint: dict[str, object],
    histogram_bins: int,
    histogram_logit_min: float,
    histogram_logit_max: float,
) -> dict[str, object]:
    if not promoter_indices or not tf_indices:
        raise ValueError(f"{name} has no promoters or TFs")
    loader = DataLoader(
        Subset(dataset, promoter_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=partial(collate_promoter_batch, input_mode=input_mode),
    )
    tf_tensor = torch.tensor(tf_indices, dtype=torch.long, device=device)
    n_tfs = len(tf_indices)
    strict_hist = LogitHistogram(histogram_bins, histogram_logit_min, histogram_logit_max)
    dilated_hist = LogitHistogram(histogram_bins, histogram_logit_min, histogram_logit_max)
    strict_counts = {key: np.zeros(n_tfs, dtype=np.int64) for key in ("tp", "fp", "fn", "calls", "total")}
    dilated_counts = {key: np.zeros(n_tfs, dtype=np.int64) for key in ("tp", "fp", "fn", "calls", "total")}
    pair_score_chunks: list[np.ndarray] = []
    pair_target_chunks: list[np.ndarray] = []
    per_tf_pair_scores: list[list[np.ndarray]] = [[] for _ in range(n_tfs)]
    per_tf_pair_targets: list[list[np.ndarray]] = [[] for _ in range(n_tfs)]
    total_loss = 0.0
    total_examples = 0
    loss_name = str(saved_option(checkpoint, "loss", "bce"))
    focal_gamma = float(saved_option(checkpoint, "focal_gamma", 2.0))
    focal_alpha = saved_option(checkpoint, "focal_alpha", None)
    focal_alpha = float(focal_alpha) if focal_alpha is not None else None
    rank_temperature = float(saved_option(checkpoint, "rank_temperature", 1.0))
    rank_negative_weight = float(saved_option(checkpoint, "rank_negative_weight", 1.0))
    rank_negative_top_k = int(saved_option(checkpoint, "rank_negative_top_k", 10))
    event_pool_radius_bp = int(saved_option(checkpoint, "event_pool_radius_bp", 10))
    event_pool_temperature = float(saved_option(checkpoint, "event_pool_temperature", 0.5))
    event_negative_top_k = int(saved_option(checkpoint, "event_negative_top_k", 32))
    event_negative_weight = float(saved_option(checkpoint, "event_negative_weight", 1.0))

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {name}"):
            x = prepare_x(batch["x"].to(device, non_blocking=True), input_mode)
            hard_y = batch["hard_y"].to(device, non_blocking=True).index_select(1, tf_tensor)
            dilated_y = batch["y"].to(device, non_blocking=True).index_select(1, tf_tensor)
            mask = batch["mask"].to(device, non_blocking=True)
            logits = split_output(forward_dense_model(model, x, tf_tensor))
            if logits.shape != hard_y.shape:
                raise ValueError(f"{name} logit/label mismatch: {logits.shape} vs {hard_y.shape}")
            loss_target = hard_y if loss_name in {"rank", "event_mil"} else dilated_y
            loss = dense_loss(
                logits,
                loss_target,
                mask,
                None,
                loss_name=loss_name,
                focal_gamma=focal_gamma,
                focal_alpha=focal_alpha,
                rank_temperature=rank_temperature,
                rank_negative_weight=rank_negative_weight,
                rank_negative_top_k=rank_negative_top_k,
                event_pool_radius_bp=event_pool_radius_bp,
                event_pool_temperature=event_pool_temperature,
                event_negative_top_k=event_negative_top_k,
                event_negative_weight=event_negative_weight,
            )
            total_loss += float(loss.detach().cpu()) * x.shape[0]
            total_examples += x.shape[0]

            logits_cpu = logits.detach().cpu().numpy().astype(np.float32, copy=False)
            hard_cpu = (hard_y.detach().cpu().numpy() >= 0.5)
            dilated_cpu = (dilated_y.detach().cpu().numpy() >= 0.5)
            mask_cpu = mask.detach().cpu().numpy().astype(bool, copy=False)
            valid = np.broadcast_to(mask_cpu[:, None, :], logits_cpu.shape)
            prediction = logits_cpu >= 0.0

            strict_hist.update(logits_cpu[valid], hard_cpu[valid])
            dilated_hist.update(logits_cpu[valid], dilated_cpu[valid])
            for counts, target in ((strict_counts, hard_cpu), (dilated_counts, dilated_cpu)):
                counts["tp"] += np.count_nonzero(prediction & target & valid, axis=(0, 2))
                counts["fp"] += np.count_nonzero(prediction & ~target & valid, axis=(0, 2))
                counts["fn"] += np.count_nonzero(~prediction & target & valid, axis=(0, 2))
                counts["calls"] += np.count_nonzero(prediction & valid, axis=(0, 2))
                counts["total"] += np.count_nonzero(valid, axis=(0, 2))

            promoter_scores = np.where(valid, logits_cpu, -np.inf).max(axis=-1)
            promoter_targets = (hard_cpu & valid).any(axis=-1)
            pair_score_chunks.append(promoter_scores.reshape(-1))
            pair_target_chunks.append(promoter_targets.reshape(-1))
            for tf_idx in range(n_tfs):
                per_tf_pair_scores[tf_idx].append(promoter_scores[:, tf_idx])
                per_tf_pair_targets[tf_idx].append(promoter_targets[:, tf_idx])

    strict_global = metric_counts_from_arrays(strict_counts)
    dilated_global = metric_counts_from_arrays(dilated_counts)
    pair_scores = np.concatenate(pair_score_chunks)
    pair_targets = np.concatenate(pair_target_chunks)
    pair_ap = binary_average_precision(pair_scores, pair_targets)
    pair_auc = binary_roc_auc(pair_scores, pair_targets)
    per_tf_pairs = []
    for local_idx, dataset_idx in enumerate(tf_indices):
        scores = np.concatenate(per_tf_pair_scores[local_idx])
        targets = np.concatenate(per_tf_pair_targets[local_idx])
        per_tf_pairs.append(
            {
                "tf_name": str(dataset.tf_names[dataset_idx]),
                "positive_promoters": int(targets.sum()),
                "promoter_tf_pairs": int(len(targets)),
                "promoter_pair_average_precision": binary_average_precision(scores, targets),
                "promoter_pair_roc_auc": binary_roc_auc(scores, targets),
                "strict": metric_counts_from_arrays_at_index(strict_counts, local_idx),
                "dilated": metric_counts_from_arrays_at_index(dilated_counts, local_idx),
            }
        )

    valid_pair_aps = [
        float(row["promoter_pair_average_precision"])
        for row in per_tf_pairs
        if row["promoter_pair_average_precision"] is not None
    ]
    valid_pair_aucs = [
        float(row["promoter_pair_roc_auc"])
        for row in per_tf_pairs
        if row["promoter_pair_roc_auc"] is not None
    ]
    return {
        "loss": total_loss / max(total_examples, 1),
        "n_examples": total_examples,
        "n_tfs": n_tfs,
        "micro": {
            "strict": {**strict_global, **strict_hist.metrics()},
            "dilated": {**dilated_global, **dilated_hist.metrics()},
        },
        "promoter_pair": {
            "average_precision": pair_ap,
            "roc_auc": pair_auc,
            "macro_average_precision": float(np.mean(valid_pair_aps)) if valid_pair_aps else None,
            "median_average_precision": float(np.median(valid_pair_aps)) if valid_pair_aps else None,
            "macro_roc_auc": float(np.mean(valid_pair_aucs)) if valid_pair_aucs else None,
            "median_roc_auc": float(np.median(valid_pair_aucs)) if valid_pair_aucs else None,
        },
        "per_tf": per_tf_pairs,
    }


def metric_counts_from_arrays(counts: dict[str, np.ndarray]) -> dict[str, float | int]:
    return metric_counts_from_values(
        int(counts["tp"].sum()),
        int(counts["fp"].sum()),
        int(counts["fn"].sum()),
        int(counts["calls"].sum()),
        int(counts["total"].sum()),
    )


def metric_counts_from_arrays_at_index(
    counts: dict[str, np.ndarray], index: int
) -> dict[str, float | int]:
    return metric_counts_from_values(
        int(counts["tp"][index]),
        int(counts["fp"][index]),
        int(counts["fn"][index]),
        int(counts["calls"][index]),
        int(counts["total"][index]),
    )


def metric_counts_from_values(
    tp: int, fp: int, fn: int, calls: int, total: int
) -> dict[str, float | int]:
    positives = tp + fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "positive_positions": positives,
        "valid_positions": total,
        "positive_prevalence": positives / max(total, 1),
        "predicted_positive_rate": calls / max(total, 1),
    }


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    device = get_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    checkpoint_tf_names = checkpoint.get("tf_names")
    if not isinstance(checkpoint_tf_names, list) or not checkpoint_tf_names:
        raise ValueError("Checkpoint has no tf_names list")
    tf_names = [str(name) for name in checkpoint_tf_names]
    config = resolve_export_config(export_config_args(args), checkpoint)
    dataset = build_dataset(
        config,
        args.max_regions,
        tf_name_filter={name.upper() for name in tf_names},
    )
    if dataset.tf_names != tf_names:
        raise ValueError("Dataset TF order does not match checkpoint TF order")
    promoter_split = normalize_promoter_split(
        load_promoter_split(args.promoter_split_path), dataset.records
    )
    tf_split = normalize_tf_split(load_tf_split(args.tf_split_path), dataset.tf_names)
    model = build_model(checkpoint, config, dataset, tf_names, device)

    results: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "metric_protocol": {
            "basewise_ranking": "streaming fixed-logit histogram approximation",
            "threshold": "logit >= 0.0 (probability >= 0.5)",
            "promoter_pair": "exact max-logit-over-promoter ranking",
            "split_pairs": list(SPLIT_PAIRS),
        },
        "promoter_split_counts": promoter_split["counts"],
        "tf_split_counts": tf_split["counts"],
    }
    for name, promoter_name, tf_name in SPLIT_PAIRS:
        results[name] = evaluate_split(
            name=name,
            model=model,
            dataset=dataset,
            promoter_indices=split_indices(promoter_split, promoter_name),
            tf_indices=tf_split_indices(tf_split, tf_name),
            input_mode=str(config["input_mode"]),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            checkpoint=checkpoint,
            histogram_bins=args.histogram_bins,
            histogram_logit_min=args.histogram_logit_min,
            histogram_logit_max=args.histogram_logit_max,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as handle:
        json.dump(results, handle, indent=2)
    print(f"Wrote split metrics: {args.output}")


if __name__ == "__main__":
    main()
