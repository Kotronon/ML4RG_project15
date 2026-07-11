from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

try:
    from .export_promoter_dense_predictions import (
        PROTEIN_DENSE_MODEL_NAMES,
        build_dataset,
        checkpoint_model_config,
        get_device,
        infer_input_channels,
        load_checkpoint,
        resolve_export_config,
    )
    from .model import build_dense_model
    from .promoter_splits import load_promoter_split, normalize_promoter_split, split_indices
    from .tf_embeddings import load_tf_embeddings
    from .tf_splits import load_tf_split, normalize_tf_split, tf_split_indices
    from .train_promoter_dense import (
        collate_promoter_batch,
        forward_dense_model,
        prepare_x,
    )
except ImportError:
    from export_promoter_dense_predictions import (
        PROTEIN_DENSE_MODEL_NAMES,
        build_dataset,
        checkpoint_model_config,
        get_device,
        infer_input_channels,
        load_checkpoint,
        resolve_export_config,
    )
    from model import build_dense_model
    from promoter_splits import load_promoter_split, normalize_promoter_split, split_indices
    from tf_embeddings import load_tf_embeddings
    from tf_splits import load_tf_split, normalize_tf_split, tf_split_indices
    from train_promoter_dense import collate_promoter_batch, forward_dense_model, prepare_x


CALL_SCHEMA = {
    "tf_name": pl.String,
    "gene_id": pl.String,
    "chrom": pl.String,
    "strand": pl.String,
    "promoter_offset": pl.Int64,
    "genomic_position": pl.Int64,
    "start": pl.Int64,
    "end": pl.Int64,
    "logit": pl.Float32,
}


def parse_csv_ints(value: str) -> list[int]:
    values = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one integer")
    if any(item < 0 for item in values):
        raise ValueError("Values must be non-negative")
    return sorted(set(values))


def parse_csv_floats(value: str) -> list[float]:
    values = [float(part.strip()) for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one float")
    if any(not 0.0 < item <= 1.0 for item in values):
        raise ValueError("Peak fractions must be in (0, 1]")
    return sorted(set(values))


def parse_csv_strings(value: str) -> list[str]:
    values = [part.strip() for part in value.split(",") if part.strip()]
    if not values:
        raise ValueError("Expected at least one value")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Tune or apply point-peak post-processing for a dense TFBS model. "
            "Tune on validation only, then apply the locked configuration to test."
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
    parser.add_argument("--promoter-split", choices=("val", "test"), required=True)
    parser.add_argument("--tf-split", choices=("val", "test"), required=True)
    parser.add_argument("--mode", choices=("tune", "apply"), required=True)
    parser.add_argument(
        "--selection-path",
        type=Path,
        help="Validation selected_config.json required for --mode apply.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--nms-radii-bp",
        default="0,5,10,20",
        help="Comma-separated NMS radii tested in --mode tune.",
    )
    parser.add_argument(
        "--peak-fractions",
        default="0.0001,0.0003,0.001,0.003,0.01",
        help=(
            "Fraction of positions retained before NMS. For global thresholds this "
            "sets one validation-derived logit threshold; for per_tf it is an "
            "unlabeled per-TF score quantile."
        ),
    )
    parser.add_argument(
        "--threshold-scopes",
        default="global,per_tf",
        help="Comma-separated threshold scopes to test: global,per_tf.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("dilated_f1", "dilated_precision", "strict_f1"),
        default="dilated_f1",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def export_config_args(args: argparse.Namespace) -> SimpleNamespace:
    """Adapt the focused peak-calling CLI to the shared checkpoint resolver."""
    return SimpleNamespace(
        checkpoint=args.checkpoint,
        sites_path=args.sites_path,
        regions_path=args.regions_path,
        input_mode=args.input_mode,
        embeddings_path=args.embeddings_path,
        embedding_column=args.embedding_column,
        embedding_key_column=args.embedding_key_column,
        tf_embeddings_path=args.tf_embeddings_path,
        tf_embedding_key_column=args.tf_embedding_key_column,
        tf_embedding_column=args.tf_embedding_column,
        tf_name_map=args.tf_name_map,
        model=None,
        hidden_channels=None,
        kernel_size=None,
        dilations=None,
        dropout=None,
        min_sites_per_tf=None,
        sequence_orientation=None,
        include_terminal_atg=False,
    )


def split_output(logits_or_outputs: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(logits_or_outputs, dict):
        if "logits" not in logits_or_outputs:
            raise ValueError("Model output dictionary has no logits")
        return logits_or_outputs["logits"]
    return logits_or_outputs


def build_model(
    checkpoint: dict[str, object],
    config: dict[str, object],
    dataset,
    tf_names: list[str],
    device: torch.device,
) -> torch.nn.Module:
    if str(config["label_mode"]) != "tf":
        raise ValueError("Peak post-processing currently requires a per-TF label-mode checkpoint")

    input_channels = int(
        checkpoint_model_config(checkpoint).get(
            "input_channels",
            infer_input_channels(dataset, str(config["input_mode"])),
        )
    )
    tf_embeddings = None
    if str(config["model_name"]) in PROTEIN_DENSE_MODEL_NAMES:
        tf_embeddings, _ = load_tf_embeddings(
            config["tf_embeddings_path"],
            tf_names,
            key_column=config["tf_embedding_key_column"],
            embedding_column=str(config["tf_embedding_column"]),
            name_mapping_path=config["tf_name_map"],
        )

    model = build_dense_model(
        str(config["model_name"]),
        n_tfs=len(tf_names),
        input_channels=input_channels,
        tf_embeddings=tf_embeddings,
        hidden_channels=int(config["hidden_channels"]),
        kernel_size=int(config["kernel_size"]),
        dropout=float(config["dropout"]),
        dilations=config["dilations"],
        tf_embedding_dropout=float(config["tf_embedding_dropout"]),
        cross_attention_gate_logit_init=float(config["cross_attention_gate_logit_init"]),
        cross_attention_context_pool_sizes=config["cross_attention_context_pool_sizes"],
        dna_attention_window_bp=int(config["dna_attention_window_bp"]),
        dna_attention_layers=int(config["dna_attention_layers"]),
        dna_attention_heads=int(config["dna_attention_heads"]),
        dna_attention_ffn_multiplier=float(config["dna_attention_ffn_multiplier"]),
        motif_kernel_sizes=config["motif_kernel_sizes"],
        protein_noise_std=float(config["protein_noise_std"]),
        protein_l2_normalize=bool(config["protein_l2_normalize"]),
        protein_delta_gate_logit_init=float(config["protein_delta_gate_logit_init"]),
        scorer=str(config["scorer"]),
        scorer_heads=int(config["scorer_heads"]),
        scorer_pair_dim=int(config["scorer_pair_dim"]),
        scorer_hidden_dim=int(config["scorer_hidden_dim"]),
        scorer_bias_mode=str(config["scorer_bias_mode"]),
        film_eval_tf_chunk_size=int(config["film_eval_tf_chunk_size"]),
        window_pooling=str(config["window_pooling"]),
        window_pooling_top_k=int(config["window_pooling_top_k"]),
    ).to(device)
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError("Checkpoint has no model_state_dict")
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def collect_tracks(
    *,
    model: torch.nn.Module,
    dataset,
    promoter_indices: list[int],
    tf_indices: list[int],
    input_mode: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[object]]:
    if not promoter_indices or not tf_indices:
        raise ValueError("Selected promoter and TF splits must both be non-empty")
    records = [dataset.records[index] for index in promoter_indices]
    loader = DataLoader(
        Subset(dataset, promoter_indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=partial(collate_promoter_batch, input_mode=input_mode),
    )
    tf_tensor = torch.tensor(tf_indices, dtype=torch.long, device=device)
    score_chunks: list[np.ndarray] = []
    strict_chunks: list[np.ndarray] = []
    dilated_chunks: list[np.ndarray] = []
    mask_chunks: list[np.ndarray] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Scoring dense peak tracks"):
            x = prepare_x(batch["x"].to(device, non_blocking=True), input_mode)
            outputs = forward_dense_model(model, x, tf_tensor)
            logits = split_output(outputs)
            if logits.shape[1] != len(tf_indices):
                raise ValueError(
                    f"Expected {len(tf_indices)} selected TF logits, got {logits.shape[1]}"
                )
            strict = batch["hard_y"].index_select(1, tf_tensor.cpu())
            dilated = batch["y"].index_select(1, tf_tensor.cpu())
            score_chunks.append(logits.detach().cpu().numpy().astype(np.float32, copy=False))
            strict_chunks.append((strict.numpy() >= 0.5).astype(np.bool_, copy=False))
            dilated_chunks.append((dilated.numpy() >= 0.5).astype(np.bool_, copy=False))
            mask_chunks.append(batch["mask"].numpy().astype(np.bool_, copy=False))

    return (
        np.concatenate(score_chunks, axis=0),
        np.concatenate(strict_chunks, axis=0),
        np.concatenate(dilated_chunks, axis=0),
        np.concatenate(mask_chunks, axis=0),
        records,
    )


def quantile(values: np.ndarray, q: float) -> float:
    try:
        return float(np.quantile(values, q, method="higher"))
    except TypeError:
        return float(np.quantile(values, q, interpolation="higher"))


def threshold_spec(
    scores: np.ndarray,
    mask: np.ndarray,
    *,
    scope: str,
    peak_fraction: float,
    fixed_global_threshold: float | None = None,
) -> tuple[np.ndarray, dict[str, float | None]]:
    valid = np.broadcast_to(mask[:, None, :], scores.shape)
    if scope == "global":
        threshold = (
            float(fixed_global_threshold)
            if fixed_global_threshold is not None
            else quantile(scores[valid], 1.0 - peak_fraction)
        )
        return np.full(scores.shape[1], threshold, dtype=np.float32), {
            "threshold_global": threshold,
            "threshold_min": threshold,
            "threshold_median": threshold,
            "threshold_max": threshold,
        }
    if scope == "per_tf":
        thresholds = np.asarray(
            [
                quantile(scores[:, tf_idx, :][mask], 1.0 - peak_fraction)
                for tf_idx in range(scores.shape[1])
            ],
            dtype=np.float32,
        )
        return thresholds, {
            "threshold_global": None,
            "threshold_min": float(thresholds.min()),
            "threshold_median": float(np.median(thresholds)),
            "threshold_max": float(thresholds.max()),
        }
    raise ValueError(f"Unknown threshold scope: {scope!r}")


def nms_mask_1d(scores: np.ndarray, valid: np.ndarray, radius_bp: int) -> np.ndarray:
    selected = np.zeros(scores.shape, dtype=bool)
    positions = np.flatnonzero(valid)
    if radius_bp == 0:
        selected[positions] = True
        return selected

    order = positions[np.argsort(-scores[positions], kind="stable")]
    suppressed = np.zeros(scores.shape, dtype=bool)
    for position in order:
        if suppressed[position]:
            continue
        selected[position] = True
        lo = max(0, int(position) - radius_bp)
        hi = min(len(scores), int(position) + radius_bp + 1)
        suppressed[lo:hi] = True
    return selected


def nms_mask(scores: np.ndarray, mask: np.ndarray, radius_bp: int) -> np.ndarray:
    peaks = np.zeros(scores.shape, dtype=bool)
    for promoter_idx in tqdm(range(scores.shape[0]), desc=f"NMS radius {radius_bp}", leave=False):
        for tf_idx in range(scores.shape[1]):
            peaks[promoter_idx, tf_idx] = nms_mask_1d(
                scores[promoter_idx, tf_idx], mask[promoter_idx], radius_bp
            )
    return peaks


def binary_metrics(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    valid = np.broadcast_to(mask[:, None, :], prediction.shape)
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
        "calls": calls,
        "calls_per_kb_per_tf": calls / max(total, 1) * 1000.0,
    }


def metric_row(
    *,
    prediction: np.ndarray,
    hard_target: np.ndarray,
    dilated_target: np.ndarray,
    mask: np.ndarray,
    radius_bp: int,
    scope: str,
    peak_fraction: float,
    threshold_summary: dict[str, float | None],
) -> dict[str, object]:
    strict = binary_metrics(prediction, hard_target, mask)
    dilated = binary_metrics(prediction, dilated_target, mask)
    return {
        "nms_radius_bp": radius_bp,
        "threshold_scope": scope,
        "peak_fraction": peak_fraction,
        **threshold_summary,
        **{f"strict_{name}": value for name, value in strict.items()},
        **{f"dilated_{name}": value for name, value in dilated.items()},
    }


def genomic_positions(record: object, offsets: np.ndarray, sequence_orientation: str) -> np.ndarray:
    if sequence_orientation == "strand-aware" and str(getattr(record, "strand")) == "-":
        return int(getattr(record, "model_end")) - 1 - offsets
    return int(getattr(record, "model_start")) + offsets


def calls_frame(
    *,
    prediction: np.ndarray,
    scores: np.ndarray,
    mask: np.ndarray,
    records: list[object],
    tf_names: list[str],
    sequence_orientation: str,
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for promoter_idx, record in enumerate(records):
        for tf_idx, tf_name in enumerate(tf_names):
            offsets = np.flatnonzero(prediction[promoter_idx, tf_idx] & mask[promoter_idx])
            if len(offsets) == 0:
                continue
            genomic = genomic_positions(record, offsets, sequence_orientation)
            for offset, position in zip(offsets.tolist(), genomic.tolist()):
                rows.append(
                    {
                        "tf_name": tf_name,
                        "gene_id": str(getattr(record, "gene_id")),
                        "chrom": str(getattr(record, "chrom")),
                        "strand": str(getattr(record, "strand")),
                        "promoter_offset": int(offset),
                        "genomic_position": int(position),
                        "start": int(position),
                        "end": int(position) + 1,
                        "logit": float(scores[promoter_idx, tf_idx, offset]),
                    }
                )
    if not rows:
        return pl.DataFrame(schema=CALL_SCHEMA)
    return pl.DataFrame(rows, schema=CALL_SCHEMA)


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def choose_best(rows: list[dict[str, object]], metric: str) -> dict[str, object]:
    candidates = [row for row in rows if isinstance(row.get(metric), (float, int))]
    if not candidates:
        raise ValueError(f"No finite rows available for selection metric {metric!r}")
    return max(
        candidates,
        key=lambda row: (
            float(row[metric]),
            float(row["dilated_precision"]),
            -float(row["dilated_calls_per_kb_per_tf"]),
        ),
    )


def selected_prediction(
    *,
    scores: np.ndarray,
    mask: np.ndarray,
    radius_bp: int,
    scope: str,
    peak_fraction: float,
    fixed_global_threshold: float | None = None,
) -> tuple[np.ndarray, dict[str, float | None]]:
    thresholds, summary = threshold_spec(
        scores,
        mask,
        scope=scope,
        peak_fraction=peak_fraction,
        fixed_global_threshold=fixed_global_threshold,
    )
    peaks = nms_mask(scores, mask, radius_bp)
    prediction = peaks & (scores >= thresholds[None, :, None])
    return prediction, summary


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.mode == "apply" and args.selection_path is None:
        raise ValueError("--selection-path is required for --mode apply")
    if args.mode == "tune" and args.selection_path is not None:
        raise ValueError("--selection-path is only used by --mode apply")

    device = get_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    checkpoint_tf_names = checkpoint.get("tf_names")
    if not isinstance(checkpoint_tf_names, list) or not checkpoint_tf_names:
        raise ValueError("Checkpoint has no tf_names list")
    tf_names = [str(name) for name in checkpoint_tf_names]
    config = resolve_export_config(export_config_args(args), checkpoint)
    tf_filter = {name.upper() for name in tf_names}
    dataset = build_dataset(config, args.max_regions, tf_name_filter=tf_filter)
    if dataset.tf_names != tf_names:
        raise ValueError("Dataset TF order does not match the checkpoint TF order")

    promoter_split = normalize_promoter_split(
        load_promoter_split(args.promoter_split_path), dataset.records
    )
    tf_split = normalize_tf_split(load_tf_split(args.tf_split_path), dataset.tf_names)
    promoter_indices = split_indices(promoter_split, args.promoter_split)
    selected_tf_indices = tf_split_indices(tf_split, args.tf_split)
    selected_tf_names = [dataset.tf_names[index] for index in selected_tf_indices]
    model = build_model(checkpoint, config, dataset, tf_names, device)

    print("Checkpoint:", args.checkpoint)
    print("Mode:", args.mode)
    print("Promoter split:", args.promoter_split, len(promoter_indices))
    print("TF split:", args.tf_split, len(selected_tf_indices))
    print("Device:", device)
    scores, hard_target, dilated_target, mask, records = collect_tracks(
        model=model,
        dataset=dataset,
        promoter_indices=promoter_indices,
        tf_indices=selected_tf_indices,
        input_mode=str(config["input_mode"]),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_metadata = {
        "checkpoint": str(args.checkpoint),
        "promoter_split": args.promoter_split,
        "tf_split": args.tf_split,
        "n_promoters": len(records),
        "n_tfs": len(selected_tf_names),
        "n_positions_per_track": int(scores.shape[-1]),
        "sequence_orientation": str(config["sequence_orientation"]),
    }

    if args.mode == "tune":
        radii = parse_csv_ints(args.nms_radii_bp)
        peak_fractions = parse_csv_floats(args.peak_fractions)
        scopes = parse_csv_strings(args.threshold_scopes)
        unsupported_scopes = sorted(set(scopes) - {"global", "per_tf"})
        if unsupported_scopes:
            raise ValueError(f"Unknown threshold scopes: {unsupported_scopes}")

        rows: list[dict[str, object]] = []
        masks_by_radius = {radius: nms_mask(scores, mask, radius) for radius in radii}
        for radius in radii:
            peak_mask = masks_by_radius[radius]
            for scope in scopes:
                for peak_fraction in peak_fractions:
                    thresholds, threshold_summary = threshold_spec(
                        scores,
                        mask,
                        scope=scope,
                        peak_fraction=peak_fraction,
                    )
                    prediction = peak_mask & (scores >= thresholds[None, :, None])
                    rows.append(
                        metric_row(
                            prediction=prediction,
                            hard_target=hard_target,
                            dilated_target=dilated_target,
                            mask=mask,
                            radius_bp=radius,
                            scope=scope,
                            peak_fraction=peak_fraction,
                            threshold_summary=threshold_summary,
                        )
                    )

        sweep_path = args.output_dir / "validation_peak_sweep.tsv"
        pl.DataFrame(rows).write_csv(sweep_path, separator="\t")
        best = choose_best(rows, args.selection_metric)
        selection = {
            **base_metadata,
            "selection_metric": args.selection_metric,
            "selected": best,
        }
        selection_path = args.output_dir / "selected_config.json"
        save_json(selection_path, selection)
        prediction, _ = selected_prediction(
            scores=scores,
            mask=mask,
            radius_bp=int(best["nms_radius_bp"]),
            scope=str(best["threshold_scope"]),
            peak_fraction=float(best["peak_fraction"]),
            fixed_global_threshold=(
                float(best["threshold_global"])
                if best["threshold_scope"] == "global"
                else None
            ),
        )
        calls = calls_frame(
            prediction=prediction,
            scores=scores,
            mask=mask,
            records=records,
            tf_names=selected_tf_names,
            sequence_orientation=str(config["sequence_orientation"]),
        )
        calls_path = args.output_dir / "validation_peak_calls.parquet"
        calls.write_parquet(calls_path)
        print("Wrote sweep:", sweep_path)
        print("Selected configuration:", selection_path)
        print("Validation calls:", calls_path, calls.height)
        print("Selected metrics:", best)
        return

    with args.selection_path.open() as handle:
        selection = json.load(handle)
    selected = selection.get("selected")
    if not isinstance(selected, dict):
        raise ValueError(f"Invalid selection file: {args.selection_path}")
    scope = str(selected["threshold_scope"])
    peak_fraction = float(selected["peak_fraction"])
    radius = int(selected["nms_radius_bp"])
    fixed_global_threshold = (
        float(selected["threshold_global"]) if scope == "global" else None
    )
    prediction, threshold_summary = selected_prediction(
        scores=scores,
        mask=mask,
        radius_bp=radius,
        scope=scope,
        peak_fraction=peak_fraction,
        fixed_global_threshold=fixed_global_threshold,
    )
    metrics = metric_row(
        prediction=prediction,
        hard_target=hard_target,
        dilated_target=dilated_target,
        mask=mask,
        radius_bp=radius,
        scope=scope,
        peak_fraction=peak_fraction,
        threshold_summary=threshold_summary,
    )
    result = {
        **base_metadata,
        "selection_path": str(args.selection_path),
        "selected_validation_config": selected,
        "applied_metrics": metrics,
    }
    metrics_path = args.output_dir / "applied_peak_metrics.json"
    save_json(metrics_path, result)
    calls = calls_frame(
        prediction=prediction,
        scores=scores,
        mask=mask,
        records=records,
        tf_names=selected_tf_names,
        sequence_orientation=str(config["sequence_orientation"]),
    )
    calls_path = args.output_dir / "applied_peak_calls.parquet"
    calls.write_parquet(calls_path)
    print("Applied metrics:", metrics_path)
    print("Applied calls:", calls_path, calls.height)
    print("Metrics:", metrics)


if __name__ == "__main__":
    main()
