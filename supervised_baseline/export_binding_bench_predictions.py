from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
from tqdm import tqdm

try:
    from .dataset import DEFAULT_REGIONS_PATH
    from .model import smallCNN
except ImportError:
    from dataset import DEFAULT_REGIONS_PATH
    from model import smallCNN


DEFAULT_PROJECT = Path("/s/project/ml4rg_students/2026/project15")
DEFAULT_MODEL_DIR = (
    DEFAULT_PROJECT / "working/supervised_baseline/models/small_cnn_overfit"
)
DEFAULT_INPUT_DIR = DEFAULT_PROJECT / "working/binding_bench_inputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan Binding Bench regions with the supervised CNN and export top "
            "model predictions in Binding Bench discrete format."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_MODEL_DIR / "best.pt")
    parser.add_argument("--regions-path", type=Path, default=DEFAULT_REGIONS_PATH)
    parser.add_argument(
        "--predictions-out",
        type=Path,
        default=DEFAULT_INPUT_DIR / "supervised_small_cnn_predictions.parquet",
    )
    parser.add_argument(
        "--feature-ranks-out",
        type=Path,
        default=DEFAULT_INPUT_DIR / "supervised_small_cnn_feature_ranks.parquet",
    )
    parser.add_argument("--window-size", type=int)
    parser.add_argument(
        "--sequence-orientation",
        choices=("strand-aware", "genomic"),
        help="Defaults to the value saved in the training checkpoint.",
    )
    parser.add_argument("--scan-stride", type=int, default=1)
    parser.add_argument("--top-k-per-tf", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--score-mode", choices=("sigmoid", "logit"), default="sigmoid")
    parser.add_argument(
        "--max-regions",
        type=int,
        help="Debug option: only scan the first N regions.",
    )
    return parser.parse_args()


def load_checkpoint(path: Path, device: torch.device) -> dict[str, object]:
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def get_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def checkpoint_arg(checkpoint: dict[str, object], name: str, default: object) -> object:
    args = checkpoint.get("args", {})
    if isinstance(args, dict):
        return args.get(name, default)
    return default


def one_hot_batch(sequences: list[str], window_size: int) -> torch.Tensor:
    joined = "".join(sequences).encode("ascii")
    bases = np.frombuffer(joined, dtype=np.uint8).reshape(len(sequences), window_size)
    encoded = np.zeros((len(sequences), 4, window_size), dtype=np.float32)
    encoded[:, 0, :] = (bases == ord("A")) | (bases == ord("a"))
    encoded[:, 1, :] = (bases == ord("C")) | (bases == ord("c"))
    encoded[:, 2, :] = (bases == ord("G")) | (bases == ord("g"))
    encoded[:, 3, :] = (bases == ord("T")) | (bases == ord("t"))
    return torch.from_numpy(encoded)


class TopKBuffer:
    def __init__(self, n_features: int, k: int) -> None:
        self.k = k
        self.scores = np.full((k, n_features), -np.inf, dtype=np.float32)
        self.chrom_codes = np.full((k, n_features), -1, dtype=np.int32)
        self.starts = np.zeros((k, n_features), dtype=np.int64)

    def update(
        self,
        batch_scores: np.ndarray,
        batch_chrom_codes: np.ndarray,
        batch_starts: np.ndarray,
    ) -> None:
        if batch_scores.shape[0] == 0:
            return

        local_k = min(self.k, batch_scores.shape[0])
        if batch_scores.shape[0] > local_k:
            local_rows = np.argpartition(batch_scores, -local_k, axis=0)[-local_k:, :]
            local_scores = np.take_along_axis(batch_scores, local_rows, axis=0)
            local_chrom_codes = batch_chrom_codes[local_rows]
            local_starts = batch_starts[local_rows]
        else:
            local_scores = batch_scores
            local_chrom_codes = np.broadcast_to(
                batch_chrom_codes[:, None], local_scores.shape
            )
            local_starts = np.broadcast_to(batch_starts[:, None], local_scores.shape)

        scores = np.concatenate([self.scores, local_scores], axis=0)
        chrom_codes = np.concatenate([self.chrom_codes, local_chrom_codes], axis=0)
        starts = np.concatenate([self.starts, local_starts], axis=0)

        keep_rows = np.argpartition(scores, -self.k, axis=0)[-self.k:, :]
        self.scores = np.take_along_axis(scores, keep_rows, axis=0)
        self.chrom_codes = np.take_along_axis(chrom_codes, keep_rows, axis=0)
        self.starts = np.take_along_axis(starts, keep_rows, axis=0)


def read_regions(path: Path, max_regions: int | None) -> list[dict[str, object]]:
    regions = pl.read_parquet(path).select("chrom", "start", "end", "strand", "seq")
    regions = regions.with_columns(
        pl.col("chrom").cast(pl.Utf8),
        pl.col("start").cast(pl.Int64),
        pl.col("end").cast(pl.Int64),
        pl.col("strand").cast(pl.Utf8),
        pl.col("seq").cast(pl.Utf8),
    )
    if max_regions is not None:
        regions = regions.head(max_regions)
    return list(regions.iter_rows(named=True))


def add_windows_for_region(
    *,
    region: dict[str, object],
    half_window: int,
    scan_stride: int,
    sequence_orientation: str,
    chrom_to_code: dict[str, int],
    chrom_names: list[str],
    windows: list[str],
    chrom_codes: list[int],
    starts: list[int],
) -> None:
    chrom = str(region["chrom"])
    if chrom not in chrom_to_code:
        chrom_to_code[chrom] = len(chrom_names)
        chrom_names.append(chrom)
    chrom_code = chrom_to_code[chrom]

    region_start = int(region["start"])
    region_end = int(region["end"])
    seq = str(region["seq"])
    strand = str(region["strand"])

    lo = region_start + half_window
    hi = region_end - half_window
    if hi <= lo:
        return

    for center in range(lo, hi, scan_stride):
        if sequence_orientation == "strand-aware" and strand == "-":
            offset = region_end - center - 1
        else:
            offset = center - region_start

        left = offset - half_window
        right = offset + half_window + 1
        if left < 0 or right > len(seq):
            continue

        windows.append(seq[left:right])
        chrom_codes.append(chrom_code)
        starts.append(center)


def score_batch(
    *,
    model: torch.nn.Module,
    device: torch.device,
    windows: list[str],
    window_size: int,
    score_mode: str,
) -> np.ndarray:
    x = one_hot_batch(windows, window_size).to(device, non_blocking=True)
    with torch.no_grad():
        logits = model(x)
        if score_mode == "sigmoid":
            scores = logits.sigmoid()
        else:
            scores = logits
    return scores.detach().cpu().numpy().astype(np.float32, copy=False)


def write_outputs(
    *,
    predictions_out: Path,
    feature_ranks_out: Path,
    topk: TopKBuffer,
    tf_names: list[str],
    chrom_names: list[str],
) -> None:
    prediction_frames = []
    feature_scores = []

    for feature_idx, tf_name in enumerate(tf_names):
        scores = topk.scores[:, feature_idx]
        valid = np.isfinite(scores) & (topk.chrom_codes[:, feature_idx] >= 0)
        if not valid.any():
            continue

        order = np.argsort(-scores[valid])
        starts = topk.starts[valid, feature_idx][order]
        chrom_codes = topk.chrom_codes[valid, feature_idx][order]
        sorted_scores = scores[valid][order]
        feature_scores.append((tf_name, float(sorted_scores[0])))

        prediction_frames.append(
            pl.DataFrame(
                {
                    "chrom": [chrom_names[int(code)] for code in chrom_codes],
                    "start": starts,
                    "end": starts + 1,
                    "feature_idx": [tf_name] * len(starts),
                    "score": sorted_scores,
                    "strand": ["."] * len(starts),
                }
            )
        )

    if not prediction_frames:
        raise RuntimeError("No predictions were produced")

    predictions = pl.concat(prediction_frames, how="vertical")
    predictions_out.parent.mkdir(parents=True, exist_ok=True)
    predictions.write_parquet(predictions_out)

    ranks = (
        pl.DataFrame(
            feature_scores,
            schema=["feature_idx", "feature_score"],
            orient="row",
        )
        .sort(["feature_score", "feature_idx"], descending=[True, False])
        .with_row_index("feature_rank", offset=1)
        .select("feature_idx", "feature_rank", "feature_score")
    )
    feature_ranks_out.parent.mkdir(parents=True, exist_ok=True)
    ranks.write_parquet(feature_ranks_out)

    print(f"Wrote predictions: {predictions_out} ({predictions.height:,} rows)")
    print(f"Wrote feature ranks: {feature_ranks_out} ({ranks.height:,} features)")
    print("Preview:")
    print(predictions.head(5))


def main() -> None:
    args = parse_args()
    if args.scan_stride <= 0:
        raise ValueError("--scan-stride must be positive")
    if args.top_k_per_tf <= 0:
        raise ValueError("--top-k-per-tf must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")

    device = get_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    tf_names = checkpoint.get("tf_names")
    if not isinstance(tf_names, list) or not tf_names:
        raise ValueError(f"Checkpoint has no tf_names list: {args.checkpoint}")

    window_size = args.window_size or int(checkpoint_arg(checkpoint, "window_size", 101))
    if window_size <= 0 or window_size % 2 == 0:
        raise ValueError("window size must be a positive odd integer")
    sequence_orientation = args.sequence_orientation or str(
        checkpoint_arg(checkpoint, "sequence_orientation", "strand-aware")
    )
    if sequence_orientation not in {"strand-aware", "genomic"}:
        raise ValueError(f"Unsupported sequence orientation: {sequence_orientation}")

    model = smallCNN(n_tfs=len(tf_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    regions = read_regions(args.regions_path, args.max_regions)
    topk = TopKBuffer(n_features=len(tf_names), k=args.top_k_per_tf)
    chrom_to_code: dict[str, int] = {}
    chrom_names: list[str] = []

    windows: list[str] = []
    chrom_codes: list[int] = []
    starts: list[int] = []
    n_scanned = 0
    half_window = window_size // 2

    print(f"Checkpoint:           {args.checkpoint}")
    print(f"Regions:              {args.regions_path}")
    print(f"Device:               {device}")
    print(f"TF outputs:           {len(tf_names)}")
    print(f"Window size:          {window_size}")
    print(f"Sequence orientation: {sequence_orientation}")
    print(f"Scan stride:          {args.scan_stride}")
    print(f"Top K per TF:         {args.top_k_per_tf}")

    def flush() -> None:
        nonlocal windows, chrom_codes, starts, n_scanned
        if not windows:
            return
        scores = score_batch(
            model=model,
            device=device,
            windows=windows,
            window_size=window_size,
            score_mode=args.score_mode,
        )
        topk.update(
            scores,
            np.asarray(chrom_codes, dtype=np.int32),
            np.asarray(starts, dtype=np.int64),
        )
        n_scanned += len(windows)
        windows = []
        chrom_codes = []
        starts = []

    for region in tqdm(regions, desc="Scanning regions"):
        add_windows_for_region(
            region=region,
            half_window=half_window,
            scan_stride=args.scan_stride,
            sequence_orientation=sequence_orientation,
            chrom_to_code=chrom_to_code,
            chrom_names=chrom_names,
            windows=windows,
            chrom_codes=chrom_codes,
            starts=starts,
        )
        while len(windows) >= args.batch_size:
            batch_windows = windows[: args.batch_size]
            batch_chrom_codes = chrom_codes[: args.batch_size]
            batch_starts = starts[: args.batch_size]
            scores = score_batch(
                model=model,
                device=device,
                windows=batch_windows,
                window_size=window_size,
                score_mode=args.score_mode,
            )
            topk.update(
                scores,
                np.asarray(batch_chrom_codes, dtype=np.int32),
                np.asarray(batch_starts, dtype=np.int64),
            )
            n_scanned += len(batch_windows)
            del windows[: args.batch_size]
            del chrom_codes[: args.batch_size]
            del starts[: args.batch_size]

    flush()
    print(f"Scanned windows:      {n_scanned:,}")

    write_outputs(
        predictions_out=args.predictions_out,
        feature_ranks_out=args.feature_ranks_out,
        topk=topk,
        tf_names=[str(name) for name in tf_names],
        chrom_names=chrom_names,
    )

    metadata = {
        "checkpoint": str(args.checkpoint),
        "regions_path": str(args.regions_path),
        "window_size": window_size,
        "sequence_orientation": sequence_orientation,
        "scan_stride": args.scan_stride,
        "top_k_per_tf": args.top_k_per_tf,
        "score_mode": args.score_mode,
        "n_scanned_windows": n_scanned,
        "n_tfs": len(tf_names),
    }
    metadata_path = args.predictions_out.with_suffix(".metadata.json")
    with metadata_path.open("w") as handle:
        json.dump(metadata, handle, indent=2)
    print(f"Wrote metadata: {metadata_path}")


if __name__ == "__main__":
    main()
