from __future__ import annotations

import argparse
import os
from pathlib import Path

import polars as pl


DEFAULT_PROJECT = Path(os.environ.get("ML4RG_PROJECT", "/s/project/ml4rg_students/2026/project15"))
DEFAULT_DATASET = "DNA_rossi_chipexo"
DEFAULT_RUN_DIR = (
    DEFAULT_PROJECT
    / "working/binding_bench_runs/supervised_res_dilated_cnn_logit_nms50"
)
DEFAULT_SUMMARY = Path("baseline_evaluations/streme_fimo_n150_filtered_baseline.tsv")
SUMMARY_COLUMNS = [
    "method",
    "metric",
    "assigned",
    "positive",
    "meaningful_ge_0.01",
    "score_sum",
    "mean_positive",
    "best_score",
    "best_tf",
    "best_feature",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Append Binding Bench best-assignment summary rows to the shared "
            "baseline evaluation TSV."
        )
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help=(
            "Binding Bench run directory. Accepts either the run root or the "
            "binding_bench/discrete/<dataset> directory."
        ),
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument(
        "--summary-tsv",
        type=Path,
        default=DEFAULT_SUMMARY,
        help="TSV file to append to.",
    )
    parser.add_argument(
        "--method",
        default="Supervised ResDilatedCNN logit NMS50",
        help="Method label to write into the summary table.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["jaccard", "precision_lb", "recall_lb"],
    )
    parser.add_argument("--meaningful-threshold", type=float, default=0.01)
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="Remove existing rows with the same method before appending.",
    )
    return parser.parse_args()


def resolve_discrete_dir(run_dir: Path, dataset: str) -> Path:
    candidates = [
        run_dir,
        run_dir / "binding_bench" / "discrete" / dataset,
    ]
    for candidate in candidates:
        if (candidate / "full_best_assnt").is_dir():
            return candidate
    checked = "\n".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        "Could not find Binding Bench full_best_assnt directory. Checked:\n"
        f"{checked}"
    )


def best_assignment_path(discrete_dir: Path, metric: str, dataset: str) -> Path:
    return (
        discrete_dir
        / "full_best_assnt"
        / f"best_assnt_metrics_{metric}_ds_{dataset}.parquet"
    )


def summarize_metric(
    *,
    discrete_dir: Path,
    dataset: str,
    method: str,
    metric: str,
    meaningful_threshold: float,
) -> dict[str, object]:
    path = best_assignment_path(discrete_dir, metric, dataset)
    if not path.is_file():
        raise FileNotFoundError(f"Missing best-assignment table: {path}")

    df = pl.read_parquet(path).with_columns(
        pl.col(metric).cast(pl.Float64).fill_null(0.0)
    )
    positive = df.filter(pl.col(metric) > 0)
    meaningful = df.filter(pl.col(metric) >= meaningful_threshold)
    best = df.sort(metric, descending=True).row(0, named=True)

    return {
        "method": method,
        "metric": metric,
        "assigned": df.select(pl.col("feature_idx").is_not_null().sum()).item(),
        "positive": positive.height,
        "meaningful_ge_0.01": meaningful.height,
        "score_sum": df.select(pl.col(metric).sum()).item(),
        "mean_positive": positive.select(pl.col(metric).mean()).item()
        if positive.height
        else None,
        "best_score": float(best[metric]),
        "best_tf": "" if best["name"] is None else str(best["name"]),
        "best_feature": ""
        if best["feature_idx"] is None
        else str(best["feature_idx"]),
    }


def read_existing_summary(path: Path) -> pl.DataFrame | None:
    if not path.is_file():
        return None
    return pl.read_csv(path, separator="\t", infer_schema_length=0)


def main() -> None:
    args = parse_args()
    discrete_dir = resolve_discrete_dir(args.run_dir, args.dataset)

    rows = [
        summarize_metric(
            discrete_dir=discrete_dir,
            dataset=args.dataset,
            method=args.method,
            metric=metric,
            meaningful_threshold=args.meaningful_threshold,
        )
        for metric in args.metrics
    ]
    new_summary = pl.DataFrame(rows).select(SUMMARY_COLUMNS)

    existing = read_existing_summary(args.summary_tsv)
    if existing is not None:
        existing = existing.select(SUMMARY_COLUMNS)
        if args.replace_existing:
            existing = existing.filter(pl.col("method") != args.method)
        summary = pl.concat([existing, new_summary], how="vertical_relaxed")
    else:
        summary = new_summary

    args.summary_tsv.parent.mkdir(parents=True, exist_ok=True)
    summary.write_csv(args.summary_tsv, separator="\t")

    print(f"Read Binding Bench results from: {discrete_dir}")
    print(f"Wrote summary: {args.summary_tsv}")
    print(new_summary)


if __name__ == "__main__":
    main()
