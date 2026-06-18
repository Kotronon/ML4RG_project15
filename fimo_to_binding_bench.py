from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path

import pandas as pd


DEFAULT_PROJECT = "/s/project/ml4rg_students/2026/project15"
DEFAULT_DATASET = "_saccharomyces_cerevisiae_sequence_mapper"
FIMO_SOURCES = ("fimo_streme", "fimo_jaspar")


def parse_fasta_header(sequence_name: str) -> dict[str, object]:
    """Parse headers written by Convert_parquet_to_fasta.ipynb."""
    parts = str(sequence_name).split("|")
    metadata: dict[str, object] = {"gene_id": parts[0], "sequence_name": sequence_name}

    for part in parts[1:]:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        metadata[key] = value

    if "chr" not in metadata and "chrom" in metadata:
        metadata["chr"] = metadata["chrom"]

    required = ("chr", "start", "end", "strand")
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError(
            f"FIMO sequence_name is missing {missing}: {sequence_name!r}. "
            "Expected headers like gene|chr=...|start=...|end=...|strand=..."
        )

    metadata["start"] = int(metadata["start"])
    metadata["end"] = int(metadata["end"])
    metadata["strand"] = str(metadata["strand"])
    return metadata


def read_fimo(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", comment="#")
    if df.empty:
        raise ValueError(f"FIMO table is empty: {path}")

    required = {"motif_id", "sequence_name", "start", "stop", "strand"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"FIMO table {path} is missing columns: {sorted(missing)}")

    df["start"] = pd.to_numeric(df["start"], errors="raise").astype(int)
    df["stop"] = pd.to_numeric(df["stop"], errors="raise").astype(int)
    return df


def _flip_strand(strand: str) -> str:
    if strand == "+":
        return "-"
    if strand == "-":
        return "+"
    return "."


def genomic_interval(
    region_start: int,
    region_end: int,
    region_strand: str,
    fimo_start: int,
    fimo_stop: int,
    sequence_orientation: str,
) -> tuple[int, int]:
    """Convert FIMO 1-based inclusive sequence coordinates to 0-based genomic BED."""
    seq_start0 = min(fimo_start, fimo_stop) - 1
    seq_end0 = max(fimo_start, fimo_stop)

    if sequence_orientation == "strand-aware" and region_strand == "-":
        return region_end - seq_end0, region_end - seq_start0

    return region_start + seq_start0, region_start + seq_end0


def genomic_strand(
    region_strand: str,
    fimo_strand: str,
    sequence_orientation: str,
) -> str:
    if fimo_strand not in {"+", "-"}:
        return "."
    if sequence_orientation == "strand-aware" and region_strand == "-":
        return _flip_strand(fimo_strand)
    return fimo_strand


def as_single_base_interval(start: int, end: int, mode: str) -> tuple[int, int]:
    if mode == "interval":
        return start, end
    if mode == "start":
        position = start
    elif mode == "end":
        position = end - 1
    elif mode == "center":
        position = (start + end - 1) // 2
    else:
        raise ValueError(f"Unknown site mode: {mode}")
    return position, position + 1


def score_predictions(df: pd.DataFrame, score_mode: str) -> pd.Series:
    if score_mode == "fimo-score":
        return pd.to_numeric(df["score"], errors="raise")

    if score_mode == "neg-log10-pvalue":
        column = "p-value"
    elif score_mode == "neg-log10-qvalue":
        column = "q-value"
    else:
        raise ValueError(f"Unknown score mode: {score_mode}")

    if column not in df.columns:
        raise ValueError(f"FIMO table has no {column!r} column for {score_mode}")

    values = pd.to_numeric(df[column], errors="coerce")
    tiny = 1e-300
    return values.apply(lambda value: -math.log10(max(float(value), tiny)))


def motif_order(feature_idx: object) -> int | None:
    match = re.match(r"^(\d+)-", str(feature_idx))
    return int(match.group(1)) if match else None


def build_feature_ranks(predictions: pd.DataFrame, rank_method: str) -> pd.DataFrame:
    grouped = predictions.groupby("feature_idx", dropna=False)

    if rank_method == "auto":
        orders = predictions["feature_idx"].drop_duplicates().map(motif_order)
        rank_method = "motif-order" if orders.notna().all() else "max-score"

    if rank_method == "motif-order":
        ranks = (
            grouped["score"]
            .max()
            .reset_index(name="max_score")
            .assign(_motif_order=lambda df: df["feature_idx"].map(motif_order))
        )
        missing = ranks["_motif_order"].isna()
        if missing.any():
            bad = ", ".join(map(str, ranks.loc[missing, "feature_idx"].head(5)))
            raise ValueError(
                f"Cannot use motif-order ranking for feature IDs without numeric prefix: {bad}"
            )
        ranks = ranks.sort_values(["_motif_order", "max_score"], ascending=[True, False])
        ranks["feature_score"] = -ranks["_motif_order"].astype(float)
        ranks = ranks.drop(columns=["_motif_order", "max_score"])
    elif rank_method == "max-score":
        ranks = grouped["score"].max().reset_index(name="feature_score")
        ranks = ranks.sort_values(["feature_score", "feature_idx"], ascending=[False, True])
    elif rank_method == "hit-count":
        ranks = grouped.size().reset_index(name="feature_score")
        ranks = ranks.sort_values(["feature_score", "feature_idx"], ascending=[False, True])
    else:
        raise ValueError(f"Unknown rank method: {rank_method}")

    ranks.insert(1, "feature_rank", range(1, len(ranks) + 1))
    return ranks[["feature_idx", "feature_rank", "feature_score"]]


def convert_fimo(
    fimo_path: Path,
    sequence_orientation: str,
    site_mode: str,
    score_mode: str,
    top_n_per_feature: int | None,
) -> pd.DataFrame:
    fimo = read_fimo(fimo_path).copy()
    metadata = (
        fimo["sequence_name"]
        .map(parse_fasta_header)
        .apply(pd.Series)
        .rename(
            columns={
                "chr": "region_chrom",
                "start": "region_start",
                "end": "region_end",
                "strand": "region_strand",
            }
        )
        .drop(columns=["sequence_name"], errors="ignore")
    )

    fimo = fimo.rename(
        columns={
            "start": "fimo_start",
            "stop": "fimo_stop",
            "strand": "fimo_strand",
        }
    )
    combined = pd.concat([fimo.reset_index(drop=True), metadata.reset_index(drop=True)], axis=1)

    intervals = [
        genomic_interval(
            int(row["region_start"]),
            int(row["region_end"]),
            str(row["region_strand"]),
            int(row["fimo_start"]),
            int(row["fimo_stop"]),
            sequence_orientation,
        )
        for row in combined.to_dict(orient="records")
    ]

    starts: list[int] = []
    ends: list[int] = []
    for start, end in intervals:
        site_start, site_end = as_single_base_interval(start, end, site_mode)
        starts.append(site_start)
        ends.append(site_end)

    predictions = pd.DataFrame(
        {
            "chrom": combined["region_chrom"].astype(str),
            "start": starts,
            "end": ends,
            "feature_idx": combined["motif_id"].astype(str),
            "score": score_predictions(combined, score_mode),
            "strand": [
                genomic_strand(str(region_strand), str(fimo_strand), sequence_orientation)
                for region_strand, fimo_strand in zip(
                    combined["region_strand"],
                    combined["fimo_strand"],
                )
            ]
            if "fimo_strand" in combined.columns
            else ".",
            "gene_id": combined["gene_id"].astype(str),
            "sequence_name": combined["sequence_name"].astype(str),
            "fimo_start": pd.to_numeric(combined["fimo_start"], errors="coerce"),
            "fimo_stop": pd.to_numeric(combined["fimo_stop"], errors="coerce"),
        }
    )

    predictions = predictions.dropna(subset=["score"])
    predictions = predictions[predictions["end"] > predictions["start"]]
    predictions = predictions.sort_values(
        ["feature_idx", "score", "chrom", "start"],
        ascending=[True, False, True, True],
    )

    if top_n_per_feature is not None:
        predictions = predictions.groupby("feature_idx", group_keys=False).head(top_n_per_feature)

    return predictions.reset_index(drop=True)


def write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        try:
            df.to_parquet(path, index=False)
        except ImportError as exc:
            try:
                import polars as pl
            except ImportError:
                raise ImportError(
                    f"Writing parquet requires pyarrow/fastparquet for pandas "
                    f"or polars as a fallback. Install one of them or write a "
                    f".tsv/.csv instead: {path}"
                ) from exc

            pl.DataFrame({column: df[column].tolist() for column in df.columns}).write_parquet(path)
    elif suffix == ".tsv":
        df.to_csv(path, sep="\t", index=False)
    elif suffix == ".csv":
        df.to_csv(path, index=False)
    else:
        raise ValueError(f"Unsupported output extension {suffix!r}: {path}")


def default_fimo_path(project: Path, source: str, dataset: str) -> Path:
    return project / "working" / "streme_results" / source / dataset / "fimo.tsv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MEME Suite FIMO hits to Binding Bench prediction inputs."
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path(os.environ.get("ML4RG_PROJECT", DEFAULT_PROJECT)),
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--source", choices=FIMO_SOURCES, default="fimo_streme")
    parser.add_argument("--fimo", type=Path, help="Override input fimo.tsv path.")
    parser.add_argument(
        "--predictions-out",
        type=Path,
        help="Output predictions table (.parquet, .tsv, or .csv).",
    )
    parser.add_argument(
        "--feature-ranks-out",
        type=Path,
        help="Output feature-rank table (.parquet, .tsv, or .csv).",
    )
    parser.add_argument(
        "--sequence-orientation",
        choices=("strand-aware", "genomic"),
        default="strand-aware",
        help=(
            "Use 'strand-aware' if FASTA seq is reverse-complemented for '-' regions; "
            "use 'genomic' if FASTA seq is always chrom:start-end on the plus reference."
        ),
    )
    parser.add_argument(
        "--site-mode",
        choices=("center", "start", "end", "interval"),
        default="center",
        help="How to turn each FIMO match into a Binding Bench predicted site.",
    )
    parser.add_argument(
        "--score-mode",
        choices=("neg-log10-pvalue", "neg-log10-qvalue", "fimo-score"),
        default="neg-log10-pvalue",
    )
    parser.add_argument(
        "--rank-method",
        choices=("auto", "motif-order", "max-score", "hit-count"),
        default="auto",
    )
    parser.add_argument(
        "--top-n-per-feature",
        type=int,
        help="Keep only the top N predictions per feature after scoring.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fimo_path = args.fimo or default_fimo_path(args.project, args.source, args.dataset)
    if not fimo_path.is_file():
        raise FileNotFoundError(f"FIMO table not found: {fimo_path}")

    output_dir = args.project / "working" / "binding_bench_inputs"
    stem = f"{args.source}_{args.dataset}"
    predictions_out = args.predictions_out or output_dir / f"{stem}_predictions.parquet"
    feature_ranks_out = args.feature_ranks_out or output_dir / f"{stem}_feature_ranks.parquet"

    predictions = convert_fimo(
        fimo_path=fimo_path,
        sequence_orientation=args.sequence_orientation,
        site_mode=args.site_mode,
        score_mode=args.score_mode,
        top_n_per_feature=args.top_n_per_feature,
    )
    feature_ranks = build_feature_ranks(predictions, args.rank_method)

    write_table(predictions, predictions_out)
    write_table(feature_ranks, feature_ranks_out)

    print(f"Read FIMO hits: {fimo_path}")
    print(f"Wrote predictions: {predictions_out} ({len(predictions):,} rows)")
    print(f"Wrote feature ranks: {feature_ranks_out} ({len(feature_ranks):,} features)")
    print("Preview:")
    print(predictions.head().to_string(index=False))


if __name__ == "__main__":
    main()
