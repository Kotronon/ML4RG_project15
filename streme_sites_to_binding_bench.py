from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

from fimo_to_binding_bench import (
    as_single_base_interval,
    build_feature_ranks,
    genomic_interval,
    genomic_strand,
    is_low_complexity_motif,
    parse_fasta_header,
    write_table,
)


DEFAULT_PROJECT = "/s/project/ml4rg_students/2026/project15"
DEFAULT_DATASET = "_saccharomyces_cerevisiae_sequence_mapper"


def default_sites_path(project: Path, result_tag: str, dataset: str) -> Path:
    result_dir = project / "working" / "streme_results"
    if result_tag:
        result_dir = result_dir / result_tag
    return result_dir / "streme" / dataset / "sites.tsv"


def read_streme_sites(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", comment="#")
    if df.empty:
        raise ValueError(f"STREME sites table is empty: {path}")

    required = {
        "motif_ID",
        "seq_ID",
        "site_Start",
        "site_End",
        "site_Strand",
        "site_Score",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"STREME sites table {path} is missing columns: {sorted(missing)}")

    df["site_Start"] = pd.to_numeric(df["site_Start"], errors="raise").astype(int)
    df["site_End"] = pd.to_numeric(df["site_End"], errors="raise").astype(int)
    df["site_Score"] = pd.to_numeric(df["site_Score"], errors="coerce")
    return df


def convert_streme_sites(
    sites_path: Path,
    sequence_orientation: str,
    site_mode: str,
    top_n_per_feature: int | None,
) -> pd.DataFrame:
    sites = read_streme_sites(sites_path).copy()
    metadata = (
        sites["seq_ID"]
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

    combined = pd.concat([sites.reset_index(drop=True), metadata.reset_index(drop=True)], axis=1)
    intervals = [
        genomic_interval(
            int(row["region_start"]),
            int(row["region_end"]),
            str(row["region_strand"]),
            int(row["site_Start"]),
            int(row["site_End"]),
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
            "feature_idx": combined["motif_ID"].astype(str),
            "score": combined["site_Score"],
            "strand": [
                genomic_strand(str(region_strand), str(site_strand), sequence_orientation)
                for region_strand, site_strand in zip(
                    combined["region_strand"],
                    combined["site_Strand"],
                )
            ],
            "gene_id": combined["gene_id"].astype(str),
            "sequence_name": combined["seq_ID"].astype(str),
            "site_start": pd.to_numeric(combined["site_Start"], errors="coerce"),
            "site_end": pd.to_numeric(combined["site_End"], errors="coerce"),
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert MEME Suite STREME sites.tsv to Binding Bench prediction inputs."
    )
    parser.add_argument(
        "--project",
        type=Path,
        default=Path(os.environ.get("ML4RG_PROJECT", DEFAULT_PROJECT)),
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument(
        "--result-tag",
        default="",
        help="Tagged streme_results subdirectory, for example 'nmotifs_150'.",
    )
    parser.add_argument("--sites", type=Path, help="Override input STREME sites.tsv path.")
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
    )
    parser.add_argument(
        "--site-mode",
        choices=("center", "start", "end", "interval"),
        default="center",
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
    parser.add_argument(
        "--filter-low-complexity",
        action="store_true",
        help=(
            "Remove motifs with long homopolymers, simple dinucleotide repeats, "
            "or at least 85%% A/T content at length 10 or greater."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sites_path = args.sites or default_sites_path(args.project, args.result_tag, args.dataset)
    if not sites_path.is_file():
        raise FileNotFoundError(f"STREME sites table not found: {sites_path}")

    output_dir = args.project / "working" / "binding_bench_inputs"
    tag = f"_{args.result_tag}" if args.result_tag else ""
    stem = f"streme_sites_{args.dataset}{tag}"
    predictions_out = args.predictions_out or output_dir / f"{stem}_predictions.parquet"
    feature_ranks_out = args.feature_ranks_out or output_dir / f"{stem}_feature_ranks.parquet"

    predictions = convert_streme_sites(
        sites_path=sites_path,
        sequence_orientation=args.sequence_orientation,
        site_mode=args.site_mode,
        top_n_per_feature=args.top_n_per_feature,
    )
    removed_features: list[str] = []
    if args.filter_low_complexity:
        features = predictions["feature_idx"].drop_duplicates()
        removed_features = sorted(
            str(feature) for feature in features if is_low_complexity_motif(feature)
        )
        predictions = predictions.loc[
            ~predictions["feature_idx"].isin(removed_features)
        ].reset_index(drop=True)
        if predictions.empty:
            raise ValueError("Low-complexity filtering removed every motif.")

    feature_ranks = build_feature_ranks(predictions, args.rank_method)

    write_table(predictions, predictions_out)
    write_table(feature_ranks, feature_ranks_out)

    print(f"Read STREME sites: {sites_path}")
    print(f"Wrote predictions: {predictions_out} ({len(predictions):,} rows)")
    print(f"Wrote feature ranks: {feature_ranks_out} ({len(feature_ranks):,} features)")
    if args.filter_low_complexity:
        print(
            f"Removed low-complexity motifs ({len(removed_features)}): "
            + ", ".join(removed_features)
        )
    print("Preview:")
    print(predictions.head().to_string(index=False))


if __name__ == "__main__":
    main()
