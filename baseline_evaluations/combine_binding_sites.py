from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


REQUIRED_COLUMNS = ("chrom", "start", "end", "name", "score", "strand")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine Binding Bench-compatible binding-site parquet files."
    )
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        type=Path,
        help="Input sites parquet. Pass multiple times.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--dedupe-columns",
        default="chrom,start,end,strand,name",
        help="Comma-separated columns used to drop duplicate sites.",
    )
    return parser.parse_args()


def read_sites(path: Path, source_idx: int) -> pl.DataFrame:
    table = pl.read_parquet(path)
    missing = set(REQUIRED_COLUMNS) - set(table.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return table.with_columns(
        pl.col("chrom").cast(pl.Utf8),
        pl.col("start").cast(pl.Int64),
        pl.col("end").cast(pl.Int64),
        pl.col("name").cast(pl.Utf8),
        pl.col("score").cast(pl.Utf8),
        pl.col("strand").cast(pl.Utf8),
        pl.lit(str(path)).alias("source_path"),
        pl.lit(source_idx).alias("source_idx"),
    )


def main() -> None:
    args = parse_args()
    dedupe_columns = [
        column.strip()
        for column in args.dedupe_columns.split(",")
        if column.strip()
    ]
    if not dedupe_columns:
        raise ValueError("--dedupe-columns must not be empty")

    tables = [read_sites(path, idx) for idx, path in enumerate(args.input)]
    combined = pl.concat(tables, how="diagonal_relaxed")
    before = combined.height
    combined = combined.unique(subset=dedupe_columns, keep="first").sort(
        ["chrom", "start", "end", "name"]
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(args.output)

    summary = {
        "inputs": [str(path) for path in args.input],
        "output": str(args.output),
        "dedupe_columns": dedupe_columns,
        "rows_before_deduplication": before,
        "rows_after_deduplication": combined.height,
        "duplicates_removed": before - combined.height,
        "n_tfs": combined.select(pl.col("name").n_unique()).item(),
    }
    summary_path = args.output.with_suffix(".summary.json")
    with summary_path.open("w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
