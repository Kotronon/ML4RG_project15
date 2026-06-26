from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .dataset import DEFAULT_SITES_PATH
except ImportError:
    from dataset import DEFAULT_SITES_PATH

import polars as pl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export TF labels present in a Binding Bench sites parquet."
    )
    parser.add_argument("--sites-path", type=Path, default=DEFAULT_SITES_PATH)
    parser.add_argument("--output", type=Path, default=Path("binding_bench_tf_names.txt"))
    parser.add_argument("--min-sites-per-tf", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sites = pl.read_parquet(args.sites_path)
    if "name" not in sites.columns:
        raise ValueError(f"Sites table has no 'name' column: {args.sites_path}")
    names = (
        sites.with_columns(pl.col("name").cast(pl.Utf8))
        .filter(pl.len().over("name") >= args.min_sites_per_tf)
        .select("name")
        .unique()
        .sort("name")
        .get_column("name")
        .to_list()
    )
    if not names:
        raise ValueError("No TF names remain after filtering")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(str(name) for name in names) + "\n")
    print(f"Wrote {len(names)} TF names to {args.output}")


if __name__ == "__main__":
    main()
