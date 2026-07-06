from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from baseline_evaluations.combine_binding_sites import read_sites
from supervised_baseline.dataset import (  # noqa: E402
    DEFAULT_REGIONS_PATH,
    DEFAULT_SITES_PATH,
    BindingBenchPromoterSequenceDataset,
)
from supervised_baseline.promoter_splits import (  # noqa: E402
    make_random_promoter_split,
    normalize_promoter_split,
    save_promoter_split,
)
from supervised_baseline.tf_embeddings import (  # noqa: E402
    available_embedding_keys,
    load_tf_embeddings,
)
from supervised_baseline.tf_splits import (  # noqa: E402
    make_named_similarity_holdout_tf_split,
    make_random_tf_split,
    normalize_tf_split,
    save_tf_split,
)


FULL_PUGH_SITES_PATH = Path(
    "/s/project/multispecies/fungi_code/tf_sae/processed/binding_sites/dna/"
    "_saccharomyces_cerevisiae/_saccharomyces_cerevisiae_pugh_chipexo.parquet"
)
ROSSI_TEST_SITES_PATH = Path(
    "/s/project/multispecies/fungi_code/tf_sae/binding_bench_datasets/test/dna/"
    "_saccharomyces_cerevisiae/DNA_rossi_chipexo_sites.parquet"
)
DEFAULT_TF_EMBEDDINGS_PATH = Path(
    "/s/project/ml4rg_students/2026/project15/working/protein_embeddings/"
    "scer_esm2_tf_emb.parquet"
)
DEFAULT_OUTPUT_DIR = Path(
    "/s/project/ml4rg_students/2026/project15/working/binding_datasets/"
    "scer_pugh_rossi_dedup"
)


def parse_name_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [
        item.strip()
        for item in value.replace(";", ",").split(",")
        if item.strip()
    ]


def write_lines(path: Path, values: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for value in values:
            handle.write(f"{value}\n")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2)


def combine_sites(
    inputs: list[Path],
    output: Path,
    dedupe_columns: list[str],
) -> dict[str, object]:
    tables = [read_sites(path, idx) for idx, path in enumerate(inputs)]
    combined = pl.concat(tables, how="diagonal_relaxed")
    before = combined.height
    combined = combined.unique(subset=dedupe_columns, keep="first").sort(
        ["chrom", "start", "end", "name"]
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(output)

    summary = {
        "inputs": [str(path) for path in inputs],
        "output": str(output),
        "dedupe_columns": dedupe_columns,
        "rows_before_deduplication": before,
        "rows_after_deduplication": combined.height,
        "duplicates_removed": before - combined.height,
        "n_tfs": combined.select(pl.col("name").n_unique()).item(),
    }
    write_json(output.with_suffix(".summary.json"), summary)
    return summary


def split_counts(split: dict[str, object], prefix: str) -> dict[str, int]:
    return {
        name: len(split.get(f"{name}_{prefix}", []))
        for name in ("train", "val", "test")
    } | {
        "omitted": len(split.get(f"omitted_{prefix}", [])),
    }


def make_embedding_command(
    *,
    output_path: Path,
    keep_names_path: Path,
    repo_root: Path,
) -> str:
    return "\n".join(
        [
            f"OUTPUT={output_path} \\",
            f"KEEP_NAMES={keep_names_path} \\",
            "BATCH_SIZE=8 \\",
            "MAX_LENGTH=1024 \\",
            f"sbatch {repo_root / 'slurm' / 'embed_esm2.sbatch'}",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deduplicated Saccharomyces binding-site bundle with "
            "ready-to-use promoter and TF split JSON files."
        )
    )
    parser.add_argument(
        "--sites-input",
        action="append",
        type=Path,
        default=None,
        help=(
            "Input Binding Bench-compatible sites parquet. Pass multiple times. "
            "Defaults to current Rossi val sites, full Pugh sites, and Rossi test sites."
        ),
    )
    parser.add_argument("--regions-path", type=Path, default=DEFAULT_REGIONS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prefix", default="scer_pugh_rossi_dedup")
    parser.add_argument(
        "--dedupe-columns",
        default="chrom,start,end,strand,name",
        help="Comma-separated columns used to drop duplicate sites.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--min-sites-per-tf", type=int, default=15)
    parser.add_argument(
        "--tf-embeddings-path",
        type=Path,
        default=DEFAULT_TF_EMBEDDINGS_PATH,
        help=(
            "Protein embedding table used to restrict protein-model splits to TFs "
            "with available embeddings. Set to an absent path to write an ESM-2 "
            "generation command instead."
        ),
    )
    parser.add_argument("--tf-embedding-key-column", default="gene")
    parser.add_argument("--tf-embedding-column", default="emb")
    parser.add_argument("--tf-name-map", type=Path, default=None)
    parser.add_argument(
        "--named-test-tfs",
        default="",
        help=(
            "Comma- or semicolon-separated seed TFs for the named-similarity "
            "test split. Related TFs above --similarity-threshold are held out too."
        ),
    )
    parser.add_argument(
        "--named-val-tfs",
        default="",
        help="Optional comma- or semicolon-separated seed TFs for validation.",
    )
    parser.add_argument("--similarity-threshold", type=float, default=0.99)
    parser.add_argument(
        "--max-regions",
        type=int,
        default=None,
        help="Debug/smoke-test option. Leave unset for the real shared bundle.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sites_inputs = args.sites_input or [
        DEFAULT_SITES_PATH,
        FULL_PUGH_SITES_PATH,
        ROSSI_TEST_SITES_PATH,
    ]
    dedupe_columns = [
        column.strip()
        for column in args.dedupe_columns.split(",")
        if column.strip()
    ]
    if not dedupe_columns:
        raise ValueError("--dedupe-columns must not be empty")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sites_path = args.output_dir / f"{args.prefix}.sites.parquet"
    sites_summary = combine_sites(sites_inputs, sites_path, dedupe_columns)

    all_tf_names = sorted(
        pl.read_parquet(sites_path)
        .select(pl.col("name").cast(pl.Utf8).unique().sort())
        .get_column("name")
        .to_list()
    )
    all_tf_names_path = args.output_dir / f"{args.prefix}.all_tf_names.txt"
    write_lines(all_tf_names_path, all_tf_names)

    embedding_filter: set[str] | None = None
    embedding_status: dict[str, object]
    if args.tf_embeddings_path.exists():
        embedding_filter = available_embedding_keys(
            args.tf_embeddings_path,
            key_column=args.tf_embedding_key_column,
            name_mapping_path=args.tf_name_map,
        )
        embedding_status = {
            "mode": "existing",
            "path": str(args.tf_embeddings_path),
            "key_column": args.tf_embedding_key_column,
            "embedding_column": args.tf_embedding_column,
            "n_embedding_keys": len(embedding_filter),
        }
    else:
        embedding_status = {
            "mode": "missing",
            "path": str(args.tf_embeddings_path),
            "message": "Embedding table not found; use the generated ESM-2 command.",
        }

    dataset = BindingBenchPromoterSequenceDataset(
        sites_path=sites_path,
        regions_path=args.regions_path,
        min_sites_per_tf=args.min_sites_per_tf,
        tf_name_filter=embedding_filter,
        max_regions=args.max_regions,
    )
    dataset_summary = dataset.summary()

    tf_names = [str(name) for name in dataset.tf_names]
    tf_names_path = args.output_dir / f"{args.prefix}.tf_names.txt"
    write_lines(tf_names_path, tf_names)

    promoter_split = normalize_promoter_split(
        make_random_promoter_split(
            dataset.records,
            seed=args.seed,
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
        ),
        dataset.records,
    )
    promoter_split_path = (
        args.output_dir / f"{args.prefix}.promoter_split_random_80_10_10.json"
    )
    save_promoter_split(promoter_split_path, promoter_split)

    random_tf_split = normalize_tf_split(
        make_random_tf_split(
            tf_names,
            seed=args.seed,
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
        ),
        tf_names,
    )
    random_tf_split_path = args.output_dir / f"{args.prefix}.tf_split_random_80_10_10.json"
    save_tf_split(random_tf_split_path, random_tf_split)

    named_tf_split_path: Path | None = None
    named_tf_split_counts: dict[str, int] | None = None
    named_test_tfs = parse_name_list(args.named_test_tfs)
    named_val_tfs = parse_name_list(args.named_val_tfs)
    if named_test_tfs:
        if not args.tf_embeddings_path.exists():
            raise FileNotFoundError(
                "--named-test-tfs requires --tf-embeddings-path for similarity clusters"
            )
        tf_embeddings, tf_embedding_metadata = load_tf_embeddings(
            args.tf_embeddings_path,
            tf_names,
            key_column=args.tf_embedding_key_column,
            embedding_column=args.tf_embedding_column,
            name_mapping_path=args.tf_name_map,
        )
        named_tf_split = normalize_tf_split(
            make_named_similarity_holdout_tf_split(
                tf_names,
                tf_embeddings.numpy(),
                seed=args.seed,
                test_tfs=named_test_tfs,
                val_tfs=named_val_tfs,
                train_fraction=args.train_fraction,
                val_fraction=args.val_fraction,
                similarity_threshold=args.similarity_threshold,
            ),
            tf_names,
        )
        named_tf_split["metadata"]["tf_embedding_metadata"] = tf_embedding_metadata
        named_tf_split_path = (
            args.output_dir / f"{args.prefix}.tf_split_named_similarity_80_10_10.json"
        )
        save_tf_split(named_tf_split_path, named_tf_split)
        named_tf_split_counts = split_counts(named_tf_split, "tf_names")

    planned_embedding_path = args.output_dir / f"{args.prefix}.esm2_tf_emb.parquet"
    embedding_command = make_embedding_command(
        output_path=planned_embedding_path,
        keep_names_path=all_tf_names_path,
        repo_root=REPO_ROOT,
    )
    embedding_command_path = args.output_dir / f"{args.prefix}.embed_esm2_command.sh"
    embedding_command_path.write_text(embedding_command)

    manifest = {
        "bundle": {
            "output_dir": str(args.output_dir),
            "prefix": args.prefix,
            "sites_path": str(sites_path),
            "regions_path": str(args.regions_path),
            "min_sites_per_tf": args.min_sites_per_tf,
            "max_regions": args.max_regions,
            "seed": args.seed,
            "train_fraction": args.train_fraction,
            "val_fraction": args.val_fraction,
            "test_fraction": 1.0 - args.train_fraction - args.val_fraction,
        },
        "sites": sites_summary,
        "dataset": dataset_summary,
        "tf_names": {
            "all_site_tf_names_path": str(all_tf_names_path),
            "effective_tf_names_path": str(tf_names_path),
            "n_all_site_tfs": len(all_tf_names),
            "n_effective_tfs": len(tf_names),
            "effective_tf_names_are_embedding_filtered": embedding_filter is not None,
        },
        "protein_embeddings": {
            **embedding_status,
            "esm2_command_path": str(embedding_command_path),
            "planned_esm2_output": str(planned_embedding_path),
        },
        "splits": {
            "promoter_random_80_10_10": {
                "path": str(promoter_split_path),
                "counts": split_counts(promoter_split, "gene_ids"),
            },
            "tf_random_80_10_10": {
                "path": str(random_tf_split_path),
                "counts": split_counts(random_tf_split, "tf_names"),
            },
            "tf_named_similarity_80_10_10": (
                {
                    "path": str(named_tf_split_path),
                    "counts": named_tf_split_counts,
                    "requested_test_tfs": named_test_tfs,
                    "requested_val_tfs": named_val_tfs,
                }
                if named_tf_split_path is not None
                else None
            ),
        },
        "training_example_env": {
            "SITES_PATH": str(sites_path),
            "PROMOTER_SPLIT_PATH": str(promoter_split_path),
            "TF_SPLIT_PATH": str(named_tf_split_path or random_tf_split_path),
            "TF_EMBEDDINGS_PATH": str(args.tf_embeddings_path),
            "MIN_SITES_PER_TF": str(args.min_sites_per_tf),
        },
    }
    manifest_path = args.output_dir / f"{args.prefix}.manifest.json"
    write_json(manifest_path, manifest)

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
