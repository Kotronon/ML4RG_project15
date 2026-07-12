from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


DEFAULT_SITES = Path(
    "project15/working/binding_datasets/scer_pugh_rossi_dedup/"
    "scer_pugh_rossi_dedup.sites.parquet"
)
DEFAULT_REGIONS = Path(
    "project15/working/sequence_datasets/fungi_upstream_ATG_1000/"
    "_saccharomyces_cerevisiae_sequence_mapper.parquet"
)
DEFAULT_PROMOTER_SPLIT = Path(
    "project15/working/binding_datasets/scer_pugh_rossi_dedup/"
    "scer_pugh_rossi_dedup.promoter_split_random_80_10_10.json"
)
DEFAULT_TF_SPLIT = Path(
    "project15/working/binding_datasets/scer_pugh_rossi_dedup/"
    "scer_pugh_rossi_dedup.tf_split_named_similarity_80_10_10.json"
)
DEFAULT_OUTPUT_DIR = Path("project15/working/binding_bench_inputs/dna_only_heldout_eval")
MINIMAL_SITE_COLUMNS = ["chrom", "start", "end", "name", "score", "strand"]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_names(names: list[str], path: Path) -> None:
    path.write_text("\n".join(names) + "\n")


def normalize_names(names: list[object]) -> list[str]:
    return sorted({str(name).strip().lower() for name in names if str(name).strip()})


def subset_regions(regions: pd.DataFrame, gene_ids: list[object]) -> pd.DataFrame:
    gene_id_set = {str(gene_id) for gene_id in gene_ids}
    out = regions[regions["gene_id"].astype(str).isin(gene_id_set)].copy()
    if out.empty:
        raise ValueError("No promoter regions remain after split filtering")
    return out.reset_index(drop=True)


def tf_sites(sites: pd.DataFrame, tf_names: list[str]) -> pd.DataFrame:
    tf_set = set(tf_names)
    out = sites[sites["name"].astype(str).str.lower().isin(tf_set)].copy()
    if out.empty:
        raise ValueError(f"No sites remain for TF names: {tf_names[:5]}")
    out["name"] = out["name"].astype(str).str.lower()
    out["score"] = 1.0
    out["strand"] = "."
    return (
        out[MINIMAL_SITE_COLUMNS]
        .drop_duplicates(["chrom", "start", "end", "name"])
        .sort_values(["chrom", "start", "end", "name"])
        .reset_index(drop=True)
    )


def merged_sites(sites: pd.DataFrame, tf_names: list[str], merged_name: str) -> pd.DataFrame:
    out = tf_sites(sites, tf_names)
    out["name"] = merged_name
    return (
        out[MINIMAL_SITE_COLUMNS]
        .drop_duplicates(["chrom", "start", "end", "name"])
        .sort_values(["chrom", "start", "end", "name"])
        .reset_index(drop=True)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare Binding Bench sites/region/filter inputs from the exact "
            "promoter and TF split JSON files used for supervised training."
        )
    )
    parser.add_argument("--sites-path", type=Path, default=DEFAULT_SITES)
    parser.add_argument("--regions-path", type=Path, default=DEFAULT_REGIONS)
    parser.add_argument("--promoter-split-path", type=Path, default=DEFAULT_PROMOTER_SPLIT)
    parser.add_argument("--tf-split-path", type=Path, default=DEFAULT_TF_SPLIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    sites = pd.read_parquet(args.sites_path)
    regions = pd.read_parquet(args.regions_path)
    promoter_split = read_json(args.promoter_split_path)
    tf_split = read_json(args.tf_split_path)

    train_tf_names = normalize_names(tf_split["train_tf_names"])
    val_tf_names = normalize_names(tf_split.get("val_tf_names", []))
    test_tf_names = normalize_names(tf_split["test_tf_names"])

    outputs: dict[str, Path] = {}
    for split_name in ("train", "val", "test"):
        key = f"{split_name}_gene_ids"
        if key not in promoter_split:
            continue
        path = args.output_dir / f"{split_name}_promoter_regions.parquet"
        subset_regions(regions, promoter_split[key]).to_parquet(path, index=False)
        outputs[f"{split_name}_regions"] = path

    name_sets = {
        "train_tfs": train_tf_names,
        "val_tfs": val_tf_names,
        "test_tfs": test_tf_names,
    }
    for label, names in name_sets.items():
        if not names:
            continue
        names_path = args.output_dir / f"{label}_names.txt"
        filter_path = args.output_dir / f"{label}_feature_filter.txt"
        sites_path = args.output_dir / f"{label}_sites.parquet"
        write_names(names, names_path)
        write_names(names, filter_path)
        tf_sites(sites, names).to_parquet(sites_path, index=False)
        outputs[f"{label}_names"] = names_path
        outputs[f"{label}_feature_filter"] = filter_path
        outputs[f"{label}_sites"] = sites_path

        legacy_name = label.removesuffix("s")
        legacy_names_path = args.output_dir / f"{legacy_name}_names.txt"
        if legacy_names_path != names_path:
            write_names(names, legacy_names_path)
            outputs[f"{legacy_name}_names"] = legacy_names_path

    merged_specs = {
        "merged_train_tfs": train_tf_names,
        "merged_test_tfs": test_tf_names,
    }
    for merged_name, names in merged_specs.items():
        sites_path = args.output_dir / f"{merged_name}_sites.parquet"
        filter_path = args.output_dir / f"{merged_name}_feature_filter.txt"
        merged_sites(sites, names, merged_name).to_parquet(sites_path, index=False)
        write_names([merged_name], filter_path)
        outputs[f"{merged_name}_sites"] = sites_path
        outputs[f"{merged_name}_feature_filter"] = filter_path

    print("Prepared Binding Bench split inputs:")
    for key, path in sorted(outputs.items()):
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
