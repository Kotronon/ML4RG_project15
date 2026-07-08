from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl


DEFAULT_PROJECT = Path("/s/project/ml4rg_students/2026/project15")
DEFAULT_SITES_PATH = Path(
    "/s/project/multispecies/fungi_code/tf_sae/binding_bench_datasets/val/dna/"
    "_saccharomyces_cerevisiae/DNA_rossi_chipexo_sites.parquet"
)
DEFAULT_EMBEDDINGS_PATH = (
    DEFAULT_PROJECT / "working/protein_embeddings/scer_esmdbp_tf_emb.parquet"
)
DEFAULT_FEATURE_DIR = (
    DEFAULT_PROJECT / "working/protein_embeddings/esmdbp_features_scer_tf/features"
)
DEFAULT_FASTA = DEFAULT_PROJECT / "working/SGD/orf_trans_all.fasta.gz"

KEY_COLUMN_CANDIDATES = ("tf", "name", "gene", "orf", "protein_id", "id")
FEATURE_SUFFIXES = (".fea", ".npy", ".npz", ".txt", ".csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether BindingBench TF names resolve to the intended protein "
            "embedding rows and whether parquet vectors match the corresponding "
            "ESM-DBP feature files."
        )
    )
    parser.add_argument("--sites-path", type=Path, default=DEFAULT_SITES_PATH)
    parser.add_argument("--embeddings-path", type=Path, default=DEFAULT_EMBEDDINGS_PATH)
    parser.add_argument("--feature-dir", type=Path, default=DEFAULT_FEATURE_DIR)
    parser.add_argument("--fasta", type=Path, default=DEFAULT_FASTA)
    parser.add_argument("--key-column", default=None)
    parser.add_argument("--embedding-column", default="emb")
    parser.add_argument("--min-sites-per-tf", type=int, default=15)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument(
        "--sample-tfs",
        default="abf1,cbf1,rap1,reb1,brn1,sua7,soh1,tpk1",
        help="Comma-separated TF labels to print explicit mapping rows for.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        help="Optional path for a machine-readable summary JSON.",
    )
    return parser.parse_args()


def normalize(value: object) -> str:
    return str(value).upper()


def sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")


def choose_key_column(table: pl.DataFrame, requested: str | None) -> str:
    if requested is not None:
        if requested not in table.columns:
            raise ValueError(
                f"Embedding table has no key column {requested!r}; "
                f"available columns: {table.columns}"
            )
        return requested
    for column in KEY_COLUMN_CANDIDATES:
        if column in table.columns:
            return column
    raise ValueError(
        f"Could not infer key column. Expected one of {KEY_COLUMN_CANDIDATES}; "
        f"available columns: {table.columns}"
    )


def parse_sgd_gene(description: str) -> str | None:
    parts = description.split()
    if len(parts) >= 2 and re.fullmatch(r"[A-Z0-9-]+", parts[1]):
        return parts[1]
    match = re.search(r"gene=([^\s,;]+)", description)
    if match:
        return match.group(1)
    return None


def read_fasta_metadata(path: Path) -> dict[str, dict[str, str | None]]:
    if not path.exists():
        return {}
    opener = gzip.open if path.suffix == ".gz" else open
    metadata: dict[str, dict[str, str | None]] = {}
    with opener(path, "rt") as handle:
        for line in handle:
            if not line.startswith(">"):
                continue
            description = line[1:].strip()
            if not description:
                continue
            protein_id = description.split()[0]
            gene = parse_sgd_gene(description)
            metadata[normalize(protein_id)] = {
                "protein_id": protein_id,
                "gene": gene,
                "description": description,
            }
    return metadata


def read_bindingbench_tfs(path: Path, min_sites_per_tf: int) -> list[str]:
    sites = pl.read_parquet(path)
    if "name" not in sites.columns:
        raise ValueError(f"Sites table has no 'name' column: {path}")
    sites = sites.with_columns(pl.col("name").cast(pl.Utf8))
    if min_sites_per_tf > 1:
        sites = sites.filter(pl.len().over("name") >= min_sites_per_tf)
    return sites.get_column("name").unique().sort().to_list()


def embedding_list_to_array(value: object) -> np.ndarray:
    if isinstance(value, list):
        return np.asarray(value, dtype=np.float32)
    if hasattr(value, "to_list"):
        return np.asarray(value.to_list(), dtype=np.float32)
    return np.asarray(value, dtype=np.float32)


def load_feature_file(path: Path) -> np.ndarray:
    if path.suffix in {".npy", ".npz"}:
        loaded = np.load(path, allow_pickle=True)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            if not loaded.files:
                raise ValueError(f"Feature file is an empty npz: {path}")
            arr = loaded[loaded.files[0]]
        else:
            arr = loaded
    else:
        arr = np.loadtxt(path, dtype=np.float32)
    arr = np.asarray(arr, dtype=np.float32).squeeze()
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        return arr.mean(axis=0)
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0].mean(axis=0)
    raise ValueError(f"Unsupported feature shape for {path}: {arr.shape}")


def candidate_stems(protein_id: str | None, gene: str | None) -> list[tuple[str, str]]:
    stems: list[tuple[str, str]] = []
    for label, value in (("protein_id", protein_id), ("gene", gene)):
        if not value:
            continue
        raw = str(value)
        for stem in (raw, sanitize(raw)):
            if stem:
                stems.append((label, stem))
        if raw.startswith("Y") and len(raw) > 1:
            stripped = raw[1:]
            stems.append((f"{label}_stripped_leading_Y", stripped))
            stems.append((f"{label}_stripped_leading_Y", sanitize(stripped)))
    return list(dict.fromkeys(stems))


def find_feature(
    feature_dir: Path,
    protein_id: str | None,
    gene: str | None,
) -> tuple[Path | None, str | None]:
    for match_type, stem in candidate_stems(protein_id, gene):
        for suffix in FEATURE_SUFFIXES:
            candidate = feature_dir / f"{stem}{suffix}"
            if candidate.exists():
                return candidate, match_type
    return None, None


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0 or not math.isfinite(denom):
        return float("nan")
    return float(np.dot(a, b) / denom)


def parse_sample_tfs(value: str) -> list[str]:
    return [part.strip() for part in value.replace(";", ",").split(",") if part.strip()]


def print_list(label: str, values: Iterable[str], limit: int = 20) -> None:
    values = sorted(str(value) for value in values)
    preview = values[:limit]
    suffix = "" if len(values) <= limit else f", ... ({len(values)} total)"
    print(f"{label}: {preview}{suffix}")


def main() -> None:
    args = parse_args()

    embeddings = pl.read_parquet(args.embeddings_path)
    if args.embedding_column not in embeddings.columns:
        raise ValueError(
            f"Embedding table has no column {args.embedding_column!r}; "
            f"available columns: {embeddings.columns}"
        )
    key_column = choose_key_column(embeddings, args.key_column)
    tfs = read_bindingbench_tfs(args.sites_path, args.min_sites_per_tf)
    fasta = read_fasta_metadata(args.fasta)

    rows_by_key: dict[str, dict[str, object]] = {}
    duplicate_keys: list[str] = []
    for row in embeddings.iter_rows(named=True):
        key = normalize(row[key_column])
        if key in rows_by_key:
            duplicate_keys.append(key)
        rows_by_key[key] = dict(row)

    tf_keys = {normalize(name) for name in tfs}
    embedding_keys = set(rows_by_key)
    missing_tfs = sorted(tf_keys - embedding_keys)
    extra_embedding_keys = sorted(embedding_keys - tf_keys)

    print("=== Key Mapping ===")
    print(f"sites_path:       {args.sites_path}")
    print(f"embeddings_path:  {args.embeddings_path}")
    print(f"key_column:       {key_column}")
    print(f"embedding_column: {args.embedding_column}")
    print(f"BindingBench TFs after min-sites filter: {len(tfs)}")
    print(f"Embedding rows: {embeddings.height}")
    print(f"Unique embedding keys: {len(embedding_keys)}")
    print(f"Duplicate embedding keys: {len(duplicate_keys)}")
    print(f"Missing TF embeddings: {len(missing_tfs)}")
    print(f"Extra embedding keys: {len(extra_embedding_keys)}")
    print_list("Missing", missing_tfs)
    print_list("Extra", extra_embedding_keys)

    print("\n=== Sample TF Resolution ===")
    for tf_name in parse_sample_tfs(args.sample_tfs):
        row = rows_by_key.get(normalize(tf_name))
        if row is None:
            print(f"{tf_name}: MISSING")
            continue
        print(
            f"{tf_name}: key={row.get(key_column)!r} "
            f"protein_id={row.get('protein_id')!r} "
            f"orf={row.get('orf')!r} gene={row.get('gene')!r}"
        )

    fasta_mismatches = []
    if fasta:
        for row in embeddings.iter_rows(named=True):
            protein_id = str(row["protein_id"]) if "protein_id" in row else None
            gene = str(row["gene"]) if "gene" in row and row["gene"] is not None else None
            if protein_id is None:
                continue
            meta = fasta.get(normalize(protein_id))
            if meta is None:
                fasta_mismatches.append(
                    {"protein_id": protein_id, "gene": gene, "reason": "protein_id_not_in_fasta"}
                )
                continue
            fasta_gene = meta.get("gene")
            if gene and fasta_gene and normalize(gene) != normalize(fasta_gene):
                fasta_mismatches.append(
                    {
                        "protein_id": protein_id,
                        "gene": gene,
                        "fasta_gene": fasta_gene,
                        "reason": "gene_mismatch",
                    }
                )

    print("\n=== FASTA Metadata ===")
    print(f"fasta: {args.fasta}")
    print(f"FASTA records parsed: {len(fasta)}")
    print(f"FASTA mismatches: {len(fasta_mismatches)}")
    for item in fasta_mismatches[:20]:
        print("  ", item)

    print("\n=== Feature File / Parquet Vector Check ===")
    feature_stats = {
        "checked": 0,
        "matched_within_tolerance": 0,
        "missing_feature_file": 0,
        "shape_mismatch": 0,
        "value_mismatch": 0,
        "stripped_leading_y_matches": 0,
    }
    worst: list[dict[str, object]] = []
    examples: list[dict[str, object]] = []

    if not args.feature_dir.exists():
        print(f"feature_dir does not exist: {args.feature_dir}")
    else:
        for row in embeddings.iter_rows(named=True):
            protein_id = str(row["protein_id"]) if "protein_id" in row else None
            gene = str(row["gene"]) if "gene" in row and row["gene"] is not None else None
            parquet_vec = embedding_list_to_array(row[args.embedding_column]).reshape(-1)
            feature_path, match_type = find_feature(args.feature_dir, protein_id, gene)
            if feature_path is None:
                feature_stats["missing_feature_file"] += 1
                examples.append(
                    {
                        "protein_id": protein_id,
                        "gene": gene,
                        "reason": "missing_feature_file",
                    }
                )
                continue
            if match_type and "stripped_leading_Y" in match_type:
                feature_stats["stripped_leading_y_matches"] += 1
            feature_stats["checked"] += 1
            feature_vec = load_feature_file(feature_path).reshape(-1)
            if feature_vec.shape != parquet_vec.shape:
                feature_stats["shape_mismatch"] += 1
                examples.append(
                    {
                        "protein_id": protein_id,
                        "gene": gene,
                        "feature": str(feature_path),
                        "match_type": match_type,
                        "parquet_shape": list(parquet_vec.shape),
                        "feature_shape": list(feature_vec.shape),
                    }
                )
                continue
            diff = float(np.max(np.abs(parquet_vec - feature_vec)))
            cos = cosine(parquet_vec, feature_vec)
            item = {
                "protein_id": protein_id,
                "gene": gene,
                "feature": str(feature_path),
                "match_type": match_type,
                "max_abs_diff": diff,
                "cosine": cos,
            }
            worst.append(item)
            if diff <= args.tolerance:
                feature_stats["matched_within_tolerance"] += 1
            else:
                feature_stats["value_mismatch"] += 1
                if len(examples) < 20:
                    examples.append(item)

        worst.sort(key=lambda item: float(item["max_abs_diff"]), reverse=True)
        print(f"feature_dir: {args.feature_dir}")
        for key, value in feature_stats.items():
            print(f"{key}: {value}")
        print("Worst differences:")
        for item in worst[:10]:
            print(
                "  "
                f"{item['protein_id']} {item['gene']} "
                f"match={item['match_type']} "
                f"max_abs_diff={item['max_abs_diff']:.6g} "
                f"cosine={item['cosine']:.6g} "
                f"file={Path(str(item['feature'])).name}"
            )
        if examples:
            print("Problem examples:")
            for item in examples[:20]:
                print("  ", item)

    summary = {
        "sites_path": str(args.sites_path),
        "embeddings_path": str(args.embeddings_path),
        "feature_dir": str(args.feature_dir),
        "fasta": str(args.fasta),
        "key_column": key_column,
        "embedding_column": args.embedding_column,
        "n_bindingbench_tfs": len(tfs),
        "n_embedding_rows": embeddings.height,
        "n_unique_embedding_keys": len(embedding_keys),
        "duplicate_embedding_keys": duplicate_keys,
        "missing_tfs": missing_tfs,
        "extra_embedding_keys": extra_embedding_keys,
        "fasta_mismatches": fasta_mismatches,
        "feature_stats": feature_stats,
    }
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w") as handle:
            json.dump(summary, handle, indent=2)
        print(f"\nWrote summary JSON: {args.json_out}")


if __name__ == "__main__":
    main()
