from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable


SPLIT_NAMES = ("train", "val", "test")


def _record_identity(record: object, idx: int) -> dict[str, object]:
    return {
        "idx": idx,
        "gene_id": str(getattr(record, "gene_id")),
        "chrom": str(getattr(record, "chrom")),
        "start": int(getattr(record, "start")),
        "end": int(getattr(record, "end")),
        "strand": str(getattr(record, "strand")),
    }


def _records_by_gene_id(records: Iterable[object]) -> dict[str, int]:
    by_gene: dict[str, int] = {}
    for idx, record in enumerate(records):
        gene_id = str(getattr(record, "gene_id"))
        if gene_id in by_gene:
            raise ValueError(f"Duplicate gene_id in promoter records: {gene_id}")
        by_gene[gene_id] = idx
    return by_gene


def make_all_train_split(records: list[object]) -> dict[str, object]:
    return {
        "metadata": {
            "method": "none",
            "n_records": len(records),
        },
        "train_gene_ids": [str(getattr(record, "gene_id")) for record in records],
        "val_gene_ids": [],
        "test_gene_ids": [],
    }


def make_random_promoter_split(
    records: list[object],
    *,
    seed: int,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
) -> dict[str, object]:
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("--train-fraction must be between 0 and 1")
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("--val-fraction must be between 0 and 1")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("Train and validation fractions must leave room for test promoters")
    if len(records) < 3:
        raise ValueError("At least three promoters are required for a train/val/test split")

    identities = [_record_identity(record, idx) for idx, record in enumerate(records)]
    random.Random(seed).shuffle(identities)

    n_records = len(identities)
    n_train = int(round(n_records * train_fraction))
    n_val = int(round(n_records * val_fraction))
    n_train = min(max(n_train, 1), n_records - 2)
    n_val = min(max(n_val, 1), n_records - n_train - 1)

    train = identities[:n_train]
    val = identities[n_train : n_train + n_val]
    test = identities[n_train + n_val :]

    return {
        "metadata": {
            "method": "random",
            "seed": seed,
            "train_fraction": train_fraction,
            "val_fraction": val_fraction,
            "test_fraction": 1.0 - train_fraction - val_fraction,
            "n_records": n_records,
        },
        "train_gene_ids": sorted(str(row["gene_id"]) for row in train),
        "val_gene_ids": sorted(str(row["gene_id"]) for row in val),
        "test_gene_ids": sorted(str(row["gene_id"]) for row in test),
    }


def _parse_chroms(chroms: str | Iterable[str]) -> set[str]:
    if isinstance(chroms, str):
        raw = chroms.replace(";", ",").split(",")
    else:
        raw = list(chroms)
    return {str(chrom).strip() for chrom in raw if str(chrom).strip()}


def make_chromosome_promoter_split(
    records: list[object],
    *,
    val_chroms: str | Iterable[str],
    test_chroms: str | Iterable[str],
) -> dict[str, object]:
    val = _parse_chroms(val_chroms)
    test = _parse_chroms(test_chroms)
    if not val:
        raise ValueError("--val-chroms is required for chromosome promoter split")
    if not test:
        raise ValueError("--test-chroms is required for chromosome promoter split")
    overlap = sorted(val & test)
    if overlap:
        raise ValueError(f"Validation and test chromosomes overlap: {overlap}")

    train_gene_ids: list[str] = []
    val_gene_ids: list[str] = []
    test_gene_ids: list[str] = []
    seen_chroms: set[str] = set()

    for record in records:
        gene_id = str(getattr(record, "gene_id"))
        chrom = str(getattr(record, "chrom"))
        seen_chroms.add(chrom)
        if chrom in val:
            val_gene_ids.append(gene_id)
        elif chrom in test:
            test_gene_ids.append(gene_id)
        else:
            train_gene_ids.append(gene_id)

    missing_val = sorted(val - seen_chroms)
    missing_test = sorted(test - seen_chroms)
    if missing_val or missing_test:
        raise ValueError(
            "Requested holdout chromosomes are absent from promoter records: "
            f"val={missing_val}, test={missing_test}"
        )

    return {
        "metadata": {
            "method": "chromosome",
            "val_chroms": sorted(val),
            "test_chroms": sorted(test),
            "n_records": len(records),
        },
        "train_gene_ids": sorted(train_gene_ids),
        "val_gene_ids": sorted(val_gene_ids),
        "test_gene_ids": sorted(test_gene_ids),
    }


def load_promoter_split(path: str | Path) -> dict[str, object]:
    with Path(path).open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Promoter split must be a JSON object: {path}")
    return payload


def normalize_promoter_split(
    split: dict[str, object],
    records: list[object],
) -> dict[str, object]:
    by_gene = _records_by_gene_id(records)
    normalized: dict[str, object] = {
        "metadata": dict(split.get("metadata", {}))
        if isinstance(split.get("metadata", {}), dict)
        else {},
    }
    seen: set[str] = set()

    for split_name in SPLIT_NAMES:
        key = f"{split_name}_gene_ids"
        raw_gene_ids = split.get(key, split.get(split_name, []))
        if not isinstance(raw_gene_ids, list):
            raise ValueError(f"Promoter split field {key!r} must be a list")
        gene_ids = [str(gene_id) for gene_id in raw_gene_ids]
        missing = sorted(gene_id for gene_id in gene_ids if gene_id not in by_gene)
        if missing:
            preview = ", ".join(missing[:10])
            raise ValueError(
                f"{len(missing)} promoters in split {split_name!r} are absent "
                f"from the dataset: {preview}"
            )
        overlap = sorted(seen.intersection(gene_ids))
        if overlap:
            preview = ", ".join(overlap[:10])
            raise ValueError(
                f"Promoter split {split_name!r} overlaps with an earlier split: {preview}"
            )
        seen.update(gene_ids)
        normalized[key] = gene_ids
        normalized[f"{split_name}_indices"] = [by_gene[gene_id] for gene_id in gene_ids]

    if not normalized["train_indices"]:
        raise ValueError("Promoter split must contain at least one train promoter")

    all_gene_ids = {str(getattr(record, "gene_id")) for record in records}
    normalized["omitted_gene_ids"] = sorted(all_gene_ids - seen)
    normalized["counts"] = {
        split_name: len(normalized[f"{split_name}_indices"])
        for split_name in SPLIT_NAMES
    } | {"omitted": len(normalized["omitted_gene_ids"])}
    normalized["chrom_counts"] = {
        split_name: _chrom_counts(
            records,
            [int(idx) for idx in normalized[f"{split_name}_indices"]],
        )
        for split_name in SPLIT_NAMES
    }
    return normalized


def _chrom_counts(records: list[object], indices: list[int]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for idx in indices:
        chrom = str(getattr(records[idx], "chrom"))
        counts[chrom] = counts.get(chrom, 0) + 1
    return dict(sorted(counts.items()))


def save_promoter_split(path: str | Path, split: dict[str, object]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        json.dump(split, handle, indent=2)


def split_indices(split: dict[str, object], split_name: str) -> list[int]:
    if split_name not in SPLIT_NAMES:
        raise ValueError(f"Unknown promoter split: {split_name!r}")
    value = split.get(f"{split_name}_indices", [])
    if not isinstance(value, list):
        raise ValueError(f"Promoter split has invalid {split_name}_indices")
    return [int(idx) for idx in value]
