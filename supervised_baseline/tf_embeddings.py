from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import polars as pl
import torch


DEFAULT_EMBEDDING_COLUMN = "emb"
KEY_COLUMN_CANDIDATES = ("tf", "name", "gene", "orf", "protein_id", "id")


def normalize_key(value: object) -> str:
    return str(value).upper()


def read_name_mapping(path: str | Path | None) -> dict[str, str]:
    if path is None:
        return {}
    with Path(path).open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"TF name mapping must be a JSON object: {path}")
    return {str(key): str(value) for key, value in payload.items()}


def choose_key_column(table: pl.DataFrame, requested: str | None = None) -> str:
    if requested is not None:
        if requested not in table.columns:
            raise ValueError(
                f"Embedding table has no key column {requested!r}. "
                f"Available columns: {table.columns}"
            )
        return requested

    for column in KEY_COLUMN_CANDIDATES:
        if column in table.columns:
            return column
    raise ValueError(
        "Could not infer embedding key column. Expected one of "
        f"{KEY_COLUMN_CANDIDATES}; available columns: {table.columns}"
    )


def available_embedding_keys(
    path: str | Path,
    *,
    key_column: str | None = None,
    name_mapping_path: str | Path | None = None,
) -> set[str]:
    table = pl.read_parquet(path)
    key_column = choose_key_column(table, key_column)
    keys = {normalize_key(value) for value in table.get_column(key_column).to_list()}
    keys.update(normalize_key(key) for key in read_name_mapping(name_mapping_path))
    return keys


def load_tf_embeddings(
    path: str | Path,
    tf_names: Iterable[str],
    *,
    key_column: str | None = None,
    embedding_column: str = DEFAULT_EMBEDDING_COLUMN,
    name_mapping_path: str | Path | None = None,
) -> tuple[torch.Tensor, dict[str, object]]:
    table = pl.read_parquet(path)
    if embedding_column not in table.columns:
        raise ValueError(
            f"Embedding table has no embedding column {embedding_column!r}. "
            f"Available columns: {table.columns}"
        )
    key_column = choose_key_column(table, key_column)
    name_mapping = read_name_mapping(name_mapping_path)

    embeddings_by_key: dict[str, tuple[str, list[float]]] = {}
    for row in table.select(key_column, embedding_column).iter_rows(named=True):
        key = str(row[key_column])
        normalized_key = normalize_key(key)
        if normalized_key in embeddings_by_key:
            raise ValueError(f"Duplicate TF embedding key {key!r} in {path}")
        embedding = row[embedding_column]
        if not isinstance(embedding, list):
            raise ValueError(
                f"Embedding for key {key!r} is not a list in column {embedding_column!r}"
            )
        embeddings_by_key[normalized_key] = (key, [float(value) for value in embedding])

    vectors: list[list[float]] = []
    missing: list[str] = []
    resolved_keys: dict[str, str] = {}
    for tf_name in tf_names:
        tf_name = str(tf_name)
        key = name_mapping.get(tf_name, tf_name)
        resolved = embeddings_by_key.get(normalize_key(key))
        if resolved is None:
            missing.append(tf_name)
            continue
        resolved_key, embedding = resolved
        resolved_keys[tf_name] = resolved_key
        vectors.append(embedding)

    if missing:
        preview = ", ".join(missing[:20])
        suffix = "" if len(missing) <= 20 else f", ... ({len(missing)} total)"
        raise ValueError(
            "Missing protein embeddings for TF labels: "
            f"{preview}{suffix}. Provide --tf-name-map if dataset labels and "
            "embedding keys use different identifiers."
        )
    if not vectors:
        raise ValueError(f"No embeddings loaded from {path}")

    width = len(vectors[0])
    if any(len(vector) != width for vector in vectors):
        raise ValueError("Protein embeddings have inconsistent dimensions")

    tensor = torch.tensor(vectors, dtype=torch.float32)
    metadata = {
        "path": str(path),
        "key_column": key_column,
        "embedding_column": embedding_column,
        "name_mapping_path": str(name_mapping_path) if name_mapping_path else None,
        "embedding_dim": width,
        "resolved_keys": resolved_keys,
    }
    return tensor, metadata
