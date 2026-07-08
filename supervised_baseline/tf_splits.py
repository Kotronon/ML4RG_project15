from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np


SPLIT_NAMES = ("train", "val", "test")


def make_all_train_tf_split(tf_names: Iterable[str]) -> dict[str, object]:
    names = [str(name) for name in tf_names]
    return {
        "metadata": {
            "method": "none",
            "n_tfs": len(names),
        },
        "train_tf_names": names,
        "val_tf_names": [],
        "test_tf_names": [],
    }


def make_random_tf_split(
    tf_names: Iterable[str],
    *,
    seed: int,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
) -> dict[str, object]:
    names = [str(name) for name in tf_names]
    if len(names) < 3:
        raise ValueError("At least three TFs are required for a train/val/test split")
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("--tf-train-fraction must be between 0 and 1")
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("--tf-val-fraction must be between 0 and 1")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("Train and validation fractions must leave room for test TFs")

    shuffled = names[:]
    random.Random(seed).shuffle(shuffled)

    n_tfs = len(shuffled)
    n_train = int(round(n_tfs * train_fraction))
    n_val = int(round(n_tfs * val_fraction))
    n_train = min(max(n_train, 1), n_tfs - 2)
    n_val = min(max(n_val, 1), n_tfs - n_train - 1)

    return {
        "metadata": {
            "method": "random",
            "seed": seed,
            "train_fraction": train_fraction,
            "val_fraction": val_fraction,
            "test_fraction": 1.0 - train_fraction - val_fraction,
            "n_tfs": n_tfs,
        },
        "train_tf_names": sorted(shuffled[:n_train]),
        "val_tf_names": sorted(shuffled[n_train : n_train + n_val]),
        "test_tf_names": sorted(shuffled[n_train + n_val :]),
    }


def _normalize_names(names: Iterable[str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for name in names:
        value = str(name)
        key = value.upper()
        if key in normalized and normalized[key] != value:
            raise ValueError(
                f"TF names collide after case normalization: {normalized[key]!r}, {value!r}"
            )
        normalized[key] = value
    return normalized


def _parse_tf_names(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        raw = value.replace(";", ",").split(",")
    else:
        raw = list(value)
    return [str(name).strip() for name in raw if str(name).strip()]


def make_named_holdout_tf_split(
    tf_names: Iterable[str],
    *,
    test_tfs: str | Iterable[str],
    val_tfs: str | Iterable[str] = (),
) -> dict[str, object]:
    names = [str(name) for name in tf_names]
    name_lookup = _normalize_names(names)
    requested_test = _parse_tf_names(test_tfs)
    requested_val = _parse_tf_names(val_tfs)
    if not requested_test:
        raise ValueError("--tf-test-names is required for named TF split")

    test_keys = {name.upper() for name in requested_test}
    val_keys = {name.upper() for name in requested_val}
    overlap = sorted(test_keys & val_keys)
    if overlap:
        raise ValueError(f"Validation and test TF names overlap: {overlap}")

    missing = sorted((test_keys | val_keys) - set(name_lookup))
    if missing:
        raise ValueError(f"Requested holdout TFs are absent from the dataset: {missing}")

    test = sorted(name_lookup[key] for key in test_keys)
    val = sorted(name_lookup[key] for key in val_keys)
    holdout = set(test) | set(val)
    train = sorted(name for name in names if name not in holdout)
    if not train:
        raise ValueError("Named TF split would leave no training TFs")

    return {
        "metadata": {
            "method": "named",
            "val_tfs": val,
            "test_tfs": test,
            "n_tfs": len(names),
        },
        "train_tf_names": train,
        "val_tf_names": val,
        "test_tf_names": test,
    }


def _connected_similarity_clusters(
    names: list[str],
    embeddings: np.ndarray,
    threshold: float,
) -> list[list[str]]:
    if embeddings.ndim != 2:
        raise ValueError("TF embeddings for similarity split must have shape [n_tfs, dim]")
    if embeddings.shape[0] != len(names):
        raise ValueError(
            f"Expected {len(names)} TF embeddings for similarity split, "
            f"got {embeddings.shape[0]}"
        )
    if not (-1.0 <= threshold <= 1.0):
        raise ValueError("--tf-similarity-threshold must be between -1 and 1")

    matrix = embeddings.astype(np.float64, copy=True)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.maximum(norms, 1e-12)
    similarity = matrix @ matrix.T

    seen = np.zeros(len(names), dtype=bool)
    clusters: list[list[str]] = []
    for start in range(len(names)):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        members: list[int] = []
        while stack:
            idx = stack.pop()
            members.append(idx)
            neighbors = np.flatnonzero((similarity[idx] >= threshold) & ~seen)
            for neighbor in neighbors.tolist():
                seen[neighbor] = True
                stack.append(int(neighbor))
        clusters.append(sorted(names[idx] for idx in members))
    return clusters


def _cluster_split_similarity(
    split_names: dict[str, list[str]],
    names: list[str],
    embeddings: np.ndarray,
) -> dict[str, float | None]:
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    matrix = embeddings.astype(np.float64, copy=True)
    matrix = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)

    out: dict[str, float | None] = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        left_idx = [name_to_idx[name] for name in split_names[left]]
        right_idx = [name_to_idx[name] for name in split_names[right]]
        key = f"max_{left}_{right}_cosine"
        if not left_idx or not right_idx:
            out[key] = None
            continue
        values = matrix[left_idx] @ matrix[right_idx].T
        out[key] = float(values.max())
    return out


def make_similarity_holdout_tf_split(
    tf_names: Iterable[str],
    embeddings: np.ndarray,
    *,
    seed: int,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    similarity_threshold: float = 0.9,
) -> dict[str, object]:
    names = [str(name) for name in tf_names]
    if len(names) < 3:
        raise ValueError("At least three TFs are required for a train/val/test split")
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("--tf-train-fraction must be between 0 and 1")
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("--tf-val-fraction must be between 0 and 1")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("Train and validation fractions must leave room for test TFs")

    embeddings = np.asarray(embeddings)
    clusters = _connected_similarity_clusters(names, embeddings, similarity_threshold)
    if len(clusters) < 3:
        raise ValueError(
            "Similarity threshold produced fewer than three TF clusters. "
            "Increase --tf-similarity-threshold."
        )

    n_tfs = len(names)
    n_train = int(round(n_tfs * train_fraction))
    n_val = int(round(n_tfs * val_fraction))
    n_train = min(max(n_train, 1), n_tfs - 2)
    n_val = min(max(n_val, 1), n_tfs - n_train - 1)
    targets = {
        "train": n_train,
        "val": n_val,
        "test": n_tfs - n_train - n_val,
    }

    rng = random.Random(seed)
    rng.shuffle(clusters)
    clusters.sort(key=lambda cluster: (-len(cluster), cluster[0]))

    split_clusters: dict[str, list[list[str]]] = {"train": [], "val": [], "test": []}
    split_counts = {name: 0 for name in split_clusters}

    for split_name, cluster in zip(("train", "val", "test"), clusters[:3]):
        split_clusters[split_name].append(cluster)
        split_counts[split_name] += len(cluster)

    for cluster in clusters[3:]:
        def deficit(split_name: str) -> tuple[float, int]:
            target = max(targets[split_name], 1)
            return (
                (targets[split_name] - split_counts[split_name]) / target,
                -split_counts[split_name],
            )

        split_name = max(split_clusters, key=deficit)
        split_clusters[split_name].append(cluster)
        split_counts[split_name] += len(cluster)

    split_names = {
        split_name: sorted(name for cluster in clusters_ for name in cluster)
        for split_name, clusters_ in split_clusters.items()
    }
    cluster_sizes = sorted((len(cluster) for cluster in clusters), reverse=True)
    similarity_summary = _cluster_split_similarity(split_names, names, embeddings)

    return {
        "metadata": {
            "method": "similarity",
            "seed": seed,
            "train_fraction": train_fraction,
            "val_fraction": val_fraction,
            "test_fraction": 1.0 - train_fraction - val_fraction,
            "similarity_threshold": similarity_threshold,
            "n_tfs": n_tfs,
            "n_clusters": len(clusters),
            "cluster_size_min": min(cluster_sizes),
            "cluster_size_median": cluster_sizes[len(cluster_sizes) // 2],
            "cluster_size_max": max(cluster_sizes),
            **similarity_summary,
        },
        "train_tf_names": split_names["train"],
        "val_tf_names": split_names["val"],
        "test_tf_names": split_names["test"],
    }


def make_named_similarity_holdout_tf_split(
    tf_names: Iterable[str],
    embeddings: np.ndarray,
    *,
    seed: int,
    test_tfs: str | Iterable[str],
    val_tfs: str | Iterable[str] = (),
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    similarity_threshold: float = 0.9,
) -> dict[str, object]:
    """Hold out named TFs, then distribute remaining similarity clusters.

    Clusters containing requested validation or test TFs are pinned to that
    split. This keeps near-duplicate TFs with the named holdout and avoids
    train/test leakage. Unpinned clusters are assigned greedily to the split
    with the largest remaining size deficit.
    """

    names = [str(name) for name in tf_names]
    if len(names) < 3:
        raise ValueError("At least three TFs are required for a train/val/test split")
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("--tf-train-fraction must be between 0 and 1")
    if not (0.0 < val_fraction < 1.0):
        raise ValueError("--tf-val-fraction must be between 0 and 1")
    if train_fraction + val_fraction >= 1.0:
        raise ValueError("Train and validation fractions must leave room for test TFs")

    name_lookup = _normalize_names(names)
    requested_test = _parse_tf_names(test_tfs)
    requested_val = _parse_tf_names(val_tfs)
    if not requested_test:
        raise ValueError("--tf-test-names is required for named_similarity TF split")

    test_keys = {name.upper() for name in requested_test}
    val_keys = {name.upper() for name in requested_val}
    overlap = sorted(test_keys & val_keys)
    if overlap:
        raise ValueError(f"Validation and test TF names overlap: {overlap}")

    missing = sorted((test_keys | val_keys) - set(name_lookup))
    if missing:
        raise ValueError(f"Requested holdout TFs are absent from the dataset: {missing}")

    fixed_test = {name_lookup[key] for key in test_keys}
    fixed_val = {name_lookup[key] for key in val_keys}

    embeddings = np.asarray(embeddings)
    clusters = _connected_similarity_clusters(names, embeddings, similarity_threshold)
    if len(clusters) < 3:
        raise ValueError(
            "Similarity threshold produced fewer than three TF clusters. "
            "Increase --tf-similarity-threshold."
        )

    n_tfs = len(names)
    n_train = int(round(n_tfs * train_fraction))
    n_val = int(round(n_tfs * val_fraction))
    n_train = min(max(n_train, 1), n_tfs - 2)
    n_val = min(max(n_val, 1), n_tfs - n_train - 1)
    targets = {
        "train": n_train,
        "val": n_val,
        "test": n_tfs - n_train - n_val,
    }

    rng = random.Random(seed)
    rng.shuffle(clusters)
    clusters.sort(key=lambda cluster: (-len(cluster), cluster[0]))

    split_clusters: dict[str, list[list[str]]] = {"train": [], "val": [], "test": []}
    split_counts = {name: 0 for name in split_clusters}
    unpinned_clusters: list[list[str]] = []
    pinned_val_clusters = 0
    pinned_test_clusters = 0

    for cluster in clusters:
        cluster_names = set(cluster)
        has_val = bool(cluster_names & fixed_val)
        has_test = bool(cluster_names & fixed_test)
        if has_val and has_test:
            raise ValueError(
                "A similarity cluster contains both validation and test holdout TFs: "
                + ", ".join(cluster)
            )
        if has_val:
            split_clusters["val"].append(cluster)
            split_counts["val"] += len(cluster)
            pinned_val_clusters += 1
        elif has_test:
            split_clusters["test"].append(cluster)
            split_counts["test"] += len(cluster)
            pinned_test_clusters += 1
        else:
            unpinned_clusters.append(cluster)

    if not split_clusters["val"] and unpinned_clusters:
        cluster = unpinned_clusters.pop(0)
        split_clusters["val"].append(cluster)
        split_counts["val"] += len(cluster)
    if not split_clusters["train"] and unpinned_clusters:
        cluster = unpinned_clusters.pop(0)
        split_clusters["train"].append(cluster)
        split_counts["train"] += len(cluster)

    for cluster in unpinned_clusters:
        def deficit(split_name: str) -> tuple[float, int]:
            target = max(targets[split_name], 1)
            return (
                (targets[split_name] - split_counts[split_name]) / target,
                -split_counts[split_name],
            )

        split_name = max(split_clusters, key=deficit)
        split_clusters[split_name].append(cluster)
        split_counts[split_name] += len(cluster)

    split_names = {
        split_name: sorted(name for cluster in clusters_ for name in cluster)
        for split_name, clusters_ in split_clusters.items()
    }
    if not split_names["train"]:
        raise ValueError("named_similarity TF split would leave no training TFs")
    if not split_names["val"]:
        raise ValueError("named_similarity TF split would leave no validation TFs")
    if not split_names["test"]:
        raise ValueError("named_similarity TF split would leave no test TFs")

    cluster_sizes = sorted((len(cluster) for cluster in clusters), reverse=True)
    similarity_summary = _cluster_split_similarity(split_names, names, embeddings)

    return {
        "metadata": {
            "method": "named_similarity",
            "seed": seed,
            "train_fraction": train_fraction,
            "val_fraction": val_fraction,
            "test_fraction": 1.0 - train_fraction - val_fraction,
            "similarity_threshold": similarity_threshold,
            "requested_val_tfs": sorted(fixed_val),
            "requested_test_tfs": sorted(fixed_test),
            "pinned_val_clusters": pinned_val_clusters,
            "pinned_test_clusters": pinned_test_clusters,
            "n_tfs": n_tfs,
            "n_clusters": len(clusters),
            "cluster_size_min": min(cluster_sizes),
            "cluster_size_median": cluster_sizes[len(cluster_sizes) // 2],
            "cluster_size_max": max(cluster_sizes),
            **similarity_summary,
        },
        "train_tf_names": split_names["train"],
        "val_tf_names": split_names["val"],
        "test_tf_names": split_names["test"],
    }


def load_tf_split(path: str | Path) -> dict[str, object]:
    with Path(path).open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"TF split must be a JSON object: {path}")
    return payload


def normalize_tf_split(
    split: dict[str, object],
    tf_names: Iterable[str],
) -> dict[str, object]:
    all_names = [str(name) for name in tf_names]
    name_to_idx = {name: idx for idx, name in enumerate(all_names)}
    normalized: dict[str, object] = {
        "metadata": dict(split.get("metadata", {}))
        if isinstance(split.get("metadata", {}), dict)
        else {},
    }
    seen: set[str] = set()

    for split_name in SPLIT_NAMES:
        key = f"{split_name}_tf_names"
        raw_names = split.get(key, split.get(split_name, []))
        if not isinstance(raw_names, list):
            raise ValueError(f"TF split field {key!r} must be a list")
        names = [str(name) for name in raw_names]
        missing = sorted(name for name in names if name not in name_to_idx)
        if missing:
            preview = ", ".join(missing[:10])
            raise ValueError(
                f"{len(missing)} TFs in split {split_name!r} are absent "
                f"from the dataset: {preview}"
            )
        overlap = sorted(seen.intersection(names))
        if overlap:
            preview = ", ".join(overlap[:10])
            raise ValueError(f"TF split {split_name!r} overlaps earlier split: {preview}")
        seen.update(names)
        normalized[key] = names
        normalized[f"{split_name}_indices"] = [name_to_idx[name] for name in names]

    if not normalized["train_indices"]:
        raise ValueError("TF split must contain at least one train TF")

    normalized["omitted_tf_names"] = sorted(set(all_names) - seen)
    normalized["counts"] = {
        split_name: len(normalized[f"{split_name}_indices"])
        for split_name in SPLIT_NAMES
    } | {"omitted": len(normalized["omitted_tf_names"])}
    return normalized


def save_tf_split(path: str | Path, split: dict[str, object]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        json.dump(split, handle, indent=2)


def tf_split_indices(split: dict[str, object], split_name: str) -> list[int]:
    if split_name not in SPLIT_NAMES:
        raise ValueError(f"Unknown TF split: {split_name!r}")
    value = split.get(f"{split_name}_indices", [])
    if not isinstance(value, list):
        raise ValueError(f"TF split has invalid {split_name}_indices")
    return [int(idx) for idx in value]
