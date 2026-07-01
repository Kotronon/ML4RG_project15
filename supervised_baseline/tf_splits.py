from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterable


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
