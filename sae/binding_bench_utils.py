"""Shared helpers for SAE → Binding Bench export and matched-feature evaluation."""

from __future__ import annotations

from pathlib import Path

try:
    import polars as pl
except ModuleNotFoundError:
    import subprocess
    import sys

    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "polars"])
    import polars as pl

DEFAULT_DATASET = "DNA_rossi_chipexo"


def genomic_pos_for_offset(region: dict[str, object], pos_idx: int) -> int:
    """Map a 0-based sequence offset to an absolute genomic coordinate.

    Matches the alignment used in ``NpyPositionWithLabelsDataset``.
    """
    reg_start = int(region["start"])
    reg_end = int(region["end"])
    strand = str(region["strand"])
    if strand == "+":
        return reg_start + pos_idx
    return reg_end - 1 - pos_idx


def resolve_best_assignment_path(
    path: Path,
    *,
    metric: str = "jaccard",
    dataset: str = DEFAULT_DATASET,
) -> Path:
    """Resolve a Binding Bench discrete dir or direct parquet path."""
    if path.is_file():
        return path

    candidates = [
        path,
        path / "binding_bench" / "discrete" / dataset,
        path / "full_best_assnt" / f"best_assnt_metrics_{metric}_ds_{dataset}.parquet",
    ]
    for candidate in candidates:
        direct = candidate / "full_best_assnt" / f"best_assnt_metrics_{metric}_ds_{dataset}.parquet"
        if direct.is_file():
            return direct
        if candidate.is_file() and candidate.suffix == ".parquet":
            return candidate

    checked = "\n".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"Could not resolve best-assignment parquet from {path}. Checked:\n{checked}"
    )


def load_tf_feature_assignments(path: Path) -> dict[str, int]:
    """Load TF → SAE feature index assignments from a Binding Bench table."""
    df = pl.read_parquet(path)
    required = {"name", "feature_idx"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Best-assignment table missing columns: {sorted(missing)}")

    assignments: dict[str, int] = {}
    for row in df.iter_rows(named=True):
        tf_name = row["name"]
        feature_idx = row["feature_idx"]
        if tf_name is None or feature_idx is None:
            continue
        assignments[str(tf_name)] = int(feature_idx)
    if not assignments:
        raise ValueError(f"No TF→feature assignments found in {path}")
    return assignments
