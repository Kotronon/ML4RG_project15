from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset


DEFAULT_SITES_PATH = Path(
    "/s/project/multispecies/fungi_code/tf_sae/"
    "binding_bench_datasets/val/dna/_saccharomyces_cerevisiae/"
    "DNA_rossi_chipexo_sites.parquet"
)
DEFAULT_REGIONS_PATH = Path(
    "/s/project/multispecies/fungi_code/tf_sae/processed/"
    "sequence_datasets/fungi_upstream_ATG_1000/"
    "_saccharomyces_cerevisiae_sequence_mapper.parquet"
)

SequenceOrientation = Literal["genomic", "strand-aware"]


@dataclass(frozen=True)
class WindowRecord:
    chrom: str
    start: int
    end: int
    center: int
    region_start: int
    region_end: int
    region_strand: str
    gene_id: str
    sequence: str
    labels: np.ndarray
    is_positive: bool


def one_hot_encode_dna(sequence: str) -> torch.Tensor:
    """Encode DNA as a float tensor with shape [4, len(sequence)]."""
    lookup = {
        "A": 0,
        "C": 1,
        "G": 2,
        "T": 3,
        "a": 0,
        "c": 1,
        "g": 2,
        "t": 3,
    }
    encoded = torch.zeros((4, len(sequence)), dtype=torch.float32)
    for pos, base in enumerate(sequence):
        idx = lookup.get(base)
        if idx is not None:
            encoded[idx, pos] = 1.0
    return encoded


class BindingBenchWindowDataset(Dataset):
    """Fixed-width DNA windows with multi-label TF targets.

    The labels come from Binding Bench sites. The sequence comes from the
    sequence-mapper parquet, which already contains promoter DNA in a `seq`
    column. This class deliberately keeps model logic out; training code should
    consume `x`, `y`, and `meta` from `__getitem__`.
    """

    def __init__(
        self,
        sites_path: str | Path = DEFAULT_SITES_PATH,
        regions_path: str | Path = DEFAULT_REGIONS_PATH,
        *,
        window_size: int = 101,
        negative_ratio: float = 1.0,
        min_sites_per_tf: int = 15,
        seed: int = 42,
        sequence_orientation: SequenceOrientation = "strand-aware",
        negative_exclusion_bp: int | None = None,
        max_positive_windows: int | None = None,
    ) -> None:
        if window_size <= 0 or window_size % 2 == 0:
            raise ValueError("window_size must be a positive odd integer")
        if negative_ratio < 0:
            raise ValueError("negative_ratio must be non-negative")

        self.sites_path = Path(sites_path)
        self.regions_path = Path(regions_path)
        self.window_size = window_size
        self.half_window = window_size // 2
        self.sequence_orientation = sequence_orientation
        self.negative_exclusion_bp = (
            self.half_window if negative_exclusion_bp is None else negative_exclusion_bp
        )
        self.rng = np.random.default_rng(seed)

        self.sites = self._read_sites(min_sites_per_tf)
        self.regions = self._read_regions()
        self.tf_names = self.sites["name"].unique().sort().to_list()
        self.tf_to_idx = {name: idx for idx, name in enumerate(self.tf_names)}
        self.idx_to_tf = {idx: name for name, idx in self.tf_to_idx.items()}

        regions_by_key = self._build_region_index(self.regions)
        sites_by_chrom = self._build_sites_by_chrom(self.sites)

        records = self._make_positive_records(regions_by_key, sites_by_chrom)
        if max_positive_windows is not None:
            records = records[:max_positive_windows]
        records.extend(self._make_negative_records(regions_by_key, sites_by_chrom, negative_ratio))

        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, object]:
        record = self.records[idx]
        return {
            "x": one_hot_encode_dna(record.sequence),
            "y": torch.tensor(record.labels, dtype=torch.float32),
            "meta": {
                "chrom": record.chrom,
                "start": record.start,
                "end": record.end,
                "center": record.center,
                "gene_id": record.gene_id,
                "region_start": record.region_start,
                "region_end": record.region_end,
                "region_strand": record.region_strand,
                "is_positive": record.is_positive,
            },
        }

    def summary(self) -> dict[str, object]:
        positives = sum(record.is_positive for record in self.records)
        return {
            "n_windows": len(self.records),
            "n_positive_windows": positives,
            "n_negative_windows": len(self.records) - positives,
            "n_tfs": len(self.tf_names),
            "window_size": self.window_size,
            "sequence_orientation": self.sequence_orientation,
        }

    def _read_sites(self, min_sites_per_tf: int) -> pl.DataFrame:
        sites = pl.read_parquet(self.sites_path)
        required = {"chrom", "start", "end", "strand", "name"}
        missing = required - set(sites.columns)
        if missing:
            raise ValueError(f"Sites table is missing columns: {sorted(missing)}")

        sites = sites.with_columns(
            pl.col("chrom").cast(pl.Utf8),
            pl.col("start").cast(pl.Int64),
            pl.col("end").cast(pl.Int64),
            pl.col("strand").cast(pl.Utf8),
            pl.col("name").cast(pl.Utf8),
        )
        if min_sites_per_tf > 1:
            sites = sites.filter(pl.len().over("name") >= min_sites_per_tf)
        if sites.is_empty():
            raise ValueError("No sites remain after filtering")
        return sites

    def _read_regions(self) -> pl.DataFrame:
        regions = pl.read_parquet(self.regions_path)
        required = {"chrom", "start", "end", "strand", "seq", "gene_id"}
        missing = required - set(regions.columns)
        if missing:
            raise ValueError(f"Regions table is missing columns: {sorted(missing)}")

        return regions.with_columns(
            pl.col("chrom").cast(pl.Utf8),
            pl.col("start").cast(pl.Int64),
            pl.col("end").cast(pl.Int64),
            pl.col("strand").cast(pl.Utf8),
            pl.col("seq").cast(pl.Utf8),
            pl.col("gene_id").cast(pl.Utf8),
        )

    @staticmethod
    def _build_region_index(regions: pl.DataFrame) -> dict[str, list[dict[str, object]]]:
        by_chrom: dict[str, list[dict[str, object]]] = {}
        for row in regions.iter_rows(named=True):
            by_chrom.setdefault(str(row["chrom"]), []).append(row)
        for rows in by_chrom.values():
            rows.sort(key=lambda row: int(row["start"]))
        return by_chrom

    def _build_sites_by_chrom(self, sites: pl.DataFrame) -> dict[str, list[dict[str, object]]]:
        by_chrom: dict[str, list[dict[str, object]]] = {}
        for row in sites.iter_rows(named=True):
            row = dict(row)
            row["center"] = self._site_center(row)
            by_chrom.setdefault(str(row["chrom"]), []).append(row)
        for rows in by_chrom.values():
            rows.sort(key=lambda row: int(row["center"]))
        return by_chrom

    @staticmethod
    def _site_center(site: dict[str, object]) -> int:
        start = int(site["start"])
        end = int(site["end"])
        if end <= start:
            return start
        return (start + end - 1) // 2

    def _find_region(
        self, regions_by_key: dict[str, list[dict[str, object]]], chrom: str, center: int
    ) -> dict[str, object] | None:
        for region in regions_by_key.get(chrom, []):
            if int(region["start"]) <= center < int(region["end"]):
                return region
        return None

    def _make_positive_records(
        self,
        regions_by_key: dict[str, list[dict[str, object]]],
        sites_by_chrom: dict[str, list[dict[str, object]]],
    ) -> list[WindowRecord]:
        records: list[WindowRecord] = []
        seen: set[tuple[str, int]] = set()
        for chrom, sites in sites_by_chrom.items():
            for site in sites:
                center = int(site["center"])
                key = (chrom, center)
                if key in seen:
                    continue
                seen.add(key)

                region = self._find_region(regions_by_key, chrom, center)
                if region is None:
                    continue

                labels = self._labels_for_window(chrom, center, sites_by_chrom)
                records.append(self._record_from_center(region, center, labels, True))
        return records

    def _make_negative_records(
        self,
        regions_by_key: dict[str, list[dict[str, object]]],
        sites_by_chrom: dict[str, list[dict[str, object]]],
        negative_ratio: float,
    ) -> list[WindowRecord]:
        n_positives = sum(len(sites) for sites in sites_by_chrom.values())
        n_negatives = int(round(n_positives * negative_ratio))
        if n_negatives == 0:
            return []

        flat_regions = [region for regions in regions_by_key.values() for region in regions]
        records: list[WindowRecord] = []
        max_attempts = max(1000, n_negatives * 100)
        attempts = 0

        while len(records) < n_negatives and attempts < max_attempts:
            attempts += 1
            region = flat_regions[int(self.rng.integers(0, len(flat_regions)))]
            start = int(region["start"]) + self.half_window
            end = int(region["end"]) - self.half_window
            if end <= start:
                continue

            center = int(self.rng.integers(start, end))
            chrom = str(region["chrom"])
            if self._has_nearby_site(chrom, center, sites_by_chrom, self.negative_exclusion_bp):
                continue

            labels = np.zeros(len(self.tf_names), dtype=np.float32)
            records.append(self._record_from_center(region, center, labels, False))

        if len(records) < n_negatives:
            raise RuntimeError(
                f"Could only sample {len(records)} negative windows out of requested "
                f"{n_negatives}. Try lowering negative_ratio or negative_exclusion_bp."
            )
        return records

    def _labels_for_window(
        self,
        chrom: str,
        center: int,
        sites_by_chrom: dict[str, list[dict[str, object]]],
    ) -> np.ndarray:
        labels = np.zeros(len(self.tf_names), dtype=np.float32)
        lo = center - self.half_window
        hi = center + self.half_window + 1
        for site in sites_by_chrom.get(chrom, []):
            site_center = int(site["center"])
            if site_center < lo:
                continue
            if site_center >= hi:
                break
            labels[self.tf_to_idx[str(site["name"])]] = 1.0
        return labels

    @staticmethod
    def _has_nearby_site(
        chrom: str,
        center: int,
        sites_by_chrom: dict[str, list[dict[str, object]]],
        radius: int,
    ) -> bool:
        for site in sites_by_chrom.get(chrom, []):
            site_center = int(site["center"])
            if site_center < center - radius:
                continue
            if site_center > center + radius:
                break
            return True
        return False

    def _record_from_center(
        self,
        region: dict[str, object],
        center: int,
        labels: np.ndarray,
        is_positive: bool,
    ) -> WindowRecord:
        seq = self._extract_window_sequence(region, center)
        start = center - self.half_window
        end = center + self.half_window + 1
        return WindowRecord(
            chrom=str(region["chrom"]),
            start=start,
            end=end,
            center=center,
            region_start=int(region["start"]),
            region_end=int(region["end"]),
            region_strand=str(region["strand"]),
            gene_id=str(region["gene_id"]),
            sequence=seq,
            labels=labels,
            is_positive=is_positive,
        )

    def _extract_window_sequence(self, region: dict[str, object], center: int) -> str:
        seq = str(region["seq"])
        region_start = int(region["start"])
        region_end = int(region["end"])
        strand = str(region["strand"])

        if self.sequence_orientation == "genomic" or strand != "-":
            offset = center - region_start
        elif self.sequence_orientation == "strand-aware":
            offset = region_end - center - 1
        else:
            raise ValueError(f"Unsupported sequence_orientation: {self.sequence_orientation}")

        left = offset - self.half_window
        right = offset + self.half_window + 1
        pad_left = max(0, -left)
        pad_right = max(0, right - len(seq))
        clipped = seq[max(0, left) : min(len(seq), right)]
        return "N" * pad_left + clipped + "N" * pad_right
