from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

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
        tf_name_filter: Iterable[str] | None = None,
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

        self.sites = self._read_sites(min_sites_per_tf, tf_name_filter)
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
                "record_idx": idx,
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

    def _read_sites(
        self,
        min_sites_per_tf: int,
        tf_name_filter: Iterable[str] | None,
    ) -> pl.DataFrame:
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
        if tf_name_filter is not None:
            keep_names = {str(name).upper() for name in tf_name_filter}
            sites = sites.filter(pl.col("name").str.to_uppercase().is_in(keep_names))
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


@dataclass(frozen=True)
class PromoterRecord:
    chrom: str
    start: int
    end: int
    strand: str
    gene_id: str
    sequence: str
    label_intervals: tuple[tuple[int, int, int], ...]


class BindingBenchPromoterBaseDataset(Dataset):
    """Whole-promoter records with dense TF-by-position Binding Bench labels.

    The model position axis follows the returned input sequence. With
    ``sequence_orientation="strand-aware"``, labels for minus-strand promoters
    are reversed so offset 0 corresponds to the first base of the stored
    promoter sequence.
    """

    def __init__(
        self,
        sites_path: str | Path = DEFAULT_SITES_PATH,
        regions_path: str | Path = DEFAULT_REGIONS_PATH,
        *,
        min_sites_per_tf: int = 15,
        sequence_orientation: SequenceOrientation = "strand-aware",
        tf_name_filter: Iterable[str] | None = None,
        max_regions: int | None = None,
    ) -> None:
        if sequence_orientation not in {"genomic", "strand-aware"}:
            raise ValueError(f"Unsupported sequence_orientation: {sequence_orientation}")

        self.sites_path = Path(sites_path)
        self.regions_path = Path(regions_path)
        self.sequence_orientation = sequence_orientation

        self.sites = self._read_sites(min_sites_per_tf, tf_name_filter)
        self.regions = self._read_regions()
        if max_regions is not None:
            self.regions = self.regions.head(max_regions)

        self.tf_names = self.sites["name"].unique().sort().to_list()
        self.tf_to_idx = {name: idx for idx, name in enumerate(self.tf_names)}
        self.idx_to_tf = {idx: name for name, idx in self.tf_to_idx.items()}

        sites_by_chrom = self._build_sites_by_chrom(self.sites)
        self.records = self._make_records(self.regions, sites_by_chrom)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, object]:
        record = self.records[idx]
        x = self._get_x(idx, record)
        labels = self._dense_labels(record)
        mask = self._position_mask(record)
        return {
            "x": x,
            "y": torch.tensor(labels, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.bool),
            "meta": {
                "record_idx": idx,
                "chrom": record.chrom,
                "start": record.start,
                "end": record.end,
                "strand": record.strand,
                "gene_id": record.gene_id,
                "length": len(record.sequence),
                "n_label_intervals": len(record.label_intervals),
            },
        }

    def _get_x(self, idx: int, record: PromoterRecord) -> torch.Tensor:
        raise NotImplementedError

    def summary(self) -> dict[str, object]:
        total_positions = sum(len(record.sequence) for record in self.records)
        positive_tf_positions = 0
        bound_promoters = 0
        for record in self.records:
            if record.label_intervals:
                bound_promoters += 1
            positive_tf_positions += sum(
                max(0, hi - lo) for _, lo, hi in record.label_intervals
            )
        return {
            "n_promoters": len(self.records),
            "n_bound_promoters": bound_promoters,
            "n_tfs": len(self.tf_names),
            "sequence_orientation": self.sequence_orientation,
            "total_positions": total_positions,
            "positive_tf_positions": positive_tf_positions,
        }

    def genomic_to_offset(self, region: dict[str, object], genomic_pos: int) -> int:
        start = int(region["start"])
        end = int(region["end"])
        strand = str(region["strand"])

        if not (start <= genomic_pos < end):
            raise ValueError(f"Position {genomic_pos} is outside region {start}-{end}")

        if self.sequence_orientation == "strand-aware" and strand == "-":
            return end - genomic_pos - 1
        return genomic_pos - start

    def offset_to_genomic(self, region: dict[str, object], offset: int) -> int:
        start = int(region["start"])
        end = int(region["end"])
        strand = str(region["strand"])
        length = end - start

        if not (0 <= offset < length):
            raise ValueError(f"Offset {offset} is outside region length {length}")

        if self.sequence_orientation == "strand-aware" and strand == "-":
            return end - offset - 1
        return start + offset

    def _read_sites(
        self,
        min_sites_per_tf: int,
        tf_name_filter: Iterable[str] | None,
    ) -> pl.DataFrame:
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
        if tf_name_filter is not None:
            keep_names = {str(name).upper() for name in tf_name_filter}
            sites = sites.filter(pl.col("name").str.to_uppercase().is_in(keep_names))
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
    def _build_sites_by_chrom(
        sites: pl.DataFrame,
    ) -> dict[str, list[dict[str, object]]]:
        by_chrom: dict[str, list[dict[str, object]]] = {}
        for row in sites.iter_rows(named=True):
            site = dict(row)
            by_chrom.setdefault(str(site["chrom"]), []).append(site)
        for rows in by_chrom.values():
            rows.sort(key=lambda row: int(row["start"]))
        return by_chrom

    def _make_records(
        self,
        regions: pl.DataFrame,
        sites_by_chrom: dict[str, list[dict[str, object]]],
    ) -> list[PromoterRecord]:
        records: list[PromoterRecord] = []
        for region_idx, region_row in enumerate(regions.iter_rows(named=True)):
            region = dict(region_row)
            sequence = str(region["seq"])
            region_start = int(region["start"])
            region_end = int(region["end"])
            length = len(sequence)
            genomic_length = region_end - region_start
            if length != genomic_length:
                raise ValueError(
                    "Promoter sequence length does not match genomic span for "
                    f"row {region_idx}: len(seq)={length}, span={genomic_length}"
                )

            intervals = self._label_intervals_for_region(region, sites_by_chrom)
            records.append(
                PromoterRecord(
                    chrom=str(region["chrom"]),
                    start=region_start,
                    end=region_end,
                    strand=str(region["strand"]),
                    gene_id=str(region["gene_id"]),
                    sequence=sequence,
                    label_intervals=tuple(intervals),
                )
            )
        return records

    def _label_intervals_for_region(
        self,
        region: dict[str, object],
        sites_by_chrom: dict[str, list[dict[str, object]]],
    ) -> list[tuple[int, int, int]]:
        chrom = str(region["chrom"])
        region_start = int(region["start"])
        region_end = int(region["end"])
        intervals: list[tuple[int, int, int]] = []

        for site in sites_by_chrom.get(chrom, []):
            site_start = int(site["start"])
            site_end = int(site["end"])
            if site_end <= region_start:
                continue
            if site_start >= region_end:
                break

            overlap_start = max(region_start, site_start)
            overlap_end = min(region_end, site_end)
            if overlap_end <= overlap_start:
                continue

            tf_idx = self.tf_to_idx[str(site["name"])]
            lo, hi = self._interval_to_offsets(region, overlap_start, overlap_end)
            if hi > lo:
                intervals.append((tf_idx, lo, hi))
        return intervals

    def _interval_to_offsets(
        self,
        region: dict[str, object],
        genomic_start: int,
        genomic_end: int,
    ) -> tuple[int, int]:
        region_start = int(region["start"])
        region_end = int(region["end"])
        strand = str(region["strand"])

        if self.sequence_orientation == "strand-aware" and strand == "-":
            lo = region_end - genomic_end
            hi = region_end - genomic_start
        else:
            lo = genomic_start - region_start
            hi = genomic_end - region_start

        length = region_end - region_start
        lo = max(0, min(length, lo))
        hi = max(0, min(length, hi))
        return lo, hi

    def _dense_labels(self, record: PromoterRecord) -> np.ndarray:
        labels = np.zeros((len(self.tf_names), len(record.sequence)), dtype=np.float32)
        for tf_idx, lo, hi in record.label_intervals:
            labels[tf_idx, lo:hi] = 1.0
        return labels

    @staticmethod
    def _position_mask(record: PromoterRecord) -> np.ndarray:
        valid_bases = {"A", "C", "G", "T", "a", "c", "g", "t"}
        return np.fromiter(
            (base in valid_bases for base in record.sequence),
            dtype=bool,
            count=len(record.sequence),
        )


class BindingBenchPromoterSequenceDataset(BindingBenchPromoterBaseDataset):
    """Whole-promoter raw DNA dataset returning one-hot inputs [4, L]."""

    def _get_x(self, idx: int, record: PromoterRecord) -> torch.Tensor:
        return one_hot_encode_dna(record.sequence)


class BindingBenchPromoterDataset(BindingBenchPromoterSequenceDataset):
    """Backward-compatible name for the raw promoter sequence dataset."""


class BindingBenchPromoterEmbeddingDataset(BindingBenchPromoterBaseDataset):
    """Whole-promoter dataset returning precomputed position embeddings [L, D].

    ``.npy`` embeddings are interpreted as an array with shape [N, L, D] in the
    same order as ``regions_path``. Parquet embeddings may either be row-aligned
    with regions or keyed by ``gene_id``/another requested column.
    """

    def __init__(
        self,
        embeddings_path: str | Path,
        sites_path: str | Path = DEFAULT_SITES_PATH,
        regions_path: str | Path = DEFAULT_REGIONS_PATH,
        *,
        embedding_column: str = "emb",
        key_column: str | None = None,
        min_sites_per_tf: int = 15,
        sequence_orientation: SequenceOrientation = "strand-aware",
        tf_name_filter: Iterable[str] | None = None,
        max_regions: int | None = None,
    ) -> None:
        self.embeddings_path = Path(embeddings_path)
        self.embedding_column = embedding_column
        self.key_column = key_column
        self._embedding_array: np.ndarray | None = None
        self._embedding_rows: list[object] | None = None
        self._embedding_by_key: dict[str, object] | None = None

        super().__init__(
            sites_path=sites_path,
            regions_path=regions_path,
            min_sites_per_tf=min_sites_per_tf,
            sequence_orientation=sequence_orientation,
            tf_name_filter=tf_name_filter,
            max_regions=max_regions,
        )
        self._load_embeddings()

    def _load_embeddings(self) -> None:
        suffix = self.embeddings_path.suffix.lower()
        if suffix == ".npy":
            array = np.load(self.embeddings_path, mmap_mode="r")
            if array.ndim != 3:
                raise ValueError(
                    f"Expected .npy embeddings with shape [N, L, D], got {array.shape}"
                )
            if array.shape[0] < len(self.records):
                raise ValueError(
                    f"Embedding array has {array.shape[0]} rows but dataset has "
                    f"{len(self.records)} promoter records"
                )
            self._embedding_array = array
            return

        if suffix != ".parquet":
            raise ValueError(
                "Unsupported embeddings format. Use .npy [N, L, D] or parquet "
                "with a list-valued position-embedding column."
            )

        table = pl.read_parquet(self.embeddings_path)
        if self.embedding_column not in table.columns:
            raise ValueError(
                f"Embedding table has no column {self.embedding_column!r}. "
                f"Available columns: {table.columns}"
            )

        if self.key_column is None:
            if table.height < len(self.records):
                raise ValueError(
                    f"Embedding table has {table.height} rows but dataset has "
                    f"{len(self.records)} promoter records"
                )
            self._embedding_rows = table.get_column(self.embedding_column).to_list()
            return

        if self.key_column not in table.columns:
            raise ValueError(
                f"Embedding table has no key column {self.key_column!r}. "
                f"Available columns: {table.columns}"
            )
        self._embedding_by_key = {
            str(row[self.key_column]): row[self.embedding_column]
            for row in table.select(self.key_column, self.embedding_column).iter_rows(named=True)
        }

    def _get_x(self, idx: int, record: PromoterRecord) -> torch.Tensor:
        if self._embedding_array is not None:
            embedding = np.asarray(self._embedding_array[idx], dtype=np.float32)
        elif self._embedding_rows is not None:
            embedding = np.asarray(self._embedding_rows[idx], dtype=np.float32)
        elif self._embedding_by_key is not None:
            try:
                embedding = np.asarray(
                    self._embedding_by_key[record.gene_id], dtype=np.float32
                )
            except KeyError as exc:
                raise KeyError(
                    f"Missing embeddings for gene_id {record.gene_id!r}"
                ) from exc
        else:
            raise RuntimeError("Embeddings were not loaded")

        if embedding.ndim != 2:
            raise ValueError(
                f"Expected position embeddings [L, D] for {record.gene_id}, "
                f"got shape {embedding.shape}"
            )
        if embedding.shape[0] != len(record.sequence):
            raise ValueError(
                f"Embedding length mismatch for {record.gene_id}: "
                f"{embedding.shape[0]} embeddings vs {len(record.sequence)} bases"
            )
        return torch.from_numpy(np.asarray(embedding, dtype=np.float32))
