from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset, get_worker_info


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
LabelSmoothingMode = Literal["hard", "hard-dilate", "linear", "gaussian"]


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
    model_start: int
    model_end: int
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
        trim_terminal_atg: bool = True,
        label_smoothing_mode: LabelSmoothingMode = "hard",
        label_smoothing_radius_bp: int = 0,
        label_smoothing_sigma_bp: float | None = None,
    ) -> None:
        if sequence_orientation not in {"genomic", "strand-aware"}:
            raise ValueError(f"Unsupported sequence_orientation: {sequence_orientation}")
        if label_smoothing_mode not in {"hard", "hard-dilate", "linear", "gaussian"}:
            raise ValueError(f"Unsupported label_smoothing_mode: {label_smoothing_mode}")
        if label_smoothing_radius_bp < 0:
            raise ValueError("label_smoothing_radius_bp must be non-negative")
        if label_smoothing_sigma_bp is not None and label_smoothing_sigma_bp <= 0:
            raise ValueError("label_smoothing_sigma_bp must be positive")

        self.sites_path = Path(sites_path)
        self.regions_path = Path(regions_path)
        self.sequence_orientation = sequence_orientation
        self.trim_terminal_atg = trim_terminal_atg
        self.label_smoothing_mode = label_smoothing_mode
        self.label_smoothing_radius_bp = label_smoothing_radius_bp
        self.label_smoothing_sigma_bp = label_smoothing_sigma_bp

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
        hard_labels = self._hard_dense_labels(record)
        labels = self._smooth_dense_labels(record, hard_labels.copy())
        mask = self._position_mask(record)
        return {
            "x": x,
            "y": torch.tensor(labels, dtype=torch.float32),
            "hard_y": torch.tensor(hard_labels, dtype=torch.float32),
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
        coordinate_positions = sum(self._coordinate_length(record) for record in self.records)
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
            "trim_terminal_atg": self.trim_terminal_atg,
            "label_smoothing_mode": self.label_smoothing_mode,
            "label_smoothing_radius_bp": self.label_smoothing_radius_bp,
            "label_smoothing_sigma_bp": self.label_smoothing_sigma_bp,
            "total_positions": total_positions,
            "coordinate_positions": coordinate_positions,
            "positive_tf_positions": positive_tf_positions,
        }

    def genomic_to_offset(self, region: dict[str, object], genomic_pos: int) -> int:
        model_start, model_end = self._model_coordinate_span(region)
        strand = str(region["strand"])

        if not (model_start <= genomic_pos < model_end):
            raise ValueError(
                f"Position {genomic_pos} is outside model span "
                f"{model_start}-{model_end}"
            )

        if self.sequence_orientation == "strand-aware" and strand == "-":
            return model_end - genomic_pos - 1
        return genomic_pos - model_start

    def offset_to_genomic(self, region: dict[str, object], offset: int) -> int:
        model_start, model_end = self._model_coordinate_span(region)
        strand = str(region["strand"])
        length = model_end - model_start

        if not (0 <= offset < length):
            raise ValueError(f"Offset {offset} is outside region length {length}")

        if self.sequence_orientation == "strand-aware" and strand == "-":
            return model_end - offset - 1
        return model_start + offset

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
            sequence = self._model_sequence(region)
            region_start = int(region["start"])
            region_end = int(region["end"])
            model_start, model_end = self._model_coordinate_span(region)
            if len(sequence) != model_end - model_start:
                raise ValueError(
                    "Promoter model sequence length does not match model span for "
                    f"row {region_idx}: len(seq)={len(sequence)}, "
                    f"span={model_end - model_start}, "
                    f"raw_start={region_start}, raw_end={region_end}"
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
                    model_start=model_start,
                    model_end=model_end,
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
        model_start, model_end = self._model_coordinate_span(region)
        intervals: list[tuple[int, int, int]] = []

        for site in sites_by_chrom.get(chrom, []):
            site_start = int(site["start"])
            site_end = int(site["end"])
            if site_end <= model_start:
                continue
            if site_start >= model_end:
                break

            overlap_start = max(model_start, site_start)
            overlap_end = min(model_end, site_end)
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
        model_start, model_end = self._model_coordinate_span(region)
        strand = str(region["strand"])

        if self.sequence_orientation == "strand-aware" and strand == "-":
            lo = model_end - genomic_end
            hi = model_end - genomic_start
        else:
            lo = genomic_start - model_start
            hi = genomic_end - model_start

        length = model_end - model_start
        lo = max(0, min(length, lo))
        hi = max(0, min(length, hi))
        return lo, hi

    def _hard_dense_labels(self, record: PromoterRecord) -> np.ndarray:
        labels = np.zeros((len(self.tf_names), len(record.sequence)), dtype=np.float32)
        for tf_idx, lo, hi in record.label_intervals:
            labels[tf_idx, lo:hi] = 1.0
        return labels

    def _dense_labels(self, record: PromoterRecord) -> np.ndarray:
        hard_labels = self._hard_dense_labels(record)
        return self._smooth_dense_labels(record, hard_labels)

    def _smooth_dense_labels(
        self,
        record: PromoterRecord,
        labels: np.ndarray,
    ) -> np.ndarray:
        radius = self.label_smoothing_radius_bp
        if self.label_smoothing_mode == "hard" or radius <= 0:
            return labels

        length = labels.shape[1]
        for tf_idx, lo, hi in record.label_intervals:
            if hi <= lo:
                continue
            smooth_lo = max(0, lo - radius)
            smooth_hi = min(length, hi + radius)
            if smooth_hi <= smooth_lo:
                continue

            if self.label_smoothing_mode == "hard-dilate":
                labels[tf_idx, smooth_lo:smooth_hi] = 1.0
                continue

            positions = np.arange(smooth_lo, smooth_hi)
            distances = np.zeros_like(positions, dtype=np.float32)
            before = positions < lo
            after = positions >= hi
            distances[before] = lo - positions[before]
            distances[after] = positions[after] - hi + 1

            if self.label_smoothing_mode == "linear":
                values = np.maximum(0.0, 1.0 - distances / float(radius + 1))
            elif self.label_smoothing_mode == "gaussian":
                sigma = (
                    self.label_smoothing_sigma_bp
                    if self.label_smoothing_sigma_bp is not None
                    else max(radius / 2.0, 1.0)
                )
                values = np.exp(-0.5 * (distances / float(sigma)) ** 2)
            else:
                raise ValueError(
                    f"Unsupported label_smoothing_mode: {self.label_smoothing_mode}"
                )

            labels[tf_idx, smooth_lo:smooth_hi] = np.maximum(
                labels[tf_idx, smooth_lo:smooth_hi],
                values.astype(np.float32),
            )
        return labels

    @staticmethod
    def _coordinate_length(record: PromoterRecord) -> int:
        return min(len(record.sequence), max(0, record.model_end - record.model_start))

    def _model_sequence(self, region: dict[str, object]) -> str:
        sequence = str(region["seq"])
        if self.trim_terminal_atg:
            return sequence[:-3]
        return sequence

    def _model_coordinate_span(self, region: dict[str, object]) -> tuple[int, int]:
        raw_start = int(region["start"])
        raw_end_inclusive = int(region["end"])
        raw_end_exclusive = raw_end_inclusive + 1
        strand = str(region["strand"])

        if not self.trim_terminal_atg:
            return raw_start, raw_end_exclusive

        if self.sequence_orientation == "strand-aware" and strand == "-":
            return raw_start + 3, raw_end_exclusive
        return raw_start, raw_end_exclusive - 3

    @staticmethod
    def _position_mask(record: PromoterRecord) -> np.ndarray:
        valid_bases = {"A", "C", "G", "T", "a", "c", "g", "t"}
        base_mask = np.fromiter(
            (base in valid_bases for base in record.sequence),
            dtype=bool,
            count=len(record.sequence),
        )
        coordinate_mask = np.zeros(len(record.sequence), dtype=bool)
        coordinate_mask[: BindingBenchPromoterBaseDataset._coordinate_length(record)] = True
        return base_mask & coordinate_mask


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
        trim_terminal_atg: bool = True,
        label_smoothing_mode: LabelSmoothingMode = "hard",
        label_smoothing_radius_bp: int = 0,
        label_smoothing_sigma_bp: float | None = None,
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
            trim_terminal_atg=trim_terminal_atg,
            label_smoothing_mode=label_smoothing_mode,
            label_smoothing_radius_bp=label_smoothing_radius_bp,
            label_smoothing_sigma_bp=label_smoothing_sigma_bp,
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
            embedding = self._embedding_array[idx]
        elif self._embedding_rows is not None:
            embedding = np.asarray(self._embedding_rows[idx])
        elif self._embedding_by_key is not None:
            try:
                embedding = np.asarray(
                    self._embedding_by_key[record.gene_id]
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
        if self.trim_terminal_atg and embedding.shape[0] == len(record.sequence) + 3:
            embedding = embedding[:-3]
        if embedding.shape[0] != len(record.sequence):
            raise ValueError(
                f"Embedding length mismatch for {record.gene_id}: "
                f"{embedding.shape[0]} embeddings vs {len(record.sequence)} bases"
            )
        return torch.from_numpy(np.asarray(embedding, dtype=np.float32))


class BindingBenchSampledPromoterWindowDataset(Dataset):
    """Sample TF-conditioned promoter windows while keeping base-wise labels.

    Each item is one ``(DNA window, TF)`` pair. Positive windows contain an
    annotated site for the sampled TF at a randomized offset, hard negatives
    contain a site for another sampled TF but not the target TF, and easy
    negatives avoid sampled-TF sites in the crop.
    """

    def __init__(
        self,
        base_dataset: BindingBenchPromoterBaseDataset,
        *,
        promoter_indices: Iterable[int],
        tf_indices: Iterable[int],
        window_size: int = 200,
        samples_per_epoch: int = 50_000,
        positive_fraction: float = 0.5,
        hard_negative_fraction: float = 0.25,
        margin_bp: int = 30,
        negative_exclusion_bp: int = 10,
        seed: int = 42,
        max_sampling_attempts: int = 100,
    ) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        if samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be positive")
        if not 0.0 <= positive_fraction <= 1.0:
            raise ValueError("positive_fraction must be in [0, 1]")
        if not 0.0 <= hard_negative_fraction <= 1.0:
            raise ValueError("hard_negative_fraction must be in [0, 1]")
        if positive_fraction + hard_negative_fraction > 1.0:
            raise ValueError("positive and hard-negative fractions must sum to <= 1")
        if margin_bp < 0:
            raise ValueError("margin_bp must be non-negative")
        if margin_bp * 2 >= window_size:
            raise ValueError("margin_bp must leave room inside the window")
        if negative_exclusion_bp < 0:
            raise ValueError("negative_exclusion_bp must be non-negative")
        if max_sampling_attempts <= 0:
            raise ValueError("max_sampling_attempts must be positive")

        self.base_dataset = base_dataset
        self.promoter_indices = [int(idx) for idx in promoter_indices]
        self.tf_indices = sorted({int(idx) for idx in tf_indices})
        if not self.promoter_indices:
            raise ValueError("At least one promoter index is required")
        if not self.tf_indices:
            raise ValueError("At least one TF index is required")
        if window_size > min(len(base_dataset.records[idx].sequence) for idx in self.promoter_indices):
            raise ValueError("window_size must fit inside every sampled promoter")

        self.window_size = int(window_size)
        self.samples_per_epoch = int(samples_per_epoch)
        self.positive_fraction = float(positive_fraction)
        self.hard_negative_fraction = float(hard_negative_fraction)
        self.margin_bp = int(margin_bp)
        self.negative_exclusion_bp = int(negative_exclusion_bp)
        self.seed = int(seed)
        self.max_sampling_attempts = int(max_sampling_attempts)
        self.tf_set = set(self.tf_indices)
        self._rng: np.random.Generator | None = None
        self._rng_seed: int | None = None

        self.positive_anchors_by_tf: dict[int, list[tuple[int, int, int]]] = {
            tf_idx: [] for tf_idx in self.tf_indices
        }
        self.flat_positive_anchors: list[tuple[int, int, int, int]] = []
        self._build_positive_anchors()
        self.positive_tfs = [
            tf_idx
            for tf_idx, anchors in self.positive_anchors_by_tf.items()
            if anchors
        ]
        if self.positive_fraction > 0 and not self.positive_tfs:
            raise ValueError("Positive sampling requested but no positive anchors exist")
        if self.hard_negative_fraction > 0 and len(self.flat_positive_anchors) == 0:
            raise ValueError("Hard-negative sampling requested but no anchors exist")

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, idx: int) -> dict[str, object]:
        del idx
        rng = self._worker_rng()
        kind_value = float(rng.random())
        if kind_value < self.positive_fraction:
            return self._sample_positive(rng)
        if kind_value < self.positive_fraction + self.hard_negative_fraction:
            return self._sample_hard_negative(rng)
        return self._sample_easy_negative(rng)

    def summary(self) -> dict[str, object]:
        return {
            "n_base_promoters": len(self.base_dataset),
            "n_sampled_promoters": len(self.promoter_indices),
            "n_sampled_tfs": len(self.tf_indices),
            "n_positive_tfs": len(self.positive_tfs),
            "n_positive_anchors": len(self.flat_positive_anchors),
            "window_size": self.window_size,
            "samples_per_epoch": self.samples_per_epoch,
            "positive_fraction": self.positive_fraction,
            "hard_negative_fraction": self.hard_negative_fraction,
            "easy_negative_fraction": (
                1.0 - self.positive_fraction - self.hard_negative_fraction
            ),
            "margin_bp": self.margin_bp,
            "negative_exclusion_bp": self.negative_exclusion_bp,
        }

    def _worker_rng(self) -> np.random.Generator:
        worker = get_worker_info()
        if worker is None:
            seed = self.seed
        else:
            seed = int(torch.initial_seed() % (2**32))
        if self._rng is None or self._rng_seed != seed:
            self._rng = np.random.default_rng(seed)
            self._rng_seed = seed
        return self._rng

    def _build_positive_anchors(self) -> None:
        promoter_set = set(self.promoter_indices)
        for promoter_idx in self.promoter_indices:
            record = self.base_dataset.records[promoter_idx]
            for tf_idx, lo, hi in record.label_intervals:
                tf_idx = int(tf_idx)
                if tf_idx not in self.tf_set or hi <= lo:
                    continue
                anchor = (promoter_idx, int(lo), int(hi))
                self.positive_anchors_by_tf[tf_idx].append(anchor)
                self.flat_positive_anchors.append((tf_idx, *anchor))

        # Avoid holding stale indices if the input iterable had duplicates.
        self.promoter_indices = sorted(promoter_set)

    def _sample_positive(self, rng: np.random.Generator) -> dict[str, object]:
        for _ in range(self.max_sampling_attempts):
            tf_idx = int(rng.choice(self.positive_tfs))
            promoter_idx, lo, hi = self.positive_anchors_by_tf[tf_idx][
                int(rng.integers(len(self.positive_anchors_by_tf[tf_idx])))
            ]
            start = self._window_start_around_interval(rng, promoter_idx, lo, hi)
            if start is not None:
                return self._make_item(promoter_idx, tf_idx, start, "positive")
        return self._fallback_positive(rng)

    def _sample_hard_negative(self, rng: np.random.Generator) -> dict[str, object]:
        if len(self.tf_indices) < 2:
            return self._sample_easy_negative(rng)

        for _ in range(self.max_sampling_attempts):
            other_tf, promoter_idx, lo, hi = self.flat_positive_anchors[
                int(rng.integers(len(self.flat_positive_anchors)))
            ]
            candidate_tfs = [tf_idx for tf_idx in self.tf_indices if tf_idx != other_tf]
            target_tf = int(rng.choice(candidate_tfs))
            start = self._window_start_around_interval(rng, promoter_idx, lo, hi)
            if start is None:
                continue
            end = start + self.window_size
            if self._has_tf_overlap(
                self.base_dataset.records[promoter_idx],
                target_tf,
                start,
                end,
                buffer_bp=self.negative_exclusion_bp,
            ):
                continue
            return self._make_item(promoter_idx, target_tf, start, "hard_negative")

        return self._sample_easy_negative(rng)

    def _sample_easy_negative(self, rng: np.random.Generator) -> dict[str, object]:
        for _ in range(self.max_sampling_attempts):
            promoter_idx = int(rng.choice(self.promoter_indices))
            record = self.base_dataset.records[promoter_idx]
            max_start = len(record.sequence) - self.window_size
            if max_start < 0:
                continue
            start = int(rng.integers(max_start + 1))
            end = start + self.window_size
            if self._has_any_tf_overlap(
                record,
                start,
                end,
                buffer_bp=self.negative_exclusion_bp,
            ):
                continue
            target_tf = int(rng.choice(self.tf_indices))
            return self._make_item(promoter_idx, target_tf, start, "easy_negative")

        for _ in range(self.max_sampling_attempts):
            promoter_idx = int(rng.choice(self.promoter_indices))
            record = self.base_dataset.records[promoter_idx]
            max_start = len(record.sequence) - self.window_size
            if max_start < 0:
                continue
            start = int(rng.integers(max_start + 1))
            end = start + self.window_size
            target_tf = int(rng.choice(self.tf_indices))
            if not self._has_tf_overlap(
                record,
                target_tf,
                start,
                end,
                buffer_bp=self.negative_exclusion_bp,
            ):
                return self._make_item(promoter_idx, target_tf, start, "easy_negative")

        return self._fallback_negative(rng)

    def _fallback_positive(self, rng: np.random.Generator) -> dict[str, object]:
        tf_idx, promoter_idx, lo, hi = self.flat_positive_anchors[
            int(rng.integers(len(self.flat_positive_anchors)))
        ]
        record = self.base_dataset.records[promoter_idx]
        base = int(rng.integers(lo, hi))
        start = min(max(0, base - self.window_size // 2), len(record.sequence) - self.window_size)
        return self._make_item(promoter_idx, tf_idx, start, "positive_fallback")

    def _fallback_negative(self, rng: np.random.Generator) -> dict[str, object]:
        promoter_idx = int(rng.choice(self.promoter_indices))
        record = self.base_dataset.records[promoter_idx]
        max_start = len(record.sequence) - self.window_size
        start = int(rng.integers(max_start + 1))
        target_tf = int(rng.choice(self.tf_indices))
        return self._make_item(promoter_idx, target_tf, start, "negative_fallback")

    def _window_start_around_interval(
        self,
        rng: np.random.Generator,
        promoter_idx: int,
        lo: int,
        hi: int,
    ) -> int | None:
        record = self.base_dataset.records[promoter_idx]
        if len(record.sequence) < self.window_size:
            return None

        base = int(rng.integers(lo, hi))
        offset = int(rng.integers(self.margin_bp, self.window_size - self.margin_bp))
        start = base - offset
        if 0 <= start <= len(record.sequence) - self.window_size:
            return start

        valid_lo = max(0, base - (self.window_size - self.margin_bp - 1))
        valid_hi = min(base - self.margin_bp, len(record.sequence) - self.window_size)
        if valid_hi >= valid_lo:
            return int(rng.integers(valid_lo, valid_hi + 1))
        return None

    def _make_item(
        self,
        promoter_idx: int,
        tf_idx: int,
        window_start: int,
        sampling_kind: str,
    ) -> dict[str, object]:
        record = self.base_dataset.records[promoter_idx]
        window_end = window_start + self.window_size
        x_full = self.base_dataset._get_x(promoter_idx, record)
        if x_full.ndim != 2:
            raise ValueError(f"Expected 2D promoter input, got {x_full.shape}")
        if x_full.shape[0] == len(record.sequence):
            x = x_full[window_start:window_end, :].clone()
        else:
            x = x_full[:, window_start:window_end].clone()

        labels, hard_labels = self._window_labels(record, tf_idx, window_start, window_end)
        mask = np.ones(self.window_size, dtype=bool)
        return {
            "x": x,
            "y": torch.tensor(labels, dtype=torch.float32),
            "hard_y": torch.tensor(hard_labels, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.bool),
            "tf_idx": torch.tensor(tf_idx, dtype=torch.long),
            "meta": {
                "record_idx": promoter_idx,
                "chrom": record.chrom,
                "start": record.start,
                "end": record.end,
                "strand": record.strand,
                "gene_id": record.gene_id,
                "window_start": window_start,
                "window_end": window_end,
                "tf_idx": tf_idx,
                "tf_name": self.base_dataset.tf_names[tf_idx],
                "sampling_kind": sampling_kind,
                "has_positive_label": bool(hard_labels.any()),
            },
        }

    def _window_labels(
        self,
        record: PromoterRecord,
        tf_idx: int,
        window_start: int,
        window_end: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        hard = np.zeros((1, self.window_size), dtype=np.float32)
        labels = np.zeros((1, self.window_size), dtype=np.float32)
        radius = self.base_dataset.label_smoothing_radius_bp
        mode = self.base_dataset.label_smoothing_mode
        sigma_bp = self.base_dataset.label_smoothing_sigma_bp

        for interval_tf, lo, hi in record.label_intervals:
            if int(interval_tf) != int(tf_idx):
                continue
            overlap_lo = max(int(lo), window_start)
            overlap_hi = min(int(hi), window_end)
            if overlap_hi <= overlap_lo:
                continue

            rel_lo = overlap_lo - window_start
            rel_hi = overlap_hi - window_start
            hard[0, rel_lo:rel_hi] = 1.0
            labels[0, rel_lo:rel_hi] = 1.0

            if mode == "hard" or radius <= 0:
                continue

            smooth_lo = max(0, rel_lo - radius)
            smooth_hi = min(self.window_size, rel_hi + radius)
            if smooth_hi <= smooth_lo:
                continue

            if mode == "hard-dilate":
                labels[0, smooth_lo:smooth_hi] = 1.0
                continue

            positions = np.arange(smooth_lo, smooth_hi)
            distances = np.zeros_like(positions, dtype=np.float32)
            before = positions < rel_lo
            after = positions >= rel_hi
            distances[before] = rel_lo - positions[before]
            distances[after] = positions[after] - rel_hi + 1

            if mode == "linear":
                values = np.maximum(0.0, 1.0 - distances / float(radius + 1))
            elif mode == "gaussian":
                sigma = sigma_bp if sigma_bp is not None else max(radius / 2.0, 1.0)
                values = np.exp(-0.5 * (distances / float(sigma)) ** 2)
            else:
                raise ValueError(f"Unsupported label_smoothing_mode: {mode}")

            labels[0, smooth_lo:smooth_hi] = np.maximum(
                labels[0, smooth_lo:smooth_hi],
                values.astype(np.float32),
            )

        return labels, hard

    def _has_tf_overlap(
        self,
        record: PromoterRecord,
        tf_idx: int,
        window_start: int,
        window_end: int,
        *,
        buffer_bp: int,
    ) -> bool:
        query_start = window_start - buffer_bp
        query_end = window_end + buffer_bp
        for interval_tf, lo, hi in record.label_intervals:
            if int(interval_tf) != int(tf_idx):
                continue
            if int(hi) > query_start and int(lo) < query_end:
                return True
        return False

    def _has_any_tf_overlap(
        self,
        record: PromoterRecord,
        window_start: int,
        window_end: int,
        *,
        buffer_bp: int,
    ) -> bool:
        query_start = window_start - buffer_bp
        query_end = window_end + buffer_bp
        for interval_tf, lo, hi in record.label_intervals:
            if int(interval_tf) not in self.tf_set:
                continue
            if int(hi) > query_start and int(lo) < query_end:
                return True
        return False
