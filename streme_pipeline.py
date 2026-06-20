from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


JASPAR_URL = (
    "https://jaspar.elixir.no/download/data/2026/CORE/"
    "JASPAR2026_CORE_fungi_non-redundant_pfms_meme.txt"
)
STAGES = ("streme", "fimo_jaspar", "tomtom", "fimo_streme")
PRINT_LOCK = threading.Lock()


def safe_print(*args, **kwargs) -> None:
    with PRINT_LOCK:
        print(*args, **kwargs, flush=True)


@dataclass(frozen=True)
class PipelineConfig:
    project: Path
    fasta_dir: Path
    result_dir: Path
    log_dir: Path
    streme: Path
    tomtom: Path
    fimo: Path
    jaspar_fungi: Path

    @classmethod
    def default(cls, project: Path | str | None = None) -> "PipelineConfig":
        project_path = Path(
            project
            or os.environ.get(
                "ML4RG_PROJECT",
                "/s/project/ml4rg_students/2026/project15",
            )
        )
        meme_bin = Path(
            os.environ.get(
                "MEME_BIN",
                "/opt/modules/i12g/anaconda/envs/meme_env/bin",
            )
        )
        result_dir = project_path / "working" / "streme_results"
        return cls(
            project=project_path,
            fasta_dir=project_path / "working" / "sequence_datasets_fastas",
            result_dir=result_dir,
            log_dir=project_path / "working" / "logs" / "streme_pipeline",
            streme=meme_bin / "streme",
            tomtom=meme_bin / "tomtom",
            fimo=meme_bin / "fimo",
            jaspar_fungi=(
                project_path
                / "working"
                / "jaspar"
                / "JASPAR2026_CORE_fungi_non-redundant_pfms_meme.txt"
            ),
        )

    def ensure_directories(self) -> None:
        for path in (
            self.result_dir / "streme",
            self.result_dir / "tomtom",
            self.result_dir / "fimo_jaspar",
            self.result_dir / "fimo_streme",
            self.result_dir / "summary_tables",
            self.log_dir,
            self.jaspar_fungi.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def ensure_jaspar(self, download: bool = True) -> None:
        if exists_nonempty(self.jaspar_fungi):
            return
        if not download:
            raise FileNotFoundError(f"Missing JASPAR database: {self.jaspar_fungi}")
        safe_print(f"Downloading JASPAR database to {self.jaspar_fungi}")
        urllib.request.urlretrieve(JASPAR_URL, self.jaspar_fungi)

    def validate(self, download_jaspar: bool = True) -> None:
        self.ensure_directories()
        self.ensure_jaspar(download=download_jaspar)
        missing = [
            path
            for path in (self.streme, self.tomtom, self.fimo)
            if not path.is_file()
        ]
        if missing:
            formatted = "\n".join(f"- {path}" for path in missing)
            raise FileNotFoundError(f"Missing MEME Suite executables:\n{formatted}")


@dataclass(frozen=True)
class PipelineParameters:
    streme_time: int = 1800
    minw: int = 6
    maxw: int = 20
    nmotifs: int = 10
    fimo_thresh: str = "1e-4"
    fimo_max_stored_scores: int = 100_000
    fimo_skip_matched_sequence: bool = False


def discover_fastas(fasta_dir: Path) -> list[Path]:
    return sorted({*fasta_dir.glob("*.fa"), *fasta_dir.glob("*.fasta")})


def fasta_name(fasta_path: Path) -> str:
    return fasta_path.name.removesuffix(".fasta").removesuffix(".fa")


def paths_for_fasta(config: PipelineConfig, fasta_path: Path) -> dict[str, Path | str]:
    name = fasta_name(fasta_path)
    streme_out = config.result_dir / "streme" / name
    tomtom_out = config.result_dir / "tomtom" / name
    fimo_jaspar_out = config.result_dir / "fimo_jaspar" / name
    fimo_streme_out = config.result_dir / "fimo_streme" / name
    return {
        "name": name,
        "streme_out": streme_out,
        "tomtom_out": tomtom_out,
        "fimo_jaspar_out": fimo_jaspar_out,
        "fimo_streme_out": fimo_streme_out,
        "streme_txt": streme_out / "streme.txt",
        "tomtom_tsv": tomtom_out / "tomtom.tsv",
        "fimo_jaspar_tsv": fimo_jaspar_out / "fimo.tsv",
        "fimo_streme_tsv": fimo_streme_out / "fimo.tsv",
    }


def exists_nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _file_signature(path: Path) -> dict[str, str | int]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _step_signature(cmd: Sequence[object], inputs: Sequence[Path]) -> dict:
    payload = {
        "command": [str(value) for value in cmd],
        "inputs": [_file_signature(path) for path in inputs],
    }
    serialized = json.dumps(payload, sort_keys=True).encode()
    payload["fingerprint"] = hashlib.sha256(serialized).hexdigest()
    return payload


def _manifest_path(expected_output: Path) -> Path:
    return expected_output.parent / ".pipeline_done.json"


def _read_manifest(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None


def _write_manifest(path: Path, signature: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(signature, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def run_command(cmd: Sequence[object], log_prefix: Path) -> None:
    log_prefix.parent.mkdir(parents=True, exist_ok=True)
    stdout_file = log_prefix.parent / f"{log_prefix.name}.out"
    stderr_file = log_prefix.parent / f"{log_prefix.name}.err"
    command = [str(value) for value in cmd]

    safe_print(f"Running: {' '.join(command)}")
    try:
        with stdout_file.open("w") as stdout, stderr_file.open("w") as stderr:
            process = subprocess.Popen(
                command,
                stdout=stdout,
                stderr=stderr,
            )
            try:
                return_code = process.wait()
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
            if return_code:
                raise subprocess.CalledProcessError(return_code, command)
    except subprocess.CalledProcessError as exc:
        stderr_tail = _tail_file(stderr_file)
        details = (
            f"\nstderr tail:\n{stderr_tail}" if stderr_tail else ""
        )
        raise RuntimeError(
            f"Command failed with exit code {exc.returncode}. "
            f"See {stderr_file}.{details}"
        ) from exc


def _tail_file(path: Path, lines: int = 25) -> str:
    try:
        return "\n".join(path.read_text(errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def run_step_if_needed(
    *,
    sample_name: str,
    label: str,
    expected_output: Path,
    cmd: Sequence[object],
    inputs: Sequence[Path],
    log_file: Path,
    force: bool = False,
    accept_legacy: bool = True,
) -> str:
    signature = _step_signature(cmd, inputs)
    manifest_path = _manifest_path(expected_output)
    manifest = _read_manifest(manifest_path)

    if exists_nonempty(expected_output) and not force:
        if manifest and manifest.get("fingerprint") == signature["fingerprint"]:
            safe_print(f"[{sample_name}] {label}: cached")
            return "cached"
        if manifest is None and accept_legacy:
            safe_print(f"[{sample_name}] {label}: existing legacy result accepted")
            return "legacy"

    safe_print(f"[{sample_name}] {label}: running")
    run_command(cmd, log_file)
    if not exists_nonempty(expected_output):
        stderr_file = log_file.parent / f"{log_file.name}.err"
        stderr_tail = _tail_file(stderr_file)
        details = (
            f"\nstderr tail:\n{stderr_tail}" if stderr_tail else ""
        )
        raise RuntimeError(
            f"{label} completed without creating expected output: {expected_output}. "
            f"See {stderr_file}.{details}"
        )
    _write_manifest(manifest_path, signature)
    return "completed"


def _commands_for_fasta(
    config: PipelineConfig,
    fasta: Path,
    parameters: PipelineParameters,
) -> dict[str, tuple[list[object], list[Path], Path, Path]]:
    paths = paths_for_fasta(config, fasta)
    name = str(paths["name"])

    streme_cmd: list[object] = [
        config.streme,
        "--dna",
        "--p",
        fasta,
        "--oc",
        paths["streme_out"],
        "--minw",
        parameters.minw,
        "--maxw",
        parameters.maxw,
        "--nmotifs",
        parameters.nmotifs,
        "--time",
        parameters.streme_time,
        "--verbosity",
        1,
    ]
    fimo_options: list[object] = [
        "--thresh",
        parameters.fimo_thresh,
        "--max-stored-scores",
        parameters.fimo_max_stored_scores,
    ]
    if parameters.fimo_skip_matched_sequence:
        raise ValueError(
            "fimo_skip_matched_sequence=True is incompatible with this pipeline: "
            "FIMO would write text to stdout instead of fimo.tsv and disable "
            "q-value calculation."
        )

    fimo_jaspar_cmd: list[object] = [
        config.fimo,
        "--oc",
        paths["fimo_jaspar_out"],
        *fimo_options,
        config.jaspar_fungi,
        fasta,
    ]
    tomtom_cmd: list[object] = [
        config.tomtom,
        "-oc",
        paths["tomtom_out"],
        "-verbosity",
        1,
        "-min-overlap",
        5,
        "-dist",
        "pearson",
        "-evalue",
        "-thresh",
        10,
        paths["streme_txt"],
        config.jaspar_fungi,
    ]
    fimo_streme_cmd: list[object] = [
        config.fimo,
        "--oc",
        paths["fimo_streme_out"],
        *fimo_options,
        paths["streme_txt"],
        fasta,
    ]

    return {
        "streme": (
            streme_cmd,
            [fasta],
            Path(paths["streme_txt"]),
            config.log_dir / f"{name}.streme",
        ),
        "fimo_jaspar": (
            fimo_jaspar_cmd,
            [config.jaspar_fungi, fasta],
            Path(paths["fimo_jaspar_tsv"]),
            config.log_dir / f"{name}.fimo_jaspar",
        ),
        "tomtom": (
            tomtom_cmd,
            [Path(paths["streme_txt"]), config.jaspar_fungi],
            Path(paths["tomtom_tsv"]),
            config.log_dir / f"{name}.tomtom",
        ),
        "fimo_streme": (
            fimo_streme_cmd,
            [Path(paths["streme_txt"]), fasta],
            Path(paths["fimo_streme_tsv"]),
            config.log_dir / f"{name}.fimo_streme",
        ),
    }


def run_stage_for_fasta(
    config: PipelineConfig,
    fasta: Path,
    stage: str,
    *,
    parameters: PipelineParameters | None = None,
    force: bool = False,
    accept_legacy: bool = True,
) -> str:
    if stage not in STAGES:
        raise ValueError(f"Unknown stage {stage!r}; choose one of {STAGES}")
    parameters = parameters or PipelineParameters()
    paths = paths_for_fasta(config, fasta)
    name = str(paths["name"])
    command, inputs, expected_output, log_file = _commands_for_fasta(
        config, fasta, parameters
    )[stage]

    missing_inputs = [path for path in inputs if not exists_nonempty(path)]
    if missing_inputs:
        formatted = ", ".join(str(path) for path in missing_inputs)
        raise FileNotFoundError(f"[{name}] {stage}: missing input(s): {formatted}")

    Path(expected_output).parent.mkdir(parents=True, exist_ok=True)
    return run_step_if_needed(
        sample_name=name,
        label=stage,
        expected_output=expected_output,
        cmd=command,
        inputs=inputs,
        log_file=log_file,
        force=force,
        accept_legacy=accept_legacy,
    )


def run_pipeline_for_fasta(
    config: PipelineConfig,
    fasta: Path,
    *,
    parameters: PipelineParameters | None = None,
    force: bool = False,
    accept_legacy: bool = True,
) -> dict[str, str]:
    results = {}
    for stage in STAGES:
        results[stage] = run_stage_for_fasta(
            config,
            fasta,
            stage,
            parameters=parameters,
            force=force,
            accept_legacy=accept_legacy,
        )
    return results


def allocated_cpus() -> int:
    slurm_candidates = (
        os.environ.get("SLURM_CPUS_PER_TASK"),
        os.environ.get("SLURM_CPUS_ON_NODE"),
    )
    for candidate in slurm_candidates:
        if candidate:
            try:
                return max(1, int(candidate))
            except ValueError:
                continue

    local_limit = os.environ.get("PIPELINE_MAX_WORKERS")
    if local_limit:
        try:
            return max(1, int(local_limit))
        except ValueError:
            pass
    return min(4, os.cpu_count() or 1)


def run_stage_for_fastas(
    config: PipelineConfig,
    fastas: Iterable[Path],
    stage: str,
    *,
    parameters: PipelineParameters | None = None,
    force: bool = False,
    accept_legacy: bool = True,
    max_workers: int | None = None,
) -> pd.DataFrame:
    fasta_list = list(fastas)
    if not fasta_list:
        return pd.DataFrame(columns=["fasta", "stage", "status", "error"])

    workers = min(len(fasta_list), max_workers or allocated_cpus())
    safe_print(
        f"Stage {stage}: {len(fasta_list)} FASTAs, max_workers={workers}"
    )
    rows: list[dict[str, str]] = []
    executor = ThreadPoolExecutor(max_workers=workers)
    futures = {}
    try:
        futures = {
            executor.submit(
                run_stage_for_fasta,
                config,
                fasta,
                stage,
                parameters=parameters,
                force=force,
                accept_legacy=accept_legacy,
            ): fasta
            for fasta in fasta_list
        }
        for index, future in enumerate(as_completed(futures), start=1):
            fasta = futures[future]
            try:
                status = future.result()
                error = ""
            except Exception as exc:
                status = "failed"
                error = str(exc)
            rows.append(
                {
                    "fasta": str(fasta),
                    "stage": stage,
                    "status": status,
                    "error": error,
                }
            )
            message = (
                f"{stage}: {index}/{len(fasta_list)} {fasta.name} -> {status}"
            )
            if error:
                message += f": {error}"
            safe_print(message)
    except BaseException:
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    return pd.DataFrame(rows)


def run_all_stages(
    config: PipelineConfig,
    fastas: Iterable[Path],
    *,
    parameters: PipelineParameters | None = None,
    force: bool = False,
    accept_legacy: bool = True,
    max_workers: int | None = None,
) -> pd.DataFrame:
    fasta_list = list(fastas)
    reports = [
        run_stage_for_fastas(
            config,
            fasta_list,
            stage,
            parameters=parameters,
            force=force,
            accept_legacy=accept_legacy,
            max_workers=max_workers,
        )
        for stage in STAGES
    ]
    return pd.concat(reports, ignore_index=True)


def build_status(config: PipelineConfig, fastas: Iterable[Path]) -> pd.DataFrame:
    rows = []
    for fasta in fastas:
        paths = paths_for_fasta(config, fasta)
        rows.append(
            {
                "name": paths["name"],
                "fasta": str(fasta),
                "streme": exists_nonempty(Path(paths["streme_txt"])),
                "tomtom": exists_nonempty(Path(paths["tomtom_tsv"])),
                "fimo_jaspar": exists_nonempty(Path(paths["fimo_jaspar_tsv"])),
                "fimo_streme": exists_nonempty(Path(paths["fimo_streme_tsv"])),
            }
        )
    return pd.DataFrame(rows)


def dataset_metadata(dataset: str) -> dict[str, str]:
    cleaned = dataset.strip("_").removesuffix("_sequence_mapper")
    tokens = [token for token in cleaned.split("_") if token]
    assembly_match = re.search(r"(gca_\d+|gcf_\d+)", cleaned, flags=re.IGNORECASE)
    return {
        "dataset": dataset,
        "genus": tokens[0].capitalize() if tokens else "",
        "species": tokens[1].lower() if len(tokens) > 1 else "",
        "species_name": (
            f"{tokens[0].capitalize()} {tokens[1].lower()}"
            if len(tokens) > 1
            else (tokens[0].capitalize() if tokens else "")
        ),
        "assembly": assembly_match.group(1).upper() if assembly_match else "",
    }


def fasta_statistics(fasta: Path) -> dict[str, int | float | str]:
    sequence_count = 0
    total_bp = 0
    min_length: int | None = None
    max_length = 0
    current_length = 0

    def finish_sequence() -> None:
        nonlocal sequence_count, total_bp, min_length, max_length, current_length
        if sequence_count == 0 and current_length == 0:
            return
        total_bp += current_length
        min_length = (
            current_length if min_length is None else min(min_length, current_length)
        )
        max_length = max(max_length, current_length)
        current_length = 0

    with fasta.open() as handle:
        for line in handle:
            if line.startswith(">"):
                if sequence_count:
                    finish_sequence()
                sequence_count += 1
            else:
                current_length += len(line.strip())
    if sequence_count:
        finish_sequence()

    mean_length = total_bp / sequence_count if sequence_count else 0.0
    return {
        "fasta": str(fasta),
        "sequence_count": sequence_count,
        "total_bp": total_bp,
        "mean_sequence_length": mean_length,
        "min_sequence_length": min_length or 0,
        "max_sequence_length": max_length,
    }


def build_dataset_statistics(fastas: Iterable[Path]) -> pd.DataFrame:
    fasta_list = list(fastas)
    rows = []
    for index, fasta in enumerate(fasta_list, start=1):
        dataset = fasta_name(fasta)
        rows.append(
            {
                **dataset_metadata(dataset),
                **fasta_statistics(fasta),
            }
        )
        if index % 100 == 0 or index == len(fasta_list):
            safe_print(f"FASTA statistics: {index}/{len(fasta_list)}")
    return pd.DataFrame(rows)


def _read_tsv_columns(path: Path) -> list[str]:
    return list(pd.read_csv(path, sep="\t", comment="#", nrows=0).columns)


def summarize_fimo_motifs(
    config: PipelineConfig,
    fastas: Iterable[Path],
    result_key: str,
    output_name: str,
    *,
    dataset_stats: pd.DataFrame | None = None,
) -> Path | None:
    fasta_list = list(fastas)
    stats = (
        dataset_stats
        if dataset_stats is not None
        else build_dataset_statistics(fasta_list)
    )
    stats_by_dataset = stats.set_index("dataset")
    rows = []

    for index, fasta in enumerate(fasta_list, start=1):
        if index == 1 or index % 100 == 0 or index == len(fasta_list):
            safe_print(
                f"{result_key} summary: {index}/{len(fasta_list)} FASTAs"
            )
        paths = paths_for_fasta(config, fasta)
        dataset = str(paths["name"])
        input_path = Path(paths[result_key])
        if not exists_nonempty(input_path):
            continue

        available = _read_tsv_columns(input_path)
        requested = [
            column
            for column in (
                "motif_id",
                "motif_alt_id",
                "sequence_name",
                "p-value",
                "q-value",
            )
            if column in available
        ]
        if "motif_id" not in requested:
            continue
        frame = pd.read_csv(
            input_path,
            sep="\t",
            comment="#",
            usecols=requested,
        )
        if frame.empty:
            continue

        group_columns = ["motif_id"]
        if "motif_alt_id" in frame:
            frame["motif_alt_id"] = frame["motif_alt_id"].fillna("")
            group_columns.append("motif_alt_id")
        aggregation: dict[str, tuple[str, str]] = {
            "hit_count": ("motif_id", "size"),
        }
        if "sequence_name" in frame:
            aggregation["sequences_with_hit"] = ("sequence_name", "nunique")
        if "p-value" in frame:
            aggregation["min_p_value"] = ("p-value", "min")
        if "q-value" in frame:
            aggregation["min_q_value"] = ("q-value", "min")

        motif_summary = frame.groupby(group_columns, dropna=False).agg(**aggregation)
        motif_summary = motif_summary.reset_index()
        dataset_row = stats_by_dataset.loc[dataset]
        motif_summary.insert(0, "dataset", dataset)
        motif_summary["sequence_count"] = int(dataset_row["sequence_count"])
        motif_summary["total_bp"] = int(dataset_row["total_bp"])
        motif_summary["hits_per_mbp"] = (
            motif_summary["hit_count"] / motif_summary["total_bp"] * 1_000_000
        )
        if "sequences_with_hit" in motif_summary:
            motif_summary["sequence_fraction"] = (
                motif_summary["sequences_with_hit"]
                / motif_summary["sequence_count"]
            )
        rows.append(motif_summary)

    if not rows:
        return None
    output_path = config.result_dir / "summary_tables" / output_name
    pd.concat(rows, ignore_index=True).to_csv(output_path, sep="\t", index=False)
    return output_path


def summarize_tomtom_matches(
    config: PipelineConfig,
    fastas: Iterable[Path],
    *,
    q_value_threshold: float = 0.05,
) -> dict[str, Path | None]:
    summary_dir = config.result_dir / "summary_tables"
    significant_frames = []

    fasta_list = list(fastas)
    for index, fasta in enumerate(fasta_list, start=1):
        if index == 1 or index % 100 == 0 or index == len(fasta_list):
            safe_print(f"TOMTOM summary: {index}/{len(fasta_list)} FASTAs")
        paths = paths_for_fasta(config, fasta)
        input_path = Path(paths["tomtom_tsv"])
        if not exists_nonempty(input_path):
            continue
        frame = pd.read_csv(input_path, sep="\t", comment="#")
        if frame.empty or "q-value" not in frame:
            continue
        frame.insert(0, "dataset", paths["name"])
        significant_frames.append(frame.loc[frame["q-value"] <= q_value_threshold])

    if not significant_frames:
        return {"significant": None, "best": None, "recurrence": None}

    significant = pd.concat(significant_frames, ignore_index=True)
    if significant.empty:
        return {"significant": None, "best": None, "recurrence": None}
    significant_path = summary_dir / "tomtom_significant.tsv"
    significant.to_csv(significant_path, sep="\t", index=False)

    sort_columns = [
        column
        for column in ("q-value", "E-value", "p-value")
        if column in significant
    ]
    best = (
        significant.sort_values(sort_columns)
        .drop_duplicates(["dataset", "Query_ID"])
        .sort_values(["dataset", "Query_ID"])
    )
    best_path = summary_dir / "tomtom_best_matches.tsv"
    best.to_csv(best_path, sep="\t", index=False)

    target_columns = ["Target_ID"]
    if "Target_alt_ID" in best:
        target_columns.append("Target_alt_ID")
    recurrence = (
        best.groupby(target_columns, dropna=False)
        .agg(
            datasets_matched=("dataset", "nunique"),
            de_novo_motifs_matched=("Query_ID", "size"),
            best_q_value=("q-value", "min"),
        )
        .reset_index()
        .sort_values(
            ["datasets_matched", "de_novo_motifs_matched", "best_q_value"],
            ascending=[False, False, True],
        )
    )
    recurrence_path = summary_dir / "tomtom_target_recurrence.tsv"
    recurrence.to_csv(recurrence_path, sep="\t", index=False)
    return {
        "significant": significant_path,
        "best": best_path,
        "recurrence": recurrence_path,
    }


def combine_result_tables(
    config: PipelineConfig,
    fastas: Iterable[Path],
    result_key: str,
    output_name: str,
    *,
    columns: Sequence[str] | None = None,
) -> Path | None:
    summary_dir = config.result_dir / "summary_tables"
    summary_dir.mkdir(parents=True, exist_ok=True)
    output_path = summary_dir / output_name
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    wrote_header = False

    with temporary.open("w") as output:
        for fasta in fastas:
            paths = paths_for_fasta(config, fasta)
            input_path = Path(paths[result_key])
            if not exists_nonempty(input_path):
                continue
            frame = pd.read_csv(
                input_path,
                sep="\t",
                comment="#",
                usecols=columns,
            )
            if frame.empty:
                continue
            frame.insert(0, "dataset", paths["name"])
            frame.to_csv(
                output,
                sep="\t",
                index=False,
                header=not wrote_header,
            )
            wrote_header = True

    if not wrote_header:
        temporary.unlink(missing_ok=True)
        return None
    temporary.replace(output_path)
    return output_path


def summarize_fimo_counts(
    config: PipelineConfig,
    fastas: Iterable[Path],
    result_key: str,
    output_name: str,
) -> Path | None:
    rows = []
    for fasta in fastas:
        paths = paths_for_fasta(config, fasta)
        input_path = Path(paths[result_key])
        if not exists_nonempty(input_path):
            continue
        counts = pd.read_csv(
            input_path,
            sep="\t",
            comment="#",
            usecols=["motif_id"],
        )["motif_id"].value_counts()
        rows.extend(
            {
                "dataset": paths["name"],
                "motif_id": motif_id,
                "hit_count": hit_count,
            }
            for motif_id, hit_count in counts.items()
        )

    if not rows:
        return None
    output_path = config.result_dir / "summary_tables" / output_name
    pd.DataFrame(rows).to_csv(output_path, sep="\t", index=False)
    return output_path


def save_summaries(
    config: PipelineConfig,
    fastas: Iterable[Path],
    *,
    combine_fimo: bool = False,
) -> dict[str, Path | None]:
    fasta_list = list(fastas)
    summary_dir = config.result_dir / "summary_tables"
    summary_dir.mkdir(parents=True, exist_ok=True)
    status_path = summary_dir / "meme_pipeline_status.csv"
    build_status(config, fasta_list).to_csv(status_path, index=False)

    outputs = {
        "status": status_path,
        "tomtom": combine_result_tables(
            config, fasta_list, "tomtom_tsv", "tomtom_all.tsv"
        ),
        "fimo_jaspar_counts": summarize_fimo_counts(
            config,
            fasta_list,
            "fimo_jaspar_tsv",
            "fimo_jaspar_counts.tsv",
        ),
        "fimo_streme_counts": summarize_fimo_counts(
            config,
            fasta_list,
            "fimo_streme_tsv",
            "fimo_streme_counts.tsv",
        ),
    }
    if combine_fimo:
        outputs["fimo_jaspar_all"] = combine_result_tables(
            config, fasta_list, "fimo_jaspar_tsv", "fimo_jaspar_all.tsv"
        )
        outputs["fimo_streme_all"] = combine_result_tables(
            config, fasta_list, "fimo_streme_tsv", "fimo_streme_all.tsv"
        )
    return outputs


def save_analysis_tables(
    config: PipelineConfig,
    fastas: Iterable[Path],
    *,
    tomtom_q_value: float = 0.05,
) -> dict[str, Path | None]:
    fasta_list = list(fastas)
    summary_dir = config.result_dir / "summary_tables"
    summary_dir.mkdir(parents=True, exist_ok=True)

    dataset_stats = build_dataset_statistics(fasta_list)
    dataset_stats_path = summary_dir / "dataset_statistics.tsv"
    dataset_stats.to_csv(dataset_stats_path, sep="\t", index=False)

    fimo_jaspar_path = summarize_fimo_motifs(
        config,
        fasta_list,
        "fimo_jaspar_tsv",
        "fimo_jaspar_motif_summary.tsv",
        dataset_stats=dataset_stats,
    )
    fimo_streme_path = summarize_fimo_motifs(
        config,
        fasta_list,
        "fimo_streme_tsv",
        "fimo_streme_motif_summary.tsv",
        dataset_stats=dataset_stats,
    )
    tomtom_paths = summarize_tomtom_matches(
        config,
        fasta_list,
        q_value_threshold=tomtom_q_value,
    )
    jaspar_recurrence_path = None
    if fimo_jaspar_path:
        jaspar_summary = pd.read_csv(fimo_jaspar_path, sep="\t")
        motif_columns = ["motif_id"]
        if "motif_alt_id" in jaspar_summary:
            motif_columns.append("motif_alt_id")
        aggregations: dict[str, tuple[str, str]] = {
            "datasets_with_hits": ("dataset", "nunique"),
            "total_hits": ("hit_count", "sum"),
            "median_hits_per_mbp": ("hits_per_mbp", "median"),
            "max_hits_per_mbp": ("hits_per_mbp", "max"),
        }
        if "sequence_fraction" in jaspar_summary:
            aggregations["median_sequence_fraction"] = (
                "sequence_fraction",
                "median",
            )
        jaspar_recurrence = (
            jaspar_summary.groupby(motif_columns, dropna=False)
            .agg(**aggregations)
            .reset_index()
            .sort_values(
                ["datasets_with_hits", "median_hits_per_mbp"],
                ascending=[False, False],
            )
        )
        jaspar_recurrence_path = summary_dir / "fimo_jaspar_recurrence.tsv"
        jaspar_recurrence.to_csv(
            jaspar_recurrence_path,
            sep="\t",
            index=False,
        )

    annotated_streme_path = None
    annotated_streme_recurrence_path = None
    if fimo_streme_path and tomtom_paths["best"]:
        streme_summary = pd.read_csv(fimo_streme_path, sep="\t")
        best_matches = pd.read_csv(tomtom_paths["best"], sep="\t")
        annotation_columns = [
            column
            for column in (
                "dataset",
                "Query_ID",
                "Target_ID",
                "Target_alt_ID",
                "q-value",
                "Query_consensus",
                "Target_consensus",
            )
            if column in best_matches
        ]
        annotated_streme = streme_summary.merge(
            best_matches[annotation_columns],
            left_on=["dataset", "motif_id"],
            right_on=["dataset", "Query_ID"],
            how="left",
        )
        annotated_streme_path = summary_dir / "fimo_streme_annotated.tsv"
        annotated_streme.to_csv(annotated_streme_path, sep="\t", index=False)

        matched = annotated_streme.dropna(subset=["Target_ID"])
        if not matched.empty:
            target_columns = ["Target_ID"]
            if "Target_alt_ID" in matched:
                target_columns.append("Target_alt_ID")
            annotated_recurrence = (
                matched.groupby(target_columns, dropna=False)
                .agg(
                    datasets_with_hits=("dataset", "nunique"),
                    streme_motifs=("motif_id", "size"),
                    total_hits=("hit_count", "sum"),
                    median_hits_per_mbp=("hits_per_mbp", "median"),
                    best_tomtom_q_value=("q-value", "min"),
                )
                .reset_index()
                .sort_values(
                    ["datasets_with_hits", "median_hits_per_mbp"],
                    ascending=[False, False],
                )
            )
            annotated_streme_recurrence_path = (
                summary_dir / "fimo_streme_annotated_recurrence.tsv"
            )
            annotated_recurrence.to_csv(
                annotated_streme_recurrence_path,
                sep="\t",
                index=False,
            )

    overview = build_status(config, fasta_list).merge(
        dataset_stats.drop(columns="fasta"),
        left_on="name",
        right_on="dataset",
        how="left",
    )
    if tomtom_paths["best"]:
        best = pd.read_csv(tomtom_paths["best"], sep="\t")
        tomtom_counts = (
            best.groupby("dataset")
            .agg(
                matched_streme_motifs=("Query_ID", "nunique"),
                matched_jaspar_targets=("Target_ID", "nunique"),
            )
            .reset_index()
        )
        overview = overview.merge(tomtom_counts, on="dataset", how="left")

    for path, prefix in (
        (fimo_jaspar_path, "jaspar"),
        (fimo_streme_path, "streme"),
    ):
        if path:
            motif_summary = pd.read_csv(path, sep="\t")
            totals = (
                motif_summary.groupby("dataset")
                .agg(
                    **{
                        f"{prefix}_fimo_hits": ("hit_count", "sum"),
                        f"{prefix}_motifs_with_hits": ("motif_id", "nunique"),
                    }
                )
                .reset_index()
            )
            overview = overview.merge(totals, on="dataset", how="left")

    numeric_columns = [
        "matched_streme_motifs",
        "matched_jaspar_targets",
        "jaspar_fimo_hits",
        "jaspar_motifs_with_hits",
        "streme_fimo_hits",
        "streme_motifs_with_hits",
    ]
    for column in numeric_columns:
        if column in overview:
            overview[column] = overview[column].fillna(0).astype(int)
    for prefix in ("jaspar", "streme"):
        hits_column = f"{prefix}_fimo_hits"
        if hits_column in overview:
            overview[f"{prefix}_fimo_hits_per_mbp"] = (
                overview[hits_column] / overview["total_bp"] * 1_000_000
            ).fillna(0.0)

    overview_path = summary_dir / "dataset_overview.tsv"
    overview.to_csv(overview_path, sep="\t", index=False)
    return {
        "dataset_statistics": dataset_stats_path,
        "dataset_overview": overview_path,
        "fimo_jaspar_motifs": fimo_jaspar_path,
        "fimo_jaspar_recurrence": jaspar_recurrence_path,
        "fimo_streme_motifs": fimo_streme_path,
        "fimo_streme_annotated": annotated_streme_path,
        "fimo_streme_annotated_recurrence": annotated_streme_recurrence_path,
        "tomtom_significant": tomtom_paths["significant"],
        "tomtom_best": tomtom_paths["best"],
        "tomtom_recurrence": tomtom_paths["recurrence"],
    }


def _sequence_metadata(sequence_name: str) -> dict[str, str]:
    fields = str(sequence_name).split("|")
    metadata = {"sequence_id": fields[0]}
    for field in fields[1:]:
        key, separator, value = field.partition("=")
        if separator:
            metadata[key] = value
    required = {"chr", "start", "end", "strand"}
    missing = required - metadata.keys()
    if missing:
        raise ValueError(
            f"Sequence header lacks {sorted(missing)}: {sequence_name!r}"
        )
    return metadata


def export_fimo_for_binding_bench(
    config: PipelineConfig,
    fasta: Path,
    *,
    result_key: str = "fimo_streme_tsv",
    output_path: Path | None = None,
    sequence_is_strand_oriented: bool = True,
) -> Path:
    """Convert relative FIMO hits to BindingBench genomic point predictions.

    The FASTA headers created by ``Convert_parquet_to_fasta.ipynb`` contain the
    genomic region coordinates. When those sequences are oriented by the region
    strand, minus-strand coordinates and motif strands must be inverted.
    """
    paths = paths_for_fasta(config, fasta)
    input_path = Path(paths[result_key])
    if not exists_nonempty(input_path):
        raise FileNotFoundError(f"Missing FIMO result: {input_path}")

    frame = pd.read_csv(input_path, sep="\t", comment="#")
    required = {"motif_id", "sequence_name", "start", "stop", "strand", "p-value"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"FIMO result lacks required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"FIMO result contains no predictions: {input_path}")

    predictions = []
    for row in frame.itertuples(index=False, name=None):
        values = dict(zip(frame.columns, row))
        metadata = _sequence_metadata(values["sequence_name"])
        region_start = int(metadata["start"])
        region_end = int(metadata["end"])
        relative_start = int(values["start"])
        relative_stop = int(values["stop"])
        region_strand = metadata["strand"]
        motif_strand = values["strand"]

        if sequence_is_strand_oriented and region_strand == "-":
            genomic_start = region_end - relative_stop
            genomic_end = region_end - relative_start + 1
            genomic_strand = {"+": "-", "-": "+"}.get(motif_strand, ".")
        else:
            genomic_start = region_start + relative_start - 1
            genomic_end = region_start + relative_stop
            genomic_strand = motif_strand if motif_strand in {"+", "-"} else "."

        center = genomic_start + (genomic_end - genomic_start - 1) // 2
        p_value = float(values["p-value"])
        score = -math.log10(max(p_value, float.fromhex("0x1.0p-1022")))
        predictions.append(
            {
                "chrom": metadata["chr"],
                "start": center,
                "end": center + 1,
                "feature_idx": str(values["motif_id"]),
                "score": score,
                "strand": genomic_strand,
            }
        )

    output_dir = config.result_dir / "binding_bench"
    output_dir.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        output_path = output_dir / f"{paths['name']}_{result_key}.tsv"

    prediction_frame = (
        pd.DataFrame(predictions)
        .sort_values(["feature_idx", "score"], ascending=[True, False])
        .drop_duplicates(["chrom", "start", "end", "feature_idx"])
    )
    prediction_frame.to_csv(output_path, sep="\t", index=False)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the STREME/MEME pipeline.")
    parser.add_argument("--project", type=Path)
    parser.add_argument(
        "--result-tag",
        help=(
            "Store results and logs in a tagged subdirectory, for example "
            "'nmotifs_100', without overwriting the default run."
        ),
    )
    parser.add_argument("--fasta", type=Path)
    parser.add_argument("--array-index", type=int)
    parser.add_argument(
        "--stage",
        choices=("all", *STAGES),
        default="all",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Create analysis tables for all available FASTA results.",
    )
    parser.add_argument("--tomtom-q-value", type=float, default=0.05)
    parser.add_argument(
        "--strict-cache",
        action="store_true",
        help="Recompute existing results that have no matching cache manifest.",
    )
    parser.add_argument("--streme-time", type=int, default=1800)
    parser.add_argument("--minw", type=int, default=6)
    parser.add_argument("--maxw", type=int, default=20)
    parser.add_argument("--nmotifs", type=int, default=10)
    parser.add_argument("--fimo-thresh", default="1e-4")
    parser.add_argument("--fimo-max-stored-scores", type=int, default=100_000)
    parser.add_argument(
        "--skip-matched-sequence",
        action="store_true",
        help=(
            "Unsupported with file-based output because FIMO writes to stdout "
            "and disables q-values."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = PipelineConfig.default(args.project)
    if args.result_tag:
        if Path(args.result_tag).name != args.result_tag or args.result_tag in {".", ".."}:
            raise ValueError("--result-tag must be a single directory name.")
        config = replace(
            config,
            result_dir=config.result_dir / args.result_tag,
            log_dir=config.log_dir / args.result_tag,
        )
    if args.summarize:
        config.ensure_directories()
    else:
        config.validate()
    fastas = discover_fastas(config.fasta_dir)
    if not fastas:
        raise FileNotFoundError(f"No .fa or .fasta files found in {config.fasta_dir}")

    if args.summarize:
        outputs = save_analysis_tables(
            config,
            fastas,
            tomtom_q_value=args.tomtom_q_value,
        )
        safe_print(
            json.dumps(
                {
                    key: str(path) if path else None
                    for key, path in outputs.items()
                },
                indent=2,
            )
        )
        return

    if args.fasta and args.array_index is not None:
        raise ValueError("Use either --fasta or --array-index, not both.")
    if args.fasta:
        fasta = args.fasta
    elif args.array_index is not None:
        try:
            fasta = fastas[args.array_index]
        except IndexError as exc:
            raise IndexError(
                f"Array index {args.array_index} is outside 0..{len(fastas) - 1}"
            ) from exc
    else:
        raise ValueError("Specify --fasta or --array-index.")

    parameters = PipelineParameters(
        streme_time=args.streme_time,
        minw=args.minw,
        maxw=args.maxw,
        nmotifs=args.nmotifs,
        fimo_thresh=args.fimo_thresh,
        fimo_max_stored_scores=args.fimo_max_stored_scores,
        fimo_skip_matched_sequence=args.skip_matched_sequence,
    )
    kwargs = {
        "parameters": parameters,
        "force": args.force,
        "accept_legacy": not args.strict_cache,
    }
    if args.stage == "all":
        result = run_pipeline_for_fasta(config, fasta, **kwargs)
    else:
        result = {args.stage: run_stage_for_fasta(config, fasta, args.stage, **kwargs)}
    safe_print(json.dumps({"fasta": str(fasta), "result": result}, indent=2))


if __name__ == "__main__":
    main()
