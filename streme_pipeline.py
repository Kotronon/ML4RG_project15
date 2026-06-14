from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
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
    fimo_skip_matched_sequence: bool = True


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
    with stdout_file.open("w") as stdout, stderr_file.open("w") as stderr:
        subprocess.run(
            command,
            stdout=stdout,
            stderr=stderr,
            check=True,
        )


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
        raise RuntimeError(
            f"{label} completed without creating expected output: {expected_output}"
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
        fimo_options.append("--skip-matched-sequence")

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
    with ThreadPoolExecutor(max_workers=workers) as executor:
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
            safe_print(
                f"{stage}: {index}/{len(fasta_list)} {fasta.name} -> {status}"
            )
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the STREME/MEME pipeline.")
    parser.add_argument("--project", type=Path)
    parser.add_argument("--fasta", type=Path)
    parser.add_argument("--array-index", type=int)
    parser.add_argument(
        "--stage",
        choices=("all", *STAGES),
        default="all",
    )
    parser.add_argument("--force", action="store_true")
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
    parser.add_argument("--keep-matched-sequence", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = PipelineConfig.default(args.project)
    config.validate()
    fastas = discover_fastas(config.fasta_dir)

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
        fimo_skip_matched_sequence=not args.keep_matched_sequence,
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
