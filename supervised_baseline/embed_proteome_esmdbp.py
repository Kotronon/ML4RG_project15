from __future__ import annotations

import argparse
import gzip
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable

import numpy as np
import polars as pl
import torch
from Bio import SeqIO


DEFAULT_PROJECT = Path("/s/project/ml4rg_students/2026/project15")
DEFAULT_MODEL_DIR = DEFAULT_PROJECT / "working/protein_models/ESM-DBP"
DEFAULT_OUTPUT = DEFAULT_PROJECT / "working/protein_embeddings/scer_esmdbp_tf_emb.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run ESM-DBP on a protein FASTA and convert the generated .fea files "
            "to the project protein-embedding parquet schema."
        )
    )
    parser.add_argument("fasta", type=Path, help="Protein FASTA, e.g. SGD orf_trans_all.fasta")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output parquet with columns protein_id, orf, gene, emb.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=DEFAULT_MODEL_DIR,
        help="Directory containing the downloaded ESM-DBP model files.",
    )
    parser.add_argument(
        "--runner-script",
        type=Path,
        help=(
            "Path to the ESM-DBP runner, e.g. TransBind/Predict_new_TF/ESM_DBP.py "
            "or ESM-DBP/code/prediction.py. If omitted, common locations are tried."
        ),
    )
    parser.add_argument(
        "--esm-dbp-code-dir",
        type=Path,
        help="Optional checkout of the ESM-DBP or TransBind code used to find the runner.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Directory for temporary filtered FASTA and ESM-DBP .fea outputs.",
    )
    parser.add_argument("--device", default="cuda", help="Device argument passed to ESM-DBP.")
    parser.add_argument(
        "--python-bin",
        default=None,
        help="Python executable used to run the ESM-DBP runner. Defaults to this interpreter.",
    )
    parser.add_argument(
        "--keep-names",
        type=Path,
        help=(
            "Optional file with protein IDs/gene names to embed. Supports one name "
            "per line, a JSON list, or a JSON mapping whose values are used."
        ),
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Do not call ESM-DBP; only convert existing .fea files in --work-dir.",
    )
    parser.add_argument(
        "--pooling",
        choices=("mean", "first"),
        default="mean",
        help="How to pool per-residue feature matrices into one vector per protein.",
    )
    parser.add_argument(
        "--runner-input-mode",
        choices=("auto", "fasta", "single-sequence"),
        default="auto",
        help=(
            "How to pass proteins to the ESM-DBP runner. The standalone "
            "code/prediction.py script expects one headerless sequence file per "
            "protein, while TransBind-style runners may accept a multi-record FASTA."
        ),
    )
    parser.add_argument(
        "--overwrite-work-dir",
        action="store_true",
        help="Delete --work-dir before running ESM-DBP.",
    )
    return parser.parse_args()


def parse_sgd_gene(description: str) -> str | None:
    parts = description.split()
    if len(parts) >= 2 and re.fullmatch(r"[A-Z0-9-]+", parts[1]):
        return parts[1]
    match = re.search(r"gene=([^\s,;]+)", description)
    if match:
        return match.group(1)
    return None


def clean_sequence(sequence: str) -> str:
    return str(sequence).rstrip("*").replace("*", "")


def assert_probably_fasta(path: Path) -> None:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        prefix = handle.read(512).lstrip()
    if prefix.startswith(b"<!DOCTYPE") or prefix.startswith(b"<html") or prefix.startswith(b"<HTML"):
        raise ValueError(
            f"{path} looks like an HTML page, not a FASTA file. Download the actual "
            "orf_trans_all.fasta file from the SGD directory listing instead of the "
            "directory page."
        )


def read_keep_names(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    text = path.read_text().strip()
    if not text:
        raise ValueError(f"Keep-name file is empty: {path}")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        if isinstance(payload, dict):
            return {str(value).upper() for value in payload.values()}
        if isinstance(payload, list):
            return {str(value).upper() for value in payload}
        raise ValueError(f"Unsupported JSON keep-name payload in {path}")
    return {line.strip().upper() for line in text.splitlines() if line.strip()}


def read_fasta_records(path: Path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as handle:
            return list(SeqIO.parse(handle, "fasta-pearson"))
    return list(SeqIO.parse(path, "fasta-pearson"))


def filter_records(records, keep_names: set[str] | None):
    if keep_names is None:
        return records

    kept = []
    missing = set(keep_names)
    for record in records:
        protein_id = str(record.id)
        gene = parse_sgd_gene(str(record.description))
        keys = {protein_id.upper()}
        if gene:
            keys.add(gene.upper())
        if keys & keep_names:
            kept.append(record)
            missing -= keys

    if not kept:
        raise ValueError("No FASTA records matched --keep-names")
    if missing:
        preview = ", ".join(sorted(missing)[:20])
        suffix = "" if len(missing) <= 20 else f", ... ({len(missing)} total)"
        print(f"Warning: keep names not found in FASTA: {preview}{suffix}")
    return kept


def write_filtered_fasta(records, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            protein_id = str(record.id)
            gene = parse_sgd_gene(str(record.description))
            gene_part = f" {gene}" if gene else ""
            handle.write(f">{protein_id}{gene_part}\n")
            sequence = clean_sequence(str(record.seq))
            for start in range(0, len(sequence), 80):
                handle.write(sequence[start : start + 80] + "\n")


def safe_file_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return stem or "protein"


def write_plain_sequence(record, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(clean_sequence(str(record.seq)) + "\n")


def runner_candidates(model_dir: Path, code_dir: Path | None) -> list[Path]:
    roots = [model_dir]
    if code_dir is not None:
        roots.insert(0, code_dir)
    candidates: list[Path] = []
    for root in roots:
        candidates.extend(
            [
                root / "ESM_DBP.py",
                root / "Predict_new_TF/ESM_DBP.py",
                root / "prediction.py",
                root / "prediciton.py",
                root / "code/prediction.py",
                root / "code/prediciton.py",
            ]
        )
    return candidates


def find_runner(args: argparse.Namespace) -> Path:
    if args.runner_script is not None:
        if not args.runner_script.exists():
            raise FileNotFoundError(f"ESM-DBP runner script not found: {args.runner_script}")
        return args.runner_script

    for candidate in runner_candidates(args.model_dir, args.esm_dbp_code_dir):
        if candidate.exists():
            return candidate

    searched = "\n".join(str(path) for path in runner_candidates(args.model_dir, args.esm_dbp_code_dir))
    raise FileNotFoundError(
        "Could not find an ESM-DBP runner script. Pass --runner-script or "
        "--esm-dbp-code-dir. Searched:\n"
        f"{searched}"
    )


def run_esm_dbp(
    *,
    runner: Path,
    model_dir: Path,
    fasta: Path,
    records,
    output_dir: Path,
    device: str,
    python_bin: str,
    runner_input_mode: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    mode = runner_input_mode
    if mode == "auto":
        mode = "single-sequence" if runner.name.lower() in {"prediction.py", "prediciton.py"} else "fasta"

    if mode == "fasta":
        command = [
            python_bin,
            str(runner),
            str(model_dir),
            str(fasta),
            str(output_dir),
            device,
        ]
        print("Running ESM-DBP:")
        print(" ".join(command))
        subprocess.run(command, check=True, cwd=runner.parent)
        return

    single_dir = output_dir / "_single_sequence_inputs"
    single_dir.mkdir(parents=True, exist_ok=True)
    print(f"Running ESM-DBP one protein at a time for {len(records)} proteins")
    for idx, record in enumerate(records, start=1):
        protein_id = str(record.id)
        sequence_path = single_dir / safe_file_stem(protein_id)
        write_plain_sequence(record, sequence_path)
        command = [
            python_bin,
            str(runner),
            str(model_dir),
            str(sequence_path),
            str(output_dir),
            device,
        ]
        print(f"[{idx}/{len(records)}] {protein_id}")
        subprocess.run(command, check=True, cwd=runner.parent)


def _as_float_array(value) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype == object:
        if arr.shape == ():
            arr = np.asarray(arr.item())
        else:
            arr = np.asarray(arr.tolist())
    return arr.astype(np.float32, copy=False)


def load_feature_file(path: Path) -> np.ndarray:
    errors: list[str] = []

    try:
        loaded = np.load(path, allow_pickle=True)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            keys = loaded.files
            if not keys:
                raise ValueError("empty npz")
            return _as_float_array(loaded[keys[0]])
        return _as_float_array(loaded)
    except Exception as exc:  # noqa: BLE001 - keep trying supported formats.
        errors.append(f"np.load: {exc}")

    try:
        try:
            loaded = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            loaded = torch.load(path, map_location="cpu")
        if isinstance(loaded, dict):
            for key in ("emb", "embedding", "features", "feature", "representations"):
                if key in loaded:
                    return _as_float_array(loaded[key])
            if len(loaded) == 1:
                return _as_float_array(next(iter(loaded.values())))
            raise ValueError(f"torch dict keys {sorted(loaded)} did not include a feature key")
        return _as_float_array(loaded)
    except Exception as exc:  # noqa: BLE001 - keep trying supported formats.
        errors.append(f"torch.load: {exc}")

    for delimiter in (None, ",", "\t"):
        try:
            return np.loadtxt(path, dtype=np.float32, delimiter=delimiter)
        except Exception as exc:  # noqa: BLE001 - keep trying supported formats.
            label = "whitespace" if delimiter is None else repr(delimiter)
            errors.append(f"np.loadtxt({label}): {exc}")

    raise ValueError(f"Could not parse feature file {path}. Tried:\n" + "\n".join(errors))


def pool_feature(feature: np.ndarray, pooling: str) -> np.ndarray:
    arr = np.asarray(feature, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        if pooling == "first":
            return arr[0]
        return arr.mean(axis=0)
    if arr.ndim == 3 and arr.shape[0] == 1:
        return pool_feature(arr[0], pooling)
    raise ValueError(f"Expected feature vector or matrix, got shape {arr.shape}")


def safe_stems(record_id: str, gene: str | None) -> list[str]:
    raw = [record_id]
    if gene:
        raw.append(gene)
    stems = []
    for value in raw:
        stems.append(value)
        stems.append(re.sub(r"[^A-Za-z0-9_.-]+", "_", value))
    return list(dict.fromkeys(stems))


def find_feature_for_record(feature_dir: Path, record_id: str, gene: str | None) -> Path | None:
    for stem in safe_stems(record_id, gene):
        for suffix in (".fea", ".npy", ".npz", ".pt", ".pth", ".txt", ".csv"):
            candidate = feature_dir / f"{stem}{suffix}"
            if candidate.exists():
                return candidate
    return None


def feature_files(feature_dir: Path) -> list[Path]:
    suffixes = {".fea", ".npy", ".npz", ".pt", ".pth", ".txt", ".csv"}
    return sorted(path for path in feature_dir.rglob("*") if path.is_file() and path.suffix in suffixes)


def collect_embeddings(records, feature_dir: Path, pooling: str) -> list[np.ndarray]:
    vectors: list[np.ndarray] = []
    missing: list[str] = []

    for record in records:
        protein_id = str(record.id)
        gene = parse_sgd_gene(str(record.description))
        feature_path = find_feature_for_record(feature_dir, protein_id, gene)
        if feature_path is None:
            missing.append(protein_id)
            continue
        vectors.append(pool_feature(load_feature_file(feature_path), pooling))

    if not missing:
        return vectors

    files = feature_files(feature_dir)
    if len(files) == 1:
        feature = load_feature_file(files[0])
        arr = np.asarray(feature, dtype=np.float32)
        if arr.ndim == 2 and arr.shape[0] == len(records):
            return [arr[idx] for idx in range(arr.shape[0])]
        if arr.ndim == 3 and arr.shape[0] == len(records):
            return [pool_feature(arr[idx], pooling) for idx in range(arr.shape[0])]

    preview = ", ".join(missing[:20])
    suffix = "" if len(missing) <= 20 else f", ... ({len(missing)} total)"
    raise FileNotFoundError(
        "Could not find per-protein ESM-DBP feature files for: "
        f"{preview}{suffix}. Found {len(files)} candidate feature files in {feature_dir}. "
        "If the runner writes one combined feature file, ensure it has shape [N, D] "
        "or [N, L, D] in FASTA record order."
    )


def validate_vectors(vectors: Iterable[np.ndarray]) -> list[np.ndarray]:
    out = [np.asarray(vector, dtype=np.float32).reshape(-1) for vector in vectors]
    if not out:
        raise ValueError("No embeddings were collected")
    width = out[0].shape[0]
    bad = [idx for idx, vector in enumerate(out) if vector.shape[0] != width]
    if bad:
        preview = ", ".join(str(idx) for idx in bad[:20])
        raise ValueError(f"Embedding dimensions are inconsistent; bad row indices: {preview}")
    return out


def main() -> None:
    args = parse_args()
    python_bin = args.python_bin or shutil.which("python") or "python"

    assert_probably_fasta(args.fasta)
    records = read_fasta_records(args.fasta)
    if not records:
        raise ValueError(f"No FASTA records found in {args.fasta}")
    records = filter_records(records, read_keep_names(args.keep_names))

    work_dir = args.work_dir or (args.output.parent / "esmdbp_features")
    if args.overwrite_work_dir and work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    filtered_fasta = work_dir / "input.fasta"
    write_filtered_fasta(records, filtered_fasta)
    feature_dir = work_dir / "features"

    print(f"FASTA:       {args.fasta}")
    print(f"Proteins:    {len(records)}")
    print(f"Model dir:   {args.model_dir}")
    print(f"Work dir:    {work_dir}")
    print(f"Feature dir: {feature_dir}")
    print(f"Output:      {args.output}")
    print(f"Pooling:     {args.pooling}")

    if not args.skip_run:
        runner = find_runner(args)
        print(f"Runner:      {runner}")
        with tempfile.TemporaryDirectory(dir=work_dir, prefix="esmdbp_run_") as tmp:
            tmp_feature_dir = Path(tmp)
            run_esm_dbp(
                runner=runner,
                model_dir=args.model_dir,
                fasta=filtered_fasta,
                records=records,
                output_dir=tmp_feature_dir,
                device=args.device,
                python_bin=python_bin,
                runner_input_mode=args.runner_input_mode,
            )
            if feature_dir.exists():
                shutil.rmtree(feature_dir)
            shutil.copytree(tmp_feature_dir, feature_dir)
    elif not feature_dir.exists():
        raise FileNotFoundError(f"--skip-run was set but feature directory does not exist: {feature_dir}")

    vectors = validate_vectors(collect_embeddings(records, feature_dir, args.pooling))
    protein_ids = [str(record.id) for record in records]
    genes = [parse_sgd_gene(str(record.description)) for record in records]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "protein_id": protein_ids,
            "orf": protein_ids,
            "gene": genes,
            "emb": [vector.astype(float).tolist() for vector in vectors],
        }
    ).write_parquet(args.output)

    print(
        "Wrote",
        args.output,
        {
            "n_proteins": len(vectors),
            "embedding_dim": int(vectors[0].shape[0]),
        },
    )


if __name__ == "__main__":
    main()
