from __future__ import annotations

import argparse
import gzip
import json
import re
from pathlib import Path

import polars as pl
import torch
from Bio import SeqIO
from transformers import AutoTokenizer, EsmModel


DEFAULT_MODEL = "facebook/esm2_t33_650M_UR50D"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Mean-pool ESM-2 protein embeddings from a FASTA file."
    )
    parser.add_argument("fasta", type=Path, help="Protein FASTA, e.g. SGD orf_trans_all.fasta")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scer_esm2_emb.parquet"),
        help="Output parquet with columns protein_id, orf, gene, emb.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--keep-names",
        type=Path,
        help=(
            "Optional file with protein IDs/gene names to embed. Supports one name "
            "per line, a JSON list, or a JSON mapping whose values are used."
        ),
    )
    parser.add_argument(
        "--exclude-special-tokens",
        action="store_true",
        help="Mean-pool amino-acid tokens only instead of including CLS/EOS.",
    )
    return parser.parse_args()


def get_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


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


@torch.no_grad()
def embed_batch(
    sequences: list[str],
    *,
    tokenizer: AutoTokenizer,
    model: EsmModel,
    device: torch.device,
    max_length: int,
    exclude_special_tokens: bool,
) -> torch.Tensor:
    enc = tokenizer(
        sequences,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    out = model(**enc).last_hidden_state
    mask = enc.attention_mask.unsqueeze(-1).float()
    if exclude_special_tokens:
        token_ids = enc.input_ids
        special = (
            (token_ids == tokenizer.cls_token_id)
            | (token_ids == tokenizer.eos_token_id)
            | (token_ids == tokenizer.pad_token_id)
        )
        mask = mask * (~special).unsqueeze(-1).float()
    pooled = (out * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    return pooled.cpu()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.max_length <= 0:
        raise ValueError("--max-length must be positive")

    device = get_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = EsmModel.from_pretrained(args.model).to(device).eval()

    assert_probably_fasta(args.fasta)
    records = read_fasta_records(args.fasta)
    if not records:
        raise ValueError(f"No FASTA records found in {args.fasta}")

    keep_names = read_keep_names(args.keep_names)
    if keep_names is not None:
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
            raise ValueError(f"No FASTA records matched --keep-names {args.keep_names}")
        if missing:
            preview = ", ".join(sorted(missing)[:20])
            suffix = "" if len(missing) <= 20 else f", ... ({len(missing)} total)"
            print(f"Warning: keep names not found in FASTA: {preview}{suffix}")
        records = kept

    protein_ids = [str(record.id) for record in records]
    genes = [parse_sgd_gene(str(record.description)) for record in records]
    sequences = [clean_sequence(str(record.seq)) for record in records]

    print(f"Model:      {args.model}")
    print(f"Device:     {device}")
    print(f"FASTA:      {args.fasta}")
    print(f"Proteins:   {len(sequences)}")
    if args.keep_names is not None:
        print(f"Keep names: {args.keep_names}")
    print(f"Output:     {args.output}")

    embeddings = []
    for start in range(0, len(sequences), args.batch_size):
        batch = sequences[start : start + args.batch_size]
        embeddings.append(
            embed_batch(
                batch,
                tokenizer=tokenizer,
                model=model,
                device=device,
                max_length=args.max_length,
                exclude_special_tokens=args.exclude_special_tokens,
            )
        )
        print(f"Embedded {min(start + args.batch_size, len(sequences))}/{len(sequences)}")

    matrix = torch.cat(embeddings, dim=0).numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {
            "protein_id": protein_ids,
            "orf": protein_ids,
            "gene": genes,
            "emb": matrix.tolist(),
        }
    ).write_parquet(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
