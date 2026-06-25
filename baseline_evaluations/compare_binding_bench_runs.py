from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


DEFAULT_PROJECT = Path(
    os.environ.get("ML4RG_PROJECT", "/s/project/ml4rg_students/2026/project15")
)
DEFAULT_DATASET = "DNA_rossi_chipexo"
DEFAULT_OUTPUT_DIR = DEFAULT_PROJECT / "working/binding_bench_reports"
DEFAULT_SCORES = ["jaccard"]


def parse_run_arg(value: str) -> tuple[str, str]:
    """Parse --run arguments of the form 'Display name=/path/to/run_dir'."""
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "--run must use the form 'Display name=/path/to/run_dir'"
        )
    display_name, run_dir = value.split("=", 1)
    display_name = display_name.strip()
    run_dir = run_dir.strip()
    if not display_name:
        raise argparse.ArgumentTypeError("Run display name must not be empty.")
    if not run_dir:
        raise argparse.ArgumentTypeError("Run directory must not be empty.")
    return display_name, run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create Binding Bench comparison plots from existing run directories. "
            "Each run directory should be the root that contains binding_bench/."
        )
    )
    parser.add_argument(
        "--run",
        action="append",
        type=parse_run_arg,
        required=True,
        metavar="NAME=RUN_DIR",
        help=(
            "Run label and Binding Bench run root. Repeat this flag for each "
            "method, e.g. --run 'STREME=/s/project/.../binding_bench_runs/foo'."
        ),
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Binding Bench dataset to plot. Default: {DEFAULT_DATASET}",
    )
    parser.add_argument(
        "--score",
        nargs="+",
        default=DEFAULT_SCORES,
        choices=["jaccard", "precision_lb", "recall_lb", "precision", "recall", "f1"],
        help="Metric(s) to plot. One output image is written per score.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for plots. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--prefix",
        default="binding_bench_comparison",
        help="Output filename prefix. Metric and extension are appended.",
    )
    parser.add_argument(
        "--extension",
        default="png",
        choices=["png", "pdf", "svg"],
        help="Plot file extension. Default: png",
    )
    parser.add_argument(
        "--binding-bench-repo",
        type=Path,
        default=None,
        help=(
            "Optional path to a binding_bench checkout. The script adds its src/ "
            "directory to sys.path before importing binding_bench."
        ),
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively in addition to saving them.",
    )
    return parser.parse_args()


def maybe_add_binding_bench_src(cli_repo: Path | None) -> None:
    candidates: list[Path] = []
    if cli_repo is not None:
        candidates.append(cli_repo / "src")

    env_repo = os.environ.get("BINDING_BENCH_REPO")
    if env_repo:
        candidates.append(Path(env_repo) / "src")

    repo_root = Path(__file__).resolve().parents[1]
    candidates.extend(
        [
            repo_root.parent / "binding_bench" / "src",
            Path.home() / "binding_bench" / "src",
        ]
    )

    for candidate in candidates:
        if candidate.is_dir():
            sys.path.insert(0, str(candidate))
            return


def validate_run_dirs(runs: dict[str, str], dataset: str) -> None:
    missing: list[str] = []
    for name, run_dir_str in runs.items():
        run_dir = Path(run_dir_str)
        expected = run_dir / "binding_bench" / "discrete" / dataset
        if not (expected / "benchmark_conf.yaml").is_file():
            missing.append(
                f"{name}: missing {expected / 'benchmark_conf.yaml'}"
            )
        if not (expected / "best_assnt_per_rank").is_dir():
            missing.append(
                f"{name}: missing {expected / 'best_assnt_per_rank'}"
            )
    if missing:
        joined = "\n".join(missing)
        raise FileNotFoundError(
            "Some runs do not look ready for compare_runs:\n" + joined
        )


def main() -> None:
    args = parse_args()
    maybe_add_binding_bench_src(args.binding_bench_repo)

    from binding_bench.report.discrete import compare_runs, report

    run_names = [name for name, _ in args.run]
    if len(run_names) != len(set(run_names)):
        raise ValueError("Duplicate --run display names are not allowed.")
    runs = dict(args.run)
    validate_run_dirs(runs, args.dataset)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    benchmark = report(runs)

    for score in args.score:
        output_path = (args.output_dir / f"{args.prefix}_{score}.{args.extension}").resolve()
        compare_runs(
            benchmark,
            datasets=args.dataset,
            score=score,
            save_path=str(output_path),
            show=args.show,
        )
        print(f"Wrote {score} comparison plot: {output_path}")


if __name__ == "__main__":
    main()
