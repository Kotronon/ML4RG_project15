from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any


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


def _as_pandas(frame: Any):
    if hasattr(frame, "to_pandas"):
        return frame.to_pandas()
    return frame


def _nearest_rank_rows(topk_df, n_targets: int):
    selected = []
    for display_name, group in topk_df.groupby("display_name", sort=False):
        ranks = sorted(int(rank) for rank in group["rank_t"].dropna().unique())
        if not ranks:
            continue
        nearest = min(ranks, key=lambda rank: (abs(rank - n_targets), rank > n_targets))
        selected.append(group[group["rank_t"] == nearest].copy())
        print(
            f"Using rank_t={nearest} for {display_name!r} "
            f"(nearest available to n_targets={n_targets})"
        )
    if not selected:
        return topk_df.iloc[0:0].copy()
    return __import__("pandas").concat(selected, ignore_index=True)


def fallback_compare_runs(benchmark: Any, dataset: str, score: str, save_path: str) -> None:
    """Render a portable Matplotlib report when Binding Bench's plotnine layout fails."""
    import matplotlib.pyplot as plt

    data = benchmark.get(datasets=dataset, metrics=[score])
    full_metrics = _as_pandas(data[score])
    topk_metrics = _as_pandas(benchmark.get_per_rank(score, datasets=dataset))

    if full_metrics.empty or topk_metrics.empty:
        raise ValueError(f"No data available for fallback plot: {dataset=} {score=}")

    n_targets = int(full_metrics["name"].nunique())
    selected_topk = _nearest_rank_rows(topk_metrics, n_targets)
    max_rank = max(1, 2 * n_targets)
    topk_visible = topk_metrics[topk_metrics["rank_t"] <= max_rank]

    score_label = {
        "jaccard": "Jaccard Sim.",
        "precision_lb": "Precision",
        "recall_lb": "Recall",
    }.get(score, score)

    display_names = sorted(topk_metrics["display_name"].dropna().unique().tolist())
    palette = plt.get_cmap("Set2")
    colors = {
        name: palette(index % palette.N)
        for index, name in enumerate(display_names)
    }

    mean_score_bar = (
        selected_topk.groupby("display_name", as_index=False)[score]
        .sum()
        .assign(**{score: lambda df: df[score] / n_targets})
    )
    mean_score_bar = mean_score_bar.sort_values(score)
    mean_score_bar_overall = (
        full_metrics.groupby("display_name", as_index=False)[score]
        .mean()
        .sort_values(score)
    )

    tf_ranked_f_ranked = selected_topk.copy()
    tf_ranked_f_ranked["score_rank"] = (
        tf_ranked_f_ranked.groupby("display_name")[score]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    tf_ranked_all_features = full_metrics.copy()
    tf_ranked_all_features["score_rank"] = (
        tf_ranked_all_features.groupby("display_name")[score]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    grid = fig.add_gridspec(2, 3)
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    axes.extend([fig.add_subplot(grid[1, 0]), fig.add_subplot(grid[1, 1])])

    rank_sums = (
        topk_visible.groupby(["rank_t", "display_name"], as_index=False)[score]
        .sum()
        .sort_values("rank_t")
    )
    for display_name, group in rank_sums.groupby("display_name", sort=False):
        axes[0].plot(group["rank_t"], group[score], label=display_name, color=colors[display_name])
    axes[0].axvline(n_targets, color="black", linestyle="--", linewidth=1, alpha=.5)
    axes[0].set(title="Best assignment over feature rank", xlabel="Feature Rank", ylabel=f"{score_label} (Sum)")

    for axis, frame, title in (
        (axes[1], mean_score_bar, "Best assignment at nearest N-factors"),
        (axes[2], mean_score_bar_overall, "Best assignment overall"),
    ):
        axis.bar(frame["display_name"], frame[score], color=[colors[name] for name in frame["display_name"]])
        axis.set(title=title, xlabel="Setup", ylabel=f"{score_label} (Mean)")
        axis.tick_params(axis="x", rotation=45)

    for axis, frame, title in (
        (axes[3], tf_ranked_f_ranked, f"{score_label} per TF at nearest N-factors"),
        (axes[4], tf_ranked_all_features, f"{score_label} per TF overall"),
    ):
        for display_name, group in frame.groupby("display_name", sort=False):
            axis.plot(group["score_rank"], group[score], label=display_name, color=colors[display_name])
        axis.set_xscale("log")
        axis.set(title=title, xlabel="TF Rank", ylabel=score_label)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, title="Setup", loc="center right")
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


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
        try:
            compare_runs(
                benchmark,
                datasets=args.dataset,
                score=score,
                save_path=str(output_path),
                show=args.show,
            )
        except (IndexError, TypeError) as exc:
            print(
                f"Binding Bench plot failed for {score} ({exc}); "
                "writing fallback plot with nearest available rank_t."
            )
            fallback_compare_runs(
                benchmark,
                dataset=args.dataset,
                score=score,
                save_path=str(output_path),
            )
        print(f"Wrote {score} comparison plot: {output_path}")


if __name__ == "__main__":
    main()
