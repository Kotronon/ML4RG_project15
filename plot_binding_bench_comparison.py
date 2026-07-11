from __future__ import annotations

import argparse
import html
from pathlib import Path
from typing import Any

import polars as pl
import yaml


DEFAULT_METRICS = ("jaccard", "precision_lb", "recall_lb", "f1")
PALETTE = (
    "#4c78a8",
    "#f58518",
    "#54a24b",
    "#e45756",
    "#72b7b2",
    "#b279a2",
    "#ff9da6",
    "#9d755d",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create simple HTML/SVG plots for Binding Bench run comparisons."
    )
    parser.add_argument("--runs-yaml", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated metrics to plot.",
    )
    return parser.parse_args()


def read_runs(path: Path) -> dict[str, Path]:
    payload = yaml.safe_load(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping of display name to run dir in {path}")
    runs: dict[str, Path] = {}
    for name, entry in payload.items():
        if isinstance(entry, dict):
            value = entry.get("output_dir") or entry.get("benchmark_results")
        else:
            value = entry
        if value is None:
            raise ValueError(f"Run entry {name!r} has no path")
        runs[str(name)] = Path(str(value))
    return runs


def benchmark_conf(run_dir: Path, dataset: str) -> dict[str, Any]:
    path = run_dir / "binding_bench" / "discrete" / dataset / "benchmark_conf.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"Missing benchmark config: {path}")
    return yaml.safe_load(path.read_text()) or {}


def result_path(conf: dict[str, Any], section: str, metric: str | None = None) -> Path | None:
    results = conf.get("results") or {}
    if section == "raw":
        value = results.get("raw")
        return Path(value) if value else None
    if section == "best_assnt":
        value = (results.get("best_assnt") or {}).get(metric)
        return Path(value) if value else None
    raise ValueError(section)


def per_rank_path(conf: dict[str, Any], metric: str) -> Path | None:
    per_rank = results = (conf.get("results") or {}).get("best_assnt_per_rank") or {}
    if not per_rank:
        return None
    for _checksum, metric_map in per_rank.items():
        value = (metric_map or {}).get(metric)
        if value:
            return Path(value)
    return None


def load_best_summary(
    runs: dict[str, Path],
    dataset: str,
    metrics: list[str],
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for display_name, run_dir in runs.items():
        conf = benchmark_conf(run_dir, dataset)
        for metric in metrics:
            path = result_path(conf, "best_assnt", metric)
            if path is None or not path.is_file():
                continue
            df = pl.read_parquet(path)
            if metric not in df.columns:
                continue
            values = df.select(pl.col(metric).cast(pl.Float64).fill_null(0.0))[metric]
            rows.append(
                {
                    "display_name": display_name,
                    "metric": metric,
                    "mean": float(values.mean()) if len(values) else 0.0,
                    "sum": float(values.sum()) if len(values) else 0.0,
                    "n_targets": int(df["name"].n_unique()) if "name" in df.columns else len(df),
                    "path": str(path),
                }
            )
    return pl.DataFrame(rows)


def load_per_rank(
    runs: dict[str, Path],
    dataset: str,
    metric: str,
) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for display_name, run_dir in runs.items():
        conf = benchmark_conf(run_dir, dataset)
        path = per_rank_path(conf, metric)
        if path is None or not path.is_file():
            continue
        df = pl.read_parquet(path)
        if metric not in df.columns or "rank_t" not in df.columns:
            continue
        frames.append(
            df.group_by("rank_t")
            .agg(pl.col(metric).cast(pl.Float64).fill_null(0.0).sum().alias("score"))
            .with_columns(display_name=pl.lit(display_name))
            .sort("rank_t")
        )
    return pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()


def svg_bar_chart(summary: pl.DataFrame, metric: str) -> str:
    data = (
        summary.filter(pl.col("metric") == metric)
        .sort("mean", descending=True)
        .to_dicts()
    )
    if not data:
        return f"<p>No best-assignment data for {html.escape(metric)}.</p>"

    width = 900
    left = 210
    right = 90
    row_h = 34
    top = 30
    height = top + row_h * len(data) + 30
    max_value = max(float(row["mean"]) for row in data) or 1.0
    plot_w = width - left - right

    chunks = [f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    chunks.append(f'<text x="{left}" y="18" font-size="14" font-weight="700">{html.escape(metric)} mean</text>')
    for idx, row in enumerate(data):
        y = top + idx * row_h
        value = float(row["mean"])
        bar_w = plot_w * value / max_value
        color = PALETTE[idx % len(PALETTE)]
        label = html.escape(str(row["display_name"]))
        chunks.append(f'<text x="8" y="{y + 21}" font-size="12">{label}</text>')
        chunks.append(f'<rect x="{left}" y="{y + 6}" width="{bar_w:.2f}" height="20" fill="{color}" />')
        chunks.append(f'<text x="{left + bar_w + 6}" y="{y + 21}" font-size="12">{value:.4g}</text>')
    chunks.append("</svg>")
    return "\n".join(chunks)


def svg_line_chart(df: pl.DataFrame, metric: str) -> str:
    if df.is_empty():
        return f"<p>No per-rank data for {html.escape(metric)}. Run Binding Bench with --ba_by_feature_rank to enable this plot.</p>"

    width = 900
    height = 380
    left = 70
    right = 220
    top = 25
    bottom = 55
    plot_w = width - left - right
    plot_h = height - top - bottom

    max_x = int(df["rank_t"].max()) or 1
    max_y = float(df["score"].max()) or 1.0
    names = df["display_name"].unique().sort().to_list()

    def xy(rank_t: float, score: float) -> tuple[float, float]:
        x = left + plot_w * rank_t / max_x
        y = top + plot_h * (1.0 - score / max_y)
        return x, y

    chunks = [f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">']
    chunks.append(f'<text x="{left}" y="16" font-size="14" font-weight="700">{html.escape(metric)} over feature rank</text>')
    chunks.append(f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333" />')
    chunks.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333" />')
    chunks.append(f'<text x="{left}" y="{height - 15}" font-size="12">rank_t</text>')
    chunks.append(f'<text x="8" y="{top + 12}" font-size="12">sum</text>')
    chunks.append(f'<text x="{left + plot_w - 25}" y="{height - 15}" font-size="12">{max_x}</text>')
    chunks.append(f'<text x="25" y="{top + 4}" font-size="12">{max_y:.3g}</text>')

    for idx, name in enumerate(names):
        sub = df.filter(pl.col("display_name") == name).sort("rank_t")
        points = []
        for row in sub.iter_rows(named=True):
            x, y = xy(float(row["rank_t"]), float(row["score"]))
            points.append(f"{x:.2f},{y:.2f}")
        color = PALETTE[idx % len(PALETTE)]
        if points:
            chunks.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2" />')
        legend_y = top + 18 + idx * 22
        chunks.append(f'<rect x="{left + plot_w + 25}" y="{legend_y - 11}" width="12" height="12" fill="{color}" />')
        chunks.append(f'<text x="{left + plot_w + 43}" y="{legend_y}" font-size="12">{html.escape(str(name))}</text>')
    chunks.append("</svg>")
    return "\n".join(chunks)


def table_html(summary: pl.DataFrame) -> str:
    if summary.is_empty():
        return "<p>No summary data found.</p>"
    rows = summary.sort(["metric", "mean"], descending=[False, True]).to_dicts()
    out = [
        "<table>",
        "<thead><tr><th>Metric</th><th>Run</th><th>Mean</th><th>Sum</th><th>N targets</th></tr></thead>",
        "<tbody>",
    ]
    for row in rows:
        out.append(
            "<tr>"
            f"<td>{html.escape(str(row['metric']))}</td>"
            f"<td>{html.escape(str(row['display_name']))}</td>"
            f"<td>{float(row['mean']):.6g}</td>"
            f"<td>{float(row['sum']):.6g}</td>"
            f"<td>{int(row['n_targets'])}</td>"
            "</tr>"
        )
    out.extend(["</tbody>", "</table>"])
    return "\n".join(out)


def main() -> None:
    args = parse_args()
    metrics = [metric.strip() for metric in args.metrics.split(",") if metric.strip()]
    runs = read_runs(args.runs_yaml)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summary = load_best_summary(runs, args.dataset, metrics)
    summary_path = args.out_dir / f"{args.dataset}_best_assignment_summary.tsv"
    summary.write_csv(summary_path, separator="\t")

    parts = [
        "<!doctype html>",
        "<meta charset='utf-8'>",
        "<style>body{font-family:-apple-system,BlinkMacSystemFont,Segoe UI,sans-serif;margin:24px;}"
        "table{border-collapse:collapse;margin:16px 0;}td,th{border:1px solid #ddd;padding:6px 8px;}"
        "th{background:#f5f5f5;text-align:left;}section{margin-bottom:36px;}svg{max-width:100%;height:auto;}</style>",
        f"<h1>Binding Bench Comparison: {html.escape(args.dataset)}</h1>",
        f"<p>Runs config: <code>{html.escape(str(args.runs_yaml))}</code></p>",
        f"<p>Summary TSV: <code>{html.escape(str(summary_path))}</code></p>",
        "<h2>Best Assignment Summary</h2>",
        table_html(summary),
    ]

    for metric in metrics:
        parts.append(f"<section><h2>{html.escape(metric)}</h2>")
        parts.append(svg_bar_chart(summary, metric))
        parts.append(svg_line_chart(load_per_rank(runs, args.dataset, metric), metric))
        parts.append("</section>")

    out_html = args.out_dir / f"{args.dataset}_comparison.html"
    out_html.write_text("\n".join(parts))
    print(f"Wrote {out_html}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
