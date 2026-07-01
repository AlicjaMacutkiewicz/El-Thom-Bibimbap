"""Plot a like-for-like real-flight comparison before and after conditioning."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

METHODS = {
    "Pre-conditioning GRU--RK4 + physics": {
        "color": "#4C78A8",
        "position": "gru_rk4_physics_position_z_window_rmse_m",
        "acceleration": "gru_rk4_physics_acceleration_z_window_rmse",
    },
    "Conditioned generalized GRU--RK4 + physics": {
        "color": "#E45756",
        "position": "gru_plus_rk4_position_z_window_rmse_m",
        "acceleration": "gru_plus_rk4_acceleration_z_window_rmse",
    },
    "Last acceleration": {
        "color": "#F2A541",
        "position": "last_acceleration_position_z_window_rmse_m",
        "acceleration": "last_acceleration_acceleration_z_window_rmse",
    },
}

LEGEND_STYLE = {
    "frameon": False,
    "fontsize": 12,
    "handlelength": 1.8,
    "handleheight": 1.1,
    "columnspacing": 1.6,
    "labelspacing": 0.8,
}


def legend_columns(item_count: int, max_columns: int = 4) -> int:
    return min(max_columns, max(1, item_count))


def parse_args() -> argparse.Namespace:
    repo = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--before-dir",
        type=Path,
        default=repo / "source_data" / "far_out_26_data" / "real_flight_z_eval_matched_all_models_nonretrained",
    )
    parser.add_argument(
        "--after-dir",
        type=Path,
        default=repo  / "source_data" / "far_out_26_data" / "real_flight_z_eval_conditioned_generalized_axis_mapped",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo  / "source_data" / "far_out_26_data" / "real_flight_comparison_visuals",
    )
    parser.add_argument("--threshold", type=float, default=10.0)
    return parser.parse_args()


def load_aligned(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.DataFrame]:
    before = pd.read_csv(args.before_dir / "per_window_metrics.csv")
    after = pd.read_csv(args.after_dir / "per_window_metrics.csv")
    if len(before) != len(after) or not np.allclose(before["start_time_s"], after["start_time_s"]):
        raise ValueError("Before/after evaluations do not contain the same prediction windows.")
    return before, after


def method_series(
    name: str, before: pd.DataFrame, after: pd.DataFrame, kind: str
) -> np.ndarray:
    source = before if name.startswith("Pre-conditioning") else after
    return source[METHODS[name][kind]].to_numpy(dtype=float)


def summarize(
    before: pd.DataFrame, after: pd.DataFrame, threshold: float
) -> list[dict[str, float | str]]:
    rows = []
    for name in METHODS:
        position = method_series(name, before, after, "position")
        acceleration = method_series(name, before, after, "acceleration")
        rows.append(
            {
                "method": name,
                "mean_position_window_rmse_m": float(position.mean()),
                "median_position_window_rmse_m": float(np.median(position)),
                "p95_position_window_rmse_m": float(np.quantile(position, 0.95)),
                "p99_position_window_rmse_m": float(np.quantile(position, 0.99)),
                "position_failure_rate_pct": float(100.0 * np.mean(position > threshold)),
                "mean_acceleration_window_rmse": float(acceleration.mean()),
                "p95_acceleration_window_rmse": float(np.quantile(acceleration, 0.95)),
            }
        )
    return rows


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_summary(
    args: argparse.Namespace,
    before: pd.DataFrame,
    after: pd.DataFrame,
    rows: list[dict[str, float | str]],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    times = after["start_time_s"].to_numpy(dtype=float)

    for name, style in METHODS.items():
        values = method_series(name, before, after, "position")
        axes[0, 0].plot(times, values, color=style["color"], linewidth=1.25, label=name)
    axes[0, 0].axhline(args.threshold, color="black", linestyle="--", linewidth=1.2)
    axes[0, 0].set_xlabel("Prediction start time (s)")
    axes[0, 0].set_ylabel("Window RMSE (m)")
    axes[0, 0].grid(alpha=0.25)

    for name, style in METHODS.items():
        values = np.sort(method_series(name, before, after, "position"))
        fraction = np.arange(1, len(values) + 1) / len(values)
        axes[0, 1].plot(values, fraction, color=style["color"], linewidth=2, label=name)
    axes[0, 1].axvline(args.threshold, color="black", linestyle="--", linewidth=1.2)
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_xlabel("Window RMSE (m), logarithmic scale")
    axes[0, 1].set_ylabel("Fraction of prediction windows")
    axes[0, 1].grid(alpha=0.25)

    labels = ["Pre-conditioning", "Conditioned", "Last acceleration"]
    x = np.arange(len(labels))
    width = 0.34
    means = [float(row["mean_position_window_rmse_m"]) for row in rows]
    p99 = [float(row["p99_position_window_rmse_m"]) for row in rows]
    colors = [style["color"] for style in METHODS.values()]
    axes[1, 0].bar(x - width / 2, means, width, color=colors, alpha=0.9, label="Mean")
    axes[1, 0].bar(
        x + width / 2,
        p99,
        width,
        color=colors,
        alpha=0.38,
        edgecolor=colors,
        linewidth=1.4,
        label="P99",
    )
    axes[1, 0].set_xticks(x, labels, rotation=12)
    axes[1, 0].set_ylabel("Z-position window RMSE (m)")
    axes[1, 0].grid(axis="y", alpha=0.25)
    axes[1, 0].legend(loc="upper center", **LEGEND_STYLE)

    failures = [float(row["position_failure_rate_pct"]) for row in rows]
    bars = axes[1, 1].bar(labels, failures, color=colors)
    axes[1, 1].set_ylabel(f"Windows with RMSE > {args.threshold:g} m (%)")
    axes[1, 1].tick_params(axis="x", rotation=12)
    axes[1, 1].grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, failures, strict=True):
        axes[1, 1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
        )

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=legend_columns(len(legend_labels)),
        **LEGEND_STYLE,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88), pad=1.6, h_pad=2.0, w_pad=1.8)
    save_figure(fig, args.output_dir, "real_flight_before_after_summary")


def plot_timeline(
    args: argparse.Namespace, before: pd.DataFrame, after: pd.DataFrame
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
    times = after["start_time_s"].to_numpy(dtype=float)
    for name, style in METHODS.items():
        axes[0].plot(
            times,
            method_series(name, before, after, "position"),
            color=style["color"],
            linewidth=1.25,
            label=name,
        )
        axes[1].plot(
            times,
            method_series(name, before, after, "acceleration"),
            color=style["color"],
            linewidth=1.25,
            label=name,
    )
    axes[0].axhline(args.threshold, color="black", linestyle="--", linewidth=1.1)
    axes[0].set_ylabel("Z-position window RMSE (m)")
    axes[1].set_ylabel("Z-acceleration window RMSE")
    axes[1].set_xlabel("Prediction start time (s)")
    for axis in axes:
        axis.grid(alpha=0.25)
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=legend_columns(len(legend_labels)),
        **LEGEND_STYLE,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88), pad=1.6, h_pad=2.0)
    save_figure(fig, args.output_dir, "real_flight_before_after_timeline")


def main() -> int:
    args = parse_args()
    args.before_dir = args.before_dir.expanduser().resolve()
    args.after_dir = args.after_dir.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    before, after = load_aligned(args)
    rows = summarize(before, after, args.threshold)
    plot_summary(args, before, after, rows)
    plot_timeline(args, before, after)
    with (args.output_dir / "real_flight_before_after_metrics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote comparison figures and metrics to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
