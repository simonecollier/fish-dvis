#!/usr/bin/env python3
"""
Plot-only helper: recreate the per-video JSD barplot from a saved results JSON,
without recomputing JSDs or rerunning permutation tests.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def _load_per_video_jsd(results_json_path: Path) -> tuple[list[str], np.ndarray]:
    with results_json_path.open("r") as f:
        payload = json.load(f)

    try:
        per_video = payload["results"]["per_video_jsd"]
    except KeyError as e:
        raise KeyError(
            f"Expected key {e} in {results_json_path}. "
            "This script expects the JSON produced by jensen_shannon_test.py."
        )

    video_ids = list(per_video.keys())
    jsd = np.array([float(per_video[vid]) for vid in video_ids], dtype=float)
    return video_ids, jsd


def plot_per_video_jsd_ylim01(
    video_ids: list[str],
    observed_jsd: np.ndarray,
    output_path: Path,
    ymin: float = 0.0,
    ymax: float = 1.0,
) -> None:
    sorted_indices = np.argsort(observed_jsd)
    sorted_jsd = observed_jsd[sorted_indices]
    sorted_video_ids = [video_ids[i] for i in sorted_indices]

    mean_jsd = float(np.mean(observed_jsd))

    fig, ax = plt.subplots(figsize=(14, 6))

    x_pos = np.arange(len(sorted_jsd))
    ax.bar(
        x_pos,
        sorted_jsd,
        alpha=0.7,
        color="steelblue",
        edgecolor="black",
        linewidth=0.5,
    )

    ax.axhline(mean_jsd, color="red", linestyle="--", linewidth=2, label="Mean JSD")

    ax.set_xticks(x_pos)
    ax.set_xticklabels(
        [str(vid) for vid in sorted_video_ids],
        rotation=45,
        ha="right",
        fontsize=14,
    )

    ax.set_xlabel("Video ID (sorted by JSD magnitude)", fontsize=24)
    ax.set_ylabel("Jensen-Shannon Divergence", fontsize=24)
    ax.legend(loc="upper left", fontsize=14)
    ax.tick_params(axis="x", labelsize=14)
    ax.tick_params(axis="y", labelsize=20)

    ax.set_ylim(ymin, ymax)

    stats_text = (
        f"Mean JSD: {mean_jsd:.6f}\n"
        f"Std JSD: {np.std(observed_jsd):.6f}\n"
        f"Min JSD: {np.min(observed_jsd):.6f}\n"
        f"Max JSD: {np.max(observed_jsd):.6f}"
    )
    ax.text(
        0.01,
        0.85,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        fontsize=14,
        family="monospace",
    )
    ax.grid(True, alpha=0.3, linestyle="--", axis="y")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recreate jsd_per_video_barplot.png from an existing jensen_shannon_test_results.json"
    )
    parser.add_argument(
        "--results-json",
        required=True,
        type=Path,
        help="Path to jensen_shannon_test_results.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Where to save the plot (default: alongside JSON as jsd_per_video_barplot_ylim01.png)",
    )
    parser.add_argument("--ymin", type=float, default=0.0)
    parser.add_argument("--ymax", type=float, default=1.0)
    args = parser.parse_args()

    results_json: Path = args.results_json
    output: Path = (
        args.output
        if args.output is not None
        else results_json.parent / "jsd_per_video_barplot_ylim01.png"
    )

    video_ids, jsd = _load_per_video_jsd(results_json)
    plot_per_video_jsd_ylim01(video_ids, jsd, output, ymin=args.ymin, ymax=args.ymax)
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()

