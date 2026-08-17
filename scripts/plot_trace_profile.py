#!/usr/bin/env python3
"""Plot retained CSA activation statistics without requiring the raw trace."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BLUE = "#1769aa"
GREEN = "#238b57"
HISTORY_ORDER = ("2K-8K", "8K-16K", "16K-32K", "32K-64K", "64K+")


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def save(fig, output, name):
    output.mkdir(parents=True, exist_ok=True)
    fig.savefig(output / f"{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_history(rows, output):
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for phase, color, marker in (
        ("assistant", BLUE, "o"),
        ("prefill", GREEN, "^"),
    ):
        lookup = {
            row["history_bin"]: 100 * float(row["adjacent_overlap_p50"])
            for row in rows
            if row["phase"] == phase and row["layer_group"] == "all"
        }
        labels = [label for label in HISTORY_ORDER if label in lookup]
        ax.plot(
            labels,
            [lookup[label] for label in labels],
            marker=marker,
            linewidth=2,
            color=color,
            label=phase,
        )
    ax.set_xlabel("Trace history (tokens)")
    ax.set_ylabel("Median adjacent-query overlap (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend()
    save(fig, output, "csa_overlap_by_history")


def plot_lag(rows, output):
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    styles = ("-", "--", ":", "-.", (0, (3, 1, 1, 1)))
    markers = ("o", "^", "s", "D", "v")
    for index, label in enumerate(HISTORY_ORDER):
        selected = sorted(
            (row for row in rows if row["history_bin"] == label),
            key=lambda row: int(row["query_lag"]),
        )
        if not selected:
            continue
        ax.plot(
            [int(row["query_lag"]) for row in selected],
            [100 * float(row["overlap_p50"]) for row in selected],
            color=(BLUE, GREEN)[index % 2],
            linestyle=styles[index],
            marker=markers[index],
            linewidth=2,
            label=label,
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Query-row lag")
    ax.set_ylabel("Median selected-set overlap (%)")
    ax.grid(alpha=0.25)
    ax.legend(title="History")
    save(fig, output, "csa_overlap_by_query_lag")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile-dir", default="artifacts/data/csa_trace_profile/raw"
    )
    parser.add_argument("--output-dir", default="artifacts/figures/main")
    args = parser.parse_args()
    source = Path(args.profile_dir)
    output = Path(args.output_dir)
    plot_history(read_csv(source / "activation_by_condition.csv"), output)
    plot_lag(read_csv(source / "overlap_by_lag.csv"), output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
