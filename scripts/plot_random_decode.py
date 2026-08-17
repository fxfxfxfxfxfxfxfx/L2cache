#!/usr/bin/env python3
"""Plot random-index sparse-decode throughput against history length."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BLUE = "#1769aa"
HISTORIES = (4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288)
BATCHES = (1, 8, 16, 32, 64)


def format_size(value):
    return f"{value // 1024}K"


def load_rows(path):
    selected = []
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            if (
                row.get("status") == "ok"
                and row.get("pattern") == "independent"
                and int(row["B"]) in BATCHES
                and int(row["H"]) in HISTORIES
            ):
                selected.append({
                    "B": int(row["B"]),
                    "H": int(row["H"]),
                    "tflops": float(row["tflops"]),
                })
    return selected


def plot(rows, output):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), squeeze=False)
    for index, batch in enumerate(BATCHES):
        ax = axes.flat[index]
        curve = sorted(
            (row for row in rows if row["B"] == batch),
            key=lambda row: row["H"],
        )
        ax.plot(
            [row["H"] for row in curve],
            [row["tflops"] for row in curve],
            marker="o",
            linewidth=2,
            color=BLUE,
        )
        ax.set_xscale("log", base=2)
        ax.set_xticks(
            HISTORIES,
            [format_size(history) for history in HISTORIES],
            rotation=45,
        )
        ax.set_xlim(HISTORIES[0] / 1.15, HISTORIES[-1] * 1.15)
        ax.set_ylim(0, max(row["tflops"] for row in curve) * 1.12)
        ax.set_title(f"B={batch}")
        ax.set_xlabel("History KV sequence length")
        ax.set_ylabel("Selected-pair TFLOPS")
        ax.grid(alpha=0.25)
    axes.flat[-1].axis("off")
    fig.suptitle("Random-index sparse decode, Q=1, top-k=2048")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default="artifacts/data/decode_control/raw/aggregate.csv"
    )
    parser.add_argument(
        "--output", default="artifacts/figures/main/random_decode_history.png"
    )
    args = parser.parse_args()
    rows = load_rows(args.input)
    expected = len(BATCHES) * len(HISTORIES)
    if len(rows) != expected:
        raise RuntimeError(
            f"expected {expected} random decode points, found {len(rows)}"
        )
    plot(rows, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
