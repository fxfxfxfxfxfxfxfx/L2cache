#!/usr/bin/env python3
"""Plot the random-index sparse-prefill baseline against history length."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BLUE = "#1769aa"
HISTORIES = (2048, 4096, 8192, 16384, 32768, 65536,
             131072, 262144, 524288)
BATCHES = (1, 2, 4, 8, 16, 32, 64, 128)


def format_size(value):
    return f"{value // 1024}K" if value >= 1024 else str(value)


def load_rows(path, query):
    latest = {}
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                latest[row["case_id"]] = row
    selected = []
    for row in latest.values():
        if (
            row.get("stage") == "full"
            and row.get("kind") == "prefill"
            and row.get("backend") == "sglang_sparse_q8kv8_fp8"
            and row.get("cache_state") == "steady"
            and row.get("status") == "ok"
            and int(row["Q"]) == query
            and int(row["B"]) in BATCHES
            and int(row["H"]) in HISTORIES
        ):
            selected.append({
                "B": int(row["B"]),
                "H": int(row["H"]),
                "tflops": float(row["tflops"]),
            })
    return selected


def plot(rows, query, output):
    fig, axes = plt.subplots(2, 4, figsize=(17, 7.5), squeeze=False)
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
        ax.set_title(f"B={batch}")
        ax.set_xlabel("History KV sequence length")
        ax.grid(alpha=0.25)
        if index % 4 == 0:
            ax.set_ylabel("Selected-pair TFLOPS")
    fig.suptitle(f"Random-index sparse prefill, new prefill Q={format_size(query)}")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default="artifacts/data/random_baseline/raw/results.jsonl"
    )
    parser.add_argument("--query", type=int, default=512)
    parser.add_argument(
        "--output", default="artifacts/figures/main/random_prefill_q512.png"
    )
    args = parser.parse_args()
    rows = load_rows(args.input, args.query)
    expected = len(BATCHES) * len(HISTORIES)
    if len(rows) != expected:
        raise RuntimeError(
            f"expected {expected} random baseline points, found {len(rows)}"
        )
    plot(rows, args.query, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
