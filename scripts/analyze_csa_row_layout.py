#!/usr/bin/env python3
"""Compare existing batch-outer CSA results with batch-inner row ordering."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BLUE = "#1769aa"
GREEN = "#238b57"
ORANGE = "#d97706"
HISTORIES = (2048, 4096, 8192, 16384, 32768, 65536,
             131072, 262144, 524288)
BATCHES = (1, 2, 4, 8, 16, 32, 64, 128)
QUERIES = (2, 8, 32, 128, 512, 1024, 4096)


def latest_rows(paths):
    latest = {}
    for path in paths:
        with Path(path).open() as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    latest[row["case_id"]] = row
    return list(latest.values())


def aggregate(paths, prefix):
    grouped = defaultdict(list)
    terminal = latest_rows(paths)
    for row in terminal:
        if row.get("status") == "ok":
            grouped[(int(row["B"]), int(row["H"]), int(row["Q"]))].append(row)
    output = {}
    for key, rows in grouped.items():
        latency = [float(row["latency_ms_median"]) for row in rows]
        tflops = [float(row["tflops"]) for row in rows]
        output[key] = {
            f"{prefix}_latency_ms": statistics.median(latency),
            f"{prefix}_tflops": statistics.median(tflops),
            f"{prefix}_pass_spread": max(latency) / min(latency) - 1.0,
            f"{prefix}_pass_count": len({row["pass_order"] for row in rows}),
        }
    return output, terminal


def load_random_paired(path):
    output = {}
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["B"]), int(row["H"]), int(row["Q"]))
            output[key] = {
                "random_tflops": float(row["random_tflops"]),
                "random_latency_ms": float(row["random_latency_ms"]),
            }
    return output


def pair(random, outer, inner):
    rows = []
    for B, H, Q in sorted(random.keys() & outer.keys() & inner.keys()):
        key = (B, H, Q)
        row = {"B": B, "H": H, "Q": Q, **random[key], **outer[key],
               **inner[key]}
        row["inner_vs_outer_tflops"] = row["inner_tflops"] / row["outer_tflops"]
        row["inner_vs_outer_latency"] = row["inner_latency_ms"] / row["outer_latency_ms"]
        rows.append(row)
    return rows


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=float), q))


def format_size(value):
    return f"{value // 1024}K" if value >= 1024 else str(value)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def draw_curve(ax, rows, title):
    rows = sorted(rows, key=lambda row: row["H"])
    x = [row["H"] for row in rows]
    ax.plot(x, [row["random_tflops"] for row in rows], marker="o",
            linewidth=2, color=BLUE, label="random index")
    ax.plot(x, [row["outer_tflops"] for row in rows], marker="^",
            linewidth=2, linestyle="--", color=GREEN,
            label="CSA batch outer")
    ax.plot(x, [row["inner_tflops"] for row in rows], marker="s",
            linewidth=2, linestyle=":", color=ORANGE,
            label="CSA batch inner")
    ax.set_xscale("log", base=2)
    ax.set_xticks(HISTORIES, [format_size(H) for H in HISTORIES], rotation=45)
    ax.set_xlim(HISTORIES[0] / 1.15, HISTORIES[-1] * 1.15)
    ax.set_title(title)
    ax.set_xlabel("History KV sequence length")
    ax.grid(alpha=0.25)


def plot_figures(rows, figures):
    figures.mkdir(parents=True, exist_ok=True)
    for Q in QUERIES:
        fig, axes = plt.subplots(2, 4, figsize=(17, 7.5), squeeze=False)
        handles = labels = None
        for index, B in enumerate(BATCHES):
            ax = axes[index // 4, index % 4]
            curve = [row for row in rows if row["B"] == B and row["Q"] == Q]
            draw_curve(ax, curve, f"B={B}")
            if index % 4 == 0:
                ax.set_ylabel("Selected-pair TFLOPS")
            if handles is None:
                handles, labels = ax.get_legend_handles_labels()
        fig.suptitle(f"Random and CSA row layouts, new prefill Q={format_size(Q)}",
                     y=0.995)
        fig.legend(handles, labels, loc="upper center", ncol=3,
                   bbox_to_anchor=(0.5, 0.965), fontsize=9)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        for suffix in ("png", "pdf"):
            fig.savefig(figures / f"row_layout_q{Q}_overview.{suffix}",
                        dpi=180 if suffix == "png" else None,
                        bbox_inches="tight")
        plt.close(fig)


def grouped_table(rows, field):
    lines = []
    for value in field[1]:
        selected = [row["inner_vs_outer_tflops"] for row in rows
                    if row[field[0]] == value]
        lines.append(
            f"| {format_size(value)} | {len(selected)} | "
            f"{statistics.median(selected):.3f}x | "
            f"{percentile(selected, 5):.3f}x | {percentile(selected, 95):.3f}x |"
        )
    return "\n".join(lines)


def render_report(output, rows, outer_terminal, inner_terminal, inputs):
    complete = [row for row in rows
                if row["outer_pass_count"] == 2 and row["inner_pass_count"] == 2]
    ratios = [row["inner_vs_outer_tflops"] for row in complete]
    inner_spreads = [row["inner_pass_spread"] for row in complete]
    high = [row["inner_vs_outer_tflops"] for row in complete
            if row["B"] >= 64 and row["Q"] >= 512]
    worst = sorted(complete, key=lambda row: row["inner_vs_outer_tflops"])[:12]
    worst_rows = "\n".join(
        f"| {row['B']} | {format_size(row['H'])} | {format_size(row['Q'])} | "
        f"{row['outer_tflops']:.1f} | {row['inner_tflops']:.1f} | "
        f"{row['inner_vs_outer_tflops']:.3f}x |"
        for row in worst
    )
    report = f"""# Prefill Throughput: Random vs CSA Batch Outer vs Batch Inner

## 构造

- batch outer（已有数据）：`b0q0,b0q1,...,b1q0,b1q1,...`
- batch inner（本次数据）：`b0q0,b1q0,...,b0q1,b1q1,...`
- Q 和 indices 使用同一个无损行置换；KV、逻辑 CSA index、overlap、FLOPs 和
  kernel 完全不变。GPU 数值验证在逆置换后逐元素相同，`max_abs=0`
- 这项实验削弱同一 sequence 的相邻 query 执行邻近性，但不是关闭硬件 L2；
  CTA 的实际调度顺序仍由 kernel 决定

## 完整性

- random 输入：`{inputs[0]}`，共 `{len(rows)}` 个同 shape 基线点
- outer 输入：`{inputs[1]}` 及稳定性覆盖数据
- inner 输入：{', '.join(f'`{path}`' for path in inputs[2:])}
- outer/inner terminal case：`{len(outer_terminal)}/{len(inner_terminal)}`
- 完整双 pass 配对：`{len(complete)}/{len(rows)}`
- inner 双 pass spread P95/最大值：
  `{100 * percentile(inner_spreads, 95):.2f}% / {100 * max(inner_spreads):.2f}%`
- inner spread >5%：`{sum(row['inner_pass_spread'] > .05 for row in complete)}`

## 结果

- batch-inner / batch-outer TFLOPS 中位数：`{statistics.median(ratios):.3f}x`
- P5/P95：`{percentile(ratios, 5):.3f}x / {percentile(ratios, 95):.3f}x`
- `B>=64,Q>=512` 高负载子集：`{statistics.median(high):.3f}x`（{len(high)} 点）
- 全部范围：`{min(ratios):.3f}x--{max(ratios):.3f}x`

### 按 Batch

| B | 点数 | inner/outer 中位数 | P5 | P95 |
|---:|---:|---:|---:|---:|
{grouped_table(complete, ('B', BATCHES))}

### 按 Q

| Q | 点数 | inner/outer 中位数 | P5 | P95 |
|---:|---:|---:|---:|---:|
{grouped_table(complete, ('Q', QUERIES))}

### 损失最大的点

| B | H | Q | outer TFLOPS | inner TFLOPS | inner/outer |
|---:|---:|---:|---:|---:|---:|
{worst_rows}

## 三线绝对吞吐曲线

所有图共用同一绝对 TFLOPS 纵轴：蓝色为 random index，绿色为 CSA
batch-outer，橙色为 CSA batch-inner；没有比例图或第二纵轴。

![Q=2](figures/row_layout_q2_overview.png)

![Q=8](figures/row_layout_q8_overview.png)

![Q=32](figures/row_layout_q32_overview.png)

![Q=128](figures/row_layout_q128_overview.png)

![Q=512](figures/row_layout_q512_overview.png)

![Q=1K](figures/row_layout_q1024_overview.png)

![Q=4K](figures/row_layout_q4096_overview.png)

完整逐点数据见 [`raw/paired.csv`](raw/paired.csv)。
"""
    (output / "analysis.md").write_text(report)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random-paired", required=True)
    parser.add_argument("--outer-input", action="append", required=True)
    parser.add_argument("--inner-input", action="append", required=True)
    parser.add_argument("--output-dir", default="build/csa_row_layout")
    return parser.parse_args()


def main():
    args = parse_args()
    random = load_random_paired(args.random_paired)
    outer, outer_terminal = aggregate(args.outer_input, "outer")
    inner, inner_terminal = aggregate(args.inner_input, "inner")
    rows = pair(random, outer, inner)
    output = Path(args.output_dir)
    write_csv(output / "raw" / "paired.csv", rows)
    plot_figures(rows, output / "figures")
    render_report(output, rows, outer_terminal, inner_terminal,
                  [args.random_paired, args.outer_input[0], *args.inner_input])
    print(f"wrote {len(rows)} paired shapes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
