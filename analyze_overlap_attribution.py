#!/usr/bin/env python3
"""Analyze and plot the matched-overlap sparse-prefill experiment."""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PATTERNS = ("original_strided", "matched_contiguous", "inverted_contiguous")
LABELS = {
    "original_strided": "Original dispersed",
    "matched_contiguous": "Overlap-matched contiguous",
    "inverted_contiguous": "Inverse-overlap contiguous",
}
COLORS = {
    "original_strided": "#c23b32",
    "matched_contiguous": "#1769aa",
    "inverted_contiguous": "#238b57",
}


def load_rows(path):
    latest = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            latest[row["case_id"]] = row
    return list(latest.values())


def percentile(values, q):
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=float), q))


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def pair_rows(rows):
    by_shape = defaultdict(dict)
    for row in rows:
        if row.get("status") == "ok":
            key = (int(row["B"]), int(row["H"]), int(row["Q"]))
            by_shape[key][row["pattern"]] = row

    paired = []
    for (B, H, Q), pattern_rows in sorted(by_shape.items()):
        if not all(pattern in pattern_rows for pattern in PATTERNS):
            continue
        original = pattern_rows["original_strided"]
        matched = pattern_rows["matched_contiguous"]
        inverted = pattern_rows["inverted_contiguous"]
        paired.append({
            "B": B, "H": H, "Q": Q, "total_q": B * Q,
            "original_overlap": original["achieved_adjacent_overlap"],
            "matched_overlap": matched["achieved_adjacent_overlap"],
            "overlap_error": matched["achieved_adjacent_overlap"]
            - original["achieved_adjacent_overlap"],
            "original_unique_kv": original["unique_kv"],
            "matched_unique_kv": matched["unique_kv"],
            "inverted_overlap": inverted["achieved_adjacent_overlap"],
            "inverted_unique_kv": inverted["unique_kv"],
            "original_reuse": original["reuse_factor"],
            "matched_reuse": matched["reuse_factor"],
            "inverted_reuse": inverted["reuse_factor"],
            "original_tflops": original["tflops"],
            "matched_tflops": matched["tflops"],
            "inverted_tflops": inverted["tflops"],
            "matched_vs_original": matched["tflops"] / original["tflops"],
            "inverted_vs_original": inverted["tflops"] / original["tflops"],
            "matched_unique_vs_original": matched["unique_kv"]
            / original["unique_kv"],
        })
    return paired, by_shape


def choose_panels(paired, limit=8):
    coverage = defaultdict(set)
    for row in paired:
        coverage[(row["B"], row["Q"])].add(row["H"])
    preferred = [(1, 64), (1, 256), (8, 256), (64, 16)]
    chosen = [shape for shape in preferred if len(coverage.get(shape, ())) >= 2]
    ranked = sorted(
        coverage, key=lambda shape: (-len(coverage[shape]), shape[0], shape[1])
    )
    for shape in ranked:
        if shape not in chosen:
            chosen.append(shape)
        if len(chosen) >= limit:
            break
    return chosen[:limit]


def plot_history_tflops(paired, figures):
    panels = choose_panels(paired)
    if not panels:
        return
    cols = 2
    rows_n = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(12, 4.2 * rows_n), squeeze=False)
    for ax, (B, Q) in zip(axes.flat, panels):
        panel = [row for row in paired if row["B"] == B and row["Q"] == Q]
        panel.sort(key=lambda row: row["H"])
        for pattern in PATTERNS:
            key = pattern.split("_")[0] + "_tflops"
            if pattern == "original_strided":
                key = "original_tflops"
            elif pattern == "matched_contiguous":
                key = "matched_tflops"
            else:
                key = "inverted_tflops"
            ax.plot(
                [row["H"] for row in panel], [row[key] for row in panel],
                marker="o", linewidth=1.8, markersize=4,
                color=COLORS[pattern], label=LABELS[pattern],
            )
        ax.set_xscale("log", base=2)
        ax.set_title(f"B={B}, Q={Q}")
        ax.set_xlabel("History KV length")
        ax.set_ylabel("Effective TFLOPS/s")
        ax.grid(True, alpha=0.25)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.subplots_adjust(top=0.92, hspace=0.34, wspace=0.25)
    save_figure(fig, figures, "tflops_vs_history_three_index_distributions")


def plot_overlap_history(paired, figures):
    panels = choose_panels(paired)
    if not panels:
        return
    cols = 2
    rows_n = math.ceil(len(panels) / cols)
    fig, axes = plt.subplots(rows_n, cols, figsize=(12, 4.0 * rows_n), squeeze=False)
    for ax, (B, Q) in zip(axes.flat, panels):
        panel = [row for row in paired if row["B"] == B and row["Q"] == Q]
        panel.sort(key=lambda row: row["H"])
        ax.plot(
            [row["H"] for row in panel],
            [100 * row["original_overlap"] for row in panel],
            marker="o", color=COLORS["original_strided"], label="Original",
        )
        ax.plot(
            [row["H"] for row in panel],
            [100 * row["matched_overlap"] for row in panel],
            linestyle="--", color=COLORS["matched_contiguous"],
            label="Matched contiguous",
        )
        ax.plot(
            [row["H"] for row in panel],
            [100 * row["inverted_overlap"] for row in panel],
            linestyle=":", color=COLORS["inverted_contiguous"],
            label="Inverse-overlap contiguous",
        )
        ax.set_xscale("log", base=2)
        ax.set_ylim(-2, 102)
        ax.set_title(f"B={B}, Q={Q}")
        ax.set_xlabel("History KV length")
        ax.set_ylabel("Adjacent-query overlap (%)")
        ax.grid(True, alpha=0.25)
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.subplots_adjust(top=0.92, hspace=0.34, wspace=0.25)
    save_figure(fig, figures, "adjacent_overlap_vs_history")


def plot_matched_ratio(paired, figures):
    if not paired:
        return
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    x = np.asarray([100 * row["original_overlap"] for row in paired])
    y = np.asarray([row["matched_vs_original"] for row in paired])
    color = np.asarray([math.log2(row["total_q"]) for row in paired])
    scatter = ax.scatter(x, y, c=color, cmap="viridis", alpha=0.68, s=24)
    ax.axhline(1.0, color="black", linewidth=1, linestyle="--")
    ax.set_xlabel("Original adjacent-query overlap (%)")
    ax.set_ylabel("Matched-contiguous / original TFLOPS")
    ax.grid(True, alpha=0.25)
    bar = fig.colorbar(scatter, ax=ax)
    bar.set_label("log2(B x Q)")
    save_figure(fig, figures, "matched_contiguous_vs_original")


def curve_retentions(paired):
    curves = defaultdict(list)
    for row in paired:
        curves[(row["B"], row["Q"])].append(row)
    output = []
    for (B, Q), rows in curves.items():
        rows.sort(key=lambda row: row["H"])
        if len(rows) < 2:
            continue
        lo, hi = rows[0], rows[-1]
        output.append({
            "B": B, "Q": Q, "H_min": lo["H"], "H_max": hi["H"],
            "original_retention": hi["original_tflops"] / lo["original_tflops"],
            "matched_retention": hi["matched_tflops"] / lo["matched_tflops"],
            "inverted_retention": hi["inverted_tflops"] / lo["inverted_tflops"],
            "overlap_min_history": lo["original_overlap"],
            "overlap_max_history": hi["original_overlap"],
        })
    return output


def render_analysis(output_dir, all_rows, paired, retentions):
    ok = sum(row.get("status") == "ok" for row in all_rows)
    failed = sum(row.get("status") == "failed" for row in all_rows)
    skipped = len(all_rows) - ok - failed
    ratios = [row["matched_vs_original"] for row in paired]
    inverted_ratios = [row["inverted_vs_original"] for row in paired]
    overlap_errors = [abs(row["overlap_error"]) for row in paired]
    unique_ratios = [row["matched_unique_vs_original"] for row in paired]
    original_ret = [row["original_retention"] for row in retentions]
    matched_ret = [row["matched_retention"] for row in retentions]
    inverted_ret = [row["inverted_retention"] for row in retentions]

    median_ratio = statistics.median(ratios) if ratios else float("nan")
    median_unique_ratio = (
        statistics.median(unique_ratios) if unique_ratios else float("nan")
    )
    retention_gap = (
        statistics.median(abs(a - b) for a, b in zip(original_ret, matched_ret))
        if original_ret else float("nan")
    )
    falling_overlap_curves = [
        row for row in retentions
        if row["overlap_min_history"] - row["overlap_max_history"] >= 0.05
    ]
    inverse_direction_fraction = (
        sum(
            row["inverted_retention"] > row["original_retention"] + 0.05
            for row in falling_overlap_curves
        ) / len(falling_overlap_curves)
        if falling_overlap_curves else float("nan")
    )
    matched_support = (
        ratios and abs(median_ratio - 1.0) <= 0.05 and retention_gap <= 0.05
    )
    inverse_support = (
        falling_overlap_curves and inverse_direction_fraction >= 0.70
    )
    if matched_support and inverse_support:
        verdict = (
            "在本次构造范围内，连续窗口在匹配相邻-query overlap 后基本复现了原始"
            "性能及 history retention，反向-overlap 组也在多数曲线上给出反向剂量"
            "响应。这共同支持 overlap 是主导软件可见变量。"
        )
    elif matched_support:
        verdict = (
            "匹配相邻 overlap 后基本复现了原始曲线，但反向-overlap 组没有在足够多"
            "曲线上形成反向剂量响应。证据支持相关性和部分因果效应，但不足以单独把"
            "全部下降归因于相邻 overlap。"
        )
    elif ratios:
        verdict = (
            "相邻-query overlap 匹配后仍未充分复现原始性能或 history retention。"
            "因此不能只用相邻 overlap 一个标量完成归因，应联合考察整个 chunk 的"
            " unique-KV working set、非相邻 query 重用与行内离散性。"
        )
    else:
        verdict = "没有完整三组配对数据，暂时不能作归因判断。"

    text = f"""# Sparse MLA Prefill Overlap-Matched Attribution

## 实验问题

固定 Q/KV、Q8KV8 sparse-prefill kernel、`topk=2048`、shape 和计时方法，只替换
indices。比较原始互质步长分布、相邻 overlap 匹配的连续窗口，以及 overlap 固定
反向 overlap 的连续窗口。所有 history 严格大于 2048。

## 完整性

- 原始结果行：{len(all_rows)}
- 成功：{ok}
- 失败：{failed}
- 跳过：{skipped}
- 完整三组配对 shape：{len(paired)}
- overlap 匹配最大绝对误差：{max(overlap_errors, default=float('nan')):.6f}

## 汇总

- `matched_contiguous/original` TFLOPS 中位数：{median_ratio:.3f}x
- 该比值 P5/P95：{percentile(ratios, 5):.3f}x / {percentile(ratios, 95):.3f}x
- `inverted_contiguous/original` TFLOPS 中位数：{statistics.median(inverted_ratios) if inverted_ratios else float('nan'):.3f}x
- `matched/original` unique-KV 中位数：{median_unique_ratio:.3f}x
- 各 `(B,Q)` 曲线 matched 与 original retention 差值中位数：{retention_gap:.3f}
- original history retention 中位数：{statistics.median(original_ret) if original_ret else float('nan'):.3f}
- matched history retention 中位数：{statistics.median(matched_ret) if matched_ret else float('nan'):.3f}
- inverse-overlap history retention 中位数：{statistics.median(inverted_ret) if inverted_ret else float('nan'):.3f}
- 原始 overlap 明显下降的曲线数：{len(falling_overlap_curves)}
- 其中 inverse retention 比 original 高至少 0.05 的比例：{inverse_direction_fraction:.3f}

## 客观结论

{verdict}

`adjacent_overlap` 只描述相邻两行集合交集。即使该值完全相同，不同构造仍可能有
不同的全 chunk union、reuse distance 和非相邻 query 重用。因此只有在
`matched_contiguous` 同时复现性能曲线，并且 unique-KV 差异不足以解释结果时，
才能把原始下降主要归因于相邻-query overlap。

## 图

![TFLOPS versus history](figures/tflops_vs_history_three_index_distributions.png)

![Overlap versus history](figures/adjacent_overlap_vs_history.png)

![Matched versus original](figures/matched_contiguous_vs_original.png)
"""
    (output_dir / "analysis.md").write_text(text)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default="assets/overlap_attribution/raw/results.jsonl"
    )
    parser.add_argument("--output-dir", default="assets/overlap_attribution")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    rows = load_rows(Path(args.input))
    paired, _ = pair_rows(rows)
    retentions = curve_retentions(paired)
    fields = list(paired[0]) if paired else ["B", "H", "Q"]
    write_csv(output_dir / "raw" / "paired.csv", paired, fields)
    retention_fields = list(retentions[0]) if retentions else ["B", "Q"]
    write_csv(
        output_dir / "raw" / "curve_retentions.csv",
        retentions, retention_fields,
    )
    figures = output_dir / "figures"
    plot_history_tflops(paired, figures)
    plot_overlap_history(paired, figures)
    plot_matched_ratio(paired, figures)
    render_analysis(output_dir, rows, paired, retentions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
