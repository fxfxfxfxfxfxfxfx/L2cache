#!/usr/bin/env python3
"""Aggregate and plot the paired sparse-decode L2 residency experiment."""

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {"hot-resident": "#d1495b", "flushed": "#1769aa"}


def load(path):
    return [json.loads(line) for line in path.open() if line.strip()]


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        if row.get("status") == "ok":
            groups[(row["B"], row["H"], row["cache_state"])].append(row)
    output = []
    for (batch, history, state), values in groups.items():
        medians = [row["latency_ms_median"] for row in values]
        row = dict(values[-1])
        row.update(
            B=batch, H=history, cache_state=state,
            latency_ms=statistics.median(medians),
            tflops=statistics.median(value["tflops"] for value in values),
            pass_count=len({value["pass_order"] for value in values}),
            pass_spread_pct=(max(medians) / min(medians) - 1) * 100,
        )
        output.append(row)
    return output


def pair_states(rows):
    index = {(row["B"], row["H"], row["cache_state"]): row for row in rows}
    pairs = []
    for batch, history, state in sorted(index):
        if state != "hot-resident":
            continue
        hot = index[(batch, history, "hot-resident")]
        cold = index.get((batch, history, "flushed"))
        if cold is None:
            continue
        pairs.append({
            "B": batch, "H": history,
            "hot_latency_us": hot["latency_ms"] * 1000,
            "flushed_latency_us": cold["latency_ms"] * 1000,
            "cold_over_hot": cold["latency_ms"] / hot["latency_ms"],
            "hot_tflops": hot["tflops"], "flushed_tflops": cold["tflops"],
            "hot_over_flushed_tflops": hot["tflops"] / cold["tflops"],
            "selected_kv_vs_l2": hot["selected_kv_vs_l2"],
            "known_touched_vs_l2": hot["known_touched_vs_l2"],
            "selected_kv_bytes": hot["selected_kv_bytes"],
            "known_touched_bytes": hot["known_touched_bytes"],
            "hot_pass_spread_pct": hot["pass_spread_pct"],
            "flushed_pass_spread_pct": cold["pass_spread_pct"],
        })
    return pairs


def write_csv(rows, path):
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def format_h(history):
    return f"{history // 1024}K"


def save(fig, figures, name):
    fig.savefig(figures / f"{name}.png", dpi=160, bbox_inches="tight")
    fig.savefig(figures / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_latency(rows, figures):
    batches = sorted({row["B"] for row in rows})
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    for ax, batch in zip(axes.flat, batches):
        for state in ("hot-resident", "flushed"):
            points = sorted(
                (row for row in rows if row["B"] == batch
                 and row["cache_state"] == state), key=lambda row: row["H"])
            ax.plot([row["H"] for row in points],
                    [row["latency_ms"] * 1000 for row in points], marker="o",
                    linewidth=2, color=COLORS[state], label=state)
        histories = sorted({row["H"] for row in points})
        ax.set_xscale("log", base=2)
        ax.set_xticks(histories, [format_h(h) for h in histories])
        ax.set_title(f"B={batch}")
        ax.set_xlabel("history length")
        ax.set_ylabel("decode latency (us)")
        ax.grid(alpha=0.25)
    for ax in axes.flat[len(batches):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle("Native FlashMLA FP8 sparse decode: L2 residency proxies",
                 y=1.01)
    fig.tight_layout()
    save(fig, figures, "decode_l2_hot_flushed_latency")


def plot_ratio(pairs, figures):
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for batch in sorted({row["B"] for row in pairs}):
        points = [row for row in pairs if row["B"] == batch]
        axes[0].plot([row["H"] for row in points],
                     [row["cold_over_hot"] for row in points], marker="o",
                     linewidth=2, label=f"B={batch}")
    histories = sorted({row["H"] for row in pairs})
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(histories, [format_h(h) for h in histories])
    axes[0].axhline(1.0, color="#555555", linestyle="--", linewidth=1)
    axes[0].axhline(1.475, color="#d1495b", linestyle=":", linewidth=1,
                    label="controlled prefill 1.475x")
    axes[0].set_xlabel("history length")
    axes[0].set_ylabel("flushed latency / hot latency")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8, ncol=2)

    for row in pairs:
        axes[1].scatter(row["known_touched_vs_l2"], row["cold_over_hot"],
                        s=55, label=f"B={row['B']}" if row["H"] == histories[0]
                        else None)
    axes[1].axvline(1.0, color="#555555", linestyle="--", linewidth=1,
                    label="50 MiB L2")
    axes[1].axhline(1.0, color="#555555", linestyle="--", linewidth=1)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("known touched bytes / L2 capacity")
    axes[1].set_ylabel("flushed latency / hot latency")
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8, ncol=2)
    fig.suptitle("Observable decode benefit from the L2-hot proxy")
    fig.tight_layout()
    save(fig, figures, "decode_l2_hot_flushed_ratio")


def write_analysis(aggregated, pairs, output_dir):
    by_batch = defaultdict(list)
    for row in pairs:
        by_batch[row["B"]].append(row["cold_over_hot"])
    rows_text = []
    for row in pairs:
        rows_text.append(
            f"| {row['B']} | {format_h(row['H'])} | "
            f"{row['selected_kv_vs_l2']:.3f}x | "
            f"{row['known_touched_vs_l2']:.3f}x | "
            f"{row['hot_latency_us']:.3f} | {row['flushed_latency_us']:.3f} | "
            f"{row['cold_over_hot']:.3f}x |"
        )
    summary_text = []
    for batch in sorted(by_batch):
        summary_text.append(
            f"| {batch} | {statistics.median(by_batch[batch]):.3f}x |"
        )
    max_spread = max(row["pass_spread_pct"] for row in aggregated)
    fit_pairs = [row for row in pairs if row["known_touched_vs_l2"] <= 0.5]
    fit_median = statistics.median(row["cold_over_hot"] for row in fit_pairs)
    fit_max = max(row["cold_over_hot"] for row in fit_pairs)
    all_max = max(row["cold_over_hot"] for row in pairs)
    prefill_ratio = 1.475

    text = f"""# Native Sparse Decode L2 Residency Proxy

## 设计目的

这个实验在 `H=4K,8K,16K,32K,64K,128K,256K,512K` 上检验一个更具体的问题：
如果 native FlashMLA FP8 sparse decode 的
selected KV 在计时前被同一个 decode 调用预热，它相对被 256 MiB 读取驱逐后
究竟能快多少；这个差值能否达到此前 controlled prefill 的 `1.475x` 延迟差。

无 NCU 时不能声称 `hot-resident` 是 100% L2 hit，也不能声称 `flushed` 是 100%
L2 miss。它们是两个可重复且时钟预热对称的软件干预：每个 timed call 前都执行
一次相同 decode prime 和一次 256 MiB flush；hot-resident 使用
`flush -> prime -> timed`，flushed 使用 `prime -> flush -> timed`。prime 和
flush 都在 CUDA event 之外。两种状态逐次交替，history 再做
ascending/descending 两遍。

`B=1,8,16` 的已知 touched footprint 明显小于 50 MiB；`B=32` 接近容量边界；
`B=64` 仅 selected KV 就超过 L2，是容量负对照。已知 footprint 包括 selected
KV、Q、output 和 indices，不包含 kernel 内部不可见的临时状态。

## 完整结果

| B | H | Selected KV / L2 | Known touched / L2 | Hot us | Flushed us | Flushed/Hot |
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows_text)}

按 batch 聚合 history 后：

| B | Median flushed/hot |
|---:|---:|
{chr(10).join(summary_text)}

升降序 pass 的最大 latency spread 为 `{max_spread:.2f}%`。对于 known footprint
不超过 L2 一半的点，flushed/hot 中位比为 `{fit_median:.3f}x`、最大值为
`{fit_max:.3f}x`；全部点最大值为 `{all_max:.3f}x`。

![Hot 与 flushed latency](figures/decode_l2_hot_flushed_latency.png)

![Hot/Flushed 比值和容量](figures/decode_l2_hot_flushed_ratio.png)

## 与 prefill 归因的判别

此前 Q8 `N=256` controlled prefill 的 independent-dispersed 512K/2K 延迟比为
`{prefill_ratio:.3f}x`。如果 fit-in-L2 decode 的 flushed/hot 也接近这个比例，
就说明从高复用/L2-resident 状态转为 cold 状态本身足以解释相近量级的性能损失。
如果 decode 差值明显更小，则“decode 没有下降空间，因为它本来不命中 L2”不被
支持；prefill 的 47.5% 更可能来自单次 kernel 内 256 个 query rows 共同放大的
unique working set、并发访存组织和空间局部性，而不是简单的两态 L2 hit/miss。

## 证据边界

该实验只能量化预热/驱逐干预对 decode kernel latency 的可观察影响。相同 decode
prime 同时预热 KV、Q、indices、代码和其他状态，所以差值不能只归给 KV；flush
也可能影响频率和未测量的 translation/cache 状态。最终硬件 hit rate 仍需要
NCU 或等价 performance counters。
"""
    (output_dir / "analysis.md").write_text(text)
    return text


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",
                        default="assets/decode_l2_residency/raw/results.jsonl")
    parser.add_argument("--output-dir", default="assets/decode_l2_residency")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    aggregated = aggregate(load(Path(args.input)))
    pairs = pair_states(aggregated)
    if not pairs:
        raise RuntimeError("no complete hot/flushed pairs")
    raw_dir = output_dir / "raw"
    write_csv(aggregated, raw_dir / "aggregate.csv")
    write_csv(pairs, raw_dir / "paired.csv")
    plot_latency(aggregated, figures)
    plot_ratio(pairs, figures)
    print(write_analysis(aggregated, pairs, output_dir))


if __name__ == "__main__":
    main()
