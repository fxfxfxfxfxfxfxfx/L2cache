#!/usr/bin/env python3
"""Analyze native sparse-decode shared versus independent selected KV."""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = {"shared": "#2a9d8f", "independent": "#d1495b"}


def load(path):
    return [json.loads(line) for line in path.open() if line.strip()]


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        if row.get("status") == "ok":
            groups[(row["B"], row["H"], row["pattern"])].append(row)
    output = []
    for (batch, history, pattern), values in groups.items():
        medians = [row["latency_ms_median"] for row in values]
        row = dict(values[-1])
        row.update(
            B=batch, H=history, pattern=pattern,
            latency_ms=statistics.median(medians),
            tflops=statistics.median(value["tflops"] for value in values),
            pass_count=len({value["pass_order"] for value in values}),
            pass_spread_pct=(max(medians) / min(medians) - 1) * 100,
        )
        output.append(row)
    return output


def pair(rows):
    index = {(row["B"], row["H"], row["pattern"]): row for row in rows}
    output = []
    for batch, history, pattern in sorted(index):
        if pattern != "shared":
            continue
        shared = index[(batch, history, "shared")]
        independent = index[(batch, history, "independent")]
        output.append({
            "B": batch, "H": history,
            "shared_latency_us": shared["latency_ms"] * 1000,
            "independent_latency_us": independent["latency_ms"] * 1000,
            "independent_over_shared": (
                independent["latency_ms"] / shared["latency_ms"]),
            "shared_tflops": shared["tflops"],
            "independent_tflops": independent["tflops"],
            "shared_over_independent_tflops": (
                shared["tflops"] / independent["tflops"]),
            "shared_unique_kv": shared["unique_selected_kv"],
            "independent_unique_kv": independent["unique_selected_kv"],
            "shared_selected_kv_vs_l2": shared["selected_kv_vs_l2"],
            "independent_selected_kv_vs_l2": independent["selected_kv_vs_l2"],
            "shared_known_touched_vs_l2": shared["known_touched_vs_l2"],
            "independent_known_touched_vs_l2": independent["known_touched_vs_l2"],
        })
    return output


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
    histories = sorted({row["H"] for row in rows})
    ncols = 3
    nrows = math.ceil(len(batches) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows),
                             sharex=True)
    for ax, batch in zip(axes.flat, batches):
        for pattern in ("shared", "independent"):
            points = sorted(
                (row for row in rows if row["B"] == batch
                 and row["pattern"] == pattern), key=lambda row: row["H"])
            ax.plot([row["H"] for row in points],
                    [row["latency_ms"] * 1000 for row in points], marker="o",
                    linewidth=2, color=COLORS[pattern], label=pattern)
        ax.set_xscale("log", base=2)
        ax.set_xticks(histories, [format_h(h) for h in histories], rotation=30)
        ax.set_title(f"B={batch}")
        ax.set_xlabel("history length")
        ax.set_ylabel("decode latency (us)")
        ax.grid(alpha=0.25)
    for ax in axes.flat[len(batches):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 0.955))
    fig.suptitle("Native FlashMLA FP8 decode: within-call selected-KV reuse",
                 y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    save(fig, figures, "decode_shared_independent_latency")


def plot_ratio(pairs, figures):
    histories = sorted({row["H"] for row in pairs})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for batch in sorted({row["B"] for row in pairs}):
        points = [row for row in pairs if row["B"] == batch]
        axes[0].plot([row["H"] for row in points],
                     [row["independent_over_shared"] for row in points],
                     marker="o", linewidth=2, label=f"B={batch}")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(histories, [format_h(h) for h in histories], rotation=30)
    axes[0].axhline(1.0, color="#555555", linestyle="--", linewidth=1)
    axes[0].axhline(1.475, color="#d1495b", linestyle=":", linewidth=1,
                    label="controlled prefill 1.475x")
    axes[0].set_xlabel("history length")
    axes[0].set_ylabel("independent / shared latency")
    axes[0].grid(alpha=0.25)
    axes[0].legend(fontsize=8, ncol=2)

    medians = []
    batches = sorted({row["B"] for row in pairs})
    for batch in batches:
        values = [row["independent_over_shared"] for row in pairs
                  if row["B"] == batch]
        medians.append(statistics.median(values))
    axes[1].plot(batches, medians, marker="o", linewidth=2, color="#1769aa")
    axes[1].axhline(1.0, color="#555555", linestyle="--", linewidth=1)
    axes[1].axhline(1.475, color="#d1495b", linestyle=":", linewidth=1)
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(batches, [str(batch) for batch in batches])
    axes[1].set_xlabel("batch size")
    axes[1].set_ylabel("median independent / shared latency")
    axes[1].grid(alpha=0.25)
    fig.suptitle("Performance value of shared selected KV in native decode")
    fig.tight_layout()
    save(fig, figures, "decode_shared_independent_ratio")


def write_analysis(aggregated, pairs, output_dir, rerun_used,
                   supplement_used):
    by_batch = defaultdict(list)
    for row in pairs:
        by_batch[row["B"]].append(row["independent_over_shared"])
    table = []
    for row in pairs:
        table.append(
            f"| {row['B']} | {format_h(row['H'])} | "
            f"{row['shared_unique_kv']:,} | {row['independent_unique_kv']:,} | "
            f"{row['shared_latency_us']:.3f} | "
            f"{row['independent_latency_us']:.3f} | "
            f"{row['independent_over_shared']:.3f}x |"
        )
    batch_table = []
    for batch in sorted(by_batch):
        values = by_batch[batch]
        batch_table.append(
            f"| {batch} | {statistics.median(values):.3f}x | "
            f"{min(values):.3f}x | {max(values):.3f}x |"
        )
    max_spread = max(row["pass_spread_pct"] for row in aggregated)
    max_ratio = max(row["independent_over_shared"] for row in pairs)
    b64_median = statistics.median(by_batch[64])
    prefill_ratio = 1.475

    text = f"""# Native Sparse Decode Selected-KV Reuse

## 实验设计

本实验不再尝试用不同的前置 kernel 制造“100% L2 hit/miss”，因为无 NCU 无法
验证 hit rate，而且前置 workload 会混入 DVFS。这里使用单次 native FlashMLA
FP8 sparse decode kernel 内的因果干预：

- `shared`：所有 batch row 读取同一组 2,048 个物理 KV；
- `independent`：每个 batch row 读取各自独立的 2,048 个物理 KV。

两者使用同一份 Q/KV allocation、同一 kernel、batch、topk、FLOPs 和计时流程。
每次 timed call 前都执行完全相同的 256 MiB flush，因此立即前置 workload 和
clock conditioning 相同。只改变 timed kernel 内的 unique selected-KV working
set。History 为 `4K..512K` 的 2 倍递增序列，做 ascending/descending 两遍；
每点 5 warmup、30 repeat。

{("原始 full run 中 `B=1,H=64K` 和 `B=64,H=256K/512K` 出现超过 5% 的"
  "升降序绝对延迟漂移。这些 shape 在无外部 CUDA process 条件下以 50 repeat "
  "重测，以下聚合显式使用 rerun 覆盖原始异常点。" if rerun_used else "")}

{("为与 controlled prefill 的 `N=256` 对齐，另补测 `B=128/256` 的 "
  "`H=4K/32K/256K`；这些点同样做两个顺序 pass。B<=64 保留完整的八个 "
  "history 点。" if supplement_used else "")}

## 完整结果

| B | H | Shared unique KV | Independent unique KV | Shared us | Independent us | Ind/Shared |
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(table)}

按 batch 汇总：

| B | Median ratio | Min | Max |
|---:|---:|---:|---:|
{chr(10).join(batch_table)}

升降序单点最大 latency spread 为 `{max_spread:.2f}%`。全部点最大
independent/shared 为 `{max_ratio:.3f}x`，B=64 的 history 中位比为
`{b64_median:.3f}x`。

![Shared/independent latency](figures/decode_shared_independent_latency.png)

![Shared/independent ratio](figures/decode_shared_independent_ratio.png)

## 归因

`B=1` 时 shared 与 independent indices 完全相同，因此应为 1.0x，它是协议的
零效应校验。随着 batch 增长，logical pairs 不变，但 independent unique KV 从
2,048 增加到 `B*2048`；若 latency ratio 随 batch 增大，说明 native decode
kernel 确实能利用 batch rows 之间的 selected-KV 重用。这里的收益发生在单次
kernel 内，不依赖前一次调用残留的 L2。

此前 controlled prefill 的最大 history ratio 为 `{prefill_ratio:.3f}x`。本实验
最大 decode reuse ratio 为 `{max_ratio:.3f}x`。因此可以判断“decode 完全不能从
L2/缓存重用获益”是否错误，同时也能判断仅靠共享 selected KV 是否足以复现
prefill 的 47.5% 差异。

## 边界

该实验直接控制 unique physical KV addresses，但没有测量 L2 hit counter。
Shared 的速度提升可以归因于 memory-hierarchy reuse 的软件输入条件，不能进一步
声明其中多少来自 L2、L1、memory coalescing、广播或 HBM traffic reduction。
此外，普通 serving decode 的不同 sequence 通常不会故意读取同一物理 KV；
shared 模式是用于量化“可复用时的性能上限”的诊断对照，不是实际调度语义。
"""
    (output_dir / "analysis.md").write_text(text)
    return text


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="assets/decode_kv_reuse/raw/results.jsonl")
    parser.add_argument("--rerun-input", default=None)
    parser.add_argument("--supplement-input", action="append", default=[])
    parser.add_argument("--output-dir", default="assets/decode_kv_reuse")
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    aggregated = aggregate(load(Path(args.input)))
    rerun_used = bool(args.rerun_input)
    if rerun_used:
        overrides = aggregate(load(Path(args.rerun_input)))
        index = {(row["B"], row["H"], row["pattern"]): row
                 for row in aggregated}
        index.update({(row["B"], row["H"], row["pattern"]): row
                      for row in overrides})
        aggregated = list(index.values())
    supplement_used = bool(args.supplement_input)
    if supplement_used:
        index = {(row["B"], row["H"], row["pattern"]): row
                 for row in aggregated}
        for path in args.supplement_input:
            supplemental = aggregate(load(Path(path)))
            index.update({(row["B"], row["H"], row["pattern"]): row
                          for row in supplemental})
        aggregated = list(index.values())
    pairs = pair(aggregated)
    if not pairs:
        raise RuntimeError("no complete shared/independent pairs")
    raw_dir = output_dir / "raw"
    write_csv(aggregated, raw_dir / "aggregate.csv")
    write_csv(pairs, raw_dir / "paired.csv")
    manifest = {
        "base_input": args.input,
        "rerun_override_input": args.rerun_input,
        "supplement_inputs": args.supplement_input,
        "precedence": "rerun overrides matching base keys; supplements add or override",
        "aggregate_rows": len(aggregated),
        "paired_rows": len(pairs),
    }
    (raw_dir / "merge_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    plot_latency(aggregated, figures)
    plot_ratio(pairs, figures)
    print(write_analysis(aggregated, pairs, output_dir, rerun_used,
                         supplement_used))


if __name__ == "__main__":
    main()
