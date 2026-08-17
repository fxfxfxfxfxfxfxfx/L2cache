#!/usr/bin/env python3
"""Analyze sampled CSA activation rules and Q8KV8 trace-replay benchmarks."""

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
HISTORIES = (8192, 16384, 32768, 65536)


def load_latest_many(paths):
    latest = {}
    for path in paths:
        with path.open() as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    latest[row["case_id"]] = row
    return list(latest.values())


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=float), q))


def format_size(value):
    if value >= 1024 and value % 1024 == 0:
        return f"{value // 1024}K"
    return str(value)


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["B", "H", "Q"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_figure(fig, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(directory / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def aggregate(rows):
    groups = defaultdict(list)
    for row in rows:
        if row.get("status") == "ok":
            groups[(int(row["B"]), int(row["H"]), int(row["Q"]),
                    row["pattern"])].append(row)
    aggregated = []
    for (B, H, Q, pattern), group in sorted(groups.items()):
        latency = [float(row["latency_ms_median"]) for row in group]
        tflops = [float(row["tflops"]) for row in group]
        aggregated.append({
            "B": B, "H": H, "Q": Q, "pattern": pattern,
            "latency_ms": statistics.median(latency),
            "latency_ms_pass_min": min(latency),
            "latency_ms_pass_max": max(latency),
            "pass_spread": max(latency) / min(latency) - 1.0,
            "tflops": statistics.median(tflops),
            "tflops_pass_min": min(tflops), "tflops_pass_max": max(tflops),
            "adjacent_overlap": statistics.median(
                float(row["achieved_adjacent_overlap"]) for row in group
            ),
            "unique_kv": int(statistics.median(
                int(row["unique_kv"]) for row in group
            )),
            "reuse_factor": statistics.median(
                float(row["reuse_factor"]) for row in group
            ),
            "c4_consecutive_fraction": (
                statistics.median(float(row["c4_consecutive_fraction"])
                                  for row in group)
                if pattern == "csa_trace_replay" else None
            ),
            "pass_count": len({row["pass_order"] for row in group}),
        })
    return aggregated


def pair_rows(aggregated):
    grouped = defaultdict(dict)
    for row in aggregated:
        grouped[(row["B"], row["H"], row["Q"])][row["pattern"]] = row
    paired = []
    for (B, H, Q), patterns in sorted(grouped.items()):
        original = patterns.get("original_strided")
        trace = patterns.get("csa_trace_replay")
        if not original or not trace:
            continue
        paired.append({
            "B": B, "H": H, "Q": Q,
            "original_tflops": original["tflops"],
            "trace_tflops": trace["tflops"],
            "trace_vs_original_tflops": trace["tflops"] / original["tflops"],
            "original_latency_ms": original["latency_ms"],
            "trace_latency_ms": trace["latency_ms"],
            "original_overlap": original["adjacent_overlap"],
            "trace_overlap": trace["adjacent_overlap"],
            "overlap_delta": trace["adjacent_overlap"] - original["adjacent_overlap"],
            "original_unique_kv": original["unique_kv"],
            "trace_unique_kv": trace["unique_kv"],
            "original_vs_trace_unique_kv": original["unique_kv"] / trace["unique_kv"],
            "trace_c4_consecutive_fraction": trace["c4_consecutive_fraction"],
            "original_pass_spread": original["pass_spread"],
            "trace_pass_spread": trace["pass_spread"],
            "complete_two_pass": (
                original["pass_count"] == 2 and trace["pass_count"] == 2
            ),
        })
    return paired


def curve_map(paired):
    curves = defaultdict(list)
    for row in paired:
        curves[(row["B"], row["Q"])].append(row)
    for curve in curves.values():
        curve.sort(key=lambda row: row["H"])
    return curves


def plot_overview(
    curves, figures, metric, ylabel, filename, ratio=False, scale=1.0,
):
    batches = sorted({key[0] for key in curves})
    queries = sorted({key[1] for key in curves})
    fig, axes = plt.subplots(
        len(batches), len(queries), figsize=(16.5, 3.2 * len(batches)),
        squeeze=False,
    )
    legend_handles = []
    legend_labels = []
    for row_index, B in enumerate(batches):
        for column_index, Q in enumerate(queries):
            ax = axes[row_index, column_index]
            curve = curves.get((B, Q), [])
            histories = [row["H"] for row in curve]
            if ratio:
                line = ax.plot(
                    histories, [scale * row[metric] for row in curve], marker="o",
                    linewidth=2, color=GREEN, label="CSA trace / strided",
                )[0]
                ax.axhline(1.0, color="#777777", linestyle="--", linewidth=1)
                handles = [line]
            else:
                first = ax.plot(
                    histories, [scale * row[metric[0]] for row in curve], marker="o",
                    linewidth=2, color=BLUE, label="original strided",
                )[0]
                second = ax.plot(
                    histories, [scale * row[metric[1]] for row in curve], marker="^",
                    linestyle="--", linewidth=2, color=GREEN,
                    label="CSA trace replay",
                )[0]
                handles = [first, second]
            if not legend_handles:
                legend_handles = handles
                legend_labels = [line.get_label() for line in handles]
            ax.set_xscale("log", base=2)
            ax.set_xticks(histories, [format_size(H) for H in histories], rotation=45)
            ax.set_title(f"B={B}, Q={format_size(Q)}")
            ax.grid(alpha=0.25)
            if row_index == len(batches) - 1:
                ax.set_xlabel("History KV length")
            if column_index == 0:
                ax.set_ylabel(ylabel)
    fig.legend(legend_handles, legend_labels, loc="upper center",
               ncol=len(legend_handles), fontsize=9, bbox_to_anchor=(0.5, 0.985))
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    save_figure(fig, figures, filename)


def plot_individual(curves, figures):
    for (B, Q), curve in curves.items():
        histories = [row["H"] for row in curve]
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
        left, right = axes
        left.plot(histories, [row["original_tflops"] for row in curve],
                  marker="o", linewidth=2, color=BLUE, label="original strided")
        left.plot(histories, [row["trace_tflops"] for row in curve],
                  marker="^", linestyle="--", linewidth=2, color=GREEN,
                  label="CSA trace replay")
        left.set_ylabel("Selected-pair TFLOPS")
        left.legend(fontsize=8)
        right.plot(histories, [100 * row["original_overlap"] for row in curve],
                   marker="o", linewidth=2, color=BLUE, label="original strided")
        right.plot(histories, [100 * row["trace_overlap"] for row in curve],
                   marker="^", linestyle="--", linewidth=2, color=GREEN,
                   label="CSA trace replay")
        right.set_ylabel("Mean adjacent overlap (%)")
        right.set_ylim(0, 100)
        right.legend(fontsize=8)
        for ax in axes:
            ax.set_xscale("log", base=2)
            ax.set_xticks(histories, [format_size(H) for H in histories], rotation=45)
            ax.set_xlabel("History KV length")
            ax.grid(alpha=0.25)
        fig.suptitle(f"CSA trace replay: B={B}, Q={format_size(Q)}")
        fig.tight_layout()
        save_figure(fig, figures, f"trace_replay_b{B}_q{Q}")


def plot_mechanism(paired, figures):
    fig, ax = plt.subplots(figsize=(7.4, 5.3))
    x = np.asarray([row["original_vs_trace_unique_kv"] for row in paired])
    y = np.asarray([row["trace_vs_original_tflops"] for row in paired])
    ax.scatter(x, y, color=BLUE, s=42, alpha=0.72, label="Measured shape")
    if len(x) >= 2:
        coefficients = np.polyfit(x, y, 1)
        fit_x = np.linspace(float(x.min()), float(x.max()), 100)
        ax.plot(
            fit_x, np.polyval(coefficients, fit_x), color=GREEN,
            linewidth=2, label="Linear fit",
        )
    ax.axhline(1.0, color="#777777", linestyle="--", linewidth=1)
    ax.axvline(1.0, color="#777777", linestyle="--", linewidth=1)
    ax.set_xlabel("Original / CSA-trace unique KV")
    ax.set_ylabel("CSA-trace / original TFLOPS")
    ax.grid(alpha=0.25)
    ax.legend()
    save_figure(fig, figures, "speedup_vs_unique_kv_reduction")


def trace_observation_tables(trace_dir):
    conditions = list(csv.DictReader(
        (trace_dir / "raw" / "activation_by_condition.csv").open()
    ))
    lags = list(csv.DictReader((trace_dir / "raw" / "overlap_by_lag.csv").open()))
    history_rows = []
    for label in ("2K-8K", "8K-16K", "16K-32K", "32K-64K", "64K+"):
        phases = {}
        for phase in ("assistant", "prefill"):
            row = next(
                item for item in conditions
                if item["history_bin"] == label and item["phase"] == phase
                and item["layer_group"] == "all"
            )
            phases[phase] = row
        history_rows.append(
            f"| {label} | {100 * float(phases['assistant']['adjacent_overlap_p50']):.1f}% "
            f"| {100 * float(phases['prefill']['adjacent_overlap_p50']):.1f}% "
            f"| {100 * float(phases['assistant']['within_2048_tokens_mean']):.1f}% "
            f"| {100 * float(phases['prefill']['within_2048_tokens_mean']):.1f}% |"
        )
    lag_rows = []
    for label in ("2K-8K", "8K-16K", "16K-32K", "32K-64K", "64K+"):
        by_lag = {
            int(row["query_lag"]): 100 * float(row["overlap_p50"])
            for row in lags if row["history_bin"] == label
        }
        lag_rows.append(
            f"| {label} | " + " | ".join(
                f"{by_lag[lag]:.1f}%" for lag in (1, 4, 16, 64, 256)
            ) + " |"
        )
    return "\n".join(history_rows), "\n".join(lag_rows)


def render_report(output, raw_rows, paired, aggregated, input_paths, trace_dir):
    complete = [row for row in paired if row["complete_two_pass"]]
    ratios = [row["trace_vs_original_tflops"] for row in complete]
    by_history = {
        H: [row["trace_vs_original_tflops"] for row in complete if row["H"] == H]
        for H in HISTORIES
    }
    spreads = [row["pass_spread"] for row in aggregated]
    unstable = [row for row in aggregated if row["pass_spread"] > 0.05]
    unstable_text = ", ".join(
        f"`B={row['B']},H={format_size(row['H'])},Q={format_size(row['Q'])},"
        f"{row['pattern']}` ({100 * row['pass_spread']:.2f}%)"
        for row in unstable
    ) or "无"
    unique_reduction = [row["original_vs_trace_unique_kv"] for row in complete]
    overlap_delta = [row["overlap_delta"] for row in complete]
    correlation = float(np.corrcoef(unique_reduction, ratios)[0, 1])
    history_table, lag_table = trace_observation_tables(trace_dir)
    links = []
    for B, Q in sorted({(row["B"], row["Q"]) for row in complete}):
        name = f"trace_replay_b{B}_q{Q}"
        links.append(
            f"| {B} | {format_size(Q)} | [PNG](figures/{name}.png) | "
            f"[PDF](figures/{name}.pdf) |"
        )
    report = f"""# CSA Trace Activation Rules and Q8KV8 Replay Benchmark

## 范围

- 数据源：ModelScope 私有数据集 `fxiaoO/deepseek-v4-flash-swebench-csa-topk`
- 本机只下载 8/800 条 lite trace，压缩文件合计约 2.6 GiB；prompt 长度覆盖
  6.9K 到 73.7K tokens，不下载完整约 331 GB 数据集
- trace tensor 为 `int32[rows,21,512]`；21 个 CSA slot 对应 hidden layer
  2,4,...,42，512 个值是 C4 compressed KV-entry id
- 统计按 `(history bin, assistant/prefill-equivalent)` 分层，每层每档最多取 128 行；
  这些 phase 是 trace 语义标签，所有数据实际均由 full-trace prefill 收集

## Index 激活规律

| History | Assistant 相邻 overlap P50 | Prefill 相邻 overlap P50 | Assistant 最近 2K | Prefill 最近 2K |
|---|---:|---:|---:|---:|
{history_table}

相邻 overlap 随 history 增长而下降，但到 64K+ 仍远高于均匀/strided synthetic
index。长 history 下 assistant span 的 overlap 高于 system/user/tool-result span。
被选 entry 也不是独立散点：相邻 C4 id 比例从短 trace 的约 72% 降到 64K+ 的
36-40%，说明真实索引同时具有时间复用和空间簇集。

| History | lag=1 | lag=4 | lag=16 | lag=64 | lag=256 |
|---|---:|---:|---:|---:|---:|
{lag_table}

复用随 query lag 平滑衰减而非一步消失。Layer 2/4/36/42 的相邻 overlap 较高，
中间层最低；完整 layer 曲线见下图。消息 phase 边界的 overlap P50 为 84.0%，
内部为 74.4%，所以边界并不等价于索引完全刷新。

![Overlap by history and phase](../local_csa_trace_sample/figures/adjacent_overlap_by_history_phase.png)

![Overlap decay by lag](../local_csa_trace_sample/figures/overlap_decay_by_query_lag.png)

![Overlap by layer](../local_csa_trace_sample/figures/adjacent_overlap_by_layer.png)

![Recent KV fraction](../local_csa_trace_sample/figures/recent_kv_fraction_by_history.png)

## Replay 映射

对每个 benchmark shape，从真实 tensor 取 `[H-Q,H)` 的 query 行和真实 CSA layer。
每个 C4 id 展开为四个连续 KV token，因此每行仍是 2,048 个唯一 index，集合
overlap 比例保持不变，且全部 index 位于历史 prefix `[0,H)`。不同 batch element
使用独立物理 KV segment，并轮换 trace/layer。比较组仍使用原 causal coprime-stride
index；Q/KV tensor、Q8KV8 kernel、shape、FLOPs 和计时方法完全相同。

## Benchmark 完整性

- 最终输入：{', '.join(f'`{path}`' for path in input_paths)}，后列按 case id 覆盖前列
- 原始 case：{len(raw_rows)}；成功：{sum(row.get('status') == 'ok' for row in raw_rows)}
- 完整双 pass 配对：{len(complete)}/{len(paired)}
- 双 pass spread P95 / 最大值：{100 * percentile(spreads, 95):.2f}% / {100 * max(spreads):.2f}%
- 残余 spread >5%：{unstable_text}

## Benchmark 结果

- CSA trace / original strided TFLOPS：中位数 `{statistics.median(ratios):.3f}x`，
  P5/P95 `{percentile(ratios, 5):.3f}x / {percentile(ratios, 95):.3f}x`
- 按 H 的中位吞吐比：{', '.join(f'`{format_size(H)}={statistics.median(values):.3f}x`' for H, values in by_history.items())}
- 最大吞吐比：`{max(ratios):.3f}x`
- Original / trace unique-KV 中位数：`{statistics.median(unique_reduction):.3f}x`
- 相邻 overlap 增量中位数：`{100 * statistics.median(overlap_delta):.1f}` 个百分点
- 吞吐比与 unique-KV 缩减倍数相关系数：`{correlation:.3f}`

![Absolute throughput overview](figures/trace_replay_throughput_overview.png)

![Speedup overview](figures/trace_replay_speedup_overview.png)

![Overlap overview](figures/trace_replay_overlap_overview.png)

![Speedup vs unique KV](figures/speedup_vs_unique_kv_reduction.png)

## 结论与边界

真实 CSA trace 保留的跨 query 重用远高于旧 strided index，因此旧 history 曲线
高估了真实激活规律下 overlap 消失的速度。短 history 上两种分布吞吐接近；随着
H 增长，trace replay 的 working set 更小，并在 64K 上形成稳定、可测的吞吐收益。
这支持“index 激活规律会显著改变 sparse-prefill history scaling”，但不能把数值
直接外推到 DSA：CSA 与 DSA 的检索器、top-k 语义和 KV 表示不同，C4->4 token
展开也是保持工作量与 overlap 的代理映射。此外这里只抽样 8 条 lite trace，
不代表完整 SWE-bench 分布；残余双-pass 异常点也应作为误差边界保留。

## 分图

| B | Q | PNG | PDF |
|---:|---:|---|---|
{chr(10).join(links)}
"""
    (output / "analysis.md").write_text(report)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default="artifacts/data/csa_trace_replay/raw/results.jsonl"
    )
    parser.add_argument("--override-input", action="append", default=[])
    parser.add_argument(
        "--trace-analysis", default="artifacts/data/csa_trace_profile"
    )
    parser.add_argument("--output-dir", default="build/csa_trace_replay")
    return parser.parse_args()


def main():
    args = parse_args()
    input_paths = [Path(args.input), *(Path(path) for path in args.override_input)]
    rows = load_latest_many(input_paths)
    aggregated = aggregate(rows)
    paired = pair_rows(aggregated)
    curves = curve_map(paired)
    output = Path(args.output_dir)
    raw = output / "raw"
    figures = output / "figures"
    write_csv(raw / "aggregate.csv", aggregated)
    write_csv(raw / "paired.csv", paired)
    plot_overview(
        curves, figures, ("original_tflops", "trace_tflops"),
        "Selected-pair TFLOPS", "trace_replay_throughput_overview",
    )
    plot_overview(
        curves, figures, "trace_vs_original_tflops", "Trace / strided TFLOPS",
        "trace_replay_speedup_overview", ratio=True,
    )
    plot_overview(
        curves, figures, ("original_overlap", "trace_overlap"),
        "Mean adjacent overlap (%)", "trace_replay_overlap_overview",
        scale=100.0,
    )
    plot_individual(curves, figures)
    plot_mechanism(paired, figures)
    render_report(
        output, rows, paired, aggregated, input_paths, Path(args.trace_analysis)
    )
    print(f"wrote {len(paired)} paired points and {output / 'analysis.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
