#!/usr/bin/env python3
"""Plot the full random-index versus CSA-profile-simulated prefill grid."""

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
HISTORIES = (2048, 4096, 8192, 16384, 32768, 65536,
             131072, 262144, 524288)
BATCHES = (1, 2, 4, 8, 16, 32, 64, 128)
QUERIES = (2, 8, 32, 128, 512, 1024, 4096)


def latest_rows(paths):
    latest = {}
    if isinstance(paths, (str, Path)):
        paths = [paths]
    for path in paths:
        with Path(path).open() as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    latest[row["case_id"]] = row
    return list(latest.values())


def load_random(path):
    rows = []
    for row in latest_rows(path):
        if (
            row.get("stage") == "full"
            and row.get("kind") == "prefill"
            and row.get("backend") == "sglang_sparse_q8kv8_fp8"
            and row.get("cache_state") == "steady"
            and row.get("status") == "ok"
            and int(row["B"]) in BATCHES
            and int(row["H"]) in HISTORIES
            and int(row["Q"]) in QUERIES
        ):
            rows.append({
                "B": int(row["B"]), "H": int(row["H"]), "Q": int(row["Q"]),
                "random_tflops": float(row["tflops"]),
                "random_latency_ms": float(row["latency_ms_median"]),
            })
    return {(row["B"], row["H"], row["Q"]): row for row in rows}


def aggregate_sim(paths):
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
            "B": key[0], "H": key[1], "Q": key[2],
            "sim_tflops": statistics.median(tflops),
            "sim_latency_ms": statistics.median(latency),
            "sim_pass_spread": max(latency) / min(latency) - 1.0,
            "sim_overlap": statistics.median(
                float(row["achieved_adjacent_overlap"]) for row in rows
            ),
            "sim_c4_consecutive": statistics.median(
                float(row["achieved_c4_consecutive_fraction"]) for row in rows
            ),
            "unique_kv": int(statistics.median(
                int(row["unique_kv"]) for row in rows
            )),
            "extrapolated": any(
                str(row["extrapolated_64k_plus"]).lower() == "true"
                for row in rows
            ),
            "pass_count": len({row["pass_order"] for row in rows}),
        }
    return output, terminal


def pair(random, simulated):
    rows = []
    for key in sorted(random.keys() & simulated.keys()):
        row = {**random[key], **simulated[key]}
        row["sim_vs_random_tflops"] = row["sim_tflops"] / row["random_tflops"]
        rows.append(row)
    return rows


def load_paired_csv(path):
    if not path:
        return {}
    rows = {}
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["B"]), int(row["H"]), int(row["Q"]))
            rows[key] = row
    return rows


def compare_shared(rows, shared):
    comparison = []
    for row in rows:
        key = (row["B"], row["H"], row["Q"])
        if key not in shared:
            continue
        old_tflops = float(shared[key]["sim_tflops"])
        comparison.append({
            "B": row["B"], "H": row["H"], "Q": row["Q"],
            "random_tflops": row["random_tflops"],
            "shared_template_sim_tflops": old_tflops,
            "independent_seed_sim_tflops": row["sim_tflops"],
            "shared_vs_random": old_tflops / row["random_tflops"],
            "independent_vs_random": row["sim_tflops"] / row["random_tflops"],
            "independent_vs_shared": row["sim_tflops"] / old_tflops,
        })
    return comparison


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["B", "H", "Q"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def format_size(value):
    return f"{value // 1024}K" if value >= 1024 else str(value)


def save_figure(fig, figures, name):
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(figures / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def configure_history_axis(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks(HISTORIES, [format_size(H) for H in HISTORIES], rotation=45)
    ax.set_xlim(HISTORIES[0] / 1.15, HISTORIES[-1] * 1.15)
    ax.grid(alpha=0.25)


def draw_curve(ax, curve, ylabel=True, title=None, legend=False):
    curve = sorted(curve, key=lambda row: row["H"])
    histories = [row["H"] for row in curve]
    ax.plot(
        histories, [row["random_tflops"] for row in curve],
        marker="o", color=BLUE, linewidth=2, label="random index",
    )
    ax.plot(
        histories, [row["sim_tflops"] for row in curve],
        marker="^", color=GREEN, linestyle="--", linewidth=2,
        label="CSA simulated (independent batch seeds)",
    )
    if any(row["extrapolated"] for row in curve):
        ax.axvspan(73739, HISTORIES[-1] * 1.15, color=GREEN, alpha=0.045)
    configure_history_axis(ax)
    if ylabel:
        ax.set_ylabel("Selected-pair TFLOPS")
    ax.set_xlabel("History KV sequence length")
    if title:
        ax.set_title(title)
    if legend:
        ax.legend(fontsize=8)


def plot_individual(rows, figures):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["B"], row["Q"])].append(row)
    links = []
    for (B, Q), curve in sorted(grouped.items()):
        fig, ax = plt.subplots(figsize=(8.4, 4.8))
        draw_curve(
            ax, curve, title=f"B={B}, new prefill Q={format_size(Q)}",
            legend=True,
        )
        fig.tight_layout()
        name = f"prefill_throughput_b{B}_q{Q}"
        save_figure(fig, figures, name)
        links.append((B, Q, name))
    return links


def plot_query_overviews(rows, figures):
    for Q in QUERIES:
        fig, axes = plt.subplots(2, 4, figsize=(17, 7.5), squeeze=False)
        legend_handles = None
        legend_labels = None
        for index, B in enumerate(BATCHES):
            ax = axes[index // 4, index % 4]
            curve = [row for row in rows if row["B"] == B and row["Q"] == Q]
            draw_curve(
                ax, curve, ylabel=index % 4 == 0,
                title=f"B={B}", legend=False,
            )
            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()
        fig.suptitle(f"New prefill Q={format_size(Q)}", y=0.995)
        fig.legend(
            legend_handles, legend_labels, loc="upper center", ncol=2,
            bbox_to_anchor=(0.5, 0.965), fontsize=9,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        save_figure(fig, figures, f"prefill_throughput_q{Q}_overview")


def percentile(values, q):
    return float(np.percentile(np.asarray(values, dtype=float), q))


def render_report(output, rows, sim_terminal, links, input_paths, comparison):
    complete = [row for row in rows if row["pass_count"] == 2]
    ratios = [row["sim_vs_random_tflops"] for row in complete]
    spreads = [row["sim_pass_spread"] for row in complete]
    extrapolated = [row for row in complete if row["extrapolated"]]
    by_history = {
        H: [row["sim_vs_random_tflops"] for row in complete if row["H"] == H]
        for H in HISTORIES
    }
    missing = sorted(
        (B, H, Q) for B in BATCHES for H in HISTORIES for Q in QUERIES
        if not any(
            row["B"] == B and row["H"] == H and row["Q"] == Q for row in rows
        )
    )
    unstable = [row for row in complete if row["sim_pass_spread"] > 0.05]
    unstable_text = ", ".join(
        f"`B={row['B']},H={format_size(row['H'])},Q={format_size(row['Q'])}` "
        f"({100 * row['sim_pass_spread']:.1f}%)"
        for row in unstable
    ) or "无"
    link_rows = "\n".join(
        f"| {B} | {format_size(Q)} | [PNG](figures/{name}.png) | "
        f"[PDF](figures/{name}.pdf) |" for B, Q, name in links
    )
    history_text = ", ".join(
        f"`{format_size(H)}={statistics.median(values):.3f}x`"
        for H, values in by_history.items() if values
    )
    comparison_section = ""
    if comparison:
        changes = [row["independent_vs_shared"] for row in comparison]
        old_ratios = [row["shared_vs_random"] for row in comparison]
        new_ratios = [row["independent_vs_random"] for row in comparison]
        high_load = [
            row for row in comparison if row["B"] >= 64 and row["Q"] >= 512
        ]
        old_low = [row for row in comparison if row["shared_vs_random"] < 0.98]
        recovered = [
            row for row in old_low if row["independent_vs_random"] >= 0.98
        ]
        representative = sorted(
            high_load or comparison,
            key=lambda row: row["independent_vs_shared"], reverse=True,
        )[:10]
        representative_rows = "\n".join(
            f"| {row['B']} | {format_size(row['H'])} | {format_size(row['Q'])} | "
            f"{row['shared_template_sim_tflops']:.1f} | "
            f"{row['independent_seed_sim_tflops']:.1f} | "
            f"{row['independent_vs_shared']:.3f}x | "
            f"{row['independent_vs_random']:.3f}x |"
            for row in representative
        )
        high_change = [row["independent_vs_shared"] for row in high_load]
        high_text = (
            f"高负载子集 `B>=64,Q>=512` 的中位变化为 "
            f"`{statistics.median(high_change):.3f}x`（{len(high_load)} 点）。"
            if high_change else ""
        )
        batch_rows = "\n".join(
            f"| {B} | {len(values)} | {statistics.median(values):.3f}x | "
            f"{percentile(values, 5):.3f}x | {percentile(values, 95):.3f}x |"
            for B in BATCHES
            if (values := [
                row["independent_vs_shared"] for row in comparison
                if row["B"] == B
            ])
        )
        comparison_section = f"""
## Batch Seed A/B 对照

旧版对每个 batch 元素广播同一条 `[Q,512]` index 模板，仅添加物理地址 offset；
新版让每个 batch 元素使用独立 seed 生成完整序列。两版 overlap/年龄/连续簇集
profile 相同，所以此处只检验不应存在的跨 batch 模板相关性。

- 匹配点 `{len(comparison)}`；新版/旧版吞吐中位数
  `{statistics.median(changes):.3f}x`，P5/P95
  `{percentile(changes, 5):.3f}x / {percentile(changes, 95):.3f}x`
- 旧版与新版相对 random 的中位数分别为
  `{statistics.median(old_ratios):.3f}x / {statistics.median(new_ratios):.3f}x`
- 旧版低于 random `2%` 以上的 `{len(old_low)}` 点中，
  新版恢复至 random `98%` 以上的有 `{len(recovered)}` 点。{high_text}

| B | 点数 | 新版/旧版中位数 | P5 | P95 |
|---:|---:|---:|---:|---:|
{batch_rows}

以下列出高负载子集中新版相对旧版改善最大的 10 个点；完整数据见
[`raw/shared_vs_independent.csv`](raw/shared_vs_independent.csv)。

| B | H | Q | 旧版 TFLOPS | 新版 TFLOPS | 新/旧 | 新版/random |
|---:|---:|---:|---:|---:|---:|---:|
{representative_rows}
"""
    report = f"""# Full Prefill Throughput: Random vs CSA-Profile Simulation

## 实验范围

- 横轴：历史 KV sequence length `2K,4K,...,512K`
- 固定新 prefill 长度：`Q={{2,8,32,128,512,1K,4K}}`
- 固定 batch：`B={{1,2,4,8,16,32,64,128}}`
- 每个 `(B,Q)` 单独一张绝对 selected-pair TFLOPS 曲线；蓝色是之前本机实测的
  random/coprime-strided index，绿色是本次实测的 CSA-profile simulated index
- 总目标网格 504 个 shape；配对成功 `{len(rows)}`，缺失 `{len(missing)}`：
  {', '.join(f'`B={B},H={format_size(H)},Q={format_size(Q)}`' for B,H,Q in missing) or '无'}
- random 曲线复用 2026-08-11 的完整本机基线；simulated 曲线于 2026-08-12
  在同一 H800、相同 kernel/runtime 下测量，环境版本由实验入口强制比对
- simulated index 对每个 batch 元素使用独立 seed；不同 batch 不再广播同一条
  query/index 模板

## 仿真定义

仿真器从 8 条 CSA trace 的分层统计中同时读取相邻-query overlap、selected-entry
年龄 CDF 和连续 C4 比例。每个 query 用 Markov selected set 保留目标数量的上一行
entry，再按年龄分布和连续簇集补齐到 512 个唯一 C4 id；每个 id 展开为四个连续
token index，因此 kernel 每行仍读取 2,048 个唯一 KV，且严格满足 causal 范围。

`H+Q <= 73,739` 的点位于已下载 trace 的观测长度范围内；更长的
`{len(extrapolated)}` 个配对固定使用 `64K+` profile，是规律外推而非真实 trace
replay，图中以很浅的绿色背景标出。吞吐点均为 GPU 实测，没有吞吐插值。

## 完整性

- 仿真输入：{', '.join(f'`{path}`' for path in input_paths)}；后列按 case id 覆盖前列
- 仿真原始 case `{len(sim_terminal)}`；成功
  `{sum(row.get('status') == 'ok' for row in sim_terminal)}`
- 完整正序/倒序配对 `{len(complete)}/{len(rows)}`
- 仿真双 pass spread P95 / 最大值：`{100 * percentile(spreads, 95):.2f}% / {100 * max(spreads):.2f}%`
- spread >5% 的配对：`{sum(row['sim_pass_spread'] > 0.05 for row in complete)}`
- 残余顺序漂移：{unstable_text}

## 吞吐结果

- Simulated/random TFLOPS 中位数 `{statistics.median(ratios):.3f}x`，
  P5/P95 `{percentile(ratios, 5):.3f}x / {percentile(ratios, 95):.3f}x`
- 按 history 的中位吞吐比：{history_text}
- 全部配对范围 `{min(ratios):.3f}x--{max(ratios):.3f}x`

{comparison_section}

![Q=2 overview](figures/prefill_throughput_q2_overview.png)

![Q=128 overview](figures/prefill_throughput_q128_overview.png)

![Q=1K overview](figures/prefill_throughput_q1024_overview.png)

![Q=4K overview](figures/prefill_throughput_q4096_overview.png)

## 分图

| B | Q | PNG | PDF |
|---:|---:|---|---|
{link_rows}
"""
    (output / "analysis.md").write_text(report)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--random-input",
        default="artifacts/data/random_baseline/raw/results.jsonl",
    )
    parser.add_argument(
        "--sim-input", default="artifacts/data/csa_batch_outer/raw/results.jsonl"
    )
    parser.add_argument("--override-input", action="append", default=[])
    parser.add_argument("--shared-paired", default=None,
                        help="Optional paired.csv from the old shared-template run")
    parser.add_argument("--output-dir", default="build/csa_profile_sim")
    return parser.parse_args()


def main():
    args = parse_args()
    random = load_random(args.random_input)
    simulated, sim_terminal = aggregate_sim(
        [args.sim_input, *args.override_input]
    )
    rows = pair(random, simulated)
    comparison = compare_shared(rows, load_paired_csv(args.shared_paired))
    output = Path(args.output_dir)
    figures = output / "figures"
    write_csv(output / "raw" / "paired.csv", rows)
    if comparison:
        write_csv(output / "raw" / "shared_vs_independent.csv", comparison)
    links = plot_individual(rows, figures)
    plot_query_overviews(rows, figures)
    render_report(
        output, rows, sim_terminal, links,
        [args.sim_input, *args.override_input], comparison,
    )
    print(f"wrote {len(rows)} paired shapes and {len(links)} individual figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
