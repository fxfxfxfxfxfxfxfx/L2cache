#!/usr/bin/env python3
"""Plot SGLang Q8xKV8 sparse prefill against the prior BF16 benchmark."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT = Path(__file__).resolve().parent
OLD_RESULTS = PROJECT / "assets" / "raw" / "history_scaling_results.jsonl"
Q8_RESULTS = PROJECT / "assets" / "q8kv8" / "raw" / "results.jsonl"
OUT_FIG = PROJECT / "assets" / "q8kv8" / "figures"
OUT_RAW = PROJECT / "assets" / "q8kv8" / "raw"
BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
HISTORIES = [2048, 4096, 8192, 16384, 32768, 65536,
             131072, 262144, 524288]
QUERIES = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048,
           4096, 8192, 16384, 32768]
BF16_BACKEND = "flashmla_sparse_bf16"
Q8_BACKEND = "sglang_sparse_q8kv8_fp8"


def load(path):
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def latest(rows):
    return {r["case_id"]: r for r in rows}.values()


def q_label(q):
    return str(q) if q < 1024 else f"{q // 1024}K"


def h_label(h):
    return f"{h // 1024}K"


def bf16_tflops(row):
    # The old BF16 sparse path launches one attention kernel; profiler data is
    # preferred where available to match the ECHO selected-pair convention.
    return row.get("figure2_hardware_tflops", row.get("tflops"))


def save(fig, name):
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FIG / f"{name}.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUT_FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}.png/.pdf")


def build_comparison(old_rows, q8_rows):
    old = {(r["B"], r["H"], r["Q"]): r for r in latest(old_rows)
           if r.get("backend") == BF16_BACKEND
           and r.get("cache_state") == "steady"}
    q8 = {(r["B"], r["H"], r["Q"]): r for r in latest(q8_rows)
          if r.get("backend") == Q8_BACKEND
          and r.get("cache_state") == "steady"}
    output = []
    for q in QUERIES:
        for b in BATCHES:
            for h in HISTORIES:
                key = (b, h, q)
                a, z = old.get(key), q8.get(key)
                a_ok = bool(a and a.get("status") == "ok")
                z_ok = bool(z and z.get("status") == "ok")
                a_perf = bf16_tflops(a) if a_ok else None
                z_perf = z.get("tflops") if z_ok else None
                output.append({
                    "B": b, "H": h, "Q": q,
                    "bf16_status": a.get("status") if a else "missing",
                    "bf16_tflops": a_perf,
                    "bf16_latency_ms": (a.get("latency_ms_median")
                                         if a_ok else None),
                    "q8kv8_status": z.get("status") if z else "missing",
                    "q8kv8_skip_reason": z.get("skip_reason") if z else None,
                    "q8kv8_tflops": z_perf,
                    "q8kv8_latency_ms": (z.get("latency_ms_median")
                                          if z_ok else None),
                    "q8kv8_vs_bf16_tflops": (
                        z_perf / a_perf if z_perf and a_perf else None),
                })
    return output


def write_csv(rows):
    path = OUT_RAW / "q8kv8_vs_bf16_prefill.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def plot_history(rows):
    colors = {"bf16": "#1769aa", "q8": "#d1495b"}
    for q in QUERIES:
        q_rows = [r for r in rows if r["Q"] == q]
        ymax = max(
            [r["q8kv8_tflops"] or 0 for r in q_rows]
            + [r["bf16_tflops"] or 0 for r in q_rows]
        ) * 1.08
        fig, axes = plt.subplots(3, 3, figsize=(14.5, 10.5),
                                 sharex=True, sharey=True)
        for ax, b in zip(axes.flat, BATCHES):
            panel = [r for r in rows if r["Q"] == q and r["B"] == b]
            bf16 = [r for r in panel if r["bf16_status"] == "ok"]
            q8 = [r for r in panel if r["q8kv8_status"] == "ok"]
            if bf16:
                ax.plot([r["H"] for r in bf16],
                        [r["bf16_tflops"] for r in bf16], marker="o",
                        linewidth=1.8, markersize=4, color=colors["bf16"],
                        label="FlashMLA BF16 sparse")
            if q8:
                ax.plot([r["H"] for r in q8],
                        [r["q8kv8_tflops"] for r in q8], marker="s",
                        linewidth=1.8, markersize=4, color=colors["q8"],
                        label="SGLang Q8xKV8")
            skipped = [r["H"] for r in panel
                       if r["q8kv8_status"] != "ok"]
            if skipped:
                ax.scatter(skipped, [0] * len(skipped), marker="x", s=30,
                           color="#666666", label="Q8 OOM")
            ax.set_title(f"batch={b}", fontsize=10)
            ax.set_xscale("log", base=2)
            ax.set_xticks(HISTORIES, [h_label(h) for h in HISTORIES],
                          rotation=45, fontsize=7)
            ax.set_ylim(0, ymax)
            ax.grid(alpha=0.25)
        for ax in axes[-1, :]:
            ax.set_xlabel("history sequence length")
        for ax in axes[:, 0]:
            ax.set_ylabel("selected-pair TFLOPS")
        handles = [
            plt.Line2D([], [], color=colors["bf16"], marker="o",
                       label="FlashMLA BF16 sparse"),
            plt.Line2D([], [], color=colors["q8"], marker="s",
                       label="SGLang Q8xKV8"),
            plt.Line2D([], [], color="#666666", marker="x",
                       linestyle="None", label="Q8 OOM / skipped"),
        ]
        fig.legend(handles=handles, loc="upper center", ncol=3,
                   bbox_to_anchor=(0.5, 0.965))
        fig.suptitle(
            f"H800 sparse MLA prefill - new length Q={q_label(q)}",
            y=0.995,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        save(fig, f"q8kv8_history_scaling_q{q}")


def plot_speedup_summary(rows):
    selected_q = [16, 64, 256, 1024, 4096, 16384]
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 11), sharex=True,
                             sharey=True)
    for ax, q in zip(axes.flat, selected_q):
        for b in BATCHES:
            panel = [r for r in rows if r["Q"] == q and r["B"] == b
                     and r["q8kv8_vs_bf16_tflops"] is not None]
            if panel:
                ax.plot([r["H"] for r in panel],
                        [r["q8kv8_vs_bf16_tflops"] for r in panel],
                        marker="o", markersize=3, linewidth=1.3,
                        label=f"B={b}")
        ax.axhline(1.0, color="#555555", linewidth=1, linestyle="--")
        ax.set_xscale("log", base=2)
        ax.set_xticks(HISTORIES, [h_label(h) for h in HISTORIES],
                      rotation=45, fontsize=7)
        ax.set_title(f"Q={q_label(q)}")
        ax.grid(alpha=0.25)
    for ax in axes[-1, :]:
        ax.set_xlabel("history sequence length")
    for ax in axes[:, 0]:
        ax.set_ylabel("Q8xKV8 / BF16 throughput")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=9,
               bbox_to_anchor=(0.5, 0.975), fontsize=8)
    fig.suptitle("H800 sparse prefill Q8xKV8 throughput ratio", y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "q8kv8_vs_bf16_speedup")


def main():
    rows = build_comparison(load(OLD_RESULTS), load(Q8_RESULTS))
    write_csv(rows)
    plot_history(rows)
    plot_speedup_summary(rows)
    comparable = [r["q8kv8_vs_bf16_tflops"] for r in rows
                  if r["q8kv8_vs_bf16_tflops"] is not None]
    comparable.sort()
    print(f"comparable={len(comparable)} median_ratio="
          f"{comparable[len(comparable) // 2]:.3f}")


if __name__ == "__main__":
    main()
