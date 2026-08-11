#!/usr/bin/env python3
"""Generate figures and the platform summary table from assets/raw/results.jsonl.

Outputs (under assets/figures and assets/raw):
- history_scaling_q*.{png,pdf}    : decode/prefill 3x3 batch panels vs history
- prefill_tflops_vs_history_*.{png,pdf}: fixed (B,Q), TFLOPS vs history
- tflops_vs_batch_q1_h*.{png,pdf}: decode batch scaling per fixed history
- tflops_vs_batch_q1_overview.{png,pdf}: decode fixed-history overview
- sparse_decode_latency_b*.{png,pdf}: one latency-vs-history plot per batch
- decode_echo_32k_64k.{png,pdf}   : decode TFLOPS vs batch at H=32K/64K
- prefill_heatmap.{png,pdf}       : prefill B x Q TFLOPS, faceted by history
- platform_comparison_table.{png,pdf}: batch, latency, and dense/sparse work
- platform_summary.csv            : per (kind, backend, cache_state) maxima,
                                    observed utilization, and 90%/95% min-batch
                                    (censored=true when a curve never reaches it)
"""

import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(PROJECT, "assets", "raw")
FIG = os.path.join(PROJECT, "assets", "figures")
H100_REF_TFLOPS = 989.5  # reference denominator only, NOT an H800 spec
SCALING_BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
SCALING_HISTORIES = [2048, 4096, 8192, 16384, 32768, 65536,
                     131072, 262144, 524288]
SCALING_QUERIES = [
    1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
    1024, 2048, 4096, 8192, 16384, 32768,
]
FIGURE2_METRICS = os.path.join(RAW, "figure2_metrics.jsonl")


def load_rows(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ok_rows(rows):
    return [r for r in rows if r.get("status") == "ok"
            and not r["case_id"].startswith("anchor")]


def latest_per_case(rows):
    """Keep the most recent record per case_id."""
    out = {}
    for r in rows:
        out[r["case_id"]] = r
    return list(out.values())


def merge_figure2_metrics(rows):
    """Attach separately-profiled kernel timing without losing E2E metrics."""
    if not os.path.exists(FIGURE2_METRICS):
        return rows
    metrics = {r["case_id"]: r for r in load_rows(FIGURE2_METRICS)
               if r.get("status") == "ok"}
    merged = []
    for row in rows:
        metric = metrics.get(row["case_id"])
        if metric:
            row = dict(row)
            row.update({key: value for key, value in metric.items()
                        if key not in ("status", "kind", "backend", "B", "H",
                                       "Q", "cache_state")})
        merged.append(row)
    return merged


def performance_tflops(row):
    """ECHO Figure 2 metric, with legacy fallback for non-canonical rows."""
    return row.get("figure2_hardware_tflops", row["tflops"])


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=150,
                bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {name}.png/.pdf")


def _history_label(value):
    return f"{value // 1024}K"


def _query_label(value):
    if value == 1:
        return "1 (decode)"
    if value < 1024:
        return f"{value} (chunked prefill)"
    return f"{value // 1024}K (chunked prefill)"


def _target_backend_roles(Q):
    if Q == 1:
        return {
            "flashmla_dense_bf16": "dense",
            "flashmla_sparse_fp8": "sparse",
        }
    return {
        "flashmla_dense_bf16_multiquery": "dense",
        "flashmla_sparse_bf16": "sparse",
    }


def write_history_scaling_results(rows):
    """Export the latest record for every requested dense/sparse grid point."""
    target_backends = {
        1: {"flashmla_dense_bf16", "flashmla_sparse_fp8"},
        **{q: {"flashmla_dense_bf16_multiquery", "flashmla_sparse_bf16"}
           for q in SCALING_QUERIES if q != 1},
    }
    target_records = [
        r for r in rows
        if r.get("cache_state") == "steady"
        and not r.get("case_id", "").startswith("anchor")
        and r.get("B") in SCALING_BATCHES
        and r.get("H") in SCALING_HISTORIES
        and r.get("Q") in target_backends
        and r.get("backend") in target_backends[r["Q"]]
    ]
    target_records.sort(key=lambda r: (r["Q"], r["B"], r["H"],
                                       r["backend"]))
    jsonl_path = os.path.join(RAW, "history_scaling_results.jsonl")
    with open(jsonl_path, "w") as f:
        for row in target_records:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    csv_path = os.path.join(RAW, "history_scaling_results.csv")
    fields = sorted({key for row in target_records for key in row})
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(target_records)
    print(f"wrote {jsonl_path}/{os.path.basename(csv_path)} "
          f"({len(target_records)} rows)")
    return target_records


def plot_tflops_vs_batch_fixed_context(all_rows):
    """Decode TFLOPS: batch on x, with history fixed in every plot."""
    rows = [r for r in all_rows
            if r.get("cache_state") == "steady"
            and r.get("B") in SCALING_BATCHES
            and r.get("H") in SCALING_HISTORIES
            and r.get("Q") in SCALING_QUERIES
            and (r.get("status") != "ok"
                 or "figure2_hardware_tflops" in r)]
    colors = {"dense": "#1769aa", "sparse": "#d1495b"}
    markers = {"dense": "o", "sparse": "s"}
    output = []

    for Q in [1]:
        backend_roles = _target_backend_roles(Q)
        kind = "decode" if Q == 1 else "prefill"
        q_rows = [r for r in rows if r.get("Q") == Q
                  and r.get("kind") == kind
                  and r.get("backend") in backend_roles]
        successful = [r for r in q_rows if r.get("status") == "ok"]
        missing_metrics = [r["case_id"] for r in successful
                           if "figure2_hardware_tflops" not in r]
        if missing_metrics:
            raise RuntimeError(
                f"{len(missing_metrics)} successful cases lack Figure 2 metrics")
        q_ymax = max((performance_tflops(r) for r in successful), default=1.0)

        overview, axes = plt.subplots(3, 3, figsize=(14.5, 10.5),
                                      sharex=True, sharey=True)
        for overview_ax, H in zip(axes.flat, SCALING_HISTORIES):
            shape_rows = [r for r in q_rows if r["H"] == H]
            individual, ax = plt.subplots(figsize=(7.4, 4.5))

            for backend, role in backend_roles.items():
                by_b = {}
                for row in shape_rows:
                    if row["backend"] != backend:
                        continue
                    current = by_b.get(row["B"])
                    if current is None or (
                            row.get("status") == "ok"
                            and current.get("status") != "ok"):
                        by_b[row["B"]] = row
                ok = [by_b[b] for b in SCALING_BATCHES
                      if b in by_b and by_b[b].get("status") == "ok"]
                if ok:
                    batches = [r["B"] for r in ok]
                    values = [performance_tflops(r) for r in ok]
                    for target_ax in (ax, overview_ax):
                        target_ax.plot(
                            batches, values, color=colors[role],
                            marker=markers[role], linewidth=2.0, markersize=5,
                            label=role)

                skipped = [b for b in SCALING_BATCHES
                           if b in by_b and by_b[b].get("status") != "ok"]
                if skipped:
                    y_fraction = 0.96 if role == "dense" else 0.90
                    for target_ax in (ax, overview_ax):
                        target_ax.scatter(
                            skipped, [y_fraction] * len(skipped), marker="x",
                            s=34, linewidths=1.3, color=colors[role],
                            transform=target_ax.get_xaxis_transform(),
                            clip_on=False)

                for B in SCALING_BATCHES:
                    row = by_b.get(B)
                    output.append({
                        "Q": Q, "H": H, "B": B, "kind": kind,
                        "role": role, "backend": backend,
                        "status": row.get("status") if row else "missing",
                        "skip_reason": row.get("skip_reason") if row else "",
                        "figure2_hardware_tflops": (
                            row.get("figure2_hardware_tflops") if row else ""),
                        "figure2_latency_us": (
                            row.get("figure2_latency_us") if row else ""),
                    })

            for target_ax in (ax, overview_ax):
                target_ax.set_xscale("log", base=2)
                target_ax.set_xticks(SCALING_BATCHES,
                                     [str(b) for b in SCALING_BATCHES])
                target_ax.set_xlim(SCALING_BATCHES[0] / 2 ** 0.35,
                                    SCALING_BATCHES[-1] * 2 ** 0.35)
                target_ax.set_ylim(0, q_ymax * 1.10)
                target_ax.grid(alpha=0.25)

            ax.set_xlabel("batch size")
            ax.set_ylabel("hardware utilization (TFLOPS)")
            ax.set_title(
                f"H800 MLA - H={_history_label(H)}, Q={_query_label(Q)}")
            handles, labels = ax.get_legend_handles_labels()
            handles.append(plt.Line2D([], [], color="#666666", marker="x",
                                      linestyle="None", label="OOM / skipped"))
            labels.append("OOM / skipped")
            ax.legend(handles, labels, fontsize=8, loc="best")
            individual.tight_layout()
            save(individual, f"tflops_vs_batch_q{Q}_h{H}")

            overview_ax.set_title(f"history={_history_label(H)}", fontsize=10)

        for ax in axes[-1, :]:
            ax.set_xlabel("batch size")
        for ax in axes[:, 0]:
            ax.set_ylabel("hardware utilization (TFLOPS)")
        handles = [
            plt.Line2D([], [], color=colors["dense"], marker=markers["dense"],
                       label="dense"),
            plt.Line2D([], [], color=colors["sparse"], marker=markers["sparse"],
                       label="sparse"),
            plt.Line2D([], [], color="#666666", marker="x",
                       linestyle="None", label="OOM / skipped"),
        ]
        overview.legend(handles=handles, loc="upper center", ncol=3,
                        bbox_to_anchor=(0.5, 0.965))
        overview.suptitle(
            f"H800 MLA TFLOPS vs batch - Q={_query_label(Q)} - ECHO Fig. 2 convention",
            y=0.995)
        overview.tight_layout(rect=(0, 0, 1, 0.95))
        save(overview, f"tflops_vs_batch_q{Q}_overview")

    path = os.path.join(RAW, "decode_tflops_vs_batch.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(f"wrote {path} ({len(output)} rows)")


def plot_sparse_decode_latency_by_batch(all_rows):
    """One sparse decode latency-vs-history figure for every requested batch."""
    rows = [r for r in all_rows
            if r.get("kind") == "decode"
            and r.get("Q") == 1
            and r.get("backend") == "flashmla_sparse_fp8"
            and r.get("cache_state") == "steady"
            and r.get("B") in SCALING_BATCHES
            and r.get("H") in SCALING_HISTORIES]
    output = []

    for B in SCALING_BATCHES:
        by_h = {r["H"]: r for r in rows if r["B"] == B}
        ok = [by_h[h] for h in SCALING_HISTORIES
              if h in by_h and by_h[h].get("status") == "ok"]
        if not ok:
            continue

        histories = [r["H"] for r in ok]
        median_us = [r["latency_ms_median"] * 1000 for r in ok]
        p5_us = [r["latency_ms_p5"] * 1000 for r in ok]
        p95_us = [r["latency_ms_p95"] * 1000 for r in ok]

        fig, ax = plt.subplots(figsize=(7.4, 4.5))
        ax.fill_between(histories, p5_us, p95_us, color="#d1495b",
                        alpha=0.18, label="p5-p95")
        ax.plot(histories, median_us, color="#b12f40", marker="s",
                linewidth=2.0, markersize=5, label="median")

        skipped = [h for h in SCALING_HISTORIES
                   if h in by_h and by_h[h].get("status") != "ok"]
        if skipped:
            first = min(skipped)
            previous = max((h for h in histories if h < first), default=first / 2)
            left = (previous * first) ** 0.5
            ax.axvspan(left, SCALING_HISTORIES[-1] * 2 ** 0.5,
                       color="#777777", alpha=0.09)
            ax.scatter(skipped, [0.96] * len(skipped), marker="x", s=36,
                       color="#666666", transform=ax.get_xaxis_transform(),
                       clip_on=False, label="OOM / skipped")
            ax.text(first, 0.88, "OOM / skipped",
                    transform=ax.get_xaxis_transform(), color="#555555",
                    fontsize=8, ha="center")

        change = (median_us[-1] / median_us[0] - 1) * 100
        ax.text(0.02, 0.94,
                f"2K to {_history_label(histories[-1])}: {change:+.1f}%",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox={"facecolor": "white", "edgecolor": "#cccccc",
                      "alpha": 0.9, "pad": 3})
        ax.set_xscale("log", base=2)
        ax.set_xticks(SCALING_HISTORIES,
                      [_history_label(h) for h in SCALING_HISTORIES],
                      rotation=45)
        ax.set_xlim(SCALING_HISTORIES[0] / 2 ** 0.35,
                    SCALING_HISTORIES[-1] * 2 ** 0.35)
        ax.set_xlabel("history KV length")
        ax.set_ylabel("single decode latency (us)")
        ax.set_title(f"H800 FlashMLA sparse FP8 decode latency - batch={B}")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        save(fig, f"sparse_decode_latency_b{B}")

        baseline = median_us[0]
        for H in SCALING_HISTORIES:
            row = by_h.get(H)
            result = {
                "B": B, "H": H,
                "status": row.get("status") if row else "missing",
                "skip_reason": row.get("skip_reason") if row else "",
                "latency_us_median": "", "latency_us_p5": "",
                "latency_us_p95": "", "latency_vs_2k": "",
            }
            if row and row.get("status") == "ok":
                med = row["latency_ms_median"] * 1000
                result.update(
                    latency_us_median=med,
                    latency_us_p5=row["latency_ms_p5"] * 1000,
                    latency_us_p95=row["latency_ms_p95"] * 1000,
                    latency_vs_2k=med / baseline,
                )
            output.append(result)

    path = os.path.join(RAW, "sparse_decode_latency_vs_history.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(f"wrote {path} ({len(output)} rows)")


def plot_history_scaling_grids(all_rows):
    """Decode/prefill overview: fixed Q, batch panels, history on x."""
    rows = [r for r in all_rows if r.get("cache_state") == "steady"
            and not r.get("case_id", "").startswith("anchor")]
    colors = {"dense": "#1769aa", "sparse": "#d1495b"}
    coverage = []

    for Q in SCALING_QUERIES:
        kind = "decode" if Q == 1 else "prefill"
        backend_role = _target_backend_roles(Q)
        fig, axes = plt.subplots(3, 3, figsize=(14.5, 10.5), sharex=True,
                                 sharey=True)
        figure_ok = [r for r in rows if r.get("kind") == kind
                     and r.get("Q") == Q and r.get("B") in SCALING_BATCHES
                     and r.get("H") in SCALING_HISTORIES
                     and r.get("backend") in backend_role
                     and r.get("status") == "ok"]
        missing_metrics = [r["case_id"] for r in figure_ok
                           if "figure2_hardware_tflops" not in r]
        if missing_metrics:
            raise RuntimeError(
                f"{len(missing_metrics)} successful cases lack Figure 2 metrics")
        ymax = max((performance_tflops(r) for r in figure_ok), default=1.0)
        for ax, B in zip(axes.flat, SCALING_BATCHES):
            panel = [r for r in rows if r.get("kind") == kind
                     and r.get("B") == B and r.get("Q") == Q
                     and r.get("H") in SCALING_HISTORIES
                     and r.get("backend") in backend_role]
            for backend, role in backend_role.items():
                by_h = {r["H"]: r for r in panel if r["backend"] == backend}
                ok = [by_h[h] for h in SCALING_HISTORIES
                      if h in by_h and by_h[h].get("status") == "ok"]
                if ok:
                    ax.plot([r["H"] for r in ok],
                            [performance_tflops(r) for r in ok],
                            marker="o" if role == "dense" else "s",
                            color=colors[role], linewidth=1.8, markersize=4,
                            label=role)
                for H in SCALING_HISTORIES:
                    row = by_h.get(H)
                    coverage.append({
                        "B": B, "Q": Q, "kind": kind, "H": H,
                        "role": role, "backend": backend,
                        "status": row.get("status") if row else "missing",
                        "skip_reason": row.get("skip_reason") if row else "",
                        "figure2_hardware_tflops": (
                            row.get("figure2_hardware_tflops") if row else ""),
                        "effective_e2e_tflops": (
                            row.get("tflops") if row else ""),
                        "figure2_latency_us": (
                            row.get("figure2_latency_us") if row else ""),
                        "latency_ms_median": (
                            row.get("latency_ms_median") if row else ""),
                    })
            skipped_h = sorted({r["H"] for r in panel
                                if r.get("status") != "ok"})
            if skipped_h:
                ax.scatter(skipped_h, [0] * len(skipped_h), marker="x",
                           s=30, linewidths=1.3, color="#666666")
            ax.set_title(f"batch={B}", fontsize=10)
            ax.set_xscale("log", base=2)
            ax.set_xticks(SCALING_HISTORIES,
                          [_history_label(h) for h in SCALING_HISTORIES],
                          rotation=45, fontsize=7)
            ax.grid(alpha=0.25)
        axes[0, 0].set_ylim(0, ymax * 1.08)
        for ax in axes[-1, :]:
            ax.set_xlabel("history sequence length")
        for ax in axes[:, 0]:
            ax.set_ylabel("hardware utilization (TFLOPS)")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        if not handles:
            handles = [
                plt.Line2D([], [], color=colors["dense"], marker="o",
                           label="dense"),
                plt.Line2D([], [], color=colors["sparse"], marker="s",
                           label="sparse"),
            ]
            labels = ["dense", "sparse"]
        handles.append(plt.Line2D([], [], color="#666666", marker="x",
                                  linestyle="None", label="OOM / skipped"))
        labels.append("OOM / skipped")
        fig.legend(handles, labels, loc="upper center", ncol=3,
                   bbox_to_anchor=(0.5, 0.965))
        q_title = _query_label(Q)
        fig.suptitle(
            f"H800 MLA history scaling · new length={q_title} · ECHO Fig. 2 convention",
            y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        save(fig, f"history_scaling_q{Q}")

    path = os.path.join(RAW, "history_scaling_coverage.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(coverage[0]))
        writer.writeheader()
        writer.writerows(coverage)
    print(f"wrote {path} ({len(coverage)} rows)")


def _plot_prefill_axis(ax, panel, x_values, x_key, y_max=None):
    backend_roles = _target_backend_roles(1024)
    colors = {"dense": "#1769aa", "sparse": "#d1495b"}
    markers = {"dense": "o", "sparse": "s"}
    for backend, role in backend_roles.items():
        by_x = {r[x_key]: r for r in panel if r.get("backend") == backend}
        ok = [by_x[x] for x in x_values
              if x in by_x and by_x[x].get("status") == "ok"]
        if ok:
            ax.plot([r[x_key] for r in ok],
                    [performance_tflops(r) for r in ok],
                    color=colors[role], marker=markers[role], linewidth=2.0,
                    markersize=5, label=role)
        skipped = [x for x in x_values
                   if x in by_x and by_x[x].get("status") != "ok"]
        if skipped:
            ax.scatter(skipped, [0.96 if role == "dense" else 0.90]
                       * len(skipped), marker="x", s=34, linewidths=1.3,
                       color=colors[role], transform=ax.get_xaxis_transform(),
                       clip_on=False)
    ax.set_xscale("log", base=2)
    ax.set_xticks(x_values, [_history_label(x) for x in x_values], rotation=45)
    ax.set_xlim(x_values[0] / 2 ** 0.35, x_values[-1] * 2 ** 0.35)
    if y_max is not None:
        ax.set_ylim(0, y_max * 1.10)
    ax.grid(alpha=0.25)


def _prefill_legend_handles():
    return [
        plt.Line2D([], [], color="#1769aa", marker="o", label="dense"),
        plt.Line2D([], [], color="#d1495b", marker="s", label="sparse"),
        plt.Line2D([], [], color="#666666", marker="x", linestyle="None",
                   label="OOM / skipped"),
    ]


def plot_prefill_history_individual(all_rows):
    """One figure per fixed (B,Q), with history on the x-axis."""
    rows = [r for r in all_rows
            if r.get("kind") == "prefill"
            and r.get("cache_state") == "steady"
            and r.get("B") in SCALING_BATCHES
            and r.get("H") in SCALING_HISTORIES
            and r.get("Q") in SCALING_QUERIES[1:]
            and not r.get("case_id", "").startswith("anchor")
            and r.get("backend") in _target_backend_roles(1024)]
    for Q in SCALING_QUERIES[1:]:
        for B in SCALING_BATCHES:
            panel = [r for r in rows if r["Q"] == Q and r["B"] == B]
            successful = [r for r in panel if r.get("status") == "ok"]
            y_max = max((performance_tflops(r) for r in successful), default=1.0)
            fig, ax = plt.subplots(figsize=(7.4, 4.5))
            _plot_prefill_axis(ax, panel, SCALING_HISTORIES, "H", y_max)
            ax.set_xlabel("history sequence length")
            ax.set_ylabel("hardware utilization (TFLOPS)")
            ax.set_title(
                f"H800 MLA prefill - batch={B}, new length={_query_label(Q)}")
            ax.legend(handles=_prefill_legend_handles(), fontsize=8,
                      loc="best")
            fig.tight_layout()
            save(fig, f"prefill_tflops_vs_history_b{B}_q{Q}")


BACKEND_STYLE = {
    "flashmla_dense_bf16": ("o-", "dense (FlashMLA BF16)"),
    "flashmla_sparse_fp8": ("s-", "sparse (FlashMLA FP8)"),
    "flashmla_sparse_fp8_fulltopk": ("^--", "full-topk (FlashMLA FP8)"),
    "flashinfer_fa3_dense_bf16": ("o-", "dense (FlashInfer FA3)"),
    "flashmla_dense_bf16_multiquery":
        ("o-", "dense (FlashMLA multi-query decode)"),
    "flashmla_sparse_bf16": ("s-", "sparse (FlashMLA BF16)"),
    "flashmla_sparse_bf16_fulltopk": ("^--", "full-topk (FlashMLA BF16)"),
}


def plot_decode_echo(rows):
    dec = [r for r in rows if r["kind"] == "decode"
           and r["cache_state"] == "steady"
           and "figure2_hardware_tflops" in r]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, H in zip(axes, [32768, 65536]):
        for backend, (style, label) in BACKEND_STYLE.items():
            pts = sorted(((r["B"], performance_tflops(r)) for r in dec
                          if r["H"] == H and r["backend"] == backend))
            if pts:
                ax.plot([p[0] for p in pts], [p[1] for p in pts], style,
                        label=label, markersize=4)
        ax.set_title(f"decode H={H // 1024}K (steady)")
        ax.set_xlabel("batch size")
        ax.set_xscale("log", base=2)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("hardware utilization (TFLOPS)")
    axes[1].legend(fontsize=8)
    fig.suptitle("H800 Sparse MLA decode — ECHO-style batch scaling")
    save(fig, "decode_echo_32k_64k")


def plot_prefill_heatmap(rows):
    pre = [r for r in rows if r["kind"] == "prefill"
           and r["cache_state"] == "steady"
           and "figure2_hardware_tflops" in r]
    histories = sorted({r["H"] for r in pre})
    backends = [b for b in ("flashmla_dense_bf16_multiquery",
                            "flashmla_sparse_bf16")
                if any(r["backend"] == b for r in pre)]
    if not backends or not histories:
        return
    fig, axes = plt.subplots(len(histories), len(backends), squeeze=False,
                             figsize=(4.9 * len(backends),
                                      3.2 * len(histories)))
    for row, H in enumerate(histories):
      for col, backend in enumerate(backends):
        ax = axes[row][col]
        pts = [r for r in pre if r["backend"] == backend and r["H"] == H]
        batches = sorted({r["B"] for r in pts}, reverse=True)
        queries = sorted({r["Q"] for r in pts})
        b_idx = {b: i for i, b in enumerate(batches)}
        q_idx = {q: i for i, q in enumerate(queries)}
        import numpy as np
        grid = np.full((len(batches), len(queries)), float("nan"))
        for r in pts:
            grid[b_idx[r["B"]], q_idx[r["Q"]]] = performance_tflops(r)
        im = ax.imshow(grid, aspect="auto", cmap="viridis", vmin=0)
        ax.set_xticks(range(len(queries)), queries, rotation=45, fontsize=8)
        ax.set_yticks(range(len(batches)), batches, fontsize=8)
        ax.set_xlabel("new length Q")
        ax.set_ylabel("batch B")
        ax.set_title(f"H={H // 1024}K · {BACKEND_STYLE[backend][1]}",
                     fontsize=9)
        for i in range(len(batches)):
            for j in range(len(queries)):
                v = grid[i, j]
                if v == v:
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                            fontsize=7, color="w")
        fig.colorbar(im, ax=ax, label="hardware utilization (TFLOPS)")
    fig.suptitle(
        "H800 Sparse MLA prefill — B × Q, ECHO Fig. 2 convention (steady)")
    fig.tight_layout(rect=(0, 0, 1, 0.99), h_pad=2.4, w_pad=1.2)
    save(fig, "prefill_heatmap")


def platform_summary(rows):
    """Per (kind, backend, cache_state) maxima + 90%/95% min-batch analysis."""
    groups = {}
    for r in rows:
        key = (r["kind"], r["backend"], r["cache_state"])
        groups.setdefault(key, []).append(r)

    out = []
    for (kind, backend, cs), rs in sorted(groups.items()):
        max_tflops = max(performance_tflops(r) for r in rs)
        max_gbps = max(r["est_logical_gbps"] for r in rs)
        out.append({
            "kind": kind, "backend": backend, "cache_state": cs,
            "platform_max_tflops": f"{max_tflops:.3f}",
            "platform_max_est_logical_gbps": f"{max_gbps:.1f}",
            "utilization_vs_h100_sxm_bf16_dense_peak":
                f"{max_tflops / H100_REF_TFLOPS:.4f}",
            "curve_H": "", "threshold": "platform_max",
            "min_batch": "", "censored": "",
            "observed_utilization_vs_platform_max": "1.0000",
        })
        if kind != "decode":
            continue
        # Per-history batch curves: smallest B reaching 90%/95% of the
        # backend's platform maximum TFLOPS; censored if never reached.
        for H in sorted({r["H"] for r in rs}):
            curve = sorted((r["B"], performance_tflops(r)) for r in rs
                           if r["H"] == H)
            for thr in (0.90, 0.95):
                target = thr * max_tflops
                hit = next((b for b, perf in curve if perf >= target), None)
                out.append({
                    "kind": kind, "backend": backend, "cache_state": cs,
                    "platform_max_tflops": f"{max_tflops:.3f}",
                    "platform_max_est_logical_gbps": f"{max_gbps:.1f}",
                    "utilization_vs_h100_sxm_bf16_dense_peak":
                        f"{max_tflops / H100_REF_TFLOPS:.4f}",
                    "curve_H": H, "threshold": f"{thr:.2f}",
                    "min_batch": hit if hit is not None else "",
                    "censored": "false" if hit is not None else "true",
                    "observed_utilization_vs_platform_max": f"{thr:.4f}",
                })
    path = os.path.join(RAW, "platform_summary.csv")
    fields = ["kind", "backend", "cache_state", "platform_max_tflops",
              "platform_max_est_logical_gbps",
              "utilization_vs_h100_sxm_bf16_dense_peak",
              "curve_H", "threshold", "min_batch", "censored",
              "observed_utilization_vs_platform_max"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out)
    print(f"wrote {path} ({len(out)} rows)")

    # observed utilization per row group -> also emit per-case observed util csv
    util_path = os.path.join(RAW, "observed_utilization.csv")
    with open(util_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "kind", "backend", "cache_state",
                    "figure2_hardware_tflops",
                    "observed_utilization_vs_platform_max"])
        for (kind, backend, cs), rs in sorted(groups.items()):
            mx = max(performance_tflops(r) for r in rs)
            for r in sorted(rs, key=lambda x: x["case_id"]):
                w.writerow([r["case_id"], kind, backend, cs,
                            f"{performance_tflops(r):.3f}",
                            f"{performance_tflops(r) / mx:.4f}"])
    print(f"wrote {util_path}")
    return out


def dense_sparse_comparison(rows):
    """Join dense/sparse cases and emit latency and effective-work ratios."""
    by_shape = {}
    for r in rows:
        if r["backend"].endswith("fulltopk"):
            continue
        key = (r["kind"], r["cache_state"], r["B"], r["H"], r["Q"])
        by_shape.setdefault(key, {})[r["backend"]] = r
    output = []
    pairs = {
        "decode": ("flashmla_dense_bf16", "flashmla_sparse_fp8"),
        "prefill": ("flashmla_dense_bf16_multiquery",
                    "flashmla_sparse_bf16"),
    }
    for key, values in sorted(by_shape.items()):
        kind, cache_state, B, H, Q = key
        dense_name, sparse_name = pairs[kind]
        dense, sparse = values.get(dense_name), values.get(sparse_name)
        if not dense or not sparse:
            continue
        output.append({
            "kind": kind, "cache_state": cache_state, "B": B, "H": H, "Q": Q,
            "dense_backend": dense_name, "sparse_backend": sparse_name,
            "dense_latency_ms": dense["latency_ms_median"],
            "sparse_latency_ms": sparse["latency_ms_median"],
            "dense_over_sparse_latency":
                dense["latency_ms_median"] / sparse["latency_ms_median"],
            "dense_pairs": dense["pairs"], "sparse_pairs": sparse["pairs"],
            "dense_over_sparse_effective_work": dense["pairs"] / sparse["pairs"],
            "dense_figure2_hardware_tflops": performance_tflops(dense),
            "sparse_figure2_hardware_tflops": performance_tflops(sparse),
            "dense_effective_e2e_tflops": dense["tflops"],
            "sparse_effective_e2e_tflops": sparse["tflops"],
        })
    path = os.path.join(RAW, "dense_sparse_comparison.csv")
    if output:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(output[0]))
            writer.writeheader()
            writer.writerows(output)
    print(f"wrote {path} ({len(output)} rows)")
    return output


def plot_comparison_table(summary, comparisons):
    """Render representative platform batch/latency/work comparisons."""
    reps = [r for r in comparisons if r["cache_state"] == "steady"
            and r["H"] in (32768, 65536)
            and ((r["kind"] == "decode" and r["B"] in (1, 32, 128))
                 or (r["kind"] == "prefill"
                     and (r["B"], r["Q"]) in ((1, 1024), (8, 256), (64, 16))))]
    reps = reps[:18]
    columns = ["kind", "H", "B", "Q", "dense_latency_ms",
               "sparse_latency_ms", "dense_over_sparse_latency",
               "dense_over_sparse_effective_work"]
    labels = ["stage", "H", "B", "Q", "dense ms", "sparse ms",
              "latency ratio", "work ratio"]
    cells = []
    for row in reps:
        cells.append([
            row["kind"], f"{row['H'] // 1024}K", row["B"], row["Q"],
            f"{row['dense_latency_ms']:.3f}", f"{row['sparse_latency_ms']:.3f}",
            f"{row['dense_over_sparse_latency']:.2f}x",
            f"{row['dense_over_sparse_effective_work']:.2f}x",
        ])
    if not cells:
        return
    fig, ax = plt.subplots(figsize=(11, 0.42 * len(cells) + 1.5))
    ax.axis("off")
    table = ax.table(cellText=cells, colLabels=labels, loc="center",
                     cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.25)
    ax.set_title("H800 dense/sparse batch, latency, and effective work")
    save(fig, "platform_comparison_table")


def main():
    jsonl = os.path.join(RAW, "results.jsonl")
    if not os.path.exists(jsonl):
        print(f"no results at {jsonl}", file=sys.stderr)
        return 1
    latest_rows = latest_per_case(load_rows(jsonl))
    write_history_scaling_results(latest_rows)
    all_rows = merge_figure2_metrics(latest_rows)
    rows = ok_rows(all_rows)
    print(f"{len(rows)} successful case records")
    if not rows:
        return 1
    plot_sparse_decode_latency_by_batch(all_rows)
    plot_tflops_vs_batch_fixed_context(all_rows)
    plot_history_scaling_grids(all_rows)
    plot_prefill_history_individual(all_rows)
    plot_decode_echo(rows)
    plot_prefill_heatmap(rows)
    summary = platform_summary(rows)
    comparisons = dense_sparse_comparison(rows)
    plot_comparison_table(summary, comparisons)
    return 0


if __name__ == "__main__":
    sys.exit(main())
