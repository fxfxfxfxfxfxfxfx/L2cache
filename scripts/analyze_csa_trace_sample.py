#!/usr/bin/env python3
"""Summarize a stratified local sample of DeepSeek-V4 CSA top-k traces."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HISTORY_BINS = (
    (2048, 8192, "2K-8K"),
    (8192, 16384, "8K-16K"),
    (16384, 32768, "16K-32K"),
    (32768, 65536, "32K-64K"),
    (65536, 1 << 60, "64K+"),
)
LAYER_GROUPS = {
    "early": tuple(range(0, 7)),
    "middle": tuple(range(7, 14)),
    "late": tuple(range(14, 21)),
}
LAGS = (1, 4, 16, 64, 256)
AGE_WINDOWS = (128, 512, 2048, 8192, 32768)
BLUE = "#1769aa"
GREEN = "#238b57"


def history_bin(query_row):
    for lower, upper, label in HISTORY_BINS:
        if lower <= query_row < upper:
            return label
    return None


def phase_labels(metadata, row_count):
    labels = np.full(row_count, "unknown", dtype=object)
    for span in metadata["message_spans"]:
        begin, end = span["transition_rows"]
        phase = (
            "assistant" if span["phase_label"] == "decode_equivalent"
            else "prefill"
        )
        labels[max(0, begin):min(row_count, end + 1)] = phase
    return labels


def pair_overlap(left, right):
    """Set intersection / 512 for arrays ending in the top-k dimension."""
    combined = np.sort(np.concatenate((left, right), axis=-1), axis=-1)
    duplicate = (combined[..., 1:] == combined[..., :-1])
    duplicate &= combined[..., 1:] >= 0
    return duplicate.sum(axis=-1) / left.shape[-1]


def evenly_spaced(values, limit):
    if len(values) <= limit:
        return values
    positions = np.linspace(0, len(values) - 1, limit, dtype=np.int64)
    return values[positions]


def describe(values):
    data = np.asarray(values, dtype=np.float64)
    if not data.size:
        return {
            "count": 0, "mean": None, "p10": None, "p50": None,
            "p90": None,
        }
    return {
        "count": int(data.size), "mean": float(data.mean()),
        "p10": float(np.percentile(data, 10)),
        "p50": float(np.percentile(data, 50)),
        "p90": float(np.percentile(data, 90)),
    }


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["empty"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def add_group_values(store, prefix, values):
    for group, slots in LAYER_GROUPS.items():
        store[prefix + (group,)].extend(values[:, slots].reshape(-1).tolist())
    store[prefix + ("all",)].extend(values.reshape(-1).tolist())


def save_fig(fig, figures, name):
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / f"{name}.png", dpi=180, bbox_inches="tight")
    fig.savefig(figures / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", default="local_data/csa_trace_source"
    )
    parser.add_argument(
        "--output-dir", default="artifacts/data/csa_trace_profile"
    )
    parser.add_argument("--rows-per-stratum", type=int, default=128)
    return parser.parse_args()


def main():
    args = parse_args()
    source = Path(args.source)
    output = Path(args.output_dir)
    raw = output / "raw"
    figures = output / "figures"
    paths = sorted(source.rglob("indexer_topk.npz"))
    if not paths:
        raise FileNotFoundError(f"no indexer_topk.npz under {source}")

    overlaps = defaultdict(list)
    lags = defaultdict(list)
    recent = defaultdict(list)
    consecutive = defaultdict(list)
    layer_values = defaultdict(list)
    boundary_values = defaultdict(list)
    trace_rows = []

    for file_index, npz_path in enumerate(paths, start=1):
        metadata_path = npz_path.with_name("metadata.json")
        metadata = json.loads(metadata_path.read_text())
        with np.load(npz_path) as archive:
            tensor = archive["indexer_topk"]
        expected_shape = tuple(metadata["tensor"]["shape"])
        if tensor.shape != expected_shape or tensor.dtype != np.int32:
            raise ValueError(
                f"{npz_path}: expected int32{expected_shape}, got "
                f"{tensor.dtype}{tensor.shape}"
            )
        row_count, layer_count, topk = tensor.shape
        if layer_count != 21 or topk != 512:
            raise ValueError(f"unsupported CSA shape {tensor.shape}")
        phases = phase_labels(metadata, row_count)
        selected = []
        for lower, upper, label in HISTORY_BINS:
            upper = min(upper, row_count)
            if lower >= upper:
                continue
            for phase in ("assistant", "prefill"):
                candidates = np.flatnonzero(phases[lower:upper] == phase) + lower
                candidates = candidates[candidates > 0]
                selected.extend(evenly_spaced(candidates, args.rows_per_stratum))
        queries = np.unique(np.asarray(selected, dtype=np.int64))
        current = tensor[queries]
        previous = tensor[queries - 1]
        if np.any((current >= 0).sum(axis=-1) != 512):
            raise ValueError(f"{npz_path}: sampled rows are not full top-k")
        if np.any(current > (queries[:, None, None] // 4)):
            raise ValueError(f"{npz_path}: sampled compressed indices are non-causal")

        adjacent = pair_overlap(previous, current)
        sorted_current = np.sort(current, axis=-1)
        consecutive_fraction = (
            (np.diff(sorted_current, axis=-1) == 1).sum(axis=-1) / (topk - 1)
        )
        age_tokens = 4 * (queries[:, None, None] // 4 - current)

        for phase in ("assistant", "prefill"):
            phase_mask = phases[queries] == phase
            for _, _, label in HISTORY_BINS:
                bin_mask = np.asarray([history_bin(q) == label for q in queries])
                mask = phase_mask & bin_mask
                if not mask.any():
                    continue
                add_group_values(overlaps, (label, phase), adjacent[mask])
                add_group_values(
                    consecutive, (label, phase), consecutive_fraction[mask]
                )
                for window in AGE_WINDOWS:
                    fractions = (age_tokens[mask] <= window).mean(axis=-1)
                    add_group_values(recent, (label, phase, window), fractions)

        for slot in range(layer_count):
            layer_values[slot].extend(adjacent[:, slot].tolist())

        is_boundary = phases[queries] != phases[queries - 1]
        boundary_values["phase_boundary"].extend(adjacent[is_boundary].reshape(-1))
        boundary_values["phase_interior"].extend(adjacent[~is_boundary].reshape(-1))

        for lag in LAGS:
            mask = queries >= lag
            lag_overlap = pair_overlap(tensor[queries[mask] - lag], current[mask])
            for _, _, label in HISTORY_BINS:
                bin_mask = np.asarray([
                    history_bin(q) == label for q in queries[mask]
                ])
                if bin_mask.any():
                    lags[(label, lag)].extend(lag_overlap[bin_mask].reshape(-1))

        trace_rows.append({
            "row": metadata["source"]["row_index"],
            "instance_id": metadata["source"]["instance_id"],
            "prompt_tokens": metadata["request"]["prompt_tokens"],
            "sampled_query_rows": len(queries),
            "assistant_fraction": (
                metadata["token_boundaries"][
                    "assistant_decode_equivalent_row_count"
                ] / row_count
            ),
            "adjacent_overlap_mean": float(adjacent.mean()),
            "adjacent_overlap_p50": float(np.median(adjacent)),
            "adjacent_overlap_p10": float(np.percentile(adjacent, 10)),
            "adjacent_overlap_p90": float(np.percentile(adjacent, 90)),
            "consecutive_c4_fraction_mean": float(consecutive_fraction.mean()),
            "npz_bytes": npz_path.stat().st_size,
        })
        print(
            f"[{file_index}/{len(paths)}] row={trace_rows[-1]['row']:04d} "
            f"tokens={row_count + 1} samples={len(queries)} "
            f"adjacent_p50={trace_rows[-1]['adjacent_overlap_p50']:.3f}",
            flush=True,
        )
        del tensor, current, previous, sorted_current, age_tokens

    metric_rows = []
    for (label, phase, group), values in sorted(overlaps.items()):
        row = {"history_bin": label, "phase": phase, "layer_group": group}
        row.update({f"adjacent_overlap_{k}": v for k, v in describe(values).items()})
        cstats = describe(consecutive[(label, phase, group)])
        row.update({f"consecutive_c4_fraction_{k}": v for k, v in cstats.items()})
        for window in AGE_WINDOWS:
            rstats = describe(recent[(label, phase, window, group)])
            row[f"within_{window}_tokens_mean"] = rstats["mean"]
            row[f"within_{window}_tokens_p50"] = rstats["p50"]
        metric_rows.append(row)

    lag_rows = []
    for (label, lag), values in sorted(lags.items()):
        row = {"history_bin": label, "query_lag": lag}
        row.update({f"overlap_{k}": v for k, v in describe(values).items()})
        lag_rows.append(row)

    layer_rows = []
    for slot, values in sorted(layer_values.items()):
        row = {"csa_slot": slot, "hidden_layer_id": 2 * (slot + 1)}
        row.update({f"adjacent_overlap_{k}": v for k, v in describe(values).items()})
        layer_rows.append(row)

    boundary_rows = []
    for kind, values in boundary_values.items():
        row = {"location": kind}
        row.update({f"adjacent_overlap_{k}": v for k, v in describe(values).items()})
        boundary_rows.append(row)

    write_csv(raw / "sample_manifest.csv", trace_rows)
    write_csv(raw / "activation_by_condition.csv", metric_rows)
    write_csv(raw / "overlap_by_lag.csv", lag_rows)
    write_csv(raw / "overlap_by_layer.csv", layer_rows)
    write_csv(raw / "overlap_at_phase_boundary.csv", boundary_rows)

    all_phase_model = {}
    for _, _, label in HISTORY_BINS:
        values = []
        for phase in ("assistant", "prefill"):
            values.extend(overlaps[(label, phase, "all")])
        if not values:
            continue
        model_stats = describe(values)
        recent_cdf = {}
        for window in AGE_WINDOWS:
            window_values = []
            for phase in ("assistant", "prefill"):
                window_values.extend(recent[(label, phase, window, "all")])
            recent_cdf[str(window)] = describe(window_values)["mean"]
        all_phase_model[label] = {
            "adjacent_overlap": model_stats,
            "recent_token_cdf": recent_cdf,
        }
    model = {
        "source": str(source.resolve()),
        "trace_count": len(trace_rows),
        "sampled_rows_per_stratum": args.rows_per_stratum,
        "csa_topk": 512,
        "c4_tokens_per_entry": 4,
        "expanded_dsa_topk": 2048,
        "history_profiles": all_phase_model,
        "boundary": {row["location"]: row for row in boundary_rows},
    }
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "activation_model.json").write_text(json.dumps(model, indent=2))

    labels = [item[2] for item in HISTORY_BINS if item[2] in all_phase_model]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for phase, color, marker in (
        ("assistant", BLUE, "o"), ("prefill", GREEN, "^"),
    ):
        values = []
        for label in labels:
            values.append(describe(overlaps[(label, phase, "all")])["p50"])
        ax.plot(labels, np.asarray(values) * 100, marker=marker, linewidth=2,
                color=color, label=phase)
    ax.set_xlabel("Trace history (tokens)")
    ax.set_ylabel("Median adjacent-query overlap (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend()
    save_fig(fig, figures, "adjacent_overlap_by_history_phase")

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    styles = ("-", "--", ":", "-.", (0, (3, 1, 1, 1)))
    markers = ("o", "^", "s", "D", "v")
    for index, label in enumerate(labels):
        rows = [row for row in lag_rows if row["history_bin"] == label]
        ax.plot([row["query_lag"] for row in rows],
                [100 * row["overlap_p50"] for row in rows],
                color=(BLUE, GREEN)[index % 2], linestyle=styles[index],
                marker=markers[index], linewidth=2, label=label)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Query-row lag")
    ax.set_ylabel("Median selected-set overlap (%)")
    ax.grid(alpha=0.25)
    ax.legend(title="History")
    save_fig(fig, figures, "overlap_decay_by_query_lag")

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot([row["hidden_layer_id"] for row in layer_rows],
            [100 * row["adjacent_overlap_p50"] for row in layer_rows],
            marker="o", color=BLUE)
    ax.set_xlabel("Hidden layer id")
    ax.set_ylabel("Median adjacent-query overlap (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    save_fig(fig, figures, "adjacent_overlap_by_layer")

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    for window, marker, color, linestyle in (
        (512, "o", BLUE, "-"),
        (2048, "s", GREEN, "--"),
        (8192, "^", BLUE, ":"),
    ):
        values = []
        for label in labels:
            pool = []
            for phase in ("assistant", "prefill"):
                pool.extend(recent[(label, phase, window, "all")])
            values.append(describe(pool)["mean"])
        ax.plot(labels, np.asarray(values) * 100, marker=marker,
                color=color, linestyle=linestyle, linewidth=2,
                label=f"age <= {window} tokens")
    ax.set_xlabel("Trace history (tokens)")
    ax.set_ylabel("Mean selected-entry fraction (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.25)
    ax.legend()
    save_fig(fig, figures, "recent_kv_fraction_by_history")

    print(f"wrote trace statistics for {len(trace_rows)} rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
