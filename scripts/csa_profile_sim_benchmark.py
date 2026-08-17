#!/usr/bin/env python3
"""Benchmark a CSA-statistics-driven index simulator over the full prefill grid."""

import argparse
import csv
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from scripts import benchmark as bm
from scripts import prefill_runtime as runtime


DEFAULT_BATCHES = (1, 2, 4, 8, 16, 32, 64, 128)
DEFAULT_HISTORIES = (2048, 4096, 8192, 16384, 32768, 65536,
                     131072, 262144, 524288)
DEFAULT_QUERIES = (2, 8, 32, 128, 512, 1024, 4096)
PASS_ORDERS = ("ascending", "descending")
PROFILE_BINS = (
    (0, 8192, "2K-8K"),
    (8192, 16384, "8K-16K"),
    (16384, 32768, "16K-32K"),
    (32768, 65536, "32K-64K"),
    (65536, 1 << 60, "64K+"),
)
AGE_WINDOWS = (128, 512, 2048, 8192, 32768)
FIELDS = [
    "case_id", "status", "skip_reason", "stage", "pattern", "pass_order",
    "cache_state", "row_layout", "B", "H", "Q", "topk", "warmup",
    "repeat", "seed",
    "latency_ms_median", "latency_ms_p5", "latency_ms_p95",
    "latency_ms_mean", "latency_ms_std", "tflops", "pairs", "flops",
    "target_adjacent_overlap", "achieved_adjacent_overlap",
    "target_c4_consecutive_fraction", "achieved_c4_consecutive_fraction",
    "unique_kv", "reuse_factor", "selected_kv_working_set_bytes",
    "selected_kv_working_set_vs_l2", "profile_bins", "extrapolated_64k_plus",
    "setup_alloc_ms", "setup_quant_ms", "setup_indices_ms",
    "peak_mem_alloc_bytes", "temp_c_before", "power_w_before",
    "sm_clock_mhz_before", "mem_clock_mhz_before", "temp_c_after",
    "power_w_after", "sm_clock_mhz_after", "mem_clock_mhz_after",
    "timestamp",
]


class Writer:
    def __init__(self, raw_dir):
        raw_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl = raw_dir / "results.jsonl"
        self.csv = raw_dir / "results.csv"
        if not self.csv.exists():
            with self.csv.open("w", newline="") as handle:
                csv.DictWriter(handle, fieldnames=FIELDS).writeheader()

    def terminal_case_ids(self):
        if not self.jsonl.exists():
            return set()
        terminal = set()
        with self.jsonl.open() as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if row.get("status") in ("ok", "skipped_memory_limit"):
                        terminal.add(row["case_id"])
        return terminal

    def append(self, row):
        with self.jsonl.open("a") as handle:
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        with self.csv.open("a", newline="") as handle:
            csv.DictWriter(
                handle, fieldnames=FIELDS, extrasaction="ignore"
            ).writerow(row)
            handle.flush()
            os.fsync(handle.fileno())


def parse_int_list(value, default):
    if not value:
        return tuple(default)
    values = tuple(sorted({int(item) for item in value.split(",") if item}))
    if not values or any(number <= 0 for number in values):
        raise ValueError("integer lists must contain positive values")
    return values


def parse_shape_triples(value):
    if not value:
        return None
    shapes = set()
    for item in value.split(","):
        parts = item.strip().split(":")
        if len(parts) != 3:
            raise ValueError("shape triples must use B:H:Q syntax")
        shape = tuple(int(part) for part in parts)
        if any(number <= 0 for number in shape):
            raise ValueError("shape triple values must be positive")
        shapes.add(shape)
    return shapes


def load_reference_shapes(path):
    shapes = set()
    with Path(path).open() as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                row.get("stage") == "full"
                and row.get("kind") == "prefill"
                and row.get("backend") == "sglang_sparse_q8kv8_fp8"
                and row.get("cache_state") == "steady"
                and row.get("status") == "ok"
            ):
                shapes.add((int(row["B"]), int(row["H"]), int(row["Q"])))
    return shapes


def load_profiles(trace_analysis):
    trace_analysis = Path(trace_analysis)
    model = json.loads(
        (trace_analysis / "raw" / "activation_model.json").read_text()
    )
    conditions = list(csv.DictReader(
        (trace_analysis / "raw" / "activation_by_condition.csv").open()
    ))
    profiles = {}
    for _, _, label in PROFILE_BINS:
        source = model["history_profiles"][label]
        phase_rows = [
            row for row in conditions
            if row["history_bin"] == label and row["layer_group"] == "all"
        ]
        counts = np.asarray([
            float(row["consecutive_c4_fraction_count"]) for row in phase_rows
        ])
        means = np.asarray([
            float(row["consecutive_c4_fraction_mean"]) for row in phase_rows
        ])
        profiles[label] = {
            "overlap": float(source["adjacent_overlap"]["mean"]),
            "age_cdf": {
                int(window): float(fraction)
                for window, fraction in source["recent_token_cdf"].items()
            },
            "consecutive": float(np.average(means, weights=counts)),
        }
    return profiles


def profile_label(visible_tokens):
    for lower, upper, label in PROFILE_BINS:
        if lower <= visible_tokens < upper:
            return label
    raise AssertionError("history profile is undefined")


def sample_age_id(universe, profile, rng):
    draw = rng.random()
    lower = 0
    for upper in AGE_WINDOWS:
        if draw <= profile["age_cdf"][upper]:
            age = rng.integers(lower, upper) if upper > lower else lower
            return max(0, universe - 1 - int(age) // 4)
        lower = upper
    max_age = max(lower + 1, universe * 4)
    age = rng.integers(lower, max_age)
    return max(0, universe - 1 - int(age) // 4)


def sample_clustered(universe, count, excluded, profile, rng):
    """Draw unique C4 ids in short runs while following the empirical age CDF."""
    chosen = set()
    stop_probability = max(0.05, 1.0 - profile["consecutive"])
    attempts = 0
    while len(chosen) < count and attempts < count * 40 + 100:
        attempts += 1
        run = min(count - len(chosen), int(rng.geometric(stop_probability)))
        anchor = sample_age_id(universe, profile, rng)
        direction = 1 if rng.random() < 0.5 else -1
        for step in range(run):
            value = anchor + direction * step
            if 0 <= value < universe and value not in excluded:
                chosen.add(value)
                if len(chosen) == count:
                    break
    if len(chosen) < count:
        start = int(rng.integers(0, universe))
        stride = max(1, universe // max(1, count - len(chosen)))
        while np.gcd(stride, universe) != 1:
            stride += 1
        for step in range(universe):
            value = (start + step * stride) % universe
            if value not in excluded:
                chosen.add(value)
                if len(chosen) == count:
                    break
    if len(chosen) != count:
        raise ValueError(
            f"cannot draw {count} entries from universe={universe} "
            f"with {len(excluded)} exclusions"
        )
    return np.fromiter(chosen, dtype=np.int32, count=count)


def select_clustered_keep(previous, count, rng):
    if count == len(previous):
        return previous.copy()
    ordered = np.sort(previous)
    breaks = np.flatnonzero(np.diff(ordered) != 1) + 1
    runs = list(np.split(ordered, breaks))
    rng.shuffle(runs)
    kept = []
    remaining = count
    for run in runs:
        if remaining <= 0:
            break
        if len(run) <= remaining:
            kept.extend(run.tolist())
            remaining -= len(run)
        else:
            start = int(rng.integers(0, len(run) - remaining + 1))
            kept.extend(run[start:start + remaining].tolist())
            remaining = 0
    return np.asarray(kept, dtype=np.int32)


def build_simulated_c4(H, Q, seed, profiles):
    rng = np.random.default_rng(seed + H * 17 + Q * 1009)
    rows = np.empty((Q, 512), dtype=np.int32)
    labels = []
    targets = []
    cluster_targets = []
    previous = None
    for query in range(Q):
        visible = H + query + 1
        universe = max(512, visible // 4)
        label = profile_label(visible)
        profile = profiles[label]
        labels.append(label)
        cluster_targets.append(profile["consecutive"])
        if previous is None:
            current = sample_clustered(
                universe, 512, set(), profile, rng
            )
        else:
            desired_keep = round(512 * profile["overlap"])
            minimum_keep = max(0, 1024 - universe)
            keep_count = max(desired_keep, minimum_keep)
            keep = select_clustered_keep(previous, keep_count, rng)
            new = sample_clustered(
                universe, 512 - keep_count, set(previous.tolist()), profile, rng
            )
            current = np.concatenate((keep, new))
            targets.append(keep_count / 512)
        rows[query] = current
        previous = current
    return rows, labels, targets, cluster_targets


def batch_seed(seed, batch):
    return seed + batch * 1000003


def build_simulated_batch(args):
    H, Q, seed, profiles = args
    return build_simulated_c4(H, Q, seed, profiles)


def simulated_indices(B, H, Q, seed, profiles, executor=None,
                      row_layout="batch-outer"):
    physical = np.empty((B, Q, bm.TOPK), dtype=np.int32)
    overlap_values = []
    target_values = []
    consecutive_values = []
    cluster_target_values = []
    unique_kv = 0
    profile_bins = []
    tasks = [
        (H, Q, batch_seed(seed, batch), profiles) for batch in range(B)
    ]
    generated = (
        executor.map(build_simulated_batch, tasks)
        if executor is not None and B > 1
        else map(build_simulated_batch, tasks)
    )
    for batch, (raw, labels, targets, cluster_targets) in enumerate(generated):
        expanded = (
            raw[..., None] * 4 + np.arange(4, dtype=np.int32)
        ).reshape(Q, bm.TOPK)
        physical[batch] = expanded + batch * (H + Q)
        if Q > 1:
            combined = np.sort(
                np.concatenate((raw[:-1], raw[1:]), axis=-1), axis=-1
            )
            overlap_values.append(float(
                (combined[:, 1:] == combined[:, :-1]).sum()
                / ((Q - 1) * raw.shape[-1])
            ))
            target_values.append(float(np.mean(targets)))
        else:
            overlap_values.append(1.0)
            target_values.append(1.0)
        ordered = np.sort(raw, axis=-1)
        consecutive_values.append(float(
            (np.diff(ordered, axis=-1) == 1).mean()
        ))
        cluster_target_values.append(float(np.mean(cluster_targets)))
        unique_kv += int(np.unique(raw).size) * 4
        for label in labels:
            if label not in profile_bins:
                profile_bins.append(label)

    if row_layout == "batch-inner":
        physical = physical.transpose(1, 0, 2).copy()
    elif row_layout != "batch-outer":
        raise ValueError(f"unknown row layout: {row_layout}")
    indices = torch.from_numpy(physical).to(device="cuda")
    indices = indices.view(B * Q, 1, bm.TOPK).contiguous()
    return {
        "indices": indices,
        "target_overlap": float(np.mean(target_values)),
        "achieved_overlap": float(np.mean(overlap_values)),
        "target_consecutive": float(np.mean(cluster_target_values)),
        "achieved_consecutive": float(np.mean(consecutive_values)),
        "unique_kv": unique_kv,
        "profile_bins": profile_bins,
        "extrapolated": H + Q > 73739,
    }


def reorder_query_rows(tensor, B, Q, row_layout):
    """Flatten logical [B,Q,...] rows in the requested execution order."""
    if row_layout == "batch-outer":
        return tensor
    if row_layout != "batch-inner":
        raise ValueError(f"unknown row layout: {row_layout}")
    trailing = tensor.shape[1:]
    return (
        tensor.view(B, Q, *trailing).transpose(0, 1).contiguous()
        .view(B * Q, *trailing)
    )


def reorder_query_inputs(inputs, B, Q, row_layout):
    if row_layout == "batch-outer":
        return
    out_shape = inputs["out"].shape
    stats_shape = inputs["max_logits"].shape
    inputs["out"] = None
    inputs["max_logits"] = None
    inputs["lse"] = None
    torch.cuda.empty_cache()
    inputs["q"] = reorder_query_rows(inputs["q"], B, Q, row_layout)
    inputs["out"] = torch.empty(
        out_shape, dtype=torch.bfloat16, device="cuda"
    )
    inputs["max_logits"] = torch.empty(
        stats_shape, dtype=torch.float32, device="cuda"
    )
    inputs["lse"] = torch.empty_like(inputs["max_logits"])


def case_id(pass_order, B, H, Q, row_layout="batch-outer"):
    layout = row_layout.replace("-", "_")
    return f"b{B}_h{H}_q{Q}_csa_profile_sim_{layout}_{pass_order}_steady"


def base_row(stage, pass_order, B, H, Q, args):
    return {
        "case_id": case_id(pass_order, B, H, Q, args.row_layout),
        "status": "ok", "skip_reason": None, "stage": stage,
        "pattern": "csa_profile_sim", "pass_order": pass_order,
        "cache_state": "steady", "row_layout": args.row_layout,
        "B": B, "H": H, "Q": Q,
        "topk": bm.TOPK, "warmup": args.warmup, "repeat": args.repeat,
        "seed": args.seed, "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def run_shape(writer, terminal, profiles, stage, pass_order, B, H, Q, args,
              executor=None):
    row = base_row(stage, pass_order, B, H, Q, args)
    if row["case_id"] in terminal:
        return
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    inputs = indices = fn = None
    try:
        inputs, alloc_ms, quant_ms = runtime.allocate_inputs(
            B, H, Q, args.seed
        )
        reorder_query_inputs(inputs, B, Q, args.row_layout)
        t0 = time.perf_counter()
        simulated = simulated_indices(
            B, H, Q, args.seed, profiles, executor=executor,
            row_layout=args.row_layout,
        )
        indices = simulated["indices"]
        simulated["indices"] = None
        torch.cuda.synchronize()
        indices_ms = (time.perf_counter() - t0) * 1e3
        fn = runtime.bind_kernel(inputs, indices)
        fn()
        torch.cuda.synchronize()
        telemetry_before = bm.gpu_telemetry()
        times = bm.time_kernel(
            fn, args.warmup, args.repeat, "steady", flush_buf=None
        )
        telemetry_after = bm.gpu_telemetry()
        stats = bm.summarize(times)
        pairs = B * Q * bm.TOPK
        flops = bm.flops_sparse_prefill(pairs)
        working_set = simulated["unique_kv"] * bm.D_QK
        row.update(
            latency_ms_median=stats["median"], latency_ms_p5=stats["p5"],
            latency_ms_p95=stats["p95"], latency_ms_mean=stats["mean"],
            latency_ms_std=stats["std"],
            tflops=flops / (stats["median"] / 1000) / 1e12,
            pairs=pairs, flops=flops,
            target_adjacent_overlap=simulated["target_overlap"],
            achieved_adjacent_overlap=simulated["achieved_overlap"],
            target_c4_consecutive_fraction=simulated["target_consecutive"],
            achieved_c4_consecutive_fraction=simulated["achieved_consecutive"],
            unique_kv=simulated["unique_kv"],
            reuse_factor=pairs / simulated["unique_kv"],
            selected_kv_working_set_bytes=working_set,
            selected_kv_working_set_vs_l2=working_set / bm.L2_BYTES,
            profile_bins=json.dumps(simulated["profile_bins"]),
            extrapolated_64k_plus=simulated["extrapolated"],
            setup_alloc_ms=alloc_ms, setup_quant_ms=quant_ms,
            setup_indices_ms=indices_ms,
            peak_mem_alloc_bytes=torch.cuda.max_memory_allocated(),
        )
        for key, value in telemetry_before.items():
            row[f"{key}_before"] = value
        for key, value in telemetry_after.items():
            row[f"{key}_after"] = value
    except torch.OutOfMemoryError as error:
        row["status"] = "skipped_memory_limit"
        row["skip_reason"] = f"input/index/kernel OOM: {error}"
    except Exception as error:
        row["status"] = "failed"
        row["skip_reason"] = f"{type(error).__name__}: {error}"
    writer.append(row)
    print(
        f"{row['case_id']}: {row['status']} "
        f"overlap={row.get('achieved_adjacent_overlap', float('nan')):.4f} "
        f"{row.get('latency_ms_median', 0):.4f} ms",
        flush=True,
    )
    del fn, indices, inputs
    torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "full"), required=True)
    parser.add_argument(
        "--trace-analysis", default="artifacts/data/csa_trace_profile"
    )
    parser.add_argument(
        "--reference-results",
        default="artifacts/data/random_baseline/raw/results.jsonl",
    )
    parser.add_argument("--output-dir", default="runs/csa_profile_sim")
    parser.add_argument("--batches")
    parser.add_argument("--histories")
    parser.add_argument("--prefill-lengths")
    parser.add_argument("--shape-triples", help="exact B:H:Q triples")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--row-layout", choices=("batch-outer", "batch-inner"),
                        default="batch-outer")
    parser.add_argument("--index-workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--time-budget-minutes", type=float, default=120.0)
    parser.add_argument("--between-pass-cooldown-seconds", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("Q8KV8 profile simulation requires an SM90 CUDA GPU")
    if args.index_workers <= 0:
        raise ValueError("--index-workers must be positive")
    if args.between_pass_cooldown_seconds < 0:
        raise ValueError("--between-pass-cooldown-seconds must be non-negative")
    reference = Path(args.reference_results)
    runtime.validate_runtime(reference)
    available = load_reference_shapes(reference)
    exact_shapes = parse_shape_triples(args.shape_triples)
    if exact_shapes is not None:
        requested = exact_shapes
    elif args.stage == "smoke":
        requested = {(1, 2048, 32), (8, 65536, 128), (1, 131072, 32)}
    else:
        batches = parse_int_list(args.batches, DEFAULT_BATCHES)
        histories = parse_int_list(args.histories, DEFAULT_HISTORIES)
        queries = parse_int_list(args.prefill_lengths, DEFAULT_QUERIES)
        requested = {(B, H, Q) for B in batches for H in histories for Q in queries}
    shapes = sorted(requested & available, key=lambda item: (item[0], item[2], item[1]))
    missing = sorted(requested - available)
    if not shapes:
        raise RuntimeError("none of the requested shapes succeeded in the random baseline")
    profiles = load_profiles(args.trace_analysis)
    output = Path(args.output_dir)
    raw = output / "raw"
    results = raw / "results.jsonl"
    if results.exists() and not args.resume:
        raise FileExistsError(f"{results} exists; use --resume or a new output")
    design = {
        "pattern": "csa_profile_sim", "pass_orders": list(PASS_ORDERS),
        "batch_index_mode": "independent-seed-per-batch",
        "row_layout": args.row_layout,
        "profile_source": str(Path(args.trace_analysis).resolve()),
        "reference_results": str(reference.resolve()),
        "index_model": (
            "Markov C4 selected sets matching empirical adjacent overlap, "
            "selected-age CDF, and consecutive-C4 clustering; each C4 id "
            "expands to four token indices"
        ),
        "extrapolation": "64K+ profile is held constant above sampled 73.7K trace",
        "shapes": [{"B": B, "H": H, "Q": Q} for B, H, Q in shapes],
        "missing_reference_shapes": [
            {"B": B, "H": H, "Q": Q} for B, H, Q in missing
        ],
        "shape_count": len(shapes), "case_count": 2 * len(shapes),
        "warmup": args.warmup, "repeat": args.repeat, "seed": args.seed,
        "index_workers": args.index_workers,
        "between_pass_cooldown_seconds": args.between_pass_cooldown_seconds,
    }
    raw.mkdir(parents=True, exist_ok=True)
    design_path = raw / "design.json"
    if args.resume and design_path.exists():
        previous = json.loads(design_path.read_text())
        for key in (
            "pattern", "pass_orders", "profile_source", "reference_results",
            "batch_index_mode", "row_layout", "index_model", "extrapolation", "shapes",
            "warmup", "repeat", "seed",
        ):
            if previous.get(key) != design.get(key):
                raise ValueError(f"resume design differs in {key}")
    design_path.write_text(json.dumps(design, indent=2))
    bm.write_environment(str(raw), str(Path(__file__).resolve().parent), args.seed)
    writer = Writer(raw)
    terminal = writer.terminal_case_ids() if args.resume else set()
    started = time.monotonic()
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=args.index_workers, mp_context=context
    ) as executor:
        for pass_index, pass_order in enumerate(PASS_ORDERS):
            if pass_index and args.between_pass_cooldown_seconds:
                time.sleep(args.between_pass_cooldown_seconds)
            pass_shapes = shapes if pass_order == "ascending" else list(reversed(shapes))
            for index, (B, H, Q) in enumerate(pass_shapes):
                if (time.monotonic() - started) / 60 >= args.time_budget_minutes:
                    remaining = pass_shapes[index:]
                    for rem_B, rem_H, rem_Q in remaining:
                        row = base_row(args.stage, pass_order, rem_B, rem_H, rem_Q, args)
                        if row["case_id"] not in terminal:
                            row.update(
                                status="skipped_time_budget",
                                skip_reason="time budget reached",
                            )
                            writer.append(row)
                    break
                run_shape(
                    writer, terminal, profiles, stage=args.stage,
                    pass_order=pass_order, B=B, H=H, Q=Q, args=args,
                    executor=executor,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
