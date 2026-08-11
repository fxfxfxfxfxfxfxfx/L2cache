#!/usr/bin/env python3
"""Matched-overlap attribution experiment for sparse MLA prefill.

The experiment keeps the Q/KV tensors, kernel, shape, and timing protocol fixed
within each shape and changes only the top-k indices:

* original_strided: the benchmark's causal coprime-stride distribution;
* matched_contiguous: contiguous windows with the same mean adjacent-query
  overlap as original_strided;
* inverted_contiguous: contiguous windows whose adjacent-query overlap is
  ``1 - original_overlap``.

Only histories greater than TOPK are included. The full stage imports the exact
set of successful Q8KV8 shapes from the preceding benchmark by default.
"""

import argparse
import csv
import json
import math
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

import benchmark as bm
from sglang_q8kv8 import sparse_mla_q8kv8_prefill_fwd


PATTERNS = ("original_strided", "matched_contiguous", "inverted_contiguous")
FIELDS = [
    "case_id", "status", "skip_reason", "stage", "pattern", "cache_state",
    "B", "H", "Q", "total_q", "topk", "warmup", "repeat", "seed",
    "latency_ms_median", "latency_ms_p5", "latency_ms_p95",
    "latency_ms_mean", "latency_ms_std", "tflops", "pairs", "flops",
    "original_adjacent_overlap", "target_adjacent_overlap",
    "achieved_adjacent_overlap",
    "overlap_error", "window_shift", "unique_kv", "reuse_factor",
    "selected_kv_working_set_bytes", "selected_kv_working_set_vs_l2",
    "setup_alloc_ms", "setup_quant_ms", "setup_original_indices_ms",
    "setup_overlap_stats_ms", "setup_pattern_indices_ms",
    "peak_mem_alloc_bytes", "temp_c_before", "power_w_before",
    "sm_clock_mhz_before", "mem_clock_mhz_before", "temp_c_after",
    "power_w_after", "sm_clock_mhz_after", "mem_clock_mhz_after",
    "timestamp",
]


class Writer:
    def __init__(self, raw_dir: Path):
        raw_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl = raw_dir / "results.jsonl"
        self.csv = raw_dir / "results.csv"
        if not self.csv.exists():
            with self.csv.open("w", newline="") as f:
                csv.DictWriter(f, fieldnames=FIELDS).writeheader()

    def existing(self):
        if not self.jsonl.exists():
            return set()
        with self.jsonl.open() as f:
            return {
                json.loads(line)["case_id"]
                for line in f
                if line.strip()
            }

    def append(self, row):
        with self.jsonl.open("a") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()
            os.fsync(f.fileno())
        with self.csv.open("a", newline="") as f:
            csv.DictWriter(
                f, fieldnames=FIELDS, extrasaction="ignore"
            ).writerow(row)
            f.flush()
            os.fsync(f.fileno())


def parse_int_list(value):
    if not value:
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def load_reference_shapes(path: Path):
    """Load unique successful steady Q8KV8 full-grid shapes."""
    shapes = set()
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if (
                row.get("stage") == "full"
                and row.get("kind") == "prefill"
                and row.get("backend") == "sglang_sparse_q8kv8_fp8"
                and row.get("cache_state") == "steady"
                and row.get("status") == "ok"
                and int(row["H"]) > bm.TOPK
                and int(row["Q"]) >= 2
            ):
                shapes.add((int(row["B"]), int(row["H"]), int(row["Q"])))
    return sorted(shapes, key=lambda x: (x[0], x[2], x[1]))


def filter_shapes(shapes, batches=None, histories=None, prefill_lengths=None):
    return [
        shape for shape in shapes
        if (batches is None or shape[0] in batches)
        and (histories is None or shape[1] in histories)
        and (prefill_lengths is None or shape[2] in prefill_lengths)
    ]


def _adjacent_overlap(indices: torch.Tensor, B: int, Q: int) -> float:
    """Exact mean |S_i intersection S_(i-1)| / topk within each batch.

    Rows are unique by construction. Sorting each pair's concatenation means
    every equality between adjacent sorted values corresponds to one member of
    the set intersection. Chunking bounds temporary sort storage.
    """
    if Q < 2:
        return float("nan")
    rows = indices.view(B, Q, bm.TOPK)
    intersections = torch.zeros((), dtype=torch.int64, device=indices.device)
    chunk_q = 64
    for begin in range(1, Q, chunk_q):
        end = min(Q, begin + chunk_q)
        pair_values = torch.cat(
            (rows[:, begin - 1:end - 1], rows[:, begin:end]), dim=-1
        ).reshape(-1, 2 * bm.TOPK)
        ordered = pair_values.sort(dim=-1).values
        intersections += (ordered[:, 1:] == ordered[:, :-1]).sum()
        del pair_values, ordered
    pair_count = B * (Q - 1)
    return intersections.item() / (pair_count * bm.TOPK)


def _unique_kv(indices: torch.Tensor, B: int, H: int, Q: int) -> int:
    """Count exact unique selected physical rows with one per-batch bitmap."""
    rows = indices.view(B, Q, bm.TOPK)
    segment = H + Q
    total = 0
    seen = torch.empty(segment, dtype=torch.bool, device=indices.device)
    for batch in range(B):
        seen.zero_()
        local = rows[batch].reshape(-1).to(torch.int64) - batch * segment
        seen[local] = True
        total += int(seen.sum().item())
        del local
    return total


def _bounce_starts(Q: int, H: int, shift: int, device) -> torch.Tensor:
    """Contiguous-window starts with an exact shift between adjacent rows.

    The walk reflects at the largest multiple of ``shift`` that fits in the
    history prefix. Consequently every selected row is one physical contiguous
    interval and every adjacent pair overlaps by TOPK-shift tokens.
    """
    if shift == 0:
        return torch.zeros(Q, dtype=torch.int64, device=device)
    max_step = (H - bm.TOPK) // shift
    if max_step < 1:
        raise ValueError(
            f"cannot place topk={bm.TOPK} windows shift={shift} in H={H}"
        )
    period = 2 * max_step
    phase = torch.arange(Q, dtype=torch.int64, device=device) % period
    step = torch.where(phase <= max_step, phase, period - phase)
    return step * shift


def make_contiguous_indices(B: int, H: int, Q: int, shift: int):
    starts = _bounce_starts(Q, H, shift, "cuda")
    offsets = (
        torch.arange(B, dtype=torch.int64, device="cuda") * (H + Q)
    )[:, None, None]
    tokens = torch.arange(bm.TOPK, dtype=torch.int64, device="cuda")
    indices = offsets + starts[None, :, None] + tokens[None, None, :]
    indices = indices.to(torch.int32).view(B * Q, 1, bm.TOPK).contiguous()
    unique_per_batch = bm.TOPK + int(starts.max().item())
    return indices, B * unique_per_batch


def make_pattern_indices(pattern, B, H, Q, seed, original_overlap):
    if pattern == "original_strided":
        indices, _ = bm._prefill_indices(B, H, Q, bm.TOPK, seed, False)
        unique_kv = _unique_kv(indices, B, H, Q)
        return indices, unique_kv, None, original_overlap
    if pattern == "matched_contiguous":
        shift = int(round(bm.TOPK * (1.0 - original_overlap)))
        shift = min(max(shift, 0), bm.TOPK)
        indices, unique_kv = make_contiguous_indices(B, H, Q, shift)
        return indices, unique_kv, shift, original_overlap
    if pattern == "inverted_contiguous":
        target_overlap = 1.0 - original_overlap
        shift = int(round(bm.TOPK * (1.0 - target_overlap)))
        shift = min(max(shift, 0), bm.TOPK)
        indices, unique_kv = make_contiguous_indices(B, H, Q, shift)
        return indices, unique_kv, shift, target_overlap
    raise ValueError(pattern)


def allocate_inputs(B, H, Q, seed):
    t0 = time.perf_counter()
    gen = torch.Generator(device="cuda").manual_seed(seed)
    total_q = B * Q
    total_kv = B * (H + Q)
    q = torch.empty(
        (total_q, bm.H_Q, bm.D_QK), dtype=torch.float8_e4m3fn, device="cuda"
    )
    kv = torch.empty(
        (total_kv, 1, bm.D_QK), dtype=torch.float8_e4m3fn, device="cuda"
    )
    out = torch.empty(
        (total_q, bm.H_Q, bm.D_V), dtype=torch.bfloat16, device="cuda"
    )
    max_logits = torch.empty(
        (total_q, bm.H_Q), dtype=torch.float32, device="cuda"
    )
    lse = torch.empty_like(max_logits)
    q_scale = torch.ones((), dtype=torch.float32, device="cuda")
    kv_scale = torch.ones((), dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()
    alloc_ms = (time.perf_counter() - t0) * 1e3

    t0 = time.perf_counter()
    bm._fill_random_fp8_chunked(q, gen)
    bm._fill_random_fp8_chunked(kv, gen)
    torch.cuda.synchronize()
    quant_ms = (time.perf_counter() - t0) * 1e3
    return {
        "q": q, "kv": kv, "out": out, "max_logits": max_logits,
        "lse": lse, "q_scale": q_scale, "kv_scale": kv_scale,
    }, alloc_ms, quant_ms


def bind_kernel(inputs, indices):
    def fn():
        return sparse_mla_q8kv8_prefill_fwd(
            inputs["q"], inputs["kv"], indices, bm.SOFTMAX_SCALE,
            inputs["q_scale"], inputs["kv_scale"], d_v=bm.D_V,
            out=inputs["out"], max_logits=inputs["max_logits"],
            lse=inputs["lse"],
        )
    return fn


def case_id(pattern, B, H, Q):
    return f"b{B}_h{H}_q{Q}_{pattern}_steady"


def skip_shape(writer, existing, stage, B, H, Q, status, reason, args):
    for pattern in PATTERNS:
        cid = case_id(pattern, B, H, Q)
        if cid in existing:
            continue
        writer.append({
            "case_id": cid, "status": status, "skip_reason": reason,
            "stage": stage, "pattern": pattern, "cache_state": "steady",
            "B": B, "H": H, "Q": Q, "total_q": B * Q,
            "topk": bm.TOPK, "warmup": args.warmup,
            "repeat": args.repeat, "seed": args.seed,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })


def run_shape(writer, existing, stage, B, H, Q, args):
    if all(case_id(pattern, B, H, Q) in existing for pattern in PATTERNS):
        return
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    inputs = original_indices = None
    try:
        inputs, alloc_ms, quant_ms = allocate_inputs(B, H, Q, args.seed)

        t0 = time.perf_counter()
        original_indices, _ = bm._prefill_indices(
            B, H, Q, bm.TOPK, args.seed, False
        )
        torch.cuda.synchronize()
        original_indices_ms = (time.perf_counter() - t0) * 1e3

        t0 = time.perf_counter()
        original_overlap = _adjacent_overlap(original_indices, B, Q)
        torch.cuda.synchronize()
        overlap_stats_ms = (time.perf_counter() - t0) * 1e3
        del original_indices
        original_indices = None
        torch.cuda.empty_cache()
    except torch.OutOfMemoryError as exc:
        skip_shape(
            writer, existing, stage, B, H, Q, "skipped_memory_limit",
            f"allocation/index statistics OOM: {exc}", args,
        )
        del inputs, original_indices
        torch.cuda.empty_cache()
        return
    except Exception as exc:
        skip_shape(
            writer, existing, stage, B, H, Q, "failed",
            f"setup {type(exc).__name__}: {exc}", args,
        )
        del inputs, original_indices
        torch.cuda.empty_cache()
        return

    pattern_order = list(PATTERNS)
    random.Random(args.seed + B * 17 + H * 31 + Q * 43).shuffle(pattern_order)
    for pattern in pattern_order:
        cid = case_id(pattern, B, H, Q)
        if cid in existing:
            continue
        indices = fn = None
        row = {
            "case_id": cid, "status": "ok", "skip_reason": None,
            "stage": stage, "pattern": pattern, "cache_state": "steady",
            "B": B, "H": H, "Q": Q, "total_q": B * Q,
            "topk": bm.TOPK, "warmup": args.warmup,
            "repeat": args.repeat, "seed": args.seed,
            "original_adjacent_overlap": original_overlap,
            "setup_alloc_ms": alloc_ms, "setup_quant_ms": quant_ms,
            "setup_original_indices_ms": original_indices_ms,
            "setup_overlap_stats_ms": overlap_stats_ms,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            t0 = time.perf_counter()
            indices, unique_kv, shift, target_overlap = make_pattern_indices(
                pattern, B, H, Q, args.seed, original_overlap
            )
            achieved_overlap = _adjacent_overlap(indices, B, Q)
            torch.cuda.synchronize()
            pattern_indices_ms = (time.perf_counter() - t0) * 1e3

            fn = bind_kernel(inputs, indices)
            fn()
            torch.cuda.synchronize()
            tele0 = bm.gpu_telemetry()
            times = bm.time_kernel(
                fn, args.warmup, args.repeat, "steady", flush_buf=None
            )
            tele1 = bm.gpu_telemetry()
            stats = bm.summarize(times)
            median_s = stats["median"] / 1000.0
            pairs = B * Q * bm.TOPK
            flops = bm.flops_sparse_prefill(pairs)
            working_set = unique_kv * bm.D_QK
            row.update(
                latency_ms_median=stats["median"],
                latency_ms_p5=stats["p5"], latency_ms_p95=stats["p95"],
                latency_ms_mean=stats["mean"], latency_ms_std=stats["std"],
                tflops=flops / median_s / 1e12, pairs=pairs, flops=flops,
                target_adjacent_overlap=target_overlap,
                achieved_adjacent_overlap=achieved_overlap,
                overlap_error=achieved_overlap - target_overlap,
                window_shift=shift, unique_kv=unique_kv,
                reuse_factor=pairs / unique_kv,
                selected_kv_working_set_bytes=working_set,
                selected_kv_working_set_vs_l2=working_set / bm.L2_BYTES,
                setup_pattern_indices_ms=pattern_indices_ms,
                peak_mem_alloc_bytes=torch.cuda.max_memory_allocated(),
            )
            for key, value in tele0.items():
                row[f"{key}_before"] = value
            for key, value in tele1.items():
                row[f"{key}_after"] = value
        except torch.OutOfMemoryError as exc:
            row["status"] = "skipped_memory_limit"
            row["skip_reason"] = f"pattern index/kernel OOM: {exc}"
        except Exception as exc:
            row["status"] = "failed"
            row["skip_reason"] = f"{type(exc).__name__}: {exc}"
        writer.append(row)
        print(
            f"{cid}: {row['status']} overlap="
            f"{row.get('achieved_adjacent_overlap', float('nan')):.4f} "
            f"{row.get('latency_ms_median', 0):.4f} ms",
            flush=True,
        )
        del indices, fn
        torch.cuda.empty_cache()

    del inputs
    torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "full"), required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--time-budget-minutes", type=float, default=55.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--reference-results", default="assets/q8kv8/raw/results.jsonl"
    )
    parser.add_argument(
        "--output-dir", default="assets/overlap_attribution"
    )
    parser.add_argument("--batches")
    parser.add_argument("--histories")
    parser.add_argument("--prefill-lengths")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("Q8KV8 kernel requires SM90")

    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    writer = Writer(raw_dir)
    bm.write_environment(
        str(raw_dir), str(Path(__file__).resolve().parent), args.seed
    )
    shapes = load_reference_shapes(Path(args.reference_results))
    shapes = filter_shapes(
        shapes, parse_int_list(args.batches), parse_int_list(args.histories),
        parse_int_list(args.prefill_lengths),
    )
    if args.stage == "smoke":
        requested = {(1, 4096, 64), (1, 32768, 256), (8, 32768, 256)}
        shapes = [shape for shape in shapes if shape in requested]
    if not shapes:
        raise RuntimeError("no reference shapes selected")

    design = {
        "patterns": list(PATTERNS), "topk": bm.TOPK,
        "reference_results": str(Path(args.reference_results).resolve()),
        "shape_count": len(shapes), "histories_gt_topk_only": True,
        "warmup": args.warmup, "repeat": args.repeat,
        "time_budget_minutes": args.time_budget_minutes, "seed": args.seed,
        "matched_contiguous_definition": (
            "single contiguous topk windows following a reflecting walk; "
            "integer window shift is round(topk*(1-original_mean_overlap))"
        ),
        "inverted_contiguous_definition": (
            "the same reflecting contiguous-window construction with target "
            "overlap=1-original_mean_overlap"
        ),
    }
    raw_dir.mkdir(parents=True, exist_ok=True)
    with (raw_dir / "design.json").open("w") as f:
        json.dump(design, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

    existing = writer.existing() if args.resume else set()
    started = time.monotonic()
    for index, (B, H, Q) in enumerate(shapes):
        elapsed_minutes = (time.monotonic() - started) / 60.0
        if elapsed_minutes >= args.time_budget_minutes:
            for remaining in shapes[index:]:
                skip_shape(
                    writer, existing, args.stage, *remaining,
                    "skipped_time_budget", "time budget reached", args,
                )
            break
        run_shape(writer, existing, args.stage, B, H, Q, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
