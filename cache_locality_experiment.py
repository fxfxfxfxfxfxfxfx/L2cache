#!/usr/bin/env python3
"""Controlled H800 experiment for sparse-MLA history-scaling causes.

This isolates three variables without Nsight Compute:
1. spatial locality within each query's selected 2048 KV tokens;
2. selected-KV overlap across query rows in the same sequence;
3. inter-iteration L2 residency (steady vs a 256 MiB L2 flush).

The full run repeats histories in ascending and descending order. It also
compares real FlashMLA FP8 decode with query-row-matched sparse prefill cases.
"""

import argparse
import csv
import json
import math
import os
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

import benchmark as bm
from sglang_q8kv8 import sparse_mla_q8kv8_prefill_fwd


HISTORIES = [2048, 32768, 524288]
PASSES = [("ascending", HISTORIES), ("descending", list(reversed(HISTORIES)))]
SHARED_PATTERNS = [
    "shared_contiguous",
    "shared_dispersed",
    "independent_contiguous",
    "independent_dispersed",
]
ISOLATED_PATTERNS = ["isolated_contiguous", "isolated_dispersed"]
CACHE_STATES = ["steady", "l2-cold"]
FIELDS = [
    "case_id", "status", "skip_reason", "pass_order", "backend",
    "cache_state", "pattern", "segment_mode", "N", "H", "topk",
    "warmup", "repeat", "latency_ms_median", "latency_ms_p5",
    "latency_ms_p95", "latency_ms_mean", "latency_ms_std", "tflops",
    "pairs", "flops", "unique_kv", "reuse_factor",
    "adjacent_overlap", "selected_kv_working_set_bytes",
    "selected_kv_working_set_vs_l2", "kv_token_bytes",
    "peak_mem_alloc_bytes", "temp_c_before", "power_w_before",
    "sm_clock_mhz_before", "mem_clock_mhz_before", "temp_c_after",
    "power_w_after", "sm_clock_mhz_after", "mem_clock_mhz_after",
    "seed", "timestamp",
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
        return {json.loads(line)["case_id"] for line in self.jsonl.open()
                if line.strip()}

    def append(self, row):
        with self.jsonl.open("a") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()
            os.fsync(f.fileno())
        with self.csv.open("a", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS,
                           extrasaction="ignore").writerow(row)
            f.flush()
            os.fsync(f.fileno())


def _coprime_stride(history):
    stride = max(1, math.ceil(history / bm.TOPK))
    while math.gcd(stride, history) != 1:
        stride += 1
    return stride


def make_indices(n_rows, history, pattern, seed):
    """Return global int32 indices and exact locality statistics."""
    j = torch.arange(bm.TOPK, dtype=torch.int64)
    stride = _coprime_stride(history)
    rows = []
    for row in range(n_rows):
        if pattern.endswith("contiguous"):
            if pattern.startswith("shared"):
                start = 0
            elif pattern.startswith("independent"):
                start = row * max(1, history // n_rows)
            else:
                start = 0
            local = (start + j) % history
        else:
            start = 0 if pattern.startswith(("shared", "isolated")) else (
                seed * 1000003 + row * 10007) % history
            local = (start + j * stride) % history
        if pattern.startswith("isolated"):
            local = local + row * history
        rows.append(local)
    cpu = torch.stack(rows)
    unique_kv = int(torch.unique(cpu).numel())
    overlaps = []
    for row in range(1, n_rows):
        overlaps.append(float(torch.isin(cpu[row], cpu[row - 1]).float().mean()))
    adjacent_overlap = statistics.fmean(overlaps) if overlaps else 1.0
    indices = cpu.to(torch.int32).view(n_rows, 1, bm.TOPK).cuda()
    return indices.contiguous(), unique_kv, adjacent_overlap


def allocate_prefill(backend, n_rows, history, segment_mode):
    segments = 1 if segment_mode == "shared" else n_rows
    total_kv = segments * history
    if backend == "sglang_q8kv8":
        q = torch.zeros((n_rows, bm.H_Q, bm.D_QK),
                        dtype=torch.float8_e4m3fn, device="cuda")
        kv = torch.zeros((total_kv, 1, bm.D_QK),
                         dtype=torch.float8_e4m3fn, device="cuda")
        out = torch.empty((n_rows, bm.H_Q, bm.D_V), dtype=torch.bfloat16,
                          device="cuda")
        max_logits = torch.empty((n_rows, bm.H_Q), dtype=torch.float32,
                                 device="cuda")
        lse = torch.empty_like(max_logits)
        q_scale = torch.ones((), dtype=torch.float32, device="cuda")
        kv_scale = torch.ones((), dtype=torch.float32, device="cuda")
        token_bytes = bm.D_QK

        def bind(indices):
            def fn():
                return sparse_mla_q8kv8_prefill_fwd(
                    q, kv, indices, bm.SOFTMAX_SCALE, q_scale, kv_scale,
                    d_v=bm.D_V, out=out, max_logits=max_logits, lse=lse,
                )
            return fn

        holders = (q, kv, out, max_logits, lse, q_scale, kv_scale)
    elif backend == "flashmla_bf16":
        import flash_mla
        q = torch.zeros((n_rows, bm.H_Q, bm.D_QK),
                        dtype=torch.bfloat16, device="cuda")
        kv = torch.zeros((total_kv, 1, bm.D_QK),
                         dtype=torch.bfloat16, device="cuda")
        topk_length = torch.full((n_rows,), bm.TOPK, dtype=torch.int32,
                                 device="cuda")
        token_bytes = bm.D_QK * 2

        def bind(indices):
            def fn():
                return flash_mla.flash_mla_sparse_fwd(
                    q, kv, indices, bm.SOFTMAX_SCALE, d_v=bm.D_V,
                    topk_length=topk_length,
                )
            return fn

        holders = (q, kv, topk_length)
    else:
        raise ValueError(backend)
    torch.cuda.synchronize()
    return bind, holders, token_bytes


def case_id(pass_order, backend, cache_state, pattern, n_rows, history):
    return (f"{pass_order}_{backend}_{cache_state}_{pattern}_"
            f"n{n_rows}_h{history}")


def time_bound_case(fn, warmup, repeat, cache_state, flush_buf):
    return bm.time_kernel(fn, warmup, repeat, cache_state, flush_buf)


def run_prefill_group(writer, existing, pass_order, backend, n_rows, history,
                      segment_mode, patterns, args, flush_buf):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    bind = holders = None
    try:
        bind, holders, token_bytes = allocate_prefill(
            backend, n_rows, history, segment_mode)
    except torch.OutOfMemoryError as exc:
        for pattern in patterns:
            for cache_state in CACHE_STATES:
                cid = case_id(pass_order, backend, cache_state, pattern,
                              n_rows, history)
                if cid in existing:
                    continue
                writer.append({
                    "case_id": cid, "status": "skipped_memory_limit",
                    "skip_reason": f"allocation OOM: {exc}",
                    "pass_order": pass_order, "backend": backend,
                    "cache_state": cache_state, "pattern": pattern,
                    "segment_mode": segment_mode, "N": n_rows,
                    "H": history, "topk": bm.TOPK, "warmup": args.warmup,
                    "repeat": args.repeat, "seed": args.seed,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
        torch.cuda.empty_cache()
        return

    prepared = {}
    for pattern in patterns:
        indices, unique_kv, overlap = make_indices(
            n_rows, history, pattern, args.seed)
        prepared[pattern] = (indices, unique_kv, overlap, bind(indices))

    order = [(pattern, cache) for pattern in patterns for cache in CACHE_STATES]
    random.Random(args.seed + history + n_rows
                  + (0 if pass_order == "ascending" else 1)).shuffle(order)
    for pattern, cache_state in order:
        cid = case_id(pass_order, backend, cache_state, pattern, n_rows, history)
        if cid in existing:
            continue
        indices, unique_kv, overlap, fn = prepared[pattern]
        tele0 = bm.gpu_telemetry()
        row = {
            "case_id": cid, "status": "ok", "skip_reason": None,
            "pass_order": pass_order, "backend": backend,
            "cache_state": cache_state, "pattern": pattern,
            "segment_mode": segment_mode, "N": n_rows, "H": history,
            "topk": bm.TOPK, "warmup": args.warmup, "repeat": args.repeat,
            "seed": args.seed, "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            times = time_bound_case(fn, args.warmup, args.repeat,
                                    cache_state, flush_buf)
            stats = bm.summarize(times)
            median_s = stats["median"] / 1000
            pairs = n_rows * bm.TOPK
            flops = bm.flops_sparse_prefill(pairs)
            working_set = unique_kv * token_bytes
            row.update(
                latency_ms_median=stats["median"], latency_ms_p5=stats["p5"],
                latency_ms_p95=stats["p95"], latency_ms_mean=stats["mean"],
                latency_ms_std=stats["std"], tflops=flops / median_s / 1e12,
                pairs=pairs, flops=flops, unique_kv=unique_kv,
                reuse_factor=pairs / unique_kv, adjacent_overlap=overlap,
                selected_kv_working_set_bytes=working_set,
                selected_kv_working_set_vs_l2=working_set / bm.L2_BYTES,
                kv_token_bytes=token_bytes,
                peak_mem_alloc_bytes=torch.cuda.max_memory_allocated(),
            )
        except Exception as exc:
            row["status"] = "failed"
            row["skip_reason"] = f"{type(exc).__name__}: {exc}"
        tele1 = bm.gpu_telemetry()
        for key, value in tele0.items():
            row[f"{key}_before"] = value
        for key, value in tele1.items():
            row[f"{key}_after"] = value
        writer.append(row)
        print(f"{cid}: {row['status']} "
              f"{row.get('latency_ms_median', 0):.4f} ms", flush=True)

    del prepared, bind, holders
    torch.cuda.empty_cache()


def run_decode_case(writer, existing, pass_order, history, cache_state, args,
                    flush_buf):
    backend = "flashmla_fp8_decode"
    pattern = "native_independent_dispersed"
    cid = case_id(pass_order, backend, cache_state, pattern, 64, history)
    if cid in existing:
        return
    tele0 = bm.gpu_telemetry()
    row = {
        "case_id": cid, "status": "ok", "skip_reason": None,
        "pass_order": pass_order, "backend": backend,
        "cache_state": cache_state, "pattern": pattern,
        "segment_mode": "isolated", "N": 64, "H": history,
        "topk": bm.TOPK, "warmup": args.warmup, "repeat": args.repeat,
        "seed": args.seed, "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    fn = holders = None
    try:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        fn, account, _, holders = bm.setup_sparse_decode(64, history, args.seed)
        times = bm.time_kernel(fn, args.warmup, args.repeat, cache_state,
                               flush_buf)
        stats = bm.summarize(times)
        median_s = stats["median"] / 1000
        unique_kv = 64 * bm.TOPK
        working_set = unique_kv * bm.FP8_TOKEN_BYTES
        row.update(
            latency_ms_median=stats["median"], latency_ms_p5=stats["p5"],
            latency_ms_p95=stats["p95"], latency_ms_mean=stats["mean"],
            latency_ms_std=stats["std"],
            tflops=account["flops"] / median_s / 1e12,
            pairs=account["pairs"], flops=account["flops"],
            unique_kv=unique_kv, reuse_factor=1.0, adjacent_overlap=0.0,
            selected_kv_working_set_bytes=working_set,
            selected_kv_working_set_vs_l2=working_set / bm.L2_BYTES,
            kv_token_bytes=bm.FP8_TOKEN_BYTES,
            peak_mem_alloc_bytes=torch.cuda.max_memory_allocated(),
        )
    except torch.OutOfMemoryError as exc:
        row["status"] = "skipped_memory_limit"
        row["skip_reason"] = f"runtime OOM: {exc}"
    except Exception as exc:
        row["status"] = "failed"
        row["skip_reason"] = f"{type(exc).__name__}: {exc}"
    finally:
        del fn, holders
        torch.cuda.empty_cache()
    tele1 = bm.gpu_telemetry()
    for key, value in tele0.items():
        row[f"{key}_before"] = value
    for key, value in tele1.items():
        row[f"{key}_after"] = value
    writer.append(row)
    print(f"{cid}: {row['status']} {row.get('latency_ms_median', 0):.4f} ms",
          flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["smoke", "full"], required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", default="assets/cache_locality")
    return parser.parse_args()


def main():
    args = parse_args()
    assert torch.cuda.is_available()
    assert torch.cuda.get_device_capability() == (9, 0)
    raw_dir = Path(args.output_dir) / "raw"
    writer = Writer(raw_dir)
    bm.write_environment(str(raw_dir), str(Path(__file__).resolve().parent),
                         args.seed)
    design = {
        "histories": HISTORIES,
        "passes": [name for name, _ in PASSES],
        "shared_patterns": SHARED_PATTERNS,
        "isolated_patterns": ISOLATED_PATTERNS,
        "cache_states": CACHE_STATES,
        "query_rows": [64, 256],
        "topk": bm.TOPK,
        "l2_bytes": bm.L2_BYTES,
        "flush_bytes": bm.FLUSH_BUF_BYTES,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "seed": args.seed,
    }
    with (raw_dir / "design.json").open("w") as f:
        json.dump(design, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    existing = writer.existing() if args.resume else set()
    flush_buf = torch.ones(bm.FLUSH_BUF_BYTES // 4, dtype=torch.float32,
                           device="cuda")

    if args.stage == "smoke":
        for h in (2048, 32768):
            run_prefill_group(writer, existing, "smoke", "sglang_q8kv8", 8,
                              h, "shared", SHARED_PATTERNS, args, flush_buf)
        return 0

    for pass_order, histories in PASSES:
        print(f"== {pass_order} ==", flush=True)
        for history in histories:
            groups = [
                ("sglang_q8kv8", 64, "shared", SHARED_PATTERNS),
                ("sglang_q8kv8", 64, "isolated", ISOLATED_PATTERNS),
                ("flashmla_bf16", 64, "shared", SHARED_PATTERNS),
                ("flashmla_bf16", 64, "isolated", ISOLATED_PATTERNS),
                ("sglang_q8kv8", 256, "shared", SHARED_PATTERNS),
            ]
            random.Random(args.seed + history).shuffle(groups)
            for backend, n_rows, segment_mode, patterns in groups:
                run_prefill_group(
                    writer, existing, pass_order, backend, n_rows, history,
                    segment_mode, patterns, args, flush_buf,
                )
            decode_states = list(CACHE_STATES)
            random.Random(args.seed + history + 17).shuffle(decode_states)
            for cache_state in decode_states:
                run_decode_case(writer, existing, pass_order, history,
                                cache_state, args, flush_buf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
