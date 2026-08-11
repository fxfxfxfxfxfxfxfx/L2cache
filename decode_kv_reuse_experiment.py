#!/usr/bin/env python3
"""Native sparse-decode selected-KV reuse experiment without NCU.

Both patterns start every timed call after the same 256 MiB L2 flush:

* shared: every batch row selects the same 2048 physical KV tokens;
* independent: every batch row selects its own 2048 physical KV tokens.

Kernel, Q, KV allocation, batch, topk, FLOPs, flush, and timing protocol are
identical. Only the number of unique selected KV tokens inside the timed kernel
changes. This measures the performance value of within-invocation KV reuse;
it does not claim a hardware-measured L2 hit rate.
"""

import argparse
import csv
import json
import os
import random
from datetime import datetime, timezone
from pathlib import Path

import torch

import benchmark as bm


HISTORIES = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]
BATCHES = [1, 8, 16, 32, 64]
PATTERNS = ["shared", "independent"]
FIELDS = [
    "case_id", "status", "skip_reason", "pass_order", "pattern", "B", "H",
    "topk", "warmup", "repeat", "latency_ms_median", "latency_ms_p5",
    "latency_ms_p95", "latency_ms_mean", "latency_ms_std", "tflops", "pairs",
    "flops", "unique_selected_kv", "selected_kv_bytes",
    "selected_kv_vs_l2", "known_touched_bytes", "known_touched_vs_l2",
    "l2_bytes", "flush_bytes", "setup_alloc_ms", "setup_indices_ms",
    "setup_quant_ms", "setup_metadata_ms", "peak_mem_alloc_bytes",
    "temp_c_before", "power_w_before", "sm_clock_mhz_before",
    "mem_clock_mhz_before", "temp_c_after", "power_w_after",
    "sm_clock_mhz_after", "mem_clock_mhz_after", "seed", "timestamp",
]


class Writer:
    def __init__(self, raw_dir: Path):
        raw_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl = raw_dir / "results.jsonl"
        self.csv = raw_dir / "results.csv"
        if not self.csv.exists():
            with self.csv.open("w", newline="") as f:
                csv.DictWriter(f, fieldnames=FIELDS).writeheader()

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


def make_decode_fn(q, kv_fp8, indices):
    import flash_mla
    sched, _ = flash_mla.get_mla_metadata()

    def fn():
        return flash_mla.flash_mla_with_kvcache(
            q, kv_fp8, None, None, bm.D_V, sched, None,
            bm.SOFTMAX_SCALE, False, True, indices)

    return fn


def flush(flush_buf):
    result = flush_buf.sum()
    torch.cuda.synchronize()
    del result


def time_interleaved(functions, flush_buf, warmup, repeat, seed):
    for _ in range(warmup):
        for pattern in PATTERNS:
            flush(flush_buf)
            output = functions[pattern]()
            torch.cuda.synchronize()
            del output

    times = {pattern: [] for pattern in PATTERNS}
    first = list(PATTERNS)
    random.Random(seed).shuffle(first)
    for iteration in range(repeat):
        order = first if iteration % 2 == 0 else list(reversed(first))
        for pattern in order:
            # Identical immediate predecessor and clock conditioning.
            flush(flush_buf)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = functions[pattern]()
            end.record()
            torch.cuda.synchronize()
            times[pattern].append(start.elapsed_time(end))
            del output
    return times


def case_id(pass_order, pattern, batch, history):
    return f"{pass_order}_{pattern}_b{batch}_h{history}"


def run_case(writer, pass_order, batch, history, args, flush_buf):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    tele0 = bm.gpu_telemetry()
    base_fn = holders = None
    try:
        base_fn, account, setup, holders = bm.setup_sparse_decode(
            batch, history, args.seed)
        independent_indices = holders["indices"]
        shared_indices = independent_indices[0:1].expand(
            batch, -1, -1).contiguous()
        assert int(torch.unique(shared_indices).numel()) == bm.TOPK
        assert int(torch.unique(independent_indices).numel()) == batch * bm.TOPK
        functions = {
            "shared": make_decode_fn(holders["q"], holders["kv_fp8"],
                                     shared_indices),
            "independent": make_decode_fn(holders["q"], holders["kv_fp8"],
                                          independent_indices),
        }
        times = time_interleaved(
            functions, flush_buf, args.warmup, args.repeat,
            args.seed + batch * 1009 + history
            + (0 if pass_order == "ascending" else 1))
        peak_mem = torch.cuda.max_memory_allocated()
        tele1 = bm.gpu_telemetry()
        for pattern in PATTERNS:
            unique_kv = bm.TOPK if pattern == "shared" else batch * bm.TOPK
            selected_bytes = unique_kv * bm.FP8_TOKEN_BYTES
            q_bytes = batch * bm.H_Q * bm.D_QK * 2
            output_bytes = batch * bm.H_Q * bm.D_V * 2
            index_bytes = batch * bm.TOPK * 4
            known_touched = selected_bytes + q_bytes + output_bytes + index_bytes
            stats = bm.summarize(times[pattern])
            median_s = stats["median"] / 1000
            row = {
                "case_id": case_id(pass_order, pattern, batch, history),
                "status": "ok", "skip_reason": None,
                "pass_order": pass_order, "pattern": pattern,
                "B": batch, "H": history, "topk": account["topk"],
                "warmup": args.warmup, "repeat": args.repeat,
                "latency_ms_median": stats["median"],
                "latency_ms_p5": stats["p5"],
                "latency_ms_p95": stats["p95"],
                "latency_ms_mean": stats["mean"],
                "latency_ms_std": stats["std"],
                "tflops": account["flops"] / median_s / 1e12,
                "pairs": account["pairs"], "flops": account["flops"],
                "unique_selected_kv": unique_kv,
                "selected_kv_bytes": selected_bytes,
                "selected_kv_vs_l2": selected_bytes / bm.L2_BYTES,
                "known_touched_bytes": known_touched,
                "known_touched_vs_l2": known_touched / bm.L2_BYTES,
                "l2_bytes": bm.L2_BYTES, "flush_bytes": bm.FLUSH_BUF_BYTES,
                "setup_alloc_ms": setup["alloc"],
                "setup_indices_ms": setup["indices"],
                "setup_quant_ms": setup["quant"],
                "setup_metadata_ms": setup["metadata"],
                "peak_mem_alloc_bytes": peak_mem,
                "seed": args.seed,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            for key, value in tele0.items():
                row[f"{key}_before"] = value
            for key, value in tele1.items():
                row[f"{key}_after"] = value
            writer.append(row)
        shared = bm.summarize(times["shared"])["median"]
        independent = bm.summarize(times["independent"])["median"]
        print(f"{pass_order} B={batch} H={history}: "
              f"shared={shared * 1000:.3f} us "
              f"independent={independent * 1000:.3f} us "
              f"ind/shared={independent / shared:.3f}x", flush=True)
    except Exception as exc:
        tele1 = bm.gpu_telemetry()
        for pattern in PATTERNS:
            row = {
                "case_id": case_id(pass_order, pattern, batch, history),
                "status": "failed", "skip_reason": f"{type(exc).__name__}: {exc}",
                "pass_order": pass_order, "pattern": pattern,
                "B": batch, "H": history, "topk": bm.TOPK,
                "warmup": args.warmup, "repeat": args.repeat,
                "seed": args.seed,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            for key, value in tele0.items():
                row[f"{key}_before"] = value
            for key, value in tele1.items():
                row[f"{key}_after"] = value
            writer.append(row)
        raise
    finally:
        del base_fn, holders
        torch.cuda.empty_cache()


def parse_ints(value):
    return [int(item) for item in value.split(",") if item]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["smoke", "full"], required=True)
    parser.add_argument("--batches", default=",".join(map(str, BATCHES)))
    parser.add_argument("--histories", default=",".join(map(str, HISTORIES)))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--output-dir", default="assets/decode_kv_reuse")
    return parser.parse_args()


def main():
    args = parse_args()
    assert torch.cuda.is_available()
    assert torch.cuda.get_device_capability() == (9, 0)
    batches = parse_ints(args.batches)
    histories = parse_ints(args.histories)
    assert all(history > bm.TOPK for history in histories)
    if args.stage == "smoke":
        batches, histories = [1, 16, 64], [4096]

    output_dir = Path(args.output_dir)
    raw_dir = output_dir / "raw"
    writer = Writer(raw_dir)
    bm.write_environment(str(raw_dir), str(Path(__file__).resolve().parent),
                         args.seed)
    design = {
        "histories": histories, "batches": batches,
        "patterns": PATTERNS, "passes": ["ascending", "descending"],
        "topk": bm.TOPK, "l2_bytes": bm.L2_BYTES,
        "flush_bytes": bm.FLUSH_BUF_BYTES, "warmup": args.warmup,
        "repeat": args.repeat, "seed": args.seed,
        "timing": "identical 256 MiB flush immediately before every timed call",
        "interpretation": ("selected-KV reuse intervention; hardware L2 hit "
                           "rates are not observed"),
    }
    with (raw_dir / "design.json").open("w") as f:
        json.dump(design, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

    flush_buf = torch.ones(bm.FLUSH_BUF_BYTES // 4, dtype=torch.float32,
                           device="cuda")
    for pass_order, ordered_histories in (
            ("ascending", sorted(histories)),
            ("descending", sorted(histories, reverse=True))):
        for history in ordered_histories:
            ordered_batches = list(batches)
            random.Random(args.seed + history
                          + (0 if pass_order == "ascending" else 1)).shuffle(
                              ordered_batches)
            for batch in ordered_batches:
                run_case(writer, pass_order, batch, history, args, flush_buf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
