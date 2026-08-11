#!/usr/bin/env python3
"""Paired L2-hot/L2-cold proxy experiment for native sparse MLA decode.

Without hardware counters this experiment cannot assert a 100% L2 hit or miss
rate.  Instead it compares two interventions immediately before every timed
decode call:

Both states execute one identical decode prime and one 256 MiB flush before the
timed call, so their clock-conditioning work is symmetric.  Only the order is
changed:

* hot-resident: flush, then prime the identical decode, then time it;
* flushed: prime the identical decode, then flush, then time it.

The two states are interleaved and the prime/flush work is outside CUDA events.
"""

import argparse
import csv
import json
import os
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path

import torch

import benchmark as bm


HISTORIES = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288]
BATCHES = [1, 8, 16, 32, 64]
PASSES = [("ascending", HISTORIES),
          ("descending", list(reversed(HISTORIES)))]
STATES = ["hot-resident", "flushed"]
FIELDS = [
    "case_id", "status", "skip_reason", "pass_order", "cache_state",
    "B", "H", "topk", "warmup", "repeat", "latency_ms_median",
    "latency_ms_p5", "latency_ms_p95", "latency_ms_mean",
    "latency_ms_std", "tflops", "pairs", "flops", "selected_kv_bytes",
    "known_touched_bytes", "selected_kv_vs_l2", "known_touched_vs_l2",
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


def run_and_sync(fn):
    output = fn()
    torch.cuda.synchronize()
    del output


def paired_times(fn, flush_buf, warmup, repeat, order_seed):
    """Measure interleaved states; preparation is outside event intervals."""
    for _ in range(warmup):
        flush_result = flush_buf.sum()
        torch.cuda.synchronize()
        del flush_result
        run_and_sync(fn)
        run_and_sync(fn)
        flush_result = flush_buf.sum()
        torch.cuda.synchronize()
        del flush_result
        run_and_sync(fn)

    times = {state: [] for state in STATES}
    rng = random.Random(order_seed)
    first = list(STATES)
    rng.shuffle(first)
    for iteration in range(repeat):
        order = first if iteration % 2 == 0 else list(reversed(first))
        for state in order:
            # Both states execute exactly one flush and one decode prime.  The
            # last operation determines whether selected KV is recent in L2.
            if state == "hot-resident":
                flush_result = flush_buf.sum()
                torch.cuda.synchronize()
                del flush_result
                run_and_sync(fn)
            else:
                run_and_sync(fn)
                flush_result = flush_buf.sum()
                torch.cuda.synchronize()
                del flush_result
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = fn()
            end.record()
            torch.cuda.synchronize()
            times[state].append(start.elapsed_time(end))
            del output
    return times


def case_id(pass_order, state, batch, history):
    return f"{pass_order}_{state}_b{batch}_h{history}"


def run_case(writer, existing, pass_order, batch, history, args, flush_buf):
    missing_states = [s for s in STATES
                      if case_id(pass_order, s, batch, history) not in existing]
    if not missing_states:
        return

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    tele0 = bm.gpu_telemetry()
    fn = holders = None
    try:
        fn, account, setup, holders = bm.setup_sparse_decode(
            batch, history, args.seed)
        times_by_state = paired_times(
            fn, flush_buf, args.warmup, args.repeat,
            args.seed + batch * 1009 + history
            + (0 if pass_order == "ascending" else 1),
        )
        peak_mem = torch.cuda.max_memory_allocated()
        tele1 = bm.gpu_telemetry()
        topk = account["topk"]
        selected_kv_bytes = batch * topk * bm.FP8_TOKEN_BYTES
        q_bytes = batch * bm.H_Q * bm.D_QK * 2
        output_bytes = batch * bm.H_Q * bm.D_V * 2
        index_bytes = batch * topk * 4
        known_touched = selected_kv_bytes + q_bytes + output_bytes + index_bytes
        for state in missing_states:
            stats = bm.summarize(times_by_state[state])
            median_s = stats["median"] / 1000
            row = {
                "case_id": case_id(pass_order, state, batch, history),
                "status": "ok", "skip_reason": None,
                "pass_order": pass_order, "cache_state": state,
                "B": batch, "H": history, "topk": topk,
                "warmup": args.warmup, "repeat": args.repeat,
                "latency_ms_median": stats["median"],
                "latency_ms_p5": stats["p5"],
                "latency_ms_p95": stats["p95"],
                "latency_ms_mean": stats["mean"],
                "latency_ms_std": stats["std"],
                "tflops": account["flops"] / median_s / 1e12,
                "pairs": account["pairs"], "flops": account["flops"],
                "selected_kv_bytes": selected_kv_bytes,
                "known_touched_bytes": known_touched,
                "selected_kv_vs_l2": selected_kv_bytes / bm.L2_BYTES,
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
        hot = bm.summarize(times_by_state["hot-resident"])["median"]
        cold = bm.summarize(times_by_state["flushed"])["median"]
        print(f"{pass_order} B={batch} H={history}: "
              f"hot={hot * 1000:.3f} us cold={cold * 1000:.3f} us "
              f"cold/hot={cold / hot:.3f}x", flush=True)
    except torch.OutOfMemoryError as exc:
        tele1 = bm.gpu_telemetry()
        for state in missing_states:
            row = {
                "case_id": case_id(pass_order, state, batch, history),
                "status": "skipped_memory_limit",
                "skip_reason": f"runtime OOM: {exc}",
                "pass_order": pass_order, "cache_state": state,
                "B": batch, "H": history, "topk": bm.TOPK,
                "warmup": args.warmup, "repeat": args.repeat,
                "l2_bytes": bm.L2_BYTES, "flush_bytes": bm.FLUSH_BUF_BYTES,
                "seed": args.seed,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            for key, value in tele0.items():
                row[f"{key}_before"] = value
            for key, value in tele1.items():
                row[f"{key}_after"] = value
            writer.append(row)
    finally:
        del fn, holders
        torch.cuda.empty_cache()


def parse_csv_ints(value):
    return [int(item) for item in value.split(",") if item]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["smoke", "full"], required=True)
    parser.add_argument("--batches", default=",".join(map(str, BATCHES)))
    parser.add_argument("--histories", default=",".join(map(str, HISTORIES)))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", default="assets/decode_l2_residency")
    return parser.parse_args()


def main():
    args = parse_args()
    assert torch.cuda.is_available()
    assert torch.cuda.get_device_capability() == (9, 0)
    batches = parse_csv_ints(args.batches)
    histories = parse_csv_ints(args.histories)
    assert all(history > bm.TOPK for history in histories)
    if args.stage == "smoke":
        batches = [1, 16]
        histories = [4096]

    raw_dir = Path(args.output_dir) / "raw"
    writer = Writer(raw_dir)
    existing = writer.existing() if args.resume else set()
    bm.write_environment(str(raw_dir), str(Path(__file__).resolve().parent),
                         args.seed)
    design = {
        "histories": histories, "batches": batches,
        "passes": [name for name, _ in PASSES], "states": STATES,
        "topk": bm.TOPK, "l2_bytes": bm.L2_BYTES,
        "flush_bytes": bm.FLUSH_BUF_BYTES, "warmup": args.warmup,
        "repeat": args.repeat, "seed": args.seed,
        "interpretation": ("hot-resident and flushed are controlled proxies; "
                           "both execute one flush and one prime in opposite "
                           "order; hardware L2 hit/miss rates are not observed"),
    }
    with (raw_dir / "design.json").open("w") as f:
        json.dump(design, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

    flush_buf = torch.ones(bm.FLUSH_BUF_BYTES // 4, dtype=torch.float32,
                           device="cuda")
    pass_defs = [("ascending", sorted(histories)),
                 ("descending", sorted(histories, reverse=True))]
    for pass_order, ordered_histories in pass_defs:
        for history in ordered_histories:
            ordered_batches = list(batches)
            random.Random(args.seed + history
                          + (0 if pass_order == "ascending" else 1)).shuffle(
                              ordered_batches)
            for batch in ordered_batches:
                run_case(writer, existing, pass_order, batch, history, args,
                         flush_buf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
