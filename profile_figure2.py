#!/usr/bin/env python3
"""Profile successful cases using the ECHO Figure 2 / FlashMLA convention.

Dense uses the main flash_fwd_splitkv_mla_kernel duration and rectangular
B*Q*cache_seqlen FLOPs, matching FlashMLA's official dense benchmark. Sparse
decode uses split-to-combine E2E duration and selected-token FLOPs; sparse
prefill uses sparse_attn_fwd duration and selected-token FLOPs.
"""

import argparse
import csv
import json
import os
import sys
import time

import torch

import benchmark as bm


PROJECT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT = os.path.join(
    PROJECT, "assets", "raw", "history_scaling_results.jsonl")
DEFAULT_OUTPUT = os.path.join(
    PROJECT, "assets", "raw", "figure2_metrics.jsonl")


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def case_from_row(row):
    backend = row["backend"]
    if backend in ("flashmla_dense_bf16",
                   "flashmla_dense_bf16_multiquery"):
        role = "dense"
    elif backend in ("flashmla_sparse_fp8", "flashmla_sparse_bf16"):
        role = "sparse"
    else:
        raise ValueError(f"unsupported backend: {backend}")
    return {
        "kind": row["kind"], "backend": role, "cache_state": "steady",
        "B": row["B"], "H": row["H"], "Q": row["Q"],
        "topk_variant": None,
    }


def initialize_profiler():
    # PyTorch 2.9/CUDA 13 can return an empty first CUDA profile in a process.
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]):
        torch.ones(1, device="cuda").add_(1)
        torch.cuda.synchronize()


def profile_events(fn, repeat):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(repeat):
            fn()
        torch.cuda.synchronize()
    return list(prof.events())


def matching(events, substring):
    return [e for e in events if substring in e.name]


def extract_latency_us(row, events, repeat):
    backend = row["backend"]
    if backend in ("flashmla_dense_bf16",
                   "flashmla_dense_bf16_multiquery"):
        kernels = matching(events, "flash_fwd_splitkv_mla_kernel")
        kernels = [e for e in kernels if "sparse" not in e.name]
        if len(kernels) != repeat:
            raise RuntimeError(f"dense split kernel count {len(kernels)} != {repeat}")
        return (sum(e.device_time_total for e in kernels) / repeat,
                "dense split-kv kernel only", kernels[0].name)

    if backend == "flashmla_sparse_fp8":
        split = matching(events, "flash_fwd_splitkv_mla_fp8_sparse_kernel")
        combine = matching(events, "flash_fwd_mla_combine_kernel")
        if len(split) != repeat:
            raise RuntimeError(f"sparse split kernel count {len(split)} != {repeat}")
        if not combine:
            latency = sum(e.device_time_total for e in split) / repeat
            return latency, "sparse split-kv (no combine launched)", split[0].name
        if len(combine) != repeat:
            raise RuntimeError(
                f"sparse combine kernel count {len(combine)} != {repeat}")
        spans = [combine[i].time_range.end - split[i].time_range.start
                 for i in range(repeat)]
        return (sum(spans) / repeat, "sparse split-to-combine E2E",
                f"{split[0].name} -> {combine[0].name}")

    if backend == "flashmla_sparse_bf16":
        kernels = matching(events, "sparse_attn_fwd_kernel")
        if len(kernels) != repeat:
            raise RuntimeError(
                f"sparse prefill kernel count {len(kernels)} != {repeat}")
        return (sum(e.device_time_total for e in kernels) / repeat,
                "sparse prefill kernel only", kernels[0].name)

    raise ValueError(backend)


def figure2_flops(row):
    if row["backend"] == "flashmla_dense_bf16_multiquery":
        pairs = row["B"] * row["Q"] * (row["H"] + row["Q"])
        return 2 * pairs * bm.H_Q * (bm.D_QK + bm.D_V), pairs, \
            "rectangular B*Q*(H+Q), FlashMLA dense benchmark convention"
    return row["flops"], row["pairs"], \
        "selected/attended pairs, FlashMLA sparse benchmark convention"


def append_fsync(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


def rewrite_csv(jsonl_path):
    rows = load_jsonl(jsonl_path)
    latest = {row["case_id"]: row for row in rows}
    rows = sorted(latest.values(), key=lambda row: row["case_id"])
    path = os.path.splitext(jsonl_path)[0] + ".csv"
    fields = sorted({key for row in rows for key in row})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                        help="profile only the first N successful cases")
    parser.add_argument("--case-ids", default=None,
                        help="comma-separated exact case IDs to profile")
    return parser.parse_args()


def main():
    args = parse_args()
    source = load_jsonl(args.input)
    successful = [row for row in source if row.get("status") == "ok"]
    if args.case_ids:
        selected = set(args.case_ids.split(","))
        successful = [row for row in successful if row["case_id"] in selected]
        found = {row["case_id"] for row in successful}
        missing = selected - found
        if missing:
            raise ValueError(f"unknown or unsuccessful case IDs: {sorted(missing)}")
    if args.limit is not None:
        successful = successful[:args.limit]
    existing = set()
    if args.resume and os.path.exists(args.output):
        existing = {row["case_id"] for row in load_jsonl(args.output)
                    if row.get("status") == "ok"}
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    initialize_profiler()
    counts = {"ok": 0, "failed": 0, "resumed": 0}
    start = time.monotonic()

    for index, source_row in enumerate(successful, 1):
        case_id = source_row["case_id"]
        if case_id in existing:
            counts["resumed"] += 1
            continue
        case = case_from_row(source_row)
        fn = holders = None
        result = {
            "case_id": case_id, "status": "ok",
            "kind": source_row["kind"], "backend": source_row["backend"],
            "B": source_row["B"], "H": source_row["H"], "Q": source_row["Q"],
            "profile_repeat": args.repeat, "cache_state": "steady",
            "effective_e2e_tflops": source_row["tflops"],
            "effective_e2e_latency_ms": source_row["latency_ms_median"],
        }
        try:
            fn, _, _, holders = bm.setup_case(case, source_row["seed"])
            events = profile_events(fn, args.repeat)
            latency_us, scope, kernel_name = extract_latency_us(
                source_row, events, args.repeat)
            flops, pairs, formula = figure2_flops(source_row)
            result.update(
                figure2_latency_us=latency_us,
                figure2_hardware_flops=flops,
                figure2_pairs=pairs,
                figure2_hardware_tflops=flops / (latency_us * 1e-6) / 1e12,
                figure2_timing_scope=scope,
                figure2_flops_formula=formula,
                profiled_kernel_name=kernel_name,
            )
            counts["ok"] += 1
            print(f"[{index}/{len(successful)}] {case_id}: "
                  f"{result['figure2_hardware_tflops']:.1f} TFLOPS", flush=True)
        except Exception as error:
            result["status"] = "failed"
            result["error"] = f"{type(error).__name__}: {error}"
            counts["failed"] += 1
            print(f"[{index}/{len(successful)}] {case_id}: FAILED "
                  f"{result['error']}", flush=True)
        finally:
            del fn, holders
            torch.cuda.empty_cache()
        append_fsync(args.output, result)

    csv_path = rewrite_csv(args.output)
    print(f"done: {counts}, elapsed_min={(time.monotonic() - start) / 60:.2f}")
    print(f"wrote {args.output} and {csv_path}")
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
