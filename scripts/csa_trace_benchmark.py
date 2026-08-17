#!/usr/bin/env python3
"""Replay sampled DeepSeek-V4 CSA activation windows in the Q8KV8 kernel."""

import argparse
import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from scripts import benchmark as bm
from scripts import prefill_runtime as runtime


PASS_ORDERS = ("ascending", "descending")
PATTERNS = ("original_strided", "csa_trace_replay")
DEFAULT_BATCHES = (1, 8, 32, 128)
DEFAULT_HISTORIES = (8192, 16384, 32768, 65536)
DEFAULT_QUERIES = (64, 256, 1024, 2048)
FIELDS = [
    "case_id", "status", "skip_reason", "stage", "pattern", "pass_order",
    "cache_state", "B", "H", "Q", "topk", "warmup", "repeat", "seed",
    "latency_ms_median", "latency_ms_p5", "latency_ms_p95",
    "latency_ms_mean", "latency_ms_std", "tflops", "pairs", "flops",
    "achieved_adjacent_overlap", "unique_kv", "reuse_factor",
    "selected_kv_working_set_bytes", "selected_kv_working_set_vs_l2",
    "c4_consecutive_fraction", "trace_rows", "hidden_layer_ids",
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
    values = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    if not values or any(value <= 0 for value in values):
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


def expand_c4_window(raw_indices, history):
    """Map 512 compressed entries to 2048 token addresses in [0, history)."""
    raw = np.asarray(raw_indices, dtype=np.int32)
    if raw.ndim != 2 or raw.shape[1] != 512:
        raise ValueError(f"expected [Q,512] C4 window, got {raw.shape}")
    if np.any(raw < 0):
        raise ValueError("C4 replay window contains padding")
    ordered = np.sort(raw, axis=-1)
    if np.any(np.diff(ordered, axis=-1) == 0):
        raise ValueError("C4 replay row contains duplicate entries")
    expanded = (
        raw[..., None] * 4 + np.arange(4, dtype=np.int32)
    ).reshape(raw.shape[0], bm.TOPK)
    if int(expanded.max()) >= history:
        raise ValueError(
            f"expanded C4 index {expanded.max()} exceeds history {history}"
        )
    return np.ascontiguousarray(expanded)


def adjacent_overlap_c4(raw):
    if len(raw) < 2:
        return 1.0
    combined = np.sort(np.concatenate((raw[:-1], raw[1:]), axis=-1), axis=-1)
    intersection = (combined[:, 1:] == combined[:, :-1]).sum()
    return float(intersection / ((len(raw) - 1) * raw.shape[-1]))


def consecutive_c4_fraction(raw):
    ordered = np.sort(raw, axis=-1)
    return float((np.diff(ordered, axis=-1) == 1).mean())


class TraceCorpus:
    def __init__(self, source):
        self.rows = []
        for npz_path in sorted(Path(source).rglob("indexer_topk.npz")):
            metadata = json.loads(npz_path.with_name("metadata.json").read_text())
            with np.load(npz_path) as archive:
                tensor = archive["indexer_topk"]
            if tensor.dtype != np.int32 or tensor.ndim != 3:
                raise ValueError(f"invalid trace tensor {npz_path}: {tensor.shape}")
            if tensor.shape[1:] != (21, 512):
                raise ValueError(f"unsupported trace tensor {npz_path}: {tensor.shape}")
            self.rows.append({
                "row": int(metadata["source"]["row_index"]),
                "instance_id": metadata["source"]["instance_id"],
                "tensor": tensor,
                "path": str(npz_path.resolve()),
            })
            print(
                f"loaded trace row={self.rows[-1]['row']:04d} "
                f"shape={tensor.shape}", flush=True,
            )
        if not self.rows:
            raise FileNotFoundError(f"no sampled traces under {source}")

    def eligible(self, history):
        rows = [row for row in self.rows if row["tensor"].shape[0] >= history]
        if not rows:
            raise ValueError(f"no sampled trace reaches history={history}")
        return rows

    def build_indices(self, B, H, Q, seed):
        candidates = self.eligible(H)
        expanded_windows = []
        trace_rows = []
        hidden_layers = []
        overlaps = []
        unique_counts = []
        consecutive_fractions = []
        variants_per_end = len(candidates) * 21
        for batch in range(B):
            variant = (seed + batch * 17) % variants_per_end
            source = candidates[variant % len(candidates)]
            slot = (variant // len(candidates)) % 21
            cycle = batch // variants_per_end
            retreat = cycle * max(1, Q // 2)
            end = H - retreat
            if end - Q < 2048:
                end = H
            raw = np.ascontiguousarray(source["tensor"][end - Q:end, slot, :])
            expanded = expand_c4_window(raw, H)
            expanded_windows.append(expanded)
            trace_rows.append(source["row"])
            hidden_layers.append(2 * (slot + 1))
            overlaps.append(adjacent_overlap_c4(raw))
            unique_counts.append(int(np.unique(raw).size) * 4)
            consecutive_fractions.append(consecutive_c4_fraction(raw))

        physical = np.stack(expanded_windows)
        offsets = (
            np.arange(B, dtype=np.int32) * (H + Q)
        )[:, None, None]
        physical += offsets
        indices = torch.from_numpy(physical).to(device="cuda", non_blocking=False)
        indices = indices.view(B * Q, 1, bm.TOPK).contiguous()
        # Sum per batch rather than per distinct template: physical KV segments
        # are disjoint across batch elements.
        return {
            "indices": indices,
            "adjacent_overlap": float(np.mean(overlaps)),
            "unique_kv": int(sum(unique_counts)),
            "c4_consecutive_fraction": float(np.mean(consecutive_fractions)),
            "trace_rows": trace_rows,
            "hidden_layer_ids": hidden_layers,
        }


def case_id(pattern, pass_order, B, H, Q):
    return f"b{B}_h{H}_q{Q}_{pattern}_{pass_order}_steady"


def case_order(pass_order):
    return PATTERNS if pass_order == "ascending" else tuple(reversed(PATTERNS))


def base_row(stage, pattern, pass_order, B, H, Q, args):
    return {
        "case_id": case_id(pattern, pass_order, B, H, Q),
        "status": "ok", "skip_reason": None, "stage": stage,
        "pattern": pattern, "pass_order": pass_order,
        "cache_state": "steady", "B": B, "H": H, "Q": Q,
        "topk": bm.TOPK, "warmup": args.warmup, "repeat": args.repeat,
        "seed": args.seed, "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def skip_shape(writer, terminal, stage, B, H, Q, status, reason, args):
    for pass_order in PASS_ORDERS:
        for pattern in case_order(pass_order):
            row = base_row(stage, pattern, pass_order, B, H, Q, args)
            if row["case_id"] in terminal:
                continue
            row["status"] = status
            row["skip_reason"] = reason
            writer.append(row)


def run_shape(writer, terminal, corpus, stage, B, H, Q, args):
    pending = [
        (pattern, pass_order)
        for pass_order in PASS_ORDERS
        for pattern in case_order(pass_order)
        if case_id(pattern, pass_order, B, H, Q) not in terminal
    ]
    if not pending:
        return
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    inputs = None
    try:
        inputs, alloc_ms, quant_ms = runtime.allocate_inputs(B, H, Q, args.seed)
    except torch.OutOfMemoryError as error:
        skip_shape(
            writer, terminal, stage, B, H, Q, "skipped_memory_limit",
            f"input allocation OOM: {error}", args,
        )
        del inputs
        torch.cuda.empty_cache()
        return

    for pattern, pass_order in pending:
        row = base_row(stage, pattern, pass_order, B, H, Q, args)
        row.update(setup_alloc_ms=alloc_ms, setup_quant_ms=quant_ms)
        indices = fn = None
        try:
            t0 = time.perf_counter()
            if pattern == "original_strided":
                indices, _ = bm._prefill_indices(
                    B, H, Q, bm.TOPK, args.seed, False
                )
                achieved = runtime.adjacent_overlap(indices, B, Q)
                unique_kv = runtime.unique_kv(indices, B, H, Q)
                c4_fraction = None
                trace_rows = None
                hidden_layers = None
            else:
                trace = corpus.build_indices(B, H, Q, args.seed)
                indices = trace["indices"]
                achieved = trace["adjacent_overlap"]
                unique_kv = trace["unique_kv"]
                c4_fraction = trace["c4_consecutive_fraction"]
                trace_rows = json.dumps(trace["trace_rows"], separators=(",", ":"))
                hidden_layers = json.dumps(
                    trace["hidden_layer_ids"], separators=(",", ":")
                )
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
            working_set = unique_kv * bm.D_QK
            row.update(
                latency_ms_median=stats["median"], latency_ms_p5=stats["p5"],
                latency_ms_p95=stats["p95"], latency_ms_mean=stats["mean"],
                latency_ms_std=stats["std"],
                tflops=flops / (stats["median"] / 1000) / 1e12,
                pairs=pairs, flops=flops,
                achieved_adjacent_overlap=achieved, unique_kv=unique_kv,
                reuse_factor=pairs / unique_kv,
                selected_kv_working_set_bytes=working_set,
                selected_kv_working_set_vs_l2=working_set / bm.L2_BYTES,
                c4_consecutive_fraction=c4_fraction,
                trace_rows=trace_rows, hidden_layer_ids=hidden_layers,
                setup_indices_ms=indices_ms,
                peak_mem_alloc_bytes=torch.cuda.max_memory_allocated(),
            )
            for key, value in telemetry_before.items():
                row[f"{key}_before"] = value
            for key, value in telemetry_after.items():
                row[f"{key}_after"] = value
        except torch.OutOfMemoryError as error:
            row["status"] = "skipped_memory_limit"
            row["skip_reason"] = f"index/kernel OOM: {error}"
        except Exception as error:
            row["status"] = "failed"
            row["skip_reason"] = f"{type(error).__name__}: {error}"
        writer.append(row)
        print(
            f"{row['case_id']}: {row['status']} overlap="
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
    parser.add_argument("--trace-source", default="local_data/csa_trace_source")
    parser.add_argument(
        "--reference-results",
        default="artifacts/data/random_baseline/raw/results.jsonl",
    )
    parser.add_argument("--output-dir", default="runs/csa_trace_replay")
    parser.add_argument("--batches")
    parser.add_argument("--histories")
    parser.add_argument("--prefill-lengths")
    parser.add_argument("--shape-triples", help="exact B:H:Q triples")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260803)
    parser.add_argument("--time-budget-minutes", type=float, default=120.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("Q8KV8 trace replay requires an SM90 CUDA GPU")
    reference = Path(args.reference_results)
    runtime.validate_runtime(reference)
    available = set(runtime.load_reference_shapes(reference))
    exact_shapes = parse_shape_triples(args.shape_triples)
    if exact_shapes is not None:
        requested = exact_shapes
    elif args.stage == "smoke":
        requested = {(1, 16384, 256), (8, 16384, 256)}
    else:
        batches = parse_int_list(args.batches, DEFAULT_BATCHES)
        histories = parse_int_list(args.histories, DEFAULT_HISTORIES)
        queries = parse_int_list(args.prefill_lengths, DEFAULT_QUERIES)
        requested = {(B, H, Q) for B in batches for H in histories for Q in queries}
    shapes = sorted(requested & available, key=lambda item: (item[0], item[2], item[1]))
    if not shapes:
        raise RuntimeError("no requested shapes are successful in the reference grid")

    output = Path(args.output_dir)
    raw = output / "raw"
    results_path = raw / "results.jsonl"
    if results_path.exists() and not args.resume:
        raise FileExistsError(f"{results_path} exists; use --resume or a new output")
    corpus = TraceCorpus(args.trace_source)
    for _, H, Q in shapes:
        if Q > H - 2048:
            raise ValueError(f"trace replay requires Q <= H-2048, got H={H}, Q={Q}")
        corpus.eligible(H)

    design = {
        "patterns": list(PATTERNS), "pass_orders": list(PASS_ORDERS),
        "mapping": "each real C4 entry expands to four consecutive KV tokens",
        "trace_window": "real tensor rows [H-Q,H), all expanded indices in [0,H)",
        "trace_source": str(Path(args.trace_source).resolve()),
        "reference_results": str(reference.resolve()),
        "shapes": [{"B": B, "H": H, "Q": Q} for B, H, Q in shapes],
        "shape_count": len(shapes), "case_count": 4 * len(shapes),
        "warmup": args.warmup, "repeat": args.repeat, "seed": args.seed,
    }
    raw.mkdir(parents=True, exist_ok=True)
    design_path = raw / "design.json"
    if args.resume and design_path.exists():
        previous = json.loads(design_path.read_text())
        for key in ("patterns", "pass_orders", "mapping", "trace_window",
                    "trace_source", "reference_results", "shapes", "warmup",
                    "repeat", "seed"):
            if previous.get(key) != design.get(key):
                raise ValueError(f"resume design differs in {key}")
    with design_path.open("w") as handle:
        json.dump(design, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    bm.write_environment(str(raw), str(Path(__file__).resolve().parent), args.seed)

    writer = Writer(raw)
    terminal = writer.terminal_case_ids() if args.resume else set()
    started = time.monotonic()
    for index, (B, H, Q) in enumerate(shapes):
        if (time.monotonic() - started) / 60 >= args.time_budget_minutes:
            for remaining in shapes[index:]:
                skip_shape(
                    writer, terminal, args.stage, *remaining,
                    "skipped_time_budget", "time budget reached", args,
                )
            break
        run_shape(writer, terminal, corpus, args.stage, B, H, Q, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
