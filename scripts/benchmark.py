#!/usr/bin/env python3
"""Sparse MLA benchmark for a single NVIDIA SM90 GPU.

Backends
--------
- decode dense        : FlashMLA BF16 paged-KV decode (SM90)
- decode sparse       : FlashMLA FP8 sparse decode, 656-byte KV-with-scales layout
- decode full-topk    : same FP8 sparse kernel with topk covering the full history
- prefill dense       : FlashInfer BatchMLAPagedAttentionWrapper(backend="fa3"),
                        causal incremental prefill
- prefill sparse      : FlashMLA flash_mla_sparse_fwd (BF16)
- prefill sparse-q8   : SGLang SM90 sparse MLA Q8xKV8 prefill (FP8)
- prefill full-topk   : same sparse kernel with topk covering the full visible range

Fixed shape: h_q=64, d_qk=576, d_v=512, topk=2048.
softmax_scale = 1/sqrt(256) — GLM-5.1 qk_head_dim=256 (576 is only the
absorbed Q/K compute dim).
"""

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone

import torch

# CUDA 13 ships CCCL (libcu++) under include/cccl; FlashInfer JIT and other
# host-compiled extensions need it on the include path.
os.environ["CPLUS_INCLUDE_PATH"] = (
    "/usr/local/cuda-13.0/include/cccl"
    + (":" + os.environ["CPLUS_INCLUDE_PATH"]
       if os.environ.get("CPLUS_INCLUDE_PATH") else ""))
# The venv's own bin/ (ninja et al.) must be on PATH for JIT builds even when
# the interpreter is invoked as .venv/bin/python without activation.
_venv_bin = os.path.dirname(sys.executable)
if _venv_bin not in os.environ.get("PATH", "").split(":"):
    os.environ["PATH"] = _venv_bin + ":" + os.environ.get("PATH", "")

# ---------------------------------------------------------------- constants
H_Q = 64
D_QK = 576
D_V = 512
TOPK = 2048
SOFTMAX_SCALE = 1.0 / math.sqrt(256.0)  # GLM-5.1 qk_head_dim = 256
PAGE = 64                     # FlashMLA paged decode requires page_block_size=64
FP8_TOKEN_BYTES = 656         # 512 fp8 nope + 4 fp32 scales + 64 bf16 rope
H100_SXM_BF16_DENSE_TFLOPS = 989.5  # reference denominator only, NOT H800 spec
FLUSH_BUF_BYTES = 256 * 1024 * 1024  # 256 MiB >> 50 MiB L2
L2_BYTES = 50 * 1024 * 1024
FULLTOPK_MAX_VISIBLE = 8192
FULLTOPK_MAX_INDICES_BYTES = 4 * 1024**3
QUANT_CHUNK_TOKENS = 1 << 20  # bound FP8-cache setup temporaries to ~3 GiB
Q8_FILL_CHUNK_ROWS = 1 << 17  # bound BF16 staging for contiguous FP8 tensors

BATCHES_FULL = [1, 2, 4, 8, 16, 32, 64, 128, 256]
DECODE_HISTORIES_FULL = [
    2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288,
]
PREFILL_HISTORIES_FULL = DECODE_HISTORIES_FULL
PREFILL_LENGTHS_FULL = [
    2, 4, 8, 16, 32, 64, 128, 256, 512,
    1024, 2048, 4096, 8192, 16384, 32768,
]

DECODE_BATCHES_SMOKE = [1, 32, 128]
HISTORIES_SMOKE = [4096, 32768]
PREFILL_BQ_SMOKE = [(1, 16), (8, 256)]

# cold对照 subsets used when --stage full --cache-state both
COLD_DECODE_BATCHES = [1, 32, 128]
COLD_DECODE_HISTORIES = [32768, 65536]
COLD_PREFILL_BQ = [(1, 1024), (8, 256), (64, 16)]
COLD_PREFILL_HISTORIES = [32768, 65536]

# anchors run before and after the full grid
ANCHOR_DECODE = dict(B=64, H=32768)
ANCHOR_PREFILL = dict(B=1, H=32768, Q=1024)
ANCHOR_DRIFT_TOL = 0.05

RESULT_FIELDS = [
    "case_id", "stage", "kind", "backend", "cache_state", "status", "skip_reason",
    "B", "H", "Q", "topk", "h_q", "d_qk", "d_v",
    "warmup", "repeat",
    "latency_ms_median", "latency_ms_p5", "latency_ms_p95",
    "latency_ms_mean", "latency_ms_std",
    "setup_alloc_ms", "setup_indices_ms", "setup_quant_ms", "setup_metadata_ms",
    "setup_total_ms",
    "flops", "pairs", "tflops", "pairs_per_s",
    "est_logical_bytes", "est_logical_gbps",
    "utilization_vs_h100_sxm_bf16_dense_peak",
    "observed_utilization_vs_platform_max",
    "peak_mem_alloc_bytes", "peak_mem_reserved_bytes",
    "temp_c_before", "power_w_before", "sm_clock_mhz_before", "mem_clock_mhz_before",
    "temp_c_after", "power_w_after", "sm_clock_mhz_after", "mem_clock_mhz_after",
    "seed", "timestamp",
]


# ---------------------------------------------------------------- fp8 layout
# Vendored from FlashMLA @ 15f13e5 tests/quant.py (official V3.2 sparse FP8
# KV-cache layout helpers), kept byte-compatible with the kernel contract.
def _cast_scale_inv_to_ue8m0(scale_inv: torch.Tensor) -> torch.Tensor:
    scale_inv = torch.clamp(scale_inv, min=2**-127, max=448.0)
    return torch.exp2(torch.ceil(torch.log2(scale_inv)))


def quantize_k_cache(input_k_cache: torch.Tensor) -> torch.Tensor:
    """(num_blocks, block_size, 1, 576) bf16 -> (num_blocks, block_size, 1, 656) fp8.

    Per-token 656-byte layout: 512 B fp8_e4m3 NoPE | 16 B (4 x fp32 scales) |
    128 B bf16 RoPE (unquantized).
    """
    d, d_nope, d_rope, tile_size = 576, 512, 64, 128
    num_tiles = d_nope // tile_size
    assert input_k_cache.shape[-1] == d
    num_blocks, block_size, h_k, _ = input_k_cache.shape
    assert h_k == 1
    input_k_cache = input_k_cache.squeeze(2)  # [num_blocks, block_size, d]
    input_elem_size = input_k_cache.element_size()

    bytes_per_token = d_nope + num_tiles * 4 + input_elem_size * d_rope
    assert bytes_per_token == FP8_TOKEN_BYTES
    result = torch.empty(
        (num_blocks, block_size + 1, bytes_per_token),
        dtype=torch.float8_e4m3fn, device=input_k_cache.device,
    )[:, :block_size, :]
    result_k_nope_part = result[..., :d_nope]
    result_k_scale_factor = result[..., d_nope:d_nope + num_tiles * 4].view(torch.float32)
    result_k_rope_part = result[..., d_nope + num_tiles * 4:].view(input_k_cache.dtype)
    result_k_rope_part[:] = input_k_cache[..., d_nope:]

    for tile_idx in range(num_tiles):
        cur_scale_inv = torch.abs(
            input_k_cache[..., tile_idx * tile_size:(tile_idx + 1) * tile_size]
        ).max(dim=-1).values.float() / 448.0
        cur_scale_inv = _cast_scale_inv_to_ue8m0(cur_scale_inv)
        result_k_scale_factor[:, :, tile_idx] = cur_scale_inv
        cur_scale_inv.unsqueeze_(-1)
        result_k_nope_part[..., tile_idx * tile_size:(tile_idx + 1) * tile_size] = (
            input_k_cache[..., tile_idx * tile_size:(tile_idx + 1) * tile_size].float()
            / cur_scale_inv.float()
        ).to(torch.float8_e4m3fn)
    return result.view(num_blocks, block_size, 1, -1)


def fill_random_k_cache_chunked(result: torch.Tensor,
                                gen: torch.Generator) -> None:
    """Fill a contiguous FP8 cache without materializing full BF16 staging.

    Quantization is setup-only. Generating at most QUANT_CHUNK_TOKENS BF16
    tokens at a time keeps its FP32 expression temporaries independent of the
    total history length.
    """
    num_blocks = result.shape[0]
    chunk_blocks = max(1, QUANT_CHUNK_TOKENS // PAGE)
    for begin in range(0, num_blocks, chunk_blocks):
        end = min(num_blocks, begin + chunk_blocks)
        bf16_chunk = _randn((end - begin, PAGE, 1, D_QK), gen)
        quantized_chunk = quantize_k_cache(bf16_chunk)
        result[begin:end].copy_(quantized_chunk)
        del bf16_chunk, quantized_chunk


def dequantize_k_cache(quant_k_cache: torch.Tensor) -> torch.Tensor:
    """Inverse of quantize_k_cache -> (num_blocks, block_size, 1, 576) bf16."""
    d, d_nope, d_rope, tile_size = 576, 512, 64, 128
    num_tiles = d_nope // tile_size
    num_blocks, block_size, h_k, bytes_per_token = quant_k_cache.shape
    assert h_k == 1 and bytes_per_token == FP8_TOKEN_BYTES
    flat = quant_k_cache.view(num_blocks, block_size, bytes_per_token)
    nope = flat[..., :d_nope]
    scales = flat[..., d_nope:d_nope + num_tiles * 4].view(torch.float32)
    rope = flat[..., d_nope + num_tiles * 4:].view(torch.bfloat16)
    out = torch.empty(num_blocks, block_size, d, dtype=torch.bfloat16,
                      device=quant_k_cache.device)
    for tile_idx in range(num_tiles):
        lo, hi = tile_idx * tile_size, (tile_idx + 1) * tile_size
        out[..., lo:hi] = (nope[..., lo:hi].float()
                           * scales[..., tile_idx:tile_idx + 1]).to(torch.bfloat16)
    out[..., d_nope:] = rope
    return out.view(num_blocks, block_size, 1, d)


def abs_indices2indices_in_kvcache(abs_indices: torch.Tensor,
                                   block_table: torch.Tensor) -> torch.Tensor:
    """Per-request absolute token index -> flat paged-cache index, -1 preserved.

    abs_indices: (b, ..., topk) int64/int32; block_table: (b, max_blocks) int.
    """
    out = torch.full_like(abs_indices, -1, dtype=torch.int32)
    valid = abs_indices >= 0
    idx = abs_indices.clamp(min=0).long()
    for b in range(abs_indices.shape[0]):
        bt = block_table[b].long()
        blk = torch.div(idx[b], PAGE, rounding_mode="floor")
        off = idx[b] % PAGE
        out[b][valid[b]] = (bt[blk[valid[b]]] * PAGE + off[valid[b]]).to(torch.int32)
    return out


# ---------------------------------------------------------------- indices
def gen_strided_positions(visible: int, n: int, seed: int, batch: int,
                          query: int) -> torch.Tensor:
    """n unique positions in [0, visible): (start + j*stride) mod visible.

    start is determined by (seed, batch, query); stride is the first integer
    >= ceil(visible/n) coprime with visible, guaranteeing uniqueness.
    """
    assert 1 <= n <= visible
    start = (seed * 1000003 + batch * 10007 + query * 101) % visible
    stride = max(1, -(-visible // n))
    while math.gcd(stride, visible) != 1:
        stride += 1
    j = torch.arange(n, dtype=torch.int64)
    pos = (start + j * stride) % visible
    # invariants: uniqueness, range, causality (positions address only < visible)
    assert torch.unique(pos).numel() == n
    assert int(pos.min()) >= 0 and int(pos.max()) < visible
    return pos


# ---------------------------------------------------------------- telemetry
def gpu_telemetry() -> dict:
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=temperature.gpu,power.draw,clocks.sm,clocks.mem",
             "--format=csv,noheader,nounits"], text=True).strip()
        t, p, sm, mem = [x.strip() for x in out.split(",")]
        return {"temp_c": float(t), "power_w": float(p),
                "sm_clock_mhz": float(sm), "mem_clock_mhz": float(mem)}
    except Exception:
        return {"temp_c": None, "power_w": None,
                "sm_clock_mhz": None, "mem_clock_mhz": None}


# ---------------------------------------------------------------- io
class ResultWriter:
    def __init__(self, raw_dir: str):
        os.makedirs(raw_dir, exist_ok=True)
        self.jsonl_path = os.path.join(raw_dir, "results.jsonl")
        self.csv_path = os.path.join(raw_dir, "results.csv")
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=RESULT_FIELDS).writeheader()

    def append(self, row: dict):
        line = json.dumps(row) + "\n"
        with open(self.jsonl_path, "a") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())
        with open(self.csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=RESULT_FIELDS,
                           extrasaction="ignore").writerow(row)
            f.flush()
            os.fsync(f.fileno())

    def existing_keys(self) -> set:
        keys = set()
        if os.path.exists(self.jsonl_path):
            with open(self.jsonl_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    # Successful and physically terminal memory cases are not
                    # retried. Legacy policy skips (B*Q, 60 GiB) and time-budget
                    # skips still get another chance after a policy change or
                    # a subsequent resume.
                    reason = str(r.get("skip_reason", ""))
                    terminal_memory = reason.startswith((
                        "runtime OOM:",
                        "skipped_memory_limit: estimated live tensors",
                        "skipped_memory_limit: larger history after runtime OOM",
                    ))
                    if r.get("status") == "ok" or terminal_memory:
                        keys.add(r.get("case_id"))
        return keys

    def finalize_observed_utilization(self):
        """Fill per-case observed utilization and atomically rewrite both files."""
        if not os.path.exists(self.jsonl_path):
            return
        rows = []
        with open(self.jsonl_path) as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        maxima = {}
        for row in rows:
            if row.get("status") != "ok" or row.get("case_id", "").startswith("anchor"):
                continue
            key = (row["kind"], row["backend"], row["cache_state"])
            maxima[key] = max(maxima.get(key, 0.0), row["tflops"])
        for row in rows:
            key = (row.get("kind"), row.get("backend"), row.get("cache_state"))
            maximum = maxima.get(key)
            row["observed_utilization_vs_platform_max"] = (
                row.get("tflops") / maximum
                if row.get("status") == "ok" and maximum else None
            )
        json_tmp = self.jsonl_path + ".tmp"
        csv_tmp = self.csv_path + ".tmp"
        with open(json_tmp, "w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
            f.flush()
            os.fsync(f.fileno())
        with open(csv_tmp, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS,
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        os.replace(json_tmp, self.jsonl_path)
        os.replace(csv_tmp, self.csv_path)


def sha256_file(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_environment(raw_dir: str, project_dir: str, seed: int):
    import platform
    env = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "label": "H800 measured (local single-GPU benchmark)",
        "seed": seed,
        "gpu": {}, "driver": {}, "software": {}, "wheels": {}, "notes": [],
    }
    try:
        q = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=name,uuid,driver_version,power.limit,clocks.sm,"
             "clocks.mem,memory.total",
             "--format=csv,noheader"], text=True).strip()
        name, uuid, drv, plimit, sm, mem, memtotal = [x.strip() for x in q.split(",")]
        env["gpu"] = {"name": name, "uuid": uuid, "memory_total": memtotal,
                      "power_limit": plimit, "sm_clock_mhz": sm,
                      "mem_clock_mhz": mem, "l2_cache_bytes": L2_BYTES}
        env["driver"] = {"driver_version": drv}
    except Exception as e:
        env["notes"].append(f"nvidia-smi query failed: {e}")
    try:
        env["driver"]["cuda_runtime_nvcc"] = subprocess.check_output(
            ["/usr/local/cuda-13.0/bin/nvcc", "--version"], text=True
        ).strip().splitlines()[-1].strip()
    except Exception as e:
        env["notes"].append(f"nvcc query failed: {e}")
    import torch as _t
    env["software"] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": _t.__version__, "torch_cuda": _t.version.cuda,
    }
    try:
        import flash_mla
        env["software"]["flash_mla"] = flash_mla.__version__
        env["software"]["flash_mla_commit"] = subprocess.check_output(
            ["git", "-C", os.path.join(project_dir, "third_party/FlashMLA"),
             "rev-parse", "HEAD"], text=True).strip()
    except Exception as e:
        env["notes"].append(f"flash_mla version failed: {e}")
    try:
        import flashinfer
        env["software"]["flashinfer"] = flashinfer.__version__
    except Exception as e:
        env["notes"].append(f"flashinfer version failed: {e}")
    try:
        from scripts.sglang_q8kv8 import source_manifest
        env["software"]["sglang_q8kv8"] = source_manifest()
    except Exception as e:
        env["notes"].append(f"sglang q8kv8 source verification failed: {e}")
    wheels_dir = os.path.join(project_dir, ".cache", "wheels")
    if os.path.isdir(wheels_dir):
        for fn in sorted(os.listdir(wheels_dir)):
            if fn.endswith(".whl"):
                env["wheels"][fn] = sha256_file(os.path.join(wheels_dir, fn))
    try:
        pip = os.path.join(project_dir, ".venv", "bin", "pip")
        env["software"]["pip_freeze"] = subprocess.check_output(
            [pip, "freeze"], text=True).strip().splitlines()
    except Exception as e:
        env["notes"].append(f"pip freeze failed: {e}")
    path = os.path.join(raw_dir, "environment.json")
    with open(path, "w") as f:
        json.dump(env, f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    return env


# ---------------------------------------------------------------- accounting
def flops_dense_decode(B, H):
    return 2 * B * 1 * H_Q * H * (D_QK + D_V)


def flops_sparse_decode(B, topk):
    return 2 * B * 1 * H_Q * topk * (D_QK + D_V)


def flops_dense_prefill(B, H, Q):
    visible_sum = B * (Q * H + Q * (Q + 1) // 2)  # sum_i (H+i+1), i in [0,Q)
    return 2 * H_Q * (D_QK + D_V) * visible_sum


def flops_sparse_prefill(topk_sum):
    return 2 * H_Q * (D_QK + D_V) * topk_sum


def bytes_dense_decode(B, H):
    return B * (1 * H_Q * D_QK * 2 + H * D_QK * 2 + 1 * H_Q * D_V * 2)


def bytes_sparse_decode(B, n_unique):
    return 2 * B * 1 * H_Q * D_QK + n_unique * FP8_TOKEN_BYTES + 2 * B * 1 * H_Q * D_V


def bytes_dense_prefill(B, H, Q):
    total_q = B * Q
    visible_sum = B * (Q * H + Q * (Q + 1) // 2)
    return 2 * total_q * H_Q * D_QK + visible_sum * D_QK * 2 + 2 * total_q * H_Q * D_V


def bytes_sparse_prefill(n_valid, total_q):
    return n_valid * D_QK * 2 + total_q * H_Q * (D_QK + D_V) * 2


def bytes_sparse_prefill_q8kv8(n_valid, total_q):
    return n_valid * D_QK + total_q * H_Q * (D_QK + D_V * 2)


def est_live_bytes(kind, backend, B, H, Q):
    """Rough live-tensor estimate for skip logic (bytes)."""
    blocks = B * (-(-H // PAGE))
    if kind == "decode":
        q_out = B * H_Q * (D_QK + D_V) * 2
        if backend == "dense":
            return blocks * PAGE * D_QK * 2 + q_out + B * TOPK * 4
        kv_fp8 = blocks * PAGE * FP8_TOKEN_BYTES
        chunk_tokens = min(blocks * PAGE, QUANT_CHUNK_TOKENS)
        # One BF16 chunk, its temporary FP8 result, and two conservative FP32
        # tile intermediates. None scales with the total history after this.
        quant_tmp = chunk_tokens * (D_QK * 2 + FP8_TOKEN_BYTES + 2 * 128 * 4)
        topk = TOPK if backend == "sparse" else H
        return kv_fp8 + quant_tmp + B * topk * 4 + q_out
    else:
        total_kv = B * (H + Q)
        total_q = B * Q
        q_out = total_q * H_Q * (D_QK + D_V) * 2
        if backend == "dense_flashmla":
            padded_kv = B * (-(-(H + Q) // PAGE)) * PAGE * D_QK * 2
            num_sm_parts = max(132 // math.ceil(Q * H_Q / 64), 1)
            total_splits = B + num_sm_parts
            accum = total_splits * Q * H_Q * (D_V + 1) * 4
            return padded_kv + q_out + accum
        if backend == "dense":
            return total_kv * (512 + 64) * 2 + q_out + 136 * 1024 * 1024
        topk = TOPK if backend in ("sparse", "sparse_q8kv8") else H + Q
        if backend == "sparse_q8kv8":
            fp8_q_kv = (total_kv + total_q * H_Q) * D_QK
            outputs = total_q * H_Q * (D_V * 2 + 2 * 4)
            indices = total_q * topk * 4
            fill_tmp = min(max(total_kv, total_q * H_Q),
                           Q8_FILL_CHUNK_ROWS) * D_QK * 2
            return fp8_q_kv + outputs + indices + fill_tmp
        return total_kv * D_QK * 2 + total_q * topk * 4 + q_out


# ---------------------------------------------------------------- setups
def _randn(shape, gen, dtype=torch.bfloat16):
    return torch.randn(*shape, generator=gen, device="cuda", dtype=dtype)


def setup_dense_decode(B, H, seed):
    """FlashMLA BF16 paged decode. Returns (fn, account, setup_ms, holders)."""
    t0 = time.perf_counter()
    cache_gen = torch.Generator(device="cuda").manual_seed(seed)
    meta_gen = torch.Generator(device="cuda").manual_seed(seed + 1)
    blocks_per_seq = -(-H // PAGE)
    num_blocks = B * blocks_per_seq
    k_cache = _randn((num_blocks, PAGE, 1, D_QK), cache_gen)
    # shuffled block table so address conversion is genuinely exercised
    block_table = torch.randperm(num_blocks, generator=meta_gen, device="cuda") \
        .view(B, blocks_per_seq).to(torch.int32).contiguous()
    cache_seqlens = torch.full((B,), H, dtype=torch.int32, device="cuda")
    q = _randn((B, 1, H_Q, D_QK), meta_gen)
    torch.cuda.synchronize()
    t_alloc = time.perf_counter() - t0

    import flash_mla
    sched, _ = flash_mla.get_mla_metadata()

    def fn():
        return flash_mla.flash_mla_with_kvcache(
            q, k_cache, block_table, cache_seqlens, D_V, sched, None,
            softmax_scale=SOFTMAX_SCALE, causal=True)

    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    t_meta = time.perf_counter() - t0

    account = dict(flops=flops_dense_decode(B, H),
                   pairs=B * H,
                   est_bytes=bytes_dense_decode(B, H), topk=None)
    setup = dict(alloc=t_alloc * 1e3, indices=None, quant=None,
                 metadata=t_meta * 1e3)
    holders = dict(q=q, k_cache=k_cache, block_table=block_table,
                   cache_seqlens=cache_seqlens)
    return fn, account, setup, holders


def setup_sparse_decode(B, H, seed, full_topk=False):
    """FlashMLA FP8 sparse decode (656 B/token). Returns same contract."""
    t0 = time.perf_counter()
    cache_gen = torch.Generator(device="cuda").manual_seed(seed)
    meta_gen = torch.Generator(device="cuda").manual_seed(seed + 1)
    blocks_per_seq = -(-H // PAGE)
    num_blocks = B * blocks_per_seq
    kv_fp8 = torch.empty(
        (num_blocks, PAGE, 1, FP8_TOKEN_BYTES),
        dtype=torch.float8_e4m3fn, device="cuda",
    )
    block_table = torch.randperm(num_blocks, generator=meta_gen, device="cuda") \
        .view(B, blocks_per_seq).to(torch.int32).contiguous()
    q = _randn((B, 1, H_Q, D_QK), meta_gen)
    torch.cuda.synchronize()
    t_alloc = time.perf_counter() - t0

    t0 = time.perf_counter()
    topk = H if full_topk else min(TOPK, H)
    rows = []
    for b in range(B):
        pos = gen_strided_positions(H, topk, seed, b, 0).cuda()
        rows.append(pos)
    abs_idx = torch.stack(rows).view(B, 1, topk)
    indices = abs_indices2indices_in_kvcache(abs_idx, block_table)
    indices = indices.contiguous()
    torch.cuda.synchronize()
    t_idx = time.perf_counter() - t0

    t0 = time.perf_counter()
    fill_random_k_cache_chunked(kv_fp8, cache_gen)
    torch.cuda.synchronize()
    t_quant = time.perf_counter() - t0

    import flash_mla
    sched, _ = flash_mla.get_mla_metadata()

    def fn():
        return flash_mla.flash_mla_with_kvcache(
            q, kv_fp8, None, None, D_V, sched, None,
            SOFTMAX_SCALE, False, True, indices)

    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    t_meta = time.perf_counter() - t0

    n_attended = B * topk
    n_unique = int(indices[indices >= 0].unique().numel())
    account = dict(flops=flops_sparse_decode(B, topk),
                   pairs=n_attended,
                   est_bytes=bytes_sparse_decode(B, n_unique), topk=topk)
    setup = dict(alloc=t_alloc * 1e3, indices=t_idx * 1e3,
                 quant=t_quant * 1e3, metadata=t_meta * 1e3)
    holders = dict(q=q, kv_fp8=kv_fp8, indices=indices,
                   block_table=block_table)
    return fn, account, setup, holders


def setup_dense_prefill(B, H, Q, seed):
    """FlashInfer fa3 MLA causal incremental prefill (page_size=1)."""
    import flashinfer
    t0 = time.perf_counter()
    gen = torch.Generator(device="cuda").manual_seed(seed)
    total_q = B * Q
    total_kv = B * (H + Q)
    ckv = _randn((total_kv, 1, 512), gen)
    kpe = _randn((total_kv, 1, 64), gen)
    q_nope = _randn((total_q, H_Q, 512), gen)
    q_pe = _randn((total_q, H_Q, 64), gen)
    qo_indptr = torch.arange(0, B + 1, dtype=torch.int32, device="cuda") * Q
    kv_indptr = torch.arange(0, B + 1, dtype=torch.int32, device="cuda") * (H + Q)
    kv_indices = torch.arange(total_kv, dtype=torch.int32, device="cuda")
    kv_len = torch.full((B,), H + Q, dtype=torch.int32, device="cuda")
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda")
    output = torch.empty(total_q, H_Q, D_V, dtype=torch.bfloat16,
                         device="cuda")
    torch.cuda.synchronize()
    t_alloc = time.perf_counter() - t0

    t0 = time.perf_counter()
    wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(workspace,
                                                           backend="fa3")
    wrapper.plan(qo_indptr, kv_indptr, kv_indices, kv_len, H_Q, 512, 64, 1,
                 True, SOFTMAX_SCALE, torch.bfloat16, torch.bfloat16)
    torch.cuda.synchronize()
    t_plan = time.perf_counter() - t0

    def fn():
        return wrapper.run(q_nope, q_pe, ckv, kpe, out=output)

    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    t_first_run = time.perf_counter() - t0

    account = dict(flops=flops_dense_prefill(B, H, Q),
                   pairs=B * (Q * H + Q * (Q + 1) // 2),
                   est_bytes=bytes_dense_prefill(B, H, Q), topk=None)
    setup = dict(alloc=t_alloc * 1e3, indices=None, quant=None,
                 metadata=(t_plan + t_first_run) * 1e3)
    holders = dict(q_nope=q_nope, q_pe=q_pe, ckv=ckv, kpe=kpe,
                   wrapper=wrapper, workspace=workspace, output=output)
    return fn, account, setup, holders


def setup_flashmla_dense_multiquery(B, H, Q, seed):
    """FlashMLA SM90 dense decode kernel in causal multi-query mode.

    This is not the SM100 dense-MHA prefill kernel. With cache_seqlens=H+Q and
    causal=True, query i sees exactly H+i+1 KV tokens, matching chunked-prefill
    semantics while exercising FlashMLA's native SM90 dense MLA path.
    """
    t0 = time.perf_counter()
    gen = torch.Generator(device="cuda").manual_seed(seed)
    visible = H + Q
    blocks_per_seq = -(-visible // PAGE)
    num_blocks = B * blocks_per_seq
    k_cache = _randn((num_blocks, PAGE, 1, D_QK), gen)
    block_table = torch.randperm(num_blocks, generator=gen, device="cuda") \
        .view(B, blocks_per_seq).to(torch.int32).contiguous()
    cache_seqlens = torch.full((B,), visible, dtype=torch.int32, device="cuda")
    q = _randn((B, Q, H_Q, D_QK), gen)
    torch.cuda.synchronize()
    t_alloc = time.perf_counter() - t0

    import flash_mla
    sched, _ = flash_mla.get_mla_metadata()

    def fn():
        return flash_mla.flash_mla_with_kvcache(
            q, k_cache, block_table, cache_seqlens, D_V, sched, None,
            softmax_scale=SOFTMAX_SCALE, causal=True)

    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    t_meta = time.perf_counter() - t0

    account = dict(flops=flops_dense_prefill(B, H, Q),
                   pairs=B * (Q * H + Q * (Q + 1) // 2),
                   est_bytes=bytes_dense_prefill(B, H, Q), topk=None)
    setup = dict(alloc=t_alloc * 1e3, indices=None, quant=None,
                 metadata=t_meta * 1e3)
    holders = dict(q=q, k_cache=k_cache, block_table=block_table,
                   cache_seqlens=cache_seqlens, sched=sched)
    return fn, account, setup, holders


def _prefill_indices(B, H, Q, topk, seed, full_topk):
    """Generate causal strided indices on GPU without per-query CPU tensors."""
    visible_cpu = [H + i + 1 for i in range(Q)]
    selected_cpu = [v if full_topk else min(TOPK, v) for v in visible_cpu]
    strides_cpu = []
    for visible, selected in zip(visible_cpu, selected_cpu):
        stride = max(1, -(-visible // selected))
        while math.gcd(stride, visible) != 1:
            stride += 1
        strides_cpu.append(stride)
    visible = torch.tensor(visible_cpu, dtype=torch.int64, device="cuda")
    selected = torch.tensor(selected_cpu, dtype=torch.int64, device="cuda")
    strides = torch.tensor(strides_cpu, dtype=torch.int64, device="cuda")
    batch_ids = torch.arange(B, dtype=torch.int64, device="cuda")[:, None]
    query_ids = torch.arange(Q, dtype=torch.int64, device="cuda")[None, :]
    starts = torch.remainder(
        seed * 1000003 + batch_ids * 10007 + query_ids * 101,
        visible[None, :],
    )
    output = torch.full((B, Q, topk), -1, dtype=torch.int32, device="cuda")
    j = torch.arange(topk, dtype=torch.int64, device="cuda")
    for begin in range(0, Q, 32):
        end = min(Q, begin + 32)
        positions = torch.remainder(
            starts[:, begin:end, None]
            + strides[None, begin:end, None] * j[None, None, :],
            visible[None, begin:end, None],
        )
        valid = j[None, None, :] < selected[None, begin:end, None]
        offsets = (batch_ids * (H + Q))[:, :, None]
        output[:, begin:end] = torch.where(
            valid, positions + offsets, -1
        ).to(torch.int32)
    # Counts/range are checked for all rows; uniqueness follows from the
    # coprime stride and is explicitly checked on the two edge rows.
    local = torch.where(
        output >= 0,
        output - (batch_ids * (H + Q)).to(torch.int32)[:, :, None],
        output,
    )
    valid = output >= 0
    assert torch.equal(valid.sum(-1), selected[None, :].expand(B, -1))
    assert not bool(((local >= visible[None, :, None]) & valid).any())
    for b, q_index in ((0, 0), (B - 1, Q - 1)):
        row = local[b, q_index][valid[b, q_index]]
        assert torch.unique(row).numel() == row.numel()
    return output.view(B * Q, 1, topk).contiguous(), selected.repeat(B).to(torch.int32)


def setup_sparse_prefill(B, H, Q, seed, full_topk=False):
    """FlashMLA flash_mla_sparse_fwd, flattened per-batch KV segments."""
    import flash_mla
    t0 = time.perf_counter()
    gen = torch.Generator(device="cuda").manual_seed(seed)
    total_q = B * Q
    total_kv = B * (H + Q)
    kv = _randn((total_kv, 1, D_QK), gen)
    q = _randn((total_q, H_Q, D_QK), gen)
    torch.cuda.synchronize()
    t_alloc = time.perf_counter() - t0

    t0 = time.perf_counter()
    max_visible = H + Q
    topk = max_visible if full_topk else TOPK
    if full_topk:
        # SM90 sparse-prefill kernel requires topk % (2*B_TOPK) == 0
        # (B_TOPK=64); pad with -1 entries up to a multiple of 128.
        topk = -(-topk // 128) * 128
    indices, topk_length = _prefill_indices(B, H, Q, topk, seed, full_topk)
    topk_sum = int(topk_length.sum())
    torch.cuda.synchronize()
    t_idx = time.perf_counter() - t0

    def fn():
        return flash_mla.flash_mla_sparse_fwd(q, kv, indices, SOFTMAX_SCALE,
                                              d_v=D_V,
                                              topk_length=topk_length)

    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    t_metadata = time.perf_counter() - t0

    n_valid = topk_sum
    account = dict(flops=flops_sparse_prefill(topk_sum),
                   pairs=topk_sum,
                   est_bytes=bytes_sparse_prefill(n_valid, total_q), topk=topk)
    setup = dict(alloc=t_alloc * 1e3, indices=t_idx * 1e3, quant=None,
                 metadata=t_metadata * 1e3)
    holders = dict(q=q, kv=kv, indices=indices, topk_length=topk_length)
    return fn, account, setup, holders


def _fill_random_fp8_chunked(tensor, gen, scale=0.05):
    """Initialize a large contiguous FP8 tensor with bounded BF16 staging."""
    flat = tensor.view(-1, tensor.shape[-1])
    for begin in range(0, flat.shape[0], Q8_FILL_CHUNK_ROWS):
        end = min(flat.shape[0], begin + Q8_FILL_CHUNK_ROWS)
        staging = torch.randn(
            (end - begin, flat.shape[1]), dtype=torch.bfloat16,
            device="cuda", generator=gen,
        )
        flat[begin:end].copy_((staging * scale).to(torch.float8_e4m3fn))
        del staging


def setup_sparse_prefill_q8kv8(B, H, Q, seed, full_topk=False):
    """SGLang native SM90 Q8xKV8 sparse MLA prefill kernel."""
    from scripts.sglang_q8kv8 import sparse_mla_q8kv8_prefill_fwd

    t0 = time.perf_counter()
    gen = torch.Generator(device="cuda").manual_seed(seed)
    total_q = B * Q
    total_kv = B * (H + Q)
    kv = torch.empty((total_kv, 1, D_QK), dtype=torch.float8_e4m3fn,
                     device="cuda")
    q = torch.empty((total_q, H_Q, D_QK), dtype=torch.float8_e4m3fn,
                    device="cuda")
    out = torch.empty((total_q, H_Q, D_V), dtype=torch.bfloat16,
                      device="cuda")
    max_logits = torch.empty((total_q, H_Q), dtype=torch.float32,
                             device="cuda")
    lse = torch.empty_like(max_logits)
    q_scale = torch.ones((), dtype=torch.float32, device="cuda")
    kv_scale = torch.ones((), dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()
    t_alloc = time.perf_counter() - t0

    t0 = time.perf_counter()
    _fill_random_fp8_chunked(q, gen)
    _fill_random_fp8_chunked(kv, gen)
    torch.cuda.synchronize()
    t_quant = time.perf_counter() - t0

    t0 = time.perf_counter()
    max_visible = H + Q
    topk = max_visible if full_topk else TOPK
    if full_topk:
        topk = -(-topk // 128) * 128
    indices, topk_length = _prefill_indices(B, H, Q, topk, seed, full_topk)
    topk_sum = int(topk_length.sum())
    torch.cuda.synchronize()
    t_idx = time.perf_counter() - t0

    # Histories are >= 2048, so ordinary sparse rows all have exactly TOPK
    # entries. Only padded full-topk diagnostics need topk_length dispatch.
    dispatch_lengths = topk_length if full_topk else None

    def fn():
        return sparse_mla_q8kv8_prefill_fwd(
            q, kv, indices, SOFTMAX_SCALE, q_scale, kv_scale,
            d_v=D_V, topk_length=dispatch_lengths, out=out,
            max_logits=max_logits, lse=lse,
        )

    t0 = time.perf_counter()
    fn()  # includes one-time JIT compilation, outside kernel timing
    torch.cuda.synchronize()
    t_metadata = time.perf_counter() - t0

    account = dict(
        flops=flops_sparse_prefill(topk_sum), pairs=topk_sum,
        est_bytes=bytes_sparse_prefill_q8kv8(topk_sum, total_q), topk=topk,
    )
    setup = dict(alloc=t_alloc * 1e3, indices=t_idx * 1e3,
                 quant=t_quant * 1e3, metadata=t_metadata * 1e3)
    holders = dict(q=q, kv=kv, indices=indices, topk_length=topk_length,
                   q_scale=q_scale, kv_scale=kv_scale, out=out,
                   max_logits=max_logits, lse=lse)
    return fn, account, setup, holders


# ---------------------------------------------------------------- timing
def time_kernel(fn, warmup, repeat, cache_state, flush_buf):
    """Per-iteration CUDA-event timing. Setup/flush never enter the window."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeat):
        if cache_state == "l2-cold":
            _ = flush_buf.sum()  # read 256 MiB >> 50 MiB L2; not timed
            torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return times


def summarize(times):
    s = sorted(times)
    n = len(s)
    q = lambda p: s[min(n - 1, max(0, int(round(p * (n - 1)))))]
    return dict(median=q(0.5), p5=q(0.05), p95=q(0.95),
                mean=statistics.fmean(s),
                std=statistics.pstdev(s) if n > 1 else 0.0)


# ---------------------------------------------------------------- cases
def case_id(kind, backend, cache_state, B, H, Q, topk_variant):
    q = f"q{Q}" if kind == "prefill" else ""
    tv = f"_{topk_variant}" if topk_variant else ""
    return f"{kind}{q}_b{B}_h{H}_{backend}{tv}_{cache_state}"


def build_case_list(stage, cache_states, batches, histories, prefill_lengths,
                    prefill_sparse_kernel="bf16"):
    """Returns list of dicts describing every grid point incl. full-topk."""
    cases = []
    for cs in cache_states:
        if stage == "smoke":
            d_batches, d_hist = DECODE_BATCHES_SMOKE, HISTORIES_SMOKE
            p_bq, p_hist = PREFILL_BQ_SMOKE, HISTORIES_SMOKE
            if batches:
                d_batches = [b for b in d_batches if b in batches]
                p_bq = [(b, q) for b, q in p_bq if b in batches]
            if histories:
                d_hist = [h for h in d_hist if h in histories]
                p_hist = [h for h in p_hist if h in histories]
            if prefill_lengths:
                p_bq = [(b, q) for b, q in p_bq if q in prefill_lengths]
        else:
            d_batches = batches or BATCHES_FULL
            d_hist = histories or DECODE_HISTORIES_FULL
            p_batches = batches or BATCHES_FULL
            p_lengths = prefill_lengths or PREFILL_LENGTHS_FULL
            p_bq = [(b, q) for b in p_batches for q in p_lengths]
            p_hist = histories or PREFILL_HISTORIES_FULL
            if cs == "l2-cold":
                d_batches = [b for b in d_batches if b in COLD_DECODE_BATCHES]
                d_hist = [h for h in d_hist if h in COLD_DECODE_HISTORIES]
                p_bq = [x for x in p_bq if x in COLD_PREFILL_BQ]
                p_hist = [h for h in p_hist if h in COLD_PREFILL_HISTORIES]
        for B in d_batches:
            for H in d_hist:
                cases.append(dict(kind="decode", backend="dense", cache_state=cs,
                                  B=B, H=H, Q=1, topk_variant=None))
                cases.append(dict(kind="decode", backend="sparse", cache_state=cs,
                                  B=B, H=H, Q=1, topk_variant=None))
        for (B, Q) in p_bq:
            for H in p_hist:
                cases.append(dict(kind="prefill", backend="dense", cache_state=cs,
                                  B=B, H=H, Q=Q, topk_variant=None))
                sparse_backend = ("sparse_q8kv8"
                                  if prefill_sparse_kernel == "q8kv8"
                                  else "sparse")
                cases.append(dict(kind="prefill", backend=sparse_backend, cache_state=cs,
                                  B=B, H=H, Q=Q, topk_variant=None))
                max_visible = H + Q
                # Always materialize the diagnostic case in the result grid.
                # check_skip records unsupported index sizes explicitly instead
                # of silently dropping them.
                cases.append(dict(kind="prefill", backend=sparse_backend,
                                  cache_state=cs, B=B, H=H, Q=Q,
                                  topk_variant="fulltopk"))
    return cases


def setup_case(case, seed):
    kind, backend = case["kind"], case["backend"]
    ft = case["topk_variant"] == "fulltopk"
    if kind == "decode" and backend == "dense":
        return setup_dense_decode(case["B"], case["H"], seed)
    if kind == "decode" and backend == "sparse":
        return setup_sparse_decode(case["B"], case["H"], seed, full_topk=ft)
    if kind == "prefill" and backend == "dense":
        return setup_flashmla_dense_multiquery(
            case["B"], case["H"], case["Q"], seed)
    if kind == "prefill" and backend == "sparse":
        return setup_sparse_prefill(case["B"], case["H"], case["Q"], seed,
                                    full_topk=ft)
    if kind == "prefill" and backend == "sparse_q8kv8":
        return setup_sparse_prefill_q8kv8(
            case["B"], case["H"], case["Q"], seed, full_topk=ft)
    raise ValueError(f"unknown case {case}")


def backend_label(case):
    ft = case["topk_variant"] == "fulltopk"
    if case["kind"] == "decode":
        return {"dense": "flashmla_dense_bf16",
                "sparse": "flashmla_sparse_fp8_fulltopk" if ft
                else "flashmla_sparse_fp8"}[case["backend"]]
    return {
        "dense": "flashmla_dense_bf16_multiquery",
        "sparse": ("flashmla_sparse_bf16_fulltopk" if ft
                   else "flashmla_sparse_bf16"),
        "sparse_q8kv8": ("sglang_sparse_q8kv8_fp8_fulltopk" if ft
                         else "sglang_sparse_q8kv8_fp8"),
    }[case["backend"]]


def base_row(case, stage, args):
    cid_backend = case["backend"]
    if case["kind"] == "prefill" and case["backend"] == "dense":
        cid_backend = "dense_flashmla_multiquery"
    return dict(
        case_id=case_id(case["kind"], cid_backend, case["cache_state"],
                        case["B"], case["H"], case["Q"], case["topk_variant"]),
        stage=stage, kind=case["kind"], backend=backend_label(case),
        cache_state=case["cache_state"], status="ok", skip_reason=None,
        observed_utilization_vs_platform_max=None,
        B=case["B"], H=case["H"], Q=case["Q"],
        topk=case["H"] if (case["topk_variant"] and case["kind"] == "decode")
        else (case["H"] + case["Q"] if case["topk_variant"] else TOPK),
        h_q=H_Q, d_qk=D_QK, d_v=D_V,
        warmup=args.warmup, repeat=args.repeat, seed=args.seed,
        timestamp=datetime.now(timezone.utc).isoformat())


def check_skip(case):
    """Returns skip reason or None."""
    if case["kind"] == "prefill" and case["topk_variant"] == "fulltopk":
        max_visible = case["H"] + case["Q"]
        index_bytes = case["B"] * case["Q"] * max_visible * 4
        if max_visible > FULLTOPK_MAX_VISIBLE:
            return ("skipped_memory_limit: full-topk max visible "
                    f"{max_visible} > {FULLTOPK_MAX_VISIBLE}")
        if index_bytes > FULLTOPK_MAX_INDICES_BYTES:
            return ("skipped_memory_limit: full-topk indices "
                    f"{index_bytes / 2**30:.1f} GiB > 4 GiB")
    backend_key = ("dense_flashmla"
                   if case["kind"] == "prefill" and case["backend"] == "dense"
                   else "dense") if case["backend"] == "dense" else (
        case["backend"] if not case["topk_variant"] else "fulltopk")
    est = est_live_bytes(case["kind"], backend_key, case["B"], case["H"],
                         case["Q"])
    free, _ = torch.cuda.mem_get_info()
    if est > free:
        return (f"skipped_memory_limit: estimated live tensors "
                f"{est / 2**30:.1f} GiB exceed current free GPU memory "
                f"{free / 2**30:.1f} GiB")
    return None


def run_case(case, stage, args, flush_buf):
    row = base_row(case, stage, args)
    tele0 = gpu_telemetry()
    fn, account, setup, holders = None, None, None, None
    try:
        torch.cuda.reset_peak_memory_stats()
        fn, account, setup, holders = setup_case(case, args.seed)
        times = time_kernel(fn, args.warmup, args.repeat, case["cache_state"],
                            flush_buf)
        stats = summarize(times)
        med_s = stats["median"] / 1e3
        row.update(
            latency_ms_median=stats["median"], latency_ms_p5=stats["p5"],
            latency_ms_p95=stats["p95"], latency_ms_mean=stats["mean"],
            latency_ms_std=stats["std"],
            setup_alloc_ms=setup["alloc"], setup_indices_ms=setup["indices"],
            setup_quant_ms=setup["quant"], setup_metadata_ms=setup["metadata"],
            setup_total_ms=sum(v for v in setup.values() if v is not None),
            flops=account["flops"], pairs=account["pairs"],
            tflops=account["flops"] / med_s / 1e12,
            pairs_per_s=account["pairs"] / med_s,
            est_logical_bytes=account["est_bytes"],
            est_logical_gbps=account["est_bytes"] / med_s / 1e9,
            utilization_vs_h100_sxm_bf16_dense_peak=(
                account["flops"] / med_s / 1e12) / H100_SXM_BF16_DENSE_TFLOPS,
            peak_mem_alloc_bytes=torch.cuda.max_memory_allocated(),
            peak_mem_reserved_bytes=torch.cuda.max_memory_reserved(),
        )
    except torch.OutOfMemoryError as e:
        row["status"] = "skipped_memory_limit"
        row["skip_reason"] = f"runtime OOM: {e}"
    except Exception as e:
        row["status"] = "failed"
        row["skip_reason"] = f"{type(e).__name__}: {e}"
    finally:
        del fn, holders
        torch.cuda.empty_cache()
    tele1 = gpu_telemetry()
    for k, v in tele0.items():
        row[f"{k}_before"] = v
    for k, v in tele1.items():
        row[f"{k}_after"] = v
    return row


def skip_row(case, stage, args, reason):
    row = base_row(case, stage, args)
    row["status"] = (reason.split(":")[0] if reason.startswith("skipped")
                     else "skipped")
    row["skip_reason"] = reason
    return row


# ---------------------------------------------------------------- anchors
def anchor_cases():
    d = ANCHOR_DECODE
    p = ANCHOR_PREFILL
    return [
        dict(kind="decode", backend="dense", cache_state="steady",
             B=d["B"], H=d["H"], Q=1, topk_variant=None),
        dict(kind="decode", backend="sparse", cache_state="steady",
             B=d["B"], H=d["H"], Q=1, topk_variant=None),
        dict(kind="prefill", backend="dense", cache_state="steady",
             B=p["B"], H=p["H"], Q=p["Q"], topk_variant=None),
        dict(kind="prefill", backend="sparse", cache_state="steady",
             B=p["B"], H=p["H"], Q=p["Q"], topk_variant=None),
    ]


def run_anchors(phase, stage, args, flush_buf, writer):
    """phase: 'pre' | 'post' | 'rerun'. Returns {backend_label: median_ms}."""
    medians = {}
    for case in anchor_cases():
        row = run_case(case, stage, args, flush_buf)
        row["case_id"] = f"anchor_{phase}_" + row["case_id"]
        writer.append(row)
        if row["status"] == "ok":
            medians[backend_label(case)] = row["latency_ms_median"]
        print(f"  anchor[{phase}] {row['case_id']}: "
              f"{row.get('latency_ms_median')} ms ({row['status']})", flush=True)
    return medians


def check_anchor_drift(pre, post, stage, args, flush_buf, writer):
    """Returns (stable: bool, report: list[str])."""
    report = []
    drifting = []
    for k, pre_ms in pre.items():
        post_ms = post.get(k)
        if post_ms is None:
            continue
        drift = abs(post_ms - pre_ms) / pre_ms
        report.append(f"anchor drift {k}: {drift * 100:.2f}% "
                      f"(pre {pre_ms:.3f} ms, post {post_ms:.3f} ms)")
        if drift > ANCHOR_DRIFT_TOL:
            drifting.append(k)
    if not drifting:
        return True, report
    tele = gpu_telemetry()
    report.append(f"drift > {ANCHOR_DRIFT_TOL * 100:.0f}% on {drifting}; "
                  f"telemetry now: {tele}; cooling down 60 s and re-running")
    time.sleep(60)
    rerun = run_anchors("rerun", stage, args, flush_buf, writer)
    still = []
    for k in drifting:
        re_ms = rerun.get(k)
        if re_ms is None:
            continue
        drift = abs(re_ms - pre[k]) / pre[k]
        report.append(f"anchor re-run drift {k}: {drift * 100:.2f}%")
        if drift > ANCHOR_DRIFT_TOL:
            still.append(k)
    if still:
        report.append(f"RUN UNSTABLE: anchors still drifting after cooldown: "
                      f"{still}")
        return False, report
    return True, report


# ---------------------------------------------------------------- main
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["smoke", "full"], required=True)
    p.add_argument("--batches", type=str, default=None,
                   help="comma list, e.g. 1,8,32,64,128")
    p.add_argument("--histories", type=str, default=None,
                   help="comma list, e.g. 4096,32768")
    p.add_argument("--prefill-lengths", type=str, default=None,
                   help="comma list of new-token lengths, e.g. 1,16,256,1024")
    p.add_argument(
        "--backend-roles", type=str, default=None,
        help=("comma list from decode-dense,decode-sparse,prefill-dense,"
              "prefill-sparse,fulltopk; default runs all"),
    )
    p.add_argument(
        "--prefill-sparse-kernel", choices=["bf16", "q8kv8"], default="bf16",
        help=("sparse prefill implementation: pinned FlashMLA BF16 or "
              "vendored SGLang SM90 Q8xKV8 FP8"),
    )
    p.add_argument(
        "--case-ids", type=str, default=None,
        help="comma list of exact generated case IDs to run",
    )
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--repeat", type=int, default=30)
    p.add_argument("--cache-state", choices=["steady", "l2-cold", "both"],
                   default=None,
                   help="default: steady for smoke, both for full")
    p.add_argument("--time-budget-minutes", type=float, default=60.0)
    p.add_argument("--resume", action="store_true",
                   help="skip case_ids already present in results.jsonl")
    p.add_argument("--skip-anchors", action="store_true",
                   help="do not run full-stage pre/post stability anchors")
    p.add_argument("--output-dir", type=str, default=None,
                   help="default: <project>/runs/random_baseline")
    p.add_argument("--seed", type=int, default=20260803)
    return p.parse_args(argv)


def parse_int_list(s):
    return [int(x) for x in s.split(",") if x.strip()]


def main(argv=None):
    args = parse_args(argv)
    project_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = args.output_dir or os.path.join(
        project_dir, "runs", "random_baseline"
    )
    raw_dir = os.path.join(out_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    assert torch.cuda.is_available() and \
        torch.cuda.get_device_capability(0) == (9, 0), "need a single SM90 GPU"

    cache_states = (["steady", "l2-cold"] if args.cache_state == "both"
                    else [args.cache_state] if args.cache_state
                    else ["steady"] if args.stage == "smoke"
                    else ["steady", "l2-cold"])
    batches = parse_int_list(args.batches) if args.batches else None
    histories = parse_int_list(args.histories) if args.histories else None
    if histories:
        ignored = [h for h in histories if h < 2048]
        if ignored:
            print(f"ignoring histories < 2048: {ignored}", file=sys.stderr)
        histories = [h for h in histories if h >= 2048]
        if not histories:
            raise ValueError("all requested histories are < 2048")
    prefill_lengths = parse_int_list(args.prefill_lengths) \
        if args.prefill_lengths else None

    env = write_environment(raw_dir, project_dir, args.seed)
    print(f"environment: {env['gpu'].get('name')} "
          f"driver {env['driver'].get('driver_version')} "
          f"torch {env['software'].get('torch')}", flush=True)

    writer = ResultWriter(raw_dir)
    existing = writer.existing_keys() if args.resume else set()
    flush_buf = torch.empty(FLUSH_BUF_BYTES // 4, dtype=torch.float32,
                            device="cuda")

    t_start = time.monotonic()
    budget_s = args.time_budget_minutes * 60.0
    stop_new_s = max(0.0, budget_s - 5 * 60.0)

    counts = {"ok": 0, "failed": 0, "skipped_memory_limit": 0,
              "skipped_time_budget": 0, "resumed": 0}
    anchor_pre = {}
    stable = None

    if args.stage == "full" and not args.skip_anchors:
        print("== pre-run anchors ==", flush=True)
        anchor_pre = run_anchors("pre", args.stage, args, flush_buf, writer)

    cases = build_case_list(
        args.stage, cache_states, batches, histories, prefill_lengths,
        prefill_sparse_kernel=args.prefill_sparse_kernel,
    )
    if args.backend_roles:
        requested_roles = set(x.strip() for x in args.backend_roles.split(",")
                              if x.strip())
        valid_roles = {"decode-dense", "decode-sparse", "prefill-dense",
                       "prefill-sparse", "fulltopk"}
        unknown_roles = requested_roles - valid_roles
        if unknown_roles:
            raise ValueError(f"unknown --backend-roles: {sorted(unknown_roles)}")

        def role(case):
            if case["topk_variant"] == "fulltopk":
                return "fulltopk"
            if case["kind"] == "prefill" and \
                    case["backend"] == "sparse_q8kv8":
                return "prefill-sparse"
            return f"{case['kind']}-{case['backend']}"

        cases = [case for case in cases if role(case) in requested_roles]
    if args.case_ids:
        requested_case_ids = set(x.strip() for x in args.case_ids.split(",")
                                 if x.strip())

        def generated_case_id(case):
            cid_backend = case["backend"]
            if case["kind"] == "prefill" and case["backend"] == "dense":
                cid_backend = "dense_flashmla_multiquery"
            return case_id(case["kind"], cid_backend, case["cache_state"],
                           case["B"], case["H"], case["Q"],
                           case["topk_variant"])

        cases = [case for case in cases
                 if generated_case_id(case) in requested_case_ids]
        found_case_ids = {generated_case_id(case) for case in cases}
        missing_case_ids = requested_case_ids - found_case_ids
        if missing_case_ids:
            raise ValueError(
                f"--case-ids not present in selected grid: "
                f"{sorted(missing_case_ids)}")
    print(f"grid: {len(cases)} cases "
          f"(cache_states={cache_states}, budget={args.time_budget_minutes} min)",
          flush=True)

    time_up = False
    oom_frontiers = {}
    for i, case in enumerate(cases):
        cid_backend = case["backend"]
        if case["kind"] == "prefill" and case["backend"] == "dense":
            cid_backend = "dense_flashmla_multiquery"
        cid = case_id(case["kind"], cid_backend, case["cache_state"],
                      case["B"], case["H"], case["Q"], case["topk_variant"])
        if cid in existing:
            counts["resumed"] += 1
            continue
        frontier_key = (case["kind"], case["backend"],
                        case["topk_variant"], case["B"], case["Q"])
        if case["H"] >= oom_frontiers.get(frontier_key, 1 << 62):
            writer.append(skip_row(
                case, args.stage, args,
                "skipped_memory_limit: larger history after runtime OOM "
                "on the same (kind, backend, B, Q) curve"))
            counts["skipped_memory_limit"] += 1
            continue
        elapsed = time.monotonic() - t_start
        if time_up or elapsed > stop_new_s:
            time_up = True
            writer.append(skip_row(case, args.stage, args,
                                   "skipped_time_budget: past "
                                   f"{stop_new_s / 60:.0f} min of "
                                   f"{args.time_budget_minutes} min budget"))
            counts["skipped_time_budget"] += 1
            continue
        reason = check_skip(case)
        if reason:
            writer.append(skip_row(case, args.stage, args, reason))
            counts["skipped_memory_limit"] += 1
            print(f"[{i + 1}/{len(cases)}] {cid}: {reason}", flush=True)
            continue
        print(f"[{i + 1}/{len(cases)}] {cid} ...", flush=True)
        row = run_case(case, args.stage, args, flush_buf)
        writer.append(row)
        counts[row["status"] if row["status"] in counts else "failed"] += 1
        if row["status"] == "skipped_memory_limit" and \
                str(row.get("skip_reason", "")).startswith("runtime OOM"):
            oom_frontiers[frontier_key] = case["H"]
        if row["status"] == "ok":
            print(f"    {row['latency_ms_median']:.3f} ms median, "
                  f"{row['tflops']:.1f} TFLOPS, "
                  f"{row['est_logical_gbps']:.0f} GB/s(est-logical)",
                  flush=True)
        else:
            print(f"    FAILED: {row['skip_reason']}", flush=True)

    if args.stage == "full" and not args.skip_anchors:
        print("== post-run anchors ==", flush=True)
        anchor_post = run_anchors("post", args.stage, args, flush_buf, writer)
        stable, report = check_anchor_drift(anchor_pre, anchor_post, args.stage,
                                            args, flush_buf, writer)
        for line in report:
            print("  " + line, flush=True)

    writer.finalize_observed_utilization()
    counts["anchors_stable"] = stable
    print(f"done: {counts}", flush=True)
    summary_path = os.path.join(raw_dir, "last_run_summary.json")
    with open(summary_path, "w") as f:
        json.dump({"counts": {k: v for k, v in counts.items()},
                   "stage": args.stage, "seed": args.seed,
                   "prefill_sparse_kernel": args.prefill_sparse_kernel,
                   "cache_states": cache_states,
                   "elapsed_min": (time.monotonic() - t_start) / 60.0,
                   "timestamp": datetime.now(timezone.utc).isoformat()},
                  f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    return 0


if __name__ == "__main__":
    sys.exit(main())
