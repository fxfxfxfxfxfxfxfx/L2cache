"""Shared runtime helpers for Q8KV8 sparse-prefill experiments."""

import json
import time
from pathlib import Path

import torch

from scripts import benchmark as bm
from scripts.sglang_q8kv8 import sparse_mla_q8kv8_prefill_fwd


def load_reference_shapes(path: Path):
    """Load successful steady-state Q8KV8 prefill shapes."""
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
                and int(row["H"]) > bm.TOPK
                and int(row["Q"]) >= 2
            ):
                shapes.add((int(row["B"]), int(row["H"]), int(row["Q"])))
    return sorted(shapes, key=lambda item: (item[0], item[2], item[1]))


def validate_runtime(reference_results: Path):
    """Require a run to match the environment of its random baseline."""
    environment_path = Path(reference_results).parent / "environment.json"
    if not environment_path.exists():
        raise FileNotFoundError(
            f"reference environment is missing: {environment_path}"
        )
    reference = json.loads(environment_path.read_text())
    expected_gpu = reference.get("gpu", {}).get("name")
    expected_software = reference.get("software", {})
    mismatches = []
    actual_gpu = torch.cuda.get_device_name()
    if expected_gpu and actual_gpu != expected_gpu:
        mismatches.append(f"GPU expected {expected_gpu!r}, got {actual_gpu!r}")
    if torch.__version__ != expected_software.get("torch"):
        mismatches.append(
            f"torch expected {expected_software.get('torch')!r}, "
            f"got {torch.__version__!r}"
        )
    if torch.version.cuda != expected_software.get("torch_cuda"):
        mismatches.append(
            f"torch CUDA expected {expected_software.get('torch_cuda')!r}, "
            f"got {torch.version.cuda!r}"
        )
    try:
        import flashinfer
        actual_flashinfer = flashinfer.__version__
    except Exception as error:
        mismatches.append(f"flashinfer import failed: {error}")
    else:
        if actual_flashinfer != expected_software.get("flashinfer"):
            mismatches.append(
                f"flashinfer expected {expected_software.get('flashinfer')!r}, "
                f"got {actual_flashinfer!r}"
            )
    if mismatches:
        raise RuntimeError(
            "runtime does not match the reference Q8KV8 grid:\n- "
            + "\n- ".join(mismatches)
        )


def adjacent_overlap(indices: torch.Tensor, batch: int, query: int) -> float:
    """Return exact mean adjacent selected-set intersection over top-k."""
    rows = indices.view(batch, query, bm.TOPK)
    intersections = torch.zeros((), dtype=torch.int64, device=indices.device)
    for begin in range(1, query, 64):
        end = min(query, begin + 64)
        pair_values = torch.cat(
            (rows[:, begin - 1:end - 1], rows[:, begin:end]), dim=-1
        ).reshape(-1, 2 * bm.TOPK)
        ordered = pair_values.sort(dim=-1).values
        intersections += (ordered[:, 1:] == ordered[:, :-1]).sum()
    return intersections.item() / (batch * (query - 1) * bm.TOPK)


def unique_kv(indices: torch.Tensor, batch: int, history: int, query: int) -> int:
    """Count unique selected KV tokens independently for each sequence."""
    rows = indices.view(batch, query, bm.TOPK)
    segment = history + query
    total = 0
    seen = torch.empty(segment, dtype=torch.bool, device=indices.device)
    for batch_index in range(batch):
        seen.zero_()
        local = (
            rows[batch_index].reshape(-1).to(torch.int64)
            - batch_index * segment
        )
        seen[local] = True
        total += int(seen.sum().item())
    return total


def allocate_inputs(batch, history, query, seed):
    t0 = time.perf_counter()
    generator = torch.Generator(device="cuda").manual_seed(seed)
    total_q = batch * query
    total_kv = batch * (history + query)
    q = torch.empty(
        (total_q, bm.H_Q, bm.D_QK),
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    kv = torch.empty(
        (total_kv, 1, bm.D_QK),
        dtype=torch.float8_e4m3fn,
        device="cuda",
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
    bm._fill_random_fp8_chunked(q, generator)
    bm._fill_random_fp8_chunked(kv, generator)
    torch.cuda.synchronize()
    quant_ms = (time.perf_counter() - t0) * 1e3
    return {
        "q": q,
        "kv": kv,
        "out": out,
        "max_logits": max_logits,
        "lse": lse,
        "q_scale": q_scale,
        "kv_scale": kv_scale,
    }, alloc_ms, quant_ms


def bind_kernel(inputs, indices):
    def fn():
        return sparse_mla_q8kv8_prefill_fwd(
            inputs["q"],
            inputs["kv"],
            indices,
            bm.SOFTMAX_SCALE,
            inputs["q_scale"],
            inputs["kv_scale"],
            d_v=bm.D_V,
            out=inputs["out"],
            max_logits=inputs["max_logits"],
            lse=inputs["lse"],
        )

    return fn
