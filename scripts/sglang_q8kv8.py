#!/usr/bin/env python3
"""Standalone loader for SGLang's SM90 sparse MLA Q8xKV8 prefill kernel."""

import hashlib
import os
from pathlib import Path

import torch


SGLANG_COMMIT = "5d85f25f75b6b6c937ac85bdc57ba0d19ebbbd7c"
SOURCE_BLOB_SHA1 = {
    "config.h": "f81b933655d46d6a63e1feb8e061873d56576667",
    "defines.h": "fc6758ed98ab5c82136a44ba06779acef321585d",
    "dense_fp8_transpose_v.h": "151100a3b43b6a05ef2d0cfd1c23d313d9475655",
    "dense_fp8_utils.h": "3ddcfc1a6a20014364c839529eab86f524c909fb",
    "entry.cuh": "0772b617357f327e58b5122b753a044a13d855cb",
    "helpers.h": "6bed8ed988b1f14703a38f08d249ca03cda07235",
    "kernel.cuh": "54bc2abdb751320a814e346aa8b5c42b5664924e",
    "params.h": "5a522f04e083caaf0c72d537666d7ced2400800f",
}

_MODULE = None


def _git_blob_sha1(path: Path) -> str:
    data = path.read_bytes()
    return hashlib.sha1(
        b"blob " + str(len(data)).encode("ascii") + b"\0" + data
    ).hexdigest()


def verify_sources() -> dict[str, str]:
    source_dir = Path(__file__).resolve().parents[1] / "third_party" / "sglang_q8kv8"
    actual = {}
    for name, expected in SOURCE_BLOB_SHA1.items():
        path = source_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"missing vendored SGLang source: {path}")
        actual[name] = _git_blob_sha1(path)
        if actual[name] != expected:
            raise RuntimeError(
                f"SGLang source hash mismatch for {name}: "
                f"expected {expected}, got {actual[name]}"
            )
    return actual


def _load_module():
    global _MODULE
    if _MODULE is not None:
        return _MODULE

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0):
        raise RuntimeError("SGLang sparse MLA Q8xKV8 prefill requires SM90")

    verify_sources()
    import flashinfer
    import tvm_ffi

    project_dir = Path(__file__).resolve().parents[1]
    source_dir = project_dir / "third_party" / "sglang_q8kv8"
    flashinfer_dir = Path(flashinfer.__file__).resolve().parent
    cutlass_dir = flashinfer_dir / "data" / "cutlass"
    include_paths = [
        str(source_dir),
        str(cutlass_dir / "include"),
        str(cutlass_dir / "tools" / "util" / "include"),
    ]
    for path in include_paths:
        if not Path(path).is_dir():
            raise FileNotFoundError(f"required include directory is missing: {path}")

    os.environ.setdefault("TVM_FFI_CUDA_ARCH_LIST", "9.0a")
    source = f'''#include "{source_dir / 'entry.cuh'}"
TVM_FFI_DLL_EXPORT_TYPED_FUNC(dispatch, sglang::sparse_prefill_q8kv8_dispatch);
TVM_FFI_DLL_EXPORT_TYPED_FUNC(dispatch_topk_length, sglang::sparse_prefill_q8kv8_dispatch_topk_length);
'''
    build_dir = project_dir / ".cache" / "sglang_q8kv8"
    build_dir.mkdir(parents=True, exist_ok=True)
    _MODULE = tvm_ffi.cpp.load_inline(
        "sglang_sparse_mla_q8kv8_sm90",
        cuda_sources=[source],
        extra_cuda_cflags=[
            "-DSGL_CUDA_ARCH=900",
            "-std=c++20",
            "-O3",
            "--expt-relaxed-constexpr",
            "-DNDEBUG",
            "-DCUTE_USE_PACKED_TUPLE=1",
            "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
            "--use_fast_math",
        ],
        extra_include_paths=include_paths,
        build_directory=str(build_dir),
    )
    return _MODULE


def sparse_mla_q8kv8_prefill_fwd(
    q: torch.Tensor,
    kv: torch.Tensor,
    indices: torch.Tensor,
    sm_scale: float,
    q_scale: torch.Tensor,
    kv_scale: torch.Tensor,
    *,
    d_v: int = 512,
    topk_length: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    max_logits: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the vendored SGLang Q8xKV8 sparse prefill kernel."""
    if q.dtype != torch.float8_e4m3fn or kv.dtype != torch.float8_e4m3fn:
        raise TypeError("q and kv must use torch.float8_e4m3fn")
    if indices.dtype != torch.int32:
        raise TypeError("indices must use torch.int32")
    if q.ndim != 3 or kv.ndim != 3 or indices.ndim != 3:
        raise ValueError("expected q, kv, and indices to be rank-3 tensors")
    s_q, h_q, d_qk = q.shape
    s_kv, h_kv, kv_d = kv.shape
    if kv_d != d_qk or h_kv != 1:
        raise ValueError("kv must have shape [s_kv, 1, d_qk]")
    if indices.shape[:2] != (s_q, h_kv):
        raise ValueError("indices must have shape [s_q, h_kv, topk]")
    if d_qk not in (512, 576) or d_v != 512:
        raise ValueError("kernel supports d_qk in {512,576} and d_v=512")
    topk = indices.shape[-1]
    if topk % 128:
        raise ValueError("topk must be a multiple of 128")
    tensors = (q, kv, indices, q_scale, kv_scale)
    if any(not x.is_cuda or not x.is_contiguous() for x in tensors):
        raise ValueError("all inputs must be contiguous CUDA tensors")
    if q_scale.dtype != torch.float32 or kv_scale.dtype != torch.float32:
        raise TypeError("q_scale and kv_scale must be float32 CUDA tensors")

    out = out if out is not None else torch.empty(
        (s_q, h_q, d_v), dtype=torch.bfloat16, device=q.device
    )
    max_logits = max_logits if max_logits is not None else torch.empty(
        (s_q, h_q), dtype=torch.float32, device=q.device
    )
    lse = lse if lse is not None else torch.empty_like(max_logits)
    stream = torch._C._cuda_getCurrentRawStream(q.device.index or 0)
    module = _load_module()
    args = (
        q, kv, indices, q_scale, kv_scale, out, max_logits, lse,
        s_q, s_kv, h_q, h_kv, d_qk, d_v, topk, sm_scale, stream,
    )
    if topk_length is None:
        module.dispatch(*args)
    else:
        if topk_length.dtype != torch.int32 or not topk_length.is_contiguous():
            raise TypeError("topk_length must be contiguous int32")
        module.dispatch_topk_length(
            q, kv, indices, q_scale, kv_scale, topk_length,
            out, max_logits, lse, s_q, s_kv, h_q, h_kv, d_qk, d_v,
            topk, sm_scale, stream,
        )
    return out, max_logits, lse


def source_manifest() -> dict:
    return {
        "sglang_commit": SGLANG_COMMIT,
        "source_git_blob_sha1": verify_sources(),
    }
