# H800 FlashMLA Dense/Sparse Benchmark Log

## Environment

- NVIDIA H800 SXM 80GB, SM90, 700W power limit
- Driver 580.126.20, CUDA toolkit 13.0.48
- PyTorch 2.9.1+cu130
- FlashMLA 1.0.0 at `15f13e5030374295491c5ce31b02d7e63a7772c6`
- FlashInfer 0.6.11.post1 is installed for independent correctness checks only
- fixed shape: `h_q=64`, `d_qk=576`, `d_v=512`, `topk=2048`

## Backend Decision

The pinned FlashMLA support matrix exposes SM90 dense MLA decode and SM90
sparse MLA prefill, but no SM90 dense MLA prefill entry. The requested dense
chunk curves therefore use `flash_mla_with_kvcache` in its native SM90 BF16
causal multi-query mode. Setting cache length to `H+Q` makes query `i` see
exactly `H+i+1` KV tokens. Results use backend label
`flashmla_dense_bf16_multiquery` and separate case IDs; no FA3 result enters
the README history-scaling plots or canonical result set.

The follow-up run additionally vendors SGLang's native SM90 Q8xKV8 sparse
prefill kernel from commit `5d85f25f75b6b6c937ac85bdc57ba0d19ebbbd7c`.
It is a separate backend and does not replace or relabel the pinned FlashMLA
BF16 measurements.

## Verification

`test_correctness.py`: 12/12 passed.

- dense FlashMLA multi-query vs prefix+chunk FP32 reference: 100% strict pass,
  max abs error `1.706e-3`, cosine diff `1.598e-6`
- dense decode vs reference: 100% strict pass
- sparse FP8 decode vs dequantized reference: 100% strict pass
- sparse FP8 decode vs original BF16 dense cosine similarity: `0.999604`
- sparse BF16 full-topk vs independent FA3 correctness reference: 100% strict
  pass at `atol=8e-4`, `rtol=3.01/128`
- SGLang Q8xKV8 sparse prefill vs selected-token FP32 reference: 100% strict
  pass, max abs error `7.553e-5`, cosine diff `1.009e-4`
- FP8 656-byte layout, paged addressing, index causality, FLOPs formulas, and
  timing-window separation passed

## Extended Grid Run

Grid:

- B: `1,2,4,8,16,32,64,128,256`
- H: `2K,4K,8K,16K,32K,64K,128K,256K,512K`
- Q: `1,2,4,8,16,32,64,128,256,512,1K,2K,4K,8K,16K,32K`
- steady-state, 10 warmup, 30 measured iterations

The extended run removed the earlier `B*Q <= 32768` policy gate and fixed
60 GiB live-tensor cap. Repeated `--resume` passes completed every target case;
the only remaining boundary is estimated physical live memory versus the
runtime free-memory reading (about 78.6 GiB), plus actual CUDA OOM.

Canonical coverage (`history_scaling_results.*`):

- target rows: 2,592
- successful measurements: 2,309
- physical `skipped_memory_limit`: 283
- missing: 0
- time-budget skips: 0
- final kernel failures: 0

All 283 unavailable cases exceed actual available memory in the live-tensor
preflight. After switching sparse decode setup to chunked FP8 cache filling,
the final canonical set has no allocation-stage CUDA OOM. A runtime OOM would
stop larger histories on the same `(kind, backend, B, Q)` curve. Sparse FP8
quantization preflight includes only its bounded chunk temporary.

## ECHO Figure 2 Metric Correction

The initial plots used causal attended-pair FLOPs divided by full-call CUDA
event time. That is an effective end-to-end metric, but it is not the metric
used by ECHO Figure 2 or FlashMLA's official dense benchmark.

The corrected `figure2_hardware_tflops` uses:

- dense: rectangular `B*Q*cache_seqlen` FLOPs divided only by the
  `flash_fwd_splitkv_mla_kernel` duration
- sparse decode: selected-pair FLOPs divided by split-to-combine duration
- sparse prefill: selected-pair FLOPs divided by `sparse_attn_fwd` duration

All 2,309 runnable cases were profiled with 10 iterations, matching the official
dense benchmark's run count. One dense point fell from its expected curve
under sustained thermal load (`358.3 TFLOPS`); after cooling to 42 C, an exact
case-ID reprofile produced `665.1 TFLOPS` and overrides the anomalous record.
The final set has 2,309 unique cases and zero profiler failures. A neighbor-curve
scan found no remaining isolated drop below 75% of adjacent geometric mean.

For `Q>1`, rectangular counting includes causally masked upper-triangle work.
Consequently nominal values can exceed the `989.5 TFLOPS` H100 reference;
these values must not be interpreted as physical H800 peak utilization. The
original causal/full-call values remain as `effective_e2e_tflops`.

## Results

- Figure 2 decode maximum: dense `383.2 TFLOPS`, sparse `339.2 TFLOPS`
- Figure 2 dense causal multi-query nominal maximum: `1151.5 TFLOPS`
- Figure 2 sparse BF16 prefill maximum: `649.5 TFLOPS`
- at long history, dense chunk performance generally remains around
  `604-653 TFLOPS`
- fixed-topk sparse hardware-utilization TFLOPS falls with history (often
  toward `342-348 TFLOPS` at 512K), even though selected work stays at
  topk=2048; latency and logical bandwidth are retained in raw records

All sixteen decode/prefill history-axis overview figures remain reproducible
as PNG and PDF; the batch-axis views remain the primary decode presentation.

## Final Plot Organization

All plots use `figure2_hardware_tflops`, and unavailable points are markers
rather than zero-valued curve samples.

- decode TFLOPS: batch on x, one figure per fixed history, plus one overview
- decode/prefill history scaling: history on x, 135 fixed prefill `(B,Q)`
  figures and sixteen fixed-Q overview sheets (including Q=1 decode)
- sparse decode latency diagnostic: history on x, one figure per batch

The corresponding tables are `decode_tflops_vs_batch.csv` and
`history_scaling_coverage.csv`.

## Sparse Decode Latency Scaling

Nine per-batch figures use the 30-repeat CUDA-event median and p5-p95 interval
for native FlashMLA FP8 sparse decode. Each plot fixes batch and varies history
from 2K through 512K; unavailable suffixes remain explicit memory-limit skips.

From 4K to the largest runnable history for each batch, median latency changes
between `-1.54%` and `+2.66%`. With selected length fixed at 2048, increasing
the addressable KV history therefore has no material latency trend in these
measurements. Batch is the dominant factor: roughly 33 us at B=1 and 242-245
us at the runnable B=256 points.

## SGLang Q8xKV8 Sparse Prefill Follow-up

Command:

```bash
.venv/bin/python benchmark.py --stage full --cache-state steady --resume \
  --skip-anchors --backend-roles prefill-sparse \
  --prefill-sparse-kernel q8kv8 --time-budget-minutes 60 \
  --output-dir assets/q8kv8
.venv/bin/python plot_q8kv8_results.py
```

The kernel consumes contiguous E4M3 Q and KV, int32 top-k indices, GPU scalar
Q/KV scales, and writes BF16 output. FP8 conversion, index construction, and
JIT compilation are setup-only. The vendored source manifest verifies eight
Git blob SHA1 values before compilation; the JIT target is `sm_90a` only.

Coverage:

- target rows: 1,215
- successful measurements: 1,116
- physical `skipped_memory_limit`: 99 (six runtime allocation OOMs)
- time-budget skips: 0
- kernel failures: 0
- maximum selected-pair throughput: `728.7 TFLOPS`

There are 1,094 same-shape BF16/Q8 comparisons. Median Q8/BF16 selected-pair
throughput is `1.263x`; the per-history median ratio rises from `1.185x` at 2K
to `1.478x` at 512K. Among 107 curves with both endpoints, median 512K/2K
throughput retention is `0.731` for Q8 and `0.587` for BF16. This supports a
memory-access contribution to the long-history prefill decline: FP8 reduces
the decline but does not remove it.

Raw results are under `assets/q8kv8/raw/`; comparison plots and PDFs are under
`assets/q8kv8/figures/`.

## Controlled Cache-Locality Follow-up (No NCU)

The first attempt overlapped an unrelated GPU benchmark that occupied about
60 GiB and 100% GPU utilization. It is excluded and explicitly marked by
`assets/cache_locality/CONTAMINATED.md`. The authoritative run under
`assets/cache_locality_clean/` started only after two consecutive idle-GPU
checks.

Design:

- histories: `2K,32K,512K`; `topk=2048`
- ascending and descending history-order passes
- steady and 256 MiB flush-buffer L2-cold states
- SGLang Q8xKV8 at 64 and 256 query rows
- FlashMLA BF16 sparse prefill at 64 query rows
- shared versus independent selected sets, each contiguous or dispersed
- 64-sequence isolated Q8 prefill and native FlashMLA FP8 decode controls

All 204 raw cases succeeded. The 102 aggregate points are medians of the two
order passes. Maximum order spread was `4.61%`; median cold/steady ratio was
`1.015x`.

At Q8 N=256, independent-dispersed latency increased `1.475x` from 2K to
512K, while shared-dispersed increased only `1.003x`. At H=512K, removing
cross-query overlap with contiguous selections cost `1.336x`; dispersion by
itself with full overlap cost `1.003x`, and adding dispersion after overlap
was removed cost another `1.097x`. The selected working set grew from 2,048
unique tokens (`0.023x` of the 50 MiB L2) to 524,288 (`5.760x`). BF16 N=64
showed the same direction (`1.220x` independent versus `1.002x` shared).

The real FP8 decode control rose only `1.046x`; the same Q8 prefill kernel with
64 independent sequences rose `1.058x`. Thus the history-dependent prefill
decline is not caused by addressable history alone or by an intrinsic prefill
kernel property. At the software-visible level, its primary cause is loss of
within-invocation cross-query selected-KV overlap and growth of the unique KV
working set. Spatial dispersion is a secondary effect. Cross-invocation L2
persistence is not dominant.

Without NCU counters this result does not identify the final hardware stall
mix. HBM traffic, L2 misses, TLB pressure, sector efficiency, and long
scoreboard stalls cannot be apportioned. The conclusion is therefore stated
as increased memory-hierarchy pressure, not as a directly measured bandwidth-
utilization loss. Synthetic indices also limit extrapolation to real indexer
traces.

Artifacts:

- `assets/cache_locality_clean/raw/results.{jsonl,csv}`
- `assets/cache_locality_clean/raw/aggregate.csv`
- `assets/cache_locality_clean/raw/{design,environment}.json`
- `assets/cache_locality_clean/analysis.md`
- `assets/cache_locality_clean/figures/` (five PNG/PDF figure pairs)

## Native Decode Selected-KV Reuse Follow-up

An initial hot-versus-flushed design was rejected because the 256 MiB flush
also changed the immediate DVFS state. A symmetric prime/flush revision still
showed immediate-predecessor sensitivity. Both diagnostic trials are preserved
under `assets/decode_l2_residency_*` and explicitly marked invalid or
diagnostic-only; neither enters the conclusion.

The final design removes that confound by giving both timed cases the same
256 MiB flush immediately before the native FlashMLA FP8 sparse-decode call.
Only physical indices change: all batch rows either share the same 2,048 KV
tokens or read `B*2048` independent tokens. Kernel, Q/KV allocation, FLOPs,
topk, batch, and clock-conditioning work remain identical.

The clean full run covers `B=1,8,16,32,64` and `H=4K..512K`; supplemental
`B=128,256` points cover `H=4K,32K,256K`. Every shape has ascending and
descending passes. Three shapes with over 5% absolute-latency drift were
rerun at 50 repeats and explicitly override the original aggregate. Final
maximum pass spread is `3.38%`.

Median independent/shared latency ratios by batch are `0.999x, 1.032x,
1.042x, 1.059x, 1.081x, 1.096x, 1.159x`; the maximum observed ratio is
`1.165x` at B=256. Native decode therefore has observable benefit from shared
selected KV, falsifying the absolute claim that it has no cache-reuse upside.
However, even with 256 query rows the benefit is materially smaller than the
controlled prefill `1.475x`. The prefill decline cannot be reduced to a binary
L2-hit versus L2-miss transition. Ordinary decode starts with independent
per-sequence selected sets whose unique count stays fixed as history grows;
prefill additionally loses within-chunk overlap and changes the unique working
set and concurrent memory-access organization.

No hardware hit-rate claim is made without NCU. Results establish a
memory-hierarchy reuse effect at the controlled software-input level only.
Final artifacts are under `assets/decode_kv_reuse/`; override inputs are under
`assets/decode_kv_reuse_rerun/` and `assets/decode_kv_reuse_large_batch/`.
