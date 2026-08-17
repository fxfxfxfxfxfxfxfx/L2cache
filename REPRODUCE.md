# Reproducing the Sparse MLA Prefill Study

The checked-in artifacts are sufficient to regenerate every README figure on a
CPU. GPU commands are only needed to collect new measurements.

## Analysis-only reproduction

```bash
cd /root/L2cache
./bootstrap.sh
PYTHON=.venv/bin/python scripts/reproduce_figures.sh
.venv/bin/python -m tests.test_csa_trace
.venv/bin/python -c \
  'from scripts.sglang_q8kv8 import verify_sources; print(verify_sources())'
```

`scripts/reproduce_figures.sh` uses a temporary build directory and writes only
the curated PNG files under `artifacts/figures/`.

## Downloading the public CSA sample

The source dataset is public:

<https://modelscope.cn/datasets/fxiaoO/deepseek-v4-flash-swebench-csa-topk>

The downloader selects metadata rows 0–23 and trace rows
`0,3,4,5,8,12,18,19`. Anonymous access is used by default; an optional
`MODELSCOPE_API_TOKEN` environment variable is supported.

```bash
.venv/bin/python -m scripts.download_csa_trace_sample
.venv/bin/python -m scripts.analyze_csa_trace_sample \
  --source local_data/csa_trace_source \
  --output-dir runs/csa_trace_profile \
  --rows-per-stratum 128
```

The download is approximately 2.6GB. `local_data/` is intentionally ignored.

## GPU environment

The reported main results used:

```text
NVIDIA H800 PCIe 80GB, SM90, 350W
Driver 580.82.07
Python 3.12.13
PyTorch 2.9.1+cu128 (CUDA 12.8)
FlashInfer 0.6.3
SGLang kernel commit 5d85f25f75b6b6c937ac85bdc57ba0d19ebbbd7c
```

The benchmark environment must provide CUDA-enabled PyTorch, FlashInfer,
TVM-FFI and Ninja. `scripts.prefill_runtime.validate_runtime` rejects a runtime
that differs from the retained random baseline before writing measurements.

## Random history baseline

```bash
python -m scripts.benchmark --stage full --cache-state steady --resume \
  --skip-anchors --backend-roles prefill-sparse \
  --prefill-sparse-kernel q8kv8 \
  --batches 1,2,4,8,16,32,64,128 \
  --histories 2048,4096,8192,16384,32768,65536,131072,262144,524288 \
  --prefill-lengths 2,8,32,128,512,1024,4096 \
  --output-dir runs/random_baseline \
  --time-budget-minutes 180
```

## Trace replay

```bash
python -m scripts.csa_trace_benchmark --stage full \
  --trace-source local_data/csa_trace_source \
  --reference-results artifacts/data/random_baseline/raw/results.jsonl \
  --output-dir runs/csa_trace_replay \
  --batches 1,8,32,128 \
  --histories 8192,16384,32768,65536 \
  --prefill-lengths 64,256,1024,2048 \
  --warmup 10 --repeat 30 --time-budget-minutes 120
```

## CSA profile full grid

Batch outer:

```bash
python -m scripts.csa_profile_sim_benchmark --stage full \
  --trace-analysis artifacts/data/csa_trace_profile \
  --reference-results artifacts/data/random_baseline/raw/results.jsonl \
  --output-dir runs/csa_batch_outer \
  --row-layout batch-outer \
  --batches 1,2,4,8,16,32,64,128 \
  --histories 2048,4096,8192,16384,32768,65536,131072,262144,524288 \
  --prefill-lengths 2,8,32,128,512,1024,4096 \
  --warmup 10 --repeat 30 --index-workers 16 \
  --time-budget-minutes 180
```

Batch inner uses the same command with:

```text
--row-layout batch-inner --output-dir runs/csa_batch_inner
```

Every full run executes shapes in ascending and descending order. Use
`--resume` only with an output directory whose retained `design.json` matches
the current command.

## Artifact schemas

- `results.jsonl`: one authoritative terminal row per case ID.
- `results.csv`: the same rows in tabular form.
- `paired.csv`: pass-aggregated, same-shape comparisons used by the report.
- `environment.json`: hardware, software and vendored source identity.
- `design.json`: shape grid, seed, ordering and simulation definition.

The original append-only inputs and rerun precedence are preserved outside the
repository as documented in [ARCHIVE_MANIFEST.md](ARCHIVE_MANIFEST.md).
