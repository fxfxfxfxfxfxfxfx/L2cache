#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$(mktemp -d)"
PYTHON_BIN="${PYTHON:-python}"
trap 'rm -rf "${BUILD_DIR}"' EXIT

cd "${PROJECT_DIR}"

"${PYTHON_BIN}" -m scripts.plot_trace_profile \
  --profile-dir artifacts/data/csa_trace_profile/raw \
  --output-dir artifacts/figures/main

"${PYTHON_BIN}" -m scripts.plot_random_decode \
  --input artifacts/data/decode_control/raw/aggregate.csv \
  --output artifacts/figures/main/random_decode_history.png

"${PYTHON_BIN}" -m scripts.plot_random_baseline \
  --input artifacts/data/random_baseline/raw/results.jsonl \
  --query 512 \
  --output artifacts/figures/main/random_prefill_q512.png

"${PYTHON_BIN}" -m scripts.analyze_csa_trace_benchmark \
  --input artifacts/data/csa_trace_replay/raw/results.jsonl \
  --trace-analysis artifacts/data/csa_trace_profile \
  --output-dir "${BUILD_DIR}/trace-replay"

"${PYTHON_BIN}" -m scripts.analyze_csa_profile_sim \
  --random-input artifacts/data/random_baseline/raw/results.jsonl \
  --sim-input artifacts/data/csa_batch_outer/raw/results.jsonl \
  --output-dir "${BUILD_DIR}/csa-outer"

"${PYTHON_BIN}" -m scripts.analyze_csa_row_layout \
  --random-paired artifacts/data/csa_batch_outer/raw/paired.csv \
  --outer-input artifacts/data/csa_batch_outer/raw/results.jsonl \
  --inner-input artifacts/data/csa_batch_inner/raw/results.jsonl \
  --output-dir "${BUILD_DIR}/row-layout"

cp "${BUILD_DIR}/trace-replay/figures/trace_replay_throughput_overview.png" \
  artifacts/figures/supplement/csa_trace_replay_throughput.png

for query in 2 8 32 128 512 1024 4096; do
  cp "${BUILD_DIR}/row-layout/figures/row_layout_q${query}_overview.png" \
    "artifacts/figures/supplement/row_layout_q${query}.png"
done

cp artifacts/figures/supplement/row_layout_q512.png \
  artifacts/figures/main/random_csa_row_layout_q512.png
cp "${BUILD_DIR}/trace-replay/figures/trace_replay_overlap_overview.png" \
  artifacts/figures/supplement/csa_trace_replay_overlap.png

echo "Figures written to artifacts/figures"
