#!/usr/bin/env bash
set -euo pipefail

# L2 residency measurements are invalid under concurrent GPU load.  Require
# two consecutive idle observations before smoke and the full run.
idle_count=0
while (( idle_count < 2 )); do
    active="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        | sed '/^[[:space:]]*$/d' | wc -l)"
    if (( active == 0 )); then
        idle_count=$((idle_count + 1))
    else
        idle_count=0
    fi
    gpu="$(nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw \
        --format=csv,noheader,nounits)"
    echo "idle confirmation ${idle_count}/2; active=${active}; gpu=${gpu}"
    sleep 15
done

echo "GPU idle; starting decode L2 residency smoke"
.venv/bin/python decode_l2_residency_experiment.py \
    --stage smoke --warmup 3 --repeat 10 \
    --output-dir assets/decode_l2_residency_smoke

echo "Smoke passed; starting full paired run"
.venv/bin/python decode_l2_residency_experiment.py \
    --stage full --warmup 5 --repeat 30 \
    --output-dir assets/decode_l2_residency

echo "Generating analysis"
exec .venv/bin/python analyze_decode_l2_residency.py \
    --input assets/decode_l2_residency/raw/results.jsonl \
    --output-dir assets/decode_l2_residency
