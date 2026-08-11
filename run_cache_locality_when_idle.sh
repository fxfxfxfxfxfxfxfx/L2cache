#!/usr/bin/env bash
set -euo pipefail

watched_parent=58315
while kill -0 "${watched_parent}" 2>/dev/null; do
    count="$(wc -l < /root/sglang_indexer_benchmark/runs/full/raw/results.jsonl)"
    gpu="$(nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw \
        --format=csv,noheader,nounits)"
    echo "waiting: external grid rows=${count}; gpu=${gpu}"
    sleep 30
done

# Require two consecutive idle observations so a restarted CUDA process is
# not mistaken for a free interval.
idle_count=0
while (( idle_count < 2 )); do
    active="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
        | sed '/^[[:space:]]*$/d' | wc -l)"
    if (( active == 0 )); then
        idle_count=$((idle_count + 1))
    else
        idle_count=0
    fi
    echo "idle confirmation ${idle_count}/2; active CUDA processes=${active}"
    sleep 10
done

echo "GPU idle; starting clean cache-locality experiment"
exec .venv/bin/python cache_locality_experiment.py \
    --stage full --warmup 10 --repeat 30 \
    --output-dir assets/cache_locality_clean
