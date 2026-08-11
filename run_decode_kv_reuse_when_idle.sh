#!/usr/bin/env bash
set -euo pipefail

poll_seconds=15
required_idle_seconds=180

wait_sustained_idle() {
    local idle_seconds=0
    while (( idle_seconds < required_idle_seconds )); do
        local active
        active="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits \
            | sed '/^[[:space:]]*$/d' | wc -l)"
        if (( active == 0 )); then
            idle_seconds=$((idle_seconds + poll_seconds))
        else
            idle_seconds=0
        fi
        echo "sustained idle ${idle_seconds}/${required_idle_seconds}s; active=${active}"
        sleep "${poll_seconds}"
    done
}

run_exclusive() {
    local label="$1"
    shift
    "$@" &
    local bench_pid=$!
    local contaminated=0
    echo "${label} started as PID ${bench_pid}"
    while kill -0 "${bench_pid}" 2>/dev/null; do
        while read -r active_pid; do
            active_pid="${active_pid//[[:space:]]/}"
            [[ -z "${active_pid}" ]] && continue
            if [[ "${active_pid}" != "${bench_pid}" ]]; then
                echo "CONTAMINATION: external CUDA PID ${active_pid} appeared"
                contaminated=1
            fi
        done < <(nvidia-smi --query-compute-apps=pid \
            --format=csv,noheader,nounits)
        if (( contaminated != 0 )); then
            kill -INT "${bench_pid}" 2>/dev/null || true
            wait "${bench_pid}" || true
            return 2
        fi
        sleep 5
    done
    wait "${bench_pid}"
}

wait_sustained_idle
run_exclusive smoke .venv/bin/python decode_kv_reuse_experiment.py \
    --stage smoke --warmup 3 --repeat 15 \
    --output-dir assets/decode_kv_reuse_smoke

# Catch chained external jobs that start shortly after the first idle window.
wait_sustained_idle
run_exclusive full .venv/bin/python decode_kv_reuse_experiment.py \
    --stage full --warmup 5 --repeat 30 \
    --output-dir assets/decode_kv_reuse

exec .venv/bin/python analyze_decode_kv_reuse.py \
    --input assets/decode_kv_reuse/raw/results.jsonl \
    --output-dir assets/decode_kv_reuse
