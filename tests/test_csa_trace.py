#!/usr/bin/env python3
"""CPU checks for CSA C4 trace replay mapping."""

import numpy as np

from scripts.csa_trace_benchmark import (
    adjacent_overlap_c4,
    consecutive_c4_fraction,
    expand_c4_window,
    parse_shape_triples,
)
from scripts.analyze_csa_trace_benchmark import aggregate, load_latest_many, pair_rows
from scripts.download_csa_trace_sample import METADATA_ROWS, TRACE_ROWS, sample_paths
from scripts.csa_profile_sim_benchmark import (
    batch_seed,
    build_simulated_c4,
    load_profiles,
    parse_shape_triples as parse_sim_shape_triples,
    profile_label,
)
from scripts.analyze_csa_profile_sim import latest_rows as latest_profile_sim_rows


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def token_overlap(expanded):
    values = []
    for left, right in zip(expanded, expanded[1:]):
        values.append(len(set(left.tolist()) & set(right.tolist())) / len(left))
    return float(np.mean(values))


def test_c4_expansion():
    raw = np.stack([
        np.arange(offset, offset + 512, dtype=np.int32)
        for offset in (0, 64, 128)
    ])
    expanded = expand_c4_window(raw, history=4096)
    check(expanded.shape == (3, 2048), "C4 expands 512 entries to topk=2048")
    check(expanded.dtype == np.int32, "expanded indices remain int32")
    check(int(expanded.min()) == 0 and int(expanded.max()) < 4096,
          "expanded indices stay in the historical prefix")
    check(all(np.unique(row).size == 2048 for row in expanded),
          "every expanded row contains 2048 unique tokens")
    check(np.isclose(adjacent_overlap_c4(raw), token_overlap(expanded)),
          "four-token expansion preserves set-overlap fraction")


def test_invalid_c4_rows():
    padded = np.tile(np.arange(512, dtype=np.int32), (2, 1))
    padded[0, 0] = -1
    try:
        expand_c4_window(padded, history=4096)
    except ValueError:
        pass
    else:
        raise AssertionError("padding must be rejected")

    duplicate = np.tile(np.arange(512, dtype=np.int32), (2, 1))
    duplicate[0, 1] = duplicate[0, 0]
    try:
        expand_c4_window(duplicate, history=4096)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate C4 entries must be rejected")


def test_consecutive_fraction():
    consecutive = np.tile(np.arange(512, dtype=np.int32), (2, 1))
    check(consecutive_c4_fraction(consecutive) == 1.0,
          "dense C4 range is fully consecutive")


def test_shape_triples():
    check(
        parse_shape_triples("8:8192:1024,32:65536:256")
        == {(8, 8192, 1024), (32, 65536, 256)},
        "exact rerun shapes parse as B:H:Q triples",
    )


def test_bounded_sample_selection():
    paths = sample_paths()
    check(len(METADATA_ROWS) == 24, "selection keeps first-page metadata")
    check(len(TRACE_ROWS) == 8, "selection is bounded to eight trace tensors")
    check(len(paths) == 34 and len(paths) == len(set(paths)),
          "sample paths are complete and unique")


def test_analysis_override_and_pairing(tmp_dir=None):
    import tempfile
    from pathlib import Path
    import json

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        base = root / "base.jsonl"
        override = root / "override.jsonl"
        common = {
            "status": "ok", "B": 1, "H": 8192, "Q": 64,
            "latency_ms_p5": 0.9, "latency_ms_p95": 1.1,
            "achieved_adjacent_overlap": 0.25, "unique_kv": 4096,
            "reuse_factor": 32.0, "c4_consecutive_fraction": None,
        }
        rows = []
        for pattern, latency, tflops in (
            ("original_strided", 2.0, 100.0),
            ("csa_trace_replay", 1.0, 200.0),
        ):
            for order in ("ascending", "descending"):
                row = dict(common, pattern=pattern, pass_order=order,
                           latency_ms_median=latency, tflops=tflops,
                           case_id=f"{pattern}_{order}")
                if pattern == "csa_trace_replay":
                    row.update(achieved_adjacent_overlap=0.75, unique_kv=2048,
                               c4_consecutive_fraction=0.5)
                rows.append(row)
        base.write_text("".join(json.dumps(row) + "\n" for row in rows))
        replacement = dict(rows[2], latency_ms_median=0.8, tflops=250.0)
        override.write_text(json.dumps(replacement) + "\n")
        latest = load_latest_many([base, override])
        paired = pair_rows(aggregate(latest))
        check(len(latest) == 4 and len(paired) == 1,
              "case-id override preserves one complete shape")
        check(np.isclose(paired[0]["trace_tflops"], 225.0),
              "override is included in two-pass median")
        check(np.isclose(paired[0]["trace_vs_original_tflops"], 2.25),
              "paired throughput ratio is computed from aggregates")


def test_profile_simulator():
    profiles = load_profiles("artifacts/data/csa_trace_profile")
    check(profile_label(2048) == "2K-8K", "2K uses the first trace profile")
    check(profile_label(524288) == "64K+", "512K uses the extrapolated profile")
    for H, Q in ((2048, 32), (65536, 128), (524288, 512)):
        raw, labels, targets, _ = build_simulated_c4(H, Q, 7, profiles)
        check(raw.shape == (Q, 512) and raw.dtype == np.int32,
              "simulated C4 tensor has the expected shape and dtype")
        for query, row in enumerate(raw):
            check(np.unique(row).size == 512,
                  "each simulated row has 512 unique C4 entries")
            check(int(row.min()) >= 0 and int(row.max()) < (H + query + 1) // 4,
                  "simulated C4 indices remain causal")
        if Q > 1:
            achieved = []
            for left, right in zip(raw, raw[1:]):
                achieved.append(
                    np.intersect1d(left, right, assume_unique=True).size / 512
                )
            check(np.allclose(achieved, targets),
                  "every transition realizes its quantized overlap target")
        check(labels[-1] == profile_label(H + Q),
              "profile selection follows per-query visible history")
    check(
        parse_sim_shape_triples("2:524288:2,128:4096:512")
        == {(2, 524288, 2), (128, 4096, 512)},
        "profile-simulator exact rerun shapes parse correctly",
    )
    first, _, _, _ = build_simulated_c4(8192, 32, batch_seed(7, 0), profiles)
    second, _, _, _ = build_simulated_c4(8192, 32, batch_seed(7, 1), profiles)
    check(not np.array_equal(first, second),
          "different batch elements receive independent simulated sequences")


def test_profile_sim_row_layout():
    import torch
    from scripts.csa_profile_sim_benchmark import reorder_query_rows

    rows = torch.arange(6).view(6, 1)
    outer = reorder_query_rows(rows, 2, 3, "batch-outer")
    inner = reorder_query_rows(rows, 2, 3, "batch-inner")
    check(outer[:, 0].tolist() == [0, 1, 2, 3, 4, 5],
          "batch-outer preserves [B,Q] flattening")
    check(inner[:, 0].tolist() == [0, 3, 1, 4, 2, 5],
          "batch-inner interleaves batch rows within each query position")
    restored = inner.view(3, 2, 1).transpose(0, 1).reshape(6, 1)
    check(torch.equal(restored, rows), "batch-inner row permutation is lossless")


def test_profile_sim_analysis_override():
    import tempfile
    from pathlib import Path
    import json

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        base = root / "base.jsonl"
        override = root / "override.jsonl"
        base.write_text(json.dumps({"case_id": "same", "value": 1}) + "\n")
        override.write_text(json.dumps({"case_id": "same", "value": 2}) + "\n")
        rows = latest_profile_sim_rows([base, override])
        check(len(rows) == 1 and rows[0]["value"] == 2,
              "profile-sim analysis applies later case-id overrides")


def main():
    tests = [
        test_c4_expansion, test_invalid_c4_rows, test_consecutive_fraction,
        test_shape_triples, test_bounded_sample_selection,
        test_analysis_override_and_pairing, test_profile_simulator,
        test_profile_sim_row_layout,
        test_profile_sim_analysis_override,
    ]
    for test in tests:
        test()
        print(f"[PASS] {test.__name__}")
    print(f"{len(tests)}/{len(tests)} tests passed")


if __name__ == "__main__":
    main()
