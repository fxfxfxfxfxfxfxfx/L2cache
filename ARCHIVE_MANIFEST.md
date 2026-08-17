# Local Archive Manifest

Repository cleanup was performed on 2026-08-14. No experiment data was deleted.
Non-core material was moved on the same filesystem to:

```text
/root/L2cache-archive-20260814/
```

## Inventory

| Item | Value |
|---|---:|
| Archived files | 1,183 |
| Archive bytes after extraction of core artifacts | 2,891,850,701 |
| Raw CSA sample files | 34 |
| Raw CSA sample bytes | 2,748,537,015 |

`original/assets/` is the pre-cleanup assets tree. `original/root/` contains the
old README, log, discarded analysis scripts and the exact modified versions
that existed before cleanup. `original/generated/` contains the previous local
cache and Python bytecode.

The archive includes dense/BF16 grids, Figure 2 outputs, cache-locality and L2
residency experiments, the manual overlap sweep, smoke/probe runs, shared-batch
CSA simulation, per-shape figures, PDFs and all original rerun files.

## Authoritative merge rules

- CSA trace replay: base, then `rerun_outliers`, keyed by `case_id`.
- CSA batch outer: base, then `rerun_outliers`, `rerun_residual`,
  `rerun_thermal`, keyed by `case_id`.
- CSA batch inner: base, then `supplement_memory_v2`, then `rerun_outliers`,
  keyed by `case_id`.
- Random baseline has no override.

The resulting authoritative row counts are 256 replay rows, 1,004 batch-outer
rows and 1,004 batch-inner rows. The consolidated files are checked in under
`artifacts/data/`.

## Raw trace SHA256

```text
row_0000  8b237ee188407c6313a1fbfb2e0330c65434e23fd2a9fac1aead45896818487c
row_0003  2d61f01f13ad9bcd7300f80e71dd6f1b82e94e70f2c2e35cbb583c78b8aafe15
row_0004  5fccc33e8e8fe857fcd505964da4b0e39d9cdc8a5af0497f5a4e4ad4f2a122b7
row_0005  fe981c760307a0a1173e0decf06462869f4f28fd09aca8fde8d8f7d136acce78
row_0008  3b23deb27e7538b6cc7b72fa0d380783006bdfa227e0c197a60911b699e146fc
row_0012  34c2a3b006c164be7061450db208f9827589a1cc3814005363722bba7cf98735
row_0018  305a52e4df08d7219198a359693cd5f0e4a371fe10837cb6c097d9518cf1e2bc
row_0019  dcc958553bb65274fac6766af263041d1dd052ec6291a36686cb3f492dcfbff7
```

## Pre-cleanup working tree

The worktree already contained modified and untracked experiment work. Those
versions were preserved rather than reset. The pre-cleanup status was:

```text
M  .gitignore
M  README.md
M  analyze_overlap_attribution.py
M  overlap_attribution_experiment.py
?? analyze_csa_*.py
?? csa_*benchmark.py
?? download_csa_trace_sample.py
?? test_csa_trace.py
?? test_overlap_dose_response.py
?? assets/local_*/
```

To restore an archived path, copy or move the corresponding item from
`/root/L2cache-archive-20260814/original/` back to its original relative path.
The archive is deliberately outside Git and will not accompany a clone.
