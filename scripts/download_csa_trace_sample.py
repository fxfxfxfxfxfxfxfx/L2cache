#!/usr/bin/env python3
"""Download the bounded CSA trace sample used by the local analysis."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


REPO_ID = "fxiaoO/deepseek-v4-flash-swebench-csa-topk"
TRACE_ROOT = (
    "lite_agentic_full_trace_prefill/data_full_trace_prefill/lite_agentic"
)
METADATA_ROWS = tuple(range(24))
TRACE_ROWS = (0, 3, 4, 5, 8, 12, 18, 19)


def sample_paths():
    paths = ["README.md", "lite_agentic_full_trace_prefill/README.md"]
    paths.extend(
        f"{TRACE_ROOT}/row_{row:04d}/metadata.json" for row in METADATA_ROWS
    )
    paths.extend(
        f"{TRACE_ROOT}/row_{row:04d}/indexer_topk.npz" for row in TRACE_ROWS
    )
    return paths


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", default="local_data/csa_trace_source"
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        from modelscope_hub.api import HubApi
    except ImportError as error:
        raise RuntimeError(
            "install modelscope-hub first: python -m pip install modelscope-hub"
        ) from error

    token = os.environ.get("MODELSCOPE_API_TOKEN")

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    api = HubApi(token=token)
    if token:
        api.login(token)
    downloaded = []
    for index, repo_path in enumerate(sample_paths(), start=1):
        local_path = api.download_file(
            REPO_ID, repo_type="dataset", file_path=repo_path,
            local_dir=output, force=args.force,
        )
        local_path = Path(local_path)
        downloaded.append({
            "repo_path": repo_path,
            "local_path": str(local_path.resolve()),
            "bytes": local_path.stat().st_size,
        })
        print(
            f"[{index}/{len(sample_paths())}] {repo_path} "
            f"({local_path.stat().st_size / (1 << 20):.1f} MiB)",
            flush=True,
        )

    manifest = {
        "repo_id": REPO_ID,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "selection": {
            "metadata_rows": list(METADATA_ROWS),
            "trace_rows": list(TRACE_ROWS),
            "reason": "length-stratified bounded sample from the first lite page",
        },
        "files": downloaded,
        "total_bytes": sum(item["bytes"] for item in downloaded),
    }
    manifest_path = output.parent / "sample_selection.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote non-secret manifest to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
