#!/usr/bin/env python3
"""Merge append-only JSONL result files using last-file, last-row precedence."""

import argparse
import csv
import json
from pathlib import Path


def merge_rows(paths):
    latest = {}
    fields = []
    for path in paths:
        with Path(path).open() as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                case_id = row.get("case_id")
                if not case_id:
                    raise ValueError(f"row without case_id in {path}")
                latest[case_id] = row
                for field in row:
                    if field not in fields:
                        fields.append(field)
    return [latest[key] for key in sorted(latest)], fields


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()

    rows, fields = merge_rows(args.input)
    jsonl = Path(args.output_jsonl)
    csv_path = Path(args.output_csv)
    jsonl.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} authoritative rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
