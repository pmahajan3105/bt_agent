#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from insight_opp_stage_pipeline_lib import normalize_traces, read_json, write_jsonl


def main() -> int:
    ap = argparse.ArgumentParser(description="Normalize insight opportunity-stage traces")
    ap.add_argument("--input", required=True, help="Raw traces JSON path")
    ap.add_argument("--out-candidates-jsonl", required=True, help="Normalized candidates JSONL")
    ap.add_argument("--out-rejected-jsonl", required=True, help="Rejected rows JSONL")
    args = ap.parse_args()

    raw = read_json(Path(args.input))
    rows = raw.get("data", []) if isinstance(raw, dict) and isinstance(raw.get("data"), list) else raw
    if not isinstance(rows, list):
        raise SystemExit("Input must be an array or BTQL response with data[]")

    candidates, rejects = normalize_traces(rows)
    write_jsonl(Path(args.out_candidates_jsonl), candidates)
    write_jsonl(Path(args.out_rejected_jsonl), rejects)
    print(
        {
            "input_rows": len(rows),
            "candidates": len(candidates),
            "rejected": len(rejects),
            "out_candidates_jsonl": args.out_candidates_jsonl,
            "out_rejected_jsonl": args.out_rejected_jsonl,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
