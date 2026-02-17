#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from insight_opp_stage_pipeline_lib import label_candidates, read_jsonl, write_json, write_jsonl


def main() -> int:
    ap = argparse.ArgumentParser(description="Label normalized candidates with rule-based judge")
    ap.add_argument("--input-candidates-jsonl", required=True)
    ap.add_argument("--judge-file", required=True)
    ap.add_argument("--out-judgements-jsonl", required=True)
    ap.add_argument("--out-stats-json", required=True)
    args = ap.parse_args()

    # We keep judge-file argument to make the run explicit and auditable.
    judge_text = Path(args.judge_file).read_text(encoding="utf-8")
    if not judge_text.strip():
        raise SystemExit("Judge file is empty")

    candidates = read_jsonl(Path(args.input_candidates_jsonl))
    judged = label_candidates(candidates)

    stats = {
        "input_candidates": len(candidates),
        "judged_rows": len(judged),
        "label_counts": dict(Counter(str(r.get("judge_outcome")) for r in judged)),
        "critical_violation_counts": dict(
            Counter(v for r in judged for v in (r.get("critical_violations") or []))
        ),
        "judge_file": args.judge_file,
    }

    write_jsonl(Path(args.out_judgements_jsonl), judged)
    write_json(Path(args.out_stats_json), stats)
    print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
