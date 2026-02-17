#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from insight_opp_stage_pipeline_lib import (
    curate_golden,
    read_jsonl,
    to_golden_rows,
    to_labels_only_rows,
    to_renamed_rows,
    write_json,
    write_jsonl,
)


def main() -> int:
    ap = argparse.ArgumentParser(description="Curate and build golden dataset")
    ap.add_argument("--input-judgements-jsonl", required=True)
    ap.add_argument("--target-good", type=int, default=20)
    ap.add_argument("--target-bad", type=int, default=10)
    ap.add_argument("--out-golden-jsonl", required=True)
    ap.add_argument("--out-renamed-jsonl", required=True)
    ap.add_argument("--out-labels-only-jsonl", required=True)
    ap.add_argument("--out-curation-log-jsonl", required=True)
    ap.add_argument("--out-summary-json", required=True)
    args = ap.parse_args()

    judged = read_jsonl(Path(args.input_judgements_jsonl))
    selected, curation_log, summary = curate_golden(
        judged_rows=judged,
        target_good=args.target_good,
        target_bad=args.target_bad,
    )

    golden_rows = to_golden_rows(selected)
    renamed_rows = to_renamed_rows(golden_rows)
    labels_rows = to_labels_only_rows(golden_rows)

    write_jsonl(Path(args.out_golden_jsonl), golden_rows)
    write_jsonl(Path(args.out_renamed_jsonl), renamed_rows)
    write_jsonl(Path(args.out_labels_only_jsonl), labels_rows)
    write_jsonl(Path(args.out_curation_log_jsonl), curation_log)
    write_json(Path(args.out_summary_json), summary)

    print(summary)

    selected_total = int(summary.get("selected_total", 0))
    selected_good = int(summary.get("selected_good", 0))
    selected_bad = int(summary.get("selected_bad", 0))
    if selected_total != args.target_good + args.target_bad:
        raise SystemExit(
            f"Golden count mismatch: selected={selected_total}, expected={args.target_good + args.target_bad}"
        )
    if selected_good != args.target_good:
        raise SystemExit(f"GOOD count mismatch: selected={selected_good}, expected={args.target_good}")
    if selected_bad != args.target_bad:
        raise SystemExit(f"BAD count mismatch: selected={selected_bad}, expected={args.target_bad}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
