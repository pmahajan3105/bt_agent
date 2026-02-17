#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from insight_prospecting_pipeline_lib import (
    BraintrustClient,
    PROJECT_ID,
    PROMPT_SLUG,
    build_slug_query,
    curate_golden,
    label_candidates,
    load_dotenv,
    normalize_traces,
    to_golden_rows,
    to_labels_only_rows,
    to_renamed_rows,
    write_json,
    write_jsonl,
)


def _count_query_by_slug(project_id: str, slug: str, hours: int) -> str:
    return (
        "select: count(*) as n\n"
        f"from: project_logs('{project_id}')\n"
        "spans\n"
        "filter: "
        f"created > now() - interval {hours} hour and "
        "created < now() and "
        f"metadata.prompt_slug = '{slug}' and "
        "span_attributes.parent_span_id is null\n"
    )


def _count_query_by_prompt_name(project_id: str, prompt_name: str, hours: int) -> str:
    return (
        "select: count(*) as n\n"
        f"from: project_logs('{project_id}')\n"
        "spans\n"
        "filter: "
        f"created > now() - interval {hours} hour and "
        "created < now() and "
        f"metadata.prompt_name = '{prompt_name}' and "
        "span_attributes.parent_span_id is null\n"
    )


def _extract_count(resp: Dict[str, Any]) -> int:
    data = resp.get("data")
    if isinstance(data, list) and data and isinstance(data[0], dict):
        n = data[0].get("n")
        if isinstance(n, (int, float)):
            return int(n)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Run full insight prospecting golden pipeline")
    ap.add_argument("--project-id", default=PROJECT_ID)
    ap.add_argument("--prompt-slug", default=PROMPT_SLUG)
    ap.add_argument("--prompt-name", default="account-overview-prospecting-summarizer-background")
    ap.add_argument("--judge-file", required=True)
    ap.add_argument("--out-dir", default="/Users/prashant/bt_agent/traces")
    ap.add_argument("--target-good", type=int, default=20)
    ap.add_argument("--target-bad", type=int, default=10)
    ap.add_argument("--sample-per-window", type=int, default=1000)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    queries_dir = out_dir / "queries"
    queries_dir.mkdir(parents=True, exist_ok=True)

    load_dotenv(Path("/Users/prashant/bt_agent/.env"))
    api_key = os.environ.get("BRAINTRUST_API_KEY")
    if not api_key:
        raise SystemExit("Missing BRAINTRUST_API_KEY")

    judge_text = Path(args.judge_file).read_text(encoding="utf-8").strip()
    if not judge_text:
        raise SystemExit("Judge file is empty")

    bt = BraintrustClient(
        api_key=api_key,
        base_url=os.environ.get("BRAINTRUST_BASE_URL", "https://api.braintrust.dev"),
    )

    windows = [72, 168, 720, 2160, 8760]
    all_rows_by_id: Dict[str, Dict[str, Any]] = {}

    slug_count_72 = _extract_count(bt.btql(_count_query_by_slug(args.project_id, args.prompt_slug, 72)))
    prompt_name_count_72 = _extract_count(
        bt.btql(_count_query_by_prompt_name(args.project_id, args.prompt_name, 72))
    )

    latest_selected_summary: Dict[str, Any] = {}
    final_selected: List[Dict[str, Any]] = []
    final_curation_log: List[Dict[str, Any]] = []
    final_candidates: List[Dict[str, Any]] = []
    final_rejects: List[Dict[str, Any]] = []
    final_judged: List[Dict[str, Any]] = []
    used_window = None

    for hours in windows:
        query = build_slug_query(args.project_id, args.prompt_slug, hours, sample=args.sample_per_window)
        query_path = queries_dir / f"insight_prospecting_bg_{hours}h.btql"
        query_path.write_text(query, encoding="utf-8")

        resp = bt.btql(query)
        rows = resp.get("data", []) if isinstance(resp.get("data"), list) else []

        for row in rows:
            if not isinstance(row, dict):
                continue
            trace_id = str(row.get("id") or row.get("root_span_id") or "").strip()
            if not trace_id:
                continue
            if trace_id in all_rows_by_id:
                continue
            row2 = dict(row)
            row2["window_hours"] = hours
            all_rows_by_id[trace_id] = row2

        all_rows = list(all_rows_by_id.values())
        raw_path = out_dir / "insight_prospecting_bg_raw_traces.json"
        write_json(raw_path, {"data": all_rows})

        candidates, rejects = normalize_traces(all_rows)
        for c in candidates:
            if c.get("window_hours") is None:
                c["window_hours"] = hours

        judged = label_candidates(candidates)
        selected, curation_log, summary = curate_golden(
            judged_rows=judged,
            target_good=args.target_good,
            target_bad=args.target_bad,
        )

        write_jsonl(out_dir / "insight_prospecting_bg_candidates_structured.jsonl", candidates)
        write_jsonl(out_dir / "insight_prospecting_bg_rejected_rows.jsonl", rejects)
        write_jsonl(out_dir / "insight_prospecting_bg_judgements_v1.jsonl", judged)
        write_json(
            out_dir / "insight_prospecting_bg_judgements_v1_stats.json",
            {
                "candidate_count": len(candidates),
                "label_counts": dict(Counter(str(r.get("judge_outcome")) for r in judged)),
                "critical_violation_counts": dict(
                    Counter(v for r in judged for v in (r.get("critical_violations") or []))
                ),
            },
        )

        final_candidates = candidates
        final_rejects = rejects
        final_judged = judged
        final_selected = selected
        final_curation_log = curation_log
        latest_selected_summary = summary
        used_window = hours

        if (
            int(summary.get("selected_good", 0)) >= args.target_good
            and int(summary.get("selected_bad", 0)) >= args.target_bad
        ):
            break

    if not final_selected:
        raise SystemExit("No selected traces found")

    golden_rows = to_golden_rows(final_selected)
    renamed_rows = to_renamed_rows(golden_rows)
    labels_only_rows = to_labels_only_rows(golden_rows)

    golden_path = out_dir / "insight_prospecting_bg_golden_30.jsonl"
    renamed_path = out_dir / "insight_prospecting_bg_golden_30_renamed.jsonl"
    labels_path = out_dir / "insight_prospecting_bg_golden_30_labels_only.jsonl"
    curation_log_path = out_dir / "insight_prospecting_bg_curation_log.jsonl"

    write_jsonl(golden_path, golden_rows)
    write_jsonl(renamed_path, renamed_rows)
    write_jsonl(labels_path, labels_only_rows)
    write_jsonl(curation_log_path, final_curation_log)

    total = len(golden_rows)
    good_count = sum(1 for r in golden_rows if r.get("judge_outcome") == "GOOD")
    bad_count = sum(1 for r in golden_rows if r.get("judge_outcome") == "BAD")
    unique_ids = len({str(r.get("trace_id") or "") for r in golden_rows})
    renamed_has_context = all("context" in r and "input" not in r for r in renamed_rows)

    if total != args.target_good + args.target_bad:
        raise SystemExit(f"Golden size mismatch: {total}")
    if good_count != args.target_good:
        raise SystemExit(f"GOOD count mismatch: {good_count}")
    if bad_count != args.target_bad:
        raise SystemExit(f"BAD count mismatch: {bad_count}")
    if unique_ids != total:
        raise SystemExit("Duplicate trace_id in golden output")
    if not renamed_has_context:
        raise SystemExit("Renamed file validation failed")

    selected_ids = {str(r.get("trace_id")) for r in golden_rows}
    missing_selected_in_log = [
        rid
        for rid in selected_ids
        if not any(
            str(log.get("trace_id")) == rid and str(log.get("worthy_decision")) == "selected"
            for log in final_curation_log
        )
    ]
    if missing_selected_in_log:
        raise SystemExit("Curation integrity failed: selected traces missing in curation log")

    rejected_without_reason = [
        log
        for log in final_curation_log
        if str(log.get("worthy_decision")) == "rejected"
        and (
            not isinstance(log.get("reject_reasons"), list)
            or len(log.get("reject_reasons")) == 0
        )
    ]
    if rejected_without_reason:
        raise SystemExit("Curation integrity failed: rejected entries without reject reasons")

    report_lines = [
        "# Insight Prospecting Stage Golden Dataset Report",
        "",
        "## Query Correctness",
        f"- Prompt name: `{args.prompt_name}`",
        f"- Prompt slug: `{args.prompt_slug}`",
        f"- Project id: `{args.project_id}`",
        f"- Count last 72h using `metadata.prompt_slug`: **{slug_count_72}**",
        f"- Count last 72h using `metadata.prompt_name`: **{prompt_name_count_72}**",
        "- Result: both prompt_name and prompt_slug are present, slug filter used for consistency with logs pipeline.",
        "",
        "## Pipeline Results",
        f"- Windows attempted: {windows}",
        f"- Final window used: {used_window}h",
        f"- Raw unique traces fetched: {len(all_rows_by_id)}",
        f"- Structured candidates: {len(final_candidates)}",
        f"- Rejected during normalization: {len(final_rejects)}",
        f"- Judged rows: {len(final_judged)}",
        f"- Curation reviewed rows: {len(final_curation_log)}",
        f"- Golden selected: {total} (GOOD={good_count}, BAD={bad_count})",
        "",
        "## Validation",
        f"- Exact count 30: {'PASS' if total == 30 else 'FAIL'}",
        f"- Exact split 20 GOOD / 10 BAD: {'PASS' if good_count == 20 and bad_count == 10 else 'FAIL'}",
        f"- No duplicate trace_id: {'PASS' if unique_ids == total else 'FAIL'}",
        f"- Renamed schema context/no-input: {'PASS' if renamed_has_context else 'FAIL'}",
        f"- Selected traces present in curation log: {'PASS' if not missing_selected_in_log else 'FAIL'}",
        f"- Rejected traces have reasons: {'PASS' if not rejected_without_reason else 'FAIL'}",
        "",
        "## Output Files",
        f"- `{golden_path}`",
        f"- `{renamed_path}`",
        f"- `{labels_path}`",
        f"- `{curation_log_path}`",
        f"- `{out_dir / 'insight_prospecting_bg_judgements_v1.jsonl'}`",
        f"- `{out_dir / 'insight_prospecting_bg_judgements_v1_stats.json'}`",
        "",
        "## Selection Summary",
        f"- Diversity buckets selected: {json.dumps(latest_selected_summary.get('diversity_buckets_selected', []))}",
        f"- BAD primary failures selected: {json.dumps(latest_selected_summary.get('primary_failures_selected', {}), default=str)}",
    ]

    report_path = out_dir / "insight_prospecting_bg_golden_30_report.md"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "selected_total": total,
                "selected_good": good_count,
                "selected_bad": bad_count,
                "final_window_used": used_window,
                "golden_path": str(golden_path),
                "renamed_path": str(renamed_path),
                "labels_path": str(labels_path),
                "curation_log_path": str(curation_log_path),
                "report_path": str(report_path),
            },
            indent=2,
        )
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
