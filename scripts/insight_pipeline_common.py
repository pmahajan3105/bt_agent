#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

from project_paths import ENV_FILE, TRACES_DIR


@dataclass(frozen=True)
class PipelineConfig:
    description: str
    output_prefix: str
    report_title: str
    query_result_line: str
    prompt_name_default: str | None
    sample_per_window_default: int
    prompt_name_count_source: str
    prompt_name_count_label: str


DEFAULT_WINDOWS: Sequence[int] = (72, 168, 720, 2160, 8760)


def build_parser(config: PipelineConfig, project_id: str, prompt_slug: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=config.description)
    parser.add_argument("--project-id", default=project_id)
    parser.add_argument("--prompt-slug", default=prompt_slug)
    if config.prompt_name_default is not None:
        parser.add_argument("--prompt-name", default=config.prompt_name_default)
    parser.add_argument("--judge-file", required=True)
    parser.add_argument("--out-dir", default=str(TRACES_DIR))
    parser.add_argument("--target-good", type=int, default=20)
    parser.add_argument("--target-bad", type=int, default=10)
    parser.add_argument("--sample-per-window", type=int, default=config.sample_per_window_default)
    return parser


def _count_query(project_id: str, field_name: str, field_value: str, hours: int) -> str:
    return (
        "select: count(*) as n\n"
        f"from: project_logs('{project_id}')\n"
        "spans\n"
        "filter: "
        f"created > now() - interval {hours} hour and "
        "created < now() and "
        f"metadata.{field_name} = '{field_value}' and "
        "span_attributes.parent_span_id is null\n"
    )


def _extract_count(resp: Dict[str, Any]) -> int:
    data = resp.get("data")
    if isinstance(data, list) and data and isinstance(data[0], dict):
        n = data[0].get("n")
        if isinstance(n, (int, float)):
            return int(n)
    return 0


def _rejected_without_reason(log: Dict[str, Any]) -> bool:
    if str(log.get("worthy_decision")) != "rejected":
        return False
    reasons = log.get("reject_reasons")
    return not isinstance(reasons, list) or len(reasons) == 0


def _validate_outputs(
    *,
    golden_rows: List[Dict[str, Any]],
    renamed_rows: List[Dict[str, Any]],
    curation_log: List[Dict[str, Any]],
    target_good: int,
    target_bad: int,
) -> None:
    total = len(golden_rows)
    good_count = sum(1 for row in golden_rows if row.get("judge_outcome") == "GOOD")
    bad_count = sum(1 for row in golden_rows if row.get("judge_outcome") == "BAD")
    unique_ids = len({str(row.get("trace_id") or "") for row in golden_rows})
    renamed_has_context = all("context" in row and "input" not in row for row in renamed_rows)

    if total != target_good + target_bad:
        raise SystemExit(f"Golden size mismatch: {total}")
    if good_count != target_good:
        raise SystemExit(f"GOOD count mismatch: {good_count}")
    if bad_count != target_bad:
        raise SystemExit(f"BAD count mismatch: {bad_count}")
    if unique_ids != total:
        raise SystemExit("Duplicate trace_id in golden output")
    if not renamed_has_context:
        raise SystemExit("Renamed file validation failed")

    selected_ids = {str(row.get("trace_id")) for row in golden_rows}
    missing_selected_in_log = [
        trace_id
        for trace_id in selected_ids
        if not any(
            str(log.get("trace_id")) == trace_id and str(log.get("worthy_decision")) == "selected"
            for log in curation_log
        )
    ]
    if missing_selected_in_log:
        raise SystemExit("Curation integrity failed: selected traces missing in curation log")

    if any(_rejected_without_reason(log) for log in curation_log):
        raise SystemExit("Curation integrity failed: rejected entries without reject reasons")


def run_pipeline(config: PipelineConfig, pipeline_lib: Any, args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    queries_dir = out_dir / "queries"
    queries_dir.mkdir(parents=True, exist_ok=True)

    pipeline_lib.load_dotenv(ENV_FILE)
    api_key = os.environ.get("BRAINTRUST_API_KEY")
    if not api_key:
        raise SystemExit("Missing BRAINTRUST_API_KEY")

    judge_text = Path(args.judge_file).read_text(encoding="utf-8").strip()
    if not judge_text:
        raise SystemExit("Judge file is empty")

    bt = pipeline_lib.BraintrustClient(
        api_key=api_key,
        base_url=os.environ.get("BRAINTRUST_BASE_URL", "https://api.braintrust.dev"),
    )

    prompt_name_count_value = (
        args.prompt_name if config.prompt_name_count_source == "prompt_name" else args.prompt_slug
    )
    slug_count_72 = _extract_count(bt.btql(_count_query(args.project_id, "prompt_slug", args.prompt_slug, 72)))
    prompt_name_count_72 = _extract_count(
        bt.btql(_count_query(args.project_id, "prompt_name", prompt_name_count_value, 72))
    )

    all_rows_by_id: Dict[str, Dict[str, Any]] = {}
    latest_selected_summary: Dict[str, Any] = {}
    final_selected: List[Dict[str, Any]] = []
    final_curation_log: List[Dict[str, Any]] = []
    final_candidates: List[Dict[str, Any]] = []
    final_rejects: List[Dict[str, Any]] = []
    final_judged: List[Dict[str, Any]] = []
    used_window: int | None = None

    for hours in DEFAULT_WINDOWS:
        query = pipeline_lib.build_slug_query(
            args.project_id,
            args.prompt_slug,
            hours,
            sample=args.sample_per_window,
        )
        query_path = queries_dir / f"{config.output_prefix}_{hours}h.btql"
        query_path.write_text(query, encoding="utf-8")

        resp = bt.btql(query)
        rows = resp.get("data", []) if isinstance(resp.get("data"), list) else []

        for row in rows:
            if not isinstance(row, dict):
                continue
            trace_id = str(row.get("id") or row.get("root_span_id") or "").strip()
            if not trace_id or trace_id in all_rows_by_id:
                continue
            row_copy = dict(row)
            row_copy["window_hours"] = hours
            all_rows_by_id[trace_id] = row_copy

        all_rows = list(all_rows_by_id.values())
        pipeline_lib.write_json(out_dir / f"{config.output_prefix}_raw_traces.json", {"data": all_rows})

        candidates, rejects = pipeline_lib.normalize_traces(all_rows)
        for candidate in candidates:
            if candidate.get("window_hours") is None:
                candidate["window_hours"] = hours

        judged = pipeline_lib.label_candidates(candidates)
        selected, curation_log, summary = pipeline_lib.curate_golden(
            judged_rows=judged,
            target_good=args.target_good,
            target_bad=args.target_bad,
        )

        pipeline_lib.write_jsonl(out_dir / f"{config.output_prefix}_candidates_structured.jsonl", candidates)
        pipeline_lib.write_jsonl(out_dir / f"{config.output_prefix}_rejected_rows.jsonl", rejects)
        pipeline_lib.write_jsonl(out_dir / f"{config.output_prefix}_judgements_v1.jsonl", judged)
        pipeline_lib.write_json(
            out_dir / f"{config.output_prefix}_judgements_v1_stats.json",
            {
                "candidate_count": len(candidates),
                "label_counts": dict(Counter(str(row.get("judge_outcome")) for row in judged)),
                "critical_violation_counts": dict(
                    Counter(violation for row in judged for violation in (row.get("critical_violations") or []))
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

    golden_rows = pipeline_lib.to_golden_rows(final_selected)
    renamed_rows = pipeline_lib.to_renamed_rows(golden_rows)
    labels_only_rows = pipeline_lib.to_labels_only_rows(golden_rows)

    golden_path = out_dir / f"{config.output_prefix}_golden_30.jsonl"
    renamed_path = out_dir / f"{config.output_prefix}_golden_30_renamed.jsonl"
    labels_path = out_dir / f"{config.output_prefix}_golden_30_labels_only.jsonl"
    curation_log_path = out_dir / f"{config.output_prefix}_curation_log.jsonl"

    pipeline_lib.write_jsonl(golden_path, golden_rows)
    pipeline_lib.write_jsonl(renamed_path, renamed_rows)
    pipeline_lib.write_jsonl(labels_path, labels_only_rows)
    pipeline_lib.write_jsonl(curation_log_path, final_curation_log)

    _validate_outputs(
        golden_rows=golden_rows,
        renamed_rows=renamed_rows,
        curation_log=final_curation_log,
        target_good=args.target_good,
        target_bad=args.target_bad,
    )

    total = len(golden_rows)
    good_count = sum(1 for row in golden_rows if row.get("judge_outcome") == "GOOD")
    bad_count = sum(1 for row in golden_rows if row.get("judge_outcome") == "BAD")
    unique_ids = len({str(row.get("trace_id") or "") for row in golden_rows})
    renamed_has_context = all("context" in row and "input" not in row for row in renamed_rows)
    selected_ids = {str(row.get("trace_id")) for row in golden_rows}
    missing_selected_in_log = [
        trace_id
        for trace_id in selected_ids
        if not any(
            str(log.get("trace_id")) == trace_id and str(log.get("worthy_decision")) == "selected"
            for log in final_curation_log
        )
    ]
    rejected_without_reason = [log for log in final_curation_log if _rejected_without_reason(log)]

    report_lines = [
        f"# {config.report_title}",
        "",
        "## Query Correctness",
        f"- Prompt slug: `{args.prompt_slug}`",
        f"- Project id: `{args.project_id}`",
        f"- Count last 72h using `metadata.prompt_slug`: **{slug_count_72}**",
        f"- Count last 72h using {config.prompt_name_count_label}: **{prompt_name_count_72}**",
    ]

    if hasattr(args, "prompt_name"):
        report_lines.insert(3, f"- Prompt name: `{args.prompt_name}`")

    report_lines.extend(
        [
            f"- Result: {config.query_result_line}",
            "",
            "## Pipeline Results",
            f"- Windows attempted: {list(DEFAULT_WINDOWS)}",
            f"- Final window used: {used_window}h",
            f"- Raw unique traces fetched: {len(all_rows_by_id)}",
            f"- Structured candidates: {len(final_candidates)}",
            f"- Rejected during normalization: {len(final_rejects)}",
            f"- Judged rows: {len(final_judged)}",
            f"- Curation reviewed rows: {len(final_curation_log)}",
            f"- Golden selected: {total} (GOOD={good_count}, BAD={bad_count})",
            "",
            "## Validation",
            f"- Exact count {args.target_good + args.target_bad}: {'PASS' if total == (args.target_good + args.target_bad) else 'FAIL'}",
            f"- Exact split {args.target_good} GOOD / {args.target_bad} BAD: {'PASS' if good_count == args.target_good and bad_count == args.target_bad else 'FAIL'}",
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
            f"- `{out_dir / f'{config.output_prefix}_judgements_v1.jsonl'}`",
            f"- `{out_dir / f'{config.output_prefix}_judgements_v1_stats.json'}`",
            "",
            "## Selection Summary",
            f"- Diversity buckets selected: {json.dumps(latest_selected_summary.get('diversity_buckets_selected', []))}",
            f"- BAD primary failures selected: {json.dumps(latest_selected_summary.get('primary_failures_selected', {}), default=str)}",
        ]
    )

    report_path = out_dir / f"{config.output_prefix}_golden_30_report.md"
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
