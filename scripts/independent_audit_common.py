#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _build_summary(
    audits: Sequence[Dict[str, Any]],
    label_mismatch: Sequence[Tuple[Any, Any, Sequence[str]]],
    failure_counter: Counter,
) -> Dict[str, Any]:
    bad_rows = [row for row in audits if row.get("judge_outcome") == "BAD"]
    bad_failures: Counter = Counter()
    for row in bad_rows:
        violations = row.get("independent_violations")
        if isinstance(violations, list) and violations:
            bad_failures[str(violations[0])] += 1

    overfit_flags: List[str] = []
    if bad_rows:
        top = bad_failures.most_common(1)
        if top and top[0][1] / len(bad_rows) > 0.7:
            overfit_flags.append(
                f"Single failure mode dominates BAD set: {top[0][0]} ({top[0][1]}/{len(bad_rows)})"
            )
        if len(bad_failures) < 3:
            overfit_flags.append(f"Low BAD diversity: only {len(bad_failures)} primary failure modes")

    return {
        "total_rows": len(audits),
        "good_rows": sum(1 for row in audits if row.get("judge_outcome") == "GOOD"),
        "bad_rows": sum(1 for row in audits if row.get("judge_outcome") == "BAD"),
        "label_consistency_failures": len(label_mismatch),
        "independent_failure_counts": dict(failure_counter),
        "bad_primary_failure_diversity": dict(bad_failures),
        "overfit_flags": overfit_flags,
        "prompt_validity_result": "PASS" if len(label_mismatch) == 0 else "FAIL",
        "judge_overfit_risk": "LOW" if not overfit_flags else "MEDIUM",
    }


def _write_markdown(
    output_path: Path,
    report_title: str,
    summary: Dict[str, Any],
    label_mismatch: Sequence[Tuple[Any, Any, Sequence[str]]],
) -> None:
    lines = [
        f"# {report_title}",
        "",
        "## Summary",
        f"- Total rows: {summary['total_rows']}",
        f"- GOOD rows: {summary['good_rows']}",
        f"- BAD rows: {summary['bad_rows']}",
        f"- Label consistency failures: {summary['label_consistency_failures']}",
        f"- Prompt validity result: {summary['prompt_validity_result']}",
        f"- Judge overfit risk: {summary['judge_overfit_risk']}",
        "",
        "## Failure Counts (Independent)",
    ]

    for key, count in sorted(summary["independent_failure_counts"].items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"- {key}: {count}")

    lines.extend(["", "## BAD Primary Failure Diversity"])
    for key, count in sorted(summary["bad_primary_failure_diversity"].items(), key=lambda kv: (-kv[1], kv[0])):
        lines.append(f"- {key}: {count}")

    if summary["overfit_flags"]:
        lines.extend(["", "## Overfit Flags"])
        for flag in summary["overfit_flags"]:
            lines.append(f"- {flag}")

    if label_mismatch:
        lines.extend(["", "## Label Mismatches"])
        for trace_id, label, violations in label_mismatch:
            lines.append(f"- {trace_id}: label={label}, violations={list(violations)}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_independent_audit(
    *,
    golden_path: Path,
    out_json_path: Path,
    out_md_path: Path,
    report_title: str,
    audit_row_fn: Callable[[Dict[str, Any]], List[str]],
) -> int:
    rows = read_jsonl(golden_path)
    audits: List[Dict[str, Any]] = []
    label_mismatch: List[Tuple[Any, Any, Sequence[str]]] = []
    failure_counter: Counter = Counter()

    for row in rows:
        trace_id = row.get("trace_id")
        label = row.get("judge_outcome")
        violations = audit_row_fn(row)
        for violation in violations:
            failure_counter[violation] += 1

        if label == "GOOD" and violations:
            consistency = "FAIL"
            label_mismatch.append((trace_id, label, violations))
        elif label == "BAD" and not violations:
            consistency = "FAIL"
            label_mismatch.append((trace_id, label, violations))
        else:
            consistency = "PASS"

        audits.append(
            {
                "trace_id": trace_id,
                "judge_outcome": label,
                "independent_violations": violations,
                "label_consistency": consistency,
            }
        )

    summary = _build_summary(audits, label_mismatch, failure_counter)

    out_json_path.write_text(json.dumps({"summary": summary, "rows": audits}, indent=2) + "\n", encoding="utf-8")
    _write_markdown(out_md_path, report_title, summary, label_mismatch)

    print(json.dumps(summary, indent=2))
    return 0
