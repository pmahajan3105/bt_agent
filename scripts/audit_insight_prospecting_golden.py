#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

from independent_audit_common import run_independent_audit
from project_paths import TRACES_DIR

GOLDEN = TRACES_DIR / "insight_prospecting_bg_golden_30.jsonl"
OUT_JSON = TRACES_DIR / "insight_prospecting_bg_golden_30_independent_audit.json"
OUT_MD = TRACES_DIR / "insight_prospecting_bg_golden_30_independent_audit.md"


def task_outcome_counts(prospecting_context: Dict[str, Any]) -> Tuple[int, int]:
    positive = 0
    negative = 0

    all_plays = prospecting_context.get("all_plays") if isinstance(prospecting_context.get("all_plays"), list) else []
    for play in all_plays:
        if not isinstance(play, dict):
            continue
        tasks = play.get("tasks") if isinstance(play.get("tasks"), list) else []
        for task in tasks:
            if not isinstance(task, dict):
                continue
            outcome = str(task.get("task_outcome") or "").upper()
            if outcome in {"POSITIVE_REPLY", "MEETING_BOOKED"}:
                positive += 1
            elif outcome in {"NEGATIVE_REPLY", "OPTED_OUT", "BOUNCED"}:
                negative += 1

    contact_engagement = (
        prospecting_context.get("contact_engagement")
        if isinstance(prospecting_context.get("contact_engagement"), list)
        else []
    )
    for engagement in contact_engagement:
        if not isinstance(engagement, dict):
            continue
        reply_sentiment = str(engagement.get("reply_sentiment") or "").upper()
        if reply_sentiment == "POSITIVE":
            positive += 1
        elif reply_sentiment == "NEGATIVE":
            negative += 1

    return positive, negative


def audit_row(row: Dict[str, Any]) -> List[str]:
    violations: List[str] = []
    inp = row.get("input") if isinstance(row.get("input"), dict) else {}

    try:
        out = json.loads(row.get("output_raw") or "")
    except Exception:
        return ["output_invalid_json"]

    if not isinstance(out, dict):
        return ["output_not_json_object"]

    summary = out.get("summary")
    highlights = out.get("highlights")
    recommendation = out.get("recommendation", None)
    action_reasoning = out.get("action_reasoning")

    if not isinstance(summary, str) or not summary.strip():
        violations.append("missing_summary")
    if not isinstance(highlights, list) or not all(isinstance(highlight, str) for highlight in highlights):
        violations.append("invalid_highlights")
    if not isinstance(action_reasoning, str) or not action_reasoning.strip():
        violations.append("missing_action_reasoning")
    if recommendation is not None and not (
        isinstance(recommendation, list) and all(isinstance(item, str) for item in recommendation)
    ):
        violations.append("invalid_recommendation_type")

    combined = "\n".join(
        [
            summary if isinstance(summary, str) else "",
            *([h for h in highlights if isinstance(h, str)] if isinstance(highlights, list) else []),
            *([r for r in recommendation if isinstance(r, str)] if isinstance(recommendation, list) else []),
            action_reasoning if isinstance(action_reasoning, str) else "",
        ]
    )
    low = combined.lower()

    if "—" in combined:
        violations.append("em_dash_present")

    if isinstance(summary, str) and len(re.findall(r"[A-Za-z0-9_']+", summary)) > 50:
        violations.append("summary_too_long")

    if isinstance(highlights, list) and not (2 <= len(highlights) <= 5):
        violations.append("highlights_count_out_of_range")

    has_sequences_value = inp.get("has_sequences")
    has_sequences = (
        has_sequences_value if isinstance(has_sequences_value, bool) else str(has_sequences_value).strip().lower() == "true"
    )
    if has_sequences and recommendation is not None:
        violations.append("has_sequences_requires_null_recommendation")
    if (not has_sequences) and not (isinstance(recommendation, list) and 1 <= len(recommendation) <= 3):
        violations.append("no_sequences_requires_1_to_3_recommendations")

    contact_coverage = inp.get("contact_coverage") if isinstance(inp.get("contact_coverage"), dict) else {}
    contacts = inp.get("contacts") if isinstance(inp.get("contacts"), dict) else {}
    total_buyers = contacts.get("total_buyers")
    connected_buyers = contact_coverage.get("connected_buyer_count")
    try:
        total_buyers = int(total_buyers) if total_buyers is not None else None
    except Exception:
        total_buyers = None
    try:
        connected_buyers = int(connected_buyers) if connected_buyers is not None else None
    except Exception:
        connected_buyers = None

    if total_buyers == 0:
        if "no buyers connected" in low or "single-threaded" in low:
            violations.append("coverage_guardrail_violated_total_buyers_zero")
        if "no buyer contacts identified" not in low:
            violations.append("no_buyer_contacts_identified_missing")
    elif total_buyers is not None and total_buyers > 0 and connected_buyers is not None:
        if connected_buyers == 0 and "no buyers connected" not in low:
            violations.append("no_buyers_connected_flag_missing")
        if total_buyers > 1 and connected_buyers == 1 and "single-threaded" not in low:
            violations.append("single_threaded_flag_missing")

    prospecting_context = inp.get("prospecting_context") if isinstance(inp.get("prospecting_context"), dict) else {}
    positive, negative = task_outcome_counts(prospecting_context)
    if positive + negative == 0:
        strong_engagement_terms = [
            "strong engagement",
            "high engagement",
            "momentum is strong",
            "replies received",
            "reply received",
            "engaged",
        ]
        if any(term in low for term in strong_engagement_terms) and "no replies" not in low:
            violations.append("strong_engagement_without_replies")

    return sorted(set(violations))


def main() -> int:
    return run_independent_audit(
        golden_path=GOLDEN,
        out_json_path=OUT_JSON,
        out_md_path=OUT_MD,
        report_title="Independent Audit: Insight Prospecting Golden 30",
        audit_row_fn=audit_row,
    )


if __name__ == "__main__":
    raise SystemExit(main())
