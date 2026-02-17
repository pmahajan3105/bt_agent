#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from independent_audit_common import run_independent_audit
from project_paths import TRACES_DIR

GOLDEN = TRACES_DIR / "insight_new_stage_bg_golden_30.jsonl"
OUT_JSON = TRACES_DIR / "insight_new_stage_bg_golden_30_independent_audit.json"
OUT_MD = TRACES_DIR / "insight_new_stage_bg_golden_30_independent_audit.md"


def _intent_signals(inp: Dict[str, Any]) -> List[str]:
    signals = inp.get("signals") if isinstance(inp.get("signals"), dict) else {}
    values = signals.get("intent_signals") if isinstance(signals, dict) else []
    if isinstance(values, list):
        return [str(item).strip() for item in values if str(item).strip()]
    return []


def _mentions_buyer_gap(text: str) -> bool:
    phrases = [
        "no buyers connected",
        "no connected buyer",
        "no connected buyers",
        "buyers identified but no",
        "buyer identified but no",
        "0 connected",
        "none connected",
        "no connections",
        "no warm path",
    ]
    low = text.lower()
    return any(phrase in low for phrase in phrases)


def _mentions_single_threaded(text: str) -> bool:
    phrases = [
        "single-threaded",
        "single threaded",
        "single thread",
        "one connected buyer",
        "1 connected buyer",
        "only one connected",
        "single connected buyer",
    ]
    low = text.lower()
    return any(phrase in low for phrase in phrases)


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

    if isinstance(summary, str) and len(re.findall(r"[A-Za-z0-9_']+", summary)) > 55:
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
        if connected_buyers == 0 and not _mentions_buyer_gap(low):
            violations.append("no_buyers_connected_flag_missing")
        if total_buyers > 1 and connected_buyers == 1 and not _mentions_single_threaded(low):
            violations.append("single_threaded_flag_missing")

    intent_signals = _intent_signals(inp)
    if len(intent_signals) == 0:
        strong_intent_markers = [
            "strong intent signal",
            "strong ai intent",
            "high intent",
            "hot timing",
            "clear intent",
        ]
        if any(marker in low for marker in strong_intent_markers):
            violations.append("intent_claim_without_signal")

    return sorted(set(violations))


def main() -> int:
    return run_independent_audit(
        golden_path=GOLDEN,
        out_json_path=OUT_JSON,
        out_md_path=OUT_MD,
        report_title="Independent Audit: Insight New Stage Golden 30",
        audit_row_fn=audit_row,
    )


if __name__ == "__main__":
    raise SystemExit(main())
