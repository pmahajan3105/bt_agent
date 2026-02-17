#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from typing import Any, Dict, List

from independent_audit_common import run_independent_audit
from project_paths import TRACES_DIR

GOLDEN = TRACES_DIR / "insight_opp_stage_bg_golden_30.jsonl"
OUT_JSON = TRACES_DIR / "insight_opp_stage_bg_golden_30_independent_audit.json"
OUT_MD = TRACES_DIR / "insight_opp_stage_bg_golden_30_independent_audit.md"


def extract_opps(inp: Dict[str, Any]) -> List[Dict[str, Any]]:
    opportunity_context = inp.get("opportunity_context") if isinstance(inp.get("opportunity_context"), dict) else {}
    if isinstance(opportunity_context.get("opportunities"), list):
        return [item for item in opportunity_context["opportunities"] if isinstance(item, dict)]
    if isinstance(opportunity_context.get("opportunity"), dict):
        return [opportunity_context["opportunity"]]
    return []


def is_closed_opp(opp: Dict[str, Any]) -> bool:
    if opp.get("actual_close_date"):
        return True
    stage = str(opp.get("stage") or opp.get("stage_human") or "").lower()
    return any(token in stage for token in ["closed_won", "closed_lost", "closed-won", "closed-lost", "won", "lost"])


def risk_items(opps: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for opp in opps:
        risks = opp.get("risks")
        if isinstance(risks, list):
            for risk in risks:
                if isinstance(risk, str) and risk.strip():
                    out.append(risk.strip())
                elif isinstance(risk, dict):
                    text = risk.get("text") or risk.get("risk") or risk.get("description")
                    if isinstance(text, str) and text.strip():
                        out.append(text.strip())
    return out


def token_set(text: str) -> set[str]:
    return {word.lower() for word in re.findall(r"[A-Za-z0-9_']+", text or "") if len(word) > 3}


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

    text = "\n".join(
        [
            summary if isinstance(summary, str) else "",
            *([h for h in highlights if isinstance(h, str)] if isinstance(highlights, list) else []),
            *([r for r in recommendation if isinstance(r, str)] if isinstance(recommendation, list) else []),
            action_reasoning if isinstance(action_reasoning, str) else "",
        ]
    )
    low = text.lower()

    if "—" in text:
        violations.append("em_dash_present")

    opps = extract_opps(inp)
    all_closed = bool(opps) and all(is_closed_opp(opp) for opp in opps)

    if all_closed:
        if any(token in low for token in ["at risk", "needs attention", "needs re-engagement", "inactive"]):
            violations.append("closed_misclassified")
        if not any(token in low for token in ["closed", "won", "lost"]):
            violations.append("closed_outcome_missing")
        if isinstance(recommendation, list):
            if "review other active opportunit" not in "\n".join(recommendation).lower():
                violations.append("closed_recommendation_mismatch")
    else:
        if isinstance(summary, str) and len(re.findall(r"[A-Za-z0-9_']+", summary)) > 50:
            violations.append("summary_too_long")
        if isinstance(highlights, list) and not (2 <= len(highlights) <= 6):
            violations.append("highlights_count_out_of_range")
        if not (isinstance(recommendation, list) and 1 <= len(recommendation) <= 5):
            violations.append("open_recommendation_count_or_type")

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

        opportunity_context = inp.get("opportunity_context") if isinstance(inp.get("opportunity_context"), dict) else {}
        tasks = opportunity_context.get("tasks") if isinstance(opportunity_context.get("tasks"), dict) else {}
        total_open = tasks.get("total_open")
        total_overdue = tasks.get("total_overdue")

        if isinstance(total_open, (int, float)) and total_open > 0 and not any(
            token in low for token in ["task", "overdue", "due", "follow up", "follow-up"]
        ):
            violations.append("tasks_not_surfaced")
        if isinstance(total_overdue, (int, float)) and total_overdue > 0 and "overdue" not in low:
            violations.append("overdue_not_prioritized")

        risks = risk_items(opps)
        if not risks and re.search(r"\brisks?\b", low):
            violations.append("invented_risk_without_source")

        source = " ".join(str(opp.get("summary") or "") for opp in opps)
        if source.strip() and isinstance(summary, str):
            if len(token_set(source).intersection(token_set(summary))) < 2 and any(
                phrase in summary.lower()
                for phrase in ["this opportunity is in", "deal is in", "deal is progressing", "active deal"]
            ):
                violations.append("summary_missing_source_context")

    return sorted(set(violations))


def main() -> int:
    return run_independent_audit(
        golden_path=GOLDEN,
        out_json_path=OUT_JSON,
        out_md_path=OUT_MD,
        report_title="Independent Audit: Insight Opportunity Stage Golden 30",
        audit_row_fn=audit_row,
    )


if __name__ == "__main__":
    raise SystemExit(main())
