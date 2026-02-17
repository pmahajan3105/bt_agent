#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

DEFAULT_BASE_URL = "https://api.braintrust.dev"
PROMPT_SLUG = "insight-account-overview-opportunity-stage-background"
PROJECT_ID = "1e7654f3-505d-402c-b24d-60048d3e6916"
REQUIRED_INPUT_KEYS = [
    "customer_context",
    "icp_criteria",
    "account_data",
    "contact_coverage",
    "contacts",
    "opportunity_context",
]
STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "about",
    "your",
    "their",
    "they",
    "them",
    "will",
    "would",
    "could",
    "should",
    "have",
    "has",
    "had",
    "were",
    "been",
    "being",
    "what",
    "when",
    "where",
    "which",
    "while",
    "than",
    "then",
    "also",
    "only",
    "very",
    "more",
    "less",
    "over",
    "under",
    "just",
    "into",
    "onto",
    "across",
    "around",
    "need",
    "needs",
    "risk",
    "risks",
    "deal",
    "stage",
    "opportunity",
}


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


class BraintrustClient:
    def __init__(self, api_key: str, base_url: str = DEFAULT_BASE_URL, timeout_s: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )

    def btql(self, query: str) -> Dict[str, Any]:
        resp = self.session.post(
            f"{self.base_url}/btql",
            json={"query": query, "fmt": "json"},
            timeout=self.timeout_s,
        )
        data = resp.json()
        if resp.status_code >= 400:
            raise RuntimeError(f"BTQL error ({resp.status_code}): {data}")
        return data


def build_slug_query(project_id: str, prompt_slug: str, hours: int, sample: int = 1000) -> str:
    return (
        "select: *\n"
        f"from: project_logs('{project_id}')\n"
        "spans\n"
        "filter: "
        f"created > now() - interval {hours} hour and "
        "created < now() and "
        "span_attributes.parent_span_id is null and "
        f"metadata.prompt_slug = '{prompt_slug}' and "
        "(span_attributes.name = 'chat_completions' or span_attributes.name = 'chat_completion')\n"
        f"sample: {sample}\n"
    )


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def _json_parse_value(value: Any) -> Tuple[Any, Optional[str]]:
    if isinstance(value, (dict, list)):
        return value, None
    if not isinstance(value, str):
        return None, "value_not_string_or_object"
    txt = value.strip()
    if not txt:
        return None, "value_empty_string"
    if txt[0] not in "[{":
        return None, "value_not_json_like_string"
    try:
        return json.loads(txt), None
    except Exception as e:
        return None, f"value_json_parse_error:{e.__class__.__name__}"


def normalize_trace_row(row: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    trace_id = str(row.get("id") or row.get("root_span_id") or "").strip()
    if not trace_id:
        return None, {"trace_id": "", "reason": "missing_trace_id"}

    inp = row.get("input")
    out = row.get("output")
    if not isinstance(inp, dict):
        return None, {"trace_id": trace_id, "reason": "input_not_dict_structured"}
    if not isinstance(out, str):
        return None, {"trace_id": trace_id, "reason": "output_not_string"}

    normalized_input: Dict[str, Any] = {}
    for key in REQUIRED_INPUT_KEYS:
        if key not in inp:
            return None, {"trace_id": trace_id, "reason": f"missing_input_key:{key}"}
        parsed, err = _json_parse_value(inp[key])
        if err is not None:
            return None, {"trace_id": trace_id, "reason": f"{key}:{err}"}
        normalized_input[key] = parsed

    candidate = {
        "trace_id": trace_id,
        "created": row.get("created"),
        "window_hours": row.get("window_hours"),
        "input": normalized_input,
        "output_raw": out,
        "metadata": row.get("metadata") if isinstance(row.get("metadata"), dict) else {},
        "span_attributes": row.get("span_attributes")
        if isinstance(row.get("span_attributes"), dict)
        else {},
    }
    return candidate, None


def normalize_traces(raw_rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    candidates: List[Dict[str, Any]] = []
    rejects: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for row in raw_rows:
        if not isinstance(row, dict):
            rejects.append({"trace_id": "", "reason": "row_not_object"})
            continue
        candidate, reject = normalize_trace_row(row)
        if reject is not None:
            rejects.append(reject)
            continue
        assert candidate is not None
        if candidate["trace_id"] in seen:
            continue
        seen.add(candidate["trace_id"])
        candidates.append(candidate)

    return candidates, rejects


def extract_opportunities(candidate_input: Dict[str, Any]) -> List[Dict[str, Any]]:
    oc = candidate_input.get("opportunity_context")
    if not isinstance(oc, dict):
        return []
    opps = oc.get("opportunities")
    if isinstance(opps, list):
        return [o for o in opps if isinstance(o, dict)]
    one = oc.get("opportunity")
    if isinstance(one, dict):
        return [one]
    return []


def extract_tasks(candidate_input: Dict[str, Any]) -> Dict[str, Any]:
    oc = candidate_input.get("opportunity_context")
    if isinstance(oc, dict) and isinstance(oc.get("tasks"), dict):
        return oc["tasks"]
    return {}


def extract_coverage(candidate_input: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    cc = candidate_input.get("contact_coverage")
    contacts = candidate_input.get("contacts")

    total_buyers: Optional[int] = None
    connected_buyers: Optional[int] = None

    if isinstance(contacts, dict):
        tb = contacts.get("total_buyers")
        if isinstance(tb, (int, float)):
            total_buyers = int(tb)

    if isinstance(cc, dict):
        cb = cc.get("connected_buyer_count")
        if isinstance(cb, (int, float)):
            connected_buyers = int(cb)
        if total_buyers is None:
            tb2 = cc.get("total_buyers")
            if isinstance(tb2, (int, float)):
                total_buyers = int(tb2)

    return total_buyers, connected_buyers


def _lower(s: Any) -> str:
    return s.lower() if isinstance(s, str) else ""


def _words(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9_']+", text.lower())


def _token_set(text: str) -> set[str]:
    return {w for w in _words(text) if len(w) > 3 and w not in STOPWORDS}


def _contains_any(text: str, phrases: Iterable[str]) -> bool:
    t = text.lower()
    return any(p.lower() in t for p in phrases)


def _risk_items(opps: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for opp in opps:
        risks = opp.get("risks")
        if isinstance(risks, list):
            for r in risks:
                if isinstance(r, str) and r.strip():
                    out.append(r.strip())
                elif isinstance(r, dict):
                    txt = r.get("text") or r.get("risk") or r.get("description")
                    if isinstance(txt, str) and txt.strip():
                        out.append(txt.strip())
    return out


def _stage_of_opp(opp: Dict[str, Any]) -> str:
    stage = opp.get("stage")
    if isinstance(stage, str) and stage.strip():
        return stage.strip().lower()
    stage_human = opp.get("stage_human")
    if isinstance(stage_human, str) and stage_human.strip():
        return stage_human.strip().lower()
    return "unknown"


def _is_closed_opp(opp: Dict[str, Any]) -> bool:
    if opp.get("actual_close_date"):
        return True
    stage = _stage_of_opp(opp)
    closed_tokens = ["closed_won", "closed_lost", "closed-won", "closed-lost", "won", "lost"]
    return any(tok in stage for tok in closed_tokens)


def _has_recent_signal(opps: List[Dict[str, Any]]) -> bool:
    recent_phrases = [
        "today",
        "yesterday",
        "spoke",
        "spoke yesterday",
        "met",
        "meeting",
        "call",
        "emailed",
        "replied",
        "followed up",
    ]
    for opp in opps:
        s = f"{opp.get('summary', '')}\n{opp.get('notes', '')}".lower()
        if _contains_any(s, recent_phrases):
            return True
    return False


def _collect_output_text(out_obj: Dict[str, Any]) -> str:
    parts: List[str] = []
    summary = out_obj.get("summary")
    if isinstance(summary, str):
        parts.append(summary)
    highlights = out_obj.get("highlights")
    if isinstance(highlights, list):
        parts.extend([h for h in highlights if isinstance(h, str)])
    recommendation = out_obj.get("recommendation")
    if isinstance(recommendation, list):
        parts.extend([r for r in recommendation if isinstance(r, str)])
    ar = out_obj.get("action_reasoning")
    if isinstance(ar, str):
        parts.append(ar)
    return "\n".join(parts)


def _urgency_bucket(opps: List[Dict[str, Any]]) -> str:
    vals: List[float] = []
    for opp in opps:
        dtc = opp.get("days_to_close")
        if isinstance(dtc, (int, float)):
            vals.append(float(dtc))
    if not vals:
        return "no_close_date"
    m = min(vals)
    if m < 0:
        return "overdue"
    if m < 7:
        return "imminent"
    if m <= 30:
        return "approaching"
    return "comfortable"


def _tasks_bucket(tasks: Dict[str, Any]) -> str:
    total_overdue = tasks.get("total_overdue")
    total_open = tasks.get("total_open")
    if isinstance(total_overdue, (int, float)) and total_overdue > 0:
        return "overdue_tasks"
    if isinstance(total_open, (int, float)) and total_open > 0:
        return "open_tasks"
    return "no_tasks"


def _coverage_bucket(total_buyers: Optional[int], connected_buyers: Optional[int]) -> str:
    if total_buyers is None:
        return "coverage_unknown"
    if total_buyers == 0:
        return "no_buyers_identified"
    if connected_buyers is None:
        return "coverage_unknown"
    if connected_buyers == 0:
        return "no_buyers_connected"
    if total_buyers > 1 and connected_buyers == 1:
        return "single_threaded"
    return "multi_threaded"


def _opp_shape(opps: List[Dict[str, Any]]) -> str:
    if len(opps) <= 1:
        return "single_opp"
    return "multi_opp"


@dataclass
class JudgeResult:
    judge_outcome: str
    model_critique: str
    critical_violations: List[str]
    warnings: List[str]
    output_schema_valid: bool
    diversity_bucket: str
    primary_failure: str


def judge_candidate(candidate: Dict[str, Any]) -> JudgeResult:
    inp = candidate["input"]
    opps = extract_opportunities(inp)
    tasks = extract_tasks(inp)
    total_buyers, connected_buyers = extract_coverage(inp)

    critical: List[str] = []
    warnings: List[str] = []
    critique_lines: List[str] = []

    out_raw = candidate.get("output_raw")
    out_obj: Optional[Dict[str, Any]] = None

    if not isinstance(out_raw, str):
        critical.append("output_not_string")
    else:
        try:
            parsed = json.loads(out_raw)
            if isinstance(parsed, dict):
                out_obj = parsed
            else:
                critical.append("output_not_json_object")
        except Exception:
            critical.append("output_invalid_json")

    schema_valid = True
    if out_obj is None:
        schema_valid = False
    else:
        summary = out_obj.get("summary")
        highlights = out_obj.get("highlights")
        recommendation = out_obj.get("recommendation", None)
        action_reasoning = out_obj.get("action_reasoning")

        if not isinstance(summary, str) or not summary.strip():
            critical.append("missing_or_invalid_summary")
            schema_valid = False
        if not isinstance(highlights, list) or not all(isinstance(h, str) for h in highlights):
            critical.append("missing_or_invalid_highlights")
            schema_valid = False
        if not isinstance(action_reasoning, str) or not action_reasoning.strip():
            critical.append("missing_or_invalid_action_reasoning")
            schema_valid = False
        if recommendation is not None and not (
            isinstance(recommendation, list) and all(isinstance(x, str) for x in recommendation)
        ):
            critical.append("invalid_recommendation_type")
            schema_valid = False

    all_text = _collect_output_text(out_obj or {})
    all_text_lower = all_text.lower()

    if "—" in all_text:
        critical.append("em_dash_present")

    all_closed = bool(opps) and all(_is_closed_opp(o) for o in opps)
    any_closed = any(_is_closed_opp(o) for o in opps)

    if all_closed and out_obj is not None:
        if _contains_any(all_text_lower, ["at risk", "needs attention", "needs re-engagement", "inactive"]):
            critical.append("closed_deal_misclassified_as_risky")
        summary_l = _lower(out_obj.get("summary"))
        if not _contains_any(summary_l, ["closed", "won", "lost"]):
            critical.append("closed_summary_missing_outcome")
        rec = out_obj.get("recommendation")
        rec_text = "\n".join(rec) if isinstance(rec, list) else ""
        if "review other active opportunit" not in rec_text.lower():
            critical.append("closed_recommendation_not_followed")

    if any_closed and not all_closed and out_obj is not None:
        warnings.append("mixed_closed_and_open_opportunities")

    if not all_closed and out_obj is not None:
        summary_txt = out_obj.get("summary") if isinstance(out_obj.get("summary"), str) else ""
        highlights = out_obj.get("highlights") if isinstance(out_obj.get("highlights"), list) else []
        recommendation = out_obj.get("recommendation") if isinstance(out_obj.get("recommendation"), list) else []

        if len(_words(summary_txt)) > 50:
            critical.append("open_summary_too_long")

        if len(highlights) < 2 or len(highlights) > 6:
            critical.append("open_highlights_count_out_of_range")

        if len(recommendation) < 1 or len(recommendation) > 5:
            critical.append("open_recommendation_count_out_of_range")

        total_open = tasks.get("total_open")
        total_overdue = tasks.get("total_overdue")
        titles: List[str] = []
        open_tasks = tasks.get("open_tasks")
        if isinstance(open_tasks, list):
            for t in open_tasks:
                if isinstance(t, dict) and isinstance(t.get("title"), str):
                    titles.append(t["title"])

        has_task_signal = _contains_any(all_text_lower, ["task", "overdue", "due", "follow up", "follow-up"])
        if not has_task_signal:
            for title in titles:
                tok = _token_set(title)
                if tok and len(tok.intersection(_token_set(all_text_lower))) >= 1:
                    has_task_signal = True
                    break

        if isinstance(total_open, (int, float)) and total_open > 0 and not has_task_signal:
            critical.append("tasks_not_surfaced")
        if isinstance(total_overdue, (int, float)) and total_overdue > 0 and "overdue" not in all_text_lower:
            critical.append("overdue_tasks_not_prioritized")

        if total_buyers == 0:
            if _contains_any(all_text_lower, ["no buyers connected", "single-threaded"]):
                critical.append("coverage_guardrail_violated_total_buyers_zero")
            if "no buyer contacts identified" not in all_text_lower:
                critical.append("no_buyer_contacts_identified_missing")
        elif total_buyers is not None and total_buyers > 0 and connected_buyers is not None:
            if connected_buyers == 0 and "no buyers connected" not in all_text_lower:
                critical.append("no_buyers_connected_flag_missing")
            if total_buyers > 1 and connected_buyers == 1 and "single-threaded" not in all_text_lower:
                critical.append("single_threaded_flag_missing")

        risks = _risk_items(opps)
        if not risks and re.search(r"\brisks?\b", all_text_lower):
            critical.append("invented_risk_without_source")
        if risks:
            risk_toks = set()
            for r in risks:
                risk_toks |= _token_set(r)
            if risk_toks and len(risk_toks.intersection(_token_set(all_text_lower))) == 0:
                critical.append("source_risks_not_reflected")

        days_values = [o.get("days_in_stage") for o in opps if isinstance(o.get("days_in_stage"), (int, float))]
        high_days = any(float(v) >= 15 for v in days_values)
        if high_days and _has_recent_signal(opps) and _contains_any(all_text_lower, ["at risk", "inactive"]):
            critical.append("notes_vs_days_conflict")

        source_summaries = [str(o.get("summary") or "") for o in opps if str(o.get("summary") or "").strip()]
        if source_summaries:
            src = " ".join(source_summaries)
            src_tokens = _token_set(src)
            out_tokens = _token_set(summary_txt)
            generic_patterns = [
                "this opportunity is in",
                "deal is in",
                "deal is progressing",
                "active deal",
                "opportunity is in",
            ]
            if _contains_any(summary_txt.lower(), generic_patterns) and len(src_tokens.intersection(out_tokens)) < 2:
                critical.append("summary_missing_source_context")

    # critique assembly
    critique_lines.append("Accuracy: " + ("FAIL" if any(v in critical for v in [
        "closed_deal_misclassified_as_risky",
        "invented_risk_without_source",
        "source_risks_not_reflected",
        "notes_vs_days_conflict",
        "summary_missing_source_context",
    ]) else "PASS"))
    critique_lines.append("Structure Compliance: " + ("FAIL" if any(v in critical for v in [
        "output_not_string",
        "output_invalid_json",
        "output_not_json_object",
        "missing_or_invalid_summary",
        "missing_or_invalid_highlights",
        "missing_or_invalid_action_reasoning",
        "invalid_recommendation_type",
        "open_summary_too_long",
        "open_highlights_count_out_of_range",
        "open_recommendation_count_out_of_range",
        "em_dash_present",
    ]) else "PASS"))
    critique_lines.append("Rule Compliance: " + ("FAIL" if any(v in critical for v in [
        "closed_recommendation_not_followed",
        "tasks_not_surfaced",
        "overdue_tasks_not_prioritized",
        "coverage_guardrail_violated_total_buyers_zero",
        "no_buyer_contacts_identified_missing",
        "no_buyers_connected_flag_missing",
        "single_threaded_flag_missing",
    ]) else "PASS"))

    if critical:
        critique_lines.append("Critical violations: " + ", ".join(sorted(set(critical))))
    else:
        critique_lines.append("No critical violations detected.")

    if warnings:
        critique_lines.append("Minor notes: " + ", ".join(sorted(set(warnings))))

    outcome = "BAD" if critical else "GOOD"
    diversity_bucket = "|".join(
        [
            "closed" if all_closed else ("mixed" if any_closed else "open"),
            _urgency_bucket(opps),
            _tasks_bucket(tasks),
            _coverage_bucket(total_buyers, connected_buyers),
            _opp_shape(opps),
        ]
    )

    primary_failure = "none"
    if critical:
        priority = [
            "closed_deal_misclassified_as_risky",
            "notes_vs_days_conflict",
            "invented_risk_without_source",
            "source_risks_not_reflected",
            "tasks_not_surfaced",
            "overdue_tasks_not_prioritized",
            "coverage_guardrail_violated_total_buyers_zero",
            "no_buyer_contacts_identified_missing",
            "no_buyers_connected_flag_missing",
            "single_threaded_flag_missing",
            "output_invalid_json",
            "output_not_json_object",
            "missing_or_invalid_summary",
            "missing_or_invalid_highlights",
            "missing_or_invalid_action_reasoning",
            "open_summary_too_long",
            "open_highlights_count_out_of_range",
            "open_recommendation_count_out_of_range",
            "summary_missing_source_context",
        ]
        for p in priority:
            if p in critical:
                primary_failure = p
                break
        if primary_failure == "none":
            primary_failure = critical[0]

    return JudgeResult(
        judge_outcome=outcome,
        model_critique=" ".join(critique_lines),
        critical_violations=sorted(set(critical)),
        warnings=sorted(set(warnings)),
        output_schema_valid=schema_valid,
        diversity_bucket=diversity_bucket,
        primary_failure=primary_failure,
    )


def label_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for c in candidates:
        jr = judge_candidate(c)
        row = {
            "trace_id": c["trace_id"],
            "created": c.get("created"),
            "window_hours": c.get("window_hours"),
            "input": c["input"],
            "output_raw": c["output_raw"],
            "judge_outcome": jr.judge_outcome,
            "model_critique": jr.model_critique,
            "critical_violations": jr.critical_violations,
            "warnings": jr.warnings,
            "output_schema_valid": jr.output_schema_valid,
            "diversity_bucket": jr.diversity_bucket,
            "primary_failure": jr.primary_failure,
        }
        out.append(row)
    return out


def _candidate_signal_score(row: Dict[str, Any]) -> float:
    inp = row.get("input") if isinstance(row.get("input"), dict) else {}
    opps = extract_opportunities(inp)
    tasks = extract_tasks(inp)
    total_buyers, connected_buyers = extract_coverage(inp)
    score = 0.0
    if opps:
        score += 1.5
    if len(opps) > 1:
        score += 1.0
    if _urgency_bucket(opps) in {"overdue", "imminent"}:
        score += 1.0
    if _tasks_bucket(tasks) == "overdue_tasks":
        score += 1.0
    if _tasks_bucket(tasks) == "open_tasks":
        score += 0.5
    cov = _coverage_bucket(total_buyers, connected_buyers)
    if cov in {"no_buyers_identified", "no_buyers_connected", "single_threaded"}:
        score += 0.8
    src_text = "\n".join(str(o.get("summary") or "") + "\n" + str(o.get("notes") or "") for o in opps)
    if len(src_text.strip()) > 40:
        score += 0.8
    return score


def assess_worthiness(row: Dict[str, Any]) -> Tuple[bool, List[str], List[str], float]:
    worthy_reasons: List[str] = []
    reject_reasons: List[str] = []

    if not isinstance(row.get("input"), dict):
        reject_reasons.append("input_not_structured_dict")
    else:
        worthy_reasons.append("input_structured_and_parseable")

    if row.get("output_schema_valid") is True:
        worthy_reasons.append("output_valid_json_schema")
    else:
        reject_reasons.append("output_schema_invalid")

    signal_score = _candidate_signal_score(row)
    if signal_score >= 2.0:
        worthy_reasons.append("material_eval_signal")
    else:
        reject_reasons.append("low_signal_or_noise")

    if isinstance(row.get("diversity_bucket"), str) and row["diversity_bucket"].strip():
        worthy_reasons.append("has_diversity_bucket")
    else:
        reject_reasons.append("missing_diversity_features")

    # Judge output is intentionally not used to determine worthiness.
    # Worthiness remains prompt/data-quality driven so judge stays independent.
    worthy_reasons.append("worthiness_independent_of_judge")

    # Require non-ambiguous source context for selected set.
    inp = row.get("input") if isinstance(row.get("input"), dict) else {}
    opps = extract_opportunities(inp)
    if not opps:
        reject_reasons.append("ambiguous_no_opportunities")
    else:
        src_text = "\n".join(str(o.get("summary") or "") + "\n" + str(o.get("notes") or "") for o in opps)
        if len(src_text.strip()) >= 20:
            worthy_reasons.append("source_context_present")
        else:
            reject_reasons.append("source_context_too_sparse")

    is_worthy = len(reject_reasons) == 0
    quality_score = signal_score + (0.8 if row.get("output_schema_valid") is True else 0.0)
    return is_worthy, worthy_reasons, reject_reasons, quality_score


def select_with_diversity(rows: List[Dict[str, Any]], target: int, outcome: str) -> List[Dict[str, Any]]:
    pool = [r for r in rows if r.get("judge_outcome") == outcome and r.get("is_worthy") is True]
    selected: List[Dict[str, Any]] = []
    selected_ids: set[str] = set()
    bucket_counts: Counter[str] = Counter()

    while len(selected) < target:
        best_idx = -1
        best_score = -10_000.0

        for i, row in enumerate(pool):
            tid = row.get("trace_id")
            if not isinstance(tid, str) or tid in selected_ids:
                continue
            score = float(row.get("quality_score", 0.0))
            bucket = str(row.get("diversity_bucket") or "unknown")
            score += 2.5 if bucket_counts[bucket] == 0 else (0.8 if bucket_counts[bucket] == 1 else -0.6)

            summary = ""
            inp = row.get("input")
            if isinstance(inp, dict):
                opps = extract_opportunities(inp)
                if opps:
                    summary = str(opps[0].get("summary") or "")
            if len(summary) < 20:
                score -= 0.5

            if score > best_score:
                best_score = score
                best_idx = i

        if best_idx < 0:
            break

        pick = pool[best_idx]
        tid = str(pick["trace_id"])
        selected.append(pick)
        selected_ids.add(tid)
        bucket_counts[str(pick.get("diversity_bucket") or "unknown")] += 1

    return selected


def curate_golden(
    judged_rows: List[Dict[str, Any]],
    target_good: int,
    target_bad: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    reviewed: List[Dict[str, Any]] = []

    for row in judged_rows:
        is_worthy, worthy_reasons, reject_reasons, quality_score = assess_worthiness(row)
        row["is_worthy"] = is_worthy
        row["worthy_reasons"] = worthy_reasons
        row["reject_reasons"] = reject_reasons
        row["quality_score"] = quality_score

    selected_good = select_with_diversity(judged_rows, target_good, "GOOD")
    selected_bad = select_with_diversity(judged_rows, target_bad, "BAD")

    selected_ids = {str(r["trace_id"]) for r in selected_good + selected_bad}

    # Build curation ledger for every reviewed row.
    for row in judged_rows:
        tid = str(row.get("trace_id") or "")
        selected = tid in selected_ids
        reject_reasons = [] if selected else list(row.get("reject_reasons") or [])
        if not selected and row.get("is_worthy") is True:
            reject_reasons.append("not_selected_due_to_quota_or_diversity")

        review_notes = []
        if selected:
            review_notes.append("Selected after per-trace worthiness review")
            review_notes.append(f"Diversity bucket: {row.get('diversity_bucket')}")
            if row.get("judge_outcome") == "BAD":
                review_notes.append(f"Primary failure mode: {row.get('primary_failure')}")
            review_notes.append(f"Quality score: {row.get('quality_score')}")
        else:
            review_notes.append("Rejected during per-trace worthiness/diversity review")
            review_notes.append(f"Quality score: {row.get('quality_score')}")

        reviewed.append(
            {
                "trace_id": tid,
                "window": row.get("window_hours"),
                "judge_outcome": row.get("judge_outcome"),
                "worthy_decision": "selected" if selected else "rejected",
                "worthy_reasons": row.get("worthy_reasons") or [],
                "reject_reasons": reject_reasons,
                "review_notes": "; ".join(review_notes),
            }
        )

    final_rows = selected_good + selected_bad
    # Stable sort for reproducibility.
    final_rows.sort(key=lambda r: (r.get("judge_outcome") != "GOOD", str(r.get("created") or "")))

    summary = {
        "reviewed_total": len(judged_rows),
        "worthy_total": sum(1 for r in judged_rows if r.get("is_worthy") is True),
        "selected_total": len(final_rows),
        "selected_good": sum(1 for r in final_rows if r.get("judge_outcome") == "GOOD"),
        "selected_bad": sum(1 for r in final_rows if r.get("judge_outcome") == "BAD"),
        "diversity_buckets_selected": sorted({str(r.get("diversity_bucket") or "") for r in final_rows}),
        "primary_failures_selected": Counter(
            str(r.get("primary_failure") or "none")
            for r in final_rows
            if r.get("judge_outcome") == "BAD"
        ),
    }

    return final_rows, reviewed, summary


def to_golden_rows(selected_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in selected_rows:
        out.append(
            {
                "trace_id": row.get("trace_id"),
                "judge_outcome": row.get("judge_outcome"),
                "model_critique": row.get("model_critique"),
                "input": row.get("input"),
                "output_raw": row.get("output_raw"),
            }
        )
    return out


def to_renamed_rows(golden_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in golden_rows:
        out.append(
            {
                "trace_id": row.get("trace_id"),
                "judge_outcome": row.get("judge_outcome"),
                "model_critique": row.get("model_critique"),
                "output_raw": row.get("output_raw"),
                "context": row.get("input"),
            }
        )
    return out


def to_labels_only_rows(golden_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in golden_rows:
        out.append(
            {
                "trace_id": row.get("trace_id"),
                "judge_outcome": row.get("judge_outcome"),
                "model_critique": row.get("model_critique"),
            }
        )
    return out
