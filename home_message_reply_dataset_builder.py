#!/usr/bin/env python3
"""
Build a sanitized + ranked dataset from `home_message_reply_traces.json`.

Primary goals:
1) Redact *Monaco-domain* email addresses (e.g., *@monaco.com, *@monaco.co, *@monaco.ai)
2) Heuristically score traces for "Founder-led sales email reply" usefulness
3) Emit JSON/JSONL artifacts for review and eval-set curation

This is intentionally dependency-free (stdlib only).
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


EMAIL_RE = re.compile(r"\b[A-Z0-9._%+-]+@([A-Z0-9.-]+\.[A-Z]{2,})\b", re.IGNORECASE)
EMAIL_FULL_RE = re.compile(r"\b([A-Z0-9._%+-]+)@([A-Z0-9.-]+\.[A-Z]{2,})\b", re.IGNORECASE)
CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")

EMAIL_BLOCK_RE = re.compile(
    r"(Email #\d+ \| (INBOUND|OUTBOUND) \|[^\n]*\n)(.*?)(?=Email #\d+ \||\Z)",
    re.DOTALL,
)

BANNED_PHRASES = [
    "i hope this finds you well",
    "kindly,",
    "please do not hesitate",
    "checking in",
    "touching base",
]

UNVERIFIED_MEETING_PHRASES = [
    "as we discussed",
    "as discussed",
    "as we spoke",
    "as we talked",
    "as mentioned on our call",
    "as mentioned on the call",
    "great speaking",
    "great talking",
    "good talking",
    "good speaking",
    "as per our conversation",
]

SALES_KEYWORDS = [
    "demo",
    "pricing",
    "quote",
    "proposal",
    "pilot",
    "trial",
    "security review",
    "soc 2",
    "msa",
    "nda",
    "next steps",
    "timeline",
    "implementation",
    "procurement",
    "intro",
    "introduction",
    "rfp",
    "poc",
    "proof of concept",
    "customer",
    "use case",
    "pipeline",
    "opportunity",
    "potential fit",
    "fit",
    "connect",
    "chat",
    "quick call",
    "call",
    "meeting",
    "catch up",
    "partner",
    "partnership",
]

AUTOMATION_HINTS = [
    "noreply",
    "no-reply",
    "do-not-reply",
    "notifications@github.com",
    "calendar",
    "invite",
    "[bot]",
    "document shared with you",
    "you are receiving this",
    "unsubscribe",
    "password reset",
    "verify your",
]

NON_SALES_HINTS = [
    "pull request",
    "monacoinc/monaco",
    "issue_event",
    "commit summary",
    "merge branch",
    "document shared with you",
    "drive-shares",
    "o-1",
    "o1",
    "o-1a",
    "rfe",
    "immigration",
    "attorney",
    "visa",
    "airbnb",
    "printer",
    "packages arriving",
    "grab a bite",
    "assignment",
    "payment request",
    "invoice",
    "coi",
    "certificate of insurance",
    "insurance",
    "w9",
    "accounts payable",
    "vendor",
    "hotel",
    "hyatt",
    "reservation",
    "itinerary",
    "application",
    "candidate",
    "resume",
    "job",
    "role",
    "hiring",
    "recruit",
    "talent",
    "weekly analytics report",
    "analytics report",
]

GOLDEN_EXCLUDE_SUBJECT_HINTS = [
    "report",
    "proof",
    "layout",
    "project #",
    "magazine",
    "event",
    "newsletter",
    "meeting summary",
    "weekly checkin",
    "checkin meeting summary",
    "industry report",
    "graphics",
]

def _extract_emails(text: str) -> List[str]:
    if not text:
        return []
    return [f"{local}@{domain}" for local, domain in EMAIL_FULL_RE.findall(text)]


def _has_control_chars(text: str) -> bool:
    return bool(text and CONTROL_CHAR_RE.search(text))


def _is_non_sales_thread(message_thread: str) -> bool:
    lower = (message_thread or "").lower()
    return any(h in lower for h in NON_SALES_HINTS)


def _is_monaco_domain(domain: str) -> bool:
    labels = [p for p in domain.lower().split(".") if p]
    if not labels:
        return False
    if "monaco" in labels:
        return True
    if "monacoinc" in labels:
        return True
    return False


@dataclass
class RedactionStats:
    total_replacements: int = 0
    unique_monaco_emails: int = 0
    traces_with_monaco_emails: int = 0


class EmailRedactor:
    def __init__(self) -> None:
        self._map: Dict[str, str] = {}
        self._counter = 0

    def replacement_for(self, email: str) -> str:
        if email in self._map:
            return self._map[email]
        self._counter += 1
        # Deterministic-ish but opaque: user_<n>+<hash>@example.com
        digest = hashlib.sha1(email.encode("utf-8")).hexdigest()[:8]
        repl = f"monaco_user_{self._counter}_{digest}@example.com"
        self._map[email] = repl
        return repl

    @property
    def unique_emails(self) -> int:
        return len(self._map)


def _redact_monaco_emails_in_text(text: str, redactor: EmailRedactor) -> Tuple[str, int]:
    replacements = 0

    def _sub(m: re.Match[str]) -> str:
        nonlocal replacements
        local, domain = m.group(1), m.group(2)
        if not _is_monaco_domain(domain):
            return m.group(0)
        replacements += 1
        return redactor.replacement_for(f"{local}@{domain}")

    return EMAIL_FULL_RE.sub(_sub, text), replacements


def _walk_and_redact(obj: Any, redactor: EmailRedactor) -> Tuple[Any, int]:
    if obj is None:
        return obj, 0
    if isinstance(obj, str):
        return _redact_monaco_emails_in_text(obj, redactor)
    if isinstance(obj, list):
        out: List[Any] = []
        n = 0
        for item in obj:
            item2, k = _walk_and_redact(item, redactor)
            out.append(item2)
            n += k
        return out, n
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        n = 0
        for k0, v0 in obj.items():
            v1, k = _walk_and_redact(v0, redactor)
            out[k0] = v1
            n += k
        return out, n
    return obj, 0


@dataclass
class TraceScore:
    score: float
    issues: List[str]
    subject_line: str
    inbound_count: int
    outbound_count: int
    is_automated: bool
    is_sales_like: bool
    output_valid_json: bool


def _safe_lower(s: str) -> str:
    return s.lower() if isinstance(s, str) else ""


def _extract_subject_line(message_thread: str) -> str:
    if not message_thread:
        return ""
    m = re.search(r"(?m)^Subject:\s*(.*)$", message_thread)
    return (m.group(0).strip() if m else "")[:200]

def _extract_latest_inbound_from(message_thread: str) -> str:
    if not message_thread:
        return ""
    last_inbound = None
    for m in EMAIL_BLOCK_RE.finditer(message_thread):
        if (m.group(2) or "").upper() == "INBOUND":
            last_inbound = m
    if not last_inbound:
        return ""
    block = last_inbound.group(3)
    m_from = re.search(r"(?m)^From:\s.*\(([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})\)\s*$", block, re.IGNORECASE)
    if m_from:
        return m_from.group(1)
    m_from2 = re.search(r"(?m)^From:\s*([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})\s*$", block, re.IGNORECASE)
    return m_from2.group(1) if m_from2 else ""


def _analyze_thread(message_thread: str) -> Tuple[int, int, bool, bool, str]:
    inbound = len(re.findall(r"\| INBOUND \|", message_thread or ""))
    outbound = len(re.findall(r"\| OUTBOUND \|", message_thread or ""))
    lower = _safe_lower(message_thread or "")
    latest_inbound_from = _safe_lower(_extract_latest_inbound_from(message_thread))
    # Determine "automated" mostly from the latest inbound sender (thread reality).
    is_automated = (
        any(h in latest_inbound_from for h in ["noreply", "no-reply", "do-not-reply"])
        or "notifications@github.com" in latest_inbound_from
        or any(h in lower for h in AUTOMATION_HINTS)
    )
    is_non_sales = _is_non_sales_thread(message_thread)
    is_sales_like = any(k in lower for k in SALES_KEYWORDS) and not is_non_sales
    subject = _extract_subject_line(message_thread or "")
    return inbound, outbound, is_automated, is_sales_like, subject


def _analyze_output(output_raw: Any) -> Tuple[bool, List[str], Optional[Dict[str, Any]]]:
    issues: List[str] = []
    if not isinstance(output_raw, str):
        return False, ["output_not_string"], None

    try:
        parsed = json.loads(output_raw)
    except Exception:
        return False, ["output_invalid_json"], None

    if not isinstance(parsed, dict):
        return False, ["output_not_object"], None

    if "proposed_response_body" not in parsed or "reasoning" not in parsed:
        issues.append("output_missing_required_keys")
    body = parsed.get("proposed_response_body")
    reasoning = parsed.get("reasoning")
    if not isinstance(body, str) or not isinstance(reasoning, str):
        issues.append("output_keys_not_strings")

    text_to_check = f"{body}\n{reasoning}" if isinstance(body, str) and isinstance(reasoning, str) else output_raw
    lower = _safe_lower(text_to_check)
    if "—" in text_to_check:
        issues.append("uses_em_dash")
    for phrase in BANNED_PHRASES:
        if phrase in lower:
            issues.append(f"banned_phrase:{phrase}")
    if isinstance(body, str) and "<p" not in body:
        issues.append("body_missing_html_p")
    if isinstance(output_raw, str) and _has_control_chars(output_raw):
        issues.append("output_has_control_chars")
    return True, issues, parsed


def _score_trace(trace: Dict[str, Any]) -> TraceScore:
    issues: List[str] = []
    inp = trace.get("input") if isinstance(trace, dict) else None
    if not isinstance(inp, dict):
        return TraceScore(
            score=-10.0,
            issues=["missing_input"],
            subject_line="",
            inbound_count=0,
            outbound_count=0,
            is_automated=True,
            is_sales_like=False,
            output_valid_json=False,
        )

    message_thread = inp.get("message_thread") or inp.get("email_thread") or ""
    inbound, outbound, is_automated, is_sales_like, subject = _analyze_thread(str(message_thread))

    output_valid, output_issues, _parsed = _analyze_output(trace.get("output"))
    issues.extend(output_issues)

    # Detect unverified meeting/call references when there's no evidence in thread/summaries.
    past = inp.get("past_meeting_summaries")
    past_empty = (past in ("", None, []) or past is False)
    mt_lower = str(message_thread).lower() if message_thread is not None else ""
    if past_empty and not any(x in mt_lower for x in ["calendar", "invite", "zoom", "meet", "meeting", "call"]):
        out_str = trace.get("output") if isinstance(trace.get("output"), str) else ""
        out_lower = out_str.lower()
        if any(p in out_lower for p in UNVERIFIED_MEETING_PHRASES):
            issues.append("unverified_meeting_reference")

    # Detect hallucinated email addresses in the response (not present in thread, not current_user_email).
    try:
        parsed = json.loads(trace.get("output")) if isinstance(trace.get("output"), str) else None
    except Exception:
        parsed = None
    if isinstance(parsed, dict):
        body = parsed.get("proposed_response_body") if isinstance(parsed.get("proposed_response_body"), str) else ""
        reasoning = parsed.get("reasoning") if isinstance(parsed.get("reasoning"), str) else ""
        output_emails = set(_extract_emails(body) + _extract_emails(reasoning))
        allowed = set(_extract_emails(str(message_thread)))
        cur_email = inp.get("current_user_email")
        if isinstance(cur_email, str) and "@" in cur_email:
            allowed.add(cur_email)
        unexpected = {e for e in output_emails if e not in allowed}
        if unexpected:
            issues.append("hallucinated_email")

    opportunity = inp.get("current_opportunity_info")
    opportunity_obj: Any = opportunity
    if isinstance(opportunity, str) and opportunity.strip().startswith(("{", "[")):
        try:
            opportunity_obj = json.loads(opportunity)
        except Exception:
            issues.append("opportunity_string_not_json")
    has_opportunity = isinstance(opportunity_obj, dict) and bool(opportunity_obj)
    if opportunity not in ("", None) and not isinstance(opportunity_obj, dict):
        issues.append("opportunity_not_object_or_empty")

    company_info = inp.get("company_info")
    has_company = isinstance(company_info, str) and company_info.strip() != ""

    has_past = bool(past) and past != ""

    # Heuristic scoring (higher is better for eval utility).
    score = 0.0
    if inbound >= 1:
        score += 2.0
    if outbound >= 1:
        score += 1.0
    if inbound >= 2:
        score += 1.0
    if has_opportunity:
        score += 1.0
    if has_company:
        score += 0.5
    if has_past:
        score += 0.5
    if is_sales_like:
        score += 2.0
    if is_automated:
        score -= 2.0
        issues.append("automated_or_system_thread")
    if not output_valid:
        score -= 2.0
    if "uses_em_dash" in issues:
        score -= 1.0
    if any(i.startswith("banned_phrase:") for i in issues):
        score -= 1.0
    if "body_missing_html_p" in issues:
        score -= 0.5

    if not isinstance(message_thread, str) or message_thread.strip() == "":
        score -= 5.0
        issues.append("missing_message_thread")

    return TraceScore(
        score=score,
        issues=issues,
        subject_line=subject,
        inbound_count=inbound,
        outbound_count=outbound,
        is_automated=is_automated,
        is_sales_like=is_sales_like,
        output_valid_json=output_valid,
    )


def _trace_id(trace: Dict[str, Any]) -> str:
    return str(trace.get("id") or trace.get("root_span_id") or "")


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to traces JSON (list)")
    ap.add_argument("--out-sanitized-traces", required=True, help="Write sanitized traces JSON here")
    ap.add_argument("--out-ranked", required=True, help="Write ranked trace metadata JSONL here")
    ap.add_argument("--out-review-jsonl", required=True, help="Write review candidates JSONL here")
    ap.add_argument("--out-valid-jsonl", required=True, help="Write valid candidates JSONL here")
    ap.add_argument(
        "--out-sales-valid-jsonl",
        default="",
        help="Optional. Write sales-focused valid candidates JSONL here",
    )
    ap.add_argument(
        "--out-golden-btshape-jsonl",
        default="",
        help="Optional. Write a golden-candidate dataset in BT dataset shape (expected labels left blank).",
    )
    ap.add_argument("--golden-size", type=int, default=30)
    ap.add_argument(
        "--golden-good-only",
        action="store_true",
        help="When writing a golden file, only include traces that look GOOD (no obvious rule violations).",
    )
    ap.add_argument(
        "--drop-monaco-traces",
        action="store_true",
        help="Drop traces entirely if they contain any Monaco-domain email address.",
    )
    ap.add_argument(
        "--drop-monaco-current-user-only",
        action="store_true",
        help="Drop traces entirely if input.current_user_email is a Monaco-domain address (other Monaco emails are redacted).",
    )
    ap.add_argument("--top-review", type=int, default=80)
    ap.add_argument("--top-valid", type=int, default=30)
    ap.add_argument("--top-sales-valid", type=int, default=30)
    args = ap.parse_args()

    traces_path = Path(args.input)
    traces = json.loads(traces_path.read_text())
    if not isinstance(traces, list):
        raise SystemExit("Input JSON must be a list of traces")

    redactor = EmailRedactor()
    sanitized: List[Dict[str, Any]] = []
    stats = RedactionStats()
    dropped_monaco_traces = 0

    ranked_rows: List[Dict[str, Any]] = []
    review_rows: List[Dict[str, Any]] = []
    valid_rows: List[Dict[str, Any]] = []
    sales_valid_rows: List[Dict[str, Any]] = []
    golden_rows: List[Dict[str, Any]] = []

    def _trace_has_monaco_email(v: Any) -> bool:
        stack = [v]
        while stack:
            cur = stack.pop()
            if cur is None:
                continue
            if isinstance(cur, str):
                for _local, domain in EMAIL_FULL_RE.findall(cur):
                    if _is_monaco_domain(domain):
                        return True
            elif isinstance(cur, dict):
                stack.extend(cur.values())
            elif isinstance(cur, list):
                stack.extend(cur)
        return False

    for idx, trace in enumerate(traces):
        if not isinstance(trace, dict):
            continue
        if args.drop_monaco_traces and _trace_has_monaco_email(trace):
            dropped_monaco_traces += 1
            continue
        if args.drop_monaco_current_user_only:
            inp0 = trace.get("input") if isinstance(trace.get("input"), dict) else {}
            cue = inp0.get("current_user_email") if isinstance(inp0, dict) else None
            if isinstance(cue, str) and "@" in cue and _is_monaco_domain(cue.split("@", 1)[1]):
                dropped_monaco_traces += 1
                continue
        trace2, replacements = _walk_and_redact(trace, redactor)
        assert isinstance(trace2, dict)
        sanitized.append(trace2)
        if replacements > 0:
            stats.traces_with_monaco_emails += 1
            stats.total_replacements += replacements

        ts = _score_trace(trace2)
        inp0 = trace2.get("input") if isinstance(trace2.get("input"), dict) else {}
        message_thread0 = (inp0.get("message_thread") or inp0.get("email_thread") or "") if isinstance(inp0, dict) else ""
        latest_inbound_from = _extract_latest_inbound_from(str(message_thread0))
        is_non_sales = _is_non_sales_thread(str(message_thread0))
        inbound_external = None
        cur_email = inp0.get("current_user_email") if isinstance(inp0, dict) else None
        if isinstance(cur_email, str) and "@" in cur_email and latest_inbound_from and "@" in latest_inbound_from:
            inbound_external = (cur_email.split("@", 1)[1].lower() != latest_inbound_from.split("@", 1)[1].lower())

        ranked_rows.append(
            {
                "idx": len(sanitized) - 1,
                "id": trace2.get("id"),
                "root_span_id": trace2.get("root_span_id"),
                "score": ts.score,
                "issues": ts.issues,
                "inbound_count": ts.inbound_count,
                "outbound_count": ts.outbound_count,
                "is_automated": ts.is_automated,
                "is_sales_like": ts.is_sales_like,
                "is_non_sales": is_non_sales,
                "output_valid_json": ts.output_valid_json,
                "latest_inbound_from": latest_inbound_from,
                "inbound_external": inbound_external,
                "current_user_email": ((trace2.get("input") or {}).get("current_user_email") if isinstance(trace2.get("input"), dict) else None),
                "subject_line": ts.subject_line,
            }
        )

    stats.unique_monaco_emails = redactor.unique_emails

    # Rank descending by score, tie-breaker by idx.
    ranked_sorted = sorted(ranked_rows, key=lambda r: (-float(r.get("score", -999)), int(r.get("idx", 0))))

    by_id: Dict[str, Dict[str, Any]] = {str(r.get("id")): r for r in ranked_sorted if r.get("id")}
    by_idx: Dict[int, Dict[str, Any]] = {int(r.get("idx")): r for r in ranked_sorted}

    def _trace_for_row(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        idx0 = row.get("idx")
        if isinstance(idx0, int) and 0 <= idx0 < len(sanitized):
            return sanitized[idx0]
        return None

    for row in ranked_sorted[: max(args.top_review, args.top_valid)]:
        tr = _trace_for_row(row)
        if not tr:
            continue
        inp = tr.get("input") if isinstance(tr.get("input"), dict) else {}
        message_thread = inp.get("message_thread") or inp.get("email_thread") or ""
        output = tr.get("output")

        if len(review_rows) < args.top_review:
            review_rows.append(
                {
                    "idx": row.get("idx"),
                    "trace_id": tr.get("id"),
                    "score": row.get("score"),
                    "issues": row.get("issues", []),
                    "message_thread": message_thread,
                    "output": output,
                }
            )

        if len(valid_rows) < args.top_valid:
            # "Valid" = no obvious format violations and not automated/system-y
            issues = row.get("issues", []) or []
            if (
                row.get("output_valid_json") is True
                and not row.get("is_automated")
                and "uses_em_dash" not in issues
                and not any(str(i).startswith("banned_phrase:") for i in issues)
                and "body_missing_html_p" not in issues
            ):
                valid_rows.append(
                    {
                        "id": tr.get("id"),
                        "input": inp,
                        "output": output,
                        "metadata": {
                            "score": row.get("score"),
                            "issues": issues,
                        },
                    }
                )

        if args.out_sales_valid_jsonl and len(sales_valid_rows) < args.top_sales_valid:
            issues = row.get("issues", []) or []
            if (
                row.get("output_valid_json") is True
                and not row.get("is_automated")
                and row.get("is_sales_like") is True
                and "uses_em_dash" not in issues
                and not any(str(i).startswith("banned_phrase:") for i in issues)
                and "body_missing_html_p" not in issues
            ):
                sales_valid_rows.append(
                    {
                        "id": tr.get("id"),
                        "input": inp,
                        "output": output,
                        "metadata": {
                            "score": row.get("score"),
                            "issues": issues,
                            "sales_like": True,
                        },
                    }
                )

    if args.out_golden_btshape_jsonl:
        def _golden_core_ok(row: Dict[str, Any]) -> bool:
            return (
                row.get("output_valid_json") is True
                and not row.get("is_automated")
                and not row.get("is_non_sales")
                and (row.get("inbound_count") or 0) >= 1
            )

        def _golden_clean_ok(row: Dict[str, Any]) -> bool:
            issues = row.get("issues", []) or []
            return (
                _golden_core_ok(row)
                and "uses_em_dash" not in issues
                and "output_has_control_chars" not in issues
                and "hallucinated_email" not in issues
                and "unverified_meeting_reference" not in issues
                and not any(str(i).startswith("banned_phrase:") for i in issues)
                and "body_missing_html_p" not in issues
            )

        def _golden_sort_key(r: Dict[str, Any]) -> Tuple[int, int, float, int]:
            return (
                0 if r.get("is_sales_like") else 1,
                0 if r.get("inbound_external") is True else 1,
                -float(r.get("score", -999)),
                int(r.get("idx", 0)),
            )

        def _golden_subject_ok(r: Dict[str, Any]) -> bool:
            subj = (r.get("subject_line") or "").lower()
            return not any(h in subj for h in GOLDEN_EXCLUDE_SUBJECT_HINTS)

        clean_ranked = sorted([r for r in ranked_sorted if _golden_clean_ok(r)], key=_golden_sort_key)
        core_ranked = sorted([r for r in ranked_sorted if _golden_core_ok(r)], key=_golden_sort_key)
        preferred_clean = [r for r in clean_ranked if r.get("is_sales_like") and _golden_subject_ok(r)]
        preferred_core = [r for r in core_ranked if r.get("is_sales_like") and _golden_subject_ok(r)]
        core_subject_ok = [r for r in core_ranked if _golden_subject_ok(r)]

        picked: List[Dict[str, Any]] = []
        picked_idx: set[int] = set()

        # 1) Prefer clean, sales-relevant examples.
        for r in preferred_clean:
            if len(picked) >= args.golden_size:
                break
            idx0 = r.get("idx")
            if isinstance(idx0, int) and idx0 not in picked_idx:
                picked.append(r)
                picked_idx.add(idx0)

        # 2) Fill remaining slots (optionally) with sales-relevant but potentially-violating examples.
        if not args.golden_good_only and len(picked) < args.golden_size:
            for r in preferred_core:
                if len(picked) >= args.golden_size:
                    break
                idx0 = r.get("idx")
                if not isinstance(idx0, int) or idx0 in picked_idx:
                    continue
                picked.append(r)
                picked_idx.add(idx0)

        # 3) Next: fill from any subject-OK core example (even if not sales-like).
        if not args.golden_good_only and len(picked) < args.golden_size:
            for r in core_subject_ok:
                if len(picked) >= args.golden_size:
                    break
                idx0 = r.get("idx")
                if not isinstance(idx0, int) or idx0 in picked_idx:
                    continue
                picked.append(r)
                picked_idx.add(idx0)

        # 4) Last resort: fill from any core-eligible example (even if subject is borderline),
        # to reach the requested count.
        if not args.golden_good_only and len(picked) < args.golden_size:
            for r in core_ranked:
                if len(picked) >= args.golden_size:
                    break
                idx0 = r.get("idx")
                if not isinstance(idx0, int) or idx0 in picked_idx:
                    continue
                picked.append(r)
                picked_idx.add(idx0)

        for row in picked[: args.golden_size]:
            tr = _trace_for_row(row)
            if not tr:
                continue
            inp = tr.get("input") if isinstance(tr.get("input"), dict) else {}
            out_raw = tr.get("output")
            golden_rows.append(
                {
                    "input": {**inp, "output_raw": out_raw},
                    "expected": {"judge_outcome": "", "model_critique": ""},
                    "metadata": {
                        "trace_id": tr.get("id"),
                        "score": row.get("score"),
                        "subject_line": row.get("subject_line"),
                        "issues": row.get("issues", []),
                        "inbound_external": row.get("inbound_external"),
                    },
                    "tags": ["golden_candidate", "no_monaco"],
                }
            )

    out_sanitized = Path(args.out_sanitized_traces)
    out_ranked = Path(args.out_ranked)
    out_review = Path(args.out_review_jsonl)
    out_valid = Path(args.out_valid_jsonl)
    out_sales_valid = Path(args.out_sales_valid_jsonl) if args.out_sales_valid_jsonl else None
    out_golden = Path(args.out_golden_btshape_jsonl) if args.out_golden_btshape_jsonl else None

    _write_json(out_sanitized, sanitized)
    _write_jsonl(out_ranked, ranked_sorted)
    _write_jsonl(out_review, review_rows)
    _write_jsonl(out_valid, valid_rows)
    if out_sales_valid is not None:
        _write_jsonl(out_sales_valid, sales_valid_rows)
    if out_golden is not None:
        _write_jsonl(out_golden, golden_rows)

    summary = {
        "input_traces": len(traces),
        "sanitized_traces": len(sanitized),
        "monaco_email_replacements": stats.total_replacements,
        "unique_monaco_emails_redacted": stats.unique_monaco_emails,
        "traces_with_monaco_emails": stats.traces_with_monaco_emails,
        "dropped_monaco_traces": dropped_monaco_traces,
        "ranked_rows": len(ranked_rows),
        "top_review_written": len(review_rows),
        "top_valid_written": len(valid_rows),
        "top_sales_valid_written": len(sales_valid_rows),
        "golden_written": len(golden_rows),
        "outputs": {
            "sanitized_traces": str(out_sanitized),
            "ranked_jsonl": str(out_ranked),
            "review_jsonl": str(out_review),
            "valid_jsonl": str(out_valid),
            "sales_valid_jsonl": (str(out_sales_valid) if out_sales_valid is not None else ""),
            "golden_btshape_jsonl": (str(out_golden) if out_golden is not None else ""),
        },
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
