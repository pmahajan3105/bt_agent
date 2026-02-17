#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

GOLDEN = Path('/Users/prashant/bt_agent/traces/insight_opp_stage_bg_golden_30.jsonl')
OUT_JSON = Path('/Users/prashant/bt_agent/traces/insight_opp_stage_bg_golden_30_independent_audit.json')
OUT_MD = Path('/Users/prashant/bt_agent/traces/insight_opp_stage_bg_golden_30_independent_audit.md')


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]


def extract_opps(inp: Dict[str, Any]) -> List[Dict[str, Any]]:
    oc = inp.get('opportunity_context') if isinstance(inp.get('opportunity_context'), dict) else {}
    if isinstance(oc.get('opportunities'), list):
        return [x for x in oc['opportunities'] if isinstance(x, dict)]
    if isinstance(oc.get('opportunity'), dict):
        return [oc['opportunity']]
    return []


def is_closed_opp(opp: Dict[str, Any]) -> bool:
    if opp.get('actual_close_date'):
        return True
    stage = str(opp.get('stage') or opp.get('stage_human') or '').lower()
    return any(t in stage for t in ['closed_won', 'closed_lost', 'closed-won', 'closed-lost', 'won', 'lost'])


def risk_items(opps: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    for o in opps:
        rs = o.get('risks')
        if isinstance(rs, list):
            for r in rs:
                if isinstance(r, str) and r.strip():
                    out.append(r.strip())
                elif isinstance(r, dict):
                    t = r.get('text') or r.get('risk') or r.get('description')
                    if isinstance(t, str) and t.strip():
                        out.append(t.strip())
    return out


def token_set(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[A-Za-z0-9_']+", text or '') if len(w) > 3}


def audit_row(row: Dict[str, Any]) -> List[str]:
    violations: List[str] = []
    inp = row.get('input') if isinstance(row.get('input'), dict) else {}

    try:
        out = json.loads(row.get('output_raw') or '')
    except Exception:
        return ['output_invalid_json']

    if not isinstance(out, dict):
        return ['output_not_json_object']

    summary = out.get('summary')
    highlights = out.get('highlights')
    recommendation = out.get('recommendation', None)
    action_reasoning = out.get('action_reasoning')

    if not isinstance(summary, str) or not summary.strip():
        violations.append('missing_summary')
    if not isinstance(highlights, list) or not all(isinstance(h, str) for h in highlights):
        violations.append('invalid_highlights')
    if not isinstance(action_reasoning, str) or not action_reasoning.strip():
        violations.append('missing_action_reasoning')
    if recommendation is not None and not (isinstance(recommendation, list) and all(isinstance(x, str) for x in recommendation)):
        violations.append('invalid_recommendation_type')

    text = '\n'.join(
        [
            summary if isinstance(summary, str) else '',
            *([h for h in highlights if isinstance(h, str)] if isinstance(highlights, list) else []),
            *([r for r in recommendation if isinstance(r, str)] if isinstance(recommendation, list) else []),
            action_reasoning if isinstance(action_reasoning, str) else '',
        ]
    )
    low = text.lower()

    if '—' in text:
        violations.append('em_dash_present')

    opps = extract_opps(inp)
    all_closed = bool(opps) and all(is_closed_opp(o) for o in opps)

    if all_closed:
        if any(x in low for x in ['at risk', 'needs attention', 'needs re-engagement', 'inactive']):
            violations.append('closed_misclassified')
        if not any(x in low for x in ['closed', 'won', 'lost']):
            violations.append('closed_outcome_missing')
        if isinstance(recommendation, list):
            if 'review other active opportunit' not in '\n'.join(recommendation).lower():
                violations.append('closed_recommendation_mismatch')
    else:
        if isinstance(summary, str) and len(re.findall(r"[A-Za-z0-9_']+", summary)) > 50:
            violations.append('summary_too_long')
        if isinstance(highlights, list) and not (2 <= len(highlights) <= 6):
            violations.append('highlights_count_out_of_range')
        if not (isinstance(recommendation, list) and 1 <= len(recommendation) <= 5):
            violations.append('open_recommendation_count_or_type')

        cc = inp.get('contact_coverage') if isinstance(inp.get('contact_coverage'), dict) else {}
        contacts = inp.get('contacts') if isinstance(inp.get('contacts'), dict) else {}
        tb = contacts.get('total_buyers')
        cb = cc.get('connected_buyer_count')
        try:
            tb = int(tb) if tb is not None else None
        except Exception:
            tb = None
        try:
            cb = int(cb) if cb is not None else None
        except Exception:
            cb = None

        if tb == 0:
            if 'no buyers connected' in low or 'single-threaded' in low:
                violations.append('coverage_guardrail_violated_total_buyers_zero')
            if 'no buyer contacts identified' not in low:
                violations.append('no_buyer_contacts_identified_missing')
        elif tb is not None and tb > 0 and cb is not None:
            if cb == 0 and 'no buyers connected' not in low:
                violations.append('no_buyers_connected_flag_missing')
            if tb > 1 and cb == 1 and 'single-threaded' not in low:
                violations.append('single_threaded_flag_missing')

        oc = inp.get('opportunity_context') if isinstance(inp.get('opportunity_context'), dict) else {}
        tasks = oc.get('tasks') if isinstance(oc.get('tasks'), dict) else {}
        total_open = tasks.get('total_open')
        total_overdue = tasks.get('total_overdue')

        if isinstance(total_open, (int, float)) and total_open > 0 and not any(
            x in low for x in ['task', 'overdue', 'due', 'follow up', 'follow-up']
        ):
            violations.append('tasks_not_surfaced')
        if isinstance(total_overdue, (int, float)) and total_overdue > 0 and 'overdue' not in low:
            violations.append('overdue_not_prioritized')

        risks = risk_items(opps)
        if not risks and re.search(r'\brisks?\b', low):
            violations.append('invented_risk_without_source')

        source = ' '.join(str(o.get('summary') or '') for o in opps)
        if source.strip() and isinstance(summary, str):
            if len(token_set(source).intersection(token_set(summary))) < 2 and any(
                g in summary.lower() for g in ['this opportunity is in', 'deal is in', 'deal is progressing', 'active deal']
            ):
                violations.append('summary_missing_source_context')

    return sorted(set(violations))


def main() -> int:
    rows = read_jsonl(GOLDEN)
    audits = []
    label_mismatch = []
    failure_counter = Counter()

    for r in rows:
        tid = r.get('trace_id')
        label = r.get('judge_outcome')
        v = audit_row(r)
        for x in v:
            failure_counter[x] += 1
        if label == 'GOOD' and v:
            consistency = 'FAIL'
            label_mismatch.append((tid, label, v))
        elif label == 'BAD' and not v:
            consistency = 'FAIL'
            label_mismatch.append((tid, label, v))
        else:
            consistency = 'PASS'

        audits.append(
            {
                'trace_id': tid,
                'judge_outcome': label,
                'independent_violations': v,
                'label_consistency': consistency,
            }
        )

    bad_rows = [a for a in audits if a['judge_outcome'] == 'BAD']
    bad_failures = Counter()
    for a in bad_rows:
        if a['independent_violations']:
            bad_failures[a['independent_violations'][0]] += 1

    overfit_flags = []
    if bad_rows:
        top = bad_failures.most_common(1)
        if top and top[0][1] / len(bad_rows) > 0.7:
            overfit_flags.append(f"Single failure mode dominates BAD set: {top[0][0]} ({top[0][1]}/{len(bad_rows)})")
        if len(bad_failures) < 3:
            overfit_flags.append(f"Low BAD diversity: only {len(bad_failures)} primary failure modes")

    summary = {
        'total_rows': len(rows),
        'good_rows': sum(1 for a in audits if a['judge_outcome'] == 'GOOD'),
        'bad_rows': sum(1 for a in audits if a['judge_outcome'] == 'BAD'),
        'label_consistency_failures': len(label_mismatch),
        'independent_failure_counts': dict(failure_counter),
        'bad_primary_failure_diversity': dict(bad_failures),
        'overfit_flags': overfit_flags,
        'prompt_validity_result': 'PASS' if len(label_mismatch) == 0 else 'FAIL',
        'judge_overfit_risk': 'LOW' if not overfit_flags else 'MEDIUM',
    }

    OUT_JSON.write_text(json.dumps({'summary': summary, 'rows': audits}, indent=2) + '\n', encoding='utf-8')

    md = []
    md.append('# Independent Audit: Insight Opportunity Stage Golden 30')
    md.append('')
    md.append('## Summary')
    md.append(f"- Total rows: {summary['total_rows']}")
    md.append(f"- GOOD rows: {summary['good_rows']}")
    md.append(f"- BAD rows: {summary['bad_rows']}")
    md.append(f"- Label consistency failures: {summary['label_consistency_failures']}")
    md.append(f"- Prompt validity result: {summary['prompt_validity_result']}")
    md.append(f"- Judge overfit risk: {summary['judge_overfit_risk']}")
    md.append('')
    md.append('## Failure Counts (Independent)')
    for k, v in sorted(summary['independent_failure_counts'].items(), key=lambda kv: (-kv[1], kv[0])):
        md.append(f"- {k}: {v}")
    md.append('')
    md.append('## BAD Primary Failure Diversity')
    for k, v in sorted(summary['bad_primary_failure_diversity'].items(), key=lambda kv: (-kv[1], kv[0])):
        md.append(f"- {k}: {v}")

    if summary['overfit_flags']:
        md.append('')
        md.append('## Overfit Flags')
        for f in summary['overfit_flags']:
            md.append(f"- {f}")

    if label_mismatch:
        md.append('')
        md.append('## Label Mismatches')
        for tid, label, v in label_mismatch:
            md.append(f"- {tid}: label={label}, violations={v}")

    OUT_MD.write_text('\n'.join(md) + '\n', encoding='utf-8')

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
