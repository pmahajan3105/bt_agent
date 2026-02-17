#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

GOLDEN = Path('/Users/prashant/bt_agent/traces/insight_prospecting_bg_golden_30.jsonl')
OUT_JSON = Path('/Users/prashant/bt_agent/traces/insight_prospecting_bg_golden_30_independent_audit.json')
OUT_MD = Path('/Users/prashant/bt_agent/traces/insight_prospecting_bg_golden_30_independent_audit.md')


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]


def task_outcome_counts(pc: Dict[str, Any]) -> Tuple[int, int, int]:
    pos = neg = weak = 0
    all_plays = pc.get('all_plays') if isinstance(pc.get('all_plays'), list) else []
    for play in all_plays:
        if not isinstance(play, dict):
            continue
        tasks = play.get('tasks') if isinstance(play.get('tasks'), list) else []
        for t in tasks:
            if not isinstance(t, dict):
                continue
            o = str(t.get('task_outcome') or '').upper()
            if o in {'POSITIVE_REPLY', 'MEETING_BOOKED'}:
                pos += 1
            elif o in {'NEGATIVE_REPLY', 'OPTED_OUT', 'BOUNCED'}:
                neg += 1
            elif o in {'LINK_CLICKED', 'MESSAGE_OPENED'}:
                weak += 1

    contact_engagement = pc.get('contact_engagement') if isinstance(pc.get('contact_engagement'), list) else []
    for e in contact_engagement:
        if not isinstance(e, dict):
            continue
        rs = str(e.get('reply_sentiment') or '').upper()
        if rs == 'POSITIVE':
            pos += 1
        elif rs == 'NEGATIVE':
            neg += 1
    return pos, neg, weak


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

    combined = '\n'.join(
        [
            summary if isinstance(summary, str) else '',
            *([h for h in highlights if isinstance(h, str)] if isinstance(highlights, list) else []),
            *([r for r in recommendation if isinstance(r, str)] if isinstance(recommendation, list) else []),
            action_reasoning if isinstance(action_reasoning, str) else '',
        ]
    )
    low = combined.lower()

    if '—' in combined:
        violations.append('em_dash_present')

    if isinstance(summary, str) and len(re.findall(r"[A-Za-z0-9_']+", summary)) > 50:
        violations.append('summary_too_long')

    if isinstance(highlights, list) and not (2 <= len(highlights) <= 5):
        violations.append('highlights_count_out_of_range')

    hs = inp.get('has_sequences')
    has_sequences = hs if isinstance(hs, bool) else str(hs).strip().lower() == 'true'
    if has_sequences and recommendation is not None:
        violations.append('has_sequences_requires_null_recommendation')
    if (not has_sequences) and not (isinstance(recommendation, list) and 1 <= len(recommendation) <= 3):
        violations.append('no_sequences_requires_1_to_3_recommendations')

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

    pc = inp.get('prospecting_context') if isinstance(inp.get('prospecting_context'), dict) else {}
    pos, neg, weak = task_outcome_counts(pc)
    if pos + neg == 0:
        strong = ['strong engagement', 'high engagement', 'momentum is strong', 'replies received', 'reply received', 'engaged']
        if any(s in low for s in strong) and 'no replies' not in low:
            violations.append('strong_engagement_without_replies')

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
        audits.append({
            'trace_id': tid,
            'judge_outcome': label,
            'independent_violations': v,
            'label_consistency': consistency,
        })

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
    md.append('# Independent Audit: Insight Prospecting Golden 30')
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
