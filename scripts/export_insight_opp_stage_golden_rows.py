#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

BASE = Path('/Users/prashant/bt_agent/traces')
GOLDEN = BASE / 'insight_opp_stage_bg_golden_30.jsonl'
RAW = BASE / 'insight_opp_stage_bg_raw_traces.json'
OUT_ALL = BASE / 'insight_opp_stage_bg_golden_30_rows.json'
OUT_DIR = BASE / 'insight_opp_stage_bg_golden_30_rows'


def read_jsonl(path: Path):
    return [json.loads(l) for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]


def main() -> int:
    golden = read_jsonl(GOLDEN)
    raw = json.loads(RAW.read_text(encoding='utf-8'))
    raw_rows = raw.get('data', []) if isinstance(raw, dict) else raw
    if not isinstance(raw_rows, list):
        raise SystemExit('Raw trace file must contain data[] list')

    by_id = {str(r.get('id') or r.get('root_span_id') or ''): r for r in raw_rows if isinstance(r, dict)}

    out_rows = []
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for g in golden:
        tid = str(g.get('trace_id') or '').strip()
        if not tid:
            continue
        raw_row = by_id.get(tid)
        if not raw_row:
            raise SystemExit(f'Missing raw trace for selected trace_id: {tid}')

        output_obj = raw_row.get('output')
        parsed_output = output_obj
        if isinstance(output_obj, str):
            try:
                parsed_output = json.loads(output_obj)
            except Exception:
                parsed_output = output_obj

        row_obj = {
            '_pagination_key': raw_row.get('_pagination_key'),
            '_xact_id': raw_row.get('_xact_id'),
            'created': raw_row.get('created'),
            'flaggedForReview': False,
            'id': raw_row.get('id') or tid,
            'model': ((raw_row.get('metadata') or {}).get('model') if isinstance(raw_row.get('metadata'), dict) else None),
            'origin': raw_row.get('origin'),
            'root_span_id': raw_row.get('root_span_id') or tid,
            'span_id': raw_row.get('span_id') or tid,
            'tags': raw_row.get('tags') if isinstance(raw_row.get('tags'), list) else [],
            'userReviewId': None,
            'span_attributes': raw_row.get('span_attributes') if isinstance(raw_row.get('span_attributes'), dict) else {'name': ''},
            'metrics': raw_row.get('metrics') if isinstance(raw_row.get('metrics'), dict) else {'estimated_cost': 0},
            'scores': raw_row.get('scores') if isinstance(raw_row.get('scores'), dict) else {},
            'audit_data': raw_row.get('audit_data') if isinstance(raw_row.get('audit_data'), list) else [],
            'classifications': raw_row.get('classifications'),
            'comments': raw_row.get('comments'),
            'dataset_id': raw_row.get('dataset_id'),
            'expected': {
                'judge_outcome': g.get('judge_outcome'),
                'model_critique': g.get('model_critique'),
            },
            'facets': raw_row.get('facets'),
            'input': raw_row.get('input'),
            'is_root': bool(raw_row.get('is_root', True)),
            'metadata': raw_row.get('metadata') if isinstance(raw_row.get('metadata'), dict) else {},
            'project_id': raw_row.get('project_id'),
            'log_model_output': parsed_output,
            '__bt_assignments': raw_row.get('__bt_assignments'),
        }

        out_rows.append(row_obj)

        single_path = OUT_DIR / f'row_{tid}.json'
        single_path.write_text(json.dumps([row_obj], ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    OUT_ALL.write_text(json.dumps(out_rows, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print({
        'rows_written': len(out_rows),
        'combined_file': str(OUT_ALL),
        'per_row_dir': str(OUT_DIR),
    })
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
