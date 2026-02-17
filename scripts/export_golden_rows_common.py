#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from project_paths import TRACES_DIR


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _parse_output(raw_row: Dict[str, Any]) -> Any:
    output_obj = raw_row.get("output")
    if not isinstance(output_obj, str):
        return output_obj
    try:
        return json.loads(output_obj)
    except Exception:
        return output_obj


def _build_export_row(raw_row: Dict[str, Any], golden_row: Dict[str, Any], trace_id: str) -> Dict[str, Any]:
    metadata = raw_row.get("metadata")
    return {
        "_pagination_key": raw_row.get("_pagination_key"),
        "_xact_id": raw_row.get("_xact_id"),
        "created": raw_row.get("created"),
        "flaggedForReview": False,
        "id": raw_row.get("id") or trace_id,
        "model": metadata.get("model") if isinstance(metadata, dict) else None,
        "origin": raw_row.get("origin"),
        "root_span_id": raw_row.get("root_span_id") or trace_id,
        "span_id": raw_row.get("span_id") or trace_id,
        "tags": raw_row.get("tags") if isinstance(raw_row.get("tags"), list) else [],
        "userReviewId": None,
        "span_attributes": (
            raw_row.get("span_attributes")
            if isinstance(raw_row.get("span_attributes"), dict)
            else {"name": ""}
        ),
        "metrics": (
            raw_row.get("metrics")
            if isinstance(raw_row.get("metrics"), dict)
            else {"estimated_cost": 0}
        ),
        "scores": raw_row.get("scores") if isinstance(raw_row.get("scores"), dict) else {},
        "audit_data": raw_row.get("audit_data") if isinstance(raw_row.get("audit_data"), list) else [],
        "classifications": raw_row.get("classifications"),
        "comments": raw_row.get("comments"),
        "dataset_id": raw_row.get("dataset_id"),
        "expected": {
            "judge_outcome": golden_row.get("judge_outcome"),
            "model_critique": golden_row.get("model_critique"),
        },
        "facets": raw_row.get("facets"),
        "input": raw_row.get("input"),
        "is_root": bool(raw_row.get("is_root", True)),
        "metadata": metadata if isinstance(metadata, dict) else {},
        "project_id": raw_row.get("project_id"),
        "log_model_output": _parse_output(raw_row),
        "__bt_assignments": raw_row.get("__bt_assignments"),
    }


def export_golden_rows(prefix: str, base_dir: Path = TRACES_DIR) -> Dict[str, str | int]:
    golden_path = base_dir / f"{prefix}_golden_30.jsonl"
    raw_path = base_dir / f"{prefix}_raw_traces.json"
    out_all_path = base_dir / f"{prefix}_golden_30_rows.json"
    out_dir = base_dir / f"{prefix}_golden_30_rows"

    golden_rows = read_jsonl(golden_path)
    raw_obj = json.loads(raw_path.read_text(encoding="utf-8"))
    raw_rows = raw_obj.get("data", []) if isinstance(raw_obj, dict) else raw_obj
    if not isinstance(raw_rows, list):
        raise SystemExit("Raw trace file must contain data[] list")

    raw_by_trace_id = {
        str(row.get("id") or row.get("root_span_id") or ""): row
        for row in raw_rows
        if isinstance(row, dict)
    }

    exported_rows: List[Dict[str, Any]] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    for golden_row in golden_rows:
        trace_id = str(golden_row.get("trace_id") or "").strip()
        if not trace_id:
            continue

        raw_row = raw_by_trace_id.get(trace_id)
        if raw_row is None:
            raise SystemExit(f"Missing raw trace for selected trace_id: {trace_id}")

        export_row = _build_export_row(raw_row, golden_row, trace_id)
        exported_rows.append(export_row)

        per_row_path = out_dir / f"row_{trace_id}.json"
        per_row_path.write_text(json.dumps([export_row], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    out_all_path.write_text(json.dumps(exported_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary = {
        "rows_written": len(exported_rows),
        "combined_file": str(out_all_path),
        "per_row_dir": str(out_dir),
    }
    print(summary)
    return summary
