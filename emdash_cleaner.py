#!/usr/bin/env python3
"""
Normalize or filter em dashes (—) in datasets / traces.

Why:
- Some prompts/judges ban the em dash character (—).
- Historical traces may contain em dashes, causing noisy failures.

This tool creates a NEW output file (does not modify inputs).

Supported inputs:
- JSONL: one JSON object per line
- JSON: either a list of objects OR an object with a "rows" list

It can clean common output locations by default, and optionally clean parts
of the input payload (e.g. input.message_thread) to avoid false failures when
judges/pipelines mistakenly treat em dashes in the *thread* as violations.

If the string looks like JSON (starts with "{"), it will parse and rewrite ALL string values.
Otherwise, it treats it as plain text.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


EM_DASH = "\u2014"
EM_DASH_RE = re.compile(r"\s*\u2014\s*")


def _set_nested(obj: Dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cur: Any = obj
    for p in parts[:-1]:
        if not isinstance(cur, dict):
            return
        nxt = cur.get(p)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[p] = nxt
        cur = nxt
    if isinstance(cur, dict):
        cur[parts[-1]] = value


def _get_nested(obj: Dict[str, Any], path: str) -> Any:
    cur: Any = obj
    for p in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur


def _replace_emdash(text: str, replacement: str) -> Tuple[str, int]:
    if EM_DASH not in text:
        return text, 0
    new_text, n = EM_DASH_RE.subn(replacement, text)
    return new_text, n


def _walk_replace_strings(obj: Any, replacement: str) -> Tuple[Any, int]:
    if obj is None:
        return obj, 0
    if isinstance(obj, str):
        return _replace_emdash(obj, replacement)
    if isinstance(obj, list):
        total = 0
        out: List[Any] = []
        for v in obj:
            v2, n = _walk_replace_strings(v, replacement)
            out.append(v2)
            total += n
        return out, total
    if isinstance(obj, dict):
        total = 0
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            v2, n = _walk_replace_strings(v, replacement)
            out[k] = v2
            total += n
        return out, total
    return obj, 0


def _clean_payload_string(payload: str, replacement: str) -> Tuple[str, int]:
    s = payload.lstrip()
    if s.startswith("{"):
        try:
            parsed = json.loads(payload)
        except Exception:
            # fall back to plain text replacement
            return _replace_emdash(payload, replacement)
        parsed2, n = _walk_replace_strings(parsed, replacement)
        return json.dumps(parsed2, ensure_ascii=False), n
    return _replace_emdash(payload, replacement)


def _clean_row(row: Dict[str, Any], *, fields_to_clean: List[str], replacement: str) -> Tuple[Dict[str, Any], int]:
    total = 0
    out = dict(row)

    for field_path in fields_to_clean:
        val = _get_nested(out, field_path) if "." in field_path else out.get(field_path)
        if isinstance(val, str):
            cleaned, n = _clean_payload_string(val, replacement)
            if "." in field_path:
                _set_nested(out, field_path, cleaned)
            else:
                out[field_path] = cleaned
            total += n

    return out, total


def _row_has_emdash(row: Dict[str, Any]) -> bool:
    def has(v: Any) -> bool:
        if v is None:
            return False
        if isinstance(v, str):
            return EM_DASH in v
        if isinstance(v, list):
            return any(has(x) for x in v)
        if isinstance(v, dict):
            return any(has(x) for x in v.values())
        return False

    return has(row)


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if not isinstance(obj, dict):
                raise ValueError("JSONL rows must be JSON objects")
            rows.append(obj)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", required=True)
    ap.add_argument(
        "--mode",
        choices=["replace", "drop"],
        default="replace",
        help="replace: normalize em dashes to ' - '. drop: remove rows containing em dashes anywhere.",
    )
    ap.add_argument(
        "--scope",
        choices=["output_only", "input_and_output", "all_strings"],
        default="output_only",
        help=(
            "output_only: clean common output fields only. "
            "input_and_output: also clean input.message_thread. "
            "all_strings: replace em dashes in all string values (entire row)."
        ),
    )
    ap.add_argument(
        "--replacement",
        default=" - ",
        help="Replacement string for em dashes (default: ' - ').",
    )
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    output_fields = ["output_raw", "output", "input.output_raw"]
    if args.scope == "output_only":
        fields_to_clean = output_fields
    elif args.scope == "input_and_output":
        fields_to_clean = output_fields + ["input.message_thread"]
    else:
        # all_strings handled separately to avoid path maintenance
        fields_to_clean = []

    if in_path.suffix == ".jsonl":
        rows = _read_jsonl(in_path)
        dropped = 0
        changed_rows = 0
        replacements = 0
        out_rows: List[Dict[str, Any]] = []

        for row in rows:
            if args.mode == "drop" and _row_has_emdash(row):
                dropped += 1
                continue
            if args.scope == "all_strings":
                cleaned2, n = _walk_replace_strings(row, args.replacement)
                cleaned = cleaned2 if isinstance(cleaned2, dict) else row
            else:
                cleaned, n = _clean_row(row, fields_to_clean=fields_to_clean, replacement=args.replacement)
            if n > 0:
                changed_rows += 1
                replacements += n
            out_rows.append(cleaned)

        _write_jsonl(out_path, out_rows)
        print(
            json.dumps(
                {
                    "input_rows": len(rows),
                    "output_rows": len(out_rows),
                    "dropped_rows": dropped,
                    "changed_rows": changed_rows,
                    "emdash_replacements": replacements,
                    "out_path": str(out_path),
                },
                indent=2,
            )
        )
        return 0

    # JSON file (list or {"rows": [...]})
    data = json.loads(in_path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("rows"), list):
        rows_any = data["rows"]
        wrapper = True
    elif isinstance(data, list):
        rows_any = data
        wrapper = False
    else:
        raise SystemExit("Unsupported JSON format. Expected a list or an object with a 'rows' list.")

    if not all(isinstance(r, dict) for r in rows_any):
        raise SystemExit("All rows must be JSON objects")

    rows = [r for r in rows_any]
    dropped = 0
    changed_rows = 0
    replacements = 0
    out_rows: List[Dict[str, Any]] = []

    for row in rows:
        if args.mode == "drop" and _row_has_emdash(row):
            dropped += 1
            continue
        if args.scope == "all_strings":
            cleaned2, n = _walk_replace_strings(row, args.replacement)
            cleaned = cleaned2 if isinstance(cleaned2, dict) else row
        else:
            cleaned, n = _clean_row(row, fields_to_clean=fields_to_clean, replacement=args.replacement)
        if n > 0:
            changed_rows += 1
            replacements += n
        out_rows.append(cleaned)

    if wrapper:
        data2 = dict(data)
        data2["rows"] = out_rows
        out_path.write_text(json.dumps(data2, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    else:
        out_path.write_text(json.dumps(out_rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "input_rows": len(rows),
                "output_rows": len(out_rows),
                "dropped_rows": dropped,
                "changed_rows": changed_rows,
                "emdash_replacements": replacements,
                "out_path": str(out_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
