#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import uuid
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from project_paths import ENV_FILE, TRACES_DIR


DEFAULT_BASE_URL = "https://api.braintrust.dev"
DEFAULT_PROJECT_NAME = "llm-gateway_production"


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
    def __init__(self, api_key: str, base_url: str = DEFAULT_BASE_URL):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )

    def btql(self, query: str, query_source: Optional[str] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"query": query, "fmt": "json"}
        if query_source:
            payload["query_source"] = query_source
        resp = self.session.post(f"{self.base_url}/btql", json=payload, timeout=120)
        data = resp.json()
        if resp.status_code >= 400:
            raise RuntimeError(f"BTQL failed ({resp.status_code}): {data}")
        return data

    def get_function(self, project_id: str, slug: str) -> Optional[Dict[str, Any]]:
        resp = self.session.get(
            f"{self.base_url}/v1/function",
            params={"project_id": project_id, "slug": slug},
            timeout=120,
        )
        data = resp.json()
        if resp.status_code >= 400:
            raise RuntimeError(f"/v1/function failed ({resp.status_code}): {data}")
        objs = data.get("objects", [])
        if isinstance(objs, list) and objs:
            obj0 = objs[0]
            if isinstance(obj0, dict):
                return obj0
        return None


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def resolve_scorer_slug(config_file: Path, prompt_name: str) -> str:
    cfg = json.loads(config_file.read_text(encoding="utf-8"))
    evaluations = cfg.get("evaluations", {})
    if not isinstance(evaluations, dict):
        raise RuntimeError("Invalid config file: evaluations must be an object")
    for _, entry in evaluations.items():
        if not isinstance(entry, dict):
            continue
        if entry.get("prompt_name") != prompt_name:
            continue
        scorers = entry.get("scorers", [])
        if isinstance(scorers, list) and scorers and isinstance(scorers[0], dict):
            slug = scorers[0].get("slug")
            if isinstance(slug, str) and slug.strip():
                return slug.strip()
    raise RuntimeError(f"Could not resolve scorer slug for prompt_name={prompt_name}")


def build_query(
    project_id: str,
    prompt_name: str,
    hours: int,
    limit: int,
    before_created: Optional[str],
) -> str:
    created_filter = ""
    if before_created:
        created_filter = f"created < '{before_created}' and "

    query = (
        "select: *\n"
        f"from: project_logs('{project_id}')\n"
        "spans\n"
        "filter: "
        f"created > now() - interval {hours} hour and "
        f"{created_filter}"
        "created < now() and "
        "span_attributes.parent_span_id is null and "
        "span_attributes.name = 'chat_completions' and "
        f"metadata.prompt_name = '{prompt_name}'\n"
        "sort: created desc\n"
        f"limit: {limit}\n"
    )
    return query


def normalize_row(row: Dict[str, Any], prompt_name: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    trace_id = str(row.get("id") or row.get("root_span_id") or "").strip()
    if not trace_id:
        return None, "missing_trace_id"
    inp = row.get("input")
    out = row.get("output")
    if not isinstance(inp, dict):
        return None, "input_not_dict"
    # Flatten wrapped input payloads to avoid input.input shape in exports.
    if "input" in inp and isinstance(inp.get("input"), dict):
        inner = inp.get("input")
        if isinstance(inner, dict):
            inp = inner
    if out is None:
        return None, "output_missing"
    # Hard guardrail against unresolved nested wrapper after flatten attempt.
    if "input" in inp and isinstance(inp.get("input"), dict):
        return None, "nested_input_wrapper_detected"

    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    row_prompt_name = metadata.get("prompt_name")
    if isinstance(row_prompt_name, str) and row_prompt_name and row_prompt_name != prompt_name:
        return None, f"prompt_name_mismatch:{row_prompt_name}"

    normalized = {
        "trace_id": trace_id,
        "created": row.get("created"),
        "input": inp,
        "output": out,
        "metadata": metadata,
        "window_hours": row.get("window_hours"),
    }
    return normalized, None


def extract_rationale(score_result: Any) -> str:
    if isinstance(score_result, dict):
        metadata = score_result.get("metadata")
        if isinstance(metadata, dict):
            rationale = metadata.get("rationale")
            if isinstance(rationale, str) and rationale.strip():
                return rationale.strip()
        return json.dumps(score_result, ensure_ascii=False)[:4000]
    return str(score_result)[:4000]


def extract_score(score_result: Any) -> float:
    if isinstance(score_result, dict):
        score = score_result.get("score")
        try:
            return float(score)
        except Exception:
            return 0.0
    if isinstance(score_result, (int, float)):
        return float(score_result)
    return 0.0


def _to_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)


def _parse_output_object(output_obj: Any) -> Optional[Dict[str, Any]]:
    if isinstance(output_obj, dict):
        return output_obj
    if isinstance(output_obj, str):
        try:
            parsed = json.loads(output_obj)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            return None
    return None


def output_to_expected_value(output_obj: Any) -> Any:
    if isinstance(output_obj, (dict, list, bool, int, float)):
        return output_obj
    if output_obj is None:
        return None
    if isinstance(output_obj, str):
        txt = output_obj.strip()
        if txt.startswith("{") or txt.startswith("["):
            try:
                return json.loads(txt)
            except Exception:
                return output_obj
        return output_obj
    return output_obj


def classify_bad_reason(prompt_name: str, output_obj: Any, rationale: str) -> str:
    parsed = _parse_output_object(output_obj)
    if parsed is None:
        return "output_not_json"

    if "risk" in prompt_name.lower():
        if "risks" not in parsed or "justification" not in parsed:
            return "missing_required_keys"
        risks = parsed.get("risks")
        just = str(parsed.get("justification") or "").strip()
        if not isinstance(risks, list):
            return "risks_not_list"
        if len(risks) > 2:
            return "risk_count_too_high"
        if len(just) < 30:
            return "justification_too_short"

    rt = rationale.lower()
    if "halluc" in rt:
        return "hallucination"
    if "latest" in rt and "not" in rt and ("tied" in rt or "exchange" in rt):
        return "not_grounded_to_latest_exchange"
    if "too large" in rt or "too many" in rt or "over" in rt:
        return "over_flagging"
    return "judge_failed_other"


def independent_worthiness(
    *,
    prompt_name: str,
    input_obj: Dict[str, Any],
    output_obj: Any,
    signature_counts: Dict[str, int],
) -> Tuple[bool, List[str], List[str], str]:
    worthy_reasons: List[str] = []
    reject_reasons: List[str] = []

    # 1) Input has material context.
    input_text = _to_text(input_obj)
    if len(input_text.strip()) >= 180:
        worthy_reasons.append("input_has_material_context")
    else:
        reject_reasons.append("input_too_sparse")

    # 2) Output must be parseable and contract-usable.
    parsed_output: Any = None
    if isinstance(output_obj, dict):
        parsed_output = output_obj
    elif isinstance(output_obj, str):
        try:
            parsed_output = json.loads(output_obj)
        except Exception:
            reject_reasons.append("output_not_json_parseable")
    else:
        reject_reasons.append("output_invalid_type")

    signature = ""
    if isinstance(parsed_output, dict):
        worthy_reasons.append("output_parseable_json_object")
        signature = _to_text(parsed_output)[:220].lower()
    else:
        signature = _to_text(output_obj)[:220].lower()

    # 3) Guardrail against nested input wrapper.
    if set(input_obj.keys()) == {"input"} and isinstance(input_obj.get("input"), dict):
        reject_reasons.append("nested_input_wrapper_detected")
    else:
        worthy_reasons.append("no_nested_input_wrapper")

    # 4) Prompt-specific light checks for risk extractor.
    if "risk" in prompt_name.lower():
        if isinstance(parsed_output, dict):
            risks = parsed_output.get("risks")
            justification = str(parsed_output.get("justification") or "").strip()
            if not isinstance(risks, list):
                reject_reasons.append("risks_not_list")
            else:
                worthy_reasons.append("risks_list_present")
                if len(risks) > 2:
                    reject_reasons.append("risk_count_too_high")
            if len(justification) < 30:
                reject_reasons.append("justification_too_short")
            else:
                worthy_reasons.append("justification_present")
        else:
            reject_reasons.append("risk_prompt_output_contract_invalid")

    # 5) Diversity guard (do not over-represent near-duplicate signatures).
    sig_key = signature[:80]
    existing_count = signature_counts.get(sig_key, 0)
    if existing_count >= 2:
        reject_reasons.append("over_represented_signature")

    is_worthy = len(reject_reasons) == 0
    return is_worthy, worthy_reasons, reject_reasons, sig_key


def independent_bad_worthiness(
    *,
    prompt_name: str,
    input_obj: Dict[str, Any],
    output_obj: Any,
    rationale: str,
    signature_counts: Dict[str, int],
    bad_reason_counts: Dict[str, int],
    max_bad_reason_share: int,
) -> Tuple[bool, List[str], List[str], str, str]:
    worthy_reasons: List[str] = []
    reject_reasons: List[str] = []

    input_text = _to_text(input_obj)
    if len(input_text.strip()) >= 180:
        worthy_reasons.append("input_has_material_context")
    else:
        reject_reasons.append("input_too_sparse")

    if set(input_obj.keys()) == {"input"} and isinstance(input_obj.get("input"), dict):
        reject_reasons.append("nested_input_wrapper_detected")
    else:
        worthy_reasons.append("no_nested_input_wrapper")

    if rationale.lower().startswith("scorer_error:"):
        reject_reasons.append("scorer_error_not_allowed_for_bad_set")

    bad_reason = classify_bad_reason(prompt_name, output_obj, rationale)
    worthy_reasons.append(f"bad_reason:{bad_reason}")

    parsed_output = _parse_output_object(output_obj)
    if parsed_output is not None:
        signature = _to_text(parsed_output)[:220].lower()
    else:
        signature = _to_text(output_obj)[:220].lower()
    sig_key = signature[:80]
    if signature_counts.get(sig_key, 0) >= 2:
        reject_reasons.append("over_represented_signature")

    if bad_reason_counts.get(bad_reason, 0) >= max_bad_reason_share:
        reject_reasons.append("over_represented_bad_reason")

    is_worthy = len(reject_reasons) == 0
    return is_worthy, worthy_reasons, reject_reasons, sig_key, bad_reason


def percentile(values: List[float], p: float) -> Optional[float]:
    if not values:
        return None
    vs = sorted(values)
    idx = int(round((len(vs) - 1) * p))
    idx = max(0, min(len(vs) - 1, idx))
    return float(vs[idx])


def score_with_subprocess(
    *,
    api_key: str,
    project_name: str,
    scorer_slug: str,
    input_obj: Dict[str, Any],
    output_obj: Any,
    timeout_seconds: int,
) -> Any:
    payload = {"input": input_obj, "output": output_obj}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tmp:
        json.dump(payload, tmp, ensure_ascii=False)
        tmp_path = tmp.name

    score_snippet = r"""
import json
import os
import sys
import braintrust
from braintrust import init_function

payload_path = sys.argv[1]
project_name = sys.argv[2]
scorer_slug = sys.argv[3]

with open(payload_path, "r", encoding="utf-8") as f:
    payload = json.load(f)

braintrust.login(api_key=os.environ["BRAINTRUST_API_KEY"])
fn = init_function(project_name=project_name, slug=scorer_slug, version=None)
res = fn(payload["input"], payload["output"], {"judge_outcome": "GOOD"})
print(json.dumps(res, ensure_ascii=False))
"""
    env = dict(os.environ)
    env["BRAINTRUST_API_KEY"] = api_key
    try:
        proc = subprocess.run(
            [sys.executable, "-c", score_snippet, tmp_path, project_name, scorer_slug],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
            check=False,
        )
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass

    if proc.returncode != 0:
        raise RuntimeError(f"subprocess_score_failed: {proc.stderr.strip()[:400]}")
    stdout = (proc.stdout or "").strip()
    if not stdout:
        raise RuntimeError("subprocess_score_empty_stdout")
    last_line = stdout.splitlines()[-1]
    return json.loads(last_line)


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch expand mixed GOOD/BAD golden rows for a prompt.")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--project-name", default=DEFAULT_PROJECT_NAME)
    parser.add_argument("--prompt-name", required=True)
    parser.add_argument("--scorer-slug", default=None)
    parser.add_argument(
        "--config-file",
        default=None,
        help="Path to production_trace_evals.json; required when --scorer-slug is not provided.",
    )
    parser.add_argument(
        "--output-root",
        default=str(TRACES_DIR / "golden_positive_expansion"),
    )
    parser.add_argument("--target-good", type=int, default=30)
    parser.add_argument("--target-bad", type=int, default=20)
    parser.add_argument("--confidence-threshold", type=float, default=0.85)
    parser.add_argument("--bad-max-confidence", type=float, default=0.35)
    parser.add_argument("--windows", default="168,720,2160")
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--max-batches-per-window", type=int, default=100)
    parser.add_argument("--scorer-timeout-seconds", type=int, default=25)
    parser.add_argument("--require-independent-curation", action="store_true", default=True)
    parser.add_argument("--max-bad-reason-share", type=int, default=10)
    parser.add_argument(
        "--output-mode",
        choices=["ids", "artifacts"],
        default="ids",
        help="ids: print only selected trace IDs (default); artifacts: write dataset files and reports.",
    )
    parser.add_argument("--env-file", default=str(ENV_FILE))
    args = parser.parse_args()

    load_dotenv(Path(args.env_file))
    api_key = os.environ.get("BRAINTRUST_API_KEY")
    if not api_key:
        raise SystemExit("Missing BRAINTRUST_API_KEY")

    scorer_slug = args.scorer_slug
    if not scorer_slug:
        if not args.config_file:
            raise SystemExit("Provide either --scorer-slug or --config-file")
        scorer_slug = resolve_scorer_slug(Path(args.config_file), args.prompt_name)

    out_dir = Path(args.output_root) / args.prompt_name
    out_dir.mkdir(parents=True, exist_ok=True)

    bt = BraintrustClient(api_key=api_key, base_url=os.environ.get("BRAINTRUST_BASE_URL", DEFAULT_BASE_URL))

    windows = [int(x.strip()) for x in args.windows.split(",") if x.strip()]

    seen_trace_ids: set[str] = set()
    raw_rows: List[Dict[str, Any]] = []
    normalized_rows: List[Dict[str, Any]] = []
    rejected_rows: List[Dict[str, Any]] = []
    row_reviews: List[Dict[str, Any]] = []
    curation_log: List[Dict[str, Any]] = []
    selected_good_rows: List[Dict[str, Any]] = []
    selected_bad_rows: List[Dict[str, Any]] = []
    selected_good_signature_counts: Dict[str, int] = {}
    selected_bad_signature_counts: Dict[str, int] = {}
    selected_bad_reason_counts: Dict[str, int] = {}
    window_stats: Dict[str, Dict[str, int]] = {}

    for hours in windows:
        if len(selected_good_rows) >= args.target_good and len(selected_bad_rows) >= args.target_bad:
            break
        before_created: Optional[str] = None
        batches = 0
        window_fetched = 0
        window_new = 0

        while batches < args.max_batches_per_window:
            if len(selected_good_rows) >= args.target_good and len(selected_bad_rows) >= args.target_bad:
                break
            query = build_query(
                project_id=args.project_id,
                prompt_name=args.prompt_name,
                hours=hours,
                limit=args.batch_size,
                before_created=before_created,
            )
            response = bt.btql(query=query, query_source=f"golden-positive-{uuid.uuid4()}")
            data = response.get("data", [])
            if not isinstance(data, list) or not data:
                break

            batches += 1
            window_fetched += len(data)
            print(
                f"[window={hours}h batch={batches}] fetched={len(data)} "
                f"good={len(selected_good_rows)}/{args.target_good} "
                f"bad={len(selected_bad_rows)}/{args.target_bad}",
                flush=True,
            )

            for raw in data:
                if not isinstance(raw, dict):
                    continue
                trace_id = str(raw.get("id") or raw.get("root_span_id") or "").strip()
                if not trace_id or trace_id in seen_trace_ids:
                    continue
                seen_trace_ids.add(trace_id)
                window_new += 1
                raw2 = dict(raw)
                raw2["window_hours"] = hours
                raw_rows.append(raw2)

                normalized, reject_reason = normalize_row(raw2, args.prompt_name)
                if reject_reason:
                    rejected_rows.append({"trace_id": trace_id, "reason": reject_reason, "window_hours": hours})
                    continue
                assert normalized is not None
                normalized_rows.append(normalized)

                # Strict row-by-row scorer review.
                try:
                    score_result = score_with_subprocess(
                        api_key=api_key,
                        project_name=args.project_name,
                        scorer_slug=scorer_slug,
                        input_obj=normalized["input"],
                        output_obj=normalized["output"],
                        timeout_seconds=args.scorer_timeout_seconds,
                    )
                    score = extract_score(score_result)
                    rationale = extract_rationale(score_result)
                except Exception as e:
                    score = 0.0
                    rationale = f"scorer_error: {type(e).__name__}: {e}"

                confidence = max(0.0, min(1.0, score))
                judge_outcome = "GOOD" if score >= 0.5 else "BAD"
                passes_scorer = (
                    judge_outcome == "GOOD"
                    and confidence >= args.confidence_threshold
                )

                independent_worthy = True
                independent_worthy_reasons: List[str] = []
                independent_reject_reasons: List[str] = []
                signature_key = ""
                if args.require_independent_curation:
                    (
                        independent_worthy,
                        independent_worthy_reasons,
                        independent_reject_reasons,
                        signature_key,
                    ) = independent_worthiness(
                        prompt_name=args.prompt_name,
                        input_obj=normalized["input"],
                        output_obj=normalized["output"],
                        signature_counts=selected_good_signature_counts,
                    )

                is_selected_good = (
                    passes_scorer
                    and independent_worthy
                    and len(selected_good_rows) < args.target_good
                )

                independent_bad_ok = True
                independent_bad_reasons: List[str] = []
                independent_bad_rejects: List[str] = []
                bad_signature_key = ""
                bad_reason = ""
                if judge_outcome == "BAD" and len(selected_bad_rows) < args.target_bad:
                    if args.require_independent_curation:
                        (
                            independent_bad_ok,
                            independent_bad_reasons,
                            independent_bad_rejects,
                            bad_signature_key,
                            bad_reason,
                        ) = independent_bad_worthiness(
                            prompt_name=args.prompt_name,
                            input_obj=normalized["input"],
                            output_obj=normalized["output"],
                            rationale=rationale,
                            signature_counts=selected_bad_signature_counts,
                            bad_reason_counts=selected_bad_reason_counts,
                            max_bad_reason_share=args.max_bad_reason_share,
                        )
                    else:
                        bad_reason = classify_bad_reason(args.prompt_name, normalized["output"], rationale)
                    is_selected_bad = (
                        confidence <= args.bad_max_confidence
                        and independent_bad_ok
                        and len(selected_bad_rows) < args.target_bad
                    )
                else:
                    is_selected_bad = False

                worthy_reasons = []
                reject_reasons = []
                selected_label = "rejected"
                if is_selected_good:
                    selected_label = "selected_good"
                    worthy_reasons = [
                        "judge_scored_good",
                        f"confidence_ge_{args.confidence_threshold}",
                    ]
                    if args.require_independent_curation:
                        worthy_reasons.extend(independent_worthy_reasons)
                elif is_selected_bad:
                    selected_label = "selected_bad"
                    worthy_reasons = [
                        "judge_scored_bad",
                        f"confidence_le_{args.bad_max_confidence}",
                    ]
                    if bad_reason:
                        worthy_reasons.append(f"bad_reason:{bad_reason}")
                    if args.require_independent_curation:
                        worthy_reasons.extend(independent_bad_reasons)
                else:
                    if judge_outcome != "GOOD":
                        reject_reasons.append("judge_scored_bad")
                    if confidence < args.confidence_threshold:
                        reject_reasons.append("confidence_below_threshold")
                    if args.require_independent_curation:
                        reject_reasons.extend(independent_reject_reasons)
                        reject_reasons.extend(independent_bad_rejects)
                    if len(selected_good_rows) >= args.target_good and judge_outcome == "GOOD":
                        reject_reasons.append("good_quota_already_met")
                    if len(selected_bad_rows) >= args.target_bad and judge_outcome == "BAD":
                        reject_reasons.append("bad_quota_already_met")

                review = {
                    "trace_id": normalized["trace_id"],
                    "created": normalized["created"],
                    "window_hours": hours,
                    "judge_outcome": judge_outcome,
                    "confidence": round(confidence, 4),
                    "model_critique": rationale,
                    "worthy_decision": selected_label,
                    "worthy_reasons": worthy_reasons,
                    "reject_reasons": reject_reasons,
                    "independent_worthy": independent_worthy,
                    "independent_worthy_reasons": independent_worthy_reasons,
                    "independent_reject_reasons": independent_reject_reasons,
                    "independent_bad_worthy": independent_bad_ok,
                    "independent_bad_reasons": independent_bad_reasons,
                    "independent_bad_rejects": independent_bad_rejects,
                    "bad_reason": bad_reason,
                    "score_raw": score,
                }
                row_reviews.append(review)
                curation_log.append(
                    {
                        "trace_id": normalized["trace_id"],
                        "window_hours": hours,
                        "worthy_decision": selected_label,
                        "worthy_reasons": worthy_reasons,
                        "reject_reasons": reject_reasons,
                        "review_notes": rationale[:1200],
                    }
                )

                if is_selected_good:
                    selected_good_rows.append(
                        {
                            "trace_id": normalized["trace_id"],
                            "created": normalized["created"],
                            "input": normalized["input"],
                            "output": normalized["output"],
                            "window_hours": hours,
                            "model_critique": rationale,
                            "selection_signature": signature_key,
                        }
                    )
                    if signature_key:
                        selected_good_signature_counts[signature_key] = (
                            selected_good_signature_counts.get(signature_key, 0) + 1
                        )
                elif is_selected_bad:
                    selected_bad_rows.append(
                        {
                            "trace_id": normalized["trace_id"],
                            "created": normalized["created"],
                            "input": normalized["input"],
                            "output": normalized["output"],
                            "window_hours": hours,
                            "model_critique": rationale,
                            "selection_signature": bad_signature_key,
                            "bad_reason": bad_reason,
                        }
                    )
                    if bad_signature_key:
                        selected_bad_signature_counts[bad_signature_key] = (
                            selected_bad_signature_counts.get(bad_signature_key, 0) + 1
                        )
                    if bad_reason:
                        selected_bad_reason_counts[bad_reason] = (
                            selected_bad_reason_counts.get(bad_reason, 0) + 1
                        )

            if len(selected_good_rows) >= args.target_good and len(selected_bad_rows) >= args.target_bad:
                break

            created_values = [
                str(r.get("created"))
                for r in data
                if isinstance(r, dict) and isinstance(r.get("created"), str)
            ]
            if not created_values:
                break
            oldest_created = min(created_values)
            if before_created is not None and oldest_created >= before_created:
                break
            before_created = oldest_created

        window_stats[str(hours)] = {
            "batches": batches,
            "rows_fetched": window_fetched,
            "new_unique_rows": window_new,
        }

    slug_counts = Counter(
        str(r.get("metadata", {}).get("prompt_slug", ""))
        for r in normalized_rows
        if isinstance(r.get("metadata"), dict) and r.get("metadata", {}).get("prompt_slug")
    )
    dominant_prompt_slug = slug_counts.most_common(1)[0][0] if slug_counts else ""

    prompt_fn = None
    judge_fn = None
    if args.output_mode == "artifacts":
        prompt_fn = bt.get_function(project_id=args.project_id, slug=dominant_prompt_slug) if dominant_prompt_slug else None
        judge_fn = bt.get_function(project_id=args.project_id, slug=scorer_slug)

    selected_good_ids = [str(row["trace_id"]) for row in selected_good_rows]
    selected_bad_ids = [str(row["trace_id"]) for row in selected_bad_rows]

    if args.output_mode == "ids":
        out = {
            "prompt_name": args.prompt_name,
            "scorer_slug": scorer_slug,
            "selected_good_trace_ids": selected_good_ids,
            "selected_bad_trace_ids": selected_bad_ids,
            "counts": {
                "good": len(selected_good_ids),
                "bad": len(selected_bad_ids),
                "reviewed": len(row_reviews),
            },
            "targets": {
                "good": args.target_good,
                "bad": args.target_bad,
            },
            "incomplete": len(selected_good_ids) < args.target_good or len(selected_bad_ids) < args.target_bad,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        print(json.dumps(out, indent=2))
        return 0

    good_rows: List[Dict[str, Any]] = []
    for row in selected_good_rows:
        good_rows.append(
            {
                "input": row["input"],
                "output": row["output"],
                "expected": {
                    "judge_outcome": "GOOD",
                    "model_critique": row["model_critique"],
                },
                "metadata": {
                    "trace_id": row["trace_id"],
                    "prompt_name": args.prompt_name,
                    "prompt_slug": dominant_prompt_slug,
                    "scorer_slug": scorer_slug,
                    "window_hours": row["window_hours"],
                },
            }
        )

    bad_rows: List[Dict[str, Any]] = []
    for row in selected_bad_rows:
        bad_rows.append(
            {
                "input": row["input"],
                "output": row["output"],
                "expected": {
                    "judge_outcome": "BAD",
                    "model_critique": row["model_critique"],
                },
                "metadata": {
                    "trace_id": row["trace_id"],
                    "prompt_name": args.prompt_name,
                    "prompt_slug": dominant_prompt_slug,
                    "scorer_slug": scorer_slug,
                    "window_hours": row["window_hours"],
                    "bad_reason": row.get("bad_reason"),
                },
            }
        )

    mixed_rows = good_rows + bad_rows

    # Braintrust-style row exports (shape like production dataset rows).
    raw_by_trace_id: Dict[str, Dict[str, Any]] = {}
    for r in raw_rows:
        if not isinstance(r, dict):
            continue
        tid = str(r.get("id") or r.get("root_span_id") or "").strip()
        if tid and tid not in raw_by_trace_id:
            raw_by_trace_id[tid] = r

    bt_style_rows: List[Dict[str, Any]] = []

    for row in selected_good_rows:
        tid = row["trace_id"]
        base_row = dict(raw_by_trace_id.get(tid, {}))
        md = dict(base_row.get("metadata", {})) if isinstance(base_row.get("metadata"), dict) else {}
        md["golden_label"] = "GOOD"
        md["golden_model_critique"] = row["model_critique"][:4000]
        base_row["metadata"] = md
        base_row["id"] = base_row.get("id") or tid
        base_row["root_span_id"] = base_row.get("root_span_id") or tid
        base_row["span_id"] = base_row.get("span_id") or tid
        base_row["input"] = row["input"]
        base_row["expected"] = output_to_expected_value(row["output"])
        bt_style_rows.append(base_row)

    for row in selected_bad_rows:
        tid = row["trace_id"]
        base_row = dict(raw_by_trace_id.get(tid, {}))
        md = dict(base_row.get("metadata", {})) if isinstance(base_row.get("metadata"), dict) else {}
        md["golden_label"] = "BAD"
        md["golden_model_critique"] = row["model_critique"][:4000]
        md["golden_bad_reason"] = row.get("bad_reason")
        base_row["metadata"] = md
        base_row["id"] = base_row.get("id") or tid
        base_row["root_span_id"] = base_row.get("root_span_id") or tid
        base_row["span_id"] = base_row.get("span_id") or tid
        base_row["input"] = row["input"]
        base_row["expected"] = output_to_expected_value(row["output"])
        bt_style_rows.append(base_row)

    # Write artifacts.
    write_json(out_dir / "raw_traces.json", {"data": raw_rows})
    write_jsonl(out_dir / "normalized_candidates.jsonl", normalized_rows)
    write_jsonl(out_dir / "rejected_rows.jsonl", rejected_rows)
    write_jsonl(out_dir / "row_reviews.jsonl", row_reviews)
    write_jsonl(out_dir / "curation_log.jsonl", curation_log)
    write_jsonl(out_dir / "golden_positive_30.jsonl", good_rows)
    write_jsonl(out_dir / "golden_bad_20.jsonl", bad_rows)
    write_jsonl(out_dir / "golden_mixed_50.jsonl", mixed_rows)
    write_json(out_dir / "golden_mixed_50_bt_rows.json", bt_style_rows)
    bt_rows_dir = out_dir / "golden_mixed_50_bt_rows"
    bt_rows_dir.mkdir(parents=True, exist_ok=True)
    for row in bt_style_rows:
        tid = str(row.get("id") or row.get("root_span_id") or "unknown")
        write_json(bt_rows_dir / f"row_{tid}.json", [row])

    # Monitoring outputs.
    bad_review_rows = [r for r in row_reviews if r.get("judge_outcome") == "BAD"]
    bad_reason_hist = Counter(str(r.get("bad_reason") or "unknown") for r in bad_review_rows)
    bad_reason_selected_hist = Counter(str(r.get("bad_reason") or "unknown") for r in row_reviews if r.get("worthy_decision") == "selected_bad")
    reviewed_conf = [float(r.get("confidence", 0.0)) for r in row_reviews]
    bad_conf = [float(r.get("confidence", 0.0)) for r in bad_review_rows]
    good_conf = [float(r.get("confidence", 0.0)) for r in row_reviews if r.get("judge_outcome") == "GOOD"]
    bad_rate_reviewed = (len(bad_review_rows) / len(row_reviews)) if row_reviews else 0.0

    monitor = {
        "prompt_name": args.prompt_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reviewed": len(row_reviews),
        "good_selected": len(good_rows),
        "bad_selected": len(bad_rows),
        "bad_rate_reviewed": round(bad_rate_reviewed, 4),
        "confidence": {
            "all": {
                "min": min(reviewed_conf) if reviewed_conf else None,
                "p50": percentile(reviewed_conf, 0.5),
                "p90": percentile(reviewed_conf, 0.9),
                "max": max(reviewed_conf) if reviewed_conf else None,
            },
            "good": {
                "p50": percentile(good_conf, 0.5),
                "p90": percentile(good_conf, 0.9),
            },
            "bad": {
                "p50": percentile(bad_conf, 0.5),
                "p90": percentile(bad_conf, 0.9),
            },
        },
        "bad_reason_histogram": dict(bad_reason_hist.most_common()),
        "bad_reason_selected_histogram": dict(bad_reason_selected_hist.most_common()),
    }

    # Baseline delta vs recent monitor history (last 7 days).
    history_path = out_dir / "monitor_history.jsonl"
    prior_rates: List[float] = []
    now_ts = datetime.now(timezone.utc)
    if history_path.exists():
        for line in history_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            ts_raw = obj.get("generated_at")
            rate = obj.get("bad_rate_reviewed")
            if not isinstance(ts_raw, str) or not isinstance(rate, (int, float)):
                continue
            try:
                ts = datetime.fromisoformat(ts_raw.replace("Z", "+00:00"))
            except Exception:
                continue
            if (now_ts - ts).total_seconds() <= 7 * 24 * 3600:
                prior_rates.append(float(rate))

    baseline = (sum(prior_rates) / len(prior_rates)) if prior_rates else None
    delta = (bad_rate_reviewed - baseline) if baseline is not None else None
    monitor["bad_rate_7d_baseline"] = round(baseline, 4) if baseline is not None else None
    monitor["bad_rate_delta_vs_7d"] = round(delta, 4) if delta is not None else None

    top_reason_share = 0.0
    if bad_reason_hist and len(bad_review_rows) > 0:
        top_reason_share = bad_reason_hist.most_common(1)[0][1] / len(bad_review_rows)
    monitor["top_bad_reason_share"] = round(top_reason_share, 4)

    alerts: List[str] = []
    if baseline is not None and delta is not None and delta > 0.10:
        alerts.append("bad_rate_jump_over_10pp_vs_7d")
    if bad_rate_reviewed > 0.25:
        alerts.append("bad_rate_over_25pct")
    if top_reason_share > 0.50:
        alerts.append("single_bad_reason_over_50pct")
    monitor["alerts"] = alerts

    write_json(out_dir / "monitor_summary.json", monitor)
    with history_path.open("a", encoding="utf-8") as hf:
        hf.write(json.dumps(monitor, ensure_ascii=False) + "\n")

    # Rolling BAD queue for manual review.
    bad_queue = [r for r in row_reviews if r.get("judge_outcome") == "BAD"]
    bad_queue.sort(key=lambda r: str(r.get("created") or ""), reverse=True)
    write_jsonl(out_dir / "bad_queue_latest_50.jsonl", bad_queue[:50])

    summary = {
        "prompt_name": args.prompt_name,
        "project_id": args.project_id,
        "project_name": args.project_name,
        "scorer_slug": scorer_slug,
        "windows_hours": windows,
        "window_stats": window_stats,
        "trace_counts": {
            "raw_unique": len(raw_rows),
            "normalized": len(normalized_rows),
            "rejected": len(rejected_rows),
            "reviewed": len(row_reviews),
            "selected_good": len(good_rows),
            "selected_bad": len(bad_rows),
        },
        "target_good": args.target_good,
        "target_bad": args.target_bad,
        "incomplete": len(good_rows) < args.target_good or len(bad_rows) < args.target_bad,
        "dominant_prompt_slug": dominant_prompt_slug,
        "prompt_function_found": bool(prompt_fn),
        "judge_function_found": bool(judge_fn),
        "prompt_message_count": len((((prompt_fn or {}).get("prompt_data") or {}).get("prompt") or {}).get("messages") or [])
        if prompt_fn
        else 0,
        "judge_message_count": len((((judge_fn or {}).get("prompt_data") or {}).get("prompt") or {}).get("messages") or [])
        if judge_fn
        else 0,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(out_dir / "summary.json", summary)

    report_lines = [
        f"# Golden Positive Expansion Report: {args.prompt_name}",
        "",
        f"- Scorer slug: `{scorer_slug}`",
        f"- Target GOOD rows: {args.target_good}",
        f"- Target BAD rows: {args.target_bad}",
        f"- Selected GOOD rows: {len(good_rows)}",
        f"- Selected BAD rows: {len(bad_rows)}",
        f"- Incomplete: {'YES' if (len(good_rows) < args.target_good or len(bad_rows) < args.target_bad) else 'NO'}",
        f"- Bad rate reviewed: {round(bad_rate_reviewed, 4)}",
        f"- Alerts: {', '.join(alerts) if alerts else 'none'}",
        "",
        "## Window Stats",
    ]
    for h in windows:
        ws = window_stats.get(str(h), {})
        report_lines.append(
            f"- {h}h: batches={ws.get('batches', 0)}, fetched={ws.get('rows_fetched', 0)}, unique={ws.get('new_unique_rows', 0)}"
        )
    report_lines.extend(
        [
            "",
            "## Artifacts",
            f"- `{out_dir / 'raw_traces.json'}`",
            f"- `{out_dir / 'normalized_candidates.jsonl'}`",
            f"- `{out_dir / 'rejected_rows.jsonl'}`",
            f"- `{out_dir / 'row_reviews.jsonl'}`",
            f"- `{out_dir / 'curation_log.jsonl'}`",
            f"- `{out_dir / 'golden_positive_30.jsonl'}`",
            f"- `{out_dir / 'golden_bad_20.jsonl'}`",
            f"- `{out_dir / 'golden_mixed_50.jsonl'}`",
            f"- `{out_dir / 'golden_mixed_50_bt_rows.json'}`",
            f"- `{out_dir / 'golden_mixed_50_bt_rows'}`",
            f"- `{out_dir / 'monitor_summary.json'}`",
            f"- `{out_dir / 'bad_queue_latest_50.jsonl'}`",
            f"- `{out_dir / 'summary.json'}`",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
