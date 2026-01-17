import argparse
import json
import os
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional

import requests

DEFAULT_BT_BASE_URL = "https://api.braintrust.dev"


class BraintrustApiError(RuntimeError):
    pass


def _load_dotenv_always() -> None:
    """
    Load environment variables from a .env file (if present).

    Behavior:
    - Looks for `.env` in the same directory as this script by default.
    - You can override the path by setting `BT_AGENT_ENV_FILE` to an absolute/relative path.
    - Does NOT override already-set environment variables.
    - If python-dotenv isn't installed (or the file doesn't exist), it no-ops.
    """
    env_file = os.environ.get("BT_AGENT_ENV_FILE")
    dotenv_path = (
        Path(env_file).expanduser().resolve()
        if env_file
        else (Path(__file__).resolve().parent / ".env")
    )

    if not dotenv_path.exists():
        return

    # Prefer python-dotenv if installed (handles more edge cases), but fall back to a tiny
    # built-in parser so .env loading still works in minimal environments.
    try:
        from dotenv import load_dotenv  # type: ignore

        load_dotenv(dotenv_path=dotenv_path, override=False)
        return
    except Exception:
        pass

    try:
        content = dotenv_path.read_text(encoding="utf-8")
    except Exception:
        return

    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        if not key or not key.replace("_", "").isalnum():
            continue

        # Strip trailing inline comments only when the value is unquoted.
        if val and val[0] not in {"'", '"'} and " #" in val:
            val = val.split(" #", 1)[0].rstrip()

        # Strip surrounding quotes.
        if len(val) >= 2 and ((val[0] == val[-1] == '"') or (val[0] == val[-1] == "'")):
            val = val[1:-1]

        if key not in os.environ:
            os.environ[key] = val


@dataclass(frozen=True)
class PlannedAction:
    type: Literal["dataset.patch", "btql.query"]
    args: Dict[str, Any]
    reason: str = ""


class BraintrustApiClient:
    def __init__(self, api_key: str, base_url: str = DEFAULT_BT_BASE_URL, timeout_s: int = 60):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.session = requests.Session()
        self.session.headers.update(
            {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        )

    def get_dataset(self, dataset_id: str) -> Dict[str, Any]:
        resp = self.session.get(f"{self.base_url}/v1/dataset/{dataset_id}", timeout=self.timeout_s)
        return _json_or_raise(resp)

    def patch_dataset(self, dataset_id: str, body: Dict[str, Any]) -> Dict[str, Any]:
        resp = self.session.patch(
            f"{self.base_url}/v1/dataset/{dataset_id}",
            json=body,
            timeout=self.timeout_s,
        )
        return _json_or_raise(resp)

    def fetch_dataset(
        self,
        dataset_id: str,
        limit: int = 1,
        offset: int = 0,
    ) -> Dict[str, Any]:
        resp = self.session.get(
            f"{self.base_url}/v1/dataset/{dataset_id}/fetch",
            params={"limit": limit, "offset": offset},
            timeout=self.timeout_s,
        )
        return _json_or_raise(resp)

    def btql(self, query: str, fmt: str = "json", query_source: Optional[str] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"query": query, "fmt": fmt}
        if query_source:
            payload["query_source"] = query_source
        resp = self.session.post(f"{self.base_url}/btql", json=payload, timeout=self.timeout_s)
        return _json_or_raise(resp)


def _json_or_raise(resp: requests.Response) -> Dict[str, Any]:
    try:
        data = resp.json()
    except Exception:
        text = (resp.text or "").strip()
        raise BraintrustApiError(f"HTTP {resp.status_code}: Non-JSON response: {text[:500]}")

    if resp.status_code >= 400:
        raise BraintrustApiError(f"HTTP {resp.status_code}: {json.dumps(data)[:2000]}")
    if not isinstance(data, dict):
        raise BraintrustApiError(f"Unexpected response type: {type(data).__name__}")
    return data


def _read_json_arg(json_str: Optional[str], json_path: Optional[str]) -> Dict[str, Any]:
    if bool(json_str) == bool(json_path):
        raise ValueError("Provide exactly one of --json or --json-file")

    raw = json_str if json_str is not None else Path(json_path or "").read_text(encoding="utf-8")
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("JSON body must be an object")
    return parsed


def _prompt_confirm(prompt: str, yes: bool) -> None:
    if yes:
        return
    sys.stdout.write(prompt)
    sys.stdout.flush()
    answer = sys.stdin.readline().strip().lower()
    if answer not in {"y", "yes"}:
        raise SystemExit("Cancelled.")


def _extract_first_json_object(text: str) -> Dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model response did not contain a JSON object")
    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("Model JSON must be an object")
    return parsed


def _llm_plan(
    instruction: str,
    available_action_types: List[str],
    model: str,
    instructions_file_text: str = "",
) -> List[PlannedAction]:
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY (or LLM_API_KEY) to use the agent planner.")

    try:
        import litellm
    except Exception as e:
        raise RuntimeError("Agent mode requires 'litellm' installed in your environment") from e

    system = """You are a careful operator that turns a user instruction into an executable plan.
Only use the allowed action types.
Return ONLY a single JSON object.
Never invent dataset_ids or project_ids.
"""

    user = {
        "instruction": instruction,
        "operator_instructions": instructions_file_text,
        "allowed_action_types": available_action_types,
        "required_output": {
            "actions": [
                {
                    "type": "dataset.patch | btql.query",
                    "args": {},
                    "reason": "optional",
                }
            ]
        },
    }

    resp = litellm.completion(
        model=model,
        api_key=api_key,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user)},
        ],
        response_format={"type": "json_object"},
    )

    content = resp["choices"][0]["message"]["content"] if isinstance(resp, dict) else ""
    plan_obj = _extract_first_json_object(content or "")

    actions_raw = plan_obj.get("actions", [])
    if not isinstance(actions_raw, list):
        raise ValueError("Plan JSON must contain an 'actions' array")

    actions: List[PlannedAction] = []
    for item in actions_raw:
        if not isinstance(item, dict):
            continue
        action_type = item.get("type")
        if action_type not in available_action_types:
            continue
        args = item.get("args", {})
        if not isinstance(args, dict):
            continue
        actions.append(
            PlannedAction(
                type=action_type,
                args=args,
                reason=str(item.get("reason", "")),
            )
        )

    return actions


def _render_plan(actions: List[PlannedAction]) -> str:
    if not actions:
        return "Plan:\n  (no actions)\n"

    lines: List[str] = ["Plan:"]
    for i, a in enumerate(actions, start=1):
        lines.append(f"  {i}. {a.type}")
        if a.reason:
            lines.append(f"     reason: {a.reason}")
        lines.append(f"     args: {json.dumps(a.args, indent=2, sort_keys=True)}")
    return "\n".join(lines) + "\n"


def _build_prompt_name_filter(prompt_name: str, hours: int) -> str:
    parts = [
        f"created > now() - interval {hours} hour",
        "created < now()",
        "span_attributes.parent_span_id is null",
        "span_attributes.name = 'chat_completions'",
        f"metadata.prompt_name = '{prompt_name}'",
    ]
    return " and ".join(parts)


def _fetch_traces_btql(
    client: BraintrustApiClient,
    project_id: str,
    prompt_name: str,
    hours: int,
    max_traces: int,
    query_source: Optional[str] = None,
) -> List[Dict[str, Any]]:
    where_clause = _build_prompt_name_filter(prompt_name, hours)
    query = (
        "select: *\n"
        f"from: project_logs('{project_id}')\n"
        "spans\n"
        f"filter: {where_clause}\n"
        f"sample: {max_traces}\n"
    )
    resp = client.btql(query, fmt="json", query_source=query_source)
    data = resp.get("data", [])
    return data if isinstance(data, list) else []


def _analyze_traces(traces: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    total = 0
    prompt_names: Dict[str, int] = {}
    models: Dict[str, int] = {}
    earliest: Optional[str] = None
    latest: Optional[str] = None

    for t in traces:
        if not isinstance(t, dict):
            continue
        total += 1

        meta = t.get("metadata") or {}
        if isinstance(meta, dict):
            pn = meta.get("prompt_name")
            if isinstance(pn, str) and pn:
                prompt_names[pn] = prompt_names.get(pn, 0) + 1

        attrs = t.get("span_attributes") or {}
        if isinstance(attrs, dict):
            m = attrs.get("model") or attrs.get("llm_model")
            if isinstance(m, str) and m:
                models[m] = models.get(m, 0) + 1

        created = t.get("created")
        if isinstance(created, str):
            earliest = created if earliest is None or created < earliest else earliest
            latest = created if latest is None or created > latest else latest

    return {
        "total": total,
        "prompt_names": dict(sorted(prompt_names.items(), key=lambda kv: (-kv[1], kv[0]))),
        "models": dict(sorted(models.items(), key=lambda kv: (-kv[1], kv[0]))),
        "earliest": earliest,
        "latest": latest,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }


def _auto_dataset_description(dataset: Dict[str, Any], sample_events: List[Dict[str, Any]]) -> str:
    """
    Generate a short, human-friendly dataset description primarily from *actual sample event content*.
    (We intentionally avoid listing "Inputs include: ..." since users often don't want that noise.)
    """
    name = str(dataset.get("name") or "").strip()

    sample = sample_events[0] if sample_events else {}
    inp = sample.get("input") if isinstance(sample, dict) else None
    exp = sample.get("expected") if isinstance(sample, dict) else None

    # Heuristics from real content.
    if isinstance(inp, dict):
        # Opportunity stage classification dataset.
        if "call_transcript" in inp or "email_thread" in inp:
            # Try to name the label field if present.
            label_field = None
            if isinstance(exp, dict):
                for k in ("opportunity_stage", "stage", "label", "class"):
                    if k in exp:
                        label_field = k
                        break
            label_part = f" labeled with `{label_field}`" if label_field else " with stage labels"
            return (
                "Evaluation dataset for classifying sales opportunity stage from customer communications "
                f"(call transcripts and/or email threads){label_part}."
            )

        # Chat query parser / meeting search intent dataset.
        if "user_query" in inp and ("index_rules" in inp or "index_mapping" in inp):
            target = str(inp.get("target_entity") or "").strip()
            target_part = f" for `{target}`" if target else ""
            return (
                "Evaluation dataset of natural-language questions used to test a query parser/translator"
                f"{target_part} (queries plus retrieval/index rules)."
            )

    # Fall back to name-only.
    if name:
        return f"Evaluation dataset for `{name}`."
    return "Evaluation dataset."


def _llm_dataset_description(
    dataset: Dict[str, Any],
    sample_events: List[Dict[str, Any]],
    model: str,
) -> str:
    """
    Use an LLM to propose a concise dataset description based on the dataset metadata + 1 sample event.
    Returns a single short paragraph.
    """
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("LLM_API_KEY")
    if not api_key:
        raise ValueError("Set OPENAI_API_KEY (or LLM_API_KEY) to generate an AI description.")

    try:
        import litellm
    except Exception as e:
        raise RuntimeError("AI description requires 'litellm' installed in your environment") from e

    sample = sample_events[0] if sample_events else {}
    # Keep prompt payload bounded.
    ds_text = json.dumps(dataset, sort_keys=True)
    sample_text = json.dumps(sample, sort_keys=True)
    max_chars = 12000
    if len(ds_text) > max_chars:
        ds_text = ds_text[:max_chars] + "...(truncated)"
    if len(sample_text) > max_chars:
        sample_text = sample_text[:max_chars] + "...(truncated)"

    system = (
        "You write short dataset descriptions for ML evaluation datasets.\n"
        "Requirements:\n"
        "- Output ONLY the description text (no JSON, no quotes, no markdown).\n"
        "- 1 sentence preferred, 2 sentences max.\n"
        "- Do NOT list input keys.\n"
        "- Mention the task and the type of examples/labels.\n"
        "- Avoid internal IDs/UUIDs.\n"
    )

    user = (
        "Propose a description for this dataset.\n\n"
        f"DATASET_METADATA_JSON:\n{ds_text}\n\n"
        f"SAMPLE_EVENT_JSON:\n{sample_text}\n"
    )

    resp = litellm.completion(
        model=model,
        api_key=api_key,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    content = resp["choices"][0]["message"]["content"] if isinstance(resp, dict) else ""
    desc = (content or "").strip()
    # Basic cleanup: collapse whitespace and trim.
    desc = " ".join(desc.split())
    if not desc:
        raise ValueError("Model did not return a description.")
    return desc


def _execute_actions(
    client: BraintrustApiClient,
    actions: List[PlannedAction],
    yes: bool,
    out_dir: Optional[str],
) -> None:
    out_base = Path(out_dir).expanduser().resolve() if out_dir else None
    if out_base:
        out_base.mkdir(parents=True, exist_ok=True)

    for idx, action in enumerate(actions, start=1):
        if action.type == "dataset.patch":
            dataset_id = str(action.args.get("dataset_id", "")).strip()
            body = action.args.get("body")
            if not dataset_id or not isinstance(body, dict):
                raise ValueError("dataset.patch requires args.dataset_id and args.body (object)")
            _prompt_confirm(
                f"\nApply action {idx}/{len(actions)}: PATCH dataset {dataset_id}? [y/N] ",
                yes,
            )
            res = client.patch_dataset(dataset_id, body)
            sys.stdout.write(json.dumps(res, indent=2, sort_keys=True) + "\n")
            continue

        if action.type == "btql.query":
            query = action.args.get("query")
            if not isinstance(query, str) or not query.strip():
                raise ValueError("btql.query requires args.query (string)")
            query_source = str(action.args.get("query_source") or uuid.uuid4())
            _prompt_confirm(f"\nRun action {idx}/{len(actions)}: BTQL query? [y/N] ", yes)
            res = client.btql(query, fmt="json", query_source=query_source)
            save_as = action.args.get("save_as")
            if out_base and isinstance(save_as, str) and save_as.strip():
                out_path = out_base / save_as
                out_path.write_text(json.dumps(res, indent=2, sort_keys=True), encoding="utf-8")
                sys.stdout.write(f"Wrote: {out_path}\n")
            else:
                sys.stdout.write(json.dumps(res, indent=2, sort_keys=True) + "\n")
            continue

        raise ValueError(f"Unsupported action type: {action.type}")


def main(argv: Optional[List[str]] = None) -> int:
    _load_dotenv_always()

    parser = argparse.ArgumentParser(
        prog="bt_agent",
        description="Local Braintrust CLI + optional agent planner (plan/confirm/execute).",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("BRAINTRUST_BASE_URL", DEFAULT_BT_BASE_URL),
        help=f"Braintrust API base URL (default: {DEFAULT_BT_BASE_URL})",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("BRAINTRUST_API_KEY"),
        help="Braintrust API key (or set BRAINTRUST_API_KEY)",
    )
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompts")

    sub = parser.add_subparsers(dest="cmd", required=True)

    ds = sub.add_parser("dataset", help="Dataset operations")
    ds_sub = ds.add_subparsers(dest="dataset_cmd", required=True)
    ds_get = ds_sub.add_parser("get", help="Get a dataset by id")
    ds_get.add_argument("--id", required=True, help="Dataset id (UUID)")
    ds_patch = ds_sub.add_parser("patch", help="PATCH /v1/dataset/{id}")
    ds_patch.add_argument("--id", required=True, help="Dataset id (UUID)")
    ds_patch.add_argument("--json", help="PATCH body as JSON string (object)")
    ds_patch.add_argument("--json-file", help="PATCH body as JSON file path")
    ds_fetch = ds_sub.add_parser("fetch", help="Fetch dataset rows (GET /v1/dataset/{id}/fetch)")
    ds_fetch.add_argument("--id", required=True, help="Dataset id (UUID)")
    ds_fetch.add_argument("--limit", type=int, default=1, help="Number of rows to fetch (default: 1)")
    ds_fetch.add_argument("--offset", type=int, default=0, help="Offset (default: 0)")
    ds_describe = ds_sub.add_parser(
        "describe",
        help="Auto-generate a dataset description from schema + 1 sample row; optionally patch it",
    )
    ds_describe.add_argument("--id", required=True, help="Dataset id (UUID)")
    ds_describe.add_argument(
        "--ai",
        action="store_true",
        help="Use an LLM to generate the description from the sample event (requires OPENAI_API_KEY/LLM_API_KEY)",
    )
    ds_describe.add_argument(
        "--model",
        default=os.environ.get("LLM_MODEL", "gpt-4o"),
        help="LLM model for --ai (default: env LLM_MODEL or gpt-4o)",
    )
    ds_describe.add_argument(
        "--apply",
        action="store_true",
        help="Patch the generated description onto the dataset (will ask to confirm unless --yes)",
    )

    btql = sub.add_parser("btql", help="Run BTQL queries")
    btql.add_argument("--query", help="BTQL query string")
    btql.add_argument("--query-file", help="Path to a BTQL query file")
    btql.add_argument("--query-source", help="query_source override")
    btql.add_argument("--save-as", help="Write JSON response to file path")

    traces = sub.add_parser("traces", help="Fetch and analyze production traces via BTQL")
    traces.add_argument(
        "--project-id",
        default=os.environ.get("BRAINTRUST_PROJECT_ID"),
        help="Braintrust project id (UUID) (or set BRAINTRUST_PROJECT_ID)",
    )
    traces.add_argument("--prompt-name", required=True, help="metadata.prompt_name value")
    traces.add_argument("--hours", type=int, default=24)
    traces.add_argument("--max-traces", type=int, default=100)
    traces.add_argument("--query-source", help="query_source override")
    traces.add_argument("--save-traces-as", help="Write traces JSON array to file path")
    traces.add_argument("--save-analysis-as", help="Write analysis JSON to file path")

    agent = sub.add_parser("agent", help="LLM-powered planning (review + confirm) for BT API actions")
    agent.add_argument("instruction", help="Natural language instruction")
    agent.add_argument("--model", default=os.environ.get("LLM_MODEL", "gpt-4o"))
    agent.add_argument(
        "--instruction-file",
        help="Path to a Markdown/text file with safety instructions for the planner (optional)",
    )
    agent.add_argument("--out-dir", help="Directory to save outputs from actions (optional)")
    agent.add_argument("--save-plan-as", help="Write the plan JSON to file path")

    args = parser.parse_args(argv)

    api_key = args.api_key
    if not api_key:
        raise ValueError("Missing Braintrust API key. Provide --api-key or set BRAINTRUST_API_KEY.")

    client = BraintrustApiClient(api_key=api_key, base_url=args.base_url)

    if args.cmd == "dataset":
        if args.dataset_cmd == "get":
            res = client.get_dataset(args.id)
            sys.stdout.write(json.dumps(res, indent=2, sort_keys=True) + "\n")
            return 0

        if args.dataset_cmd == "patch":
            body = _read_json_arg(args.json, args.json_file)
            _prompt_confirm(f"PATCH dataset {args.id} on {client.base_url}? [y/N] ", args.yes)
            res = client.patch_dataset(args.id, body)
            sys.stdout.write(json.dumps(res, indent=2, sort_keys=True) + "\n")
            return 0

        if args.dataset_cmd == "fetch":
            res = client.fetch_dataset(args.id, limit=args.limit, offset=args.offset)
            sys.stdout.write(json.dumps(res, indent=2, sort_keys=True) + "\n")
            return 0

        if args.dataset_cmd == "describe":
            ds_obj = client.get_dataset(args.id)
            fetch_obj = client.fetch_dataset(args.id, limit=1, offset=0)
            rows = fetch_obj.get("data", fetch_obj.get("rows", fetch_obj.get("events", [])))
            sample_events = rows if isinstance(rows, list) else []

            # Show the full sample event so the operator can verify what the dataset contains.
            sys.stdout.write("\nSample event (limit=1):\n")
            sys.stdout.write(json.dumps(sample_events[0] if sample_events else {}, indent=2, sort_keys=True) + "\n")

            if args.ai:
                desc = _llm_dataset_description(ds_obj, sample_events, model=args.model)
            else:
                desc = _auto_dataset_description(ds_obj, sample_events)

            sys.stdout.write("\nSuggested description:\n")
            sys.stdout.write(desc + "\n\n")

            if not args.apply:
                sys.stdout.write("Dry-run only. Re-run with --apply to patch this description onto the dataset.\n")
                return 0

            if args.yes:
                raise ValueError(
                    "Refusing to run dataset describe --apply with --yes. "
                    "Re-run without --yes so you can verify before patching."
                )

            _prompt_confirm(f"PATCH dataset {args.id} description on {client.base_url}? [y/N] ", args.yes)
            res = client.patch_dataset(args.id, {"description": desc})
            sys.stdout.write(json.dumps(res, indent=2, sort_keys=True) + "\n")
            return 0

    if args.cmd == "btql":
        if bool(args.query) == bool(args.query_file):
            raise ValueError("Provide exactly one of --query or --query-file")
        query = args.query or Path(args.query_file).read_text(encoding="utf-8")
        _prompt_confirm("Run BTQL query? [y/N] ", args.yes)
        res = client.btql(query, fmt="json", query_source=args.query_source)
        if args.save_as:
            Path(args.save_as).expanduser().resolve().write_text(
                json.dumps(res, indent=2, sort_keys=True), encoding="utf-8"
            )
            sys.stdout.write(f"Wrote: {Path(args.save_as).expanduser().resolve()}\n")
        else:
            sys.stdout.write(json.dumps(res, indent=2, sort_keys=True) + "\n")
        return 0

    if args.cmd == "traces":
        if not args.project_id:
            raise ValueError(
                "Missing Braintrust project id. Provide --project-id or set BRAINTRUST_PROJECT_ID."
            )
        traces_list = _fetch_traces_btql(
            client=client,
            project_id=args.project_id,
            prompt_name=args.prompt_name,
            hours=args.hours,
            max_traces=args.max_traces,
            query_source=args.query_source,
        )
        analysis = _analyze_traces(traces_list)
        sys.stdout.write(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
        if args.save_traces_as:
            Path(args.save_traces_as).expanduser().resolve().write_text(
                json.dumps(traces_list, indent=2, sort_keys=True), encoding="utf-8"
            )
        if args.save_analysis_as:
            Path(args.save_analysis_as).expanduser().resolve().write_text(
                json.dumps(analysis, indent=2, sort_keys=True), encoding="utf-8"
            )
        return 0

    if args.cmd == "agent":
        available = ["dataset.patch", "btql.query"]
        instructions_text = ""
        if args.instruction_file:
            instructions_text = Path(args.instruction_file).read_text(encoding="utf-8")
        actions = _llm_plan(
            args.instruction,
            available,
            model=args.model,
            instructions_file_text=instructions_text,
        )
        sys.stdout.write(_render_plan(actions))

        plan_obj = {
            "actions": [{"type": a.type, "args": a.args, "reason": a.reason} for a in actions],
            "instruction": args.instruction,
            "model": args.model,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        if args.save_plan_as:
            Path(args.save_plan_as).expanduser().resolve().write_text(
                json.dumps(plan_obj, indent=2, sort_keys=True), encoding="utf-8"
            )
            sys.stdout.write(f"Wrote: {Path(args.save_plan_as).expanduser().resolve()}\n")

        if not actions:
            return 0

        _prompt_confirm("\nExecute this plan? [y/N] ", args.yes)
        _execute_actions(client=client, actions=actions, yes=args.yes, out_dir=args.out_dir)
        return 0

    raise ValueError(f"Unknown command: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())
