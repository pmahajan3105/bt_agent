#!/usr/bin/env python3
"""
Utility script for working with Braintrust prompts and traces.

Provides subcommands for fetching prompt definitions, traces, and saving
golden datasets - designed to be used by Claude for LLM-guided trace selection.

Usage:
    # Fetch and display a prompt's system message
    python prompt_utils.py fetch-prompt --slug email-tagger

    # Fetch traces for a prompt
    python prompt_utils.py fetch-traces --slug email-tagger --hours 168 --limit 500

    # Save traces to eval-compatible format
    python prompt_utils.py save-golden --input traces.json --output golden.json

Environment Variables:
    BRAINTRUST_API_KEY    - Required: Your Braintrust API key
    BRAINTRUST_PROJECT_ID - Required: Default project ID
    BRAINTRUST_BASE_URL   - Optional: API base URL (default: https://api.braintrust.dev)
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# Defaults
DEFAULT_BASE_URL = "https://api.braintrust.dev"
DEFAULT_HOURS = 168  # 7 days
DEFAULT_LIMIT = 500


def load_dotenv() -> None:
    """Load .env file from script directory if it exists."""
    dotenv_path = Path(__file__).resolve().parent / ".env"
    if not dotenv_path.exists():
        return

    for line in dotenv_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, val = line.split("=", 1)
        key, val = key.strip(), val.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = val


class BraintrustClient:
    """Simple Braintrust API client."""

    def __init__(self, api_key: str, base_url: str = DEFAULT_BASE_URL, timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })

    def btql(self, query: str) -> Dict[str, Any]:
        """Execute a BTQL query and return results."""
        resp = self.session.post(
            f"{self.base_url}/btql",
            json={"query": query, "fmt": "json"},
            timeout=self.timeout,
        )

        if resp.status_code >= 400:
            try:
                err = resp.json()
                msg = err.get("Message", resp.text[:500])
            except Exception:
                msg = resp.text[:500]
            raise RuntimeError(f"BTQL error ({resp.status_code}): {msg}")

        return resp.json()

    def get_prompts(self, project_id: str) -> List[Dict[str, Any]]:
        """Get all prompts for a project."""
        resp = self.session.get(
            f"{self.base_url}/v1/prompt",
            params={"project_id": project_id},
            timeout=self.timeout,
        )

        if resp.status_code >= 400:
            raise RuntimeError(f"API error ({resp.status_code}): {resp.text[:500]}")

        return resp.json().get("objects", [])

    def get_prompt(self, project_id: str, slug: str) -> Optional[Dict[str, Any]]:
        """Get a specific prompt by slug."""
        resp = self.session.get(
            f"{self.base_url}/v1/prompt",
            params={"project_id": project_id, "slug": slug},
            timeout=self.timeout,
        )

        if resp.status_code >= 400:
            raise RuntimeError(f"API error ({resp.status_code}): {resp.text[:500]}")

        prompts = resp.json().get("objects", [])
        return prompts[0] if prompts else None


def get_system_message(prompt: Dict[str, Any]) -> Optional[str]:
    """Extract the system message from a prompt definition."""
    prompt_data = prompt.get("prompt_data", {})
    prompt_section = prompt_data.get("prompt", {})
    messages = prompt_section.get("messages", [])

    for msg in messages:
        if msg.get("role") == "system":
            content = msg.get("content", "")
            # Handle both string and structured content
            if isinstance(content, str):
                return content
            elif isinstance(content, list):
                # Extract text from content parts
                texts = [p.get("text", "") for p in content if p.get("type") == "text"]
                return "\n".join(texts)
    return None


def cmd_fetch_prompt(args) -> int:
    """Fetch and display a prompt's definition."""
    client = BraintrustClient(
        api_key=args.api_key,
        base_url=args.base_url,
    )

    prompt = client.get_prompt(args.project_id, args.slug)
    if not prompt:
        print(f"Error: Prompt '{args.slug}' not found in project", file=sys.stderr)
        return 1

    system_msg = get_system_message(prompt)

    print(f"# Prompt: {args.slug}")
    print(f"# ID: {prompt.get('id', 'N/A')}")
    print(f"# Description: {prompt.get('description', 'N/A')}")
    print()
    print("## System Message")
    print()
    if system_msg:
        print(system_msg)
    else:
        print("(No system message found)")

    # Also output JSON if requested
    if args.json:
        print()
        print("## Full Prompt JSON")
        print()
        print(json.dumps(prompt, indent=2))

    return 0


def cmd_fetch_traces(args) -> int:
    """Fetch traces for a prompt."""
    client = BraintrustClient(
        api_key=args.api_key,
        base_url=args.base_url,
    )

    # Build BTQL query
    query = f"""select: *
from: project_logs('{args.project_id}')
spans
filter: created > now() - interval {args.hours} hour and created < now() and span_attributes.parent_span_id is null and span_attributes.name = 'chat_completions' and metadata.prompt_name = '{args.slug}' and error is null
sort: created desc
limit: {args.limit}"""

    print(f"Fetching traces for prompt: {args.slug}", file=sys.stderr)
    print(f"  Time range: last {args.hours} hours", file=sys.stderr)
    print(f"  Limit: {args.limit}", file=sys.stderr)

    result = client.btql(query)
    traces = result.get("data", [])

    print(f"  Found: {len(traces)} traces", file=sys.stderr)

    # Output traces
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(traces, indent=2), encoding="utf-8")
        print(f"  Saved to: {output_path}", file=sys.stderr)
    else:
        # Output to stdout
        print(json.dumps(traces, indent=2))

    return 0


def cmd_save_golden(args) -> int:
    """Convert traces to eval-compatible golden dataset format."""
    # Read input traces
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    traces = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(traces, list):
        print("Error: Input must be a JSON array of traces", file=sys.stderr)
        return 1

    # Convert to eval format
    rows = []
    for trace in traces:
        # Extract input (user message content)
        input_data = trace.get("input", [])

        # Extract output (expected result)
        output = trace.get("output")
        if isinstance(output, str):
            expected = output
        else:
            expected = json.dumps(output)

        row = {
            "input": input_data,
            "expected": expected,
            "metadata": {
                "source_trace_id": trace.get("id", trace.get("span_id", "unknown")),
                "source_created": trace.get("created"),
            }
        }
        rows.append(row)

    # Build output structure
    output_data = {
        "metadata": {
            "source": "golden_trace_selector",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "trace_count": len(rows),
        },
        "rows": rows,
    }

    # Write output
    output_path = Path(args.output)
    output_path.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
    print(f"Saved {len(rows)} golden traces to: {output_path}", file=sys.stderr)

    return 0


def cmd_list_prompts(args) -> int:
    """List all prompts in a project."""
    client = BraintrustClient(
        api_key=args.api_key,
        base_url=args.base_url,
    )

    prompts = client.get_prompts(args.project_id)

    # Filter out judges if requested
    if not args.include_judges:
        prompts = [p for p in prompts if "judge" not in p.get("slug", "").lower()]

    print(f"Found {len(prompts)} prompts in project:\n")

    for p in sorted(prompts, key=lambda x: x.get("slug", "")):
        slug = p.get("slug", "N/A")
        desc = (p.get("description") or "")[:60]
        print(f"  {slug:40s} {desc}")

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Utility for working with Braintrust prompts and traces",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Common arguments
    parser.add_argument(
        "--api-key",
        default=os.environ.get("BRAINTRUST_API_KEY"),
        help="Braintrust API key (or set BRAINTRUST_API_KEY)",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("BRAINTRUST_BASE_URL", DEFAULT_BASE_URL),
        help=f"Braintrust API base URL (default: {DEFAULT_BASE_URL})",
    )
    parser.add_argument(
        "--project-id",
        default=os.environ.get("BRAINTRUST_PROJECT_ID"),
        help="Braintrust project ID (or set BRAINTRUST_PROJECT_ID)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # fetch-prompt command
    fetch_prompt_parser = subparsers.add_parser(
        "fetch-prompt",
        help="Fetch and display a prompt's definition",
    )
    fetch_prompt_parser.add_argument(
        "--slug", "-s",
        required=True,
        help="Prompt slug to fetch",
    )
    fetch_prompt_parser.add_argument(
        "--json", "-j",
        action="store_true",
        help="Also output full JSON",
    )
    fetch_prompt_parser.set_defaults(func=cmd_fetch_prompt)

    # fetch-traces command
    fetch_traces_parser = subparsers.add_parser(
        "fetch-traces",
        help="Fetch traces for a prompt",
    )
    fetch_traces_parser.add_argument(
        "--slug", "-s",
        required=True,
        help="Prompt slug to fetch traces for",
    )
    fetch_traces_parser.add_argument(
        "--hours",
        type=int,
        default=DEFAULT_HOURS,
        help=f"Hours of history to fetch (default: {DEFAULT_HOURS})",
    )
    fetch_traces_parser.add_argument(
        "--limit", "-n",
        type=int,
        default=DEFAULT_LIMIT,
        help=f"Max traces to fetch (default: {DEFAULT_LIMIT})",
    )
    fetch_traces_parser.add_argument(
        "--output", "-o",
        help="Output file (default: stdout)",
    )
    fetch_traces_parser.set_defaults(func=cmd_fetch_traces)

    # save-golden command
    save_golden_parser = subparsers.add_parser(
        "save-golden",
        help="Convert traces to eval-compatible golden dataset format",
    )
    save_golden_parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input traces JSON file",
    )
    save_golden_parser.add_argument(
        "--output", "-o",
        required=True,
        help="Output golden dataset file",
    )
    save_golden_parser.set_defaults(func=cmd_save_golden)

    # list-prompts command
    list_prompts_parser = subparsers.add_parser(
        "list-prompts",
        help="List all prompts in a project",
    )
    list_prompts_parser.add_argument(
        "--include-judges",
        action="store_true",
        help="Include judge prompts in listing",
    )
    list_prompts_parser.set_defaults(func=cmd_list_prompts)

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return 1

    # Validate common args
    if not args.api_key:
        print("Error: Missing API key. Set --api-key or BRAINTRUST_API_KEY", file=sys.stderr)
        return 1

    if not args.project_id:
        print("Error: Missing project ID. Set --project-id or BRAINTRUST_PROJECT_ID", file=sys.stderr)
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
