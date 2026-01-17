#!/usr/bin/env python3
"""
Braintrust Trace Fetcher & Dataset Builder

A focused tool for fetching production traces from Braintrust for a specific prompt
and converting them into evaluation datasets.

Usage Examples:
    # Fetch 5000 traces for a prompt and save locally
    python trace_fetcher.py fetch \
        --prompt-name "my_agent_prompt" \
        --max-traces 5000 \
        --output traces_dataset.json

    # Fetch and create a Braintrust dataset
    python trace_fetcher.py fetch \
        --prompt-name "my_agent_prompt" \
        --max-traces 5000 \
        --create-dataset "My Agent Eval Dataset"

    # Preview what traces exist (dry run with analysis)
    python trace_fetcher.py preview \
        --prompt-name "my_agent_prompt" \
        --hours 168
"""

import argparse
import json
import os
import sys
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

import requests

DEFAULT_BT_BASE_URL = "https://api.braintrust.dev"
DEFAULT_BATCH_SIZE = 500  # BTQL has limits, so we paginate


class BraintrustApiError(RuntimeError):
    """Custom error for Braintrust API failures."""
    pass


def _load_dotenv() -> None:
    """Load .env file from script directory if it exists."""
    dotenv_path = Path(__file__).resolve().parent / ".env"
    if not dotenv_path.exists():
        return

    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=dotenv_path, override=False)
        return
    except ImportError:
        pass

    # Fallback: simple .env parser
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


def _json_or_raise(resp: requests.Response) -> Dict[str, Any]:
    """Parse JSON response or raise a descriptive error."""
    try:
        data = resp.json()
    except Exception:
        raise BraintrustApiError(f"HTTP {resp.status_code}: Non-JSON response: {resp.text[:500]}")

    if resp.status_code >= 400:
        raise BraintrustApiError(f"HTTP {resp.status_code}: {json.dumps(data)[:2000]}")
    return data


class BraintrustClient:
    """
    Minimal Braintrust API client focused on trace fetching and dataset operations.
    """

    def __init__(self, api_key: str, base_url: str = DEFAULT_BT_BASE_URL, timeout_s: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        })

    def btql(
        self,
        query: str,
        fmt: str = "json",
        query_source: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute a BTQL query against Braintrust."""
        payload: Dict[str, Any] = {"query": query, "fmt": fmt}
        if query_source:
            payload["query_source"] = query_source
        resp = self.session.post(
            f"{self.base_url}/btql",
            json=payload,
            timeout=self.timeout_s,
        )
        return _json_or_raise(resp)

    def get_project(self, project_id: str) -> Dict[str, Any]:
        """Get project metadata."""
        resp = self.session.get(
            f"{self.base_url}/v1/project/{project_id}",
            timeout=self.timeout_s,
        )
        return _json_or_raise(resp)

    def create_dataset(
        self,
        project_id: str,
        name: str,
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a new dataset in a project."""
        payload = {"project_id": project_id, "name": name}
        if description:
            payload["description"] = description
        resp = self.session.post(
            f"{self.base_url}/v1/dataset",
            json=payload,
            timeout=self.timeout_s,
        )
        return _json_or_raise(resp)

    def insert_dataset_rows(
        self,
        dataset_id: str,
        rows: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Insert rows into an existing dataset."""
        resp = self.session.post(
            f"{self.base_url}/v1/dataset/{dataset_id}/insert",
            json={"events": rows},
            timeout=self.timeout_s,
        )
        return _json_or_raise(resp)


@dataclass
class FetchConfig:
    """Configuration for trace fetching."""
    project_id: str
    prompt_name: str
    hours: int = 168  # Default: 7 days
    max_traces: int = 5000
    batch_size: int = DEFAULT_BATCH_SIZE
    query_source: Optional[str] = None
    include_errors: bool = False  # Whether to include traces with errors


@dataclass
class FetchResult:
    """Result of a trace fetch operation."""
    traces: List[Dict[str, Any]] = field(default_factory=list)
    total_fetched: int = 0
    prompt_name: str = ""
    time_range_hours: int = 0
    fetch_timestamp: str = ""
    errors: List[str] = field(default_factory=list)


def build_trace_query(
    project_id: str,
    prompt_name: str,
    hours: int,
    limit: int,
    offset: int = 0,
    include_errors: bool = False,
) -> str:
    """
    Build a BTQL query for fetching traces.

    BTQL is Braintrust's query language, similar to SQL but designed for trace data.
    Key concepts:
    - project_logs() is the function to query logs from a specific project
    - spans expands the query to include span-level data
    - filter applies conditions (time range, prompt name, etc.)
    - limit/offset for pagination
    """
    filters = [
        f"created > now() - interval {hours} hour",
        "created < now()",
        "span_attributes.parent_span_id is null",  # Root spans only (no nested calls)
        "span_attributes.name = 'chat_completions'",  # LLM completion spans
        f"metadata.prompt_name = '{prompt_name}'",
    ]

    if not include_errors:
        filters.append("error is null")

    where_clause = " and ".join(filters)

    query = f"""select: *
from: project_logs('{project_id}')
spans
filter: {where_clause}
order by: created desc
limit: {limit}
offset: {offset}"""

    return query


def fetch_traces_paginated(
    client: BraintrustClient,
    config: FetchConfig,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> FetchResult:
    """
    Fetch traces from Braintrust with pagination.

    This function handles the complexity of fetching large numbers of traces
    by breaking the request into smaller batches. BTQL has limits on how many
    results can be returned in a single query, so we paginate.
    """
    result = FetchResult(
        prompt_name=config.prompt_name,
        time_range_hours=config.hours,
        fetch_timestamp=datetime.now(timezone.utc).isoformat(),
    )

    query_source = config.query_source or f"trace_fetcher_{uuid.uuid4().hex[:8]}"
    offset = 0
    all_traces: List[Dict[str, Any]] = []

    while len(all_traces) < config.max_traces:
        remaining = config.max_traces - len(all_traces)
        batch_limit = min(config.batch_size, remaining)

        query = build_trace_query(
            project_id=config.project_id,
            prompt_name=config.prompt_name,
            hours=config.hours,
            limit=batch_limit,
            offset=offset,
            include_errors=config.include_errors,
        )

        try:
            resp = client.btql(query, fmt="json", query_source=query_source)
            data = resp.get("data", [])

            if not data:
                # No more results
                break

            all_traces.extend(data)
            offset += len(data)

            if progress_callback:
                progress_callback(len(all_traces), config.max_traces)

            # If we got fewer results than requested, we've exhausted the data
            if len(data) < batch_limit:
                break

        except BraintrustApiError as e:
            result.errors.append(str(e))
            break

    result.traces = all_traces
    result.total_fetched = len(all_traces)
    return result


def transform_trace_to_dataset_row(
    trace: Dict[str, Any],
    include_metadata: bool = True,
) -> Dict[str, Any]:
    """
    Transform a raw trace into a dataset row format.

    Braintrust datasets expect rows with:
    - input: The input to the model (typically messages)
    - expected: The expected/actual output (for evaluation)
    - metadata: Optional metadata for filtering/analysis

    This transformation extracts the relevant fields from trace spans
    and restructures them for dataset use.
    """
    # Extract span attributes (contains model info, messages, etc.)
    span_attrs = trace.get("span_attributes", {}) or {}
    metadata = trace.get("metadata", {}) or {}

    # The input is typically the messages sent to the model
    input_data = span_attrs.get("input", {})
    if not input_data:
        # Fallback: try to get from llm_input
        input_data = span_attrs.get("llm_input", {})

    # The output/expected is the model's response
    output_data = span_attrs.get("output", {})
    if not output_data:
        output_data = span_attrs.get("llm_output", {})

    row: Dict[str, Any] = {
        "input": input_data,
        "expected": output_data,
    }

    if include_metadata:
        row["metadata"] = {
            "source_trace_id": trace.get("id"),
            "source_span_id": trace.get("span_id"),
            "prompt_name": metadata.get("prompt_name"),
            "model": span_attrs.get("model"),
            "created": trace.get("created"),
            # Include any custom metadata from the original trace
            **{k: v for k, v in metadata.items() if k != "prompt_name"},
        }

    return row


def create_dataset_from_traces(
    traces: List[Dict[str, Any]],
    include_metadata: bool = True,
) -> List[Dict[str, Any]]:
    """
    Convert a list of traces into dataset rows.

    Filters out traces with empty inputs/outputs to ensure dataset quality.
    """
    dataset_rows = []

    for trace in traces:
        row = transform_trace_to_dataset_row(trace, include_metadata)

        # Skip rows with empty input or output
        if not row["input"] or not row["expected"]:
            continue

        dataset_rows.append(row)

    return dataset_rows


def analyze_traces(traces: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze fetched traces to provide summary statistics.

    Useful for understanding the data before creating a dataset.
    """
    if not traces:
        return {"total": 0, "message": "No traces found"}

    models: Dict[str, int] = {}
    dates: List[str] = []
    has_output_count = 0
    error_count = 0

    for trace in traces:
        span_attrs = trace.get("span_attributes", {}) or {}
        model = span_attrs.get("model") or "unknown"
        models[model] = models.get(model, 0) + 1

        if trace.get("created"):
            dates.append(trace["created"])

        if span_attrs.get("output"):
            has_output_count += 1

        if trace.get("error"):
            error_count += 1

    dates.sort()

    return {
        "total_traces": len(traces),
        "traces_with_output": has_output_count,
        "traces_with_errors": error_count,
        "models_distribution": dict(sorted(models.items(), key=lambda x: -x[1])),
        "date_range": {
            "earliest": dates[0] if dates else None,
            "latest": dates[-1] if dates else None,
        },
        "analyzed_at": datetime.now(timezone.utc).isoformat(),
    }


def print_progress(current: int, total: int) -> None:
    """Print a simple progress indicator."""
    pct = (current / total * 100) if total > 0 else 0
    sys.stdout.write(f"\r  Fetched {current:,} / {total:,} traces ({pct:.1f}%)")
    sys.stdout.flush()


def cmd_fetch(args: argparse.Namespace, client: BraintrustClient) -> int:
    """Execute the fetch command."""
    config = FetchConfig(
        project_id=args.project_id,
        prompt_name=args.prompt_name,
        hours=args.hours,
        max_traces=args.max_traces,
        batch_size=args.batch_size,
        include_errors=args.include_errors,
    )

    print(f"Fetching traces for prompt: {config.prompt_name}")
    print(f"  Project ID: {config.project_id}")
    print(f"  Time range: last {config.hours} hours")
    print(f"  Max traces: {config.max_traces:,}")
    print()

    # Fetch traces with progress
    result = fetch_traces_paginated(
        client=client,
        config=config,
        progress_callback=print_progress if sys.stdout.isatty() else None,
    )

    if sys.stdout.isatty():
        print()  # New line after progress

    if result.errors:
        print(f"\nWarnings/Errors during fetch:")
        for err in result.errors:
            print(f"  - {err}")

    print(f"\nFetched {result.total_fetched:,} traces")

    # Analyze traces
    analysis = analyze_traces(result.traces)
    print(f"\nAnalysis:")
    print(f"  Traces with output: {analysis['traces_with_output']:,}")
    print(f"  Traces with errors: {analysis['traces_with_errors']:,}")
    print(f"  Models: {analysis['models_distribution']}")
    if analysis['date_range']['earliest']:
        print(f"  Date range: {analysis['date_range']['earliest'][:10]} to {analysis['date_range']['latest'][:10]}")

    # Transform to dataset format
    dataset_rows = create_dataset_from_traces(
        result.traces,
        include_metadata=not args.no_metadata,
    )
    print(f"\nValid dataset rows (with input & output): {len(dataset_rows):,}")

    # Save to file
    if args.output:
        output_path = Path(args.output).expanduser().resolve()

        output_data = {
            "metadata": {
                "prompt_name": config.prompt_name,
                "project_id": config.project_id,
                "fetched_at": result.fetch_timestamp,
                "time_range_hours": config.hours,
                "total_traces": result.total_fetched,
                "valid_rows": len(dataset_rows),
            },
            "rows": dataset_rows,
        }

        output_path.write_text(json.dumps(output_data, indent=2), encoding="utf-8")
        print(f"\nSaved to: {output_path}")

    # Optionally save raw traces
    if args.save_raw:
        raw_path = Path(args.save_raw).expanduser().resolve()
        raw_path.write_text(json.dumps(result.traces, indent=2), encoding="utf-8")
        print(f"Raw traces saved to: {raw_path}")

    # Create Braintrust dataset if requested
    if args.create_dataset:
        print(f"\nCreating Braintrust dataset: {args.create_dataset}")
        try:
            dataset = client.create_dataset(
                project_id=config.project_id,
                name=args.create_dataset,
                description=f"Dataset from production traces of {config.prompt_name} ({len(dataset_rows):,} rows)",
            )
            dataset_id = dataset.get("id")
            print(f"  Dataset created with ID: {dataset_id}")

            # Insert rows in batches
            batch_size = 100
            for i in range(0, len(dataset_rows), batch_size):
                batch = dataset_rows[i:i + batch_size]
                client.insert_dataset_rows(dataset_id, batch)
                if sys.stdout.isatty():
                    sys.stdout.write(f"\r  Inserted {min(i + batch_size, len(dataset_rows)):,} / {len(dataset_rows):,} rows")
                    sys.stdout.flush()

            if sys.stdout.isatty():
                print()
            print(f"  Dataset populated successfully!")
            print(f"  View at: https://www.braintrust.dev/app/datasets/{dataset_id}")

        except BraintrustApiError as e:
            print(f"  Error creating dataset: {e}")
            return 1

    return 0


def cmd_preview(args: argparse.Namespace, client: BraintrustClient) -> int:
    """Execute the preview command - dry run to see what traces exist."""
    # Fetch a small sample to analyze
    config = FetchConfig(
        project_id=args.project_id,
        prompt_name=args.prompt_name,
        hours=args.hours,
        max_traces=min(args.sample_size, 100),
        include_errors=True,
    )

    print(f"Previewing traces for prompt: {config.prompt_name}")
    print(f"  Project ID: {config.project_id}")
    print(f"  Time range: last {config.hours} hours")
    print(f"  Sample size: {config.max_traces}")
    print()

    result = fetch_traces_paginated(client=client, config=config)

    analysis = analyze_traces(result.traces)
    print(json.dumps(analysis, indent=2))

    # Show a sample trace structure
    if result.traces and args.show_sample:
        print("\n--- Sample Trace Structure ---")
        sample = result.traces[0]
        # Show keys at each level
        print(f"Top-level keys: {list(sample.keys())}")
        if "span_attributes" in sample:
            print(f"span_attributes keys: {list(sample['span_attributes'].keys())}")
        if "metadata" in sample:
            print(f"metadata keys: {list(sample['metadata'].keys())}")

    return 0


def main(argv: Optional[List[str]] = None) -> int:
    _load_dotenv()

    parser = argparse.ArgumentParser(
        prog="trace_fetcher",
        description="Fetch Braintrust traces and create evaluation datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("BRAINTRUST_API_KEY"),
        help="Braintrust API key (or set BRAINTRUST_API_KEY env var)",
    )
    parser.add_argument(
        "--base-url",
        default=os.environ.get("BRAINTRUST_BASE_URL", DEFAULT_BT_BASE_URL),
        help=f"Braintrust API base URL (default: {DEFAULT_BT_BASE_URL})",
    )
    parser.add_argument(
        "--project-id",
        default=os.environ.get("BRAINTRUST_PROJECT_ID"),
        help="Braintrust project ID (or set BRAINTRUST_PROJECT_ID env var)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Fetch command
    fetch_parser = subparsers.add_parser(
        "fetch",
        help="Fetch traces and optionally create a dataset",
    )
    fetch_parser.add_argument(
        "--prompt-name",
        required=True,
        help="The prompt name to filter traces by (metadata.prompt_name)",
    )
    fetch_parser.add_argument(
        "--hours",
        type=int,
        default=168,
        help="Hours of history to fetch (default: 168 = 7 days)",
    )
    fetch_parser.add_argument(
        "--max-traces",
        type=int,
        default=5000,
        help="Maximum number of traces to fetch (default: 5000)",
    )
    fetch_parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for pagination (default: {DEFAULT_BATCH_SIZE})",
    )
    fetch_parser.add_argument(
        "--include-errors",
        action="store_true",
        help="Include traces that have errors",
    )
    fetch_parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Exclude metadata from dataset rows",
    )
    fetch_parser.add_argument(
        "--output", "-o",
        help="Output file path for the dataset JSON",
    )
    fetch_parser.add_argument(
        "--save-raw",
        help="Also save raw traces to this file path",
    )
    fetch_parser.add_argument(
        "--create-dataset",
        metavar="NAME",
        help="Create a Braintrust dataset with this name",
    )

    # Preview command
    preview_parser = subparsers.add_parser(
        "preview",
        help="Preview/analyze traces without creating a dataset",
    )
    preview_parser.add_argument(
        "--prompt-name",
        required=True,
        help="The prompt name to filter traces by",
    )
    preview_parser.add_argument(
        "--hours",
        type=int,
        default=168,
        help="Hours of history to analyze (default: 168 = 7 days)",
    )
    preview_parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of traces to sample for preview (default: 100)",
    )
    preview_parser.add_argument(
        "--show-sample",
        action="store_true",
        help="Show the structure of a sample trace",
    )

    args = parser.parse_args(argv)

    # Validate required args
    if not args.api_key:
        print("Error: Missing Braintrust API key. Set --api-key or BRAINTRUST_API_KEY.", file=sys.stderr)
        return 1

    if not args.project_id:
        print("Error: Missing project ID. Set --project-id or BRAINTRUST_PROJECT_ID.", file=sys.stderr)
        return 1

    # Create client
    client = BraintrustClient(
        api_key=args.api_key,
        base_url=args.base_url,
    )

    # Dispatch to command handler
    if args.command == "fetch":
        return cmd_fetch(args, client)
    elif args.command == "preview":
        return cmd_preview(args, client)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
