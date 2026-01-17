#!/usr/bin/env python3
"""
Fetch email-tagger traces filtered by output.email_tags category.

Fetches traces where output.email_tags contains a specific tag value,
and saves them to separate folders.
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from collections import defaultdict

import requests

DEFAULT_BT_BASE_URL = "https://api.braintrust.dev"

# Categories to fetch
CATEGORIES = [
    "ooo",
    "automated",
    "bounced",
    "calendar_invite",
    "positive_sentiment",
    "negative_sentiment",
    "default",
]


def _load_dotenv() -> None:
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


def btql_query(
    api_key: str,
    query: str,
    base_url: str = DEFAULT_BT_BASE_URL,
) -> Dict[str, Any]:
    """Execute a BTQL query."""
    resp = requests.post(
        f"{base_url}/btql",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={"query": query, "fmt": "json"},
        timeout=120,
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"BTQL error {resp.status_code}: {resp.text[:500]}")
    return resp.json()


def fetch_traces_batch(
    api_key: str,
    project_id: str,
    prompt_name: str,
    hours: int,
    limit: int,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """Fetch a batch of traces."""
    query = f"""select: *
from: project_logs('{project_id}')
spans
filter: created > now() - interval {hours} hour and created < now() and span_attributes.parent_span_id is null and span_attributes.name = 'chat_completions' and metadata.prompt_name = '{prompt_name}' and error is null
order by: created desc
limit: {limit}
offset: {offset}"""

    result = btql_query(api_key, query)
    return result.get("data", [])


def get_trace_tags(trace: Dict[str, Any]) -> List[str]:
    """Extract email_tags from trace output."""
    output = trace.get("output")
    if isinstance(output, dict):
        tags = output.get("email_tags", [])
        if isinstance(tags, list):
            return tags
    return []


def main():
    _load_dotenv()

    # Configuration - can override via env vars
    api_key = os.environ.get("BRAINTRUST_API_KEY")
    if not api_key:
        # Allow passing via command line for testing
        if len(sys.argv) > 1:
            api_key = sys.argv[1]
        else:
            print("Error: Set BRAINTRUST_API_KEY or pass as first argument")
            sys.exit(1)

    project_id = os.environ.get("BRAINTRUST_PROJECT_ID", "1e7654f3-505d-402c-b24d-60048d3e6916")
    prompt_name = "email-tagger"
    hours = 168  # 7 days
    target_per_category = 100
    batch_size = 500
    max_total_fetches = 10000  # Safety limit

    output_base = Path(__file__).resolve().parent / "traces"

    # Create output directories
    for cat in CATEGORIES:
        (output_base / cat).mkdir(parents=True, exist_ok=True)

    print(f"Fetching traces for prompt: {prompt_name}")
    print(f"Project ID: {project_id}")
    print(f"Time range: last {hours} hours")
    print(f"Target per category: {target_per_category}")
    print(f"Categories: {CATEGORIES}")
    print()

    # Track traces by category
    traces_by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    total_fetched = 0
    offset = 0

    # Check if we have enough for all categories
    def all_categories_filled() -> bool:
        return all(
            len(traces_by_category[cat]) >= target_per_category
            for cat in CATEGORIES
        )

    # Fetch until we have enough or hit limits
    while not all_categories_filled() and total_fetched < max_total_fetches:
        print(f"Fetching batch at offset {offset}...")

        batch = fetch_traces_batch(
            api_key=api_key,
            project_id=project_id,
            prompt_name=prompt_name,
            hours=hours,
            limit=batch_size,
            offset=offset,
        )

        if not batch:
            print("No more traces available.")
            break

        total_fetched += len(batch)
        offset += len(batch)

        # Categorize traces
        for trace in batch:
            tags = get_trace_tags(trace)
            for tag in tags:
                if tag in CATEGORIES and len(traces_by_category[tag]) < target_per_category:
                    traces_by_category[tag].append(trace)

        # Print progress
        status = ", ".join(f"{cat}: {len(traces_by_category[cat])}" for cat in CATEGORIES)
        print(f"  Total fetched: {total_fetched}, Progress: [{status}]")

        # Early exit if we got fewer than batch size (no more data)
        if len(batch) < batch_size:
            print("Reached end of available traces.")
            break

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)

    # Save traces to files
    for cat in CATEGORIES:
        cat_traces = traces_by_category[cat]
        count = len(cat_traces)

        if count == 0:
            print(f"  {cat}: 0 traces found (NONE AVAILABLE)")
            continue

        # Save traces
        output_file = output_base / cat / "traces.json"
        output_file.write_text(json.dumps(cat_traces, indent=2), encoding="utf-8")

        # Save summary
        summary = {
            "category": cat,
            "count": count,
            "target": target_per_category,
            "complete": count >= target_per_category,
        }
        summary_file = output_base / cat / "summary.json"
        summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        status = "✓" if count >= target_per_category else f"⚠ (only {count})"
        print(f"  {cat}: {count} traces saved {status}")

    print()
    print(f"Output directory: {output_base}")
    print("Done!")


if __name__ == "__main__":
    main()
